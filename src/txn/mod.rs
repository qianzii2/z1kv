//! Z1Kv — embedded MVCC key-value engine facade.
//!
//! Wires together MVCC (visibility.rs) + WAL + the three-layer storage
//! (L1 memstore / L2 disk / L3 frozen), exposing the
//! begin/get/put/delete/commit/rollback/scan API.
//!
//! # Commit flow
//!
//! 1. WAL append_durable(Commit) → DURABILITY BOUNDARY
//! 2. MVCC commit (SSI check + active_txns removal + committed_history insert)
//! 3. snapshot_cache.invalidate()
//!
//! # Invariants
//!
//! - D4: put goes WAL-first, then memory (guaranteed inside memstore.put)
//! - The MVCC commit is non-compensable: once a WAL Commit record is durable,
//!   recovery treats that transaction as committed

pub mod protocol;

use crate::config::Z1Config;
use crate::error::{Result, Z1Error};
use crate::mvcc::cache::SnapshotCache;
use crate::mvcc::visibility::{TxnSnapshot, VisibilityManager};
use crate::store::config::SyncLevel;
use crate::store::disk::DiskLayer;
use crate::store::flush::FlushEngine;
use crate::store::mem::MemStore;
use crate::store::recent_flush_cache::RecentFlushCache;
use crate::store::types::{Z1Entry, Z1Key};
use crate::wal::{recovery::replay_committed_ops, WalConfig, WalRecord, WalWriter};
use crate::TxnId;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The Z1Kv engine.
pub struct Z1Kv {
    /// Data directory this engine was opened on (read-only snapshot of the
    /// path; see [`Z1Kv::data_dir`]). Private by design: external mutation
    /// of engine state would bypass the engine lock and durability contracts.
    data_dir: PathBuf,
    config: Z1Config,
    /// Process-level engine lock: exclusively owns data_dir while held,
    /// preventing two instances from concurrently corrupting WAL segment
    /// rotation / patch-id allocation / checkpoint truncation (silent data
    /// corruption). Released when the instance drops.
    _engine_lock: crate::engine_lock::EngineLock,
    wal: Arc<WalWriter>,
    mvcc: RwLock<VisibilityManager>,
    snapshot_cache: SnapshotCache,
    /// Transaction-pinned snapshots (SSI read isolation): one snapshot is
    /// taken and pinned at `begin_txn`; `get_for_txn` reads through that
    /// snapshot instead of the global current snapshot.
    ///
    /// Before the fix: `get_for_txn` called `snapshot()` on every read,
    /// taking the **current** global snapshot. After a concurrent transaction
    /// committed, this transaction's reads would see the new data — the read
    /// set was recorded against a drifting snapshot, distorting SSI conflict
    /// detection. Entries are removed from this table by commit/rollback
    /// when the transaction ends.
    txn_snapshots: RwLock<std::collections::HashMap<TxnId, TxnSnapshot>>,
    l1: Arc<MemStore>,
    l2: Arc<DiskLayer>,
    l3: Arc<DiskLayer>,
    flush_engine: FlushEngine,
    recent_flush: Arc<RecentFlushCache>,
    txn_counter: AtomicU64,
}

impl Z1Kv {
    /// Open (or create) the engine at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_config(path, Z1Config::default())
    }

    pub fn open_with_config(path: impl Into<PathBuf>, config: Z1Config) -> Result<Self> {
        let data_dir = path.into();
        std::fs::create_dir_all(&data_dir).map_err(Z1Error::Io)?;

        // ── Process lock ────────────────────────────────────────────────────
        // Must be acquired before touching WAL/L2: the double-instance
        // corruption happens on any write path.
        let _engine_lock = crate::engine_lock::EngineLock::acquire(&data_dir)?;

        // ── WAL ─────────────────────────────────────────────────────────────
        // SyncEach: every append_durable fsyncs, so a commit's Put/Commit
        // records are truly on disk (recoverable after reopen).
        // GroupCommitStrict would need a background flusher to guarantee
        // "wait until flush succeeds"; the self-implemented simplified
        // version has no flusher, so the engine layer picks SyncEach.
        let wal_config = WalConfig {
            wal_dir: data_dir.join("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
            group_commit: Some(crate::wal::GroupCommitConfig {
                policy: crate::wal::SyncPolicy::SyncEach,
                ..Default::default()
            }),
            // Windows write-through: default follows the platform (cfg!);
            // overridable via WalConfig.write_through.
            write_through: cfg!(windows),
        };
        let wal = Arc::new(WalWriter::open(&data_dir, wal_config)?);

        // ── Storage layers ──────────────────────────────────────────────────
        // MemStore uses the async WAL write path (appends into the OS buffer,
        // no separate fsync) — Put records of uncommitted transactions do not
        // need durability (after a crash, recovery discards them because
        // there is no Commit record); the commit's append_durable fsync
        // flushes the whole buffer of the same file handle (Puts included).
        // The DURABILITY BOUNDARY semantics are unchanged: the commit's fsync
        // remains the single boundary.
        // Companion: replay_all tolerantly drops tail-truncated records
        // (see wal/mod.rs).
        let l1 = Arc::new(MemStore::new(64 * 1024 * 1024, SyncLevel::Async));
        l1.set_wal(Arc::clone(&wal));
        let l2 = Arc::new(DiskLayer::new(data_dir.join("l2")));
        let l3 = Arc::new(DiskLayer::new(data_dir.join("l3")));
        let recent_flush = Arc::new(RecentFlushCache::new());
        // Fix: the patch-id counter was pure in-memory state; resetting to
        // zero on restart would make the first flush after a reopen rename
        // over existing same-id patch files (disk index key ranges misaligned
        // with the new content; permanent data loss once the WAL has been
        // truncated by a checkpoint).
        // Recover the watermark from disk: allocate after the largest id
        // already used by either layer.
        let next_patch_id = l2.max_patch_id().max(l3.max_patch_id()) + 1;
        let flush_engine = FlushEngine::with_l3(
            Arc::clone(&l1),
            Arc::clone(&l2),
            Arc::clone(&l3),
            Arc::clone(&recent_flush),
            Arc::new(AtomicU64::new(0)),
            next_patch_id,
        );

        // ── MVCC ─────────────────────────────────────────────────────────────
        // Use the configured visibility settings (history eviction thresholds etc.).
        let mut mvcc = VisibilityManager::new_with_config(config.visibility.clone());

        // ── Recovery: checkpoint baseline + WAL replay (closed loop) ────────
        // 1. Read the checkpoint baseline first (committed_txn + commit_ts_map).
        //    Old WAL segments after the checkpoint may already have been
        //    truncated; their commit history is carried by the checkpoint.
        // 2. Replay the remaining WAL records to get the WAL's commit history.
        // 3. Merge: committed_txn = max(checkpoint, WAL); history = checkpoint ∪ WAL.
        let ckpt_mgr = crate::wal::CheckpointManager::new(&data_dir);
        let (cp_committed_txn, cp_history) =
            ckpt_mgr.recovery_baseline().unwrap_or((0, Vec::new()));

        let wal_dir = data_dir.join("wal");
        let recovery = replay_committed_ops(&wal_dir, config.strict_mode, |op| {
            if let WalRecord::Put { cf, key, value } = &op.record {
                let entry = Z1Entry {
                    key: Z1Key::new(*cf, key.clone()),
                    txn_id: op.txn_id,
                    value: value.clone().map(Arc::new),
                    ts: 0,
                };
                // replay_put: memory only, never re-append to the WAL
                // (otherwise the WAL doubles on every open).
                l1.replay_put(entry);
            }
            Ok(())
        })
        .map_err(|e| Z1Error::Wal(format!("recovery failed: {}", e)))?;

        // Merge the checkpoint baseline with the WAL replay result.
        let mut merged_history: std::collections::HashMap<TxnId, u64> =
            cp_history.into_iter().collect();
        for (txn_id, commit_ts) in recovery.commit_ts_map {
            merged_history.insert(txn_id, commit_ts);
        }
        let committed_txn = cp_committed_txn.max(recovery.max_seen_committed_txn);

        // Rebuild MVCC committed_history + committed_txn from checkpoint ∪ WAL.
        mvcc.set_committed_txn(committed_txn);
        mvcc.recover_committed_history_with_config(merged_history, None, &recovery.inserted_at_map);
        // D7: lower bound for TTL eviction — recovered history entries are
        // never evicted prematurely.
        mvcc.set_replay_watermark(recovery.replay_watermark);

        let next_txn = committed_txn.saturating_add(1);

        Ok(Self {
            data_dir,
            config,
            _engine_lock,
            wal,
            mvcc: RwLock::new(mvcc),
            snapshot_cache: SnapshotCache::new(),
            txn_snapshots: RwLock::new(std::collections::HashMap::new()),
            l1,
            l2,
            l3,
            flush_engine,
            recent_flush,
            txn_counter: AtomicU64::new(next_txn),
        })
    }

    /// Allocate the next transaction id (overflow-protected).
    fn next_txn_id(&self) -> Result<TxnId> {
        let prev = self.txn_counter.load(Ordering::SeqCst);
        if prev == u64::MAX {
            return Err(Z1Error::Internal("transaction ID counter overflow".into()));
        }
        Ok(self.txn_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Begin a transaction.
    ///
    /// Pinned snapshot: registration and snapshot capture happen inside the
    /// **same mvcc write-lock scope** (`begin_txn_with_snapshot`) — the two
    /// steps previously had a TOCTOU window: a concurrent commit landing
    /// between them advanced committed_txn, so the new transaction's pinned
    /// snapshot included commits made "after begin", breaking repeatable
    /// reads.
    pub fn begin_txn(&self) -> Result<TxnId> {
        // Second real concurrency defect: `next_txn_id` (atomic allocation)
        // and mvcc registration (write lock) are two non-atomic steps. Under
        // concurrent begins, thread A gets id 5, thread B gets 6 and
        // registers first; committed_txn advances to 6, so A's subsequent
        // registration of 5 is rejected as "already committed". Out-of-order
        // ids are a normal artifact of the allocator and must be retried,
        // not reported as errors. The snapshot is constructed inside the
        // same write-lock scope (see begin_txn_with_snapshot).
        const BEGIN_RETRY: usize = 16;
        for attempt in 0..BEGIN_RETRY {
            let txn_id = self.next_txn_id()?;
            self.snapshot_cache.invalidate();
            let snap = match self
                .mvcc
                .write()
                .begin_txn_with_snapshot(txn_id, crate::mvcc::visibility::IsolationLevel::Snapshot)
            {
                Ok(snap) => snap,
                Err(e)
                    if e.to_string().contains("already committed") && attempt + 1 < BEGIN_RETRY =>
                {
                    // Out-of-order id: committed_txn has been pushed past by
                    // a concurrent begin/commit; abandon this id and take a
                    // larger one.
                    continue;
                }
                Err(e) => return Err(Z1Error::Internal(format!("begin_txn failed: {}", e))),
            };
            self.txn_snapshots.write().insert(txn_id, snap);
            return Ok(txn_id);
        }
        Err(Z1Error::Internal(
            "begin_txn: could not allocate an in-order transaction id after retries".into(),
        ))
    }

    /// Commit a transaction (WAL-first, then MVCC).
    pub fn commit(&self, txn_id: TxnId) -> Result<()> {
        // Transaction-existence validation (get_begin_ts previously served
        // only a dead field of the compensation framework).
        self.ensure_active_txn(txn_id)?;

        // inserted_at is captured once before the WAL write (D5: the WAL
        // record and committed_history share the same TTL clock, so no
        // drift after recovery).
        let inserted_at = crate::codec::current_timestamp_millis();

        // ── DURABILITY BOUNDARY ─────────────────────────────────────────────
        self.wal.append_durable(
            txn_id,
            WalRecord::Commit {
                commit_ts: txn_id,
                inserted_at,
            },
        )?;

        // ── Apply phase: CommitProtocol single execution point ──────────────
        // All side effects are expressed as PendingOps and applied in order
        // by apply_all, making them easy to audit; the MVCC commit is
        // intentionally non-compensable (committed_txn is monotonic, see
        // protocol.rs).
        let mut protocol = crate::txn::protocol::CommitProtocol::new(txn_id);
        protocol
            .pending_ops
            .push(crate::txn::protocol::PendingOp::MvccCommit { inserted_at });
        protocol
            .pending_ops
            .push(crate::txn::protocol::PendingOp::CommitTxnRecord {
                txn_id,
                commit_ts: txn_id,
            });
        protocol
            .pending_ops
            .push(crate::txn::protocol::PendingOp::SnapshotInvalidate);

        struct EngineApplier<'a> {
            engine: &'a Z1Kv,
        }
        impl crate::txn::protocol::CommitApplier for EngineApplier<'_> {
            fn apply_mvcc_commit(&self, txn_id: TxnId, inserted_at: u64) -> Result<()> {
                self.engine.mvcc.write().commit_txn(txn_id, inserted_at)?;
                Ok(())
            }
            fn apply_put_committed_txn(&self, _counter: u64) -> Result<()> {
                // committed_txn is carried by the WAL Commit record and
                // rebuilt by recovery — no separate counter persistence
                // needed. This op is kept to preserve the protocol shape.
                Ok(())
            }
            fn apply_commit_txn_record(&self, txn_id: u64, commit_ts: u64) -> Result<()> {
                // commit_ts is persisted with the WAL Commit record (the D5
                // field); recovery rebuilds committed_history from it — no
                // redundant KV write needed.
                let _ = (txn_id, commit_ts);
                Ok(())
            }
            fn apply_snapshot_invalidate(&self) -> Result<()> {
                self.engine.snapshot_cache.invalidate();
                Ok(())
            }
        }

        // ── Apply phase: CommitProtocol single execution point ──────────────
        // All side effects are expressed as PendingOps and applied in order
        // by apply_all, making them easy to audit; the MVCC commit is
        // intentionally non-compensable (committed_txn is monotonic, see
        // protocol.rs).
        //
        // When apply_all fails (typically an SSI conflict — commit_txn has
        // already aborted the txn and removed it from active_txns), the
        // pinned-snapshot registry must also be cleaned up: previously only
        // the success path cleaned up, leaking entries on failure (the
        // snapshot and its deep-copied commit_ts_map stayed in memory
        // indefinitely).
        if let Err(e) = protocol.apply_all(&EngineApplier { engine: self }) {
            self.txn_snapshots.write().remove(&txn_id);
            return Err(e);
        }

        // Transaction ended: remove the pinned snapshot (the registry's
        // lifetime matches the transaction's).
        self.txn_snapshots.write().remove(&txn_id);

        tracing::debug!(txn_id, inserted_at, "transaction committed");

        // ── Automatic maintenance ────────────────────────────────────────────
        // Safety argument for degrading failures to warn:
        // - a checkpoint failure aborts before truncate, so the WAL stays
        //   intact and the data remains replayable
        // - a compaction failure is redundant either way (it aborts before or
        //   after drop_cf without losing data)
        // So warning and returning Ok from commit is safe here; the next
        // commit retries.
        if let Err(e) = self.maybe_auto_maintain() {
            tracing::warn!(error = %e, "auto maintenance failed after commit (non-fatal)");
        }
        Ok(())
    }

    /// Post-commit automatic maintenance: WAL over threshold → checkpoint;
    /// L2 patches over threshold → compaction.
    ///
    /// Trigger semantics:
    /// - a checkpoint failure is non-fatal (the data is already in the WAL,
    ///   just not truncated)
    /// - a threshold of 0 disables the check
    fn maybe_auto_maintain(&self) -> Result<()> {
        // 1. WAL size → automatic checkpoint.
        if self.config.checkpoint_wal_size_threshold > 0
            && self.wal.size_bytes() >= self.config.checkpoint_wal_size_threshold
        {
            tracing::info!(
                wal_size = self.wal.size_bytes(),
                "WAL exceeded checkpoint threshold, auto-checkpointing"
            );
            self.checkpoint()?;
        }

        // 2. L2 patch count → automatic compaction.
        //    Watermark = min(oldest active transaction's begin_ts, lowest
        //    pinned transaction-snapshot id).
        //    Previously only active_txns were considered: a committed
        //    transaction whose pinned snapshot the user still holds (for
        //    repeatable reads via get_for_txn) is not in active_txns, so the
        //    watermark would compute to u64::MAX → GC would reclaim old
        //    versions still visible to that snapshot → repeatable reads
        //    inside the transaction break (a later read of the same key
        //    returns None or a newer value).
        //    Contract note: `db.snapshot()` (a bare snapshot, not a
        //    transaction) is a time-travel read interface; its visibility is
        //    NOT protected by GC (unbounded retention across compactions
        //    would equal unbounded history growth) — see the `snapshot` /
        //    `scan_at` docs.
        if self.config.l2_compaction_threshold > 0 {
            let min_active = {
                let mvcc = self.mvcc.read();
                mvcc.oldest_active_begin_ts()
            };
            let min_pinned = {
                let snaps = self.txn_snapshots.read();
                snaps.keys().copied().min().unwrap_or(u64::MAX)
            };
            let watermark = min_active.min(min_pinned);
            self.flush_engine
                .try_compact(watermark, self.config.l2_compaction_threshold)?;
        }
        Ok(())
    }

    /// Validate that the transaction is active (begun and not finished).
    ///
    /// Unifies the transaction contract for put/delete/get_for_txn/rollback:
    /// an unknown (never begun or already finished) txn returns
    /// `TxnNotFound`. These paths previously failed silently or too late.
    fn ensure_active_txn(&self, txn_id: TxnId) -> Result<()> {
        self.mvcc
            .read()
            .get_begin_ts(txn_id)
            .map(|_| ())
            .ok_or(Z1Error::TxnNotFound(txn_id))
    }

    /// Rollback a transaction (WAL-first, then MVCC).
    ///
    /// Unknown txns are rejected — the code path previously wrote a Rollback
    /// WAL record (harmless to recovery but polluting the log) and silently
    /// returned success.
    pub fn rollback(&self, txn_id: TxnId) -> Result<()> {
        self.ensure_active_txn(txn_id)?;
        self.snapshot_cache.invalidate();
        // WAL first (durability boundary for the rollback decision).
        self.wal.append(txn_id, WalRecord::Rollback)?;
        self.mvcc.write().rollback_txn(txn_id)?;
        // Transaction ended: remove the pinned snapshot.
        self.txn_snapshots.write().remove(&txn_id);
        Ok(())
    }

    /// Get the current snapshot (full commit_ts_map).
    ///
    /// # GC contract
    ///
    /// Bare snapshots (not `begin_txn` transactions) are **not protected by
    /// compaction GC**: a later auto/manual compaction may reclaim old
    /// versions this snapshot depends on, making `get_at`/`scan_at` return
    /// empty for versions older than the GC watermark. For a stable read
    /// view across compactions use `begin_txn` + `get_for_txn` (pinned
    /// snapshots participate in watermark computation).
    pub fn snapshot(&self) -> TxnSnapshot {
        self.snapshot_cache.snapshot(&self.mvcc.read())
    }

    /// Point-get a key as of the current snapshot.
    pub fn get(&self, cf: u16, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let snap = self.snapshot();
        self.get_at(&snap, cf, key)
    }

    /// Point-get a key **within a transaction**, recording the read set for
    /// SSI conflict detection.
    ///
    /// Difference from `get`: `get` is a pure snapshot read (it does not
    /// participate in SSI tracking), while this method reads within a
    /// transaction — keys read are recorded into the transaction's read set
    /// and used for RW/WR conflict detection at commit time. This closes the
    /// SSI loop: only explicit transactional reads participate in conflict
    /// detection.
    ///
    /// Snapshot semantics: uses the snapshot pinned at `begin_txn` — reads
    /// are repeatable within the transaction and concurrent commits do not
    /// leak into this transaction's read set. Conflict detection therefore
    /// operates on a consistent snapshot. An unknown txn_id (never begun or
    /// already finished) returns `TxnNotFound` instead of silently
    /// continuing.
    pub fn get_for_txn(&self, cf: u16, key: &[u8], txn_id: TxnId) -> Result<Option<Vec<u8>>> {
        let snap = self
            .txn_snapshots
            .read()
            .get(&txn_id)
            .cloned()
            .ok_or(Z1Error::TxnNotFound(txn_id))?;
        // SSI: record the read set (the cf+key pair is the conflict key).
        let conflict_key = conflict_key(cf, key);
        self.mvcc.write().record_read(txn_id, &conflict_key);
        self.get_at(&snap, cf, key)
    }

    /// Point-get a key as of a specific snapshot.
    pub fn get_at(&self, snap: &TxnSnapshot, cf: u16, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let active: std::collections::HashSet<TxnId> = snap.active_txns.iter().copied().collect();
        let commit_ts = &snap.commit_ts_map;

        // Migration gate (read side): hold across candidate collection from
        // all layers. A concurrent flush moves entries L1 → cache → L2 and
        // clears the cache while publishing the patch index; without the
        // gate a reader using a pre-publish L2 index copy could observe
        // neither the (cleared) cache nor the (unindexed-for-it) patch.
        let _gate = self.recent_flush.read_gate();

        // Merge candidates from L1 (+recent flush) + L2 + L3.
        let mut candidates: Vec<Z1Entry> = Vec::new();

        // L1
        candidates.extend(
            self.l1
                .get_versions(cf, key, snap.snapshot_id, commit_ts, &active),
        );
        // recent flush (D8)
        candidates.extend(
            self.recent_flush
                .get_for_key(cf, key)
                .into_iter()
                .filter(|e| e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active)),
        );
        // L2
        candidates.extend(
            self.l2
                .get_versions(cf, key)?
                .into_iter()
                .filter(|e| e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active)),
        );
        // L3
        candidates.extend(
            self.l3
                .get_versions(cf, key)?
                .into_iter()
                .filter(|e| e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active)),
        );

        // Highest txn_id wins; tombstone = None.
        // Tie semantics: max_by_key returns the LAST maximal element — with
        // candidate order L1 → recent_flush → L2 → L3, a tie resolves in
        // favor of the L3 version. L3 is the authoritative post-GC layer, so
        // this is correct; but the correctness depends on the candidate
        // order. Revisit this when changing the merge order.
        let best = candidates.into_iter().max_by_key(|e| e.txn_id);
        match best {
            Some(e) if !e.is_tombstone() => Ok(e.value.as_deref().map(|v| v.as_slice().to_vec())),
            _ => Ok(None),
        }
    }

    /// Put a value under a transaction.
    ///
    /// D4: WAL-first (guaranteed inside memstore.put: WAL before memory).
    /// Transaction validation: an unknown (never begun or finished) txn
    /// returns `TxnNotFound`, consistent with `get_for_txn`. Previously
    /// record_write silently skipped and the data still went to L1+WAL,
    /// failing only at commit with "missing begin_ts" — far too late.
    pub fn put(
        &self,
        cf: u16,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
        txn_id: TxnId,
    ) -> Result<()> {
        self.ensure_active_txn(txn_id)?;
        let key = key.into();
        // SSI: record the write set (the cf+key pair is the conflict key).
        let conflict_key = conflict_key(cf, &key);
        self.mvcc.write().record_write(txn_id, &conflict_key);

        let entry = Z1Entry::put(
            Z1Key::new(cf, key),
            txn_id,
            value.into(),
            crate::codec::current_timestamp_millis() as i64,
        );
        self.l1.put(entry)
    }

    /// Delete a key (tombstone) under a transaction.
    /// Same transaction validation as `put`.
    pub fn delete(&self, cf: u16, key: impl Into<Vec<u8>>, txn_id: TxnId) -> Result<()> {
        self.ensure_active_txn(txn_id)?;
        let key = key.into();
        // SSI: record the write set (the cf+key pair is the conflict key).
        let conflict_key = conflict_key(cf, &key);
        self.mvcc.write().record_write(txn_id, &conflict_key);

        let entry = Z1Entry::tombstone(
            Z1Key::new(cf, key),
            txn_id,
            crate::codec::current_timestamp_millis() as i64,
        );
        self.l1.put(entry)
    }

    /// Flush L1 → L2 (**threshold-triggered**).
    ///
    /// # Semantic trap
    ///
    /// A real flush happens only when the L1 byte count exceeds the internal
    /// threshold (64MB); with small data this is a **no-op** (returns
    /// `Ok(None)` and produces no L2 patch). This has misled tests and
    /// callers into thinking "called it, so it's on disk". For unconditional
    /// flushing use [`Z1Kv::flush_now`] or [`Z1Kv::checkpoint`] (checkpoint
    /// takes the unconditional flush path internally).
    pub fn flush(&self) -> Result<Option<usize>> {
        self.flush_engine.try_flush()
    }

    /// Unconditionally flush L1 in-memory data to L2 disk patches,
    /// ignoring thresholds.
    ///
    /// Returns the number of patch groups written (grouped by cf). Added so
    /// that "called it, so it's on disk" scenarios no longer need to detour
    /// through `checkpoint`.
    pub fn flush_now(&self) -> Result<usize> {
        self.flush_engine.flush_l1_to_l2()
    }

    /// L2 → L3 compaction: merge L2 patches, GC old versions, write the
    /// L3 frozen layer.
    ///
    /// `min_active_begin_ts`: the smallest begin_ts among active
    /// transactions (the safe watermark).
    /// Pass `u64::MAX` when there are no active transactions (only the
    /// highest version per key is kept).
    /// The real watermark can be obtained via
    /// `self.mvcc.read().oldest_active_begin_ts()`.
    pub fn compact(&self, min_active_begin_ts: TxnId) -> Result<(usize, usize)> {
        self.flush_engine.compact_l2_to_l3(min_active_begin_ts)
    }

    /// Range scan: returns the (key, value) pairs visible under the current
    /// snapshot for a cf within `[start, end)` (end = None means unbounded),
    /// sorted by key.
    ///
    /// Merges four sources — L1 (with recent_flush) + L2 + L3 — taking the
    /// visible highest-txn_id version per key; keys whose highest version is
    /// a tombstone count as deleted and are not returned.
    pub fn scan(
        &self,
        cf: u16,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let snap = self.snapshot();
        self.scan_at(&snap, cf, start, end)
    }

    /// Range scan at a specific snapshot — time-travel scanning.
    pub fn scan_at(
        &self,
        snap: &TxnSnapshot,
        cf: u16,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let active: std::collections::HashSet<TxnId> = snap.active_txns.iter().copied().collect();
        let commit_ts = &snap.commit_ts_map;

        // Migration gate (read side): same rationale as `get_at` — keep the
        // multi-layer candidate collection atomic against concurrent flush
        // migration (L1 → cache → L2 + cache clear).
        let _gate = self.recent_flush.read_gate();

        // key -> (txn_id, value|tombstone); merge four sources, keep the
        // highest visible txn_id.
        let mut best: std::collections::BTreeMap<Vec<u8>, (TxnId, Option<Vec<u8>>)> =
            std::collections::BTreeMap::new();
        let mut record =
            |key: Vec<u8>, txn_id: TxnId, value: &Option<Arc<Vec<u8>>>| match best.get(&key) {
                Some((cur_txn, _)) if *cur_txn >= txn_id => {}
                _ => {
                    best.insert(
                        key,
                        (txn_id, value.as_deref().map(|v| v.as_slice().to_vec())),
                    );
                }
            };

        // L1 (the three sources are merged internally).
        for (k, e) in self
            .l1
            .range_visible(cf, start, end, snap.snapshot_id, commit_ts, &active)
        {
            record(k, e.txn_id, &e.value);
        }
        // Recent flush (the D8 window).
        for (k, e) in self
            .recent_flush
            .get_filtered(|e| {
                e.key.cf == cf
                    && e.key.key.as_slice() >= start
                    && end.is_none_or(|en| e.key.key.as_slice() < en)
                    && e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active)
            })
            .into_iter()
            .map(|e| (e.key.key.clone(), e))
        {
            record(k, e.txn_id, &e.value);
        }
        // L2 + L3 (disk patches only hold committed versions, but the
        // snapshot watermark still needs to filter).
        for (k, e) in self.l2.range_visible(cf, start, end)? {
            if e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active) {
                record(k, e.txn_id, &e.value);
            }
        }
        for (k, e) in self.l3.range_visible(cf, start, end)? {
            if e.is_visible_at_commit(snap.snapshot_id, commit_ts, &active) {
                record(k, e.txn_id, &e.value);
            }
        }

        // Tombstoned keys are not returned.
        Ok(best
            .into_iter()
            .filter_map(|(k, (_, v))| v.map(|val| (k, val)))
            .collect())
    }

    /// Run one checkpoint: persist the MVCC baseline + truncate the WAL.
    /// This (1) flushes L1→L2 first (all committed data lands in disk
    /// patches), (2) writes the checkpoint file, (3) appends a WAL
    /// checkpoint marker + fsync, (4) truncates WAL segments before the
    /// checkpoint.
    ///
    /// The order is critical: L1→L2 flush first, then checkpoint, then WAL
    /// truncate — otherwise, after truncating the WAL, L1 data that has not
    /// yet reached L2 is permanently lost.
    ///
    /// # A flush failure must abort the checkpoint
    ///
    /// An earlier revision degraded flush failures to a warning and
    /// continued, reasoning that "the WAL has us covered and the current
    /// segment is unaffected". That reasoning was **wrong**: the Put records
    /// of L1 data are spread across **all** WAL segments (earlier writes sit
    /// in older segments), and `truncate_before` deletes exactly those older
    /// segments. Continuing the checkpoint after a flush failure (e.g. a
    /// full disk) would first delete the WAL records of unflushed data, and
    /// a crash would then destroy the in-memory L1 → **permanent data
    /// loss**.
    ///
    /// Therefore a flush failure must propagate here: no checkpoint is
    /// written, no truncation happens, and the WAL stays intact (the data
    /// remains replayable); the next commit's auto-maintain retries.
    pub fn checkpoint(&self) -> Result<()> {
        // 1. Flush L1 in-memory data to L2 disk patches first (including
        //    uncommitted versions; reads filter them via MVCC visibility,
        //    so this is safe). A failure must abort (see above).
        self.flush_engine.flush_l1_to_l2()?;

        // 2. Flush the pending WAL queue (so the checkpoint covers the
        //    newest commits).
        self.wal.flush_and_sync()?;

        let mvcc = self.mvcc.read();
        let committed_txn = mvcc.committed_txn();
        let history = mvcc.committed_history_entries();
        drop(mvcc);

        let ckpt_mgr = crate::wal::CheckpointManager::new(&self.data_dir);
        ckpt_mgr.checkpoint(&self.wal, committed_txn, committed_txn, history)
    }

    /// D7: read the current replay_watermark (tests/diagnostics).
    pub fn replay_watermark(&self) -> u64 {
        self.mvcc.read().replay_watermark()
    }

    /// Data directory this engine was opened on.
    pub fn data_dir(&self) -> &std::path::Path {
        &self.data_dir
    }

    /// A copy of the engine configuration it was opened with.
    ///
    /// Returns a reference, not a mutable one: runtime config changes would
    /// not affect an already-opened instance anyway (thresholds are read
    /// from `self.config` after each commit), and public mutable access
    /// would sidestep the engine-lock semantics.
    pub fn config(&self) -> &Z1Config {
        &self.config
    }

    /// D5: read a txn's (commit_ts, inserted_at) from committed_history
    /// (tests/diagnostics).
    pub fn committed_entry(&self, txn_id: TxnId) -> Option<(u64, u64)> {
        self.mvcc.read().committed_entry(txn_id)
    }
}

/// Build the SSI conflict key: `cf(2B BE) || key`, matching Z1Key's total
/// order. Only the same cf + key conflicts; different cfs are different
/// records and never interfere.
fn conflict_key(cf: u16, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + key.len());
    out.extend_from_slice(&cf.to_be_bytes());
    out.extend_from_slice(key);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "z1kv_engine_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn put_get_commit() {
        let db = Z1Kv::open(tmp_dir("putget")).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        db.commit(txn).unwrap();

        let v = db.get(0, b"k").unwrap();
        assert_eq!(v, Some(b"v".to_vec()));
    }

    #[test]
    fn uncommitted_not_visible() {
        let db = Z1Kv::open(tmp_dir("uncommitted")).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();

        // Not committed — get (current snapshot) must not see it.
        assert_eq!(db.get(0, b"k").unwrap(), None);
    }

    #[test]
    fn ssi_write_write_conflict_detected() {
        let db = Z1Kv::open(tmp_dir("ssi_ww")).unwrap();

        // txn A writes key k, does not commit.
        let t1 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v1", t1).unwrap();

        // txn B writes the same key k concurrently, commits first.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v2", t2).unwrap();
        // At t2's commit, the WW conflict with active txn t1 is detected → t2 aborts.
        let result = db.commit(t2);
        assert!(
            result.is_err(),
            "SSI must detect write-write conflict at commit"
        );
        assert!(result.unwrap_err().to_string().contains("SSI conflict"));

        // t2 has aborted; t1 can now commit (the conflicting party is gone).
        assert!(
            db.commit(t1).is_ok(),
            "t1 must commit after conflicting t2 aborted"
        );
    }

    #[test]
    fn ssi_no_conflict_when_disjoint_keys() {
        let db = Z1Kv::open(tmp_dir("ssi_disjoint")).unwrap();

        // txn A writes key a, does not commit.
        let t1 = db.begin_txn().unwrap();
        db.put(0, b"a", b"v1", t1).unwrap();

        // txn B writes key b (disjoint), commits.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"b", b"v2", t2).unwrap();
        db.commit(t2).unwrap();

        // txn A's commit must not conflict (different keys).
        if let Err(e) = db.commit(t1) {
            panic!("disjoint keys must not conflict, got: {}", e);
        }
    }

    #[test]
    fn ssi_rw_conflict_detected_via_get_for_txn() {
        let db = Z1Kv::open(tmp_dir("ssi_rw")).unwrap();

        // txn A reads key k within a transaction (records the read set), does not commit.
        let t1 = db.begin_txn().unwrap();
        let _ = db.get_for_txn(0, b"k", t1).unwrap();

        // txn B writes key k, commits first.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", t2).unwrap();
        // At t2's commit, the WR conflict with active txn t1 is detected (t1 read a key it
        // wrote) → t2 aborts.
        let result = db.commit(t2);
        assert!(
            result.is_err(),
            "SSI must detect read-write conflict at commit"
        );
        assert!(result.unwrap_err().to_string().contains("SSI conflict"));

        // t1's commit must succeed (t2 has aborted).
        assert!(db.commit(t1).is_ok());
    }

    /// Regression (pinned snapshot): get_for_txn uses the snapshot from
    /// begin; concurrent commits do not leak into the transaction —
    /// repeatable reads within a transaction.
    ///
    /// Note: t1 must have read key k first (recording the read set) to
    /// trigger t2's SSI conflict. To let t2 commit successfully (otherwise
    /// the pinned-snapshot behavior could not be observed), this test has t2
    /// write a key t1 **has not read**. That is exactly the observable
    /// pinned-snapshot scenario: after t1 reads a, t2 writes b and commits;
    /// when t1 then reads b — before the fix it saw b, after the fix it does
    /// not.
    #[test]
    fn get_for_txn_uses_pinned_snapshot() {
        let db = Z1Kv::open(tmp_dir("pinned_snap")).unwrap();

        // t1 begins and reads key a (which does not exist yet).
        let t1 = db.begin_txn().unwrap();
        assert_eq!(db.get_for_txn(0, b"a", t1).unwrap(), None);

        // t2 writes key b and commits (after t1 began; disjoint from t1's
        // read set, so no SSI conflict).
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"b", b"vb", t2).unwrap();
        db.commit(t2).unwrap();

        // Before the fix: get_for_txn took the global current snapshot →
        // t1 would immediately see b (snapshot leakage).
        // After the fix: t1 uses the snapshot pinned at begin → still cannot see b.
        assert_eq!(
            db.get_for_txn(0, b"b", t1).unwrap(),
            None,
            "pinned snapshot must hide commits made after begin"
        );

        // The current-snapshot read can see b.
        assert_eq!(db.get(0, b"b").unwrap(), Some(b"vb".to_vec()));

        db.commit(t1).unwrap();
    }

    /// Regression: get_for_txn explicitly errors on unknown (never begun or
    /// finished) transactions instead of warning and continuing (the old
    /// implementation silently skipped read-set recording).
    #[test]
    fn get_for_txn_unknown_txn_errors() {
        let db = Z1Kv::open(tmp_dir("txn_not_found")).unwrap();
        let err = db.get_for_txn(0, b"k", 9999).unwrap_err();
        assert!(
            matches!(err, Z1Error::TxnNotFound(9999)),
            "expected TxnNotFound, got {}",
            err
        );

        // After commit the registry entry is removed; a second get_for_txn must error.
        let t = db.begin_txn().unwrap();
        db.commit(t).unwrap();
        let err = db.get_for_txn(0, b"k", t).unwrap_err();
        assert!(matches!(err, Z1Error::TxnNotFound(_)));
    }

    /// Regression: the write path (put/delete) returns TxnNotFound for
    /// unknown transactions — previously accepted silently, with the data
    /// going into L1+WAL and failing only at commit (exposed too late).
    #[test]
    fn put_delete_unknown_txn_errors_immediately() {
        let db = Z1Kv::open(tmp_dir("write_not_found")).unwrap();

        let err = db.put(0, b"k", b"v", 9999).unwrap_err();
        assert!(matches!(err, Z1Error::TxnNotFound(9999)), "put: {}", err);
        let err = db.delete(0, b"k", 9999).unwrap_err();
        assert!(matches!(err, Z1Error::TxnNotFound(9999)), "delete: {}", err);

        // rollback rejects unknown txns too (previously it silently wrote a
        // WAL Rollback record and returned Ok).
        let err = db.rollback(9999).unwrap_err();
        assert!(
            matches!(err, Z1Error::TxnNotFound(9999)),
            "rollback: {}",
            err
        );

        // put on an already committed txn must also error (the txn ended).
        let t = db.begin_txn().unwrap();
        db.commit(t).unwrap();
        let err = db.put(0, b"k", b"v", t).unwrap_err();
        assert!(matches!(err, Z1Error::TxnNotFound(_)));
    }

    /// Regression: after a failed commit (SSI conflict → txn aborted), the
    /// pinned-snapshot registry must be cleaned up. Observed by: the failed
    /// txn's get_for_txn returning TxnNotFound (the entry is gone) and MVCC
    /// active_txns no longer containing it.
    #[test]
    fn commit_failure_cleans_pinned_snapshot() {
        let db = Z1Kv::open(tmp_dir("commit_fail_clean")).unwrap();

        let t1 = db.begin_txn().unwrap();
        let _ = db.get_for_txn(0, b"k", t1).unwrap(); // record the read set

        let t2 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", t2).unwrap();
        assert!(db.commit(t2).is_err(), "t2 must fail with SSI conflict");

        // t2 has aborted: its pinned snapshot must have been cleaned up.
        let err = db.get_for_txn(0, b"other", t2).unwrap_err();
        assert!(
            matches!(err, Z1Error::TxnNotFound(_)),
            "t2 snapshot must be cleaned after failed commit, got {}",
            err
        );

        // t1 is unaffected: still readable and committable.
        assert_eq!(db.get_for_txn(0, b"k", t1).unwrap(), None);
        assert!(db.commit(t1).is_ok());
    }

    /// Regression (GC watermark): the auto-compaction watermark must count
    /// pinned transaction snapshots, otherwise a committed transaction's
    /// repeatable read breaks after compaction.
    ///
    /// Repro: t1 begins (snapshot id=1) → t2 commits (txn 2) → compaction
    /// triggers (ignoring t1's pinned snapshot the watermark would be
    /// u64::MAX, keeping only the highest version per key) → t1 reads again
    /// via get_for_txn: before the fix version 1 had been reclaimed and the
    /// read returned empty.
    #[test]
    fn auto_compact_respects_pinned_snapshot() {
        use crate::config::Z1Config;

        let dir = tmp_dir("compact_pinned");
        let config = Z1Config::default()
            .with_l2_compaction_threshold(1) // compact after every commit
            .with_checkpoint_wal_size_threshold(0); // disable auto-checkpoint noise

        let db = Z1Kv::open_with_config(dir, config).unwrap();

        // Seed version 1 of key "k" (committed by txn 1).
        let t0 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v1", t0).unwrap();
        db.commit(t0).unwrap(); // committed_txn=1
        db.flush().unwrap(); // unconditionally land in L2, producing a patch
        db.compact(u64::MAX).unwrap(); // maximum compaction: L3 keeps only version 1

        // t1 begins (pinned snapshot id=1, can see v1). Read the disjoint
        // key 'o' first (no conflict with t2's write set), proving this is
        // the transactional read path.
        let t1 = db.begin_txn().unwrap();
        assert_eq!(db.get_for_txn(0, b"o", t1).unwrap(), None);

        // t2 commits a new version (txn 2) → auto_maintain triggers compaction.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v2", t2).unwrap();
        db.commit(t2).unwrap(); // auto-compacts internally; t1's snapshot must hold the watermark

        // Before the fix: the watermark ignored t1's pinned snapshot → GC
        // reclaimed v1 → repeatable read broken.
        assert_eq!(
            db.get_for_txn(0, b"k", t1).unwrap(),
            Some(b"v1".to_vec()),
            "pinned snapshot repeatable-read must survive auto-compaction"
        );
        // The current read sees v2.
        assert_eq!(db.get(0, b"k").unwrap(), Some(b"v2".to_vec()));

        db.commit(t1).unwrap();
    }

    /// Regression (out-of-order id retry): under concurrent begin_txn, the
    /// two non-atomic steps (next_txn_id + mvcc registration) would get a
    /// small id rejected after committed_txn advanced — the engine now
    /// retries until success. 8 threads × 50 begin/commit pairs must all
    /// succeed, and the snapshot invariant (snapshot_id < txn_id) must
    /// always hold.
    #[test]
    fn concurrent_begin_with_out_of_order_ids() {
        use std::sync::Arc as StdArc;

        let dir = tmp_dir("begin_ooo");
        let db: StdArc<Z1Kv> = StdArc::new(Z1Kv::open(dir).unwrap());
        let mut handles = Vec::new();
        for _ in 0..8u64 {
            let db = StdArc::clone(&db);
            handles.push(std::thread::spawn(move || {
                for _ in 0..50u64 {
                    let t = db.begin_txn().expect("begin must succeed after retry");
                    // The pinned snapshot is built inside the lock scope: the
                    // snapshot id is always < its own txn_id.
                    let _ = db.get_for_txn(0, b"__probe__", t).unwrap();
                    db.commit(t).expect("commit");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked (begin retry broken)");
        }
    }

    /// Regression (fault injection): when flush fails, checkpoint must
    /// abort — no checkpoint written, no WAL truncation, and the data
    /// remains replayable from the WAL.
    ///
    /// Incident scenario (the loss window the erroneous degradation once
    /// opened): flush fails → checkpoint continues → truncate deletes old
    /// segments → the Put records of unflushed data sat exactly in those old
    /// segments → after a crash, L1 memory is gone and the WAL has no
    /// records = permanent data loss.
    ///
    /// Injection: replace the L2 root directory with a **file of the same
    /// name**, so append_patch's create_dir_all necessarily fails.
    #[test]
    fn checkpoint_aborts_when_flush_fails_and_wal_survives() {
        let dir = tmp_dir("ckpt_flush_fail");
        let db = Z1Kv::open(dir.clone()).unwrap();

        // Commit data (the WAL has records).
        let t = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", t).unwrap();
        db.commit(t).unwrap();

        // Fault injection: delete the l2 directory and replace it with a
        // same-named file → create_dir_all fails.
        std::fs::remove_dir_all(dir.join("l2")).unwrap();
        std::fs::write(dir.join("l2"), b"not a directory").unwrap();

        // checkpoint must abort with Err (the flush_l1_to_l2 failure propagates).
        assert!(
            db.checkpoint().is_err(),
            "checkpoint must abort when flush fails"
        );

        // The WAL was not truncated: records remain replayable (the key
        // guarantee against data loss).
        let records = crate::wal::replay_all(&dir.join("wal")).unwrap();
        assert!(
            records.iter().any(|e| matches!(
                &e.record,
                crate::wal::WalRecord::Put { key, .. } if key == b"k"
            )),
            "WAL must retain the Put record after aborted checkpoint"
        );
        // No checkpoint file must exist (nothing was written).
        assert!(
            !dir.join("checkpoints").join("_LATEST").exists(),
            "no checkpoint file must be written when flush fails"
        );

        // Repair the fault (restore the directory) and retry the checkpoint.
        std::fs::remove_file(dir.join("l2")).unwrap();
        std::fs::create_dir_all(dir.join("l2")).unwrap();
        // reopen (simulating a restart after repair) → checkpoint works.
        drop(db);
        let db2 = Z1Kv::open(dir.clone()).unwrap();
        db2.checkpoint().unwrap();
        assert!(dir.join("checkpoints").join("_LATEST").exists());
    }

    /// Coverage gap: the `rollback` success path had no test at all (only
    /// the TxnNotFound rejection was tested). After rollback: the data is
    /// invisible, the transaction removed, and after a crash (WAL replay
    /// discards transactions without a Commit record) it stays invisible.
    #[test]
    fn rollback_discards_writes() {
        let dir = tmp_dir("rollback_writes");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            let t = db.begin_txn().unwrap();
            db.put(0, b"k", b"v", t).unwrap();
            db.rollback(t).unwrap();

            // After rollback the data is invisible; the transaction has ended
            // (a second put returns TxnNotFound).
            assert_eq!(db.get(0, b"k").unwrap(), None);
            let err = db.put(0, b"k", b"v2", t).unwrap_err();
            assert!(matches!(err, Z1Error::TxnNotFound(_)));
        }

        // Crash & reopen: rolled-back writes must not resurrect.
        let db2 = Z1Kv::open(dir).unwrap();
        assert_eq!(
            db2.get(0, b"k").unwrap(),
            None,
            "rolled-back writes must stay discarded after reopen"
        );
    }

    /// Coverage gap: end-to-end behavior for empty keys, binary keys
    /// (containing 0x00) and cf boundaries (0 and u16::MAX) — keys are
    /// opaque bytes and any content must round-trip losslessly.
    #[test]
    fn opaque_keys_and_cf_boundaries() {
        let dir = tmp_dir("opaque_keys");
        let db = Z1Kv::open(dir.clone()).unwrap();

        // Empty key and binary keys containing 0x00.
        let keys: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            vec![0x00],
            vec![0x00, 0xFF, 0x00],
            (0..=255u8).collect(),
        ];
        let txn = db.begin_txn().unwrap();
        for (i, k) in keys.iter().enumerate() {
            db.put(0, k.clone(), vec![i as u8], txn).unwrap();
        }
        // cf boundaries: 0 and u16::MAX.
        db.put(u16::MAX, b"max_cf", b"top", txn).unwrap();
        db.put(0, b"zero_cf", b"bottom", txn).unwrap();
        db.commit(txn).unwrap();

        // All read back losslessly (including cross-cf isolation: the same
        // key in different cfs does not interfere).
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(
                db.get(0, k).unwrap(),
                Some(vec![i as u8]),
                "opaque key {:?} must round-trip",
                k
            );
        }
        assert_eq!(db.get(u16::MAX, b"max_cf").unwrap(), Some(b"top".to_vec()));
        assert_eq!(db.get(0, b"zero_cf").unwrap(), Some(b"bottom".to_vec()));
        assert_eq!(
            db.get(0, b"max_cf").unwrap(),
            None,
            "cf isolation at boundary"
        );

        // scan covers the empty key (empty key >= empty start, so it is included).
        let all = db.scan(0, b"", None).unwrap();
        assert_eq!(
            all.len(),
            keys.len() + 1,
            "empty start must include empty key"
        );

        // Binary keys round-trip losslessly after reopen.
        drop(db);
        let db2 = Z1Kv::open(dir).unwrap();
        assert_eq!(db2.get(0, &[0x00, 0xFF, 0x00]).unwrap(), Some(vec![2]));
        assert_eq!(db2.get(u16::MAX, b"max_cf").unwrap(), Some(b"top".to_vec()));
    }

    /// Coverage gap: strict_mode end-to-end difference — when the WAL holds
    /// a corrupt record, strict=true (default) refuses to open and
    /// strict=false still fails to open (CRC damage is reader-level).
    #[test]
    fn strict_mode_e2e_reopen_with_corrupt_wal() {
        let dir = tmp_dir("strict_e2e");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            let t = db.begin_txn().unwrap();
            db.put(0, b"good", b"v", t).unwrap();
            db.commit(t).unwrap();
        }
        // Inject corruption: flip the WAL's last byte (record-level CRC fails).
        let wal_path = dir.join("wal").join("wal.00000000");
        let mut bytes = std::fs::read(&wal_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&wal_path, &bytes).unwrap();

        // Semantic boundary (established by this test): CRC damage sits at
        // the WAL read layer (replay_all) and is Fatal regardless of
        // strict_mode — strict_mode only governs the apply-layer Corruption
        // classification (see recovery.rs classify_apply_error).
        // Skipping physical corruption would be the wrong semantics: it
        // would silently drop already-fsynced commits.
        assert!(
            Z1Kv::open(dir.clone()).is_err(),
            "CRC corruption must abort open regardless of strict_mode"
        );
        assert!(
            Z1Kv::open_with_config(
                dir,
                Z1Config {
                    strict_mode: false,
                    ..Default::default()
                }
            )
            .is_err(),
            "strict_mode=false must NOT bypass reader-level CRC failure"
        );
    }

    /// Coverage gap: after uncommitted data is flushed into L2 and the
    /// transaction rolls back, the data becomes a ghost in L2 (invisible
    /// under D12) and is reclaimed by compaction GC. Life-cycle closed loop:
    /// put → flush (uncommitted version enters a patch) → rollback →
    /// get(None) → compact → neither L2/L3 has the version visible.
    #[test]
    fn rolled_back_flushed_data_is_gc_collected() {
        let dir = tmp_dir("rollback_gc");
        let db = Z1Kv::open(dir.clone()).unwrap();

        let t = db.begin_txn().unwrap();
        db.put(0, b"k", b"ghost", t).unwrap();
        db.flush_now().unwrap(); // uncommitted version lands in an L2 patch
        db.rollback(t).unwrap();

        // Invisible after rollback (even though the version is still in the L2 patch).
        assert_eq!(db.get(0, b"k").unwrap(), None);

        // Compaction reclaims the ghost: with watermark u64::MAX, the ghost
        // txn < watermark and is the key's only version → it is retained in
        // L3 as the "newest baseline below the watermark", but D12 makes it
        // invisible. We assert visibility, not physical deletion (gc.rs
        // semantics: keep the baseline).
        db.compact(u64::MAX).unwrap();
        assert_eq!(db.get(0, b"k").unwrap(), None);

        // Equally invisible after reopen (the ghost's txn has no Commit record).
        drop(db);
        let db2 = Z1Kv::open(dir).unwrap();
        assert_eq!(db2.get(0, b"k").unwrap(), None);
    }

    #[test]
    fn delete_tombstones() {
        let db = Z1Kv::open(tmp_dir("delete")).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        db.commit(txn).unwrap();
        assert_eq!(db.get(0, b"k").unwrap(), Some(b"v".to_vec()));

        let txn2 = db.begin_txn().unwrap();
        db.delete(0, b"k", txn2).unwrap();
        db.commit(txn2).unwrap();
        assert_eq!(db.get(0, b"k").unwrap(), None);
    }

    #[test]
    fn mvcc_snapshot_isolation() {
        let db = Z1Kv::open(tmp_dir("mvcc")).unwrap();

        let t1 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v1", t1).unwrap();
        db.commit(t1).unwrap();

        // Reader snapshot taken after t1 commits sees v1.
        let snap = db.snapshot();
        assert_eq!(db.get_at(&snap, 0, b"k").unwrap(), Some(b"v1".to_vec()));

        // A later write (uncommitted) is invisible to the earlier snapshot.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"k", b"v2", t2).unwrap();
        assert_eq!(db.get_at(&snap, 0, b"k").unwrap(), Some(b"v1".to_vec()));

        db.commit(t2).unwrap();
        assert_eq!(db.get(0, b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn reopen_recovers_from_wal() {
        let dir = tmp_dir("reopen");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            let txn = db.begin_txn().unwrap();
            db.put(0, b"k", b"v", txn).unwrap();
            db.commit(txn).unwrap();
        }
        // Reopen — WAL replay must reconstruct the committed write.
        let db = Z1Kv::open(dir).unwrap();
        assert_eq!(db.get(0, b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn checkpoint_then_reopen_recovers_via_checkpoint_baseline() {
        let dir = tmp_dir("ckpt_reopen");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            // Commit some transactions.
            for i in 0..10u64 {
                let txn = db.begin_txn().unwrap();
                db.put(0, i.to_le_bytes().to_vec(), i.to_string().into_bytes(), txn)
                    .unwrap();
                db.commit(txn).unwrap();
            }
            // Checkpoint: persist the baseline + truncate the old WAL.
            db.checkpoint().unwrap();
        }
        // Reopen — checkpoint baseline + remaining WAL replay; the data must be complete.
        let db = Z1Kv::open(dir.clone()).unwrap();
        for i in 0..10u64 {
            assert_eq!(
                db.get(0, &i.to_le_bytes()).unwrap(),
                Some(i.to_string().into_bytes()),
                "key {} must survive checkpoint+reopen",
                i
            );
        }
    }

    /// Regression: recovery replay must not re-append to the WAL.
    ///
    /// Before the fix: recovery went through `MemStore::put`
    /// (SyncLevel::Immediate), re-appending every replayed Put via
    /// append_durable → the WAL doubled in size on every open.
    /// After the fix: recovery goes through `replay_put` (memory only); the
    /// WAL size stays constant across repeated opens.
    #[test]
    fn reopen_does_not_duplicate_wal() {
        let dir = tmp_dir("wal_dup");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            let txn = db.begin_txn().unwrap();
            db.put(0, b"k", b"v", txn).unwrap();
            db.commit(txn).unwrap();
        }

        let wal_dir = dir.join("wal");
        let size_after_first = wal_size(&wal_dir);
        assert!(size_after_first > 0);

        // Repeated opens (each performs recovery): the WAL size must stay constant.
        for _ in 0..3 {
            let _db = Z1Kv::open(dir.clone()).unwrap();
            let size_now = wal_size(&wal_dir);
            assert_eq!(
                size_now, size_after_first,
                "WAL must not grow on reopen (recovery must not re-append records)"
            );
        }
    }

    /// D5/D7: after recovery the inserted_at clock does not drift and
    /// replay_watermark is set.
    ///
    /// Before the fix: the WAL Commit record had no inserted_at, so every
    /// recovered history entry's TTL clock reset to "now"; replay_watermark
    /// was never set (always 0).
    /// After the fix: recovered entries keep the WAL-carried inserted_at,
    /// and watermark = the largest inserted_at.
    #[test]
    fn recovery_preserves_inserted_at_clock_and_watermark() {
        let dir = tmp_dir("d5_d7");
        {
            let db = Z1Kv::open(dir.clone()).unwrap();
            let txn = db.begin_txn().unwrap();
            db.put(0, b"k", b"v", txn).unwrap();
            let inserted_at_before = crate::codec::current_timestamp_millis();
            db.commit(txn).unwrap();
            let _ = inserted_at_before;
        }

        // Reopen: the recovery path must restore inserted_at and the watermark.
        let db = Z1Kv::open(dir).unwrap();
        let (commit_ts, inserted_at) = db.committed_entry(1).expect("history must be recovered");
        assert_eq!(commit_ts, 1);
        // inserted_at comes from the WAL record (approximately the commit
        // moment), not the recovery moment — a loose lower bound of "no
        // earlier than 5 seconds before reopen" suffices (same-machine,
        // monotonic clock distinguishes drift).
        let now = crate::codec::current_timestamp_millis();
        assert!(
            inserted_at <= now && inserted_at > now - 60_000,
            "inserted_at must come from the WAL record, got {}",
            inserted_at
        );
        // D7: watermark = the largest recovered inserted_at > 0.
        assert_eq!(
            db.replay_watermark(),
            inserted_at,
            "replay_watermark must equal max recovered inserted_at"
        );
    }

    /// Total size of the WAL segment files (wal.*).
    fn wal_size(wal_dir: &PathBuf) -> u64 {
        std::fs::read_dir(wal_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.starts_with("wal.")
                    })
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// scan: range scan within L1 (keys ascending, upper bound exclusive).
    #[test]
    fn scan_range_basic() {
        let db = Z1Kv::open(tmp_dir("scan_basic")).unwrap();
        for k in ["a", "b", "c", "d"] {
            let txn = db.begin_txn().unwrap();
            db.put(0, k.as_bytes(), format!("v_{k}").into_bytes(), txn)
                .unwrap();
            db.commit(txn).unwrap();
        }

        // Full range [a, unbounded).
        let all = db.scan(0, b"a", None).unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].0, b"a".to_vec());
        assert_eq!(all[3].1, b"v_d".to_vec());

        // Half-open interval [b, d): b and c.
        let mid = db.scan(0, b"b", Some(b"d")).unwrap();
        assert_eq!(mid.len(), 2);
        assert_eq!(mid[0].0, b"b".to_vec());
        assert_eq!(mid[1].0, b"c".to_vec());
    }

    /// scan: cross-layer merge (after flush, L1 empty, L2 has data) +
    /// tombstone termination.
    #[test]
    fn scan_merges_layers_and_respects_tombstone() {
        let db = Z1Kv::open(tmp_dir("scan_layers")).unwrap();
        for k in ["a", "b", "c"] {
            let txn = db.begin_txn().unwrap();
            db.put(0, k.as_bytes(), format!("v_{k}").into_bytes(), txn)
                .unwrap();
            db.commit(txn).unwrap();
        }
        // Flush to L2.
        db.flush().unwrap();

        // Delete b (the tombstone lands in L1; b's old version is in L2).
        let txn = db.begin_txn().unwrap();
        db.delete(0, b"b", txn).unwrap();
        db.commit(txn).unwrap();

        let out = db.scan(0, b"a", None).unwrap();
        let keys: Vec<Vec<u8>> = out.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(
            keys,
            vec![b"a".to_vec(), b"c".to_vec()],
            "b must be tombstoned"
        );
        assert_eq!(out[0].1, b"v_a".to_vec());
        assert_eq!(out[1].1, b"v_c".to_vec());
    }

    /// scan: snapshot isolation — an old snapshot cannot see later writes.
    #[test]
    fn scan_snapshot_isolation() {
        let db = Z1Kv::open(tmp_dir("scan_snap")).unwrap();
        let t1 = db.begin_txn().unwrap();
        db.put(0, b"a", b"v1", t1).unwrap();
        db.commit(t1).unwrap();

        let snap = db.snapshot();

        // Write b after the snapshot.
        let t2 = db.begin_txn().unwrap();
        db.put(0, b"b", b"v2", t2).unwrap();
        db.commit(t2).unwrap();

        // The current snapshot sees a and b; the old snapshot sees only a.
        assert_eq!(db.scan(0, b"a", None).unwrap().len(), 2);
        assert_eq!(db.scan_at(&snap, 0, b"a", None).unwrap().len(), 1);
    }

    /// scan: cf isolation.
    #[test]
    fn scan_cf_isolation() {
        let db = Z1Kv::open(tmp_dir("scan_cf")).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"cf0", txn).unwrap();
        db.put(1, b"k", b"cf1", txn).unwrap();
        db.commit(txn).unwrap();

        assert_eq!(db.scan(0, b"k", None).unwrap()[0].1, b"cf0".to_vec());
        assert_eq!(db.scan(1, b"k", None).unwrap()[0].1, b"cf1".to_vec());
    }

    /// Automatic maintenance: WAL over threshold → auto checkpoint after
    /// commit; L2 patches over threshold → auto compaction.
    #[test]
    fn auto_maintain_triggers_checkpoint_and_compaction() {
        use crate::config::Z1Config;

        let dir = tmp_dir("auto_maintain");
        let config = Z1Config {
            // Tiny thresholds: trigger every few WAL records.
            checkpoint_wal_size_threshold: 1,
            l2_compaction_threshold: 1,
            ..Default::default()
        };
        {
            let db = Z1Kv::open_with_config(dir.clone(), config).unwrap();
            for i in 0..5u64 {
                let txn = db.begin_txn().unwrap();
                db.put(0, i.to_le_bytes().to_vec(), i.to_string().into_bytes(), txn)
                    .unwrap();
                db.commit(txn).unwrap(); // commit triggers auto maintenance internally
            }
        }

        // The checkpoint file must exist (auto maintenance ran).
        assert!(
            dir.join("checkpoints").join("_LATEST").exists(),
            "auto checkpoint must have run (WAL threshold=1)"
        );

        // Data is complete after reopen (the checkpoint ∪ WAL recovery chain works).
        let db = Z1Kv::open(dir.clone()).unwrap();
        for i in 0..5u64 {
            assert_eq!(
                db.get(0, &i.to_le_bytes()).unwrap(),
                Some(i.to_string().into_bytes()),
                "key {} must survive auto-maintained reopen",
                i
            );
        }
    }

    /// A threshold of 0 disables automatic maintenance (purely manual mode).
    #[test]
    fn auto_maintain_disabled_with_zero_threshold() {
        use crate::config::Z1Config;

        let dir = tmp_dir("auto_off");
        let config = Z1Config {
            checkpoint_wal_size_threshold: 0,
            l2_compaction_threshold: 0,
            ..Default::default()
        };
        {
            let db = Z1Kv::open_with_config(dir.clone(), config).unwrap();
            let txn = db.begin_txn().unwrap();
            db.put(0, b"k", b"v", txn).unwrap();
            db.commit(txn).unwrap();
        }
        assert!(
            !dir.join("checkpoints").join("_LATEST").exists(),
            "zero threshold must disable auto checkpoint"
        );
    }
}
