//! L1: ping-pong memstore with ArcSwap RCU reads.
//!
//! The L1 memstore.
//!
//! - Key shape: `(Z1Key, txn_id)` — column family + user key + version
//! - Ping-pong hot/cold double buffering with an ArcSwap snapshot (RCU reads)
//! - The three SyncLevel tiers (Immediate/Batch/Async), with WAL writes
//!   converging on one function
//!
//! # Read path
//!
//! `get_visible(cf, key, snapshot_txn)` collects candidates from hot + cold +
//! snapshot and returns the key's visible version (via
//! `Z1Entry::is_visible_at_commit`).
//!
//! # Invariants
//!
//! - After `swap_and_drain`, the snapshot keeps the old hot buffer so
//!   readers inside the flush window continue to see the data

use crate::store::config::SyncLevel;
use crate::store::types::{Z1Entry, Z1Key};
use crate::wal::{WalRecord, WalWriter};
use crate::TxnId;
use arc_swap::ArcSwap;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Internal key: (Z1Key, txn_id) — the txn_id disambiguates multiple versions
/// of the same user key in the same L1 buffer.
type MemKey = (Z1Key, TxnId);

/// Snapshot type: ArcSwap<BTreeMap<MemKey, Z1Entry>> for RCU reads.
type Snapshot = ArcSwap<BTreeMap<MemKey, Z1Entry>>;

/// L1 in-memory versioned store.
pub struct MemStore {
    /// hot: accepts new writes. cold: drained by flush.
    hot: Mutex<BTreeMap<MemKey, Z1Entry>>,
    cold: Mutex<BTreeMap<MemKey, Z1Entry>>,
    /// ArcSwap snapshot — updated on every `swap_and_drain`.
    snapshot: Snapshot,
    /// Estimated size of the hot buffer in bytes.
    hot_size_bytes: AtomicU64,
    /// Flush threshold in bytes.
    flush_threshold: usize,
    /// Sync level for WAL writes.
    sync_level: SyncLevel,
    /// WAL writer (shared; None disables durability).
    wal: Mutex<Option<Arc<WalWriter>>>,
    /// Batch mode: number of writes since the last fsync (for lazy-fsync checks).
    batch_pending: AtomicU64,
}

impl MemStore {
    pub fn new(flush_threshold: usize, sync_level: SyncLevel) -> Self {
        Self {
            hot: Mutex::new(BTreeMap::new()),
            cold: Mutex::new(BTreeMap::new()),
            snapshot: ArcSwap::from_pointee(BTreeMap::new()),
            hot_size_bytes: AtomicU64::new(0),
            flush_threshold,
            sync_level,
            wal: Mutex::new(None),
            batch_pending: AtomicU64::new(0),
        }
    }

    /// Attach a WAL writer (used for durability by `put`).
    pub fn set_wal(&self, wal: Arc<WalWriter>) {
        *self.wal.lock() = Some(wal);
    }

    /// Insert a versioned entry.
    ///
    /// SyncLevel::Immediate → WAL append_durable before memory insert.
    /// SyncLevel::Batch → queue via WAL group-commit (append_durable queues).
    /// SyncLevel::Async → memory only (WAL optional, no fsync).
    pub fn put(&self, entry: Z1Entry) -> crate::error::Result<()> {
        // WAL durability first (D4: WAL is durable BEFORE L2 write; for L1
        // the memory insert is the L1 write, so WAL must precede it).
        if let Some(wal) = self.wal.lock().as_ref() {
            let record = match &entry.value {
                Some(v) => WalRecord::Put {
                    cf: entry.key.cf,
                    key: entry.key.key.clone(),
                    value: Some((**v).clone()),
                },
                None => WalRecord::Put {
                    cf: entry.key.cf,
                    key: entry.key.key.clone(),
                    value: None,
                },
            };
            match self.sync_level {
                SyncLevel::Immediate => wal.append_durable(entry.txn_id, record)?,
                SyncLevel::Batch { ms, max_pending } => {
                    // Lazy fsync (Batch semantics): the append lands in the OS
                    // file buffer (the data is in the WAL file; a crash may
                    // lose the last `ms` milliseconds — the accepted Batch
                    // trade-off). When the write count since the last fsync
                    // reaches `max_pending`, or `ms` has elapsed since it,
                    // an fsync fires (lazily, from the writer; no background
                    // thread — a Drop flush is the safety net).
                    self.batch_pending.fetch_add(1, Ordering::Relaxed);
                    wal.append(entry.txn_id, record)?;
                    if self.batch_pending.load(Ordering::Relaxed) >= max_pending as u64
                        || wal.ms_since_last_flush() >= ms
                    {
                        wal.flush_and_sync()?;
                        self.batch_pending.store(0, Ordering::Relaxed);
                    }
                }
                SyncLevel::Async => {
                    // best-effort, no fsync
                    let _ = wal.append(entry.txn_id, record);
                }
            }
        }

        let key = (entry.key.clone(), entry.txn_id);
        let est = estimate_size(&entry);
        if entry.txn_id == 65 {
            eprintln!("[trace] put txn65 into hot");
        }
        self.hot.lock().insert(key, entry);
        self.hot_size_bytes.fetch_add(est, Ordering::Relaxed);
        Ok(())
    }

    /// Replay an entry from WAL recovery — memory insert ONLY, no WAL append.
    ///
    /// Insert an entry during recovery replay.
    ///
    /// Key difference from `put`: replayed data is **already durable in the
    /// WAL**. Going through `put` would append the same record again — the
    /// WAL would double in size on every open. This method inserts into the
    /// hot buffer directly, bypassing the WAL write path.
    pub fn replay_put(&self, entry: Z1Entry) {
        let key = (entry.key.clone(), entry.txn_id);
        let est = estimate_size(&entry);
        if entry.txn_id == 65 {
            eprintln!("[trace] put txn65 into hot");
        }
        self.hot.lock().insert(key, entry);
        self.hot_size_bytes.fetch_add(est, Ordering::Relaxed);
    }

    /// Returns true if the store should be flushed (threshold exceeded).
    pub fn should_flush(&self) -> bool {
        self.hot_size_bytes.load(Ordering::Relaxed) as usize > self.flush_threshold
    }

    /// Atomically swap hot and cold buffers. Returns the drained (old cold) entries.
    /// Atomically swap hot and cold buffers. Returns the drained (old cold) entries.
    pub fn swap_and_drain(&self) -> Vec<Z1Entry> {
        // ROOT FIX (lost-read race): `old_hot` must be TAKEN, not cloned.
        //
        // The old clone-then-swap sequence had a fatal interleaving with
        // `put`:
        //   1. flusher clones hot (does NOT contain a yet-to-be-put entry)
        //   2. writer puts txn N into hot and commits
        //   3. flusher swaps hot→cold, drains cold (WITHOUT txn N), clears
        //      cold, and publishes `snapshot = old_hot` (WITHOUT txn N)
        //   4. the committed entry txn N is now in NO read source (hot was
        //      replaced, cold cleared, snapshot stale) → it is invisible
        //      until reopen, even though its WAL record is durable.
        // Taking the whole hot map under the hot lock serializes put vs
        // migration: a put either lands before the take (migrated + cached)
        // or after (lives in the fresh hot buffer). There is no third
        // interleaving.
        let old_hot: BTreeMap<MemKey, Z1Entry> = {
            let mut hot = self.hot.lock();
            let taken = std::mem::take(&mut *hot);
            if taken.keys().any(|(_, t)| *t == 65) {
                eprintln!("[trace] swap drained txn65 ({} entries)", taken.len());
            }
            taken
        };

        // `drained` = everything migrated out of the read path. Since the
        // take above leaves hot empty and every prior migration already
        // carried the previous cold contents (each flush drains cold
        // completely), the drained set is exactly `old_hot` here. Keeping
        // this a pure take (no hot↔cold swap) preserves the put-vs-migration
        // atomicity: a put either lands before the take (migrated + cached)
        // or after (lives in the fresh hot buffer).
        let drained: Vec<Z1Entry> = old_hot.values().cloned().collect();

        self.hot_size_bytes.store(0, Ordering::Relaxed);
        // Snapshot = old hot entries (visible during flush window).
        self.snapshot.store(Arc::new(old_hot));
        drained
    }

    /// Get the visible version of a key at a snapshot, given authoritative
    /// MVCC commit/active state (D12 strict visibility).
    ///
    /// Read path: merge candidates from hot + cold + snapshot and return
    /// the visible version with the highest txn_id.
    pub fn get_visible(
        &self,
        cf: u16,
        key: &[u8],
        snapshot_txn: TxnId,
        commit_ts_by_txn: &std::collections::HashMap<TxnId, u64>,
        active_txns: &std::collections::HashSet<TxnId>,
    ) -> Option<Z1Entry> {
        let zk = Z1Key::new(cf, key);
        let mut best: Option<Z1Entry> = None;

        let hot = self.hot.lock();
        let cold = self.cold.lock();
        let snap = self.snapshot.load_full();

        for entry in hot.values().chain(cold.values()).chain(snap.values()) {
            if entry.key != zk {
                continue;
            }
            if !entry.is_visible_at_commit(snapshot_txn, commit_ts_by_txn, active_txns) {
                continue;
            }
            if best.as_ref().is_none_or(|b| entry.txn_id > b.txn_id) {
                best = Some(entry.clone());
            }
        }

        best
    }

    /// Get all visible versions of a key across all three sources (for scan).
    pub fn get_versions(
        &self,
        cf: u16,
        key: &[u8],
        snapshot_txn: TxnId,
        commit_ts_by_txn: &std::collections::HashMap<TxnId, u64>,
        active_txns: &std::collections::HashSet<TxnId>,
    ) -> Vec<Z1Entry> {
        let zk = Z1Key::new(cf, key);
        let hot = self.hot.lock();
        let cold = self.cold.lock();
        let snap = self.snapshot.load_full();

        let mut out: Vec<Z1Entry> = hot
            .values()
            .chain(cold.values())
            .chain(snap.values())
            .filter(|e| e.key == zk)
            .filter(|e| e.is_visible_at_commit(snapshot_txn, commit_ts_by_txn, active_txns))
            .cloned()
            .collect();
        out.sort_by_key(|e| e.txn_id);
        out
    }

    /// Number of entries in the hot buffer.
    pub fn len(&self) -> usize {
        self.hot.lock().len()
    }

    /// Returns `true` if the hot buffer contains no entries.
    pub fn is_empty(&self) -> bool {
        self.hot.lock().is_empty()
    }

    /// Collect committed versions for a column family within the key range
    /// `[start, end)` (start inclusive, end exclusive; `end = None` means
    /// unbounded). For each key, only the **visible version with the highest
    /// txn_id** under the snapshot is returned.
    ///
    /// Merges hot + cold + snapshot, filters by visibility, and keeps one
    /// entry per key. Returns `(key_bytes, entry)` pairs sorted by key.
    pub fn range_visible(
        &self,
        cf: u16,
        start: &[u8],
        end: Option<&[u8]>,
        snapshot_txn: TxnId,
        commit_ts_by_txn: &std::collections::HashMap<TxnId, u64>,
        active_txns: &std::collections::HashSet<TxnId>,
    ) -> Vec<(Vec<u8>, Z1Entry)> {
        use std::collections::BTreeMap as StdBTreeMap;

        let hot = self.hot.lock();
        let cold = self.cold.lock();
        let snap = self.snapshot.load_full();

        // key -> highest visible version (BTreeMap keeps keys ascending).
        let mut best: StdBTreeMap<Vec<u8>, Z1Entry> = StdBTreeMap::new();

        for entry in hot.values().chain(cold.values()).chain(snap.values()) {
            if entry.key.cf != cf {
                continue;
            }
            if entry.key.key.as_slice() < start {
                continue;
            }
            if let Some(e) = end {
                if entry.key.key.as_slice() >= e {
                    continue;
                }
            }
            if !entry.is_visible_at_commit(snapshot_txn, commit_ts_by_txn, active_txns) {
                continue;
            }
            match best.get(&entry.key.key) {
                Some(cur) if cur.txn_id >= entry.txn_id => {}
                _ => {
                    best.insert(entry.key.key.clone(), entry.clone());
                }
            }
        }

        best.into_iter().collect()
    }

    /// Estimated size in bytes of the hot buffer.
    ///
    /// Not used by the production path (the flush threshold reads the
    /// internal `hot_size_bytes` counter); kept for tests and diagnostics.
    /// Zero-reference methods (`shutdown`, `all_entries`, ...) have been
    /// removed.
    pub fn size_bytes(&self) -> u64 {
        self.hot_size_bytes.load(Ordering::Relaxed)
    }
}

impl Default for MemStore {
    fn default() -> Self {
        Self::new(
            64 * 1024 * 1024,
            SyncLevel::Batch {
                ms: 100,
                max_pending: 1000,
            },
        )
    }
}

fn estimate_size(entry: &Z1Entry) -> u64 {
    let key_size = entry.key.key.len() as u64 + 2;
    let value_size = entry.value.as_ref().map(|v| v.len() as u64).unwrap_or(0);
    key_size + value_size + 32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn entry(cf: u16, key: &[u8], txn_id: TxnId, value: Option<Vec<u8>>) -> Z1Entry {
        Z1Entry {
            key: Z1Key::new(cf, key),
            txn_id,
            value: value.map(Arc::new),
            ts: 0,
        }
    }

    #[test]
    fn put_and_get_visible() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v1".to_vec()))).unwrap();

        let mut commit = HashMap::new();
        commit.insert(1, 1);
        let active = HashSet::new();

        let v = store.get_visible(0, b"k", 1, &commit, &active).unwrap();
        assert_eq!(v.value.as_deref().map(|v| v.as_slice()), Some(&b"v1"[..]));
    }

    #[test]
    fn highest_txn_wins() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v1".to_vec()))).unwrap();
        store.put(entry(0, b"k", 2, Some(b"v2".to_vec()))).unwrap();

        let mut commit = HashMap::new();
        commit.insert(1, 1);
        commit.insert(2, 2);
        let active = HashSet::new();

        let v = store.get_visible(0, b"k", 2, &commit, &active).unwrap();
        assert_eq!(v.value.as_deref().map(|v| v.as_slice()), Some(&b"v2"[..]));
    }

    #[test]
    fn tombstone_is_not_visible_value() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v".to_vec()))).unwrap();
        store.put(entry(0, b"k", 2, None)).unwrap();

        let mut commit = HashMap::new();
        commit.insert(1, 1);
        commit.insert(2, 2);
        let active = HashSet::new();

        let v = store.get_visible(0, b"k", 2, &commit, &active).unwrap();
        assert!(v.is_tombstone());
    }

    #[test]
    fn uncommitted_txn_not_visible() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v".to_vec()))).unwrap();

        let commit = HashMap::new(); // no commit history
        let active = HashSet::new();

        assert!(store.get_visible(0, b"k", 1, &commit, &active).is_none());
    }

    /// Regression: after swap_and_drain, cold must be emptied — the old
    /// implementation left old_hot stranded in cold, duplicating data across
    /// L2 patches / recent_flush with unbounded growth.
    #[test]
    fn swap_and_drain_empties_cold_buffer() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v".to_vec()))).unwrap();
        let drained = store.swap_and_drain();
        assert_eq!(drained.len(), 1, "old_hot must be returned for caching");
        // Swap again: if cold has leftovers, `drained` will be non-empty.
        let drained2 = store.swap_and_drain();
        assert!(drained2.is_empty(), "cold buffer must be empty after drain");
    }

    #[test]
    fn swap_and_drain_preserves_snapshot_for_readers() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"v".to_vec()))).unwrap();

        // Simulate flush: swap hot -> cold.
        // Semantics: swap_and_drain returns ALL data moved out of the read
        // path (including old_hot) — the caller immediately writes it to
        // recent_flush, closing the drain→cache visibility vacuum.
        let drained = store.swap_and_drain();
        assert_eq!(drained.len(), 1, "old_hot must be returned for caching");
        let mut commit = HashMap::new();
        commit.insert(1, 1);
        let active = HashSet::new();

        // Reader during flush window must still see the entry via snapshot.
        let v = store.get_visible(0, b"k", 1, &commit, &active).unwrap();
        assert_eq!(v.value.as_deref().map(|v| v.as_slice()), Some(&b"v"[..]));

        // swap_and_drain has taken everything in one shot (cold is always
        // empty now); the snapshot keeps serving readers inside the window.
        let v2 = store.get_visible(0, b"k", 1, &commit, &active).unwrap();
        assert_eq!(v2.value.as_deref().map(|v| v.as_slice()), Some(&b"v"[..]));
    }

    #[test]
    fn keys_are_separated_by_cf() {
        let store = MemStore::new(1024, SyncLevel::Async);
        store.put(entry(0, b"k", 1, Some(b"cf0".to_vec()))).unwrap();
        store.put(entry(1, b"k", 1, Some(b"cf1".to_vec()))).unwrap();

        let mut commit = HashMap::new();
        commit.insert(1, 1);
        let active = HashSet::new();

        let v0 = store.get_visible(0, b"k", 1, &commit, &active).unwrap();
        let v1 = store.get_visible(1, b"k", 1, &commit, &active).unwrap();
        assert_eq!(v0.value.as_deref().map(|v| v.as_slice()), Some(&b"cf0"[..]));
        assert_eq!(v1.value.as_deref().map(|v| v.as_slice()), Some(&b"cf1"[..]));
    }

    /// SyncLevel::Batch lazy fsync: reaching `max_pending` writes triggers an fsync.
    #[test]
    fn batch_mode_flushes_when_queue_full() {
        use crate::wal::{GroupCommitConfig, WalConfig};

        let dir = std::env::temp_dir().join(format!("z1kv_mem_batch_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let wal = Arc::new(
            WalWriter::open(
                &dir,
                WalConfig {
                    wal_dir: dir.join("wal"),
                    enabled: true,
                    group_commit: Some(GroupCommitConfig {
                        policy: crate::wal::SyncPolicy::GroupCommitStrict,
                        max_batch_size: 100,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        // max_pending = 3: the third put triggers the lazy fsync.
        let store = MemStore::new(
            1024,
            SyncLevel::Batch {
                ms: 10_000,
                max_pending: 3,
            },
        );
        store.set_wal(wal.clone());

        for i in 0..3u64 {
            store
                .put(entry(
                    0,
                    format!("k{i}").as_bytes(),
                    i + 1,
                    Some(b"v".to_vec()),
                ))
                .unwrap();
        }

        // The record is written and fsynced (ms_since_last_flush just reset).
        assert!(
            wal.ms_since_last_flush() < 10_000,
            "fsync must have just run"
        );
        let records = crate::wal::replay_all(&dir.join("wal")).unwrap();
        assert_eq!(records.len(), 3);
    }

    /// SyncLevel::Batch lazy fsync: a timeout (ms) triggers an fsync.
    #[test]
    fn batch_mode_flushes_on_timeout() {
        use crate::wal::{GroupCommitConfig, WalConfig};

        let dir = std::env::temp_dir().join(format!("z1kv_mem_batch_to_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let wal = Arc::new(
            WalWriter::open(
                &dir,
                WalConfig {
                    wal_dir: dir.join("wal"),
                    enabled: true,
                    group_commit: Some(GroupCommitConfig {
                        policy: crate::wal::SyncPolicy::GroupCommitStrict,
                        max_batch_size: 100,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        // ms = 30ms: write one entry, wait for the timeout, then the second
        // write triggers the lazy fsync.
        let store = MemStore::new(
            1024,
            SyncLevel::Batch {
                ms: 30,
                max_pending: 1000,
            },
        );
        store.set_wal(wal.clone());

        store.put(entry(0, b"k1", 1, Some(b"v".to_vec()))).unwrap();
        // Neither timeout nor threshold reached → not yet fsynced.
        assert!(wal.ms_since_last_flush() < 30, "no fsync before threshold");

        std::thread::sleep(std::time::Duration::from_millis(50));
        store.put(entry(0, b"k2", 2, Some(b"v".to_vec()))).unwrap();

        // The second put triggers the timeout fsync.
        assert!(
            wal.ms_since_last_flush() < 30,
            "timeout must trigger lazy fsync"
        );
        let records = crate::wal::replay_all(&dir.join("wal")).unwrap();
        assert_eq!(records.len(), 2);
    }

    /// replay_put does not write the WAL (regression: recovery must not
    /// double-append).
    #[test]
    fn replay_put_bypasses_wal() {
        use crate::wal::{GroupCommitConfig, WalConfig};

        let dir = std::env::temp_dir().join(format!("z1kv_mem_replay_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let wal = Arc::new(
            WalWriter::open(
                &dir,
                WalConfig {
                    wal_dir: dir.join("wal"),
                    enabled: true,
                    group_commit: Some(GroupCommitConfig {
                        policy: crate::wal::SyncPolicy::SyncEach,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        let store = MemStore::new(1024, SyncLevel::Immediate);
        store.set_wal(wal.clone());

        // put writes the WAL; replay_put does not.
        store.put(entry(0, b"k", 1, Some(b"v".to_vec()))).unwrap();
        store.replay_put(entry(0, b"replayed", 2, Some(b"v".to_vec())));

        // The replayed entry is visible (it lives in memory).
        let mut commit = HashMap::new();
        commit.insert(2, 2);
        assert!(store
            .get_visible(0, b"replayed", 2, &commit, &HashSet::new())
            .is_some());

        // But the WAL holds only the one record written by put.
        let records = crate::wal::replay_all(&dir.join("wal")).unwrap();
        assert_eq!(records.len(), 1, "replay_put must not append to WAL");
    }
}
