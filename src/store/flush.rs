//! FlushEngine — L1 → L2 flush with race-window protection.
//!
//! - D8: recent_flush_cache bridges the L1→L2 flush window (readers can see
//!   already-flushed data)
//! - D5: flush_epoch increments so concurrent readers detect a flush and retry
//! - D4: the WAL is durable before L2 (memstore.put guarantees WAL-first;
//!   flush is only memory→disk)
//!
//! Simplification: SILK foreground load coordination and EcoTune policy
//! selection were dropped (they can be brought back if needed); the core
//! correctness protocol of flush is kept.

use crate::error::Result;
use crate::store::disk::DiskLayer;
use crate::store::mem::MemStore;
use crate::store::recent_flush_cache::RecentFlushCache;
use crate::store::types::PatchEntry;
use crate::TxnId;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// L1 → L2 flush engine.
pub struct FlushEngine {
    l1: Arc<MemStore>,
    l2: Arc<DiskLayer>,
    /// L3 frozen layer (compaction output).
    l3: Arc<DiskLayer>,
    /// D8: recently flushed entries (race window protection).
    recent_flush: Arc<RecentFlushCache>,
    /// D5: flush epoch counter.
    flush_epoch: Arc<AtomicU64>,
    /// Monotonic patch id generator.
    next_patch_id: AtomicU64,
}

impl FlushEngine {
    /// Construct with a dedicated L3 layer (used by the engine).
    ///
    /// There is no `new`: the old `new` created a placeholder DiskLayer for
    /// L3 (`"l3_unused"`), which created a real directory in the current
    /// working directory at construction time (a side effect). Production
    /// code and tests always use `with_l3` with an explicit L3 root.
    ///
    /// `next_patch_id`: fix — a pure in-memory counter resets to zero on
    /// restart, making the first flush after a reopen rename over existing
    /// same-id patch files (permanent data loss once the WAL has been
    /// truncated). The caller (`Z1Kv::open`) must pass
    /// `max(l2.max_patch_id(), l3.max_patch_id()) + 1` to recover the
    /// watermark; tests may pass 0.
    pub fn with_l3(
        l1: Arc<MemStore>,
        l2: Arc<DiskLayer>,
        l3: Arc<DiskLayer>,
        recent_flush: Arc<RecentFlushCache>,
        flush_epoch: Arc<AtomicU64>,
        next_patch_id: u64,
    ) -> Self {
        Self {
            l1,
            l2,
            l3,
            recent_flush,
            flush_epoch,
            next_patch_id: AtomicU64::new(next_patch_id),
        }
    }

    /// Flush L1 to L2: drain the cold buffer, group by cf, write patches.
    ///
    /// # Invariants
    ///
    /// - D8: drained entries go into the recent_flush cache BEFORE the flush starts
    /// - D5: after the flush, clear the cache + increment flush_epoch
    /// - one group per (cf) = one patch file
    pub fn flush_l1_to_l2(&self) -> Result<usize> {
        // Visibility-window fix: staged drain so that recent_flush covers
        // every entry about to be moved **before** it leaves L1.
        //
        // Old order: drain_after_swap (L1 emptied) → cache_entries — between
        // the two steps there was a vacuum window: L1 empty, recent_flush
        // not yet populated, patch not yet written. A concurrent get landing
        // in the window returned None for committed data (violating
        // read-immediately-after-commit); the
        // `concurrent_put_during_flush_never_loses_data` stress test
        // reproduced it reliably.
        //
        // New order: swap (data moves hot→cold, still readable via get's L1
        // path — get_versions merges hot+cold+snapshot) → cache(swapped) →
        // drain_cold → cache_merge(cold) → append → clear.
        // Under any interleaving, at least one of recent_flush / the three
        // L1 sources holds the data.
        // With the fix, swap_and_drain returns ALL data moved out of the
        // read path (old cold + old_hot) at once, and cache_merge covers it
        // immediately — no vacuum window remains between drain and cache.
        //
        // Semantics change (the correct form of closing the window):
        // recent_flush carries **all data not yet durably written** —
        // cache_merge accumulates and clear runs only after every
        // append_patch succeeded. The old replace-on-flush semantics of
        // cache_entries would evict "data whose previous append failed" when
        // "this round's swapped" arrived (lost before ever being written);
        // merge + a final clear guarantees: either everything is written and
        // the cache cleared, or the data stays readable and retryable.
        //
        // Migration gate (write side): the WHOLE migration — swap L1,
        // cache the swapped data, publish patches to the L2 index, clear
        // the cache — runs under this write lock. Gated readers (see
        // `Z1Kv::get_at` / `scan_at`) either observe the data in L1/the
        // cache (flush not yet started) or in the L2 index (flush done);
        // the intermediate "moved out of L1, not yet cached / not yet
        // indexed" states are never visible to a gated reader.
        let _gate = self.recent_flush.write_gate();
        let swapped = self.l1.swap_and_drain();
        self.recent_flush.cache_merge(&swapped);

        // Patch data source = the full cache (historically unwritten data +
        // this round's swapped).
        let all = self.recent_flush.get_all();

        // Group by cf.
        let mut groups: BTreeMap<u16, Vec<PatchEntry>> = BTreeMap::new();
        for e in &all {
            let pe = PatchEntry {
                key: e.key.key.clone(),
                value: e.value.clone(),
                txn_id: e.txn_id,
            };
            groups.entry(e.key.cf).or_default().push(pe);
        }

        let num_groups = groups.len();

        for (cf, entries) in groups {
            let patch_id = self.next_patch_id.fetch_add(1, Ordering::Relaxed);
            self.l2.append_patch(cf, entries, patch_id)?;
        }

        // D8 + D5: everything written successfully → clear the cache; any
        // failure propagated via `?` above leaves the cache holding the
        // not-yet-durable data (readable, retried next time). The whole
        // migration runs under the single write gate acquired above, so
        // gated readers either see the data in the cache or in the L2
        // index, never in neither.
        self.recent_flush.clear();
        drop(_gate);
        self.recent_flush.increment_epoch();
        self.flush_epoch.fetch_add(1, Ordering::SeqCst);

        Ok(num_groups)
    }

    /// Flush if the L1 threshold is exceeded.
    pub fn try_flush(&self) -> Result<Option<usize>> {
        if self.l1.should_flush() {
            Ok(Some(self.flush_l1_to_l2()?))
        } else {
            Ok(None)
        }
    }

    /// L2 → L3 compaction: merge L2 patches into a GC'd L3 frozen layer.
    ///
    /// Steps:
    /// 1. For each cf: collect all versions of all keys in L2 (patches are
    ///    append-only; the same key may appear in several patches)
    /// 2. `gc_entries` merge: per key keep "the newest version below the
    ///    watermark + all versions above it"
    /// 3. Write the merged result as one L3 patch
    /// 4. Drop the cf's L2 data (L3 has fully taken over)
    ///
    /// `min_active_begin_ts`: the smallest begin_ts among active
    /// transactions. Passing `u64::MAX` (no active transactions) keeps only
    /// the highest version per key — maximal compaction under MVCC.
    /// Returns (number of cfs processed, number of entries GC'd).
    pub fn compact_l2_to_l3(&self, min_active_begin_ts: TxnId) -> Result<(usize, usize)> {
        let mut cfs_compacted = 0usize;
        let mut total_reclaimed = 0usize;

        // Migration gate (write side): compaction replaces the L2 index
        // contents (drop_cf) with an L3 patch. A gated reader may hold a
        // pre-compaction L2 index snapshot whose files this compaction
        // deletes — so the L2 takeover must be atomic with respect to gated
        // readers, exactly like a flush.
        let _gate = self.recent_flush.write_gate();

        // Enumerate the cfs present in L2 (by scanning directories via patch_ids).
        let cf_ids = self.l2.list_cfs();

        for cf in cf_ids {
            // 1. Collect all versions of this cf in L2 (all keys across all patches).
            let all_versions = self.l2.all_versions(cf)?;
            if all_versions.is_empty() {
                continue;
            }

            // Convert to PatchEntry for the GC merge.
            let entries: Vec<PatchEntry> = all_versions
                .into_iter()
                .map(|e| PatchEntry {
                    key: e.key.key.clone(),
                    value: e.value,
                    txn_id: e.txn_id,
                })
                .collect();

            // 2. GC merge.
            let (retained, stats) = crate::store::gc::gc_entries(entries, min_active_begin_ts);
            total_reclaimed += stats.reclaimed;

            // 3. Write L3 (one patch per cf).
            if !retained.is_empty() {
                let patch_id = self.next_patch_id.fetch_add(1, Ordering::Relaxed);
                self.l3.append_patch(cf, retained, patch_id)?;
            }

            // 4. The cf's L2 data has been taken over by L3; delete it.
            self.l2.drop_cf(cf)?;
            cfs_compacted += 1;
        }

        if cfs_compacted > 0 {
            self.flush_epoch.fetch_add(1, Ordering::SeqCst);
        }

        Ok((cfs_compacted, total_reclaimed))
    }

    /// Trigger compaction if the L2 patch count reaches the threshold.
    pub fn try_compact(
        &self,
        min_active_begin_ts: TxnId,
        l2_patch_threshold: usize,
    ) -> Result<Option<(usize, usize)>> {
        let patch_count: usize = self
            .l2
            .list_cfs()
            .iter()
            .map(|cf| self.l2.patch_ids(*cf).len())
            .sum();
        if patch_count >= l2_patch_threshold {
            Ok(Some(self.compact_l2_to_l3(min_active_begin_ts)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::config::SyncLevel;
    use crate::store::types::Z1Entry;
    use crate::store::types::Z1Key;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("z1kv_flush_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(cf: u16, key: &[u8], txn_id: u64, v: u8) -> Z1Entry {
        Z1Entry {
            key: Z1Key::new(cf, key),
            txn_id,
            value: Some(Arc::new(vec![v])),
            ts: 0,
        }
    }

    #[test]
    fn flush_moves_entries_to_l2() {
        let dir = tmp_dir("basic");
        let l1 = Arc::new(MemStore::new(1024, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(
            l1.clone(),
            l2.clone(),
            Arc::new(DiskLayer::new(dir.join("l3"))),
            recent,
            epoch,
            0,
        );

        l1.put(entry(0, b"k", 1, 1)).unwrap();
        l1.put(entry(0, b"k2", 1, 2)).unwrap();

        let groups = engine.flush_l1_to_l2().unwrap();
        assert_eq!(groups, 1); // one cf

        // L2 must now contain the entries.
        let versions = l2.get_versions(0, b"k").unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(
            versions[0].value.as_deref().map(|v| v.as_slice()),
            Some(&[1u8][..])
        );
    }

    #[test]
    fn flush_bumps_epoch_and_clears_cache() {
        let dir = tmp_dir("epoch");
        let l1 = Arc::new(MemStore::new(1024, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(
            l1.clone(),
            l2.clone(),
            Arc::new(DiskLayer::new(dir.join("l3"))),
            recent.clone(),
            epoch.clone(),
            0,
        );

        l1.put(entry(0, b"k", 1, 1)).unwrap();
        engine.flush_l1_to_l2().unwrap();

        assert_eq!(epoch.load(Ordering::SeqCst), 1);
        assert_eq!(recent.flush_epoch(), 1);
        assert!(
            recent.get_for_key(0, b"k").is_empty(),
            "cache cleared after flush"
        );
    }

    #[test]
    fn try_flush_respects_threshold() {
        let dir = tmp_dir("threshold");
        // threshold = 1 byte so any put triggers flush.
        let l1 = Arc::new(MemStore::new(1, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(
            l1.clone(),
            l2.clone(),
            Arc::new(DiskLayer::new(dir.join("l3"))),
            recent,
            epoch,
            0,
        );

        l1.put(entry(0, b"k", 1, 1)).unwrap();
        assert!(l1.should_flush());
        assert_eq!(engine.try_flush().unwrap(), Some(1));
        assert!(!l1.should_flush());
    }

    /// compaction: L2's multi-versions merge into L3, old versions are GC'd,
    /// L2 data is taken over.
    #[test]
    fn compact_l2_to_l3_merges_and_gcs() {
        let dir = tmp_dir("compact");
        let l1 = Arc::new(MemStore::new(1024, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let l3 = Arc::new(DiskLayer::new(dir.join("l3")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(
            l1.clone(),
            l2.clone(),
            l3.clone(),
            recent.clone(),
            epoch.clone(),
            0,
        );

        // Key "k" gets 5 versions flushed in 3 rounds (3 L2 patches).
        for (txn, v) in [(1u64, 1u8), (2, 2), (3, 3), (4, 4), (5, 5)] {
            l1.put(entry(0, b"k", txn, v)).unwrap();
            if txn % 2 == 0 {
                engine.flush_l1_to_l2().unwrap();
            }
        }
        engine.flush_l1_to_l2().unwrap();
        assert!(
            l2.patch_ids(0).len() >= 2,
            "L2 should hold multiple patches"
        );

        // Compaction: watermark = 5 (no active transactions) → only the highest version per key.
        let (cfs, reclaimed) = engine.compact_l2_to_l3(5).unwrap();
        assert_eq!(cfs, 1);
        assert!(
            reclaimed >= 3,
            "old versions must be GC'd, got {}",
            reclaimed
        );

        // L2 has been taken over: no data.
        assert!(l2.patch_ids(0).is_empty());
        // L3: at watermark=5, keep the newest below the watermark (txn 4) + txn 5.
        let versions = l3.get_versions(0, b"k").unwrap();
        let txns: Vec<u64> = versions.iter().map(|v| v.txn_id).collect();
        assert_eq!(txns, vec![4, 5]);

        // Verify maximal compaction too: watermark = u64::MAX (no active
        // snapshots) → only the highest version is kept.
        // (The cf was dropped; flush a new round of data into L2.)
        for (txn, v) in [(6u64, 6u8), (7, 7)] {
            l1.put(entry(0, b"k", txn, v)).unwrap();
        }
        engine.flush_l1_to_l2().unwrap();
        engine.compact_l2_to_l3(u64::MAX).unwrap();
        let versions = l3.get_versions(0, b"k").unwrap();
        // L3 now holds [4,5] (the old compaction output) + [7] (the highest
        // after this round's GC); get_versions merges and returns them, but
        // the highest txn_id = 7.
        assert_eq!(versions.last().unwrap().txn_id, 7);
    }

    /// compaction keeps the newest version below the watermark (active
    /// snapshot protection).
    #[test]
    fn compact_preserves_history_before_watermark() {
        let dir = tmp_dir("compact_hist");
        let l1 = Arc::new(MemStore::new(1024, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let l3 = Arc::new(DiskLayer::new(dir.join("l3")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(l1.clone(), l2.clone(), l3.clone(), recent, epoch, 0);

        // Key "k" versions 1..4, all flushed into L2.
        for txn in 1..=4u64 {
            l1.put(entry(0, b"k", txn, txn as u8)).unwrap();
        }
        engine.flush_l1_to_l2().unwrap();

        // Watermark = 4: keep txn 3 (newest below the watermark) + txn 4;
        // reclaim txns 1, 2.
        let (_, reclaimed) = engine.compact_l2_to_l3(4).unwrap();
        assert_eq!(reclaimed, 2);

        let versions = l3.get_versions(0, b"k").unwrap();
        let txns: Vec<u64> = versions.iter().map(|e| e.txn_id).collect();
        assert_eq!(
            txns,
            vec![3, 4],
            "history baseline (txn 3) must be preserved"
        );
    }

    /// try_compact: triggers only when the patch count reaches the threshold.
    #[test]
    fn try_compact_respects_threshold() {
        let dir = tmp_dir("try_compact");
        let l1 = Arc::new(MemStore::new(1024, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let l3 = Arc::new(DiskLayer::new(dir.join("l3")));
        let recent = Arc::new(RecentFlushCache::new());
        let epoch = Arc::new(AtomicU64::new(0));
        let engine = FlushEngine::with_l3(l1.clone(), l2.clone(), l3.clone(), recent, epoch, 0);

        // Only 1 patch flushed, threshold 2 → no trigger.
        l1.put(entry(0, b"k", 1, 1)).unwrap();
        engine.flush_l1_to_l2().unwrap();
        assert!(engine.try_compact(u64::MAX, 2).unwrap().is_none());

        // Flush one more (2 patches total), reaching the threshold → triggers.
        l1.put(entry(0, b"k", 2, 2)).unwrap();
        engine.flush_l1_to_l2().unwrap();
        let result = engine.try_compact(u64::MAX, 2).unwrap();
        assert!(result.is_some(), "2 patches should meet threshold 2");
        assert!(l2.patch_ids(0).is_empty(), "L2 taken over after compaction");
    }
}
