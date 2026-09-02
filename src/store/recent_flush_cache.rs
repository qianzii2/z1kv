//! RecentFlushCache — explicit shared cache for race window prevention.
//!
//! ## Design
//!
//! The cache bridges the L1→L2 flush window (D8): entries drained from L1
//! stay readable via this cache until their patch is written and indexed.
//!
//! Semantics note (fix): the cache holds **all data not yet durably written**
//! — `cache_merge` accumulates after the swap and `clear` runs only after
//! every patch has been written successfully; if an append fails, the data
//! stays readable in the cache and the flush can be retried (the old
//! replace-on-flush semantics would lose data from the failed round).

use crate::store::types::{Z1Entry, Z1Key};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache for recently flushed entries to prevent race window loss.
#[derive(Debug)]
pub struct RecentFlushCache {
    /// Key: (Z1Key, txn_id), Value: Z1Entry.
    cache: RwLock<BTreeMap<(Z1Key, u64), Z1Entry>>,
    /// Migration gate: readers hold this read lock while collecting
    /// candidates from all layers; the flusher holds the write lock while
    /// publishing patches to the L2 index AND clearing the cache. This
    /// closes the migration window where a reader holding a stale L2 index
    /// snapshot could observe neither the cache (already cleared) nor the
    /// new patch (not yet in the reader's index copy).
    migration_gate: RwLock<()>,
    /// Incremented after each L1→L2 flush.
    flush_epoch: AtomicU64,
}

impl Default for RecentFlushCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RecentFlushCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(BTreeMap::new()),
            migration_gate: RwLock::new(()),
            flush_epoch: AtomicU64::new(0),
        }
    }

    /// Read-side migration gate. Hold the returned guard while collecting
    /// candidates from L1 / recent_flush / L2 so a concurrent flush cannot
    /// move data out from under the read (see `migration_gate` docs).
    pub fn read_gate(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.migration_gate.read()
    }

    /// Write-side migration gate. The flusher holds it across "publish
    /// patch to index + clear cache" so those two steps are atomic with
    /// respect to gated readers.
    pub fn write_gate(&self) -> parking_lot::RwLockWriteGuard<'_, ()> {
        self.migration_gate.write()
    }

    /// Cache entries from a flush operation (replaces contents).
    pub fn cache_entries(&self, entries: &[Z1Entry]) {
        let mut cache = self.cache.write();
        cache.clear();
        for e in entries {
            cache.insert((e.key.clone(), e.txn_id), e.clone());
        }
    }

    /// Full snapshot of the cache (the not-yet-durable data set, used as
    /// the patch data source).
    pub fn get_all(&self) -> Vec<Z1Entry> {
        self.cache.read().values().cloned().collect()
    }

    /// Merge entries into the cache (no clear) — used by the staged flush:
    /// cache the swapped buffer first, then the cold buffer, so that
    /// recent_flush covers L1 data **before** it is removed, closing the
    /// visibility vacuum between drain and cache.
    pub fn cache_merge(&self, entries: &[Z1Entry]) {
        let mut cache = self.cache.write();
        for e in entries {
            cache.insert((e.key.clone(), e.txn_id), e.clone());
        }
    }

    /// Get all cached entries for a key.
    pub fn get_for_key(&self, cf: u16, key: &[u8]) -> Vec<Z1Entry> {
        let zk = Z1Key::new(cf, key);
        self.cache
            .read()
            .values()
            .filter(|e| e.key == zk)
            .cloned()
            .collect()
    }

    /// Get all cached entries matching a filter.
    pub fn get_filtered<F>(&self, filter: F) -> Vec<Z1Entry>
    where
        F: Fn(&Z1Entry) -> bool,
    {
        self.cache
            .read()
            .values()
            .filter(|e| filter(e))
            .cloned()
            .collect()
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Increment the flush epoch counter.
    pub fn increment_epoch(&self) {
        self.flush_epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Get current flush epoch.
    pub fn flush_epoch(&self) -> u64 {
        self.flush_epoch.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn entry(cf: u16, key: &[u8], txn_id: u64, v: u8) -> Z1Entry {
        Z1Entry {
            key: Z1Key::new(cf, key),
            txn_id,
            value: Some(Arc::new(vec![v])),
            ts: 0,
        }
    }

    /// Regression for the not-yet-durable-data-set semantics of
    /// `cache_merge`: merge accumulates (does not replace), `get_all`
    /// returns a snapshot, and `clear` after success empties everything.
    #[test]
    fn merge_accumulates_and_clear_commits() {
        let cache = RecentFlushCache::new();
        // Round 1: swap out A, merge into the cache.
        cache.cache_merge(&[entry(0, b"a", 1, 1)]);
        // Before round 2's swap: append failed → round-1 data must still be
        // in the cache.
        cache.cache_merge(&[entry(0, b"b", 2, 2)]);
        assert_eq!(
            cache.get_for_key(0, b"a").len(),
            1,
            "failed-round data must persist"
        );
        assert_eq!(cache.get_for_key(0, b"b").len(), 1);
        // All patches written successfully → clear.
        cache.clear();
        assert!(
            cache.get_all().is_empty(),
            "clear after success must empty cache"
        );
    }

    /// Regression: the `get_all` snapshot agrees with `get_for_key`.
    #[test]
    fn get_all_matches_get_for_key() {
        let cache = RecentFlushCache::new();
        cache.cache_merge(&[entry(0, b"x", 1, 1), entry(1, b"y", 2, 2)]);
        let all = cache.get_all();
        assert_eq!(all.len(), 2);
        assert_eq!(cache.get_for_key(0, b"x").len(), 1);
        assert_eq!(cache.get_for_key(1, b"y").len(), 1);
    }

    #[test]
    fn cache_entries_and_get_for_key() {
        let cache = RecentFlushCache::new();
        cache.cache_entries(&[entry(0, b"k", 1, 1)]);
        let got = cache.get_for_key(0, b"k");
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn cache_clear() {
        let cache = RecentFlushCache::new();
        cache.cache_entries(&[entry(0, b"k", 1, 1)]);
        cache.clear();
        assert!(cache.get_for_key(0, b"k").is_empty());
    }

    #[test]
    fn flush_epoch_monotonic() {
        let cache = RecentFlushCache::new();
        assert_eq!(cache.flush_epoch(), 0);
        cache.increment_epoch();
        assert_eq!(cache.flush_epoch(), 1);
    }
}
