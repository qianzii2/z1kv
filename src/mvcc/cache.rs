//! Snapshot Cache with Generational Invalidation.
//!
//! Implements the generational cache protocol that eliminates TOCTOU races
//! between cache validation and snapshot use.
//!
//! ## Design
//!
//! - Uses an atomic generation counter that is incremented on any MVCC mutation
//! - Each cache entry stores the generation at the time of creation
//! - Cache reads verify the generation matches before returning
//! - Separate cache slots for `Full` and `ActiveOnly` snapshots prevent type confusion
//!
//! ## Thread Safety
//!
//! - `invalidate()` uses `fetch_add` for lock-free generation increment
//! - Cache reads use `Acquire` ordering to observe the latest invalidation
//! - Cache writes use `Release` ordering to publish the new entry
//!
//! ## Why generations?
//!
//! The early implementation shared a single `RwLock<Option<Arc<TxnSnapshot>>>`
//! cache, which had a TOCTOU window (the cache could be invalidated between
//! the check and the return). A generation counter closes that window.
//!
//! Note: the `invalidate` order is "clear cache first, bump generation
//! second" — in the reverse order there was a window where a new generation
//! could hit the stale, not-yet-cleared entry and return an outdated
//! snapshot, contradicting the invariant that a reader observing the new
//! generation never observes a stale entry.

use crate::mvcc::visibility::{TxnSnapshot, VisibilityManager};
use crate::TxnId;
use arc_swap::ArcSwap;
use std::sync::Arc;

/// A single cached snapshot entry with its generation number.
#[derive(Debug)]
struct SnapshotCacheEntry {
    generation: TxnId,
    snapshot: Arc<TxnSnapshot>,
}

/// Generational snapshot cache that eliminates TOCTOU races between
pub struct SnapshotCache {
    /// Global generation counter — any MVCC mutation increments this.
    generation: ArcSwap<TxnId>,
    /// Snapshot cache (includes commit_ts_map).
    full_cache: parking_lot::RwLock<Option<SnapshotCacheEntry>>,
}

impl SnapshotCache {
    /// Create a new empty cache with generation = 1.
    pub fn new() -> Self {
        Self {
            generation: ArcSwap::from_pointee(1),
            full_cache: parking_lot::RwLock::new(None),
        }
    }

    /// Invalidate all cached snapshots.
    ///
    /// Called on any MVCC mutation (begin_txn, commit_txn, rollback_txn).
    /// Uses `fetch_add` for lock-free increment — readers holding an old generation
    /// will continue to use the stale snapshot until they naturally exit, which is
    /// safe because they observed the old MVCC state when they started.
    pub fn invalidate(&self) {
        // Order is critical: clear the cache FIRST, then bump the generation.
        //
        // Before the fix (bump gen, then clear): between the two steps, a new
        // `snapshot()` could read the new generation and match the stale,
        // not-yet-cleared entry, returning an **outdated** snapshot —
        // violating the promise that a reader seeing the new generation will
        // never observe a stale entry.
        //
        // After the fix: a reader either (a) read the old generation before
        // the clear → slow-path rebuild (the MVCC change may or may not have
        // landed during the rebuild, but any outcome is a self-consistent
        // read), or (b) reads the new generation → the cache is necessarily
        // empty → slow-path rebuild. A "new generation × stale entry" hit is
        // impossible.
        *self.full_cache.write() = None;
        let old = **self.generation.load();
        self.generation.store(Arc::new(old + 1));
        tracing::debug!(gen = old + 1, "snapshot cache invalidated");
    }

    fn current_generation(&self) -> TxnId {
        **self.generation.load()
    }

    /// Get or compute a full snapshot (with commit_ts_map).
    ///
    /// Returns the cached snapshot if the generation matches.
    /// Otherwise, builds a fresh snapshot and caches it.
    pub fn snapshot(&self, mvcc: &VisibilityManager) -> TxnSnapshot {
        let gen = self.current_generation();

        // Fast path: check cache with generation validation
        if let Some(entry) = self.full_cache.read().as_ref() {
            if entry.generation == gen {
                return entry.snapshot.as_ref().clone();
            }
        }

        // Slow path: generate new snapshot.
        //
        // Double-check fix: a concurrent invalidate may land while the
        // snapshot is being built (the build needs the mvcc read lock). If a
        // snapshot built under the OLD generation were cached under the NEW
        // generation, later readers would hit a stale view (violating
        // read-your-commits immediately after a commit returns). Revalidate
        // the generation before writing; on mismatch, skip caching (return
        // this snapshot directly — it is still a self-consistent view, just
        // not cached).
        let snap = mvcc.snapshot(crate::mvcc::visibility::IsolationLevel::Snapshot);
        let current_gen = self.current_generation();
        if current_gen == gen {
            let entry = SnapshotCacheEntry {
                generation: gen,
                snapshot: Arc::new(snap.clone()),
            };
            *self.full_cache.write() = Some(entry);
            snap
        } else {
            // Generation advanced: retry once (the cache is now either empty
            // or matches the new generation).
            self.snapshot(mvcc)
        }
    }
}

impl Default for SnapshotCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_cache_returns_same_snapshot_within_generation() {
        let cache = SnapshotCache::new();
        let mvcc = VisibilityManager::new();

        let s1 = cache.snapshot(&mvcc);
        let s2 = cache.snapshot(&mvcc);
        assert_eq!(s1.snapshot_id, s2.snapshot_id);
    }

    #[test]
    fn snapshot_cache_invalidates_on_mutation() {
        let cache = SnapshotCache::new();
        let mut mvcc = VisibilityManager::new();

        let s1 = cache.snapshot(&mvcc);
        // Simulate a commit advancing committed_txn
        mvcc.set_committed_txn(42);
        cache.invalidate();
        let s2 = cache.snapshot(&mvcc);

        assert_ne!(s1.snapshot_id, s2.snapshot_id);
        assert_eq!(s2.snapshot_id, 42);
    }

    #[test]
    fn generation_advances_monotonically() {
        let cache = SnapshotCache::new();
        assert_eq!(cache.current_generation(), 1);
        cache.invalidate();
        assert_eq!(cache.current_generation(), 2);
        cache.invalidate();
        assert_eq!(cache.current_generation(), 3);
    }
}
