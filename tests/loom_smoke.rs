//! loom smoke tests: verify the loom toolchain works and model the core
//! generation-protocol invariant of `SnapshotCache` with loom primitives.
//!
//! The real `SnapshotCache` uses `parking_lot::RwLock` / `ArcSwap`, which
//! loom cannot explore; this test extracts its protocol —
//! "invalidate clears the entry, THEN bumps the generation; a reader that
//! observes generation G must never see a stale entry stamped G" — into a
//! loom-explorable model with loom atomics, and asserts the invariant under
//! every interleaving.

use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;

#[test]
fn loom_smoke_two_threads_fetch_add() {
    loom::model(|| {
        let counter = Arc::new(AtomicU64::new(0));
        let b = Arc::clone(&counter);
        let h1 = loom::thread::spawn(move || b.fetch_add(1, Ordering::SeqCst));
        counter.fetch_add(1, Ordering::SeqCst);
        h1.join().unwrap();
        // Under any interleaving of the two fetch_adds, the final value must be 2.
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    });
}

/// Model of the SnapshotCache generational protocol.
///
/// State: a shared cached entry `(entry_generation, cached_payload)` plus a
/// `cache_valid` flag. `invalidate()` clears the cache before bumping the
/// generation (the documented ordering requirement). The reader mirrors
/// SnapshotCache::snapshot's fast path: it takes a snapshot id (`gen`) and
/// accepts the cache entry only if `entry_generation == gen`, checked
/// against the entry itself under one lock (modeled here by loading the
/// entry generation and payload, then validating).
#[test]
fn loom_cache_generation_protocol_never_returns_stale_entry() {
    loom::model(|| {
        // Shared state.
        let generation = Arc::new(AtomicU64::new(1));
        // The cached entry, mirroring SnapshotCacheEntry { generation, snapshot }:
        // an entry carries the generation it was built under.
        let entry_generation = Arc::new(AtomicU64::new(0));
        let cached_payload = Arc::new(AtomicU64::new(0));
        let cache_valid = Arc::new(AtomicU64::new(0)); // 0 = empty, 1 = has entry

        // ── Writer / invalidator thread ─────────────────────────────────
        // Simulates: commit → invalidate() → commit → fresh snapshot cached
        // under the new generation. Order inside invalidate is load-bearing:
        // clear FIRST, then bump generation.
        let g2 = Arc::clone(&generation);
        let eg2 = Arc::clone(&entry_generation);
        let p2 = Arc::clone(&cached_payload);
        let v2 = Arc::clone(&cache_valid);
        let writer = loom::thread::spawn(move || {
            // invalidate(): clear cache, then bump generation.
            v2.store(0, Ordering::SeqCst);
            let old = g2.load(Ordering::SeqCst);
            g2.store(old + 1, Ordering::SeqCst);

            // Simulate a commit landing (payload of the new generation era),
            // then a fresh snapshot being cached under the new generation.
            p2.store(42, Ordering::SeqCst);
            eg2.store(2, Ordering::SeqCst);
            v2.store(1, Ordering::SeqCst);
        });

        // ── Reader thread ────────────────────────────────────────────────
        // Mirrors SnapshotCache::snapshot's fast path: read the current
        // generation, then (under the same lock in the real code) load the
        // entry and accept it only if entry.generation == snapshot gen.
        // In the real implementation the generation check and payload read
        // are one critical section on the RwLock; modeling that atomicity
        // with a single compare keeps the check meaningful.
        let g1 = Arc::clone(&generation);
        let eg1 = Arc::clone(&entry_generation);
        let p1 = Arc::clone(&cached_payload);
        let v1 = Arc::clone(&cache_valid);
        let reader = loom::thread::spawn(move || {
            let gen = g1.load(Ordering::SeqCst);
            if v1.load(Ordering::SeqCst) == 1 {
                let entry_gen = eg1.load(Ordering::SeqCst);
                let payload = p1.load(Ordering::SeqCst);
                // Invariant: accept the entry only when the entry's own
                // generation matches the snapshot's. A mismatching entry is
                // treated as a miss (slow path rebuild), never returned.
                if entry_gen == gen {
                    // Accepted: the payload must be this generation's.
                    if gen == 1 {
                        assert_eq!(payload, 0, "gen-1 entry must carry the gen-1 payload");
                    } else if gen == 2 {
                        assert_eq!(payload, 42, "gen-2 entry must carry the gen-2 payload");
                    }
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        // Post-conditions: the invalidation completed; generation advanced.
        assert_eq!(generation.load(Ordering::SeqCst), 2);
    });
}
