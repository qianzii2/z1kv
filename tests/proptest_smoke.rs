//! Property tests: GC conservativeness + dual visibility-implementation equivalence.

use proptest::prelude::*;
use z1kv::mvcc::{IsolationLevel, VisibilityManager};
use z1kv::store::Z1Entry;

proptest! {
    // Property 1: gc_entries, for any entries and watermark, must preserve
    // each key's visible version set as of a snapshot at the watermark
    // (conservativeness).
    #[test]
    fn gc_is_conservative_at_watermark(
        entries in proptest::collection::vec(
            (0u64..8, 0u64..4, proptest::option::of(0u8..255)),
            0..16
        ).prop_map(|v| {
            v.into_iter().map(|(txn, k, val)| z1kv::store::PatchEntry {
                key: vec![k as u8],
                value: val.map(|v| std::sync::Arc::new(vec![v])),
                txn_id: txn,
            }).collect::<Vec<z1kv::store::PatchEntry>>()
        }),
        watermark in 0u64..12,
    ) {
        let (retained, _stats) = z1kv::store::gc::gc_entries(entries.clone(), watermark);

        // Verify against the watermark snapshot one key at a time: every key
        // visible before GC remains visible after GC.
        // Visibility = the key has a version with txn_id <= watermark (the max).
        let visible_before = |ks: &[z1kv::store::PatchEntry]| -> std::collections::BTreeMap<Vec<u8>, Option<u64>> {
            let mut m = std::collections::BTreeMap::new();
            for e in ks {
                if e.txn_id <= watermark {
                    let cur = m.entry(e.key.clone()).or_insert(None);
                    match cur {
                        Some(t) if *t >= e.txn_id => {}
                        _ => { *cur = Some(e.txn_id); }
                    }
                }
            }
            m
        };
        prop_assert_eq!(visible_before(&entries), visible_before(&retained));
    }

    // Property 2: dual visibility-implementation equivalence —
    // VisibilityManager::is_visible and Z1Entry::is_visible_at_commit agree
    // on any input.
    // This is the core D12 strict-snapshot-isolation invariant (a
    // generalization of 9 hand-written cases as a property test).
    // Note: only creation visibility is tested (deleted=None), because
    // is_visible_at_commit does not handle tombstones (an interface design
    // difference).
    #[test]
    fn visibility_dual_impl_equivalence(
        snapshot_id in 0u64..32,
        created in 0u64..32,
        committed in proptest::collection::vec((0u64..32, 0u64..32), 0..8),
    ) {
        // Build commit_ts_map: give mgr.recover_committed_history and
        // entry.is_visible_at_commit exactly the same input.
        let mut commit_ts_map = std::collections::HashMap::new();
        let mut inserted_at = std::collections::HashMap::new();
        for (txn, ts) in &committed {
            commit_ts_map.insert(*txn, *ts);
            inserted_at.insert(*txn, snapshot_id); // snapshot_id as inserted_at (keeps TTL out of the way)
        }
        let _active: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Build the mgr, feeding it the exact same commit_ts_map as the entry.
        let cfg = z1kv::config::VisibilityConfig::default()
            .with_max_history_entries(u64::MAX as usize)
            .with_history_ttl_secs(u64::MAX / 1000);
        let mut mgr = VisibilityManager::new_with_config(cfg);
        mgr.set_committed_txn(snapshot_id);
        mgr.recover_committed_history_with_config(commit_ts_map.clone(), None, &inserted_at);

        // Use the snapshot built by the mgr (same input as the entry), rather
        // than a manually passed snapshot_id.
        let snap = mgr.snapshot(IsolationLevel::Snapshot);
        let entry = Z1Entry {
            key: z1kv::store::Z1Key::new(0, b"k"),
            txn_id: created,
            value: Some(std::sync::Arc::new(b"v".to_vec())),
            ts: 0,
        };

        // Implementation 1: mgr.is_visible (the VisFilter for VisibilityManager).
        let via_mgr = mgr.is_visible(&snap, created, None);

        // Implementation 2: entry.is_visible_at_commit (Z1Entry's VisFilter impl).
        // Uses snap's snapshot_id/commit_ts_map (identical to mgr.is_visible).
        let active_from_snap: std::collections::HashSet<u64> =
            snap.active_txns.iter().copied().collect();
        let via_entry = entry.is_visible_at_commit(
            snap.snapshot_id, &snap.commit_ts_map, &active_from_snap);

        // Invariant: the two implementations must agree.
        prop_assert_eq!(via_mgr, via_entry);
    }
}
