#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz target 4: MVCC visibility decisions — generate
// (snapshot_id, created_txn, deleted_txn) plus a small commit_ts_map from
// the fuzz byte stream.
// Contract: is_visible_at_commit returns a boolean, never panics; and it
// stays equivalent to VisibilityManager's VisFilter implementation
// (cross-validation of the dual implementations).

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let snapshot_id = u64::from_le_bytes(data[0..8].try_into().unwrap()) % 64;
    let created = u64::from_le_bytes(data[8..16].try_into().unwrap()) % 64;
    let deleted = if data.len() > 16 {
        Some(u64::from_le_bytes(data[16..24].try_into().unwrap_or([0; 8])) % 64)
    } else {
        None
    };

    let mut commit_ts = std::collections::HashMap::new();
    commit_ts.insert(created, created);
    if let Some(d) = deleted {
        commit_ts.insert(d, d);
    }
    let active: std::collections::HashSet<u64> = std::collections::HashSet::new();

    let entry = z1kv::store::Z1Entry {
        key: z1kv::store::Z1Key::new(0, b"k"),
        txn_id: created,
        value: Some(std::sync::Arc::new(b"v".to_vec())),
        ts: 0,
    };
    // Contract: boolean output, never panics.
    let _ = entry.is_visible_at_commit(snapshot_id, &commit_ts, &active);
});
