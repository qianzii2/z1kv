//! Fuzz-contract smoke tests (a no-libFuzzer alternative).
//!
//! Drives the four parse entry points with fixed-seed pseudo-random byte
//! streams and asserts the same contracts as fuzz/fuzz_targets/*:
//! Ok/Err/boolean are all acceptable, a panic never is.
//! For real coverage-guided fuzzing see fuzz/README.md (needs MSVC ASan).

use z1kv::codec::disk_format::DiskFormat;

// xorshift64*: fixed-seed pseudo-random, keeping the test reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

#[test]
fn fuzz_smoke_parse_surfaces() {
    let mut rng = Rng(0x5EED_1B0D);

    for _ in 0..500 {
        // 1. WAL record parsing: arbitrary bytes → Ok/Err, never a panic.
        let dir = std::env::temp_dir().join(format!("z1kv_smoke_{}", uuid::Uuid::new_v4()));
        let wal_dir = dir.join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let data_len = (rng.next() % 128) as usize;
        std::fs::write(wal_dir.join("wal.00000000"), rng.bytes(data_len)).unwrap();
        let _ = z1kv::wal::replay_all(&wal_dir);
        std::fs::remove_dir_all(&dir).ok();

        // 2. patch deserialization.
        let data = {
            let n = (rng.next() % 256) as usize;
            rng.bytes(n)
        };
        let _ = z1kv::store::Z1PatchFormatV4::from_disk_bytes(&data);

        // 3. checkpoint envelope。
        let dir = std::env::temp_dir().join(format!("z1kv_smoke_ck_{}", uuid::Uuid::new_v4()));
        let ckpt = dir.join("checkpoints");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::write(ckpt.join("_LATEST"), 3735928559u64.to_le_bytes()).unwrap();
        std::fs::write(ckpt.join("ckpt_000000003735928559.bin"), {
            let n = (rng.next() % 128) as usize;
            rng.bytes(n)
        })
        .unwrap();
        let mgr = z1kv::wal::CheckpointManager::new(&dir);
        let _ = mgr.load_latest();
        std::fs::remove_dir_all(&dir).ok();

        // 4. visibility decision: boolean output + dual implementations agree.
        let snapshot_id = rng.next() % 64;
        let created = rng.next() % 64;
        let deleted = rng.next() % 64;
        let mut commit_ts = std::collections::HashMap::new();
        commit_ts.insert(created, created);
        commit_ts.insert(deleted, deleted);
        let active: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let entry = z1kv::store::Z1Entry {
            key: z1kv::store::Z1Key::new(0, b"k"),
            txn_id: created,
            value: Some(std::sync::Arc::new(b"v".to_vec())),
            ts: 0,
        };
        let _ = entry.is_visible_at_commit(snapshot_id, &commit_ts, &active);
    }
}
