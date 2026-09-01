#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz target 3: checkpoint envelope parsing — build arbitrary-byte
// checkpoint files and run them through `load_latest` (crc/length guards +
// postcard).
// Contract: returns Some/None, never panics (the out-of-bounds panic was
// fixed; this target guards against regression).

fuzz_target!(|data: &[u8]| {
    let dir = std::env::temp_dir().join(format!(
        "z1kv-fuzz-ckpt-{}",
        std::process::id()
    ));
    let ckpt_dir = dir.join("checkpoints");
    if std::fs::create_dir_all(&ckpt_dir).is_err() {
        return;
    }
    let id_bytes = 0xdead_beefu64.to_le_bytes();
    if std::fs::write(dir.join("ENGINE.lock"), b"fuzz").is_err() {
        return;
    }
    if std::fs::write(ckpt_dir.join("_LATEST"), id_bytes).is_err() {
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    if std::fs::write(ckpt_dir.join("ckpt_000000003735928559.bin"), data).is_err() {
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let mgr = z1kv::wal::CheckpointManager::new(&dir);
    let _ = mgr.load_latest(); // Some/None are both valid; a panic is a bug
    let _ = std::fs::remove_dir_all(&dir);
});
