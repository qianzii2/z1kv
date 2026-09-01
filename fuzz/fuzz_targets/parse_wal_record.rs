#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz target 1: WAL record parsing — feed arbitrary byte sequences to
// `read_one_record`'s parse path (via a temp file, matching the on-disk
// format).
// Contract: returns Ok(Some/None) or Err; never panics, never over-allocates
// (the MAX_WAL_RECORD_LEN guard).

fuzz_target!(|data: &[u8]| {
    // Minimum valid frame: 8 header bytes. Shorter than 8 is skipped
    // (the parser would only report truncated anyway).
    if data.len() < 8 {
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "z1kv-fuzz-wal-{}",
        std::process::id()
    ));
    let wal_dir = dir.join("wal");
    let _ = std::fs::create_dir_all(&wal_dir);
    let path = wal_dir.join("wal.00000000");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    // Contract: passing without a panic; Err is a valid outcome (corrupt input).
    let _ = z1kv::wal::replay_all(&wal_dir);
    let _ = std::fs::remove_dir_all(&dir);
});
