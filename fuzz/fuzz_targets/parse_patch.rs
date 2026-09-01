#![no_main]

use libfuzzer_sys::fuzz_target;
use z1kv::codec::disk_format::DiskFormat;

// Fuzz target 2: `.zpatch` file parsing — feed arbitrary bytes to
// `Z1PatchFormatV4::from_disk_bytes` (DiskFormat 18B header + postcard).
// Contract: Ok or Err, never panics.

fuzz_target!(|data: &[u8]| {
    let _ = z1kv::store::Z1PatchFormatV4::from_disk_bytes(data);
});
