//! Engine-level process lock — prevents a data dir from being opened by
//! multiple engine instances (same process or across processes).
//!
//! # Motivation
//!
//! Z1KV originally had no process-level mutual exclusion: opening the same
//! data directory twice with `Z1Kv::open` (two instances in one process, or
//! one in each of two processes) would write the WAL and L2 patches
//! concurrently — segment rotation, patch-id allocation and checkpoint
//! truncation would all be corrupted, silently. The standard remedy for
//! embedded databases is a process lock inside the data directory (same as
//! the SQLite `-journal` file or the RocksDB `LOCK` file).
//!
//! # Semantics
//!
//! - `open` creates/opens `data_dir/ENGINE.lock` in **exclusive** mode:
//!   - Windows: `CreateFileW` with `dwShareMode = 0` (the second
//!     `CreateFileW` fails)
//!   - Unix: `flock(LOCK_EX | LOCK_NB)`
//! - On success the `EngineLock` holds the handle; `Drop` closes it (the
//!   lock is released with the process/instance).
//! - On failure a `Z1Error::Internal` is returned, explicitly stating that
//!   the data directory is already in use.
//!
//! # Easy-to-confuse points
//!
//! - The lock file is an empty file whose content is meaningless. A stale
//!   `ENGINE.lock` (left over after a crash) does not block reopening — the
//!   lock lives and dies with the process handle, not with the file.
//! - Sharing a single `Z1Kv` instance across threads within one process is
//!   supported (internal RwLock/Mutex); this lock only intercepts "opening
//!   another engine instance".

use crate::error::{Result, Z1Error};
use std::path::{Path, PathBuf};

/// An acquired engine lock. Exclusively owns the data directory while held;
/// released on `Drop`.
///
/// # Send / Sync
///
/// A Windows `HANDLE` is a raw pointer and the `windows` crate does not
/// implement `Send`/`Sync` for it. Here, however, the handle is an
/// **owning** reference to a kernel object: it is closed exactly once, in
/// `Drop` (`CloseHandle` may be called from any thread), and the field is
/// only read otherwise. `unsafe impl Send/Sync` is therefore sound; sharing
/// one engine across threads via `Arc<Z1Kv>` is an explicit design goal.
pub struct EngineLock {
    /// Windows handle / Unix file object — closed on `Drop`.
    #[cfg(windows)]
    handle: windows::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    _file: std::fs::File,
    /// Lock-file path (for diagnostics).
    #[allow(dead_code)]
    path: PathBuf,
}

// SAFETY: see the type docs. The HANDLE is closed exactly once, in Drop;
// the file object itself is Send + Sync.
#[cfg(windows)]
unsafe impl Send for EngineLock {}
#[cfg(windows)]
unsafe impl Sync for EngineLock {}

impl EngineLock {
    /// Try to acquire the exclusive engine lock for `data_dir`.
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        let path = data_dir.join("ENGINE.lock");
        #[cfg(windows)]
        {
            use windows::core::PCWSTR;
            use windows::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows::Win32::Storage::FileSystem::{
                CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_MODE, OPEN_ALWAYS,
            };

            let wide_path: Vec<u16> = path
                .to_string_lossy()
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            // share_mode = 0: while this process holds the handle, any other
            // CreateFileW (including a second open in the same process)
            // fails — exactly the mutual exclusion we want.
            // GENERIC_READ | GENERIC_WRITE = 0xC0000000 (raw value, avoiding
            // the GENERIC_ACCESS_RIGHTS → FILE_ACCESS_RIGHTS conversion chain).
            let handle = unsafe {
                CreateFileW(
                    PCWSTR::from_raw(wide_path.as_ptr()),
                    0xC0000000u32, // GENERIC_READ | GENERIC_WRITE
                    FILE_SHARE_MODE(0),
                    None,
                    OPEN_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL,
                    None,
                )
            }
            .map_err(|e| {
                Z1Error::Internal(format!(
                    "cannot lock data dir {:?} (already opened by another engine instance?): {}",
                    data_dir, e
                ))
            })?;
            if handle == INVALID_HANDLE_VALUE {
                return Err(Z1Error::Internal(format!(
                    "cannot lock data dir {:?}: INVALID_HANDLE_VALUE",
                    data_dir
                )));
            }
            Ok(Self { handle, path })
        }
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::io::AsRawFd;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(false)
                .open(&path)
                .map_err(|e| {
                    Z1Error::Internal(format!("cannot open lock file {:?}: {}", path, e))
                })?;
            let rc = unsafe { flock(file.as_raw_fd(), libc_flock_flags()) };
            if rc != 0 {
                return Err(Z1Error::Internal(format!(
                    "cannot lock data dir {:?} (already opened by another engine instance?): {}",
                    data_dir,
                    std::io::Error::last_os_error()
                )));
            }
            file.write_all(b"")?;
            Ok(Self { _file: file, path })
        }
    }
}

#[cfg(unix)]
fn libc_flock_flags() -> i32 {
    // LOCK_EX = 2, LOCK_NB = 4 (same values on Linux and Darwin)
    2 | 4
}

#[cfg(unix)]
extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

impl Drop for EngineLock {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::Foundation::CloseHandle;
            let _ = CloseHandle(self.handle);
        }
        // Unix: the file object's own Drop closes the fd → flock is released.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("z1kv_elock_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn second_acquire_fails() {
        let dir = tmp_dir("second_fails");
        let _lock1 = EngineLock::acquire(&dir).unwrap();
        let err = match EngineLock::acquire(&dir) {
            Err(e) => e,
            Ok(_) => panic!("second acquire must fail"),
        };
        assert!(
            err.to_string().contains("cannot lock data dir"),
            "unexpected error: {}",
            err
        );
        // After release, acquiring again succeeds.
        drop(_lock1);
        let _lock2 = EngineLock::acquire(&dir).unwrap();
    }

    /// Regression: a stale lock file left behind by a crashed process must
    /// not block reacquisition. Mutual exclusion comes from the open handle
    /// (Windows share_mode=0 / Unix flock), not from the file's existence —
    /// the OS reclaims a dead process's handles, so the leftover file is an
    /// empty shell. Verified by pre-creating ENGINE.lock and asserting
    /// acquire still succeeds.
    #[test]
    fn stale_lock_file_does_not_block_acquire() {
        let dir = tmp_dir("stale_lock");
        std::fs::create_dir_all(&dir).unwrap();
        // Simulate a crash leftover: the lock file exists, but no process
        // holds a handle to it.
        std::fs::write(dir.join("ENGINE.lock"), b"leftover from crashed process").unwrap();
        let _lock = EngineLock::acquire(&dir).unwrap();
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = tmp_dir("drop_release");
        {
            let _lock = EngineLock::acquire(&dir).unwrap();
        } // Drop → released
        let _lock = EngineLock::acquire(&dir).unwrap();
    }
}
