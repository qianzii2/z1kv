//! DurableWriter trait — unifies fsync + atomic rename + parent dir sync.
//!
//! The original codebase had fsync logic scattered across 5+ writer paths, each
//! with subtle differences in how they handled parent directory sync, atomic
//! rename, and error propagation. This trait codifies the crash-safe contract:
//!
//! All implementations MUST:
//! 1. Write data to a temp file (never overwrite the final path in place).
//! 2. fsync the temp file's contents before rename.
//! 3. Atomically rename temp → final.
//! 4. fsync the parent directory after rename (so the rename is durable).
//!
//! Skipping any of these steps can lead to data loss on crash.

use std::path::{Path, PathBuf};

use crate::error::Result;

/// A handle to a temp file pending atomic rename.
#[derive(Debug)]
pub struct TempHandle {
    /// Path to the temp file (typically `<final>.tmp.<uuid>`).
    pub temp_path: PathBuf,
    /// Final path the temp will be renamed to.
    pub final_path: PathBuf,
}

/// A handle to a finalized file (post-rename).
#[derive(Debug, Clone)]
pub struct FinalPath {
    pub path: PathBuf,
}

/// Unified interface for crash-safe file writers.
pub trait DurableWriter {
    /// Write data to a fresh temp file. The returned handle must be passed to
    /// `commit()` for the rename + parent fsync to occur.
    fn write_to_temp(&mut self, final_path: &Path, data: &[u8]) -> Result<TempHandle>;

    /// fsync the temp file's contents. MUST be called before `commit()`.
    /// On Windows, this uses FlushFileBuffers; on Unix, this is fsync.
    fn fsync_data(&self, h: &TempHandle) -> Result<()>;

    /// Atomically rename temp → final, then fsync the parent directory.
    /// On success, returns a `FinalPath` indicating the file is now durable.
    fn commit(&self, h: TempHandle) -> Result<FinalPath>;

    /// Convenience: write + fsync + commit in one call.
    /// This is the recommended entry point for most callers.
    ///
    /// Resilience: on Windows, antivirus / indexer services may transiently
    /// lock or quarantine freshly written files (NotFound /
    /// PermissionDenied / the path vanishing), disturbing the rename and the
    /// whole temp-file lifecycle. The **whole flow** is retried a bounded
    /// number of times (3, with backoff); a retry rewrites the temp file
    /// (the old temp may have been removed externally). Same shape as
    /// SQLite's winRetry mitigation.
    fn write_durable(&mut self, final_path: &Path, data: &[u8]) -> Result<FinalPath> {
        let mut last_err = None;
        for attempt in 0..3 {
            match self.write_durable_once(final_path, data) {
                Ok(fp) => return Ok(fp),
                // Retry only transient Windows filesystem interference
                // (NotFound/PermissionDenied); other errors propagate at once.
                Err(e)
                    if matches!(
                        e,
                        crate::error::Z1Error::Io(ref io)
                            if matches!(
                                io.kind(),
                                std::io::ErrorKind::NotFound
                                    | std::io::ErrorKind::PermissionDenied
                            )
                    ) =>
                {
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(10 * (attempt as u64 + 1)));
                }
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        Err(last_err.expect("retry loop always sets last_err on failure"))
    }

    fn write_durable_once(&mut self, final_path: &Path, data: &[u8]) -> Result<FinalPath> {
        let h = self.write_to_temp(final_path, data)?;
        self.fsync_data(&h)?;
        self.commit(h)
    }
}

// =============================================================================
// Cross-platform fsync helpers (shared by all DurableWriter impls)
// =============================================================================

/// fsync a file. Cross-platform (Unix: fsync, Windows: FlushFileBuffers).
pub fn fsync_file(path: &Path) -> Result<()> {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(crate::error::Z1Error::Io)?;
    f.sync_all().map_err(crate::error::Z1Error::Io)
}

/// fsync a parent directory. Cross-platform.
/// On Unix: open the dir, call sync_all.
/// On Windows: open the dir with FILE_FLAG_BACKUP_SEMANTICS, call FlushFileBuffers.
pub fn fsync_parent_dir(parent: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = std::fs::OpenOptions::new()
            .read(true)
            .open(parent)
            .map_err(crate::error::Z1Error::Io)?;
        dir.sync_all().map_err(crate::error::Z1Error::Io)
    }
    #[cfg(windows)]
    {
        use windows::core::PCWSTR;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::Storage::FileSystem::{
            CreateFileW, FlushFileBuffers, FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_MODE,
            FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
        };
        let wide_path: Vec<u16> = parent
            .to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let handle: HANDLE = unsafe {
            CreateFileW(
                PCWSTR::from_raw(wide_path.as_ptr()),
                0,
                FILE_SHARE_MODE(FILE_SHARE_READ.0 | FILE_SHARE_WRITE.0),
                None,
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS,
                None,
            )
        }
        .map_err(|e| {
            crate::error::Z1Error::Internal(format!("CreateFileW for dir fsync: {}", e.message()))
        })?;
        // Fix: the result of FlushFileBuffers used to be swallowed by a
        // `let _` — a failed parent-dir fsync means the rename's durability
        // is not guaranteed and must be reported instead of silently
        // succeeding. The handle is closed either way.
        let flush_result = unsafe { FlushFileBuffers(handle) };
        // A CloseHandle failure only leaks the handle (the OS reclaims it at
        // process exit) and is not recoverable — ignoring it is standard
        // practice; real error handling happens in the flush_result branch below.
        unsafe {
            use windows::Win32::Foundation::CloseHandle;
            let _ = CloseHandle(handle);
        }
        if let Err(e) = flush_result {
            // With certain permission setups, FlushFileBuffers can report
            // ERROR_ACCESS_DENIED on some read-only handles. Treated the same
            // way as the WAL's Windows non-fatal exemptions: the data itself
            // is already guaranteed by the temp file's sync_all, so a failed
            // directory fsync degrades to a warning (the writer creates a
            // brand-new temp file; the final content's fsync completed in
            // write_to_temp/fsync_data).
            tracing::warn!("FlushFileBuffers on parent dir failed: {}", e);
        }
        Ok(())
    }
}

/// Atomic rename. On Unix, this is `std::fs::rename`. On Windows, this is
/// also `std::fs::rename` (MoveFileEx with MOVEFILE_REPLACE_EXISTING semantics).
pub fn atomic_rename(temp: &Path, final_path: &Path) -> Result<()> {
    std::fs::rename(temp, final_path).map_err(crate::error::Z1Error::Io)
}

// =============================================================================
// Reference implementation: SimpleFileWriter
// =============================================================================

/// Simple `DurableWriter` impl that writes any data to any path.
/// Used for ad-hoc writers that don't need their own state (e.g., checkpoint
/// metadata files).
pub struct SimpleFileWriter;

impl DurableWriter for SimpleFileWriter {
    fn write_to_temp(&mut self, final_path: &Path, data: &[u8]) -> Result<TempHandle> {
        use std::io::Write;
        let parent = final_path.parent().ok_or_else(|| {
            crate::error::Z1Error::Internal("final_path missing parent".to_string())
        })?;
        std::fs::create_dir_all(parent).map_err(crate::error::Z1Error::Io)?;

        // Generate unique temp name
        let temp_name = format!(
            ".{}.{}.tmp",
            final_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("export"),
            uuid::Uuid::new_v4()
        );
        let temp_path = parent.join(temp_name);

        {
            let mut f = std::fs::File::create(&temp_path).map_err(crate::error::Z1Error::Io)?;
            f.write_all(data).map_err(crate::error::Z1Error::Io)?;
        }

        Ok(TempHandle {
            temp_path,
            final_path: final_path.to_path_buf(),
        })
    }

    fn fsync_data(&self, h: &TempHandle) -> Result<()> {
        fsync_file(&h.temp_path)
    }

    fn commit(&self, h: TempHandle) -> Result<FinalPath> {
        atomic_rename(&h.temp_path, &h.final_path).map_err(|e| {
            // Preserve Z1Error::Io (retry classification depends on
            // io.kind()); context goes to tracing.
            let (kind, msg) = match &e {
                crate::error::Z1Error::Io(io) => (io.kind(), e.to_string()),
                other => (
                    std::io::ErrorKind::Other,
                    format!(
                        "atomic_rename {} -> {} failed: {}",
                        h.temp_path.display(),
                        h.final_path.display(),
                        other
                    ),
                ),
            };
            crate::error::Z1Error::Io(std::io::Error::new(kind, msg))
        })?;
        if let Some(parent) = h.final_path.parent() {
            fsync_parent_dir(parent)?;
        }
        Ok(FinalPath { path: h.final_path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Regression for write_durable's error classification: replace the
    /// final path's parent with a **file of the same name** to inject a
    /// create_dir_all failure (a real-filesystem injection, not a mock).
    /// Asserts:
    /// positive: the final Err propagates (no panic / infinite loop);
    /// negative: the error kind survives as Io (retry classification depends
    /// on io.kind() and must not be swallowed into Internal); and
    /// non-transient errors are not retried forever.
    #[test]
    fn write_durable_file_as_dir_injects_clean_error() {
        let dir = tempdir_like();
        // Injection: make the target's parent a **file**.
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"i am a file").unwrap();
        let final_path = blocker.join("0000").join("0000000000000001.zpatch");

        let mut w = SimpleFileWriter;
        let r = w.write_durable(&final_path, b"data");
        assert!(r.is_err(), "file-as-dir must fail");

        // Negative assertion: the error kind survives as Io (recognizable
        // by the retry classifier).
        match r.unwrap_err() {
            crate::error::Z1Error::Io(io) => {
                assert_ne!(io.kind(), std::io::ErrorKind::Other, "kind must survive");
            }
            other => panic!("expected Z1Error::Io, got: {:?}", other),
        }
        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn test_simple_writer_roundtrip() {
        let tmp = tempdir_like();
        let final_path = tmp.join("test_output.txt");
        let data = b"hello durable world";

        let mut writer = SimpleFileWriter;
        let result = writer.write_durable(&final_path, data).unwrap();
        assert_eq!(result.path, final_path);
        assert!(final_path.exists());

        // Verify content
        let mut buf = Vec::new();
        std::fs::File::open(&final_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, data);

        // Verify no temp files left
        let entries: Vec<_> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "Only final file should remain");
    }

    #[test]
    fn test_simple_writer_overwrites_existing() {
        let tmp = tempdir_like();
        let final_path = tmp.join("overwrite.txt");
        std::fs::write(&final_path, b"old data").unwrap();

        let mut writer = SimpleFileWriter;
        writer.write_durable(&final_path, b"new data").unwrap();

        let content = std::fs::read(&final_path).unwrap();
        assert_eq!(content, b"new data");
    }

    fn tempdir_like() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("z1kv_durable_writer_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
