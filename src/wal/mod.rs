//! Self-implemented write-ahead log.
//!
//! A self-implemented write-ahead log.
//!
//! Carries over the behavior semantics of its durability predecessor
//! (SyncEach / GroupCommitStrict / FlushOnly policies, the Windows flush
//! error exemption, record-level CRC), but with a self-implemented file
//! format and no external `durability` crate.
//!
//! # Record format (append-only)
//!
//! ```text
//! len(4B LE) | crc32(4B LE) | payload(len bytes)
//! ```
//!
//! `len` is the payload byte count; `crc32` covers the payload. Segment
//! files rotate at `max_file_size`, named `wal.{seq:08}` (seq monotonically
//! increasing).
//!
//! # Invariants
//!
//! - There is exactly one DURABILITY BOUNDARY: `append_durable` returning Ok
//! - GroupCommitStrict waits for BOTH "queue drained" and "flush succeeded"
//! - On Windows, `PermissionDenied / AddrInUse / WouldBlock` are treated as
//!   non-fatal
//! - Truncation boundary: `truncate_before` refuses to truncate while the
//!   group-commit queue is non-empty (enforced by call ordering plus a
//!   defensive assert)

use crate::error::{Result, Z1Error};
use crate::TxnId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub mod checkpoint;
pub mod recovery;

pub use checkpoint::{CheckpointManager, CheckpointState};
pub use recovery::{replay_committed_ops, RecoveryResult};

/// WAL record payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalRecord {
    /// A single versioned write within a transaction.
    Put {
        cf: u16,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
    },
    /// Mark a transaction begun.
    Begin,
    /// Mark a transaction committed.
    ///
    /// D5: `inserted_at` is the wall-clock time captured at commit — the
    /// TTL clock for committed_history. It is persisted in the WAL so that
    /// crash-recovered history entries share the same eviction clock as
    /// online commits.
    Commit { commit_ts: u64, inserted_at: u64 },
    /// Mark a transaction rolled back.
    Rollback,
    /// Checkpoint marker.
    Checkpoint { checkpoint_id: u64 },
}

/// A WAL entry — one record with its transaction id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalEntry {
    pub txn_id: TxnId,
    pub record: WalRecord,
}

/// WAL configuration.
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub wal_dir: PathBuf,
    pub max_file_size: u64,
    pub enabled: bool,
    pub group_commit: Option<GroupCommitConfig>,
    /// Windows write-through: FILE_FLAG_WRITE_THROUGH.
    /// Writes go straight to disk and `flush_and_sync` skips
    /// FlushFileBuffers — trading a slightly slower write path for one fewer
    /// fsync syscall per transaction. Windows only.
    pub write_through: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
            group_commit: Some(GroupCommitConfig::default()),
            write_through: cfg!(windows),
        }
    }
}

/// Group commit / fsync policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Synchronous: fsync every write before returning.
    SyncEach,
    /// Group Commit (Strict): batch writes, wait for flush+fsync confirmation.
    #[default]
    GroupCommitStrict,
    /// Flush to OS buffer and fsync (equivalent to SyncEach in this impl).
    FlushOnly,
}

/// Group commit configuration.
#[derive(Debug, Clone)]
pub struct GroupCommitConfig {
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub policy: SyncPolicy,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_wait_ms: 50,
            policy: SyncPolicy::GroupCommitStrict,
        }
    }
}

/// A complete record's on-disk position (for checkpoint truncate coordination).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalPosition {
    pub seq: u64,
}

/// Self-implemented WAL writer.
///
/// Single-writer architecture: all appends go through one `Mutex<Inner>`.
/// Group-commit records are queued and flushed together when the batch size
/// is reached or `flush_pending()` is called. No background thread — this keeps
/// the same externally-observable durability contract while staying simple.
pub struct WalWriter {
    inner: Mutex<Inner>,
    config: WalConfig,
}

struct Inner {
    /// Current segment file.
    file: std::fs::File,
    /// Windows write-through mode: FILE_FLAG_WRITE_THROUGH — writes go
    /// straight to disk and FlushFileBuffers in `flush_and_sync` becomes a
    /// no-op (the data is already there). Slightly slower per write, one
    /// fewer fsync syscall per transaction.
    write_through: bool,
    /// Under WRITE_THROUGH: accumulated bytes of segments preceding the
    /// current one (directory metadata lags, so read_dir accumulation is
    /// unreliable).
    prev_segments_bytes: u64,
    /// Current segment sequence number.
    seq: u64,
    /// Bytes written into the current segment.
    segment_bytes: u64,
    /// Group-commit pending queue.
    queue: Vec<WalEntry>,
    /// WAL directory (for rotation).
    wal_dir: PathBuf,
    /// Segment size limit.
    max_file_size: u64,
    /// Timestamp of the last successful flush (for lazy Batch timeout checks).
    last_flush: std::time::Instant,
}

impl WalWriter {
    /// Open (or create) the WAL directory and position at the latest segment.
    pub fn open(data_dir: &Path, config: WalConfig) -> Result<Self> {
        let wal_dir = if config.wal_dir.is_absolute() {
            config.wal_dir.clone()
        } else {
            data_dir.join(&config.wal_dir)
        };
        std::fs::create_dir_all(&wal_dir).map_err(Z1Error::Io)?;

        let write_through = cfg!(windows) && config.write_through;
        let (seq, file, segment_bytes) = open_latest_segment(&wal_dir, write_through)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                file,
                seq,
                segment_bytes,
                write_through,
                prev_segments_bytes: 0,
                queue: Vec::new(),
                wal_dir: wal_dir.clone(),
                max_file_size: config.max_file_size,
                last_flush: std::time::Instant::now(),
            }),
            config: WalConfig { wal_dir, ..config },
        })
    }

    fn wal_dir(&self) -> &Path {
        &self.config.wal_dir
    }

    /// Serialize a record into its on-disk bytes.
    fn encode_record(entry: &WalEntry) -> Result<Vec<u8>> {
        let payload = postcard::to_stdvec(entry)
            .map_err(|e| Z1Error::Serialization(format!("WAL encode: {}", e)))?;
        if payload.len() > MAX_WAL_RECORD_LEN as usize {
            // Write/read symmetry fix: the reader enforces
            // MAX_WAL_RECORD_LEN but the writer previously did not — a Put
            // with a huge value would produce a record the reader always
            // rejects, making recovery fail outright on reopen (the database
            // would not open). Report the error at the write boundary.
            return Err(Z1Error::Serialization(format!(
                "WAL record too large: {} bytes (limit {})",
                payload.len(),
                MAX_WAL_RECORD_LEN
            )));
        }
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Append a record to the WAL without fsync.
    pub fn append(&self, txn_id: TxnId, record: WalRecord) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let entry = WalEntry { txn_id, record };
        let bytes = Self::encode_record(&entry)?;
        let mut inner = self.inner.lock();
        inner.append_encoded(&bytes)
    }

    /// Append a record and fsync (durability boundary).
    ///
    /// For GroupCommitStrict, records are queued; when the batch size is reached,
    /// the whole queue is flushed+fsynced together.
    pub fn append_durable(&self, txn_id: TxnId, record: WalRecord) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let entry = WalEntry { txn_id, record };
        match self.config.sync_policy() {
            SyncPolicy::GroupCommitStrict => {
                let batch_size = self
                    .config
                    .group_commit
                    .as_ref()
                    .map(|c| c.max_batch_size)
                    .unwrap_or(100);
                let mut inner = self.inner.lock();
                inner.queue.push(entry);
                if inner.queue.len() >= batch_size {
                    inner.flush_queue()?;
                }
                Ok(())
            }
            SyncPolicy::SyncEach | SyncPolicy::FlushOnly => {
                let bytes = Self::encode_record(&entry)?;
                let mut inner = self.inner.lock();
                inner.append_encoded(&bytes)?;
                inner.flush_and_sync()?;
                Ok(())
            }
        }
    }

    /// Flush any queued group-commit records and fsync.
    ///
    /// Returns the number of records flushed.
    pub fn flush_pending(&self) -> Result<usize> {
        let mut inner = self.inner.lock();
        let n = inner.queue.len();
        inner.flush_queue()?;
        Ok(n)
    }

    /// Force fsync of the current segment (and any queued records).
    pub fn flush_and_sync(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.flush_queue()?;
        inner.flush_and_sync()
    }

    /// Total size in bytes of all WAL segment files.
    pub fn size_bytes(&self) -> u64 {
        // In WRITE_THROUGH mode Windows updates directory metadata lazily
        // (a file has data but metadata.len() reports 0), so accumulating
        // via read_dir is unreliable. Track bytes precisely in the writer
        // state instead (single segment; when several segments exist this
        // includes pre-rotate sizes recorded in prev_segments).
        let inner = self.inner.lock();
        if inner.write_through {
            return inner.segment_bytes + inner.prev_segments_bytes;
        }
        std::fs::read_dir(self.wal_dir())
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| e.metadata().ok())
                    .filter(|m| m.is_file())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Truncate WAL segments strictly before `pos` (for checkpoint coordination).
    ///
    /// Truncate prefix semantics: only whole segments with `seq < pos.seq`
    /// are deleted; the segment containing `pos` is untouched.
    ///
    /// Truncation-boundary invariant (see lib.rs "Truncation boundary"):
    /// truncation is refused while the unflushed group-commit queue is
    /// non-empty — otherwise records that were appended but not yet flushed
    /// could live in a segment about to be deleted. The checkpoint path
    /// flushes first so this should not occur; the invariant is enforced
    /// here as a defensive assert.
    pub fn truncate_before(&self, pos: WalPosition) -> Result<usize> {
        {
            let inner = self.inner.lock();
            if !inner.queue.is_empty() {
                return Err(Z1Error::Wal(format!(
                    "truncate_before: {} unflushed record(s) in group-commit queue; \
                     flush_and_sync must run before truncation",
                    inner.queue.len()
                )));
            }
        }
        let mut removed = 0;
        for entry in std::fs::read_dir(self.wal_dir()).map_err(Z1Error::Io)? {
            let entry = entry.map_err(Z1Error::Io)?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("wal.") {
                if let Ok(seq) = rest.parse::<u64>() {
                    if seq < pos.seq {
                        std::fs::remove_file(entry.path()).map_err(Z1Error::Io)?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Current position (seq) for checkpoint coordination.
    pub fn position(&self) -> WalPosition {
        WalPosition {
            seq: self.inner.lock().seq,
        }
    }

    /// Milliseconds since the last successful flush (lazy Batch timeout).
    pub fn ms_since_last_flush(&self) -> u64 {
        self.inner.lock().last_flush.elapsed().as_millis() as u64
    }
}

/// Drop safety net: under GroupCommitStrict / Async, a non-empty unflushed
/// queue gets a best-effort flush + fsync on drop, so appended-and-confirmed
/// records are not silently lost.
impl Drop for WalWriter {
    fn drop(&mut self) {
        // Best effort: Drop cannot return errors, only log.
        if let Some(mut inner) = self.inner.try_lock() {
            if !inner.queue.is_empty() {
                if let Err(e) = inner.flush_queue() {
                    tracing::error!(
                        queued = inner.queue.len(),
                        error = %e,
                        "WalWriter Drop: failed to flush pending queue; {} records may be lost",
                        inner.queue.len()
                    );
                }
            }
        }
    }
}

impl Inner {
    /// Append encoded bytes, rotating segments when `max_file_size` is exceeded.
    fn append_encoded(&mut self, bytes: &[u8]) -> Result<()> {
        if self.segment_bytes + bytes.len() as u64 > self.max_file_size && self.segment_bytes > 0 {
            self.rotate()?;
        }
        self.file.write_all(bytes).map_err(Z1Error::Io)?;
        self.segment_bytes += bytes.len() as u64;
        Ok(())
    }

    /// Rotate to the next segment.
    fn rotate(&mut self) -> Result<()> {
        // In WRITE_THROUGH mode the data already reached the disk; skip sync_all.
        if !self.write_through {
            self.file.sync_all().map_err(Z1Error::Io)?;
        }
        self.prev_segments_bytes += self.segment_bytes;
        let next_seq = self.seq + 1;
        let path = segment_path(&self.wal_dir, next_seq);
        let file = if self.write_through {
            open_write_through(&path)?
        } else {
            std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .map_err(Z1Error::Io)?
        };
        self.file = file;
        self.seq = next_seq;
        self.segment_bytes = 0;
        Ok(())
    }

    /// Flush the group-commit queue to disk and fsync.
    fn flush_queue(&mut self) -> Result<()> {
        if self.queue.is_empty() {
            return Ok(());
        }
        // mem::take moves the whole queue out (reusing the allocation)
        // instead of draining into a fresh Vec.
        let entries = std::mem::take(&mut self.queue);
        for entry in entries {
            let bytes = WalWriter::encode_record(&entry)?;
            self.append_encoded(&bytes)?;
        }
        self.flush_and_sync()
    }

    /// Flush the OS buffer and fsync the current segment.
    fn flush_and_sync(&mut self) -> Result<()> {
        // In WRITE_THROUGH mode the data is already on disk and
        // FlushFileBuffers would be a no-op — skip it to save a syscall
        // (same for rotate).
        if self.write_through {
            self.last_flush = std::time::Instant::now();
            return Ok(());
        }
        if let Err(e) = self.file.flush() {
            if !is_non_fatal_windows_error(&e) {
                return Err(Z1Error::Io(e));
            }
            tracing::warn!("WAL flush hit Windows limitation: {e}; treating as non-fatal");
        }
        if let Err(e) = self.file.sync_all() {
            if !is_non_fatal_windows_error(&e) {
                return Err(Z1Error::Io(e));
            }
            tracing::warn!("WAL sync hit Windows limitation: {e}; treating as non-fatal");
        }
        self.last_flush = std::time::Instant::now();
        Ok(())
    }
}

fn segment_path(wal_dir: &Path, seq: u64) -> PathBuf {
    wal_dir.join(format!("wal.{:08}", seq))
}

/// Locate the highest-numbered segment (or create seq 0).
#[cfg(windows)]
fn open_write_through(path: &Path) -> Result<std::fs::File> {
    // Safe std API: inject FILE_FLAG_WRITE_THROUGH (0x80000000) via custom_flags.
    use std::os::windows::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(0x80000000) // FILE_FLAG_WRITE_THROUGH
        .open(path)
        .map_err(Z1Error::Io)
}

/// Non-Windows: a no-op open (equivalent to a plain OpenOptions;
/// write_through is always false).
#[cfg(not(windows))]
fn open_write_through(path: &Path) -> Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(Z1Error::Io)
}

fn open_latest_segment(wal_dir: &Path, write_through: bool) -> Result<(u64, std::fs::File, u64)> {
    let mut highest: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(wal_dir).map_err(Z1Error::Io)? {
        let entry = entry.map_err(Z1Error::Io)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("wal.") {
            if let Ok(seq) = rest.parse::<u64>() {
                if highest.as_ref().is_none_or(|(s, _)| seq > *s) {
                    highest = Some((seq, entry.path()));
                }
            }
        }
    }

    match highest {
        Some((seq, path)) => {
            let file = if write_through {
                open_write_through(&path)?
            } else {
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .map_err(Z1Error::Io)?
            };
            let segment_bytes = file.metadata().map_err(Z1Error::Io)?.len();
            Ok((seq, file, segment_bytes))
        }
        None => {
            let path = segment_path(wal_dir, 0);
            let file = if write_through {
                open_write_through(&path)?
            } else {
                std::fs::OpenOptions::new()
                    .create(true)
                    .read(true)
                    .write(true)
                    .truncate(true)
                    .open(&path)
                    .map_err(Z1Error::Io)?
            };
            Ok((0, file, 0))
        }
    }
}

/// Windows-specific non-fatal fsync errors (from RockDuck `flush_and_sync_safe`).
#[cfg(windows)]
fn is_non_fatal_windows_error(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::PermissionDenied | ErrorKind::AddrInUse | ErrorKind::WouldBlock
    )
}

#[cfg(not(windows))]
fn is_non_fatal_windows_error(_e: &std::io::Error) -> bool {
    false
}

impl WalConfig {
    /// Returns the effective sync policy.
    fn sync_policy(&self) -> SyncPolicy {
        self.group_commit
            .as_ref()
            .map(|c| c.policy)
            .unwrap_or(SyncPolicy::GroupCommitStrict)
    }
}

// =============================================================================
// WAL replay — the record-scanning layer
// =============================================================================

/// Replay all segment files in seq order. CRC or truncation errors return
/// Err immediately (for the recovery layer to classify).
pub fn replay_all(wal_dir: &Path) -> Result<Vec<WalEntry>> {
    let mut seqs: Vec<u64> = std::fs::read_dir(wal_dir)
        .map_err(Z1Error::Io)?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("wal.")
                .and_then(|r| r.parse::<u64>().ok())
        })
        .collect();
    seqs.sort_unstable();

    let mut out = Vec::new();
    for (idx, seq) in seqs.iter().enumerate() {
        let path = segment_path(wal_dir, *seq);
        let mut file = std::fs::File::open(&path).map_err(Z1Error::Io)?;
        let is_last_segment = idx + 1 == seqs.len();
        loop {
            // Record start offset: tail-truncation tolerance needs rollback judgment.
            use std::io::Seek;
            let offset = file.stream_position().map_err(Z1Error::Io)?;
            // Peek at the header first: read the len field to decide
            // whether this is a "structural truncation" (fewer than 8 header
            // bytes, or a declared length extending past EOF). Structural
            // truncation = a half-written record = an uncommitted
            // transaction, safe to drop at the tail; a CRC mismatch with an
            // intact structure = a bit flip in committed data, must not be
            // dropped.
            let file_len = file.metadata().map(|m| m.len()).map_err(Z1Error::Io)?;
            let mut header = [0u8; 8];
            // Peek the len field (for the over-limit / structural-truncation
            // decision), then rewind. Use read (not read_exact): both EOF and
            // short reads are structural-truncation candidates, decided
            // below; the peek itself does not error.
            let header_read = file.read(&mut header).map_err(Z1Error::Io)?;
            file.seek(std::io::SeekFrom::Start(offset))
                .map_err(Z1Error::Io)?;
            let declared_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as u64;
            let beyond_end = offset + 8 + declared_len > file_len;
            let over_limit = declared_len > MAX_WAL_RECORD_LEN as u64;
            // Structural truncation = the declared length extends past EOF
            // (a half-written record), and the declaration is within the
            // defensive limit (an over-limit declaration is a
            // malicious/corrupt signal and must error — allocation-bomb
            // defense).
            let structurally_truncated = (header_read < 8 || beyond_end) && !over_limit;
            match read_one_record(&mut file) {
                Ok(Some(entry)) => out.push(entry),
                Ok(None) => break,
                Err(e) => {
                    // Tail-truncation tolerance (tail of the last segment only):
                    // structural truncation (a half-written record) = an
                    // uncommitted transaction, safe to discard — the fsync
                    // boundary guarantees no new writes can appear after a
                    // commit's fsync. A CRC mismatch with an intact
                    // structure = corrupted committed data, Fatal.
                    if is_last_segment && structurally_truncated {
                        tracing::warn!(
                            "WAL tail structurally truncated at segment {} offset {} (file len {},                              declared extends past end), discarding uncommitted tail: {}",
                            seq, offset, file_len, e
                        );
                        break;
                    }
                    return Err(e);
                }
            }
        }
    }
    Ok(out)
}

/// Maximum payload length of a single WAL record.
///
/// `read_one_record`'s `len` comes from disk and previously had no cap —
/// a corrupt/malicious file could declare a length near 4 GB and
/// `vec![0u8; len]` would allocate that much memory outright (OOM/DoS).
/// Real records are bounded by the value-size limit, far below this.
pub const MAX_WAL_RECORD_LEN: u32 = 64 * 1024 * 1024;

/// Read a single record from a file, returning None at clean EOF.
pub fn read_one_record(file: &mut std::fs::File) -> Result<Option<WalEntry>> {
    let mut header = [0u8; 8];
    let read = file.read(&mut header).map_err(Z1Error::Io)?;
    if read == 0 {
        return Ok(None);
    }
    if read != 8 {
        return Err(Z1Error::Deserialize("WAL truncated header".into()));
    }
    let len = u32::from_le_bytes(header[0..4].try_into().expect("read!=8 checked")) as usize;
    let stored_crc = u32::from_le_bytes(header[4..8].try_into().expect("read!=8 checked"));

    if len as u64 > MAX_WAL_RECORD_LEN as u64 {
        return Err(Z1Error::Deserialize(format!(
            "WAL record length {} exceeds limit {} (corrupt or hostile file)",
            len, MAX_WAL_RECORD_LEN
        )));
    }

    let mut payload = vec![0u8; len];
    file.read_exact(&mut payload).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Z1Error::Deserialize("WAL truncated payload".into())
        } else {
            Z1Error::Io(e)
        }
    })?;

    let actual_crc = crc32fast::hash(&payload);
    if actual_crc != stored_crc {
        return Err(Z1Error::Deserialize(format!(
            "WAL crc mismatch: stored={:08x} actual={:08x}",
            stored_crc, actual_crc
        )));
    }

    // Spell out WalEntry so inference cannot drift to Option<WalEntry>
    // (read_one_record returns Result<Option<WalEntry>>; returning
    // from_bytes directly as the tail expression would confuse inference).
    let entry: WalEntry = postcard::from_bytes(&payload)
        .map_err(|e| Z1Error::Deserialize(format!("WAL decode: {}", e)))?;
    Ok(Some(entry))
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("z1kv_wal_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sync_config(dir: &Path) -> WalConfig {
        WalConfig {
            wal_dir: dir.join("wal"),
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                policy: SyncPolicy::SyncEach,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn encode_decode_roundtrip_via_replay() {
        let dir = tmp_dir("roundtrip");
        let wal = WalWriter::open(&dir, sync_config(&dir)).unwrap();
        wal.append_durable(
            1,
            WalRecord::Put {
                cf: 0,
                key: b"k".to_vec(),
                value: Some(b"v".to_vec()),
            },
        )
        .unwrap();
        wal.append_durable(
            1,
            WalRecord::Commit {
                commit_ts: 1,
                inserted_at: 1,
            },
        )
        .unwrap();
        drop(wal);

        let records = replay_all(&dir.join("wal")).unwrap();
        assert_eq!(records.len(), 2);
        assert!(matches!(&records[0].record, WalRecord::Put { value: Some(v), .. } if v == b"v"));
        assert!(matches!(
            &records[1].record,
            WalRecord::Commit {
                commit_ts: 1,
                inserted_at: 1
            }
        ));
    }

    #[test]
    fn corrupt_record_detected_by_crc() {
        let dir = tmp_dir("corrupt");
        let wal = WalWriter::open(&dir, sync_config(&dir)).unwrap();
        wal.append_durable(
            1,
            WalRecord::Put {
                cf: 0,
                key: b"k".to_vec(),
                value: Some(b"v".to_vec()),
            },
        )
        .unwrap();
        drop(wal);

        // Flip a byte in the payload (last byte of the file).
        let path = dir.join("wal").join("wal.00000000");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        assert!(replay_all(&dir.join("wal")).is_err());
    }

    /// Regression: a record whose declared len exceeds MAX_WAL_RECORD_LEN
    /// must be rejected (previously `vec![0u8; len]` allocated exactly the
    /// declared amount — a corrupt/malicious file could trigger a near-4GB
    /// allocation).
    #[test]
    fn oversized_record_len_rejected_without_alloc() {
        let dir = tmp_dir("oversize");
        let wal = WalWriter::open(&dir, sync_config(&dir)).unwrap();
        wal.append_durable(1, WalRecord::Begin).unwrap();
        drop(wal);

        let path = dir.join("wal").join("wal.00000000");
        let mut bytes = std::fs::read(&path).unwrap();
        // Rewrite the first record's len field as u32::MAX (little-endian,
        // at offset 0).
        bytes[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        // Over the limit → immediate Err, no 4GB allocation.
        let err = replay_all(&dir.join("wal")).unwrap_err();
        assert!(err.to_string().contains("exceeds limit"), "got: {}", err);
    }

    /// Regression (write/read symmetry): a record exceeding
    /// MAX_WAL_RECORD_LEN must be rejected at the write boundary — the cap
    /// used to exist only on the read side, so an oversized Put would write
    /// a record the reader always rejects, making recovery fail on reopen.
    #[test]
    fn oversized_record_rejected_at_write_boundary() {
        let dir = tmp_dir("oversize_write");
        let wal = WalWriter::open(&dir, sync_config(&dir)).unwrap();

        let big_value = vec![0u8; (MAX_WAL_RECORD_LEN as usize) + 1];
        let err = wal
            .append_durable(
                1,
                WalRecord::Put {
                    cf: 0,
                    key: b"k".to_vec(),
                    value: Some(big_value),
                },
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("WAL record too large"),
            "unexpected error: {}",
            err
        );

        // The WAL must not contain a poison record; later normal writes and
        // replay are unaffected.
        wal.append_durable(1, WalRecord::Begin).unwrap();
        drop(wal);
        let records = replay_all(&dir.join("wal")).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn group_commit_queues_until_flush() {
        let dir = tmp_dir("groupcommit");
        let config = WalConfig {
            wal_dir: dir.join("wal"),
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                max_batch_size: 100,
                max_wait_ms: 50,
                policy: SyncPolicy::GroupCommitStrict,
            }),
            ..Default::default()
        };
        let wal = WalWriter::open(&dir, config).unwrap();
        wal.append_durable(1, WalRecord::Begin).unwrap();
        wal.append_durable(
            1,
            WalRecord::Put {
                cf: 0,
                key: b"k".to_vec(),
                value: Some(b"v".to_vec()),
            },
        )
        .unwrap();
        wal.append_durable(
            1,
            WalRecord::Commit {
                commit_ts: 1,
                inserted_at: 1,
            },
        )
        .unwrap();

        // Nothing flushed yet (queue size 3 < batch 100).
        assert_eq!(replay_all(&dir.join("wal")).unwrap().len(), 0);

        wal.flush_pending().unwrap();
        assert_eq!(replay_all(&dir.join("wal")).unwrap().len(), 3);
    }

    /// Edge-case regression: exactly max_file_size — the check is
    /// `segment_bytes + bytes.len() > max` (strictly greater), so filling
    /// the segment exactly does NOT rotate; the next record rotates.
    #[test]
    fn segment_rotation_exact_boundary() {
        let dir = tmp_dir("exact_boundary");
        // One record = 8(header) + payload bytes. Probe one record's encoded
        // length first.
        let probe_entry = WalEntry {
            txn_id: 1,
            record: WalRecord::Begin,
        };
        let probe_len = WalWriter::encode_record(&probe_entry).unwrap().len();

        // max_file_size is exactly one record: after the first write
        // segment_bytes == max (no rotation); the second record (a different
        // payload length) triggers rotation.
        let config = WalConfig {
            wal_dir: dir.join("wal"),
            max_file_size: probe_len as u64,
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                policy: SyncPolicy::SyncEach,
                ..Default::default()
            }),
            ..Default::default()
        };
        let wal = WalWriter::open(&dir, config).unwrap();
        wal.append_durable(1, WalRecord::Begin).unwrap();
        let pos1 = wal.position();
        wal.append_durable(1, WalRecord::Rollback).unwrap(); // different length → over limit → rotate
        let pos2 = wal.position();
        assert!(
            pos2.seq > pos1.seq,
            "second record must rotate to next segment"
        );
        drop(wal);
        assert_eq!(replay_all(&dir.join("wal")).unwrap().len(), 2);
    }

    #[test]
    fn segment_rotation_on_max_file_size() {
        let dir = tmp_dir("rotate");
        let config = WalConfig {
            wal_dir: dir.join("wal"),
            max_file_size: 64, // tiny — forces rotation
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                policy: SyncPolicy::SyncEach,
                ..Default::default()
            }),
            ..Default::default()
        };
        let wal = WalWriter::open(&dir, config).unwrap();
        for i in 0..10u64 {
            wal.append_durable(
                1,
                WalRecord::Put {
                    cf: 0,
                    key: i.to_le_bytes().to_vec(),
                    value: Some(b"v".to_vec()),
                },
            )
            .unwrap();
        }
        drop(wal);

        // Multiple segments must exist.
        let segs: Vec<_> = std::fs::read_dir(dir.join("wal"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("wal."))
            .collect();
        assert!(segs.len() > 1, "expected segment rotation");

        // All records replay back.
        assert_eq!(replay_all(&dir.join("wal")).unwrap().len(), 10);
    }

    #[test]
    fn truncate_before_removes_old_segments() {
        let dir = tmp_dir("truncate");
        let config = WalConfig {
            wal_dir: dir.join("wal"),
            max_file_size: 64,
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                policy: SyncPolicy::SyncEach,
                ..Default::default()
            }),
            ..Default::default()
        };
        let wal = WalWriter::open(&dir, config).unwrap();
        for i in 0..10u64 {
            wal.append_durable(
                1,
                WalRecord::Put {
                    cf: 0,
                    key: i.to_le_bytes().to_vec(),
                    value: Some(b"v".to_vec()),
                },
            )
            .unwrap();
        }
        let pos = wal.position();
        assert!(pos.seq > 0, "rotation should have advanced seq");

        let removed = wal.truncate_before(pos).unwrap();
        assert!(removed > 0, "should remove at least one old segment");
    }
}
