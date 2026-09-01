//! Checkpoint — crash-recovery baseline + WAL truncation closed loop.
//!
//! The checkpoint content converges on the MVCC baseline
//! (committed_txn + commit_ts_map).
//!
//! # Crash-safe protocol
//!
//! 1. Serialize CheckpointState → `.ckpt.{id}.tmp` (postcard + crc32 envelope)
//! 2. fsync(tmp) → rename(tmp, final)
//! 3. Write `_LATEST` pointing at the new checkpoint (tmp → fsync → rename)
//! 4. Append a WAL Checkpoint marker + flush_and_sync
//! 5. WAL truncate_before (drop old segments before the `_LATEST` position)
//!
//! On recovery: read `_LATEST` → get the committed_txn baseline → replay
//! the WAL above that baseline.
//!
//! # Easy-to-confuse points
//!
//! - A checkpoint is an optimization to speed up recovery, not the source of
//!   correctness — the WAL is always the source of truth; if a checkpoint is
//!   corrupted, recovery falls back to a full WAL replay.
//! - Truncation must strictly follow checkpoint durability (written +
//!   fsynced), otherwise committed data not yet covered by the checkpoint
//!   could be truncated away.

use crate::durable_writer::DurableWriter;
use crate::error::{Result, Z1Error};
use crate::wal::{WalRecord, WalWriter};
use crate::TxnId;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Checkpoint file magic.
const CKPT_MAGIC: [u8; 4] = *b"CKPT";
/// Checkpoint format version.
const CKPT_VERSION: u8 = 1;

/// Checkpoint content: the MVCC recovery baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    pub checkpoint_id: TxnId,
    pub committed_txn: TxnId,
    pub timestamp_ms: u64,
    /// Committed-history snapshot: txn_id -> commit_ts (restores
    /// committed_history).
    pub committed_history: Vec<(TxnId, u64)>,
}

/// Checkpoint manager.
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
}

/// Checkpoint file format with a crc32 envelope:
/// `magic(4) | version(1) | crc32(4) | payload_len(4) | payload(N)`
const ENVELOPE_HEADER: usize = 4 + 1 + 4 + 4;

impl CheckpointManager {
    pub fn new(data_dir: &Path) -> Self {
        let checkpoint_dir = data_dir.join("checkpoints");
        // A failure here is not fatal (the write-phase `write_durable`
        // creates the dir again and reports explicitly), but log it for
        // diagnostics.
        if let Err(e) = std::fs::create_dir_all(&checkpoint_dir) {
            tracing::warn!(
                "CheckpointManager: create_dir_all({:?}) failed: {}",
                checkpoint_dir,
                e
            );
        }
        Self { checkpoint_dir }
    }

    fn checkpoint_path(&self, id: TxnId) -> PathBuf {
        self.checkpoint_dir.join(format!("ckpt_{:016}.bin", id))
    }

    fn latest_path(&self) -> PathBuf {
        self.checkpoint_dir.join("_LATEST")
    }

    /// Serialize and write a checkpoint (crash-safe).
    pub fn write(&self, state: &CheckpointState) -> Result<()> {
        let payload = postcard::to_stdvec(state)
            .map_err(|e| Z1Error::Serialization(format!("checkpoint encode: {}", e)))?;
        if payload.len() > u32::MAX as usize {
            // Fix: `len` is a u32 field; silent truncation would write a
            // corrupt file — fail loudly instead.
            return Err(Z1Error::Serialization(format!(
                "checkpoint payload too large: {} bytes (u32 limit exceeded)",
                payload.len()
            )));
        }
        let crc = crc32fast::hash(&payload);
        let len = payload.len() as u32;

        let mut envelope = Vec::with_capacity(ENVELOPE_HEADER + payload.len());
        envelope.extend_from_slice(&CKPT_MAGIC);
        envelope.push(CKPT_VERSION);
        envelope.extend_from_slice(&crc.to_le_bytes());
        envelope.extend_from_slice(&len.to_le_bytes());
        envelope.extend_from_slice(&payload);

        // 1. Write tmp + fsync + rename.
        let final_path = self.checkpoint_path(state.checkpoint_id);
        let mut writer = crate::durable_writer::SimpleFileWriter;
        writer.write_durable(&final_path, &envelope)?;

        // 2. Write the _LATEST pointer.
        let latest_payload = state.checkpoint_id.to_le_bytes();
        writer.write_durable(&self.latest_path(), &latest_payload)?;

        tracing::info!(
            checkpoint_id = state.checkpoint_id,
            committed_txn = state.committed_txn,
            "checkpoint written"
        );
        Ok(())
    }

    /// Read the latest checkpoint (if present and uncorrupted). Returns
    /// None when corrupted (fall back to a full WAL replay).
    pub fn load_latest(&self) -> Option<CheckpointState> {
        let latest_path = self.latest_path();
        let id_bytes = std::fs::read(&latest_path).ok()?;
        if id_bytes.len() != 8 {
            return None;
        }
        let id = u64::from_le_bytes(id_bytes.try_into().ok()?);

        let path = self.checkpoint_path(id);
        let bytes = std::fs::read(&path).ok()?;

        // Validate the envelope.
        if bytes.len() < ENVELOPE_HEADER {
            tracing::warn!("checkpoint {} truncated, falling back to WAL replay", id);
            return None;
        }
        if bytes[..4] != CKPT_MAGIC || bytes[4] != CKPT_VERSION {
            tracing::warn!(
                "checkpoint {} magic/version mismatch, falling back to WAL replay",
                id
            );
            return None;
        }
        let stored_crc = u32::from_le_bytes(bytes[5..9].try_into().ok()?);
        let len = u32::from_le_bytes(bytes[9..13].try_into().ok()?) as usize;
        // Fix: `len` comes from disk. On a truncated file,
        // `ENVELOPE_HEADER + len` can exceed the actual byte count and a
        // naive slice would panic (crashing on open). An out-of-range len is
        // treated as corruption and falls back to WAL replay.
        let payload_end = ENVELOPE_HEADER.checked_add(len)?;
        if bytes.len() < payload_end {
            tracing::warn!(
                "checkpoint {} truncated (len {} > {} bytes), falling back to WAL replay",
                id,
                len,
                bytes.len()
            );
            return None;
        }
        let payload = &bytes[ENVELOPE_HEADER..payload_end];
        if crc32fast::hash(payload) != stored_crc {
            tracing::warn!("checkpoint {} crc mismatch, falling back to WAL replay", id);
            return None;
        }

        match postcard::from_bytes::<CheckpointState>(payload) {
            Ok(state) => Some(state),
            Err(e) => {
                tracing::warn!("checkpoint {} deserialize failed: {}, falling back", id, e);
                None
            }
        }
    }

    /// Run one checkpoint: write the state + WAL marker + WAL truncate.
    ///
    /// A 4-step condensed protocol (the WAL is not truncated into the
    /// current segment).
    pub fn checkpoint(
        &self,
        wal: &WalWriter,
        checkpoint_id: TxnId,
        committed_txn: TxnId,
        committed_history: Vec<(TxnId, u64)>,
    ) -> Result<()> {
        // 1. Write the checkpoint file (including _LATEST).
        let state = CheckpointState {
            checkpoint_id,
            committed_txn,
            timestamp_ms: crate::codec::current_timestamp_millis(),
            committed_history,
        };
        self.write(&state)?;

        // 2. WAL marker + flush (durability boundary).
        wal.append_durable(committed_txn, WalRecord::Checkpoint { checkpoint_id })?;

        // 3. WAL truncate: drop old segments before the checkpoint.
        //    `checkpoint_id` is already durable via append_durable, so
        //    truncating is safe.
        let pos = wal.position();
        let removed = wal.truncate_before(pos)?;
        tracing::info!(removed, "WAL truncated after checkpoint");

        Ok(())
    }

    /// Recovery baseline: read committed_txn + history from the checkpoint.
    /// Returns None if there is no checkpoint.
    pub fn recovery_baseline(&self) -> Option<(TxnId, Vec<(TxnId, u64)>)> {
        self.load_latest()
            .map(|s| (s.committed_txn, s.committed_history))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{GroupCommitConfig, SyncPolicy, WalConfig};

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("z1kv_ckpt_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn open_wal(dir: &Path) -> WalWriter {
        let config = WalConfig {
            wal_dir: dir.join("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
            group_commit: Some(GroupCommitConfig {
                policy: SyncPolicy::SyncEach,
                ..Default::default()
            }),
            write_through: false,
        };
        WalWriter::open(dir, config).unwrap()
    }

    #[test]
    fn checkpoint_roundtrip() {
        let dir = tmp_dir("roundtrip");
        let mgr = CheckpointManager::new(&dir);

        let state = CheckpointState {
            checkpoint_id: 42,
            committed_txn: 42,
            timestamp_ms: 123,
            committed_history: vec![(1, 1), (2, 2), (42, 42)],
        };
        mgr.write(&state).unwrap();

        let loaded = mgr.load_latest().unwrap();
        assert_eq!(loaded.committed_txn, 42);
        assert_eq!(loaded.committed_history, vec![(1, 1), (2, 2), (42, 42)]);
    }

    #[test]
    fn corrupt_checkpoint_falls_back() {
        let dir = tmp_dir("corrupt");
        let mgr = CheckpointManager::new(&dir);

        let state = CheckpointState {
            checkpoint_id: 7,
            committed_txn: 7,
            timestamp_ms: 0,
            committed_history: vec![(7, 7)],
        };
        mgr.write(&state).unwrap();

        // Flip the last payload byte.
        let path = mgr.checkpoint_path(7);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        assert!(
            mgr.load_latest().is_none(),
            "corrupt checkpoint must fall back"
        );
    }

    /// Regression: when the declared payload_len exceeds the actual file
    /// size, load must return None (fall back to WAL replay) instead of
    /// panicking on an out-of-range slice (it used to crash on open).
    #[test]
    fn truncated_len_falls_back_without_panic() {
        let dir = tmp_dir("trunc_len");
        let mgr = CheckpointManager::new(&dir);

        let state = CheckpointState {
            checkpoint_id: 9,
            committed_txn: 9,
            timestamp_ms: 0,
            committed_history: vec![(9, 9)],
        };
        mgr.write(&state).unwrap();

        // Overwrite only the len field (offset 9..13) with a huge value;
        // everything else stays intact.
        let path = mgr.checkpoint_path(9);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        assert!(
            mgr.load_latest().is_none(),
            "oversized len must fall back instead of panicking"
        );
    }

    #[test]
    fn checkpoint_truncates_wal() {
        let dir = tmp_dir("truncate");
        let wal = open_wal(&dir);
        let mgr = CheckpointManager::new(&dir);

        // Write some WAL records.
        for i in 0..5u64 {
            wal.append_durable(
                i,
                WalRecord::Put {
                    cf: 0,
                    key: i.to_le_bytes().to_vec(),
                    value: Some(vec![1]),
                },
            )
            .unwrap();
        }

        // Checkpoint + truncate.
        mgr.checkpoint(&wal, 5, 5, vec![(5, 5)]).unwrap();

        // recovery_baseline should return the checkpoint's committed_txn.
        let (committed_txn, history) = mgr.recovery_baseline().unwrap();
        assert_eq!(committed_txn, 5);
        assert_eq!(history, vec![(5, 5)]);
    }
}
