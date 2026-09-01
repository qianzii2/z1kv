//! WAL recovery — error classification and replay.
//!
//! Four-level error classification (Recoverable / Ambiguous / Corruption /
//! Fatal) with a `replay_failure` side log; the payload type is `WalRecord`.
//!
//! # Recovery phases
//!
//! 1. SCAN: `wal::replay_all` reads all WAL segments
//! 2. BUILD: group by txn_id, filter committed transactions
//! 3. REPLAY: for each committed txn, call the `apply()` callback in order
//!
//! # Invariants
//!
//! - Uncommitted transactions (no Commit record) are discarded as aborted
//! - A corrupted txn is skipped and logged to `replay_failure` without
//!   interrupting other transactions
//! - Fatal (WAL reader I/O error) aborts immediately

use crate::error::{Result, Z1Error};
use crate::wal::{replay_all, WalEntry, WalRecord};
use crate::TxnId;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Severity of a WAL replay error.
#[derive(Debug, Clone)]
pub enum ReplayErrorKind {
    /// IO error that is safe to skip — data already durable.
    Recoverable {
        reason: RecoverableReason,
        detail: String,
    },
    /// Payload format mismatch — old format, safe to skip.
    Ambiguous { detail: String },
    /// Record corrupt / unparseable — write to replay_failure, skip txn.
    Corruption {
        location: CorruptionLocation,
        detail: String,
    },
    /// Fatal I/O error from the WAL reader — cannot continue.
    Fatal { io_kind: String, detail: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoverableReason {
    AlreadyExists,
    PermissionDenied,
    Unknown,
}

impl std::fmt::Display for RecoverableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoverableReason::AlreadyExists => write!(f, "AlreadyExists"),
            RecoverableReason::PermissionDenied => write!(f, "PermissionDenied"),
            RecoverableReason::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionLocation {
    Deserialize,
    Unknown,
}

impl std::fmt::Display for CorruptionLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorruptionLocation::Deserialize => write!(f, "Deserialize"),
            CorruptionLocation::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for ReplayErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayErrorKind::Recoverable { reason, detail } => {
                write!(f, "recoverable ({:?}): {}", reason, detail)
            }
            ReplayErrorKind::Ambiguous { detail } => write!(f, "ambiguous: {}", detail),
            ReplayErrorKind::Corruption { location, detail } => {
                write!(f, "corruption ({:?}): {}", location, detail)
            }
            ReplayErrorKind::Fatal { io_kind, detail } => {
                write!(f, "fatal ({}): {}", io_kind, detail)
            }
        }
    }
}

/// WAL replay error — carries a classified kind and details.
#[derive(Debug)]
pub struct ReplayError {
    pub kind: ReplayErrorKind,
    pub txn_id: TxnId,
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WAL replay error ({}): {}", self.kind, self.txn_id)
    }
}

impl std::error::Error for ReplayError {}

/// WAL recovery result.
#[derive(Debug, Default)]
pub struct RecoveryResult {
    /// Last committed transaction ID that was both committed and durably replayed.
    pub last_committed_txn: TxnId,
    /// Highest committed transaction ID observed in the WAL, even if replay degraded.
    pub max_seen_committed_txn: TxnId,
    /// Number of committed transactions replayed.
    pub committed_count: usize,
    /// Number of uncommitted transactions discarded.
    pub discarded_count: usize,
    /// Commit timestamp map: txn_id -> commit_ts.
    pub commit_ts_map: HashMap<TxnId, u64>,
    /// D5: WAL-carried inserted_at (the TTL clock): txn_id -> inserted_at.
    /// Recovered history entries share the same eviction clock as online commits.
    pub inserted_at_map: HashMap<TxnId, u64>,
    /// D7: replay_watermark = the largest inserted_at recovered from the WAL.
    /// Serves as a lower bound for TTL eviction so recovered history entries
    /// are never evicted prematurely.
    pub replay_watermark: u64,
    /// Committed Put entries replayed (for L1 rebuild).
    pub committed_puts: Vec<(TxnId, WalRecord)>,
}

/// Classify an apply error (simplified to KV semantics).
fn classify_apply_error(err: &Z1Error, _txn_id: TxnId) -> ReplayErrorKind {
    let msg = err.to_string();
    match err {
        Z1Error::Deserialize(_) => ReplayErrorKind::Corruption {
            location: CorruptionLocation::Deserialize,
            detail: msg,
        },
        Z1Error::Io(io_err) => match io_err.kind() {
            std::io::ErrorKind::AlreadyExists => ReplayErrorKind::Recoverable {
                reason: RecoverableReason::AlreadyExists,
                detail: msg,
            },
            std::io::ErrorKind::PermissionDenied => ReplayErrorKind::Recoverable {
                reason: RecoverableReason::PermissionDenied,
                detail: msg,
            },
            _ => ReplayErrorKind::Fatal {
                io_kind: format!("{:?}", io_err.kind()),
                detail: msg,
            },
        },
        _ => ReplayErrorKind::Ambiguous { detail: msg },
    }
}

/// Replay all committed WAL operations.
///
/// Groups records by txn_id, filters committed transactions, calls `apply`
/// for each committed op, and classifies errors.
///
/// `strict_mode`: with strict=true, Corruption (an unrecoverable degradation
/// that would silently skip possibly-committed data) escalates to Fatal and
/// aborts recovery — exactly the `handle_degrade` contract: silent
/// degradations must be surfaced in strict mode. With strict=false, the
/// corrupted txn is skipped and logged to replay_failure (best-effort
/// recovery).
///
/// `Err(ReplayError)` is returned for Fatal errors and for Corruption under
/// strict mode; other classifications are logged and the txn skipped.
pub fn replay_committed_ops(
    wal_dir: &Path,
    strict_mode: bool,
    mut apply: impl FnMut(&WalEntry) -> Result<()>,
) -> std::result::Result<RecoveryResult, ReplayError> {
    let records = replay_all(wal_dir).map_err(|e| ReplayError {
        kind: ReplayErrorKind::Fatal {
            io_kind: "WAL reader".to_string(),
            detail: e.to_string(),
        },
        txn_id: 0,
    })?;

    // Phase 1: group Put/Begin by txn_id, track Commit/Rollback.
    let mut pending: HashMap<TxnId, Vec<WalEntry>> = HashMap::new();
    let mut committed_txns: HashMap<TxnId, u64> = HashMap::new(); // txn_id -> commit_ts
    let mut rolled_back: HashSet<TxnId> = HashSet::new();
    let mut max_seen_committed: TxnId = 0;
    // D5: WAL-carried inserted_at (the TTL clock).
    let mut inserted_at_map: HashMap<TxnId, u64> = HashMap::new();
    // D7: replay_watermark = the largest recovered inserted_at.
    let mut max_inserted_at: u64 = 0;

    for entry in &records {
        match &entry.record {
            WalRecord::Put { .. } => {
                pending.entry(entry.txn_id).or_default().push(entry.clone());
            }
            WalRecord::Begin => {
                pending.entry(entry.txn_id).or_default();
            }
            WalRecord::Commit {
                commit_ts,
                inserted_at,
            } => {
                committed_txns.insert(entry.txn_id, *commit_ts);
                // D5: WAL-carried inserted_at (the TTL clock).
                inserted_at_map.insert(entry.txn_id, *inserted_at);
                // D7: replay_watermark = the largest recovered inserted_at,
                // a lower bound for TTL eviction (history snapshots may
                // still reference these entries).
                max_inserted_at = max_inserted_at.max(*inserted_at);
                max_seen_committed = max_seen_committed.max(entry.txn_id);
            }
            WalRecord::Rollback => {
                rolled_back.insert(entry.txn_id);
                pending.remove(&entry.txn_id);
            }
            WalRecord::Checkpoint { .. } => {
                // marker only — no data effect
            }
        }
    }

    let mut result = RecoveryResult {
        max_seen_committed_txn: max_seen_committed,
        replay_watermark: max_inserted_at,
        inserted_at_map,
        ..Default::default()
    };

    // Discarded = txns with pending writes but no Commit and no Rollback.
    let committed_ids: HashSet<TxnId> = committed_txns.keys().copied().collect();
    result.discarded_count = pending
        .iter()
        .filter(|(id, _)| !committed_ids.contains(id) && !rolled_back.contains(id))
        .count();

    // Phase 2: replay committed txns in order.
    let mut ordered_txns: Vec<TxnId> = committed_txns.keys().copied().collect();
    ordered_txns.sort_unstable();

    for txn_id in ordered_txns {
        let commit_ts = committed_txns[&txn_id];
        let Some(ops) = pending.remove(&txn_id) else {
            continue;
        };

        let mut txn_failed = false;
        for op in &ops {
            if let Err(e) = apply(op) {
                let kind = classify_apply_error(&e, txn_id);
                match &kind {
                    ReplayErrorKind::Fatal { io_kind, detail } => {
                        tracing::error!(
                            "WAL replay fatal error for txn {}: {} ({}). Aborting recovery.",
                            txn_id,
                            detail,
                            io_kind
                        );
                        return Err(ReplayError { kind, txn_id });
                    }
                    ReplayErrorKind::Corruption { location, detail } => {
                        // Under strict_mode, Corruption is an unrecoverable
                        // degradation (it would silently skip possibly-
                        // committed data) — recovery must abort rather than
                        // skip. Without strict mode, record the degradation
                        // and skip.
                        let degrade = Z1Error::degrade("wal_record_corrupt", detail.clone());
                        if crate::metrics::handle_degrade(&degrade, strict_mode).is_break() {
                            tracing::error!(
                                "WAL replay corruption for txn {} at {:?}: {}. \
                                 strict_mode aborting recovery.",
                                txn_id,
                                location,
                                detail
                            );
                            return Err(ReplayError { kind, txn_id });
                        }
                        tracing::error!(
                            "WAL replay corruption detected for txn {} at {:?}: {}. \
                             Writing to replay_failure and skipping.",
                            txn_id,
                            location,
                            detail
                        );
                        if let Err(io_err) = write_replay_failure(wal_dir, txn_id, detail) {
                            tracing::error!(
                                "failed to write replay_failure record for txn {}: {}",
                                txn_id,
                                io_err
                            );
                        }
                    }
                    ReplayErrorKind::Recoverable { reason, detail } => {
                        tracing::warn!(
                            "WAL replay recoverable error for txn {} ({:?}): {}. Skipping.",
                            txn_id,
                            reason,
                            detail
                        );
                    }
                    ReplayErrorKind::Ambiguous { detail } => {
                        tracing::warn!(
                            "WAL replay ambiguous error for txn {}: {}. Skipping.",
                            txn_id,
                            detail
                        );
                    }
                }
                txn_failed = true;
                break;
            }
        }

        if txn_failed {
            continue;
        }

        result.last_committed_txn = result.last_committed_txn.max(txn_id);
        result.committed_count += 1;
        result.commit_ts_map.insert(txn_id, commit_ts);
        // Collect committed Puts for L1 rebuild.
        for op in &ops {
            if matches!(op.record, WalRecord::Put { .. }) {
                result.committed_puts.push((txn_id, op.record.clone()));
            }
        }
    }

    Ok(result)
}

/// Write a corruption record to the `replay_failure` file (JSONL).
fn write_replay_failure(wal_dir: &Path, txn_id: TxnId, reason: &str) -> std::io::Result<()> {
    use std::io::Write;

    #[derive(serde::Serialize)]
    struct ReplayFailureRecord {
        txn_id: u64,
        reason: String,
        timestamp: u64,
    }

    let failure_path = wal_dir.join("replay_failure");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(failure_path)?;
    let mut writer = std::io::BufWriter::new(file);

    let record = ReplayFailureRecord {
        txn_id,
        reason: reason.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    serde_json::to_writer_pretty(&mut writer, &record)?;
    writeln!(writer)?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "z1kv_recovery_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_wal(dir: &Path, entries: &[WalEntry]) {
        let config = crate::wal::WalConfig {
            wal_dir: dir.join("wal"),
            enabled: true,
            group_commit: Some(crate::wal::GroupCommitConfig {
                policy: crate::wal::SyncPolicy::SyncEach,
                ..Default::default()
            }),
            ..Default::default()
        };
        let wal = crate::wal::WalWriter::open(dir, config).unwrap();
        for e in entries {
            wal.append_durable(e.txn_id, e.record.clone()).unwrap();
        }
    }

    #[test]
    fn replay_committed_txn() {
        let dir = tmp_dir("committed");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Commit {
                        commit_ts: 1,
                        inserted_at: 1,
                    },
                },
            ],
        );

        let mut puts = Vec::new();
        let result = replay_committed_ops(&dir.join("wal"), false, |op| {
            if let WalRecord::Put { .. } = &op.record {
                puts.push(op.txn_id);
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(result.committed_count, 1);
        assert_eq!(result.last_committed_txn, 1);
        assert_eq!(result.max_seen_committed_txn, 1);
        assert_eq!(result.commit_ts_map.get(&1), Some(&1));
        assert_eq!(puts, vec![1]);
    }

    #[test]
    fn uncommitted_txn_is_discarded() {
        let dir = tmp_dir("uncommitted");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 5,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 5,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                // no Commit
            ],
        );

        let result = replay_committed_ops(&dir.join("wal"), false, |_op| Ok(())).unwrap();
        assert_eq!(result.last_committed_txn, 0);
        assert_eq!(result.committed_count, 0);
        assert_eq!(result.discarded_count, 1);
    }

    #[test]
    fn rollback_removes_pending() {
        let dir = tmp_dir("rollback");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 3,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 3,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 3,
                    record: WalRecord::Rollback,
                },
            ],
        );

        let result = replay_committed_ops(&dir.join("wal"), false, |_op| Ok(())).unwrap();
        assert_eq!(result.committed_count, 0);
        assert_eq!(result.discarded_count, 0);
    }

    #[test]
    fn corruption_records_failure_and_continues() {
        let dir = tmp_dir("corruption");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Commit {
                        commit_ts: 7,
                        inserted_at: 7,
                    },
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k2".to_vec(),
                        value: Some(b"v2".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Commit {
                        commit_ts: 8,
                        inserted_at: 8,
                    },
                },
            ],
        );

        // txn 7's apply fails with Deserialize (corruption); txn 8 succeeds.
        let result = replay_committed_ops(&dir.join("wal"), false, |op| {
            if op.txn_id == 7 {
                Err(Z1Error::Deserialize("synthetic corruption".into()))
            } else {
                Ok(())
            }
        })
        .unwrap();

        assert_eq!(result.last_committed_txn, 8);
        assert_eq!(result.max_seen_committed_txn, 8);
        assert_eq!(result.committed_count, 1);
        assert!(result.commit_ts_map.contains_key(&8));

        // replay_failure records txn 7.
        let content = std::fs::read_to_string(dir.join("wal").join("replay_failure")).unwrap();
        assert!(content.contains("\"txn_id\": 7"));
        assert!(content.contains("synthetic corruption"));
    }

    /// With strict_mode=true, Corruption escalates to Fatal and aborts recovery.
    #[test]
    fn strict_mode_aborts_on_corruption() {
        let dir = tmp_dir("strict");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 7,
                    record: WalRecord::Commit {
                        commit_ts: 7,
                        inserted_at: 7,
                    },
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k2".to_vec(),
                        value: Some(b"v2".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 8,
                    record: WalRecord::Commit {
                        commit_ts: 8,
                        inserted_at: 8,
                    },
                },
            ],
        );

        let result = replay_committed_ops(&dir.join("wal"), true, |op| {
            if op.txn_id == 7 {
                Err(Z1Error::Deserialize("synthetic corruption".into()))
            } else {
                Ok(())
            }
        });

        // strict mode: corruption must abort (Err), never be silently skipped.
        assert!(result.is_err(), "strict mode must abort on corruption");
        let err = result.unwrap_err();
        assert!(
            matches!(err.kind, ReplayErrorKind::Corruption { .. }),
            "expected Corruption kind, got {:?}",
            err.kind
        );
    }

    /// With strict_mode, recovery is completely normal when data is intact
    /// (strict mode does not affect the healthy path).
    #[test]
    fn strict_mode_normal_recovery_unchanged() {
        let dir = tmp_dir("strict_ok");
        write_wal(
            &dir,
            &[
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Begin,
                },
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Put {
                        cf: 0,
                        key: b"k".to_vec(),
                        value: Some(b"v".to_vec()),
                    },
                },
                WalEntry {
                    txn_id: 1,
                    record: WalRecord::Commit {
                        commit_ts: 1,
                        inserted_at: 1,
                    },
                },
            ],
        );

        let result = replay_committed_ops(&dir.join("wal"), true, |_op| Ok(())).unwrap();
        assert_eq!(result.committed_count, 1);
        assert_eq!(result.replay_watermark, 1, "D7 watermark from inserted_at");
    }
}
