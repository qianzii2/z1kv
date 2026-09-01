//! CommitProtocol — commit atomicity protocol.
//!
//! # Protocol
//!
//! ```text
//! 1. prepare  : pre-record side effects as in-memory PendingOps (no disk I/O)
//! 2. commit   : WAL.append_durable(Commit) → fsync → DURABILITY BOUNDARY
//! 3. apply    : apply PendingOps to MVCC in order
//! 4. finalize : snapshot_cache.invalidate()
//! ```
//!
//! # Invariants
//!
//! - There is exactly one DURABILITY BOUNDARY: the WAL `append_durable`.
//! - MVCC mutations in the apply phase are **never** rolled back (they are
//!   coupled to the persisted Commit record).
//! - The MVCC commit is intentionally non-compensable: once the WAL Commit
//!   record is durable, recovery treats the transaction as committed.

use crate::error::{Result, Z1Error};
use crate::TxnId;

/// All side effects pending for a single commit.
#[derive(Debug, Clone)]
pub struct CommitProtocol {
    pub txn_id: TxnId,
    pub pending_ops: Vec<PendingOp>,
}

/// A side effect waiting to be applied (nothing is written to disk here).
///
/// Key design: every side effect performed during a commit is expressed as a
/// `PendingOp`, which makes `apply_all` the single execution point — easy to
/// audit.
#[derive(Debug, Clone)]
pub enum PendingOp {
    /// MVCC commit: remove the txn from active_txns and record it in
    /// committed_history.
    MvccCommit { inserted_at: u64 },
    /// Persist the committed_txn counter (carried by the WAL Commit record
    /// in Z1KV; the applier treats this as a no-op).
    PutCommittedTxn { counter: u64 },
    /// Persist the commit timestamp (also carried by the WAL Commit record).
    CommitTxnRecord { txn_id: u64, commit_ts: u64 },
    /// Snapshot cache invalidation.
    SnapshotInvalidate,
}

impl CommitProtocol {
    pub fn new(txn_id: TxnId) -> Self {
        Self {
            txn_id,
            pending_ops: Vec::new(),
        }
    }

    /// Check protocol sanity: `pending_ops` must not be empty.
    pub fn validate(&self) -> Result<()> {
        if self.pending_ops.is_empty() {
            return Err(Z1Error::Internal(
                "CommitProtocol: pending_ops is empty, commit_txn has no work to do".into(),
            ));
        }
        Ok(())
    }

    /// Apply the pending ops in order; abort on the first failure.
    ///
    /// Key invariant: the WAL `append_durable` (the only DURABILITY
    /// BOUNDARY) must already have executed before calling this.
    pub fn apply_all<A: CommitApplier>(&self, applier: &A) -> Result<()> {
        self.validate()?;
        for op in &self.pending_ops {
            match op {
                PendingOp::MvccCommit { inserted_at } => {
                    applier.apply_mvcc_commit(self.txn_id, *inserted_at)?;
                }
                PendingOp::PutCommittedTxn { counter } => {
                    applier.apply_put_committed_txn(*counter)?;
                }
                PendingOp::CommitTxnRecord { txn_id, commit_ts } => {
                    applier.apply_commit_txn_record(*txn_id, *commit_ts)?;
                }
                PendingOp::SnapshotInvalidate => {
                    applier.apply_snapshot_invalidate()?;
                }
            }
        }
        Ok(())
    }
}

/// Trait implemented by the engine to apply `PendingOp`s.
pub trait CommitApplier {
    fn apply_mvcc_commit(&self, txn_id: TxnId, inserted_at: u64) -> Result<()>;
    fn apply_put_committed_txn(&self, counter: u64) -> Result<()>;
    fn apply_commit_txn_record(&self, txn_id: u64, commit_ts: u64) -> Result<()>;
    fn apply_snapshot_invalidate(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct MockApplier {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl CommitApplier for MockApplier {
        fn apply_mvcc_commit(&self, _txn: TxnId, _ts: u64) -> Result<()> {
            self.log.lock().unwrap().push("mvcc".into());
            Ok(())
        }
        fn apply_put_committed_txn(&self, c: u64) -> Result<()> {
            self.log.lock().unwrap().push(format!("pct:{}", c));
            Ok(())
        }
        fn apply_commit_txn_record(&self, t: u64, c: u64) -> Result<()> {
            self.log.lock().unwrap().push(format!("ctr:{}:{}", t, c));
            Ok(())
        }
        fn apply_snapshot_invalidate(&self) -> Result<()> {
            self.log.lock().unwrap().push("snap".into());
            Ok(())
        }
    }

    #[test]
    fn apply_all_runs_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let applier = MockApplier { log: log.clone() };
        let mut p = CommitProtocol::new(42);
        p.pending_ops.push(PendingOp::MvccCommit { inserted_at: 1 });
        p.pending_ops
            .push(PendingOp::PutCommittedTxn { counter: 42 });
        p.pending_ops.push(PendingOp::CommitTxnRecord {
            txn_id: 42,
            commit_ts: 42,
        });
        p.pending_ops.push(PendingOp::SnapshotInvalidate);
        p.apply_all(&applier).unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            vec!["mvcc", "pct:42", "ctr:42:42", "snap"]
        );
    }

    #[test]
    fn empty_protocol_rejected() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let applier = MockApplier { log };
        let p = CommitProtocol::new(1);
        assert!(p.apply_all(&applier).is_err());
    }
}
