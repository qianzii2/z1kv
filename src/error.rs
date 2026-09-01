//! Error types for Z1KV.
//!
//! Severity layering with explicit degradation-reporting semantics.
//!
//! # Severity levels
//!
//! - `Fatal`: always returned as an error
//! - `Degrade { metric, recoverable }`: upgraded to Fatal in strict mode,
//!   otherwise the operation degrades and continues
//! - `Warn { metric }`: warning only, does not affect execution
//!
//! Callers inspect `error.severity()` and combine it with `strict_mode` to
//! decide whether to escalate.

use thiserror::Error;

/// Error severity. Tells the caller how to handle an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Fatal error: must be propagated; the caller should abandon the operation.
    Fatal,
    /// Degraded error: escalated to Fatal in strict mode, otherwise a metric
    /// is recorded and execution continues. `metric` is a stable string id
    /// (e.g. `"wal_replay_skip"`) for monitoring systems. `recoverable`
    /// indicates whether the degraded result is still correct.
    Degrade {
        metric: &'static str,
        recoverable: bool,
    },
    /// Warning: logged only, does not affect execution.
    Warn { metric: &'static str },
}

impl Severity {
    /// Whether this is a Fatal error.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Severity::Fatal)
    }

    /// Whether this is a degraded error.
    pub fn is_degrade(&self) -> bool {
        matches!(self, Severity::Degrade { .. })
    }

    /// Resolve the effective behavior under the given `strict_mode`.
    ///
    /// - strict=true + Degrade → escalated to Fatal
    /// - strict=false + Degrade → stays Degrade
    /// - Fatal / Warn → unchanged
    pub fn with_strict_mode(self, strict_mode: bool) -> Severity {
        match (self, strict_mode) {
            (Severity::Degrade { .. }, true) => Severity::Fatal,
            (other, _) => other,
        }
    }

    /// Extract the metric name (for logging / monitoring).
    pub fn metric(&self) -> Option<&'static str> {
        match self {
            Severity::Fatal => None,
            Severity::Degrade { metric, .. } | Severity::Warn { metric } => Some(metric),
        }
    }
}

#[derive(Error, Debug)]
pub enum Z1Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialize(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("MVCC conflict: {0}")]
    MvccConflict(String),

    #[error("Transaction not found: txn_id={0}")]
    TxnNotFound(u64),

    /// Severity-tagged degraded error. Carries a metric + context; callers
    /// use `severity()` to decide whether to escalate.
    #[error("Degraded [{metric}]: {context}")]
    Degraded {
        metric: &'static str,
        context: String,
    },
}

/// Manual `Clone` impl because `std::io::Error` doesn't implement `Clone`.
impl Clone for Z1Error {
    fn clone(&self) -> Self {
        match self {
            Self::Io(e) => Self::Io(std::io::Error::new(e.kind(), e.to_string())),
            Self::Serialization(s) => Self::Serialization(s.clone()),
            Self::Deserialize(s) => Self::Deserialize(s.clone()),
            Self::Internal(s) => Self::Internal(s.clone()),
            Self::Wal(s) => Self::Wal(s.clone()),
            Self::MvccConflict(s) => Self::MvccConflict(s.clone()),
            Self::TxnNotFound(id) => Self::TxnNotFound(*id),
            Self::Degraded { metric, context } => Self::Degraded {
                metric,
                context: context.clone(),
            },
        }
    }
}

impl Z1Error {
    /// Create a degraded error (`Severity::Degrade`).
    pub fn degrade(metric: &'static str, context: impl Into<String>) -> Self {
        Self::Degraded {
            metric,
            context: context.into(),
        }
    }

    /// Return the error's severity. All standard variants are Fatal;
    /// only `Degraded` may be a Degrade (depending on the metric).
    pub fn severity(&self) -> Severity {
        match self {
            Self::Degraded { metric, .. } => Severity::Degrade {
                metric,
                // Default metrics are recoverable=true; unrecoverable metrics
                // must be listed explicitly in `unrecoverable_metrics()`.
                recoverable: !Self::unrecoverable_metrics().contains(metric),
            },
            _ => Severity::Fatal,
        }
    }

    /// Unrecoverable degradation metrics: even with a degradation path, these
    /// must surface as errors in strict mode because degrading would produce
    /// incorrect results.
    ///
    /// KV-semantics list:
    /// - `vis_data_unavailable`: visibility data missing — deleted data could
    ///   become readable
    /// - `wal_record_corrupt`: a corrupt WAL record was skipped — committed
    ///   writes could be lost
    /// - `version_chain_broken`: broken version chain (compaction deleted a
    ///   visible version)
    /// - `gc_snapshot_violation`: GC removed a version that an active
    ///   snapshot may still reference
    fn unrecoverable_metrics() -> &'static [&'static str] {
        &[
            "vis_data_unavailable",
            "wal_record_corrupt",
            "version_chain_broken",
            "gc_snapshot_violation",
        ]
    }
}

pub type Result<T> = std::result::Result<T, Z1Error>;

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_default_is_fatal() {
        let e = Z1Error::Internal("test".into());
        assert_eq!(e.severity(), Severity::Fatal);
    }

    #[test]
    fn degrade_severity() {
        let e = Z1Error::degrade("wal_replay_skip", "old format");
        let s = e.severity();
        assert!(s.is_degrade());
        assert_eq!(s.metric(), Some("wal_replay_skip"));
        assert!(!s.is_fatal());
    }

    #[test]
    fn strict_mode_upgrades_degrade() {
        let e = Z1Error::degrade("wal_replay_skip", "x");
        let s = e.severity();
        // non-strict → Degrade
        assert!(s.with_strict_mode(false).is_degrade());
        // strict → Fatal
        assert!(s.with_strict_mode(true).is_fatal());
    }

    #[test]
    fn unrecoverable_metrics_listed() {
        for m in &["vis_data_unavailable", "wal_record_corrupt"] {
            let e = Z1Error::degrade(m, "x");
            if let Severity::Degrade { recoverable, .. } = e.severity() {
                assert!(!recoverable, "{} should be unrecoverable", m);
            } else {
                panic!("expected Degrade");
            }
        }
    }
}
