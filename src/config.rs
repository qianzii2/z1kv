//! Configuration for Z1KV.

/// Top-level engine configuration.
///
/// `#[non_exhaustive]`: new fields may be added in minor releases without a
/// semver break; external crates cannot use struct literals — build one via
/// [`Z1Config::default()`] plus the `with_*` builder methods (as shown in
/// the README).
#[derive(Clone)]
#[non_exhaustive]
pub struct Z1Config {
    pub visibility: VisibilityConfig,
    /// Strict mode (explicit degradation reporting):
    /// - `true` (default): every `Severity::Degrade` error is upgraded to
    ///   `Fatal`. Intended for production so silent degradations are always
    ///   surfaced as errors.
    /// - `false`: silent-degradation behavior is kept (compatibility with
    ///   older deployments).
    ///
    /// Every code path returning a degraded error is governed by this flag.
    pub strict_mode: bool,
    /// WAL size threshold in bytes: once the WAL exceeds this value after a
    /// commit, a checkpoint is triggered automatically. `0` disables
    /// automatic checkpointing.
    pub checkpoint_wal_size_threshold: u64,
    /// L2 patch-count threshold: once the number of L2 patches exceeds this
    /// value after a flush, L2→L3 compaction is triggered automatically.
    /// `0` disables automatic compaction.
    pub l2_compaction_threshold: usize,
}

impl Default for Z1Config {
    fn default() -> Self {
        Self {
            visibility: VisibilityConfig::default(),
            // strict_mode defaults to on: every degradation escalates to Fatal.
            strict_mode: true,
            checkpoint_wal_size_threshold: 64 * 1024 * 1024,
            l2_compaction_threshold: 64,
        }
    }
}

impl Z1Config {
    /// Set `strict_mode` (builder style; the struct is `#[non_exhaustive]`
    /// so external crates cannot use struct literals).
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Set `checkpoint_wal_size_threshold` in bytes (builder style).
    /// `0` disables automatic checkpointing.
    pub fn with_checkpoint_wal_size_threshold(mut self, bytes: u64) -> Self {
        self.checkpoint_wal_size_threshold = bytes;
        self
    }

    /// Set `l2_compaction_threshold` (builder style).
    /// `0` disables automatic compaction.
    pub fn with_l2_compaction_threshold(mut self, n: usize) -> Self {
        self.l2_compaction_threshold = n;
        self
    }

    /// Replace the `VisibilityConfig` (builder style).
    pub fn with_visibility(mut self, visibility: VisibilityConfig) -> Self {
        self.visibility = visibility;
        self
    }
}

/// MVCC visibility manager configuration.
///
/// Note: `Serialize`/`Deserialize` derives are deliberately absent — the
/// engine never reads a config file, so this struct is never serialized.
/// If config-file support is added later, restore the derives together with
/// `deny_unknown_fields` (a misspelled config key should fail loudly, not be
/// silently ignored).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VisibilityConfig {
    /// Maximum number of entries to retain in the committed history.
    /// When exceeded, oldest 20% of entries are evicted.
    ///
    /// # Semantic boundary
    ///
    /// `0` is a legal value but with an extreme meaning: the committed
    /// history is emptied immediately, and under rule D12 **all** historical
    /// versions become invisible (every read behaves as if the key does not
    /// exist). There is no config validation layer (the engine does not read
    /// config files), so this semantics is stated explicitly here.
    pub max_history_entries: usize,
    /// Time-to-live for committed history entries, in seconds.
    /// Entries older than (now - ttl) are evicted on each commit.
    pub history_ttl_secs: u64,
}

impl Default for VisibilityConfig {
    fn default() -> Self {
        Self {
            max_history_entries: 100_000,
            history_ttl_secs: 3600,
        }
    }
}

impl VisibilityConfig {
    /// Set `max_history_entries` (builder style; the struct is
    /// `#[non_exhaustive]` so external crates cannot use struct literals).
    pub fn with_max_history_entries(mut self, n: usize) -> Self {
        self.max_history_entries = n;
        self
    }

    /// Set `history_ttl_secs` (builder style).
    pub fn with_history_ttl_secs(mut self, secs: u64) -> Self {
        self.history_ttl_secs = secs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the explicit semantics of `max_history_entries = 0` —
    /// history is emptied immediately, and under D12 all old versions become
    /// invisible. This test locks in that documented extreme semantics.
    #[test]
    fn zero_history_entries_blinds_old_versions() {
        let cfg = VisibilityConfig::default()
            .with_max_history_entries(0)
            .with_history_ttl_secs(3600);
        let mut mgr = crate::mvcc::VisibilityManager::new_with_config(cfg);

        // Commit path: register → commit → prune (threshold 0) empties history.
        mgr.begin_txn(1).unwrap();
        mgr.commit_txn(1, 0).unwrap();
        assert_eq!(mgr.committed_entry(1), None, "history pruned immediately");

        // D12: missing history → old versions are invisible.
        let snap = mgr.snapshot(crate::mvcc::IsolationLevel::Snapshot);
        assert!(!mgr.is_visible(&snap, 1, None));
    }
}
