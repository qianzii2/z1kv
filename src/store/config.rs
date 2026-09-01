//! SyncLevel — WAL synchronization policy for L1 writes.

/// Synchronization level for L1 writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncLevel {
    /// Every write is fsync'd immediately. Zero durability risk, lower throughput.
    Immediate,
    /// Group commit: flush after N ms or N pending entries, whichever first.
    Batch { ms: u64, max_pending: usize },
    /// Fully async: write to memory only, rely on background sync.
    Async,
}
