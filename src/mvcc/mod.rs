//! MVCC module.
//!
//! Visibility rules operate directly on version records — there are no
//! shadow columns as in column-oriented predecessors.

pub mod cache;
pub mod visibility;

pub use cache::SnapshotCache;
pub use visibility::{IsolationLevel, TxnSnapshot, VisFilter, VisibilityManager};
