//! Storage kernel for Z1KV — three-layer versioned store.
//!
//! - L1 memstore: in-memory ping-pong buffer with ArcSwap RCU reads
//! - L2 disk: guard-indexed patch files on disk
//! - L3 frozen: compacted immutable patches
//!
//! Race-protection invariants (recent_flush_cache / flush_epoch / D4
//! WAL-first) are enforced across all three layers.

pub mod config;
pub mod disk;
pub mod flush;
pub mod gc;
pub mod mem;
pub mod recent_flush_cache;
pub mod types;

pub use config::SyncLevel;
pub use disk::{DiskLayer, L2Disk, L3Frozen};
pub use flush::FlushEngine;
pub use gc::{gc_entries, GcStats};
pub use mem::MemStore;
pub use recent_flush_cache::RecentFlushCache;
pub use types::{
    PatchEntry, PatchZoneMap, StoreCheckpointState, Z1Entry, Z1Key, Z1Patch, Z1PatchFormatV4,
};
