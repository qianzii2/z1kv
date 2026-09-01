//! Z1KV — embedded MVCC key-value engine.
//!
//! - MVCC at record level: every version is `(cf, key, txn_id) -> value | tombstone`
//! - Visibility via `VisFilter` rules 1-4 (strict Snapshot Isolation, D12)
//! - Storage: three-layer delta stack (L1 memstore -> L2 disk patches -> L3 frozen)
//! - Durability: self-implemented WAL + crash-safe checkpoints
//! - No distribution, no CDC.
//!
//! # Invariants (correctness core)
//! - D4: WAL is durable BEFORE any L2 write
//! - Truncation boundary: WAL truncate is blocked while the group-commit
//!   queue is non-empty (defensive assert in `WalWriter::truncate_before`)
//! - D12: a txn absent from commit_ts_map is INVISIBLE (GC obeys the same rule)
//! - D5: TTL eviction clocks on `inserted_at`, never on commit_ts
//! - D7: replay watermark (`replay_watermark`) is a lower bound for TTL
//!   eviction — WAL-recovered history is never evicted prematurely
//! - D8: recent-flush cache bridges the L1->L2 race window
//! - Guard routing: patches route by the real key, never a hardcoded default

pub mod codec;
pub mod config;
pub mod durable_writer;
pub mod engine_lock;
pub mod error;
pub mod metrics;
pub mod mvcc;
pub mod store;
pub mod txn;
pub mod wal;

pub use error::{Result, Z1Error};
pub use mvcc::{IsolationLevel, TxnSnapshot, VisFilter, VisibilityManager};
pub use txn::Z1Kv;

/// Transaction ID type — a global transaction identifier.
/// WAL records, version keys `(cf, key, txn_id)` and SSI checks are all
/// expressed in terms of this type.
pub type TxnId = u64;
