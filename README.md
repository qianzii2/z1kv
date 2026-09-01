# Z1KV

An embedded MVCC key-value storage engine (pure library crate, no binary).

- **Versioning**: every version is `(cf, key, txn_id) -> value | tombstone`
- **Visibility**: strict Snapshot Isolation (a txn absent from `commit_ts_map` is invisible — rule D12) + SSI conflict detection
- **Storage**: a three-layer delta stack — L1 memory (ping-pong + RCU reads) → L2 disk patches → L3 frozen (GC-merged)
- **Durability**: a self-implemented WAL (WAL-first; `append_durable` is the single durability boundary) + crash-safe checkpoints
- **Invariants**: D4 / D5 / D7 / D8 / D12 (defined in `src/lib.rs`)

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
z1kv = "0.1"
```

```rust
use z1kv::Z1Kv;

// Open (or create) the engine; a process-level lock guarantees a data
// directory is never opened by two engine instances.
let db = Z1Kv::open("my-data-dir")?;

// Transactional write.
let txn = db.begin_txn()?;
db.put(0, b"greeting", b"hello", txn)?;        // cf, key, value, txn
db.commit(txn)?;

// Read: the current snapshot.
assert_eq!(db.get(0, b"greeting")?, Some(b"hello".to_vec()));

// Transactional read: the snapshot is pinned at begin_txn, repeatable
// reads, participates in SSI conflict detection.
let t1 = db.begin_txn()?;
assert_eq!(db.get_for_txn(0, b"greeting", t1)?, Some(b"hello".to_vec()));
db.commit(t1)?;

// Delete (tombstone).
let t2 = db.begin_txn()?;
db.delete(0, b"greeting", t2)?;
db.commit(t2)?;
assert_eq!(db.get(0, b"greeting")?, None);
```

## Range scan

```rust
// [start, end) half-open interval; end = None means unbounded.
let rows = db.scan(0, b"a", Some(b"z"))?;   // Vec<(Vec<u8>, Vec<u8>)>, sorted by key
```

## Maintenance

```rust
db.flush_now()?;        // unconditional L1 → L2 (db.flush() is threshold-triggered, a no-op on small data)
db.checkpoint()?;       // flush → write checkpoint → truncate WAL (crash-safe order)
let (cfs, reclaimed) = db.compact(u64::MAX)?; // L2 → L3 merge + GC (watermark = oldest active / pinned snapshot)
```

## Key contracts

| Contract | Meaning |
|---|---|
| Durability boundary | `commit()` returning Ok means the WAL is fsynced; committed writes always survive a crash, uncommitted writes are discarded |
| Pinned snapshots | Snapshots from `begin_txn` participate in the GC watermark and stay stable across compactions |
| Bare snapshots | `db.snapshot()` is a time-travel read; its visibility is **not** guaranteed across GC (see the `snapshot` docs) |
| Engine lock | Opening the same data directory twice fails explicitly (`ENGINE.lock`, released when the instance drops) |
| Record limits | A single WAL record is ≤ 64 MB; oversized values are rejected at the `put` boundary and never corrupt the file |
| strict_mode | On by default: recovery aborts on corrupted records instead of silently skipping them |

## Configuration

```rust
use z1kv::config::{Z1Config, VisibilityConfig};
use z1kv::Z1Kv;

let cfg = Z1Config::default()
    .with_checkpoint_wal_size_threshold(64 * 1024 * 1024) // auto-checkpoint when the WAL exceeds this; 0 = disabled
    .with_l2_compaction_threshold(64)                     // auto-compaction beyond this many L2 patches; 0 = disabled
    .with_strict_mode(true)                               // escalate degraded errors to fatal
    .with_visibility(VisibilityConfig::default());        // history eviction (count/TTL)

let db = Z1Kv::open_with_config("my-data-dir", cfg)?;
```

> `Z1Config` and `VisibilityConfig` are `#[non_exhaustive]`: construct them
> via `default()` plus the `with_*` builder methods instead of struct
> literals.

## Architecture

```
            ┌───────────────────────────────┐
   writes   │  Z1Kv (facade)                │
            │  ┌─────────┐    ┌──────────┐  │
   txn ───► │  │ MVCC    │◄──►│ WAL      │  │   D4: WAL-first
            │  └────┬────┘    └────┬─────┘  │
            │       │  L1 MemStore │        │   ping-pong hot/cold + ArcSwap RCU reads
            │       ▼              │        │
            │  ┌─────────┐   ┌─────▼─────┐  │   D8: recent-flush cache bridges the race window
            │  │ L2 disk │──►│ L3 frozen │  │   compaction + GC
            │  └─────────┘   └───────────┘  │
            └───────────────────────────────┘
                     crash-safe checkpoint → truncate WAL
```

## Testing

```sh
cargo test          # unit + integration + README verification + doc tests
cargo test --all-targets --release
```

The test suite includes a WAL byte-level crash matrix (truncation and bit-flip
injection at every boundary), a checkpoint envelope flip matrix, property
tests for GC conservativeness and dual visibility-implementation equivalence
(proptest), fuzz-contract smoke tests, and concurrency stress tests.

## License

Apache-2.0
