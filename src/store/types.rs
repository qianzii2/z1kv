//! Core types for the Z1KV versioned storage kernel.
//!
//! - `Z1Entry`: record-level multi-versioning, `(cf, key, txn_id) -> value | tombstone`
//! - `Z1PatchFormatV4`: sparse patches — a sorted `(key, tombstone|value)` sequence
//! - `PatchZoneMap`: txn-range pruning statistics with key bounds
//! - The `DiskFormat` framework is reused; MAGIC = `DLT004`,
//!   `MIN_READABLE_VERSION = 4` (a fresh project, no legacy data)
//!
//! # File Format
//!
//! Each `.zpatch` file payload starts with the DiskFormat 18-byte header
//! (`DLT004` + version + crc32 + payload_len) followed by the postcard-encoded
//! `Z1PatchFormatV4`.
//!
//! # Easy-to-confuse points
//!
//! - `Z1Entry` has no per-cell before/after images: the version chain *is*
//!   the history; readers pick the visible version by txn_id, and tombstones
//!   express deletion.
//! - `Z1Key` total order = `cf_be(2B) || key_bytes`, giving a globally
//!   consistent ordering across column families.

use crate::codec::disk_format::DiskFormat;
use crate::TxnId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// File magic bytes for `.zpatch` files (DiskFormat-protected).
const PATCH_V4_MAGIC: &[u8; 6] = b"DLT004";

/// A versioned key: column-family id + user key bytes.
///
/// Ordering: `(cf, key)` lexicographic — cf as big-endian u16 so that
/// prefix scans and guard splitting behave uniformly across CFs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Z1Key {
    pub cf: u16,
    pub key: Vec<u8>,
}

impl Z1Key {
    pub fn new(cf: u16, key: impl Into<Vec<u8>>) -> Self {
        Self {
            cf,
            key: key.into(),
        }
    }
}

/// The atomic unit of versioned storage: one version of one key.
///
/// - No `before` image: the version chain is the history.
/// - `value: None` = tombstone (delete marker).
/// - Authoritative visibility must come from commit_ts_map + active txn
///   state (D12). The migrated `committed` transitional flag was verified
///   as never read (`Z1Entry` is never serialized; the on-disk format is
///   `PatchEntry`) and has been removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Z1Entry {
    pub key: Z1Key,
    pub txn_id: TxnId,
    /// `None` = tombstone (delete marker); `Some(bytes)` = value.
    pub value: Option<Arc<Vec<u8>>>,
    /// Wall-clock timestamp (for time travel), milliseconds since epoch.
    pub ts: i64,
}

impl Z1Entry {
    pub fn tombstone(key: Z1Key, txn_id: TxnId, ts: i64) -> Self {
        Self {
            key,
            txn_id,
            value: None,
            ts,
        }
    }

    pub fn put(key: Z1Key, txn_id: TxnId, value: impl Into<Vec<u8>>, ts: i64) -> Self {
        Self {
            key,
            txn_id,
            value: Some(Arc::new(value.into())),
            ts,
        }
    }

    pub fn is_tombstone(&self) -> bool {
        self.value.is_none()
    }

    /// Returns true if this version is semantically committed for a snapshot
    /// using authoritative MVCC commit timestamps.
    /// Absent commit history means invisible (D12 strict rule).
    pub fn is_visible_at_commit(
        &self,
        snapshot_txn: TxnId,
        commit_ts_by_txn: &std::collections::HashMap<TxnId, u64>,
        active_txns: &std::collections::HashSet<TxnId>,
    ) -> bool {
        if self.txn_id > snapshot_txn {
            return false;
        }
        if active_txns.contains(&self.txn_id) {
            return false;
        }
        commit_ts_by_txn
            .get(&self.txn_id)
            .copied()
            .map(|commit_ts| commit_ts <= snapshot_txn)
            .unwrap_or(false)
    }
}

/// ZoneMap statistics for a patch — used for txn-range pruning.
/// The min/max value bounds of the predecessor design became key bounds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchZoneMap {
    /// Minimum transaction ID in this patch.
    pub min_txn: TxnId,
    /// Maximum transaction ID in this patch.
    pub max_txn: TxnId,
    /// Number of entries in this patch.
    pub entry_count: u64,
    /// Smallest key in this patch (optional, for key-range pruning).
    pub min_key: Option<Vec<u8>>,
    /// Largest key in this patch (optional, for key-range pruning).
    pub max_key: Option<Vec<u8>>,
}

/// The format of a persistent patch (V4).
///
/// - Sparse: a roaring bitmap makes no sense for arbitrary byte keys, so the
///   payload is a sorted `(key, tombstone|value)` sequence.
/// - Dense: removed — KV patches are naturally sparse, the Dense use case
///   no longer exists.
///
/// Serialization via `DiskFormat` (`DLT004` magic + crc32 protection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Z1PatchFormatV4 {
    /// Sorted `(key, tombstone|value)` entries.
    Sparse { entries: Vec<PatchEntry> },
}

/// One entry inside a sparse patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchEntry {
    pub key: Vec<u8>,
    /// `None` = tombstone.
    pub value: Option<Arc<Vec<u8>>>,
    /// Transaction ID that produced this version.
    pub txn_id: TxnId,
}

impl Z1PatchFormatV4 {
    /// Build a sparse patch from entries. Entries are sorted by key for
    /// deterministic on-disk layout and binary-searchable reads.
    pub fn sparse(mut entries: Vec<PatchEntry>) -> Self {
        entries.sort_by(|a, b| a.key.cmp(&b.key).then(a.txn_id.cmp(&b.txn_id)));
        Self::Sparse { entries }
    }

    /// Returns the number of entries.
    pub fn entry_count(&self) -> usize {
        match self {
            Self::Sparse { entries } => entries.len(),
        }
    }

    /// Returns the highest txn_id among all entries.
    pub fn max_txn_id(&self) -> Option<TxnId> {
        match self {
            Self::Sparse { entries } => entries.iter().map(|e| e.txn_id).max(),
        }
    }

    /// Returns the lowest txn_id among all entries.
    pub fn min_txn_id(&self) -> Option<TxnId> {
        match self {
            Self::Sparse { entries } => entries.iter().map(|e| e.txn_id).min(),
        }
    }

    /// Returns the smallest and largest keys.
    pub fn key_bounds(&self) -> (Option<&[u8]>, Option<&[u8]>) {
        match self {
            Self::Sparse { entries } => (
                entries.first().map(|e| e.key.as_slice()),
                entries.last().map(|e| e.key.as_slice()),
            ),
        }
    }
}

/// DiskFormat impl for the V4 patch payload.
///
/// Intentionally incompatible with any predecessor v1 format — this project
/// starts at V4 with `MIN_READABLE_VERSION = 4` (no legacy baggage).
impl DiskFormat for Z1PatchFormatV4 {
    const MAGIC: &'static [u8; 6] = PATCH_V4_MAGIC;
    const VERSION: u32 = 4;
    const MIN_READABLE_VERSION: u32 = 4;
}

/// A persistent patch — the unit stored in L2 and L3.
/// `patch_id` stays monotonically increasing across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Z1Patch {
    pub patch_id: u64,
    /// Transaction range covered by this patch.
    pub txn_range: (TxnId, TxnId),
    pub format: Z1PatchFormatV4,
    pub zone_map: PatchZoneMap,
}

/// Storage checkpoint state for durability / recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreCheckpointState {
    pub committed_txn: TxnId,
    pub l1_entry_count: usize,
    pub l1_size_bytes: u64,
    pub l2_patch_counts: u64,
    pub l3_patch_counts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    fn key(cf: u16, k: &[u8]) -> Z1Key {
        Z1Key::new(cf, k)
    }

    // ── Visibility tests (semantics carried over) ─────────────────────

    fn visible_entry(txn_id: TxnId) -> Z1Entry {
        Z1Entry {
            key: key(0, b"k"),
            txn_id,
            value: Some(Arc::new(vec![txn_id as u8])),
            ts: 0,
        }
    }

    #[test]
    fn visibility_requires_commit_history_not_legacy_flag() {
        let entry = visible_entry(42);
        let active_txns = std::collections::HashSet::new();

        assert!(!entry.is_visible_at_commit(100, &StdHashMap::new(), &active_txns));

        let mut commit_ts_by_txn = StdHashMap::new();
        commit_ts_by_txn.insert(42, 40);
        assert!(entry.is_visible_at_commit(100, &commit_ts_by_txn, &active_txns));
    }

    // ── V4 tests ────────────────────────────────────────────────────────

    #[test]
    fn tombstone_semantics() {
        let k = key(1, b"a");
        let t = Z1Entry::tombstone(k.clone(), 7, 0);
        let p = Z1Entry::put(k, 8, b"v", 0);
        assert!(t.is_tombstone());
        assert!(!p.is_tombstone());
        assert_eq!(p.value.as_deref().map(|v| v.as_slice()), Some(&b"v"[..]));
    }

    #[test]
    fn z1key_ordering_cf_then_key() {
        assert!(key(0, b"z") < key(1, b"a"), "cf dominates key order");
        assert!(key(1, b"a") < key(1, b"b"));
        assert!(
            key(1, b"ab") < key(1, b"b"),
            "prefix sorts before longer key"
        );
    }

    #[test]
    fn patch_v4_disk_format_roundtrip() {
        let patch = Z1PatchFormatV4::sparse(vec![
            PatchEntry {
                key: b"k2".to_vec(),
                value: Some(Arc::new(vec![2])),
                txn_id: 2,
            },
            PatchEntry {
                key: b"k1".to_vec(),
                value: None,
                txn_id: 1,
            },
        ]);

        let bytes = patch.to_disk_bytes().unwrap();
        let parsed = Z1PatchFormatV4::from_disk_bytes(&bytes).unwrap();

        match parsed {
            Z1PatchFormatV4::Sparse { entries } => {
                assert_eq!(entries.len(), 2);
                // sorted by key
                assert_eq!(entries[0].key, b"k1");
                assert!(entries[0].value.is_none(), "tombstone preserved");
                assert_eq!(entries[1].key, b"k2");
                assert_eq!(entries[1].txn_id, 2);
            }
        }
    }

    #[test]
    fn patch_v4_crc_corruption_rejected() {
        let patch = Z1PatchFormatV4::sparse(vec![PatchEntry {
            key: b"k".to_vec(),
            value: Some(Arc::new(vec![1])),
            txn_id: 1,
        }]);
        let mut bytes = patch.to_disk_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(Z1PatchFormatV4::from_disk_bytes(&bytes).is_err());
    }

    #[test]
    fn patch_v4_entry_count_and_bounds() {
        let patch = Z1PatchFormatV4::sparse(vec![
            PatchEntry {
                key: b"aaa".to_vec(),
                value: None,
                txn_id: 1,
            },
            PatchEntry {
                key: b"zzz".to_vec(),
                value: Some(Arc::new(vec![9])),
                txn_id: 5,
            },
        ]);
        assert_eq!(patch.entry_count(), 2);
        assert_eq!(patch.max_txn_id(), Some(5));
        assert_eq!(patch.min_txn_id(), Some(1));
        let (min, max) = patch.key_bounds();
        assert_eq!(min, Some(&b"aaa"[..]));
        assert_eq!(max, Some(&b"zzz"[..]));
    }

    /// ZoneMap regression for append_patch: min_txn must be the patch's
    /// smallest txn id (it was once wrongly assigned max_txn_id).
    #[test]
    fn zone_map_txn_bounds_are_correct() {
        use super::super::disk::DiskLayer;
        use super::super::flush::FlushEngine;
        use crate::store::config::SyncLevel;
        use crate::store::mem::MemStore;
        use crate::store::recent_flush_cache::RecentFlushCache;

        let dir = std::env::temp_dir().join(format!("z1kv_disk_zonemap_{}", uuid::Uuid::new_v4()));
        let l1 = Arc::new(MemStore::new(usize::MAX, SyncLevel::Async));
        let l2 = Arc::new(DiskLayer::new(dir.join("l2")));
        let l3 = Arc::new(DiskLayer::new(dir.join("l3")));
        let engine = FlushEngine::with_l3(
            l1,
            Arc::clone(&l2),
            l3,
            Arc::new(RecentFlushCache::new()),
            Arc::new(std::sync::atomic::AtomicU64::new(0)),
            0,
        );
        engine.flush_l1_to_l2().unwrap(); // empty flush → no patch

        l2.append_patch(
            0,
            vec![
                PatchEntry {
                    key: b"k".to_vec(),
                    value: Some(Arc::new(vec![1])),
                    txn_id: 3,
                },
                PatchEntry {
                    key: b"k2".to_vec(),
                    value: Some(Arc::new(vec![2])),
                    txn_id: 9,
                },
            ],
            7,
        )
        .unwrap();

        // Read the patch back from disk and verify its zone map.
        let bytes =
            std::fs::read(dir.join("l2").join("0000").join("0000000000000007.zpatch")).unwrap();
        let format = Z1PatchFormatV4::from_disk_bytes(&bytes).unwrap();
        assert_eq!(format.max_txn_id(), Some(9));
        assert_eq!(format.min_txn_id(), Some(3));
        let _ = engine; // keep alive
    }
}
