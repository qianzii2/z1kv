//! Disk patch layer (L2 / L3) — append-only patch files.
//!
//! The disk layers (L2 patches / L3 frozen).
//!
//! - On-disk layout: `{root}/{cf}/{patch_id:016}.zpatch`, where each patch
//!   file is a `DiskFormat`-protected `Z1PatchFormatV4`
//! - L2 and L3 share this struct with different root_dirs
//!
//! # Guard lookup (key-range pruning index)
//!
//! Each patch records its `(min_key, max_key)` into an in-memory index
//! (sorted by key). When reading a key, the index is **linearly filtered**
//! to the patch files whose key range covers it — only matching files are
//! read, preventing read amplification from growing linearly with the patch
//! count. The key bounds come from the actual key distribution at flush
//! time (PebblesDB-style guard separators).
//!
//! Note: the current implementation linearly filters rather than binary
//! searches (an earlier doc claimed binary search; corrected). At the
//! current patch counts linear filtering is sufficient; binary search on
//! `min_key` is the upgrade path for much larger patch counts.
//!
//! # Read path
//!
//! `get_versions(cf, key)` uses the index to locate the patch files whose
//! key range covers the key, reads each one, and collects all versions of
//! the key for the upper layer to filter by MVCC visibility.

use crate::codec::disk_format::DiskFormat;
use crate::durable_writer::DurableWriter;
use crate::error::{Result, Z1Error};
use crate::store::types::{PatchEntry, PatchZoneMap, Z1Entry, Z1Key, Z1Patch, Z1PatchFormatV4};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Key-range index entry for one patch file.
/// `min_key` / `max_key` are the patch's smallest/largest keys (lexicographic).
type RangeIndexEntry = (Vec<u8>, Vec<u8>, PathBuf);

/// A disk patch layer.
pub struct DiskLayer {
    root_dir: PathBuf,
    /// In-memory range index: cf -> (min_key, max_key, path) list sorted by
    /// min_key. When reading a key, only patch files with
    /// key ∈ [min_key, max_key] are consulted.
    index: RwLock<BTreeMap<u16, Vec<RangeIndexEntry>>>,
    /// Patch content cache (path -> deserialized Sparse entries).
    /// L2 patches are immutable (append-only; invalidated wholesale by
    /// drop_cf), so they are cached after the first read — turning
    /// get_versions' "read N files + deserialize per get" into memory
    /// lookups. Cache size is on the order of the L2 data (acceptable;
    /// cleaned up when drop_cf/compact removes patches).
    content: RwLock<std::collections::HashMap<PathBuf, std::sync::Arc<Vec<PatchEntry>>>>,
}

impl DiskLayer {
    pub fn new(root_dir: PathBuf) -> Self {
        // A failure here does not abort construction (the read path
        // naturally tolerates a missing directory), but is logged for
        // diagnostics; real write failures surface as Err in append_patch.
        if let Err(e) = std::fs::create_dir_all(&root_dir) {
            tracing::warn!("DiskLayer: create_dir_all({:?}) failed: {}", root_dir, e);
        }
        let index = Self::rebuild_index(&root_dir);
        Self {
            root_dir,
            index: RwLock::new(index),
            content: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Read patch contents (cached). NotFound is handled by the caller as
    /// a degradation.
    fn load_patch(&self, path: &std::path::Path) -> Result<std::sync::Arc<Vec<PatchEntry>>> {
        if let Some(hit) = self.content.read().get(path) {
            return Ok(std::sync::Arc::clone(hit));
        }
        let bytes = std::fs::read(path).map_err(Z1Error::Io)?;
        let format = Z1PatchFormatV4::from_disk_bytes(&bytes)?;
        let Z1PatchFormatV4::Sparse { entries } = format;
        let arc = std::sync::Arc::new(entries);
        self.content
            .write()
            .insert(path.to_path_buf(), std::sync::Arc::clone(&arc));
        Ok(arc)
    }

    /// Remove a patch from the content cache and the index (drop_cf / a
    /// file that disappeared).
    fn evict_patch(&self, path: &std::path::Path) {
        self.content.write().remove(path);
    }

    /// Scan disk patch files at startup and rebuild the key-range index.
    ///
    /// The index is in-memory only (lost on restart) and must be recovered
    /// from the disk patch files: each patch's payload is parsed for
    /// min_key/max_key, then sorted by min_key.
    fn rebuild_index(root_dir: &std::path::Path) -> BTreeMap<u16, Vec<RangeIndexEntry>> {
        let mut index: BTreeMap<u16, Vec<RangeIndexEntry>> = BTreeMap::new();

        let Ok(cf_entries) = std::fs::read_dir(root_dir) else {
            return index;
        };

        for cf_entry in cf_entries.flatten() {
            // cf directory name = 4 hex digits.
            let cf_name = cf_entry.file_name();
            let cf_name = cf_name.to_string_lossy();
            if cf_name.len() != 4 {
                continue;
            }
            let Ok(cf) = u16::from_str_radix(&cf_name, 16) else {
                continue;
            };
            if !cf_entry.path().is_dir() {
                continue;
            }

            let mut entries: Vec<RangeIndexEntry> = Vec::new();
            let Ok(patch_entries) = std::fs::read_dir(cf_entry.path()) else {
                continue;
            };
            for pe in patch_entries.flatten() {
                let path = pe.path();
                let Ok(bytes) = std::fs::read(&path) else {
                    tracing::warn!("DiskLayer: failed to read patch {:?}, skipping", path);
                    continue;
                };
                let Ok(format) = Z1PatchFormatV4::from_disk_bytes(&bytes) else {
                    tracing::warn!("DiskLayer: corrupt patch {:?}, skipping", path);
                    continue;
                };
                let (min, max) = format.key_bounds();
                entries.push((
                    min.map(|k| k.to_vec()).unwrap_or_default(),
                    max.map(|k| k.to_vec()).unwrap_or_default(),
                    path,
                ));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            index.insert(cf, entries);
        }

        index
    }

    fn cf_dir(&self, cf: u16) -> PathBuf {
        self.root_dir.join(format!("{:04x}", cf))
    }

    fn patch_path(&self, cf: u16, patch_id: u64) -> PathBuf {
        self.cf_dir(cf).join(format!("{:016}.zpatch", patch_id))
    }

    /// Append a patch (grouped entries) to the layer.
    ///
    /// Invariant: patches must be routed by the real key (grouped by cf
    /// here; one patch file per flush batch — a key is never routed to a
    /// hardcoded default location).
    pub fn append_patch(&self, cf: u16, entries: Vec<PatchEntry>, patch_id: u64) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let cf_dir = self.cf_dir(cf);
        std::fs::create_dir_all(&cf_dir).map_err(Z1Error::Io)?;

        let format = Z1PatchFormatV4::sparse(entries);
        let (min_key, max_key) = format.key_bounds();
        // Fix: min_txn was once wrongly assigned max_txn_id() (a
        // copy-paste slip); min/max are now taken per their field semantics.
        let zone_map = PatchZoneMap {
            min_txn: format.min_txn_id().unwrap_or(0),
            max_txn: format.max_txn_id().unwrap_or(0),
            entry_count: format.entry_count() as u64,
            min_key: min_key.map(|k| k.to_vec()),
            max_key: max_key.map(|k| k.to_vec()),
        };
        let patch = Z1Patch {
            patch_id,
            txn_range: (zone_map.min_txn, zone_map.max_txn),
            format,
            zone_map,
        };

        let path = self.patch_path(cf, patch_id);
        let bytes = patch.format.to_disk_bytes()?;

        // Atomic write via DurableWriter.
        let mut writer = crate::durable_writer::SimpleFileWriter;
        writer.write_durable(&path, &bytes)?;

        // Record in the key-range index (a non-empty patch always has
        // min/max keys).
        let min = patch.zone_map.min_key.clone().unwrap_or_default();
        let max = patch.zone_map.max_key.clone().unwrap_or_default();
        self.index
            .write()
            .entry(cf)
            .or_default()
            .push((min, max, path));
        Ok(())
    }

    /// Get all versions of a key (unsorted; caller applies MVCC visibility).
    ///
    /// Locate via the key-range index: only patch files with
    /// key ∈ [min_key, max_key] are read.
    pub fn get_versions(&self, cf: u16, key: &[u8]) -> Result<Vec<Z1Entry>> {
        let mut out = Vec::new();

        // Snapshot the index (clone this cf's path list under the read
        // lock, then release the lock before doing I/O).
        let paths: Vec<PathBuf> = {
            let idx = self.index.read();
            idx.get(&cf)
                .map(|entries| {
                    entries
                        .iter()
                        .filter(|(min, max, _)| key >= min.as_slice() && key <= max.as_slice())
                        .map(|(_, _, path)| path.clone())
                        .collect()
                })
                .unwrap_or_default()
        };

        for path in paths {
            // Resilience: when a patch file goes missing (NotFound),
            // degrade — warn, remove it from the index, and keep reading the
            // other patches. Other I/O errors still propagate.
            let entries = match self.load_patch(&path) {
                Ok(a) => a,
                Err(Z1Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::warn!("patch file vanished, skipping: {:?}", path);
                    self.remove_from_index(&path);
                    self.evict_patch(&path);
                    continue;
                }
                Err(e) => return Err(e),
            };
            for pe in entries.iter() {
                if pe.key == key {
                    out.push(Z1Entry {
                        key: Z1Key::new(cf, key),
                        txn_id: pe.txn_id,
                        value: pe.value.clone(),
                        ts: 0,
                    });
                }
            }
        }

        Ok(out)
    }

    /// List all patch ids for a cf (for compaction / GC bookkeeping).
    pub fn patch_ids(&self, cf: u16) -> Vec<u64> {
        let cf_dir = self.cf_dir(cf);
        if !cf_dir.exists() {
            return Vec::new();
        }
        std::fs::read_dir(&cf_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.strip_suffix(".zpatch")
                            .and_then(|s| s.parse::<u64>().ok())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove the patch entry at `path` from the in-memory index
    /// (degradation cleanup when a file disappears).
    fn remove_from_index(&self, path: &std::path::Path) {
        let mut idx = self.index.write();
        for entries in idx.values_mut() {
            entries.retain(|(_, _, p)| p != path);
        }
    }

    /// The highest patch id already used by this layer (across all cfs).
    ///
    /// Fix: `next_patch_id` was a pure in-memory counter that reset to zero
    /// on restart. If not recovered at open time, the first flush after a
    /// reopen would allocate ids from 0 again, and the `write_durable`
    /// rename would **overwrite** existing same-id patch files while the
    /// in-memory index entries kept the old key ranges — as soon as the WAL
    /// had been truncated by a checkpoint, the replayed data is less than
    /// the old patch content, so the overwrite meant permanent data loss.
    /// Recovery rule: `next = max(existing) + 1`; an empty layer starts at 0.
    pub fn max_patch_id(&self) -> u64 {
        self.index
            .read()
            .values()
            .flat_map(|entries| {
                entries.iter().filter_map(|(_, _, path)| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.parse::<u64>().ok())
                })
            })
            .max()
            .unwrap_or(0)
    }

    /// Collect all versions on disk patches for a cf within the key range
    /// `[start, end)`, taking the highest txn_id version per key (disk
    /// patches only hold committed versions).
    ///
    /// Pruned via the key-range index: only patch files whose range
    /// intersects the query interval are read. Returns `(key_bytes, entry)`
    /// pairs sorted by key (tombstones included; the caller filters).
    pub fn range_visible(
        &self,
        cf: u16,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Z1Entry)>> {
        use std::collections::BTreeMap as StdBTreeMap;

        // Locate the patch files whose range intersects the query.
        let paths: Vec<PathBuf> = {
            let idx = self.index.read();
            idx.get(&cf)
                .map(|entries| {
                    entries
                        .iter()
                        .filter(|(min, max, _)| {
                            // Patch range [min, max] intersects [start, end):
                            // max >= start AND (end is None OR min < end)
                            let overlaps_start = max.as_slice() >= start;
                            let overlaps_end = match end {
                                Some(e) => min.as_slice() < e,
                                None => true,
                            };
                            overlaps_start && overlaps_end
                        })
                        .map(|(_, _, path)| path.clone())
                        .collect()
                })
                .unwrap_or_default()
        };

        // key -> highest txn_id version.
        let mut best: StdBTreeMap<Vec<u8>, Z1Entry> = StdBTreeMap::new();

        for path in paths {
            let entries = match self.load_patch(&path) {
                Ok(a) => a,
                Err(Z1Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::warn!("patch file vanished, skipping: {:?}", path);
                    self.remove_from_index(&path);
                    self.evict_patch(&path);
                    continue;
                }
                Err(e) => return Err(e),
            };
            for pe in entries.iter() {
                if pe.key.as_slice() < start {
                    continue;
                }
                if let Some(e) = end {
                    if pe.key.as_slice() >= e {
                        continue;
                    }
                }
                let entry = Z1Entry {
                    key: Z1Key::new(cf, pe.key.clone()),
                    txn_id: pe.txn_id,
                    value: pe.value.clone(),
                    ts: 0,
                };
                match best.get(&entry.key.key) {
                    Some(cur) if cur.txn_id >= entry.txn_id => {}
                    _ => {
                        best.insert(entry.key.key.clone(), entry);
                    }
                }
            }
        }

        Ok(best.into_iter().collect())
    }

    /// Drop all data for a cf (for GC / test cleanup).
    pub fn drop_cf(&self, cf: u16) -> Result<()> {
        let cf_dir = self.cf_dir(cf);
        if cf_dir.exists() {
            std::fs::remove_dir_all(&cf_dir).map_err(Z1Error::Io)?;
        }
        self.index.write().remove(&cf);
        // Clean up the content cache too (drop_cf deletes the cf directory
        // wholesale).
        let prefix = self.cf_dir(cf);
        self.content.write().retain(|p, _| !p.starts_with(&prefix));
        Ok(())
    }

    /// Enumerate the cfs present in this layer (for compaction iteration).
    pub fn list_cfs(&self) -> Vec<u16> {
        self.index.read().keys().copied().collect()
    }

    /// Collect all versions for a cf (all keys across all patches,
    /// including multiple versions of the same key).
    /// Feeds the compaction GC merge.
    pub fn all_versions(&self, cf: u16) -> Result<Vec<Z1Entry>> {
        let mut out = Vec::new();
        let paths: Vec<PathBuf> = {
            let idx = self.index.read();
            idx.get(&cf)
                .map(|entries| entries.iter().map(|(_, _, p)| p.clone()).collect())
                .unwrap_or_default()
        };
        for path in paths {
            let entries = match self.load_patch(&path) {
                Ok(a) => a,
                Err(Z1Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::warn!("patch file vanished, skipping: {:?}", path);
                    self.remove_from_index(&path);
                    self.evict_patch(&path);
                    continue;
                }
                Err(e) => return Err(e),
            };
            for pe in entries.iter() {
                out.push(Z1Entry {
                    key: Z1Key::new(cf, pe.key.clone()),
                    txn_id: pe.txn_id,
                    value: pe.value.clone(),
                    ts: 0,
                });
            }
        }
        Ok(out)
    }
}

// Re-export a convenient alias: L2 is a disk layer rooted at `l2`, L3 at `l3`.
pub type L2Disk = DiskLayer;
pub type L3Frozen = DiskLayer;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("z1kv_disk_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn append_and_read_versions() {
        let dir = tmp_dir("roundtrip");
        let layer = DiskLayer::new(dir.join("l2"));

        layer
            .append_patch(
                0,
                vec![
                    PatchEntry {
                        key: b"k".to_vec(),
                        value: Some(Arc::new(vec![1])),
                        txn_id: 1,
                    },
                    PatchEntry {
                        key: b"k".to_vec(),
                        value: Some(Arc::new(vec![2])),
                        txn_id: 2,
                    },
                ],
                1,
            )
            .unwrap();

        let versions = layer.get_versions(0, b"k").unwrap();
        assert_eq!(versions.len(), 2);
        assert_eq!(
            versions.iter().map(|e| e.txn_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn cf_separation() {
        let dir = tmp_dir("cf");
        let layer = DiskLayer::new(dir.join("l2"));

        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"k".to_vec(),
                    value: Some(Arc::new(vec![1])),
                    txn_id: 1,
                }],
                1,
            )
            .unwrap();
        layer
            .append_patch(
                1,
                vec![PatchEntry {
                    key: b"k".to_vec(),
                    value: Some(Arc::new(vec![2])),
                    txn_id: 1,
                }],
                1,
            )
            .unwrap();

        let v0 = layer.get_versions(0, b"k").unwrap();
        let v1 = layer.get_versions(1, b"k").unwrap();
        assert_eq!(v0.len(), 1);
        assert_eq!(v1.len(), 1);
        assert_eq!(
            v0[0].value.as_deref().map(|v| v.as_slice()),
            Some(&[1u8][..])
        );
        assert_eq!(
            v1[0].value.as_deref().map(|v| v.as_slice()),
            Some(&[2u8][..])
        );
    }

    #[test]
    fn tombstone_preserved() {
        let dir = tmp_dir("tombstone");
        let layer = DiskLayer::new(dir.join("l2"));

        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"k".to_vec(),
                    value: None,
                    txn_id: 5,
                }],
                1,
            )
            .unwrap();

        let versions = layer.get_versions(0, b"k").unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].is_tombstone());
    }

    #[test]
    fn patch_ids_are_discoverable() {
        let dir = tmp_dir("ids");
        let layer = DiskLayer::new(dir.join("l2"));

        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"a".to_vec(),
                    value: Some(Arc::new(vec![1])),
                    txn_id: 1,
                }],
                10,
            )
            .unwrap();
        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"b".to_vec(),
                    value: Some(Arc::new(vec![2])),
                    txn_id: 2,
                }],
                11,
            )
            .unwrap();

        let mut ids = layer.patch_ids(0);
        ids.sort_unstable();
        assert_eq!(ids, vec![10, 11]);
    }

    /// The index is in-memory and must be rebuilt from disk patch files
    /// after restart.
    /// Verified by: dropping the layer (simulating a restart) → new
    /// DiskLayer → get_versions still readable.
    #[test]
    fn index_rebuilt_on_reopen() {
        let dir = tmp_dir("rebuild");
        let l2_root = dir.join("l2");

        {
            let layer = DiskLayer::new(l2_root.clone());
            layer
                .append_patch(
                    0,
                    vec![PatchEntry {
                        key: b"k1".to_vec(),
                        value: Some(Arc::new(vec![1])),
                        txn_id: 1,
                    }],
                    1,
                )
                .unwrap();
            layer
                .append_patch(
                    0,
                    vec![PatchEntry {
                        key: b"k2".to_vec(),
                        value: Some(Arc::new(vec![2])),
                        txn_id: 2,
                    }],
                    2,
                )
                .unwrap();
            // layer drop (simulating process exit)
        }

        // Restart: the new DiskLayer must rebuild its index from disk.
        let layer = DiskLayer::new(l2_root);
        let v1 = layer.get_versions(0, b"k1").unwrap();
        let v2 = layer.get_versions(0, b"k2").unwrap();
        assert_eq!(v1.len(), 1, "k1 must be readable after index rebuild");
        assert_eq!(v2.len(), 1, "k2 must be readable after index rebuild");
        assert_eq!(
            v1[0].value.as_deref().map(|v| v.as_slice()),
            Some(&[1u8][..])
        );
        assert_eq!(
            v2[0].value.as_deref().map(|v| v.as_slice()),
            Some(&[2u8][..])
        );
    }

    /// Key-range pruning: keys outside every patch range must return empty
    /// (and read no files).
    /// Regression for the NotFound degradation: when a patch file is
    /// missing, it is removed from the index, the remaining patches are
    /// read, and the read does not propagate an IO Err.
    #[test]
    fn vanished_patch_degrades_gracefully() {
        let dir = tmp_dir("vanished");
        let layer = DiskLayer::new(dir.join("l2"));

        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"good".to_vec(),
                    value: Some(Arc::new(vec![1])),
                    txn_id: 1,
                }],
                1,
            )
            .unwrap();
        layer
            .append_patch(
                0,
                vec![PatchEntry {
                    key: b"bad".to_vec(),
                    value: Some(Arc::new(vec![2])),
                    txn_id: 2,
                }],
                2,
            )
            .unwrap();

        // Injection: delete the second patch file (the index still points
        // at it).
        let bad_path = dir
            .join("l2")
            .join("0000")
            .join(format!("{:016}.zpatch", 2));
        assert!(bad_path.exists());
        std::fs::remove_file(&bad_path).unwrap();

        // Read "bad" (index hit but the file vanished): must degrade to
        // empty, not an IO Err.
        let r = layer.get_versions(0, b"bad");
        assert!(
            r.is_ok(),
            "vanished patch must degrade, not error: {:?}",
            r.err()
        );
        assert!(r.unwrap().is_empty());

        // Read "good" (the first patch still exists): must return normally.
        assert_eq!(layer.get_versions(0, b"good").unwrap().len(), 1);
    }

    #[test]
    fn range_pruning_skips_out_of_range_keys() {
        let dir = tmp_dir("prune");
        let layer = DiskLayer::new(dir.join("l2"));

        // The patch only covers [a, b].
        layer
            .append_patch(
                0,
                vec![
                    PatchEntry {
                        key: b"a".to_vec(),
                        value: Some(Arc::new(vec![1])),
                        txn_id: 1,
                    },
                    PatchEntry {
                        key: b"b".to_vec(),
                        value: Some(Arc::new(vec![2])),
                        txn_id: 2,
                    },
                ],
                1,
            )
            .unwrap();

        // "zz" is outside [a, b] — pruned by the index, no file read, empty result.
        assert!(layer.get_versions(0, b"zz").unwrap().is_empty());
        // "a" and "b" are in range and read back normally.
        assert_eq!(layer.get_versions(0, b"a").unwrap().len(), 1);
        assert_eq!(layer.get_versions(0, b"b").unwrap().len(), 1);
    }
}
