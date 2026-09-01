//! GC — version reclamation (physical MVCC cleanup).
//!
//! The tail end of MVCC: old versions must not accumulate forever. During
//! compaction, versions no longer referenced by any snapshot are physically
//! reclaimed according to the "safe watermark".
//!
//! # Safe watermark
//!
//! A version is safe to delete if and only if:
//! 1. It is **superseded by a later version** (the same key has a higher
//!    txn_id version)
//! 2. Its txn_id < `min_active_begin_ts` (no active transaction can see it
//!    in its snapshot)
//! 3. It is below `replay_watermark` (the history baseline already
//!    persisted via checkpoint)
//!
//! Retention policy — for each key keep:
//! - all versions with `txn_id >= min_active_begin_ts` (active snapshots may
//!   reference them)
//! - the **newest** version before `min_active_begin_ts` (history baseline
//!   for long-running transactions)
//!
//! All other old versions are physically deleted.
//!
//! # Easy-to-confuse points
//!
//! - GC must obey D12: a txn absent from commit_ts_map is invisible — but GC
//!   deals with committed versions, so reclamation uses the txn_id
//!   watermark, not the visibility rules.
//! - A tombstone may be deleted once its txn_id is below the watermark and
//!   no snapshot references it.

use crate::store::types::PatchEntry;
use crate::TxnId;

/// GC statistics.
#[derive(Debug, Default, Clone)]
pub struct GcStats {
    /// Number of entries scanned.
    pub scanned: usize,
    /// Number of entries retained.
    pub retained: usize,
    /// Number of entries physically reclaimed.
    pub reclaimed: usize,
}

/// GC-merge the collected versions of a column family, returning the
/// retained `PatchEntry` list.
///
/// `min_active_begin_ts`: the smallest begin_ts among active transactions
/// (the safe watermark). Among versions below the watermark, only the newest
/// one per key is kept.
///
/// Input entries need not be pre-sorted; they are sorted by
/// `(key, txn_id)` internally.
pub fn gc_entries(
    mut entries: Vec<PatchEntry>,
    min_active_begin_ts: TxnId,
) -> (Vec<PatchEntry>, GcStats) {
    let mut stats = GcStats {
        scanned: entries.len(),
        ..Default::default()
    };

    // Sort: key ascending, txn_id ascending within a key.
    entries.sort_by(|a, b| a.key.cmp(&b.key).then(a.txn_id.cmp(&b.txn_id)));

    let mut retained: Vec<PatchEntry> = Vec::new();

    // Process grouped by key: one version chain per key.
    let mut i = 0;
    while i < entries.len() {
        // Find the range [i, j) of entries sharing this key.
        let key = entries[i].key.clone();
        let mut j = i;
        while j < entries.len() && entries[j].key == key {
            j += 1;
        }
        // entries[i..j] is this key's version chain (txn_id ascending).
        let chain = &entries[i..j];

        // Index of the newest version below the watermark (if any).
        let mut last_before_watermark: Option<usize> = None;
        for (k, e) in chain.iter().enumerate() {
            if e.txn_id < min_active_begin_ts {
                last_before_watermark = Some(k);
            } else {
                break; // ascending order: everything after is >= watermark
            }
        }

        // Keep:
        // 1. the newest version below the watermark (history baseline)
        // 2. all versions >= the watermark
        for (k, e) in chain.iter().enumerate() {
            let keep = if Some(k) == last_before_watermark {
                true
            } else {
                e.txn_id >= min_active_begin_ts
            };
            if keep {
                retained.push(e.clone());
                stats.retained += 1;
            } else {
                stats.reclaimed += 1;
            }
        }

        i = j;
    }

    (retained, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn pe(key: &[u8], txn_id: TxnId, v: u8) -> PatchEntry {
        PatchEntry {
            key: key.to_vec(),
            value: Some(Arc::new(vec![v])),
            txn_id,
        }
    }

    fn tombstone(key: &[u8], txn_id: TxnId) -> PatchEntry {
        PatchEntry {
            key: key.to_vec(),
            value: None,
            txn_id,
        }
    }

    #[test]
    fn keeps_latest_version_before_watermark_plus_all_after() {
        // Key "k" has 5 versions: txn 1..5, watermark = 4.
        let entries = vec![
            pe(b"k", 1, 1),
            pe(b"k", 2, 2),
            pe(b"k", 3, 3),
            pe(b"k", 4, 4),
            pe(b"k", 5, 5),
        ];
        let (retained, stats) = gc_entries(entries, 4);

        assert_eq!(stats.scanned, 5);
        // Keep txn 3 (newest below watermark) + 4 + 5 = 3; reclaim txns 1, 2.
        assert_eq!(stats.retained, 3);
        assert_eq!(stats.reclaimed, 2);

        let txns: Vec<TxnId> = retained.iter().map(|e| e.txn_id).collect();
        assert_eq!(txns, vec![3, 4, 5]);
    }

    #[test]
    fn tombstone_below_watermark_removed_if_covered() {
        // Key "k": put(1), delete(2), watermark = 3 (no active snapshot
        // references the old versions).
        let entries = vec![pe(b"k", 1, 1), tombstone(b"k", 2)];
        let (retained, stats) = gc_entries(entries, 3);

        // Keep only txn 2 (newest below watermark = the tombstone); txn 1
        // is reclaimed.
        assert_eq!(stats.retained, 1);
        assert_eq!(stats.reclaimed, 1);
        assert!(
            retained[0].value.is_none(),
            "tombstone retained as baseline"
        );
    }

    #[test]
    fn active_snapshot_protects_old_versions() {
        // Watermark = 1 (an active snapshot started before txn 1): all
        // versions are >= the watermark, so all are kept.
        let entries = vec![pe(b"k", 1, 1), pe(b"k", 2, 2)];
        let (retained, stats) = gc_entries(entries, 1);

        assert_eq!(stats.reclaimed, 0);
        assert_eq!(retained.len(), 2);
    }

    #[test]
    fn multiple_keys_gc_independently() {
        let entries = vec![
            pe(b"a", 1, 1),
            pe(b"a", 2, 2),
            pe(b"b", 1, 1),
            pe(b"b", 2, 2),
            pe(b"b", 3, 3),
        ];
        let (retained, stats) = gc_entries(entries, 3);

        // a: keep txn 2 (newest below watermark), reclaim txn 1
        // b: keep txn 2 (newest below watermark) + 3, reclaim txn 1
        assert_eq!(stats.reclaimed, 2);
        assert_eq!(stats.retained, 3);

        let a_txns: Vec<TxnId> = retained
            .iter()
            .filter(|e| e.key == b"a")
            .map(|e| e.txn_id)
            .collect();
        let b_txns: Vec<TxnId> = retained
            .iter()
            .filter(|e| e.key == b"b")
            .map(|e| e.txn_id)
            .collect();
        assert_eq!(a_txns, vec![2]);
        assert_eq!(b_txns, vec![2, 3]);
    }
}
