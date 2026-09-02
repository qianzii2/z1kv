//! Engine integration tests: crash recovery, concurrency, the engine lock,
//! patch-id recovery and stress consistency.
//!
//! Core semantics: a crash at any moment (process exit, without flush) must
//! recover all committed writes via WAL replay; uncommitted writes are
//! discarded (WAL-first — commit returning Ok is the durability boundary).

use std::path::PathBuf;
use z1kv::Z1Kv;

fn tmp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("z1kv_crash_test_{}_{}", name, uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Crash after a single commit (drop without flush) → recover on reopen.
#[test]
fn commit_then_crash_recovers() {
    let dir = tmp_dir("single_commit");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        db.commit(txn).unwrap();
        // Drop directly, without calling flush().
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"k").unwrap(), Some(b"v".to_vec()));
}

/// Crash after many commits → all recovered.
#[test]
fn many_commits_then_crash_recovers_all() {
    let dir = tmp_dir("many_commits");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        for i in 0..100u64 {
            let txn = db.begin_txn().unwrap();
            db.put(0, i.to_le_bytes().to_vec(), i.to_string().into_bytes(), txn)
                .unwrap();
            db.commit(txn).unwrap();
        }
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    for i in 0..100u64 {
        assert_eq!(
            db.get(0, &i.to_le_bytes()).unwrap(),
            Some(i.to_string().into_bytes()),
            "key {} must survive crash",
            i
        );
    }
}

/// Crash with an uncommitted transaction → discarded.
#[test]
fn uncommitted_discarded_on_crash() {
    let dir = tmp_dir("uncommitted");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        // No commit; drop directly.
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"k").unwrap(), None);
}

/// Crash after a delete (tombstone) → still invisible after recovery.
#[test]
fn delete_then_crash_stays_deleted() {
    let dir = tmp_dir("delete");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        db.commit(txn).unwrap();
        let txn2 = db.begin_txn().unwrap();
        db.delete(0, b"k", txn2).unwrap();
        db.commit(txn2).unwrap();
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"k").unwrap(), None);
}

/// Multi-cf isolation + cross-cf transaction recovery.
#[test]
fn multi_cf_crash_recovers() {
    let dir = tmp_dir("multi_cf");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"cf0", txn).unwrap();
        db.put(1, b"k", b"cf1", txn).unwrap();
        db.commit(txn).unwrap();
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"k").unwrap(), Some(b"cf0".to_vec()));
    assert_eq!(db.get(1, b"k").unwrap(), Some(b"cf1".to_vec()));
}

/// Crash after commits + an unconditional L1→L2 flush → data recovers from
/// WAL + L2 (two layers).
///
/// Note: the earlier version called `db.flush()` — threshold-triggered, a
/// no-op with small data — so the test never actually produced L2 data
/// ("two-layer recovery" was in name only). It now uses `flush_now()`
/// (unconditional), genuinely covering the recovery scenario where L1 is
/// empty and data lives only in WAL + L2.
#[test]
fn flush_then_crash_recovers() {
    let dir = tmp_dir("flush");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let txn = db.begin_txn().unwrap();
        db.put(0, b"k", b"v", txn).unwrap();
        db.commit(txn).unwrap();
        // Unconditional flush (L1 → L2); L1 is emptied right after.
        db.flush_now().unwrap();
    }
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"k").unwrap(), Some(b"v".to_vec()));
}

/// Regression: after reopen the patch-id counter must be recovered from
/// disk; a flush must not overwrite existing patches.
///
/// Before the fix: `next_patch_id` reset to zero on restart → the first
/// flush after a reopen produced patch_id=0, whose `write_durable` rename
/// overwrote the same-id file from the previous flush, and the disk index's
/// old entry kept a misaligned key range. Earlier tests missed this because
/// recovery replays the whole WAL into L1, so a second flush rewrote the
/// data into the overwritten file — masking rather than fixing; once the
/// WAL was truncated by a checkpoint, the overwrite meant data loss.
///
/// This test truncates the WAL via checkpoint to approach the real loss
/// path, and asserts directly at the patch-file level that no overwrite
/// occurs (new and old patch ids differ).
#[test]
fn reopen_then_flush_does_not_overwrite_existing_patches() {
    let dir = tmp_dir("patchid_regress");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let t = db.begin_txn().unwrap();
        db.put(0, b"aaa", b"batch1", t).unwrap();
        db.commit(t).unwrap();
        // Checkpoint internally takes the unconditional flush_l1_to_l2 path
        // (db.flush() is the threshold-triggered try_flush, which does not
        // fire with small data) and truncates the WAL — the real pre-fix
        // data-loss path. The first patch id = 1 (recovery formula
        // max(existing)+1; an empty DB has max=0).
        db.checkpoint().unwrap();
    }

    // Crash & reopen: the counter must be recovered from disk, not reset to zero.
    let db = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db.get(0, b"aaa").unwrap(), Some(b"batch1".to_vec()));

    let t = db.begin_txn().unwrap();
    db.put(0, b"zzz", b"batch2", t).unwrap();
    db.commit(t).unwrap();
    db.checkpoint().unwrap(); // must allocate a NEW id (=2), never reuse 1

    // Both batches must be readable (the old patch was not overwritten; the
    // index is not misaligned).
    assert_eq!(
        db.get(0, b"aaa").unwrap(),
        Some(b"batch1".to_vec()),
        "batch1 must survive reopen+flush (patch must not be overwritten)"
    );
    assert_eq!(db.get(0, b"zzz").unwrap(), Some(b"batch2".to_vec()));

    // Directly verify at the disk level: the cf=0 directory must contain 2
    // patch files with different ids, and the second allocated id must be
    // strictly greater than the first (no reuse). Recovery formula is
    // max(existing)+1; the first patch id in an empty DB is 1.
    let cf_dir = dir.join("l2").join("0000");
    let mut ids: Vec<u64> = std::fs::read_dir(&cf_dir)
        .unwrap()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy().into_owned();
            name.strip_suffix(".zpatch")
                .and_then(|s| s.parse::<u64>().ok())
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2],
        "second flush must allocate a fresh patch id, not overwrite the old one"
    );
}

/// Regression (truncation path): after checkpoint truncates the WAL, a
/// reopen followed by flush must not lose the old patch data through
/// patch-id reuse — the scenario that genuinely lost data before the fix.
#[test]
fn reopen_after_checkpoint_then_flush_preserves_l2_data() {
    let dir = tmp_dir("patchid_ckpt_regress");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        let t = db.begin_txn().unwrap();
        db.put(0, b"old_key", b"old_value", t).unwrap();
        db.commit(t).unwrap();
        db.flush().unwrap(); // L2 patch id=0
        db.checkpoint().unwrap(); // WAL truncated — replay no longer covers batch1's put
    }

    let db = Z1Kv::open(dir.clone()).unwrap();
    let t = db.begin_txn().unwrap();
    db.put(0, b"new_key", b"new_value", t).unwrap();
    db.commit(t).unwrap();
    db.checkpoint().unwrap(); // unconditional flush (as above)

    // Before the fix: the second flush reused id=0 and overwrote the old
    // patch; old_key existed only in that patch (WAL truncated, L1
    // emptied), so after the overwrite get(old_key) = None.
    assert_eq!(
        db.get(0, b"old_key").unwrap(),
        Some(b"old_value".to_vec()),
        "checkpointed L2 data must survive reopen+flush (patch id reuse = data loss)"
    );
    assert_eq!(db.get(0, b"new_key").unwrap(), Some(b"new_value".to_vec()));
}

/// Regression: opening the same data_dir twice must fail explicitly
/// (preventing silent corruption from two engine instances writing
/// WAL/L2 concurrently); after the first instance drops, reopening works.
#[test]
fn double_open_is_rejected_until_first_dropped() {
    let dir = tmp_dir("double_open");

    let db1 = Z1Kv::open(dir.clone()).unwrap();
    let t = db1.begin_txn().unwrap();
    db1.put(0, b"k", b"v", t).unwrap();
    db1.commit(t).unwrap();

    // The second instance: must fail with a recognizable error message.
    let err = match Z1Kv::open(dir.clone()) {
        Err(e) => e,
        Ok(_) => panic!("double open must fail"),
    };
    assert!(
        err.to_string().contains("cannot lock data dir"),
        "unexpected error: {}",
        err
    );

    // db1 still works.
    assert_eq!(db1.get(0, b"k").unwrap(), Some(b"v".to_vec()));

    // db1 drop → lock released → reopen works, data intact.
    drop(db1);
    let db2 = Z1Kv::open(dir.clone()).unwrap();
    assert_eq!(db2.get(0, b"k").unwrap(), Some(b"v".to_vec()));
}

/// Regression: multi-threaded concurrent stress — 8 threads write to
/// their own key spaces transactionally while a background thread
/// repeatedly flushes/checkpoints/compacts. Verifies (1) no deadlock / no
/// panic, (2) all committed writes are readable after the stress, (3)
/// reopen is consistent afterward.
/// Before this test existed the codebase had zero multi-threaded tests —
/// concurrent interleavings of begin/commit/flush/checkpoint were never
/// verified.
#[test]
fn concurrent_threads_stress() {
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::sync::Arc as StdArc;
    #[allow(unused_imports)]
    use std::sync::Arc;

    let dir = tmp_dir("concurrent");
    let db: std::sync::Arc<Z1Kv> = std::sync::Arc::new(Z1Kv::open(dir.clone()).unwrap());

    let stop = StdArc::new(AtomicBool::new(false));
    let stop_bg = stop.clone();
    let db_bg = StdArc::clone(&db);

    // Background maintenance thread: repeated flush_now / checkpoint / compact.
    let bg = std::thread::spawn(move || {
        let db = db_bg;
        let mut i = 0u64;
        while !stop_bg.load(AtomicOrdering::Relaxed) {
            let _ = db.flush_now();
            let _ = db.checkpoint();
            let _ = db.compact(u64::MAX);
            i += 1;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        i
    });

    // 8 writer threads × 50 transactions, key spaces isolated per thread
    // (t{tid}_{i}).
    let mut handles = Vec::new();
    for tid in 0..8u64 {
        let db = StdArc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for i in 0..50u64 {
                let t = db.begin_txn().expect("begin");
                let key = format!("t{}_{}", tid, i);
                db.put(0, key.as_bytes(), i.to_le_bytes().to_vec(), t)
                    .expect("put");
                db.commit(t).expect("commit");
            }
        }));
    }
    for h in handles {
        h.join().expect("writer thread panicked (deadlock/bug)");
    }

    stop.store(true, AtomicOrdering::Relaxed);
    bg.join().expect("bg thread panicked");

    // All 400 keys must be readable with correct values.
    for tid in 0..8u64 {
        for i in 0..50u64 {
            let key = format!("t{}_{}", tid, i);
            let v = db
                .get(0, key.as_bytes())
                .unwrap_or_else(|e| panic!("get {} failed: {}", key, e));
            assert_eq!(
                v,
                Some(i.to_le_bytes().to_vec()),
                "key {} must hold its committed value",
                key
            );
        }
    }

    drop(db);
    // Still consistent after reopen.
    let db2 = Z1Kv::open(dir).unwrap();
    assert_eq!(
        db2.get(0, b"t7_49").unwrap(),
        Some(49u64.to_le_bytes().to_vec()),
        "reopen must preserve concurrently committed data"
    );
}

/// Regression (concurrent repeatable read): a begun transaction runs on its
/// pinned snapshot; new keys committed concurrently by a committer stay
/// invisible within the transaction (the snapshot is pinned at begin time).
/// (Note: the TOCTOU window itself is verified by the unit-level
/// discriminative test `begin_with_snapshot_pins_committed_txn` + a
/// revert-based check; this test locks in the end-to-end behavior.)
#[test]
fn concurrent_begin_pinned_repeatable_read() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = tmp_dir("begin_toctou");
    let db: std::sync::Arc<Z1Kv> = std::sync::Arc::new(Z1Kv::open(dir).unwrap());

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_c = stop.clone();
    let db_c = std::sync::Arc::clone(&db);

    // Committer: keeps committing new transactions. Being aborted on an SSI
    // read-set conflict with the reader is normal (WR conflict); skip that
    // key and continue — only an engine bug would panic.
    let committer = std::thread::spawn(move || {
        let mut i = 0u64;
        while !stop_c.load(Ordering::Relaxed) {
            let t = match db_c.begin_txn() {
                Ok(t) => t,
                Err(_) => break,
            };
            let key = format!("bg_{}", i);
            if db_c.put(0, key.as_bytes(), b"v".to_vec(), t).is_err() {
                let _ = db_c.rollback(t);
                i += 1;
                continue;
            }
            if db_c.commit(t).is_ok() {
                i += 1;
            } else {
                // SSI abort (conflict with the reader's read set) — normal;
                // continue with the next key.
                i += 1;
            }
        }
    });

    // Repeatable-read assertion: after t1 begins, the committer keeps
    // committing new transactions (writing bg_0..bg_N). Reads of those keys
    // inside t1 must all return None — the pinned snapshot is fixed at
    // begin time. Before the fix (TOCTOU): begin's registration and
    // snapshot capture were two steps; a concurrent commit in between
    // advanced committed_txn, so t1's snapshot was "fresher" and read values
    // from after begin.
    let t1 = db.begin_txn().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    for i in 0..20u64 {
        let key = format!("bg_{}", i);
        assert_eq!(
            db.get_for_txn(0, key.as_bytes(), t1).unwrap(),
            None,
            "pinned snapshot must not see commit {} made after begin",
            key
        );
    }

    // The current read can see the committer's commits (at least some bg_* exist).
    let visible_now = (0..20u64)
        .filter(|i| db.get(0, format!("bg_{}", i).as_bytes()).unwrap().is_some())
        .count();
    assert!(
        visible_now > 0,
        "committer must have made progress during the test"
    );

    // Stop the committer; t1 commits successfully.
    stop.store(true, Ordering::Relaxed);
    committer.join().unwrap();
    db.commit(t1).unwrap();
}

/// Coverage gap: write-side interleaving of flush and concurrent puts —
/// during the stress, each put's data is either in L1, in L2, or already
/// covered by a checkpoint, and everything is readable at the end.
/// (The read-side D8 window has its own test; this one locks in the write
/// side's atomicity under flush/swap interleaving.)
///
/// Regression (read-migration gate): a committed key became invisible in
/// all read sources when auto-compaction dropped the L2 index while a
/// reader was mid-`get_at`. The failure mode was: reader holds a
/// pre-compaction L2 index snapshot; compaction deletes those files and
/// publishes an L3 patch that does NOT contain the key (it was still in
/// L1, not yet flushed); a concurrent flush then migrates the key out of
/// L1 and clears the recent-flush cache. Every source appeared empty even
/// though the WAL was intact. The fix gates flush AND compaction against
/// multi-source reads via `RecentFlushCache`'s migration gate, and makes
/// `swap_and_drain` TAKE the hot buffer (serializing put vs migration) so
/// an entry can never fall between clone and swap.
#[test]
fn concurrent_put_during_flush_never_loses_data() {
    use std::sync::Arc as StdArc;

    let dir = tmp_dir("put_flush_race");
    let db: StdArc<Z1Kv> = StdArc::new(Z1Kv::open(dir.clone()).unwrap());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Flusher: medium-frequency flush_now (triggering swap_and_drain). The
    // frequency is limited by Windows filesystem rename latency under AV
    // interference (retries are built into write_durable); 1ms high frequency
    // makes the stress time out on disturbed environments.
    let stop_f = stop.clone();
    let db_f = StdArc::clone(&db);
    let flusher = std::thread::spawn(move || {
        while !stop_f.load(std::sync::atomic::Ordering::Relaxed) {
            db_f.flush_now()
                .expect("flush_now must not fail (would lose data)");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });

    // Writer: keeps committing. Immediately read-verify after each commit —
    // if the flush window loses data, get returns None.
    let mut committed_keys = Vec::new();
    for i in 0..150u64 {
        let t = db.begin_txn().unwrap();
        let key = format!("rk_{}", i);
        db.put(0, key.as_bytes(), i.to_le_bytes().to_vec(), t)
            .unwrap();
        db.commit(t).unwrap();
        // Read back immediately: committed data must be visible under any flush interleaving.
        let v = db.get(0, key.as_bytes()).unwrap();
        if v.is_none() {
            // Diagnostics: on failure, dump layer states + read disk patches directly.
            let _snap = db.snapshot();
            let mut patch_info = String::new();
            let l2_dir = dir.join("l2").join("0000");
            if let Ok(entries) = std::fs::read_dir(&l2_dir) {
                for e in entries.flatten() {
                    patch_info.push_str(&format!(
                        "
  patch: {}",
                        e.file_name().to_string_lossy()
                    ));
                }
            }
            // Read disk patches directly: parse all patches for rk_120.
            use z1kv::codec::disk_format::DiskFormat;
            let mut found_in_disk = false;
            if let Ok(entries) = std::fs::read_dir(&l2_dir) {
                for e in entries.flatten() {
                    let Ok(bytes) = std::fs::read(e.path()) else {
                        continue;
                    };
                    let Ok(fmt) = z1kv::store::Z1PatchFormatV4::from_disk_bytes(&bytes) else {
                        continue;
                    };
                    // The patch format currently has a single Sparse variant;
                    // destructure directly.
                    let z1kv::store::Z1PatchFormatV4::Sparse { entries: es } = fmt;
                    for pe in es {
                        if pe.key == key.as_bytes() {
                            found_in_disk = true;
                            patch_info.push_str(&format!(
                                "
  DISK FOUND key={:?} txn={} value={:?}",
                                pe.key, pe.txn_id, pe.value
                            ));
                        }
                    }
                }
            }
            // Distinguish transient vs permanent: retry the read, then re-read
            // after the flusher stops.
            let retry1 = db.get(0, key.as_bytes()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
            let retry2 = db.get(0, key.as_bytes()).unwrap();
            panic!(
                "lost key {} (txn {}) after commit: committed_entry={:?}                  retry1={:?} retry2={:?} disk_found={}{}",
                key, t, db.committed_entry(t), retry1, retry2, found_in_disk, patch_info,
            );
        }
        assert_eq!(
            v,
            Some(i.to_le_bytes().to_vec()),
            "committed key {} must be readable immediately (flush race)",
            key
        );
        committed_keys.push(key);
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    flusher.join().unwrap();

    // Full re-verification + reopen.
    for (i, key) in committed_keys.iter().enumerate() {
        assert_eq!(
            db.get(0, key.as_bytes()).unwrap(),
            Some((i as u64).to_le_bytes().to_vec())
        );
    }
    drop(db);
    let db2 = Z1Kv::open(dir).unwrap();
    assert_eq!(
        db2.get(0, b"rk_149").unwrap(),
        Some(149u64.to_le_bytes().to_vec())
    );
}

/// Aggressive migration-gate regression: a flusher hammers flush_now with
/// NO sleep while a compactor thread and a writer race; every committed key
/// must be readable immediately after its commit under any interleaving.
/// This pins the gate that serializes {flush, compaction} against
/// multi-source reads (`get_at`), plus the take-based `swap_and_drain`
/// that serializes put vs L1 migration. Without the gate this test loses
/// ~2 keys per 150 deterministically (the compaction/L2-index race).
#[test]
fn concurrent_flush_compact_read_gate() {
    use std::sync::atomic::{AtomicBool, Ordering as AO};
    use std::sync::Arc as StdArc;

    let dir = tmp_dir("gate_race");
    let db: StdArc<Z1Kv> = StdArc::new(Z1Kv::open(dir.clone()).unwrap());
    let stop = StdArc::new(AtomicBool::new(false));

    let stop_f = stop.clone();
    let db_f = StdArc::clone(&db);
    let flusher = std::thread::spawn(move || {
        while !stop_f.load(AO::Relaxed) {
            let _ = db_f.flush_now();
        }
    });

    let stop_c = stop.clone();
    let db_c = StdArc::clone(&db);
    let compactor = std::thread::spawn(move || {
        while !stop_c.load(AO::Relaxed) {
            let _ = db_c.compact(u64::MAX);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    let mut misses = 0;
    for i in 0..150u64 {
        let t = db.begin_txn().unwrap();
        let key = format!("rk_{}", i);
        db.put(0, key.as_bytes(), i.to_le_bytes().to_vec(), t).unwrap();
        db.commit(t).unwrap();
        let snap = db.snapshot();
        if db.get_at(&snap, 0, key.as_bytes()).unwrap().is_none() {
            misses += 1;
            eprintln!(
                "GATE MISS i={} txn={} snap_id={} committed={:?}",
                i, t, snap.snapshot_id, db.committed_entry(t)
            );
        }
    }

    stop.store(true, AO::Relaxed);
    flusher.join().unwrap();
    compactor.join().unwrap();
    assert_eq!(misses, 0, "committed keys must be readable under flush+compact race");
}

#[test]
fn stress_commit_crash_consistency() {
    let dir = tmp_dir("stress");
    {
        let db = Z1Kv::open(dir.clone()).unwrap();
        for i in 0..1000u64 {
            let txn = db.begin_txn().unwrap();
            let key = (i % 50).to_le_bytes(); // 50 keys overwritten repeatedly
            db.put(0, key.to_vec(), i.to_le_bytes().to_vec(), txn)
                .unwrap();
            db.commit(txn).unwrap();
            // Flush every 200 writes, simulating mixed states.
            if i % 200 == 0 {
                let _ = db.flush().unwrap();
            }
        }
    }

    let db = Z1Kv::open(dir.clone()).unwrap();
    for k in 0..50u64 {
        let key_bytes = k.to_le_bytes();
        // Final value = the last i that wrote this key (the largest i with i % 50 == k).
        let v = db
            .get(0, &key_bytes)
            .unwrap()
            .expect("key must exist after crash");
        assert_eq!(v.len(), 8, "value must be 8-byte txn id");
        let stored = u64::from_le_bytes(v.try_into().unwrap());
        let expected = (0..1000u64).rev().find(|&t| t % 50 == k).unwrap();
        assert_eq!(
            stored, expected,
            "key {} must hold latest committed value",
            k
        );
    }
}
