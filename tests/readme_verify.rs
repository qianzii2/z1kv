//! Verifies the README code examples compile and run correctly (guards
//! against documentation drift).
use z1kv::Z1Kv;

#[test]
fn readme_quickstart_example() {
    let dir = std::env::temp_dir().join(format!("z1kv_readme_{}", uuid::Uuid::new_v4()));
    // -- README example start --
    let db = Z1Kv::open(&dir).unwrap();

    let txn = db.begin_txn().unwrap();
    db.put(0, b"greeting", b"hello", txn).unwrap();
    db.commit(txn).unwrap();

    assert_eq!(db.get(0, b"greeting").unwrap(), Some(b"hello".to_vec()));

    let t1 = db.begin_txn().unwrap();
    assert_eq!(
        db.get_for_txn(0, b"greeting", t1).unwrap(),
        Some(b"hello".to_vec())
    );
    db.commit(t1).unwrap();

    let t2 = db.begin_txn().unwrap();
    db.delete(0, b"greeting", t2).unwrap();
    db.commit(t2).unwrap();
    assert_eq!(db.get(0, b"greeting").unwrap(), None);

    let rows = db.scan(0, b"a", Some(b"z")).unwrap();

    db.flush_now().unwrap();
    db.checkpoint().unwrap();
    let (cfs, _reclaimed) = db.compact(u64::MAX).unwrap();
    // -- README example end --
    let _ = (rows, cfs);
    drop(db); // release the engine lock first (Windows: open handles cannot be deleted), then clean up
    std::fs::remove_dir_all(&dir).ok();
}
