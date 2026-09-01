//! Timestamp utilities.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Last valid timestamp, used as the fallback when the wall clock goes
/// backwards. `AtomicU64` keeps hot-path reads lock-free.
static LAST_VALID_TS_MILLIS: AtomicU64 = AtomicU64::new(0);

/// Current UNIX timestamp in milliseconds.
///
/// If the wall clock goes backwards (NTP adjustment, VM suspend, ...), the
/// last valid timestamp is returned instead of panicking or going backwards.
pub fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            let ts = d.as_millis() as u64;
            LAST_VALID_TS_MILLIS.store(ts, Ordering::Relaxed);
            ts
        })
        .unwrap_or_else(|_| {
            tracing::warn!("Time went backwards, using last valid timestamp");
            LAST_VALID_TS_MILLIS.load(Ordering::Relaxed)
        })
}
