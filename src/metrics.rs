//! Minimal metrics module.
//!
//! Provides degradation counters for the severity layering in `error.rs`.
//! Currently a simple `AtomicU64` implementation; prometheus /
//! OpenTelemetry integration can be added later.
//!
//! # Usage
//!
//! ```text
//! Degradation point (inside an error-handling branch):
//!     INCREMENT_DEGRADE!("wal_replay_skip");
//!     tracing::warn!(metric = "wal_replay_skip", error = %e);
//!     continue;  // or abort, depending on strict_mode
//! ```
//!
//! # Easy-to-confuse points
//!
//! - Metric names are `snake_case` string literals.
//! - The same metric may be incremented from several code paths; counts add up.
//! - Counters are not persisted; they reset when the process restarts.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global metric counter table: `metric_name -> count`.
///
/// Implemented with a `static` + `Mutex`, zero dependencies. Only called at
/// degradation points, so the performance impact is negligible.
static DEGRADE_COUNTS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, AtomicU64>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Increment a degradation counter.
///
/// A macro (rather than a function) avoids copying the string literal on
/// every call site.
#[macro_export]
macro_rules! INCREMENT_DEGRADE {
    ($metric:expr) => {{
        let m: &'static str = $metric;
        let mut map = $crate::metrics::DEGRADE_COUNTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.entry(m)
            .or_insert(std::sync::atomic::AtomicU64::new(0))
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }};
}

pub use INCREMENT_DEGRADE;

/// Decide how a degraded error should be handled, given `strict_mode`.
///
/// # Return value
///
/// - `ControlFlow::Continue(_)`: the degradation stands; the caller should
///   skip the current item (e.g. `continue`).
/// - `ControlFlow::Break(_)`: the degradation was escalated to Fatal by
///   strict mode; the caller should abort.
pub fn handle_degrade(
    err: &crate::error::Z1Error,
    strict_mode: bool,
) -> std::ops::ControlFlow<(), ()> {
    use crate::error::Severity;
    use std::ops::ControlFlow;
    let sev = err.severity().with_strict_mode(strict_mode);
    if let Some(metric) = sev.metric() {
        INCREMENT_DEGRADE!(metric);
        tracing::warn!(metric = metric, error = %err, "degraded");
    }
    match sev {
        Severity::Fatal => ControlFlow::Break(()),
        _ => ControlFlow::Continue(()),
    }
}

/// Snapshot of all degradation counters.
///
/// Intended for diagnostics / monitoring endpoints. Returns
/// `Vec<(metric_name, count)>` sorted by count, descending.
pub fn snapshot() -> Vec<(&'static str, u64)> {
    let map = DEGRADE_COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<_> = map
        .iter()
        .map(|(k, v)| (*k, v.load(Ordering::Relaxed)))
        .collect();
    out.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    out
}

/// Reset all counters (test-only).
#[cfg(test)]
pub fn reset_for_tests() {
    let mut map = DEGRADE_COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    map.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Z1Error;

    #[test]
    fn increment_works() {
        reset_for_tests();
        INCREMENT_DEGRADE!("test_metric_a");
        INCREMENT_DEGRADE!("test_metric_a");
        INCREMENT_DEGRADE!("test_metric_b");
        let s = snapshot();
        // Parallel tests share the global counter table, so we cannot assume
        // only this test's metrics exist — assert by name (order-independent)
        // to avoid scheduling-order flakiness.
        let find = |name: &str| {
            s.iter()
                .find(|(m, _)| *m == name)
                .map(|(_, c)| *c)
                .unwrap_or(0)
        };
        assert_eq!(find("test_metric_a"), 2);
        assert_eq!(find("test_metric_b"), 1);
    }

    #[test]
    fn handle_degrade_strict_breaks() {
        reset_for_tests();
        let e = Z1Error::degrade("vis_data_unavailable", "x");
        use std::ops::ControlFlow;
        // strict=true + vis_data_unavailable (unrecoverable) → Break
        assert!(matches!(handle_degrade(&e, true), ControlFlow::Break(())));
        // strict=false → Continue
        assert!(matches!(
            handle_degrade(&e, false),
            ControlFlow::Continue(())
        ));
    }
}
