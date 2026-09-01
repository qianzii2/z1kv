//! Codec module — binary serialization utilities.
//!
//! - `disk_format` — uniform on-disk format framework (magic + version + CRC32)
//! - `timestamp` — monotonic timestamp clamping (never goes backwards)

pub mod disk_format;
pub mod timestamp;

pub use timestamp::current_timestamp_millis;
