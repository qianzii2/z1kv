//! DiskFormat — a uniform on-disk format framework.
//!
//! # Motivation
//!
//! An early review of this codebase found 18 on-disk serialization points
//! with wildly inconsistent protection levels:
//! - 11 used raw postcard/bincode serialization (no magic / version / CRC)
//! - 5 had full magic + version + CRC protection
//! - 2 had magic but no version (e.g. `DeltaPatchFormat`)
//!
//! Without a uniform evolution mechanism, every added field would require
//! hand-written "old format fallback" boilerplate, and CRC checks were easy
//! to forget. With this trait:
//!
//! 1. Every persisted struct implementing `DiskFormat` gets
//!    magic + version + CRC protection for free.
//! 2. A format upgrade only needs a `VERSION` bump; the framework
//!    automatically rejects files an old reader cannot understand.
//! 3. `MIN_READABLE_VERSION` lets old readers accept new files (only for
//!    backward-compatible field additions).
//!
//! # Invariants
//!
//! - `MAGIC` is 6 bytes
//! - Fixed 18-byte header: `magic(6) + version(4) + crc32(4) + payload_len(4)`
//! - CRC32 covers the whole payload, not the header
//! - Payload encoding: `postcard::to_allocvec(self)` (serde-compatible)
//!
//! # Test matrix (every implementer must cover)
//!
//! - An old v1 file must be readable by a v2 reader (via
//!   `MIN_READABLE_VERSION` compatibility)
//! - A new v2 file must be rejected by a v1 reader (magic/version mismatch)
//! - A half-written file (truncated at magic / version / crc / len / payload)
//!   must return a clear error
//! - A CRC error (1-bit flip in the payload) must return a clear error
//!
//! # Easy-to-confuse points
//!
//! - Do **not** wrap already stream-friendly formats (WAL / version logs)
//!   in `DiskFormat`.
//! - Do **not** put a version number inside `MAGIC`; version is a separate
//!   header field.
//! - `MIN_READABLE_VERSION` must be `<= VERSION`, otherwise old readers
//!   deadlock against new files.

use crate::error::{Result, Z1Error};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Uniform on-disk format trait. Every persisted struct must implement it.
///
/// # Implementation example
///
/// ```
/// use z1kv::codec::disk_format::DiskFormat;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct MyFormatV2 {
///     field_a: u32,
///     field_b: String,  // new field in v2
/// }
///
/// impl DiskFormat for MyFormatV2 {
///     const MAGIC: &'static [u8; 6] = b"MYF002";  // b"MYF" + v2 tag
///     const VERSION: u32 = 2;
///     const MIN_READABLE_VERSION: u32 = 2;
/// }
///
/// let v = MyFormatV2 { field_a: 1, field_b: "x".into() };
/// let bytes = v.to_disk_bytes().unwrap();
/// assert!(MyFormatV2::matches_magic(&bytes));
/// ```
pub trait DiskFormat: Serialize + DeserializeOwned + Sized {
    /// 6-byte magic. Convention: the first 3 bytes are the format-name
    /// abbreviation, the last 3 bytes are the version tag (e.g. `002`).
    const MAGIC: &'static [u8; 6];
    /// Current version number. Bump when adding fields; new formats start at v1.
    const VERSION: u32;
    /// Oldest readable version. Old readers only accept files with
    /// `version >= MIN_READABLE_VERSION`.
    const MIN_READABLE_VERSION: u32;

    /// Fixed header size: `magic(6) + version(4) + crc32(4) + payload_len(4)`
    /// = 18 bytes.
    const HEADER_SIZE: usize = 6 + 4 + 4 + 4;

    /// Serialize into a byte stream with magic + version + crc + payload.
    ///
    /// Default implementation: postcard-encoded payload + crc32fast check +
    /// magic/version header.
    fn to_disk_bytes(&self) -> Result<Vec<u8>> {
        let payload = postcard::to_allocvec(self)
            .map_err(|e| Z1Error::Serialization(format!("postcard encode: {}", e)))?;
        if payload.len() > u32::MAX as usize {
            // Fix: `payload_len` is a u32 field. A silent `as u32` truncation
            // would write a file whose declared length differs from the actual
            // payload (the reader would then fail CRC/length checks) — fail
            // loudly at the write boundary instead.
            return Err(Z1Error::Serialization(format!(
                "payload too large for DiskFormat: {} bytes (u32 limit exceeded)",
                payload.len()
            )));
        }
        let payload_crc = crc32fast::hash(&payload);
        let payload_len = payload.len() as u32;

        let mut out = Vec::with_capacity(Self::HEADER_SIZE + payload.len());
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&payload_crc.to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Deserialize from a byte stream. The framework validates
    /// magic / version / CRC.
    ///
    /// # Errors
    ///
    /// - `Deserialize("too short")` — byte stream shorter than the header
    /// - `Deserialize("magic mismatch")` — magic does not match
    /// - `Deserialize("unsupported version X (need Y-Z)")` — incompatible version
    /// - `Deserialize("truncated")` — `payload_len` exceeds the actual stream
    /// - `Deserialize("crc mismatch: stored=X actual=Y")` — CRC check failed
    fn from_disk_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(Z1Error::Deserialize(format!(
                "DiskFormat({}): too short ({} < {} header bytes)",
                std::str::from_utf8(Self::MAGIC).unwrap_or("??"),
                bytes.len(),
                Self::HEADER_SIZE
            )));
        }
        if &bytes[..6] != Self::MAGIC {
            return Err(Z1Error::Deserialize(format!(
                "DiskFormat magic mismatch: expected {:?}, got {:?}",
                std::str::from_utf8(Self::MAGIC).unwrap_or("??"),
                std::str::from_utf8(&bytes[..6]).unwrap_or("??")
            )));
        }

        // Version parsing cannot fail (length already validated).
        let version = u32::from_le_bytes(
            bytes[6..10]
                .try_into()
                .expect("HEADER_SIZE guarantees 10 bytes available"),
        );
        if version < Self::MIN_READABLE_VERSION || version > Self::VERSION {
            return Err(Z1Error::Deserialize(format!(
                "DiskFormat unsupported version {} (need {}-{})",
                version,
                Self::MIN_READABLE_VERSION,
                Self::VERSION
            )));
        }

        let stored_crc = u32::from_le_bytes(
            bytes[10..14]
                .try_into()
                .expect("HEADER_SIZE guarantees 14 bytes available"),
        );
        let payload_len = u32::from_le_bytes(
            bytes[14..18]
                .try_into()
                .expect("HEADER_SIZE guarantees 18 bytes available"),
        ) as usize;
        // Defense: `checked_add` prevents a usize overflow on 32-bit targets
        // (HEADER_SIZE + u32::MAX could bypass the length check and cause a
        // slicing panic). Equivalent on 64-bit targets.
        let total_expected = match Self::HEADER_SIZE.checked_add(payload_len) {
            Some(v) => v,
            None => {
                return Err(Z1Error::Deserialize(format!(
                    "DiskFormat payload_len {} overflows",
                    payload_len
                )))
            }
        };
        if bytes.len() < total_expected {
            return Err(Z1Error::Deserialize(format!(
                "DiskFormat truncated: header says {} bytes, only {} available",
                total_expected,
                bytes.len()
            )));
        }

        let payload = &bytes[Self::HEADER_SIZE..total_expected];
        let actual_crc = crc32fast::hash(payload);
        if actual_crc != stored_crc {
            return Err(Z1Error::Deserialize(format!(
                "DiskFormat crc mismatch: stored={:08x} actual={:08x}",
                stored_crc, actual_crc
            )));
        }

        postcard::from_bytes(payload)
            .map_err(|e| Z1Error::Deserialize(format!("postcard decode: {}", e)))
    }

    /// Probe whether a byte stream belongs to this format (magic check only;
    /// neither CRC nor version is validated).
    ///
    /// Useful when the on-disk file type is unknown (e.g. cold start).
    fn matches_magic(bytes: &[u8]) -> bool {
        bytes.len() >= 6 && &bytes[..6] == Self::MAGIC
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestV1 {
        a: u32,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestV2 {
        a: u32,
        b: String,
    }

    impl DiskFormat for TestV1 {
        const MAGIC: &'static [u8; 6] = b"TST001";
        const VERSION: u32 = 1;
        const MIN_READABLE_VERSION: u32 = 1;
    }

    impl DiskFormat for TestV2 {
        const MAGIC: &'static [u8; 6] = b"TST002";
        const VERSION: u32 = 2;
        const MIN_READABLE_VERSION: u32 = 1; // can read v1
    }

    #[test]
    fn roundtrip_v1() {
        let v = TestV1 { a: 42 };
        let bytes = v.to_disk_bytes().unwrap();
        let parsed = TestV1::from_disk_bytes(&bytes).unwrap();
        assert_eq!(v, parsed);
    }

    #[test]
    fn roundtrip_v2() {
        let v = TestV2 {
            a: 42,
            b: "hello".to_string(),
        };
        let bytes = v.to_disk_bytes().unwrap();
        let parsed = TestV2::from_disk_bytes(&bytes).unwrap();
        assert_eq!(v, parsed);
    }

    #[test]
    fn magic_mismatch_rejected() {
        let v = TestV1 { a: 42 };
        let bytes = v.to_disk_bytes().unwrap();
        // Wrong magic: TestV2's magic.
        let mut wrong = bytes.clone();
        wrong[..6].copy_from_slice(b"TST002");
        let r = TestV1::from_disk_bytes(&wrong);
        assert!(r.is_err());
    }

    #[test]
    fn truncated_rejected() {
        let v = TestV1 { a: 42 };
        let bytes = v.to_disk_bytes().unwrap();
        let truncated = &bytes[..bytes.len() - 2];
        let r = TestV1::from_disk_bytes(truncated);
        assert!(r.is_err());
    }

    #[test]
    fn crc_corruption_rejected() {
        let v = TestV1 { a: 42 };
        let mut bytes = v.to_disk_bytes().unwrap();
        // Flip the last payload byte.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let r = TestV1::from_disk_bytes(&bytes);
        assert!(r.is_err());
    }

    #[test]
    fn version_mismatch_rejected() {
        // Hand-craft a v99 header.
        let mut bytes = vec![];
        bytes.extend_from_slice(b"TST001");
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let r = TestV1::from_disk_bytes(&bytes);
        assert!(r.is_err());
    }

    #[test]
    fn too_short_rejected() {
        let r = TestV1::from_disk_bytes(b"abc");
        assert!(r.is_err());
    }

    #[test]
    fn matches_magic_works() {
        let v = TestV1 { a: 42 };
        let bytes = v.to_disk_bytes().unwrap();
        assert!(TestV1::matches_magic(&bytes));
        assert!(!TestV2::matches_magic(&bytes));
    }
}
