//! Error and result types for the crate.
//!
//! The variant set mirrors the de-facto standard [`bcrypt`
//! crate](https://crates.io/crates/bcrypt) (0.19) so switching crates is a
//! mechanical change; the one deliberate difference is the payload of
//! [`BcryptError::Rand`], which is this crate's own [`EntropyError`] because
//! OS entropy here is hand-declared rather than pulled from `getrandom`.

use core::fmt;

/// Convenience alias mirroring the `bcrypt` crate.
pub type BcryptResult<T> = Result<T, BcryptError>;

/// Everything that can go wrong in the public API.
///
/// No variant carries attacker-controlled content, so `Display` is safe to
/// log on an authentication path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BcryptError {
    /// Cost outside the allowed range `4..=31` (bcrypt's `log2(rounds)`).
    CostNotAllowed(u32),
    /// The hash string is malformed; the payload names the failing part.
    InvalidHash(&'static str),
    /// A `non_truncating_*` function received a password of 72 bytes or more.
    /// The payload is the input length including the terminator bcrypt appends.
    Truncation(usize),
    /// The OS CSPRNG could not produce a salt.
    Rand(EntropyError),
    /// A batch function received `passwords` and `salts` slices of different
    /// lengths. Batch outputs are position-preserving, so a mismatch has no
    /// meaningful result and is rejected up front.
    ///
    /// In [`verify_many`](crate::verify_many), which takes `hashes` rather
    /// than `salts`, the `salts` field reports the `hashes` length.
    BatchLengthMismatch {
        /// Number of passwords passed.
        passwords: usize,
        /// Number of salts (or hashes, for `verify_many`) passed.
        salts: usize,
    },
}

/// Failure of the OS entropy source.
///
/// Wraps the platform error code where one exists (`errno` on Unix, the
/// `NTSTATUS`-style status on Windows, the WASI `errno` on wasm); `None` means
/// the platform reported failure without a code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropyError {
    code: Option<i32>,
}

impl EntropyError {
    // Only `crate::random` (std-gated) ever constructs an entropy error;
    // without `std` this constructor is legitimately unused.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) const fn new(code: Option<i32>) -> Self {
        EntropyError { code }
    }

    /// The platform error code, if the platform provided one.
    #[inline]
    #[must_use]
    pub const fn code(self) -> Option<i32> {
        self.code
    }
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "OS entropy source failed (code {code})"),
            None => f.write_str("OS entropy source failed"),
        }
    }
}

impl fmt::Display for BcryptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BcryptError::CostNotAllowed(cost) => {
                write!(f, "cost {cost} not allowed: must be in 4..=31")
            }
            BcryptError::InvalidHash(part) => write!(f, "invalid hash: {part}"),
            BcryptError::Truncation(len) => {
                write!(f, "password is {len} bytes; bcrypt truncates at 72")
            }
            BcryptError::Rand(err) => write!(f, "{err}"),
            BcryptError::BatchLengthMismatch { passwords, salts } => {
                write!(
                    f,
                    "batch length mismatch: {passwords} passwords but {salts} salts"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EntropyError {}

#[cfg(feature = "std")]
impl std::error::Error for BcryptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BcryptError::Rand(err) => Some(err),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // `no_std`: `ToString` is not in the prelude, so pull it from `alloc`.
    use alloc::string::ToString;

    #[test]
    fn display_is_informative() {
        assert_eq!(
            BcryptError::CostNotAllowed(3).to_string(),
            "cost 3 not allowed: must be in 4..=31"
        );
        assert!(BcryptError::Truncation(73).to_string().contains("72"));
        assert_eq!(
            BcryptError::BatchLengthMismatch {
                passwords: 3,
                salts: 2
            }
            .to_string(),
            "batch length mismatch: 3 passwords but 2 salts"
        );
        assert_eq!(EntropyError::new(Some(1)).code(), Some(1));
        assert!(EntropyError::new(None).to_string().contains("failed"));
    }
}
