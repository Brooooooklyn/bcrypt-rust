//! The bcrypt hash-string format: `$2v$cc$<22 salt chars><31 hash chars>`.
//!
//! A hash string is exactly 60 ASCII bytes: a `$2v$` version marker, two
//! decimal cost digits, a `$`, then the salt and hash fields in bcrypt's own
//! base64 dialect (see `crate::base64`). [`HashParts`] is the parsed form.
//! Its [`FromStr`](core::str::FromStr) is deliberately strict — a string
//! that does not decode exactly is rejected, never repaired — and its
//! [`Display`](core::fmt::Display) always emits the modern `$2b$` spelling.
//!
//! Parsing and formatting are stack-only; the two `String` helpers are the
//! module's only `alloc` affordances.

use core::fmt;

#[cfg(feature = "alloc")]
use alloc::string::String;

use crate::base64;
use crate::error::BcryptError;

/// Total length of a bcrypt hash string, in bytes.
const HASH_STRING_LEN: usize = 60;
/// Offset of the 22-char salt field.
const SALT_START: usize = 7;
/// Offset of the 31-char hash field, immediately after the salt.
const HASH_START: usize = SALT_START + base64::SALT_B64_LEN;

const _: () = assert!(
    HASH_START + base64::HASH_B64_LEN == HASH_STRING_LEN,
    "field offsets must tile the 60-byte hash string exactly"
);

/// The version marker in a bcrypt hash string: the `v` in `$2v$`.
///
/// Versions are **labels**: every variant accepted by this crate computes
/// the same, correct algorithm (see the crate-level "Variant semantics"
/// section). The historical `$2x$` / 8-bit-`$2a$` bug emulations are not
/// reproduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Version {
    /// `$2a$` — the original OpenBSD spelling.
    TwoA,
    /// `$2x$` — crypt_blowfish's marker for hashes computed with its
    /// sign-extension bug. Accepted on parse so databases containing it
    /// still verify; the correct algorithm is computed regardless.
    TwoX,
    /// `$2y$` — crypt_blowfish's post-fix spelling.
    TwoY,
    /// `$2b$` — the current spelling, and what this crate emits by default.
    TwoB,
}

impl Version {
    /// The marker byte as it appears at position 2 of a hash string.
    pub(crate) const fn marker(self) -> u8 {
        match self {
            Version::TwoA => b'a',
            Version::TwoX => b'x',
            Version::TwoY => b'y',
            Version::TwoB => b'b',
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Version::TwoA => "2a",
            Version::TwoX => "2x",
            Version::TwoY => "2y",
            Version::TwoB => "2b",
        })
    }
}

/// A parsed bcrypt hash: cost, raw salt, and the 23-byte hash payload.
///
/// Obtain one from a hash string with `"...".parse()`, or from hashing with
/// [`hash_with_result`](crate::hash_with_result) /
/// [`hash_with_salt`](crate::hash_with_salt). Formatting back is
/// [`Display`](core::fmt::Display) (always `$2b$`) or
/// [`format_for_version`](HashParts::format_for_version) /
/// [`write_for_version`](HashParts::write_for_version) for a chosen marker.
///
/// The version marker of a parsed string is **not** retained: versions are
/// labels (see [`Version`]), so re-formatting a parsed `$2a$` string yields
/// `$2b$` — the same behaviour as the `bcrypt` crate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HashParts {
    cost: u32,
    salt: [u8; 16],
    hash: [u8; 23],
}

impl HashParts {
    /// Assemble the parts. `cost` must already be validated to `4..=31` by
    /// the caller (`crate::core`); the parser validates it on the way in.
    pub(crate) fn new(cost: u32, salt: [u8; 16], hash: [u8; 23]) -> Self {
        debug_assert!((crate::core::MIN_COST..=crate::core::MAX_COST).contains(&cost));
        HashParts { cost, salt, hash }
    }

    /// The cost parameter: bcrypt's `log2(rounds)`, in `4..=31`.
    #[must_use]
    pub fn get_cost(&self) -> u32 {
        self.cost
    }

    /// The salt, encoded in bcrypt's base64 dialect (22 chars).
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn get_salt(&self) -> String {
        // The encoded form is pure ASCII by construction, so the lossy
        // conversion never actually replaces anything.
        String::from_utf8_lossy(&base64::encode_16(&self.salt)).into_owned()
    }

    /// The raw 16-byte salt.
    #[must_use]
    pub fn get_salt_raw(&self) -> [u8; 16] {
        self.salt
    }

    /// The raw 23-byte hash payload: the 24-byte bcrypt ciphertext with its
    /// last byte dropped, exactly as the string's hash field encodes it.
    ///
    /// This accessor is an **extension** over the `bcrypt` crate, which
    /// exposes the payload only as an encoded string.
    #[must_use]
    pub fn get_hash(&self) -> [u8; 23] {
        self.hash
    }

    /// The 60-byte hash string with the given version marker.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn format_for_version(&self, version: Version) -> String {
        String::from_utf8_lossy(&self.format_bytes(version)).into_owned()
    }

    /// Write the 60-byte hash string with the given version marker to `w`.
    pub fn write_for_version<W: fmt::Write>(
        &self,
        version: Version,
        w: &mut W,
    ) -> fmt::Result {
        let bytes = self.format_bytes(version);
        match core::str::from_utf8(&bytes) {
            Ok(s) => w.write_str(s),
            // format_bytes only ever emits ASCII, so this branch is
            // unreachable; report it as a formatting error, never a panic.
            Err(_) => Err(fmt::Error),
        }
    }

    /// The hash string as stack bytes — the shared core both public
    /// formatting wrappers build on.
    pub(crate) fn format_bytes(&self, version: Version) -> [u8; HASH_STRING_LEN] {
        let mut out = [0u8; HASH_STRING_LEN];
        out[0] = b'$';
        out[1] = b'2';
        out[2] = version.marker();
        out[3] = b'$';
        // Cost is 4..=31 by construction, so exactly two decimal digits.
        out[4] = b'0' + (self.cost / 10) as u8;
        out[5] = b'0' + (self.cost % 10) as u8;
        out[6] = b'$';
        out[SALT_START..HASH_START].copy_from_slice(&base64::encode_16(&self.salt));
        out[HASH_START..HASH_STRING_LEN].copy_from_slice(&base64::encode_23(&self.hash));
        out
    }
}

impl core::str::FromStr for HashParts {
    type Err = BcryptError;

    /// Strict parser. Rejects — with [`InvalidHash`](BcryptError::InvalidHash)
    /// naming the failing part — anything that is not exactly 60 bytes of
    /// `$2v$cc$<salt><hash>` with `v` one of `a`/`b`/`x`/`y`, two decimal
    /// cost digits in `4..=31`, and both fields valid bcrypt base64:
    ///
    /// * `"length"` — not exactly 60 bytes;
    /// * `"prefix"` — bad magic, unknown version marker, or missing `$`;
    /// * `"cost"` — non-decimal digits or a value outside `4..=31` (an
    ///   out-of-range cost *in a string* is a parse error;
    ///   [`CostNotAllowed`](BcryptError::CostNotAllowed) is reserved for cost
    ///   *arguments* to hash functions, matching the `bcrypt` crate);
    /// * `"salt"` / `"hash"` — a character outside the bcrypt alphabet.
    ///
    /// Every check compares bytes, so a non-ASCII string fails the field it
    /// lands in; no separate ASCII pass is needed.
    fn from_str(s: &str) -> Result<Self, BcryptError> {
        let b = s.as_bytes();
        if b.len() != HASH_STRING_LEN {
            return Err(BcryptError::InvalidHash("length"));
        }
        if b[0] != b'$' || b[1] != b'2' || b[3] != b'$' {
            return Err(BcryptError::InvalidHash("prefix"));
        }
        match b[2] {
            // The marker is validated but not stored: versions are labels.
            b'a' | b'b' | b'x' | b'y' => {}
            _ => return Err(BcryptError::InvalidHash("prefix")),
        }
        if !b[4].is_ascii_digit() || !b[5].is_ascii_digit() {
            return Err(BcryptError::InvalidHash("cost"));
        }
        let cost = u32::from(b[4] - b'0') * 10 + u32::from(b[5] - b'0');
        if !(crate::core::MIN_COST..=crate::core::MAX_COST).contains(&cost) {
            return Err(BcryptError::InvalidHash("cost"));
        }
        if b[6] != b'$' {
            return Err(BcryptError::InvalidHash("prefix"));
        }
        let salt = base64::decode_16(&b[SALT_START..HASH_START])
            .map_err(|()| BcryptError::InvalidHash("salt"))?;
        let hash = base64::decode_23(&b[HASH_START..HASH_STRING_LEN])
            .map_err(|()| BcryptError::InvalidHash("hash"))?;
        Ok(HashParts { cost, salt, hash })
    }
}

impl fmt::Display for HashParts {
    /// The 60-byte hash string in the default `$2b$` spelling.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_for_version(Version::TwoB, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenBSD's own vector, `$2a$05$CCCC…` — 60 bytes.
    const GOOD: &str = "$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW";

    /// A minimal `fmt::Write` over a stack buffer, so Display and
    /// `write_for_version` can be tested without `alloc`.
    struct StackWriter<'a> {
        buf: &'a mut [u8],
        len: usize,
    }

    impl fmt::Write for StackWriter<'_> {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            let end = self.len + s.len();
            if end > self.buf.len() {
                return Err(fmt::Error);
            }
            self.buf[self.len..end].copy_from_slice(s.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    fn rejection_part(s: &str) -> &'static str {
        match s.parse::<HashParts>() {
            Err(BcryptError::InvalidHash(part)) => part,
            other => panic!("expected InvalidHash, got {other:?}"),
        }
    }

    #[test]
    fn parses_all_four_prefixes_identically() {
        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        for marker in *b"abxy" {
            let mut bytes = [0u8; HASH_STRING_LEN];
            bytes.copy_from_slice(GOOD.as_bytes());
            bytes[2] = marker;
            let s = core::str::from_utf8(&bytes).expect("still ASCII");
            assert_eq!(s.parse::<HashParts>(), Ok(parts.clone()));
        }
    }

    #[test]
    fn getters_return_the_vector_fields() {
        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        assert_eq!(parts.get_cost(), 5);
        assert_eq!(
            parts.get_salt_raw(),
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("salt decodes")
        );
        assert_eq!(
            parts.get_hash(),
            base64::decode_23(b"E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW").expect("hash decodes")
        );
    }

    #[test]
    fn format_bytes_round_trips_through_the_parser() {
        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        for (version, prefix) in [
            (Version::TwoA, "$2a$"),
            (Version::TwoB, "$2b$"),
            (Version::TwoX, "$2x$"),
            (Version::TwoY, "$2y$"),
        ] {
            let bytes = parts.format_bytes(version);
            let s = core::str::from_utf8(&bytes).expect("format_bytes emits ASCII");
            assert_eq!(s.len(), HASH_STRING_LEN);
            assert!(s.starts_with(prefix));
            assert_eq!(s.parse::<HashParts>(), Ok(parts.clone()));
        }
    }

    #[test]
    fn display_defaults_to_2b() {
        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        let expected = parts.format_bytes(Version::TwoB);
        let mut buf = [0u8; HASH_STRING_LEN];
        let mut w = StackWriter {
            buf: &mut buf,
            len: 0,
        };
        fmt::write(&mut w, format_args!("{parts}")).expect("Display fits the buffer");
        assert_eq!(w.len, HASH_STRING_LEN);
        assert_eq!(w.buf, &expected);
        assert!(expected.starts_with(b"$2b$"));
    }

    #[test]
    fn write_for_version_matches_format_bytes() {
        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        for version in [Version::TwoA, Version::TwoB, Version::TwoX, Version::TwoY] {
            let mut buf = [0u8; HASH_STRING_LEN];
            let mut w = StackWriter {
                buf: &mut buf,
                len: 0,
            };
            parts
                .write_for_version(version, &mut w)
                .expect("writing to a stack buffer fits");
            assert_eq!(w.len, HASH_STRING_LEN);
            assert_eq!(w.buf, &parts.format_bytes(version));
        }
    }

    #[test]
    fn parser_rejection_matrix() {
        // Length: 59 and 61 bytes, and the empty string.
        assert_eq!(rejection_part(""), "length");
        assert_eq!(rejection_part(&GOOD[..59]), "length");
        let mut longer = [0u8; 61];
        longer[..60].copy_from_slice(GOOD.as_bytes());
        longer[60] = b'W';
        let longer = core::str::from_utf8(&longer).expect("ASCII");
        assert_eq!(rejection_part(longer), "length");

        // Prefix: bad magic, unknown marker, and missing `$` at index 3 or 6.
        for bad in [
            "$1a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            "$2z$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            "$2aX05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            "$2a$05XCCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
        ] {
            assert_eq!(rejection_part(bad), "prefix", "{bad}");
        }

        // Cost: non-digit, below the floor, above the ceiling.
        for bad in [
            "$2a$0x$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            "$2a$03$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            "$2a$32$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
        ] {
            assert_eq!(rejection_part(bad), "cost", "{bad}");
        }

        // Salt/hash fields: one byte outside the bcrypt alphabet each.
        assert_eq!(
            rejection_part("$2a$05$CCC!CCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW"),
            "salt"
        );
        assert_eq!(
            rejection_part("$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOe*"),
            "hash"
        );

        // A non-ASCII byte inside the salt field fails the salt check, not
        // the length check: "é" is 2 bytes, so the string stays 60 bytes.
        let non_ascii = "$2a$05$éCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW";
        assert_eq!(non_ascii.len(), HASH_STRING_LEN);
        assert_eq!(rejection_part(non_ascii), "salt");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn alloc_helpers_agree_with_the_stack_core() {
        use alloc::string::{String, ToString};

        let parts: HashParts = GOOD.parse().expect("the OpenBSD vector parses");
        assert_eq!(parts.get_salt(), "CCCCCCCCCCCCCCCCCCCCC.");
        assert_eq!(parts.format_for_version(Version::TwoA), GOOD);
        // Display normalises any parsed marker to $2b$.
        assert_eq!(parts.to_string().len(), HASH_STRING_LEN);
        assert!(parts.to_string().starts_with("$2b$"));

        let mut s = String::new();
        parts
            .write_for_version(Version::TwoB, &mut s)
            .expect("writing to a String is infallible");
        assert_eq!(s, parts.format_for_version(Version::TwoB));
    }

    #[test]
    fn version_display_and_marker_agree() {
        for (version, text, marker) in [
            (Version::TwoA, "2a", b'a'),
            (Version::TwoX, "2x", b'x'),
            (Version::TwoY, "2y", b'y'),
            (Version::TwoB, "2b", b'b'),
        ] {
            assert_eq!(version.marker(), marker);
            let mut buf = [0u8; 2];
            let mut w = StackWriter {
                buf: &mut buf,
                len: 0,
            };
            fmt::write(&mut w, format_args!("{version}")).expect("two bytes fit");
            assert_eq!(w.len, 2);
            assert_eq!(w.buf, text.as_bytes());
        }
    }
}
