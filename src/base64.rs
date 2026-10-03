//! bcrypt's base64 dialect, stack-only.
//!
//! Bit packing is MSB-first exactly like RFC 4648; only the alphabet
//! (`./A-Za-z0-9`, in that order) and the absence of padding differ. bcrypt
//! uses exactly two payload sizes — 16 salt bytes in 22 chars, 23 hash bytes
//! in 31 — so the API is fixed-size arrays and never allocates.
//!
//! Neither 16 nor 23 bytes is a multiple of 3, so the final char of each
//! encoding carries spare low bits. The encoder zeroes them (canonical
//! form); the decoder ignores them, which is why decoding is only defined on
//! the exact lengths below.
//!
//! # Dispatch
//!
//! The codec is the only data-parallel step outside the Blowfish core, so
//! it carries SIMD kernels of its own: aarch64 always runs NEON (the
//! feature is mandatory on the target), x86_64 picks AVX2 then SSSE3 at
//! runtime on `std` builds (compile-time `target_feature` cfgs otherwise),
//! wasm32 runs the v128 kernel under `simd128`, and everything else runs
//! the scalar tables. Every path is byte-exact
//! identical, reject set included — the differential tests below pin that.
//! Measured share of one hash: ~20 ns against 163 µs at cost 4 (0.012%),
//! so this is completeness of the SIMD story, not a throughput play.

/// The bcrypt alphabet: index 0 is `'.'`, 1 is `'/'`, then `A`–`Z`, `a`–`z`,
/// `0`–`9` — *not* the RFC 4648 order.
pub(crate) const ALPHABET: &[u8; 64] =
    b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Encoded length of a 16-byte salt.
pub(crate) const SALT_B64_LEN: usize = 22;
/// Encoded length of a 23-byte hash payload (the 24-byte bcrypt ciphertext
/// with its last byte dropped).
pub(crate) const HASH_B64_LEN: usize = 31;

/// Marker in [`DECODE`] for bytes outside the alphabet.
pub(crate) const INVALID: u8 = 0xFF;

/// Byte → 6-bit value, built at compile time.
pub(crate) const DECODE: [u8; 256] = {
    let mut table = [INVALID; 256];
    let mut i = 0;
    while i < 64 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// The scalar tables — the reference every SIMD kernel is checked against.
/// `pub` only so `__internal` (feature `internal-api`) can bench it; the
/// module is crate-private without that feature.
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm128;

/// Encode a 16-byte salt to its 22-char bcrypt base64 form.
pub(crate) fn encode_16(bytes: &[u8; 16]) -> [u8; SALT_B64_LEN] {
    #[cfg(target_arch = "aarch64")]
    return neon::encode_16(bytes);
    #[cfg(target_arch = "x86_64")]
    return x86::encode_16(bytes);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return wasm128::encode_16(bytes);
    #[allow(unreachable_code)]
    scalar::encode_16(bytes)
}

/// Encode the 23 hash bytes to their 31-char bcrypt base64 form.
pub(crate) fn encode_23(bytes: &[u8; 23]) -> [u8; HASH_B64_LEN] {
    #[cfg(target_arch = "aarch64")]
    return neon::encode_23(bytes);
    #[cfg(target_arch = "x86_64")]
    return x86::encode_23(bytes);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return wasm128::encode_23(bytes);
    #[allow(unreachable_code)]
    scalar::encode_23(bytes)
}

/// Decode exactly [`SALT_B64_LEN`] chars back to the 16-byte salt.
pub(crate) fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
    #[cfg(target_arch = "aarch64")]
    return neon::decode_16(s);
    #[cfg(target_arch = "x86_64")]
    return x86::decode_16(s);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return wasm128::decode_16(s);
    #[allow(unreachable_code)]
    scalar::decode_16(s)
}

/// Decode exactly [`HASH_B64_LEN`] chars back to the 23 hash bytes.
pub(crate) fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
    #[cfg(target_arch = "aarch64")]
    return neon::decode_23(s);
    #[cfg(target_arch = "x86_64")]
    return x86::decode_23(s);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return wasm128::decode_23(s);
    #[allow(unreachable_code)]
    scalar::decode_23(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* — a deterministic byte source for round-trips, so the
    /// tests need no rand crate and no alloc.
    struct Rng(u64);

    impl Rng {
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                *b = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8;
            }
        }
    }

    #[test]
    fn salt_cross_check() {
        // The salt half of a well-known John the Ripper test vector.
        let bytes: [u8; 16] = [
            38, 113, 212, 141, 108, 213, 195, 166, 201, 38, 20, 13, 47, 40, 104, 18,
        ];
        assert_eq!(&encode_16(&bytes), b"HlFShUxTu4ZHHfOLJwfmCe");
        assert_eq!(decode_16(b"HlFShUxTu4ZHHfOLJwfmCe"), Ok(bytes));
    }

    #[test]
    fn round_trip_16_and_23() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut b16 = [0u8; 16];
        let mut b23 = [0u8; 23];
        for _ in 0..256 {
            rng.fill(&mut b16);
            rng.fill(&mut b23);
            assert_eq!(decode_16(&encode_16(&b16)), Ok(b16));
            assert_eq!(decode_23(&encode_23(&b23)), Ok(b23));
        }
    }

    #[test]
    fn encode_16_last_char_is_structurally_constrained() {
        // 16 bytes = 128 bits; the 22nd char carries only the top 2 of its
        // 6 bits, so its alphabet index is one of 0, 16, 32, 48.
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        let mut bytes = [0u8; 16];
        for _ in 0..64 {
            rng.fill(&mut bytes);
            let last = encode_16(&bytes)[SALT_B64_LEN - 1];
            assert!(matches!(last, b'.' | b'O' | b'e' | b'u'));
        }
    }

    #[test]
    fn encode_23_last_char_index_is_multiple_of_4() {
        // 23 bytes = 184 bits; the 31st char carries only the top 4 of its
        // 6 bits, so its alphabet index is 0 mod 4.
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        let mut bytes = [0u8; 23];
        for _ in 0..64 {
            rng.fill(&mut bytes);
            let last = encode_23(&bytes)[HASH_B64_LEN - 1];
            assert_eq!(DECODE[last as usize] % 4, 0);
        }
    }

    #[test]
    fn decode_rejects_bad_chars_and_wrong_lengths() {
        // '!' and '*' are outside the bcrypt alphabet.
        assert!(decode_16(b"CCCCCCCCCCCCCCCCCCCCC!").is_err());
        assert!(decode_23(b"7uG0VCzI2bS7j6ymqJi9CdcdxiRTWN*").is_err());
        // One char short, one char long, and a wildly wrong length.
        assert!(decode_16(b"CCCCCCCCCCCCCCCCCCCCC").is_err());
        assert!(decode_16(b"CCCCCCCCCCCCCCCCCCCCC..").is_err());
        assert!(decode_23(b"7uG0VCzI2bS7j6ymqJi9CdcdxiRTWN").is_err());
        assert!(decode_23(&[b'C'; 32]).is_err());
        assert!(decode_16(&[]).is_err());
    }

    /// The dispatched path (SIMD where compiled) must be byte-exact against
    /// the scalar tables — outputs AND the reject set. Exhaustive over every
    /// byte value at every position, plus random whole-string round-trips.
    #[test]
    fn dispatch_matches_scalar() {
        let mut rng = Rng(0xB529_7A4D_1D2B_9F83);
        let mut b16 = [0u8; 16];
        let mut b23 = [0u8; 23];
        for _ in 0..512 {
            rng.fill(&mut b16);
            rng.fill(&mut b23);
            assert_eq!(encode_16(&b16), scalar::encode_16(&b16));
            assert_eq!(encode_23(&b23), scalar::encode_23(&b23));
        }
        let base16 = *b"CCCCCCCCCCCCCCCCCCCCCC";
        let base23 = *b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";
        for pos in 0..SALT_B64_LEN {
            for c in 0..=u8::MAX {
                let mut s = base16;
                s[pos] = c;
                assert_eq!(decode_16(&s), scalar::decode_16(&s), "16: pos {pos} c {c:#04x}");
            }
        }
        for pos in 0..HASH_B64_LEN {
            for c in 0..=u8::MAX {
                let mut s = base23;
                s[pos] = c;
                assert_eq!(decode_23(&s), scalar::decode_23(&s), "23: pos {pos} c {c:#04x}");
            }
        }
    }
}
