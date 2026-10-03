//! Scalar tables — the reference implementation. Every SIMD kernel in the
//! sibling modules must produce byte-identical results to these, including
//! the exact reject set on decode (tests in the parent module pin that).

use super::{ALPHABET, DECODE, HASH_B64_LEN, INVALID, SALT_B64_LEN};

/// MSB-first packing of `bytes` into alphabet chars; `out` is filled exactly.
fn encode_into(bytes: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), (bytes.len() * 8).div_ceil(6));
    let mut o = 0;
    let mut groups = bytes.chunks_exact(3);
    for g in &mut groups {
        let (b0, b1, b2) = (g[0], g[1], g[2]);
        out[o] = ALPHABET[(b0 >> 2) as usize];
        out[o + 1] = ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize];
        out[o + 2] = ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize];
        out[o + 3] = ALPHABET[(b2 & 0x3f) as usize];
        o += 4;
    }
    // Partial final group: one leftover byte contributes its 8 bits to two
    // chars, two leftover bytes to three; the spare low bits stay zero.
    match groups.remainder() {
        [b0] => {
            let b0 = *b0;
            out[o] = ALPHABET[(b0 >> 2) as usize];
            out[o + 1] = ALPHABET[((b0 & 0x03) << 4) as usize];
        }
        [b0, b1] => {
            let (b0, b1) = (*b0, *b1);
            out[o] = ALPHABET[(b0 >> 2) as usize];
            out[o + 1] = ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize];
            out[o + 2] = ALPHABET[((b1 & 0x0f) << 2) as usize];
        }
        _ => {}
    }
}

/// The 6-bit value of one char, or `Err` for a byte outside the alphabet.
fn decode_char(c: u8) -> Result<u8, ()> {
    match DECODE[c as usize] {
        INVALID => Err(()),
        v => Ok(v),
    }
}

/// Inverse of [`encode_into`]; `out` is filled exactly. The spare low bits of
/// a final partial group are discarded, matching every other bcrypt codec.
fn decode_into(s: &[u8], out: &mut [u8]) -> Result<(), ()> {
    debug_assert_eq!(out.len(), s.len() * 6 / 8);
    let mut o = 0;
    let mut groups = s.chunks_exact(4);
    for g in &mut groups {
        let v0 = decode_char(g[0])?;
        let v1 = decode_char(g[1])?;
        let v2 = decode_char(g[2])?;
        let v3 = decode_char(g[3])?;
        out[o] = (v0 << 2) | (v1 >> 4);
        out[o + 1] = ((v1 & 0x0f) << 4) | (v2 >> 2);
        out[o + 2] = ((v2 & 0x03) << 6) | v3;
        o += 3;
    }
    match groups.remainder() {
        [] => Ok(()),
        [c0, c1] => {
            let v0 = decode_char(*c0)?;
            let v1 = decode_char(*c1)?;
            out[o] = (v0 << 2) | (v1 >> 4);
            Ok(())
        }
        [c0, c1, c2] => {
            let v0 = decode_char(*c0)?;
            let v1 = decode_char(*c1)?;
            let v2 = decode_char(*c2)?;
            out[o] = (v0 << 2) | (v1 >> 4);
            out[o + 1] = ((v1 & 0x0f) << 4) | (v2 >> 2);
            Ok(())
        }
        // A lone leftover char carries only 6 bits — never a whole byte.
        _ => Err(()),
    }
}

/// Encode a 16-byte salt to its 22-char bcrypt base64 form.
pub fn encode_16(bytes: &[u8; 16]) -> [u8; SALT_B64_LEN] {
    let mut out = [0u8; SALT_B64_LEN];
    encode_into(bytes, &mut out);
    out
}

/// Encode the 23 hash bytes to their 31-char bcrypt base64 form.
pub fn encode_23(bytes: &[u8; 23]) -> [u8; HASH_B64_LEN] {
    let mut out = [0u8; HASH_B64_LEN];
    encode_into(bytes, &mut out);
    out
}

/// Decode exactly [`SALT_B64_LEN`] chars back to the 16-byte salt.
pub fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
    if s.len() != SALT_B64_LEN {
        return Err(());
    }
    let mut out = [0u8; 16];
    decode_into(s, &mut out)?;
    Ok(out)
}

/// Decode exactly [`HASH_B64_LEN`] chars back to the 23 hash bytes.
pub fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
    if s.len() != HASH_B64_LEN {
        return Err(());
    }
    let mut out = [0u8; 23];
    decode_into(s, &mut out)?;
    Ok(out)
}
