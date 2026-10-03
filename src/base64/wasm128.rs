//! wasm v128 kernel for the bcrypt base64 dialect, compiled only under
//! `simd128` — the module `cfg` in `super`, so the whole crate build has
//! the feature and the dispatch in `super` is a compile-time choice
//! (WebAssembly has no survivable runtime feature detection; see the
//! `eks::wasm128` docs). The functions below are therefore plain safe
//! fns; the only unsafe ops are `v128_load`/`v128_store`, each carrying
//! a `// SAFETY:` comment with its in-bounds argument.
//!
//! Both payload sizes run one uniform shape: pad the input into a stack
//! buffer, process two 16-byte vectors, cut the result out of a 32-byte
//! scratch. The padding is what keeps the vector path byte-exact with
//! the scalar tables:
//!
//! * Encode zero-pads the 16/23 real bytes to 24, so every 3-byte group
//!   is complete and the spare low bits of the final kept char read zero
//!   bytes — the canonical form falls out of the uniform group math,
//!   exactly like the scalar tail handling.
//! * Decode pads the 22/31 chars to 24/32 with `'.'` (alphabet value 0),
//!   whose bits land only in output bytes the final copy drops — the
//!   scalar decoder's "spare bits ignored" behavior, since those bits
//!   never reach a kept byte.
//!
//! 6-bit expansion (encode) works on 16-bit lanes: one shuffle builds
//! the overlapping byte pairs `[b0,b1, b1,b2, b3,b4, b4,b5, ...]` (each
//! output char spans at most two input bytes), two shift/mask vectors
//! extract the 6-bit fields, and one shuffle merges them into char
//! order. Packing (decode) is the mirror image on 32-bit lanes: six
//! shift/mask terms per dword, then a compaction shuffle.
//!
//! Value <-> char translation is compare-and-select, not a LUT:
//! `i8x16_swizzle` is only a 16-entry table, so range compares plus
//! `v128_bitselect` are both smaller and branch-free. wasm byte
//! compares are SIGNED: encode biases values by 0x80 before threshold
//! tests, and decode compares ASCII directly — every alphabet range
//! sits inside 0..=127, so bytes >= 0x80 read as negative and fail
//! every lower bound, which is exactly the scalar reject set.

use core::arch::wasm32::{
    i16x8_shl, i16x8_splat, i32x4_shl, i32x4_splat, i8x16_add, i8x16_ge, i8x16_gt, i8x16_le,
    i8x16_shuffle, i8x16_splat, i8x16_sub, u16x8_shr, u32x4_shr, v128, v128_and, v128_any_true,
    v128_bitselect, v128_load, v128_not, v128_or, v128_store, v128_xor,
};

use super::{HASH_B64_LEN, SALT_B64_LEN};

/// 6-bit value -> alphabet char, branch-free: offsets 46 (v < 2), 63
/// (v < 28), 69 (v < 54), -6 (otherwise) against the `./A-Za-z0-9`
/// alphabet (26 lowercase letters, so digits start at value 54). The
/// 0x80 bias moves the unsigned thresholds into the signed-compare
/// domain; the masks are nested, so the bitselect chain needs no mask
/// arithmetic.
#[inline(always)]
fn translate_encode(vals: v128) -> v128 {
    let biased = v128_xor(vals, i8x16_splat(-128));
    let lt2 = i8x16_gt(i8x16_splat(-126), biased);
    let lt28 = i8x16_gt(i8x16_splat(-100), biased);
    let lt54 = i8x16_gt(i8x16_splat(-74), biased);
    let off = v128_bitselect(
        i8x16_splat(46),
        v128_bitselect(
            i8x16_splat(63),
            v128_bitselect(i8x16_splat(69), i8x16_splat(-6), lt54),
            lt28,
        ),
        lt2,
    );
    i8x16_add(vals, off)
}

/// Encode one 12-byte block into 16 chars. `pairs` holds the 16-bit
/// byte pairs `[b0,b1, b1,b2, b3,b4, b4,b5, ...]` (LE lanes) covering
/// four 3-byte groups; even lanes yield chars 0,1 mod 4 and odd lanes
/// chars 2,3, and the merge shuffle picks the valid lane of each.
#[inline(always)]
fn encode_block(pairs: v128) -> v128 {
    // Even lane = (b0|b1<<8) -> c0 | c1<<8, with c0 = b0>>2 and
    // c1 = (b0&3)<<4 | b1>>4.
    let lo = v128_or(
        v128_or(
            v128_and(u16x8_shr(pairs, 2), i16x8_splat(0x3f)),
            i16x8_shl(v128_and(pairs, i16x8_splat(3)), 12),
        ),
        v128_and(u16x8_shr(pairs, 4), i16x8_splat(0x0f00)),
    );
    // Odd lane = (b1|b2<<8) -> c2 | c3<<8, with c2 = (b1&0xf)<<2 | b2>>6
    // and c3 = b2&0x3f.
    let hi = v128_or(
        v128_or(
            v128_and(i16x8_shl(pairs, 2), i16x8_splat(0x3c)),
            u16x8_shr(pairs, 14),
        ),
        v128_and(pairs, i16x8_splat(0x3f00)),
    );
    let vals = i8x16_shuffle::<0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31>(lo, hi);
    translate_encode(vals)
}

/// Overlapping byte pairs `[b0,b1, b1,b2, ...]` for the first 12-byte
/// block, straight out of the block's load window.
#[inline(always)]
fn pairs_0_12(v: v128) -> v128 {
    i8x16_shuffle::<0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11>(v, v)
}

/// The same pairs for the second block: input byte 12 is byte 4 of the
/// second load window (`buf[8..24]`).
#[inline(always)]
fn pairs_12_24(v: v128) -> v128 {
    i8x16_shuffle::<4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15>(v, v)
}

/// Encode a 16-byte salt to its 22-char bcrypt base64 form.
pub(crate) fn encode_16(bytes: &[u8; 16]) -> [u8; SALT_B64_LEN] {
    let mut buf = [0u8; 24];
    buf[..16].copy_from_slice(bytes);
    // SAFETY: `buf` is a live 24-byte stack array; the two loads cover
    // bytes 0..16 and 8..24, both in full.
    let (v0, v1) = unsafe {
        (
            v128_load(buf.as_ptr().cast::<v128>()),
            v128_load(buf.as_ptr().add(8).cast::<v128>()),
        )
    };
    let c0 = encode_block(pairs_0_12(v0));
    let c1 = encode_block(pairs_12_24(v1));
    let mut scratch = [0u8; 32];
    // SAFETY: `scratch` is a live 32-byte stack array; the stores cover
    // bytes 0..16 and 16..32, both in full.
    unsafe {
        v128_store(scratch.as_mut_ptr().cast::<v128>(), c0);
        v128_store(scratch.as_mut_ptr().add(16).cast::<v128>(), c1);
    }
    // Chars 22..32 encode the zero padding; the copy drops them.
    let mut out = [0u8; SALT_B64_LEN];
    out.copy_from_slice(&scratch[..SALT_B64_LEN]);
    out
}

/// Encode the 23 hash bytes to their 31-char bcrypt base64 form.
pub(crate) fn encode_23(bytes: &[u8; 23]) -> [u8; HASH_B64_LEN] {
    let mut buf = [0u8; 24];
    buf[..23].copy_from_slice(bytes);
    // SAFETY: `buf` is a live 24-byte stack array; the two loads cover
    // bytes 0..16 and 8..24, both in full.
    let (v0, v1) = unsafe {
        (
            v128_load(buf.as_ptr().cast::<v128>()),
            v128_load(buf.as_ptr().add(8).cast::<v128>()),
        )
    };
    let c0 = encode_block(pairs_0_12(v0));
    let c1 = encode_block(pairs_12_24(v1));
    let mut scratch = [0u8; 32];
    // SAFETY: `scratch` is a live 32-byte stack array; the stores cover
    // bytes 0..16 and 16..32, both in full.
    unsafe {
        v128_store(scratch.as_mut_ptr().cast::<v128>(), c0);
        v128_store(scratch.as_mut_ptr().add(16).cast::<v128>(), c1);
    }
    // Char 31 encodes the zero padding; the copy drops it.
    let mut out = [0u8; HASH_B64_LEN];
    out.copy_from_slice(&scratch[..HASH_B64_LEN]);
    out
}

/// Validate and translate one 16-char vector to 6-bit values, then pack
/// each 4-value dword into 3 bytes. Returns the packed vector (12 real
/// bytes plus 4 filler bytes from the compaction shuffle) and a mask
/// that is all-ones in every lane holding a non-alphabet byte.
#[inline(always)]
fn decode_block(chars: v128) -> (v128, v128) {
    // All four ranges sit inside 0..=127, so a byte >= 0x80 is negative
    // to the signed compares and fails every lower bound — invalid,
    // exactly as the scalar table says.
    let dot = v128_and(
        i8x16_ge(chars, i8x16_splat(b'.' as i8)),
        i8x16_le(chars, i8x16_splat(b'/' as i8)),
    );
    let digit = v128_and(
        i8x16_ge(chars, i8x16_splat(b'0' as i8)),
        i8x16_le(chars, i8x16_splat(b'9' as i8)),
    );
    let upper = v128_and(
        i8x16_ge(chars, i8x16_splat(b'A' as i8)),
        i8x16_le(chars, i8x16_splat(b'Z' as i8)),
    );
    let lower = v128_and(
        i8x16_ge(chars, i8x16_splat(b'a' as i8)),
        i8x16_le(chars, i8x16_splat(b'z' as i8)),
    );
    let valid = v128_or(v128_or(dot, digit), v128_or(upper, lower));
    // The masks are disjoint, so any nesting order works; invalid lanes
    // take a garbage value that the error path never returns.
    let vals = v128_bitselect(
        i8x16_sub(chars, i8x16_splat(46)),
        v128_bitselect(
            i8x16_add(chars, i8x16_splat(6)),
            v128_bitselect(
                i8x16_sub(chars, i8x16_splat(63)),
                i8x16_sub(chars, i8x16_splat(69)),
                upper,
            ),
            digit,
        ),
        dot,
    );
    // Per dword [c0,c1,c2,c3] (LE): B0 = c0<<2 | c1>>4 lands at byte 0,
    // B1 = (c1&0xf)<<4 | c2>>2 at byte 1, B2 = (c2&3)<<6 | c3 at byte 2.
    let b0 = v128_or(
        v128_and(i32x4_shl(vals, 2), i32x4_splat(0xfc)),
        v128_and(u32x4_shr(vals, 12), i32x4_splat(0x0f)),
    );
    let b1 = v128_or(
        v128_and(i32x4_shl(vals, 4), i32x4_splat(0xf000)),
        v128_and(u32x4_shr(vals, 10), i32x4_splat(0x3f00)),
    );
    let b2 = v128_or(
        v128_and(i32x4_shl(vals, 6), i32x4_splat(0xc0_0000)),
        v128_and(u32x4_shr(vals, 8), i32x4_splat(0x3f_0000)),
    );
    let packed = v128_or(v128_or(b0, b1), b2);
    let compact =
        i8x16_shuffle::<0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 16, 16, 16>(packed, packed);
    (compact, v128_not(valid))
}

/// Decode exactly [`SALT_B64_LEN`] chars back to the 16-byte salt.
pub(crate) fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
    if s.len() != SALT_B64_LEN {
        return Err(());
    }
    // Pad to 24 chars with value-0 '.'; its bits only reach output bytes
    // 16..18, which the final copy drops.
    let mut buf = [0u8; 32];
    buf[..SALT_B64_LEN].copy_from_slice(s);
    buf[22] = b'.';
    buf[23] = b'.';
    // SAFETY: `buf` is a live 32-byte stack array; the two loads cover
    // bytes 0..16 and 8..24, both in full.
    let (v0, v1) = unsafe {
        (
            v128_load(buf.as_ptr().cast::<v128>()),
            v128_load(buf.as_ptr().add(8).cast::<v128>()),
        )
    };
    let (p0, bad0) = decode_block(v0);
    let (p1, bad1) = decode_block(v1);
    if v128_any_true(v128_or(bad0, bad1)) {
        return Err(());
    }
    // Block 0 (chars 0..16) yields bytes 0..12; block 1 (chars 8..24)
    // yields bytes 6..18, so the second store's overlap rewrites block
    // 0's filler with the real tail bytes.
    let mut scratch = [0u8; 32];
    // SAFETY: `scratch` is a live 32-byte stack array; the stores cover
    // bytes 0..16 and 6..22, both in full.
    unsafe {
        v128_store(scratch.as_mut_ptr().cast::<v128>(), p0);
        v128_store(scratch.as_mut_ptr().add(6).cast::<v128>(), p1);
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(&scratch[..16]);
    Ok(out)
}

/// Decode exactly [`HASH_B64_LEN`] chars back to the 23 hash bytes.
pub(crate) fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
    if s.len() != HASH_B64_LEN {
        return Err(());
    }
    // One '.' pad to 32 chars; its bits land only in output byte 23,
    // which the final copy drops.
    let mut buf = [0u8; 32];
    buf[..HASH_B64_LEN].copy_from_slice(s);
    buf[31] = b'.';
    // SAFETY: `buf` is a live 32-byte stack array; the two loads cover
    // bytes 0..16 and 16..32, both in full.
    let (v0, v1) = unsafe {
        (
            v128_load(buf.as_ptr().cast::<v128>()),
            v128_load(buf.as_ptr().add(16).cast::<v128>()),
        )
    };
    let (p0, bad0) = decode_block(v0);
    let (p1, bad1) = decode_block(v1);
    if v128_any_true(v128_or(bad0, bad1)) {
        return Err(());
    }
    // Block 0 (chars 0..16) yields bytes 0..12; block 1 (chars 16..32)
    // yields bytes 12..24.
    let mut scratch = [0u8; 32];
    // SAFETY: `scratch` is a live 32-byte stack array; the stores cover
    // bytes 0..16 and 12..28, both in full.
    unsafe {
        v128_store(scratch.as_mut_ptr().cast::<v128>(), p0);
        v128_store(scratch.as_mut_ptr().add(12).cast::<v128>(), p1);
    }
    let mut out = [0u8; 23];
    out.copy_from_slice(&scratch[..23]);
    Ok(out)
}
