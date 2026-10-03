//! NEON kernel — aarch64 always has the feature, so the parent module calls
//! these unconditionally. Byte-exact with the scalar tables, reject set
//! included (the parent's differential tests pin that).
//!
//! Both directions pad the payload to full 16-byte groups with zero bits
//! (encode: zero tail bytes; decode: '.' = alphabet value 0). That is
//! invisible to the dialect: the scalar encoder zeroes the spare tail bits
//! and the scalar decoder ignores them, so a zero-padded tail produces the
//! same kept bytes as the scalar partial-group paths. Encode pulls the
//! 24-byte window from a two-register table (`vqtbl2q_u8`): the head
//! register is input bytes 0..15, the tail is zeros (encode_16) or input
//! bytes 15..22 zero-extended (encode_23), so table indices stay in bounds
//! while reading past the payload end yields zero. Fields are translated
//! with one `vqtbl4q_u8` against the 64-entry alphabet. Decode validates
//! with unsigned range arithmetic on the five alphabet classes and packs
//! 4x6 -> 3 with the shift/or inverse; bits dropped by the byte shifts are
//! exactly the spare ones. Overlapping stores land exactly 22/31 chars and
//! 16/23 bytes with no scalar tail.

use core::arch::aarch64::*;

use super::{ALPHABET, HASH_B64_LEN, SALT_B64_LEN};

// Encode: per 4-char group, lane j of A/B names the window byte feeding
// output char j. Window 1 is input bytes 0..11, i.e. table lanes 0..11.
static IDX_A: [u8; 16] = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11];
static IDX_B: [u8; 16] = [0, 1, 2, 0, 0, 4, 5, 0, 0, 7, 8, 0, 0, 10, 11, 0];

// Window 2 is input bytes 12..23. encode_16's tail register is all zeros,
// so the table lane equals the byte position (16.. read as zero).
static IDX_A2_16: [u8; 16] = [12, 12, 13, 14, 15, 15, 16, 17, 18, 18, 19, 20, 21, 21, 22, 23];
static IDX_B2_16: [u8; 16] = [0, 13, 14, 0, 0, 16, 17, 0, 0, 19, 20, 0, 0, 22, 23, 0];

// encode_23's tail register holds input bytes 15..22 zero-extended, so
// bytes 16..22 sit at table lanes 17..23 and lane 24+ reads as zero.
static IDX_A2_23: [u8; 16] = [12, 12, 13, 14, 15, 15, 17, 18, 19, 19, 20, 21, 22, 22, 23, 24];
static IDX_B2_23: [u8; 16] = [0, 13, 14, 0, 0, 17, 18, 0, 0, 20, 21, 0, 0, 23, 24, 0];

// Per-lane signed shifts (negative = right) and masks carving 6-bit fields.
static SHIFT_A: [i8; 16] = [-2, 4, 2, 0, -2, 4, 2, 0, -2, 4, 2, 0, -2, 4, 2, 0];
static SHIFT_B: [i8; 16] = [0, -4, -6, 0, 0, -4, -6, 0, 0, -4, -6, 0, 0, -4, -6, 0];
static MASK_A: [u8; 16] = [0x3f, 0x30, 0x3c, 0x3f, 0x3f, 0x30, 0x3c, 0x3f, 0x3f, 0x30, 0x3c, 0x3f, 0x3f, 0x30, 0x3c, 0x3f];
static MASK_B: [u8; 16] = [0x00, 0x0f, 0x03, 0x00, 0x00, 0x0f, 0x03, 0x00, 0x00, 0x0f, 0x03, 0x00, 0x00, 0x0f, 0x03, 0x00];

// Decode pack: lanes 3i..3i+2 of the output take the 6-bit values of
// chars 4i..4i+3; lanes 12..15 are junk that stores never keep.
static PK_A: [u8; 16] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0];
static PK_B: [u8; 16] = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 0, 0, 0, 0];
static PK_SA: [i8; 16] = [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 0, 0, 0, 0];
static PK_SB: [i8; 16] = [-4, -2, 0, -4, -2, 0, -4, -2, 0, -4, -2, 0, 0, 0, 0, 0];

/// The 24 bytes in the `{head, tail}` window as 32 alphabet chars; callers
/// keep the front 22/31 via overlapping stores.
fn encode_core(head: uint8x16_t, tail: uint8x16_t, a2: &[u8; 16], b2: &[u8; 16]) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: NEON is mandatory on aarch64. All loads are 16-byte reads of
    // 16-byte statics, or of the 64-byte ALPHABET at offsets 0/16/32/48.
    unsafe {
        let tab = uint8x16x2_t(head, tail);
        let lut = uint8x16x4_t(
            vld1q_u8(ALPHABET.as_ptr()),
            vld1q_u8(ALPHABET.as_ptr().add(16)),
            vld1q_u8(ALPHABET.as_ptr().add(32)),
            vld1q_u8(ALPHABET.as_ptr().add(48)),
        );
        let sa = vld1q_s8(SHIFT_A.as_ptr());
        let sb = vld1q_s8(SHIFT_B.as_ptr());
        let ma = vld1q_u8(MASK_A.as_ptr());
        let mb = vld1q_u8(MASK_B.as_ptr());
        let enc = |ia: uint8x16_t, ib: uint8x16_t| {
            let a = vqtbl2q_u8(tab, ia);
            let b = vqtbl2q_u8(tab, ib);
            let fields = vorrq_u8(vandq_u8(vshlq_u8(a, sa), ma), vandq_u8(vshlq_u8(b, sb), mb));
            vqtbl4q_u8(lut, fields)
        };
        (
            enc(vld1q_u8(IDX_A.as_ptr()), vld1q_u8(IDX_B.as_ptr())),
            enc(vld1q_u8(a2.as_ptr()), vld1q_u8(b2.as_ptr())),
        )
    }
}

/// Encode a 16-byte salt to its 22-char bcrypt base64 form.
pub(crate) fn encode_16(bytes: &[u8; 16]) -> [u8; SALT_B64_LEN] {
    let mut out = [0u8; SALT_B64_LEN];
    // SAFETY: NEON is mandatory on aarch64. The 16-byte load reads exactly
    // `bytes`. The stores write 16 bytes at offsets 0 and 6 of a 22-byte
    // array — both fully in bounds; the second carries chars 6..21 via the
    // vext slide (c0[6..15], then c1[0..5]).
    unsafe {
        let head = vld1q_u8(bytes.as_ptr());
        let (c0, c1) = encode_core(head, vdupq_n_u8(0), &IDX_A2_16, &IDX_B2_16);
        vst1q_u8(out.as_mut_ptr(), c0);
        vst1q_u8(out.as_mut_ptr().add(6), vextq_u8::<6>(c0, c1));
    }
    out
}

/// Encode the 23 hash bytes to their 31-char bcrypt base64 form.
pub(crate) fn encode_23(bytes: &[u8; 23]) -> [u8; HASH_B64_LEN] {
    let mut out = [0u8; HASH_B64_LEN];
    // SAFETY: NEON is mandatory on aarch64. Loads read 16 bytes at offset 0
    // and 8 bytes at offset 15 of a 23-byte array — both in bounds. Stores
    // write 16 bytes at offsets 0 and 15 of a 31-byte array — in bounds;
    // the vext slide is (c0[15], then c1[0..14]) = chars 15..30.
    unsafe {
        let head = vld1q_u8(bytes.as_ptr());
        let tail = vcombine_u8(vld1_u8(bytes.as_ptr().add(15)), vdup_n_u8(0));
        let (c0, c1) = encode_core(head, tail, &IDX_A2_23, &IDX_B2_23);
        vst1q_u8(out.as_mut_ptr(), c0);
        vst1q_u8(out.as_mut_ptr().add(15), vextq_u8::<15>(c0, c1));
    }
    out
}

/// Char -> 6-bit value plus a validity mask (0xFF lanes), via wrapping
/// unsigned range checks on the five alphabet classes — the exact scalar
/// reject set. Classes are disjoint, so the vbsl chain order is free.
fn translate(c: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: NEON is mandatory on aarch64; register-only, no memory.
    unsafe {
        let d_punct = vsubq_u8(c, vdupq_n_u8(b'.')); // '.' -> 0, '/' -> 1
        let d0 = vsubq_u8(c, vdupq_n_u8(b'0'));
        let da = vsubq_u8(c, vdupq_n_u8(b'A'));
        let dl = vsubq_u8(c, vdupq_n_u8(b'a'));
        let m_punct = vcgtq_u8(vdupq_n_u8(2), d_punct);
        let m_digit = vcgtq_u8(vdupq_n_u8(10), d0);
        let m_upper = vcgtq_u8(vdupq_n_u8(26), da);
        let m_lower = vcgtq_u8(vdupq_n_u8(26), dl);
        let v = vbslq_u8(m_digit, vaddq_u8(d0, vdupq_n_u8(54)), d_punct);
        let v = vbslq_u8(m_upper, vaddq_u8(da, vdupq_n_u8(2)), v);
        let v = vbslq_u8(m_lower, vaddq_u8(dl, vdupq_n_u8(28)), v);
        let m = vorrq_u8(vorrq_u8(m_punct, m_digit), vorrq_u8(m_upper, m_lower));
        (v, m)
    }
}

/// 16 6-bit values -> 12 packed bytes in lanes 0..11 (12..15 are junk).
/// Byte shifts drop exactly the bits the scalar masks would clear.
fn pack(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is mandatory on aarch64; the loads are 16-byte reads of
    // 16-byte statics.
    unsafe {
        let wa = vqtbl1q_u8(v, vld1q_u8(PK_A.as_ptr()));
        let wb = vqtbl1q_u8(v, vld1q_u8(PK_B.as_ptr()));
        let la = vshlq_u8(wa, vld1q_s8(PK_SA.as_ptr()));
        let lb = vshlq_u8(wb, vld1q_s8(PK_SB.as_ptr()));
        vorrq_u8(la, lb)
    }
}

/// Translate and pack a 22- or 31-char string into 24 bytes; the caller
/// keeps the front 16/23. Tail pad chars are '.', whose zero bits land
/// only in dropped bytes — the scalar decoder ignores spare bits the same.
fn decode_packed(s: &[u8]) -> Result<[u8; 24], ()> {
    debug_assert!(s.len() == SALT_B64_LEN || s.len() == HASH_B64_LEN);
    let mut buf = [b'.'; 32];
    buf[..s.len()].copy_from_slice(s);
    let mut out = [0u8; 24];
    // SAFETY: NEON is mandatory on aarch64. `buf` is 32 bytes, so both
    // 16-byte loads are in bounds. Stores into the 24-byte `out`: 16 bytes
    // at 0 (junk lanes 12..15 overwritten next), 8 at 12, 4 at 20 — all in
    // bounds; aarch64 is little-endian, so the u32 lane is bytes 20..23.
    unsafe {
        let (v0, m0) = translate(vld1q_u8(buf.as_ptr()));
        let (v1, m1) = translate(vld1q_u8(buf.as_ptr().add(16)));
        if vminvq_u8(vandq_u8(m0, m1)) == 0 {
            return Err(());
        }
        let p1 = pack(v1);
        vst1q_u8(out.as_mut_ptr(), pack(v0));
        vst1_u8(out.as_mut_ptr().add(12), vget_low_u8(p1));
        let hi = vgetq_lane_u32::<2>(vreinterpretq_u32_u8(p1));
        (out.as_mut_ptr().add(20) as *mut u32).write_unaligned(hi);
    }
    Ok(out)
}

/// Decode exactly [`SALT_B64_LEN`] chars back to the 16-byte salt.
pub(crate) fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
    if s.len() != SALT_B64_LEN {
        return Err(());
    }
    let packed = decode_packed(s)?;
    let mut out = [0u8; 16];
    out.copy_from_slice(&packed[..16]);
    Ok(out)
}

/// Decode exactly [`HASH_B64_LEN`] chars back to the 23-byte hash payload.
pub(crate) fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
    if s.len() != HASH_B64_LEN {
        return Err(());
    }
    let packed = decode_packed(s)?;
    let mut out = [0u8; 23];
    out.copy_from_slice(&packed[..23]);
    Ok(out)
}
