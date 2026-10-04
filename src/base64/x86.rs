//! x86-64 base64 kernels for bcrypt's dialect: AVX2 and SSSE3 flavors
//! behind a runtime pick, byte-exact against [`super::scalar`] — outputs
//! and the decode reject set included. The parent module's
//! `dispatch_matches_scalar` and this module's per-flavor tests pin that.
//!
//! # The uniform-padding trick
//!
//! Both payload sizes ride one fixed 24-byte → 32-char pipeline:
//!
//! * **Encode**: the input is copied into a zeroed 32-byte stack buffer
//!   and 24 bytes are encoded to 32 chars; the first 22/31 are kept. The
//!   zero padding is what makes this byte-exact: every kept char encodes
//!   only real input bytes, and the final kept char's spare low bits come
//!   out zero — exactly the scalar encoder's canonical form (16 bytes end
//!   on a char carrying 2 of 6 bits, 23 bytes on one carrying 4 of 6).
//! * **Decode**: 32 chars are validated and decoded to 24 bytes and the
//!   first 16/23 are kept. The 32-char window is built in registers:
//!   chars 0..16 load straight from the input (always real), and chars
//!   16..32 come from a 16-byte load of the last 16 chars shifted down
//!   with the vacated tail ORed to `b'.'` (alphabet value 0). Pad chars
//!   land in the tail of the last kept group or beyond it, so their
//!   bits only reach dropped bytes — which is also why the spare low
//!   bits of the real final char are ignored, exactly like the scalar
//!   decoder. `'.'` is a valid char, so validating all 32 lanes
//!   validates exactly the input.
//!
//! Decode reads exactly `CHARS` bytes of the caller's slice; every
//! other load and store stays inside these frames' stack buffers, and
//! nothing reads or writes past the caller's slices.
//!
//! # Kernels
//!
//! Encode expands 3 bytes to four 6-bit fields with the classic Muła
//! sequence — `vpshufb` placing each group's bytes into a 32-bit lane as
//! `[b, a, c, b]`, then masked `vpmulhuw`/`vpmullw` spreading the fields
//! into bytes — followed by range-arithmetic value→char translation for
//! the bcrypt ordering: +46 for `0..2` (`'.'`/`'/'`), +63 for `2..28`
//! (`'A'−2`), +69 for `28..54` (`'a'−28`), −6 for `54..64` (`'0'−54`),
//! as a nested `vpcmpgtb` mask chain (values are 0..=63, so signed byte
//! compares are exact). Decode classifies with a two-shuffle nibble LUT
//! (escape-simd's scheme): `vpshufb(LO_TAB, c & 0xF) & vpshufb(HI_TAB,
//! c >> 4)` is nonzero exactly for alphabet chars — one `vpcmpeqb`
//! against zero turns that into the invalid mask — and
//! `c + vpshufb(OFF_TAB, c >> 4)` is the 6-bit value. Rejects on any
//! invalid lane, then packs 4×6 → 3 bytes with the classic
//! `vpmaddubsw` + `vpmaddwd` chain and a final `vpshufb` (the AVX2
//! flavor adds one `vpermd` to join the two lanes' 12-byte halves).
//!
//! # Dispatch
//!
//! The pick is per operation, measured on Zen 4 (`benches/base64.rs`,
//! medians of two criterion runs; ns):
//!
//! | op | scalar | SSSE3 | AVX2 | picked |
//! |---|---|---|---|---|
//! | encode_16 | 11.8 | **8.9** | 20.1 | SSSE3 |
//! | encode_23 | 14.2 | **10.4** | 21.8 | SSSE3 |
//! | decode_16 | 14.8 | 12.1 | **9.0** | AVX2 |
//! | decode_23 | 20.0 | 14.9 | **9.0** | AVX2 |
//!
//! (Encode rows predate the nibble-LUT rewrite — encode is untouched by
//! it; decode rows were re-measured with it.) Encode is 128-bit: at a
//! 24-byte payload the 256-bit expansion's fixed costs (the `vpermd`
//! lane-join, the 32-byte tmp store-forward) exceed the work — the same
//! reason aarch64's 128-bit NEON encode wins big. Decode is 256-bit:
//! the scalar competitor does four dependent table loads per 3 bytes,
//! so the wider LUT pays. Both decode sizes now cost the same — the
//! old decode_23 premium was the 31-byte padded stack copy, which the
//! register-built window removed. `no_std` has no
//! runtime detection and falls back to compile-time `target_feature`
//! cfgs, then scalar — the same split `crate::eks` uses. Under Miri both
//! probes answer false: Miri does not interpret these intrinsics, and the
//! `miri` CI job runs the `--lib` tests (the base64 differential
//! included) — the same Scalar pin `eks::detect` applies.
//!
//! # Rosetta note
//!
//! Like `crate::eks`: Rosetta 2 executes SSSE3 and AVX2 but does not
//! advertise AVX2 in cpuid, so `cargo test --target x86_64-apple-darwin`
//! exercises the SSSE3 flavor at the runtime pick, and
//! `RUSTFLAGS="-C target-feature=+avx2"` compiles the AVX2 probe to
//! `true`, exercising that flavor. Correctness is the gate there.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi8, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8,
    _mm_cmpgt_epi8, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8,
    _mm_mulhi_epu16, _mm_mullo_epi16, _mm_or_si128, _mm_set1_epi32, _mm_set1_epi8,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_si128, _mm_srli_epi16,
    _mm_srli_si128, _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_castsi128_si256, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_inserti128_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_movemask_epi8, _mm256_mulhi_epu16,
    _mm256_mullo_epi16, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
    _mm256_set1_epi8, _mm256_setr_epi32, _mm256_setr_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
};

use super::{HASH_B64_LEN, SALT_B64_LEN, scalar};

// Runtime feature probes. This module is compiled only on x86_64 (the
// parent gates it); with `std` the probe is a real runtime check, without
// it the answer degrades to the compile-time `target_feature` cfg — the
// same split `crate::eks` uses. Under Miri both answer false (module
// docs), pinning the scalar tables.
#[cfg(feature = "std")]
#[inline]
fn have_avx2() -> bool {
    !cfg!(miri) && std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(feature = "std"))]
#[inline]
fn have_avx2() -> bool {
    !cfg!(miri) && cfg!(target_feature = "avx2")
}

#[cfg(feature = "std")]
#[inline]
fn have_ssse3() -> bool {
    !cfg!(miri) && std::arch::is_x86_feature_detected!("ssse3")
}
#[cfg(not(feature = "std"))]
#[inline]
fn have_ssse3() -> bool {
    !cfg!(miri) && cfg!(target_feature = "ssse3")
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// The encode block shared by both flavors, at xmm width: one lane of 12
/// input bytes (byte 12..15 are ignored) becomes 16 bcrypt-base64 chars.
/// Expansion is the classic Muła `vpshufb` + `vpmulhuw`/`vpmullw`
/// sequence; translation is the bcrypt-order range chain (module docs).
#[target_feature(enable = "ssse3")]
#[inline]
fn enc_block_ssse3(x: __m128i) -> __m128i {
    let shuffled = _mm_shuffle_epi8(
        x,
        _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10),
    );
    let t0 = _mm_and_si128(shuffled, _mm_set1_epi32(0x0FC0_FC00));
    let t1 = _mm_mulhi_epu16(t0, _mm_set1_epi32(0x0400_0040));
    let t2 = _mm_and_si128(shuffled, _mm_set1_epi32(0x003F_03F0));
    let t3 = _mm_mullo_epi16(t2, _mm_set1_epi32(0x0100_0010));
    let v = _mm_or_si128(t1, t3);
    // Value → char: nested thresholds 2/28/54, later masks overriding.
    let m1 = _mm_cmpgt_epi8(v, _mm_set1_epi8(1));
    let m2 = _mm_cmpgt_epi8(v, _mm_set1_epi8(27));
    let m3 = _mm_cmpgt_epi8(v, _mm_set1_epi8(53));
    let mut off = _mm_set1_epi8(46);
    off = _mm_or_si128(_mm_and_si128(m1, _mm_set1_epi8(63)), _mm_andnot_si128(m1, off));
    off = _mm_or_si128(_mm_and_si128(m2, _mm_set1_epi8(69)), _mm_andnot_si128(m2, off));
    off = _mm_or_si128(_mm_and_si128(m3, _mm_set1_epi8(-6)), _mm_andnot_si128(m3, off));
    _mm_add_epi8(v, off)
}

/// [`enc_block_ssse3`] at ymm width: two independent 12-byte lanes
/// (lane 0 = input bytes 0..12, lane 1 = bytes 12..24) become 32 chars.
#[target_feature(enable = "avx2")]
#[inline]
fn enc_block_avx2(x: __m256i) -> __m256i {
    let shuffled = _mm256_shuffle_epi8(
        x,
        _mm256_setr_epi8(
            1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, 1, 0, 2, 1, 4, 3, 5, 4, 7, 6,
            8, 7, 10, 9, 11, 10,
        ),
    );
    let t0 = _mm256_and_si256(shuffled, _mm256_set1_epi32(0x0FC0_FC00));
    let t1 = _mm256_mulhi_epu16(t0, _mm256_set1_epi32(0x0400_0040));
    let t2 = _mm256_and_si256(shuffled, _mm256_set1_epi32(0x003F_03F0));
    let t3 = _mm256_mullo_epi16(t2, _mm256_set1_epi32(0x0100_0010));
    let v = _mm256_or_si256(t1, t3);
    let m1 = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(1));
    let m2 = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(27));
    let m3 = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(53));
    let mut off = _mm256_set1_epi8(46);
    off = _mm256_or_si256(
        _mm256_and_si256(m1, _mm256_set1_epi8(63)),
        _mm256_andnot_si256(m1, off),
    );
    off = _mm256_or_si256(
        _mm256_and_si256(m2, _mm256_set1_epi8(69)),
        _mm256_andnot_si256(m2, off),
    );
    off = _mm256_or_si256(
        _mm256_and_si256(m3, _mm256_set1_epi8(-6)),
        _mm256_andnot_si256(m3, off),
    );
    _mm256_add_epi8(v, off)
}

/// Encode `IN` bytes (16 or 23) to their `OUT`-char (22 or 31) bcrypt
/// base64 form through the shared zero-padded 24-byte pipeline (module
/// docs), SSSE3 flavor: two 16-byte blocks.
///
/// # Safety
///
/// The CPU must support SSSE3 (checked by the caller, [`have_ssse3`]).
/// All loads and stores stay inside this frame's 32-byte stack buffers.
#[target_feature(enable = "ssse3")]
unsafe fn encode_ssse3<const IN: usize, const OUT: usize>(bytes: &[u8; IN]) -> [u8; OUT] {
    // Bytes IN..32 stay zero, which is what makes the kept tail chars
    // canonical (module docs).
    let mut buf = [0u8; 32];
    buf[..IN].copy_from_slice(bytes);
    // SAFETY: `buf` is a live 32-byte stack array; both 16-byte loads
    // (offsets 0 and 12) read within it.
    let (lo, hi) = unsafe {
        (
            _mm_loadu_si128(buf.as_ptr().cast::<__m128i>()),
            _mm_loadu_si128(buf.as_ptr().add(12).cast::<__m128i>()),
        )
    };
    let chars_lo = enc_block_ssse3(lo);
    let chars_hi = enc_block_ssse3(hi);
    let mut tmp = [0u8; 32];
    // SAFETY: `tmp` is a live 32-byte stack array; both 16-byte stores
    // (offsets 0 and 16) write within it.
    unsafe {
        _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), chars_lo);
        _mm_storeu_si128(tmp.as_mut_ptr().add(16).cast::<__m128i>(), chars_hi);
    }
    let mut out = [0u8; OUT];
    out.copy_from_slice(&tmp[..OUT]);
    out
}

/// [`encode_ssse3`], AVX2 flavor: one 32-byte block (two 12-byte lanes).
///
/// # Safety
///
/// The CPU must support AVX2 (checked by the caller, [`have_avx2`]).
/// All loads and stores stay inside this frame's 32-byte stack buffers.
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2<const IN: usize, const OUT: usize>(bytes: &[u8; IN]) -> [u8; OUT] {
    let mut buf = [0u8; 32];
    buf[..IN].copy_from_slice(bytes);
    // Lane 0 holds bytes 0..16 (uses 0..12), lane 1 bytes 12..28 (uses
    // 12..24) — the two-load arrangement that gives each lane's vpshufb
    // the same group offsets.
    //
    // SAFETY: `buf` is a live 32-byte stack array; both 16-byte loads
    // (offsets 0 and 12) read within it.
    let x = unsafe {
        let lo = _mm_loadu_si128(buf.as_ptr().cast::<__m128i>());
        let hi = _mm_loadu_si128(buf.as_ptr().add(12).cast::<__m128i>());
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    };
    let chars = enc_block_avx2(x);
    let mut tmp = [0u8; 32];
    // SAFETY: `tmp` is a live 32-byte stack array, written in full.
    unsafe { _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), chars) };
    let mut out = [0u8; OUT];
    out.copy_from_slice(&tmp[..OUT]);
    out
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

// Decode classification tables, indexed by a char's nibbles
// (escape-simd's two-shuffle scheme). Row bits per high nibble: row 2
// (`./`) 0x01, row 3 (`0-9`) 0x02, row 4 (`A-O`) 0x04, row 5 (`P-Z`)
// 0x08, row 6 (`a-o`) 0x10, row 7 (`p-z`) 0x20. LO_TAB[l] ORs the bits
// of every row r whose char (r<<4 | l) is in the alphabet: l=0 → '0',
// 'P','p' → 0x2A; l=1..=9 → rows 3..=7 → 0x3E; l=A → 'J','Z','j','z' →
// 0x3C; l=B..=D → rows 4,6 → 0x14; l=E → '.','N','n' → 0x15; l=F →
// '/','O','o' → 0x15. OFF_TAB is the per-row value offset: '.'−0 →
// −46, '0'−54 → +6, 'A'−2 → −63, 'a'−28 → −69.

/// Validate 16 chars and translate to 6-bit values, in natural order:
/// with `lo = c & 0xF` and `hi = c >> 4`, `pshufb(LO_TAB, lo) &
/// pshufb(HI_TAB, hi)` is nonzero ⟺ the char is in the alphabet, and
/// `c + pshufb(OFF_TAB, hi)` (wrapping) is its value. Returns the
/// values and an INVALID mask (0xFF per invalid lane); the caller
/// rejects unless the mask is all zero, and invalid lanes produce
/// garbage values the reject path never returns. Both `vpshufb` index
/// vectors stay in 0..=15, so the high-bit lane zeroing never fires.
#[target_feature(enable = "ssse3")]
#[inline]
fn dec_translate_ssse3(c: __m128i) -> (__m128i, __m128i) {
    let nibble = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(c, nibble);
    // No byte shift on x86: shift within words, then mask the nibble.
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(c), nibble);
    let lo_tab = _mm_setr_epi8(
        0x2A, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3C, 0x14, 0x14, 0x14,
        0x15, 0x15,
    );
    let hi_tab = _mm_setr_epi8(
        0x00, 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    );
    let off_tab = _mm_setr_epi8(0, 0, -46, 6, -63, -63, -69, -69, 0, 0, 0, 0, 0, 0, 0, 0);
    let bits = _mm_and_si128(_mm_shuffle_epi8(lo_tab, lo), _mm_shuffle_epi8(hi_tab, hi));
    let bad = _mm_cmpeq_epi8(bits, _mm_setzero_si128());
    let v = _mm_add_epi8(c, _mm_shuffle_epi8(off_tab, hi));
    (v, bad)
}

/// [`dec_translate_ssse3`] at ymm width (32 chars): the same tables
/// doubled across the two 128-bit lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn dec_translate_avx2(c: __m256i) -> (__m256i, __m256i) {
    let nibble = _mm256_set1_epi8(0x0F);
    let lo = _mm256_and_si256(c, nibble);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(c), nibble);
    let lo_tab = _mm256_setr_epi8(
        0x2A, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3C, 0x14, 0x14, 0x14,
        0x15, 0x15, 0x2A, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x3C, 0x14,
        0x14, 0x14, 0x15, 0x15,
    );
    let hi_tab = _mm256_setr_epi8(
        0x00, 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    );
    let off_tab = _mm256_setr_epi8(
        0, 0, -46, 6, -63, -63, -69, -69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -46, 6, -63, -63, -69,
        -69, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let bits = _mm256_and_si256(
        _mm256_shuffle_epi8(lo_tab, lo),
        _mm256_shuffle_epi8(hi_tab, hi),
    );
    let bad = _mm256_cmpeq_epi8(bits, _mm256_setzero_si256());
    let v = _mm256_add_epi8(c, _mm256_shuffle_epi8(off_tab, hi));
    (v, bad)
}

/// The upper half of the dot-padded 32-char decode window (chars
/// 16..32), built in registers: a 16-byte load of the LAST 16 chars
/// (chars CHARS-16..CHARS) shifted down so chars 16..CHARS land at
/// positions 0..CHARS-16, with the zeroed tail ORed to `'.'` pads.
/// `psrldq` takes a const immediate and CHARS is only ever
/// [`SALT_B64_LEN`] or [`HASH_B64_LEN`], so the shift amounts are a
/// folded two-way select.
#[target_feature(enable = "ssse3")]
#[inline]
fn dec_tail_ssse3<const CHARS: usize>(s: &[u8]) -> __m128i {
    assert!(CHARS == SALT_B64_LEN || CHARS == HASH_B64_LEN);
    // SAFETY: reads exactly chars CHARS-16..CHARS of `s`, whose length
    // the caller checked to be CHARS (>= 22 > 16).
    let last16 = unsafe { _mm_loadu_si128(s.as_ptr().add(CHARS - 16).cast::<__m128i>()) };
    let dots = _mm_set1_epi8(b'.' as i8);
    if CHARS == SALT_B64_LEN {
        _mm_or_si128(
            _mm_srli_si128::<{ (32 - SALT_B64_LEN) as i32 }>(last16),
            _mm_slli_si128::<{ (SALT_B64_LEN - 16) as i32 }>(dots),
        )
    } else {
        _mm_or_si128(
            _mm_srli_si128::<{ (32 - HASH_B64_LEN) as i32 }>(last16),
            _mm_slli_si128::<{ (HASH_B64_LEN - 16) as i32 }>(dots),
        )
    }
}

/// Pack 16 6-bit values (natural char order) into 12 bytes at byte
/// positions 0..12 of the result: `vpmaddubsw` merges value pairs into
/// the two 12-bit halves of each group, `vpmaddwd` joins the halves into
/// the 24-bit group, and `vpshufb` gathers the three bytes per group.
#[target_feature(enable = "ssse3")]
#[inline]
fn dec_pack_ssse3(v: __m128i) -> __m128i {
    let merge = _mm_maddubs_epi16(v, _mm_set1_epi32(0x0140_0140));
    let n = _mm_madd_epi16(merge, _mm_set1_epi32(0x0001_1000));
    _mm_shuffle_epi8(
        n,
        _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1),
    )
}

/// [`dec_pack_ssse3`] at ymm width, plus one `vpermd` joining the two
/// lanes' 12-byte results into byte positions 0..24 of the result.
#[target_feature(enable = "avx2")]
#[inline]
fn dec_pack_avx2(v: __m256i) -> __m256i {
    let merge = _mm256_maddubs_epi16(v, _mm256_set1_epi32(0x0140_0140));
    let n = _mm256_madd_epi16(merge, _mm256_set1_epi32(0x0001_1000));
    let bytes = _mm256_shuffle_epi8(
        n,
        _mm256_setr_epi8(
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9,
            8, 14, 13, 12, -1, -1, -1, -1,
        ),
    );
    _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, -1, -1))
}

/// Decode exactly `CHARS` (22 or 31) bcrypt-base64 chars to `OUT` bytes
/// (16 or 23) through the shared dot-padded 32-char pipeline (module
/// docs), SSSE3 flavor. `Err` on wrong length or any non-alphabet byte.
///
/// # Safety
///
/// The CPU must support SSSE3 (checked by the caller, [`have_ssse3`]).
/// Reads exactly `CHARS` bytes of `s`; stores stay inside this frame's
/// 32-byte stack buffer.
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const CHARS: usize, const OUT: usize>(
    s: &[u8],
) -> Result<[u8; OUT], ()> {
    if s.len() != CHARS {
        return Err(());
    }
    // The window is built in registers, not through a padded stack
    // copy: LLVM lowered that copy to overlapping stores whose reloads
    // fail to store-forward, a stall larger than the whole translate.
    //
    // SAFETY: reads exactly chars 0..16 of `s`, whose length was just
    // checked to be CHARS (>= 22 > 16).
    let lo = unsafe { _mm_loadu_si128(s.as_ptr().cast::<__m128i>()) };
    let hi = dec_tail_ssse3::<CHARS>(s);
    let (v0, bad0) = dec_translate_ssse3(lo);
    let (v1, bad1) = dec_translate_ssse3(hi);
    // One extraction for both blocks: the pad lanes are '.', always
    // valid, so this checks exactly the CHARS real chars (module docs).
    if _mm_movemask_epi8(_mm_or_si128(bad0, bad1)) != 0 {
        return Err(());
    }
    let bytes_lo = dec_pack_ssse3(v0);
    let bytes_hi = dec_pack_ssse3(v1);
    let mut tmp = [0u8; 32];
    // SAFETY: `tmp` is a live 32-byte stack array; both 16-byte stores
    // (offsets 0 and 12) write within it.
    unsafe {
        _mm_storeu_si128(tmp.as_mut_ptr().cast::<__m128i>(), bytes_lo);
        _mm_storeu_si128(tmp.as_mut_ptr().add(12).cast::<__m128i>(), bytes_hi);
    }
    let mut out = [0u8; OUT];
    out.copy_from_slice(&tmp[..OUT]);
    Ok(out)
}

/// [`decode_ssse3`], AVX2 flavor: one 32-char block.
///
/// # Safety
///
/// The CPU must support AVX2 (checked by the caller, [`have_avx2`]).
/// Reads exactly `CHARS` bytes of `s`; stores stay inside this frame's
/// 32-byte stack buffer.
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const CHARS: usize, const OUT: usize>(
    s: &[u8],
) -> Result<[u8; OUT], ()> {
    if s.len() != CHARS {
        return Err(());
    }
    // Register-built window like the SSSE3 flavor (see there).
    //
    // SAFETY: reads exactly chars 0..16 of `s`, whose length was just
    // checked to be CHARS (>= 22 > 16).
    let x = unsafe {
        let lo = _mm_loadu_si128(s.as_ptr().cast::<__m128i>());
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), dec_tail_ssse3::<CHARS>(s))
    };
    let (v, bad) = dec_translate_avx2(x);
    // Any invalid lane sets a sign bit; the pad lanes are '.', always
    // valid (module docs).
    if _mm256_movemask_epi8(bad) != 0 {
        return Err(());
    }
    let packed = dec_pack_avx2(v);
    let mut tmp = [0u8; 32];
    // SAFETY: `tmp` is a live 32-byte stack array, written in full.
    unsafe { _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), packed) };
    let mut out = [0u8; OUT];
    out.copy_from_slice(&tmp[..OUT]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Dispatch (the parent module calls these on x86_64)
// ---------------------------------------------------------------------------

/// Encode a 16-byte salt to its 22-char bcrypt base64 form.
//
// Encode prefers SSSE3 over AVX2 — see the "Dispatch" module docs: at a
// 24-byte payload the 256-bit encode's fixed costs (vpermd lane-join +
// the 32-byte tmp store-forward) exceed the work. Measured on Zen 4.
pub(crate) fn encode_16(bytes: &[u8; 16]) -> [u8; SALT_B64_LEN] {
    if have_ssse3() {
        // SAFETY: SSSE3 was just detected (module docs for the probes).
        unsafe { encode_ssse3::<16, SALT_B64_LEN>(bytes) }
    } else if have_avx2() {
        // SAFETY: AVX2 was just detected.
        unsafe { encode_avx2::<16, SALT_B64_LEN>(bytes) }
    } else {
        scalar::encode_16(bytes)
    }
}

/// Encode the 23 hash bytes to their 31-char bcrypt base64 form.
pub(crate) fn encode_23(bytes: &[u8; 23]) -> [u8; HASH_B64_LEN] {
    if have_ssse3() {
        // SAFETY: SSSE3 was just detected.
        unsafe { encode_ssse3::<23, HASH_B64_LEN>(bytes) }
    } else if have_avx2() {
        // SAFETY: AVX2 was just detected.
        unsafe { encode_avx2::<23, HASH_B64_LEN>(bytes) }
    } else {
        scalar::encode_23(bytes)
    }
}

/// Decode exactly [`SALT_B64_LEN`] chars back to the 16-byte salt.
pub(crate) fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
    if have_avx2() {
        // SAFETY: AVX2 was just detected.
        unsafe { decode_avx2::<SALT_B64_LEN, 16>(s) }
    } else if have_ssse3() {
        // SAFETY: SSSE3 was just detected.
        unsafe { decode_ssse3::<SALT_B64_LEN, 16>(s) }
    } else {
        scalar::decode_16(s)
    }
}

/// Decode exactly [`HASH_B64_LEN`] chars back to the 23 hash bytes.
pub(crate) fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
    if have_avx2() {
        // SAFETY: AVX2 was just detected.
        unsafe { decode_avx2::<HASH_B64_LEN, 23>(s) }
    } else if have_ssse3() {
        // SAFETY: SSSE3 was just detected.
        unsafe { decode_ssse3::<HASH_B64_LEN, 23>(s) }
    } else {
        scalar::decode_23(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* — the same deterministic byte source the parent
    /// module's tests use (no rand crate, no alloc).
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

    /// One encode flavor vs scalar: every byte value at every input
    /// position (pins the canonical zero-padding of the tail chars), then
    /// random whole-buffer compares — the parent test's shape.
    macro_rules! check_encode {
        ($kernel:ident) => {{
            for pos in 0..16 {
                for b in 0..=u8::MAX {
                    let mut input = [0u8; 16];
                    input[pos] = b;
                    assert_eq!($kernel::<16, SALT_B64_LEN>(&input), scalar::encode_16(&input));
                }
            }
            for pos in 0..23 {
                for b in 0..=u8::MAX {
                    let mut input = [0u8; 23];
                    input[pos] = b;
                    assert_eq!($kernel::<23, HASH_B64_LEN>(&input), scalar::encode_23(&input));
                }
            }
            let mut rng = Rng(0xC0FF_EE12_3456_7890);
            let mut b16 = [0u8; 16];
            let mut b23 = [0u8; 23];
            for _ in 0..512 {
                rng.fill(&mut b16);
                rng.fill(&mut b23);
                assert_eq!($kernel::<16, SALT_B64_LEN>(&b16), scalar::encode_16(&b16));
                assert_eq!($kernel::<23, HASH_B64_LEN>(&b23), scalar::encode_23(&b23));
            }
        }};
    }

    /// One decode flavor vs scalar: every byte value at every string
    /// position (reject set included — the parent test's sweep), every
    /// wrong length, then random round-trips.
    macro_rules! check_decode {
        ($kernel:ident) => {{
            let base16 = *b"CCCCCCCCCCCCCCCCCCCCCC";
            let base23 = *b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";
            for pos in 0..SALT_B64_LEN {
                for c in 0..=u8::MAX {
                    let mut s = base16;
                    s[pos] = c;
                    assert_eq!(
                        $kernel::<SALT_B64_LEN, 16>(&s),
                        scalar::decode_16(&s),
                        "16: pos {pos} c {c:#04x}"
                    );
                }
            }
            for pos in 0..HASH_B64_LEN {
                for c in 0..=u8::MAX {
                    let mut s = base23;
                    s[pos] = c;
                    assert_eq!(
                        $kernel::<HASH_B64_LEN, 23>(&s),
                        scalar::decode_23(&s),
                        "23: pos {pos} c {c:#04x}"
                    );
                }
            }
            let s = [b'C'; 33];
            for len in 0..=33usize {
                assert_eq!($kernel::<SALT_B64_LEN, 16>(&s[..len]), scalar::decode_16(&s[..len]));
                assert_eq!($kernel::<HASH_B64_LEN, 23>(&s[..len]), scalar::decode_23(&s[..len]));
            }
            let mut rng = Rng(0xB529_7A4D_1D2B_9F83);
            let mut b16 = [0u8; 16];
            let mut b23 = [0u8; 23];
            for _ in 0..256 {
                rng.fill(&mut b16);
                rng.fill(&mut b23);
                assert_eq!($kernel::<SALT_B64_LEN, 16>(&scalar::encode_16(&b16)), Ok(b16));
                assert_eq!($kernel::<HASH_B64_LEN, 23>(&scalar::encode_23(&b23)), Ok(b23));
            }
        }};
    }

    #[test]
    fn avx2_matches_scalar() {
        if !have_avx2() {
            return;
        }
        // SAFETY: the AVX2 kernels run only under have_avx2().
        unsafe {
            check_encode!(encode_avx2);
            check_decode!(decode_avx2);
        }
    }

    #[test]
    fn ssse3_matches_scalar() {
        if !have_ssse3() {
            return;
        }
        // SAFETY: the SSSE3 kernels run only under have_ssse3().
        unsafe {
            check_encode!(encode_ssse3);
            check_decode!(decode_ssse3);
        }
    }
}
