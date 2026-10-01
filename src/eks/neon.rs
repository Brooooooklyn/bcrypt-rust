//! AArch64 NEON EksBlowfish: four independent bcrypt hashes in lockstep,
//! one per 32-bit vector lane.
//!
//! Ports [`super::scalar`] exactly — same OpenBSD structure, same
//! continuous salt-word stream across the P→S transition in
//! `expand_state`, same key-then-salt cost-loop order — with every 32-bit
//! operation widened to a lane-wise `uint32x4_t` one. Scalar
//! `wrapping_add` becomes `vaddq_u32` (NEON addition wraps, which is
//! exactly what Blowfish wants); every XOR becomes `veorq_u32`.
//!
//! # State layout: struct-of-arrays
//!
//! The four S-boxes are stored interleaved by lane — [`SBoxes4`] — so
//! entry `(b, i)` holds all four lanes contiguously and one `vst1q_u32`
//! writes a whole entry during the expansion chains. Lookups are gathers:
//! lane `l` of a lookup reads `box_base[idx_l * 4 + l]`. NEON has no
//! gather instruction, so [`lookup4`] does four scalar loads and one
//! `vld1q_u32`; it is deliberately the only function that touches S-box
//! memory by computed address, so a later tuning phase can swap the
//! spelling (e.g. a `vld1q_lane_u32` chain) without touching the kernel
//! around it.
//!
//! # Lookup invariant
//!
//! Every index vector handed to [`lookup4`] is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f4`], so every
//! lane is in `0..=255` and `idx * 4 + lane` stays inside the addressed
//! 256-entry box — a gather can never leave the 16 KiB [`SBoxes4`]. The
//! invariant is stated once here and referenced at every unsafe
//! dereference below.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_u32, veorq_u32, vgetq_lane_u32, vld1q_u32,
    vshrq_n_u32, vst1q_u32,
};

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call — the four 32-bit NEON lanes.
const LANES: usize = 4;

/// Four lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is the
/// `(b, i)` Blowfish S-box word of all four lanes, lane `l` at u32 offset
/// `(b * 256 + i) * 4 + l`. 16 KiB, 64-byte aligned, so every entry is
/// 16-byte aligned by construction (the entry stride divides the struct
/// alignment).
#[repr(C, align(64))]
struct SBoxes4([[u32; 4]; 1024]);

impl SBoxes4 {
    /// Base pointer of box `b` — the address lane gathers index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 1024` stays inside the 4096-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 4) }
    }

    /// Store one (l, r) pair over entries `e` and `e + 1`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn store_pair(&mut self, e: usize, l: uint32x4_t, r: uint32x4_t) {
        debug_assert!(e + 1 < 1024);
        // SAFETY: `e + 1 < 1024`, so both entry pointers land inside the
        // array; each entry is exactly one 16-byte lane vector.
        unsafe {
            vst1q_u32(self.0[e].as_mut_ptr(), l);
            vst1q_u32(self.0[e + 1].as_mut_ptr(), r);
        }
    }
}

/// The lockstep EksBlowfish state: 18 P-vectors plus the interleaved
/// S-boxes — the lane-wise image of scalar `State`.
struct State4 {
    p: [uint32x4_t; 18],
    s: SBoxes4,
}

/// Gather one S-box word per lane: lane `l` loads `box_base[idx_l * 4 + l]`.
///
/// This is spelling (a) — extract the four indices, four scalar loads,
/// reassemble with `vld1q_u32`. It is the single, deliberately tiny
/// function every S-box read goes through, so a later tuning phase can
/// swap it for a `vld1q_lane_u32` chain without touching any caller.
///
/// # Safety
///
/// `box_base` must point at the first word of one 256-entry box of an
/// [`SBoxes4`], and every lane of `idx` must be in `0..=255` — the
/// module-level lookup invariant. Then `idx_l * 4 + l < 1024` and each
/// load stays inside that box's 4 KiB.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn lookup4(box_base: *const u32, idx: uint32x4_t) -> uint32x4_t {
    let i0 = vgetq_lane_u32::<0>(idx) as usize;
    let i1 = vgetq_lane_u32::<1>(idx) as usize;
    let i2 = vgetq_lane_u32::<2>(idx) as usize;
    let i3 = vgetq_lane_u32::<3>(idx) as usize;
    let gathered = [
        // SAFETY: lookup invariant — `i0 <= 255`, so `i0 * 4` stays in the box.
        unsafe { *box_base.add(i0 * 4) },
        // SAFETY: lookup invariant — `i1 <= 255`, so `i1 * 4 + 1` stays in the box.
        unsafe { *box_base.add(i1 * 4 + 1) },
        // SAFETY: lookup invariant — `i2 <= 255`, so `i2 * 4 + 2` stays in the box.
        unsafe { *box_base.add(i2 * 4 + 2) },
        // SAFETY: lookup invariant — `i3 <= 255`, so `i3 * 4 + 3` stays in the box.
        unsafe { *box_base.add(i3 * 4 + 3) },
    ];
    // SAFETY: `gathered` is a live 16-byte stack array, readable in full.
    unsafe { vld1q_u32(gathered.as_ptr()) }
}

/// The Blowfish round function, lane-wise: `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`,
/// bytes taken MSB-first, every addition wrapping (`vaddq_u32` wraps, which
/// is what the scalar `wrapping_add` does). The byte masks below are what
/// make the [`lookup4`] gathers safe — see the module-level invariant.
#[target_feature(enable = "neon")]
#[inline]
fn f4(s: &SBoxes4, x: uint32x4_t) -> uint32x4_t {
    let byte_mask = vdupq_n_u32(0xff);
    let a = vshrq_n_u32::<24>(x);
    let b = vandq_u32(vshrq_n_u32::<16>(x), byte_mask);
    let c = vandq_u32(vshrq_n_u32::<8>(x), byte_mask);
    let d = vandq_u32(x, byte_mask);
    // SAFETY: every index lane is a byte — `a` by shifting the other bytes
    // away, `b`/`c`/`d` by the `& 0xff` mask right above — and each
    // `box_base` points at a real box of `s`, so the module-level lookup
    // invariant holds at all four gathers.
    let (va, vb, vc, vd) = unsafe {
        (
            lookup4(s.box_base(0), a),
            lookup4(s.box_base(1), b),
            lookup4(s.box_base(2), c),
            lookup4(s.box_base(3), d),
        )
    };
    vaddq_u32(veorq_u32(vaddq_u32(va, vb), vc), vd)
}

/// One Blowfish block encryption, four lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`.
#[target_feature(enable = "neon")]
#[inline]
fn encipher4(state: &State4, mut l: uint32x4_t, mut r: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    for &p in &state.p[..16] {
        l = veorq_u32(l, p);
        r = veorq_u32(r, f4(&state.s, l));
        core::mem::swap(&mut l, &mut r);
    }
    core::mem::swap(&mut l, &mut r);
    r = veorq_u32(r, state.p[16]);
    l = veorq_u32(l, state.p[17]);
    (l, r)
}

/// Lockstep `Blowfish_expandstate`: fresh state from the pi digits with the
/// key XORed into P, then 521 encryptions mixing salt words into the
/// running block and writing each ciphertext pair back over P (9 pairs)
/// and then S (512 pairs).
///
/// The salt counter `j` is one continuous stream across **both** loops —
/// the P loop consumes 18 words, so the first S-box pair XORs `salt[2]`
/// and `salt[3]`. Faithful to scalar `expand_state`; do not "tidy" the
/// counter into the S loop.
#[target_feature(enable = "neon")]
#[inline]
fn expand_state_v(state: &mut State4, swv: &[uint32x4_t; 4], kwv: &[uint32x4_t; 18]) {
    for ((p, &init), &k) in state.p.iter_mut().zip(P_INIT.iter()).zip(kwv.iter()) {
        *p = veorq_u32(vdupq_n_u32(init), k);
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 4];
    }
    let (mut l, mut r) = (vdupq_n_u32(0), vdupq_n_u32(0));
    let mut j = 0usize;
    for pair in 0..9 {
        l = veorq_u32(l, swv[j % 4]);
        j += 1;
        r = veorq_u32(r, swv[j % 4]);
        j += 1;
        (l, r) = encipher4(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            l = veorq_u32(l, swv[j % 4]);
            j += 1;
            r = veorq_u32(r, swv[j % 4]);
            j += 1;
            (l, r) = encipher4(state, l, r);
            state.s.store_pair(b * 256 + 2 * pair, l, r);
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v`] does, minus the salt mixing.
#[target_feature(enable = "neon")]
#[inline]
fn encrypt_zero_chain_v(state: &mut State4) {
    let (mut l, mut r) = (vdupq_n_u32(0), vdupq_n_u32(0));
    for pair in 0..9 {
        (l, r) = encipher4(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            (l, r) = encipher4(state, l, r);
            state.s.store_pair(b * 256 + 2 * pair, l, r);
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
#[target_feature(enable = "neon")]
#[inline]
fn expand0state_v(state: &mut State4, wv: &[uint32x4_t; 18]) {
    for (p, &w) in state.p.iter_mut().zip(wv.iter()) {
        *p = veorq_u32(*p, w);
    }
    encrypt_zero_chain_v(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words, so
/// the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "neon")]
#[inline]
fn expand0state_salt_v(state: &mut State4, swv: &[uint32x4_t; 4]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        *p = veorq_u32(*p, swv[i & 3]);
    }
    encrypt_zero_chain_v(state);
}

/// Transpose the four lanes' key words into 18 vectors: vector `i` holds
/// word `i` of lanes 0..=3. Runs once per group, so the plain gather below
/// is fine — nothing here is on the cost-loop path.
#[target_feature(enable = "neon")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> [uint32x4_t; 18] {
    debug_assert_eq!(key_words.len(), LANES);
    let mut out = [vdupq_n_u32(0); 18];
    for (i, v) in out.iter_mut().enumerate() {
        let col = [
            key_words[0][i],
            key_words[1][i],
            key_words[2][i],
            key_words[3][i],
        ];
        // SAFETY: `col` is a live 16-byte stack array, readable in full.
        *v = unsafe { vld1q_u32(col.as_ptr()) };
    }
    out
}

/// Transpose the four lanes' salt words into 4 vectors; see
/// [`transpose_keys`].
#[target_feature(enable = "neon")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> [uint32x4_t; 4] {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut out = [vdupq_n_u32(0); 4];
    for (i, v) in out.iter_mut().enumerate() {
        let col = [
            salt_words[0][i],
            salt_words[1][i],
            salt_words[2][i],
            salt_words[3][i],
        ];
        // SAFETY: `col` is a live 16-byte stack array, readable in full.
        *v = unsafe { vld1q_u32(col.as_ptr()) };
    }
    out
}

/// The whole lockstep bcrypt: one key+salt expansion, `2^cost` rounds of
/// key-then-salt expansion (OpenBSD order), then 64 encryptions of the
/// "OrpheanBeholderScryDoubt" constant and a big-endian store per lane.
///
/// Every kernel function carries the same `#[target_feature(enable =
/// "neon")]` scope — stdarch annotates the NEON intrinsics, and a
/// matching-scope caller both calls and inlines them without `unsafe` —
/// so the whole kernel compiles as one NEON unit behind the
/// [`bcrypt_lanes`] entry point.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer dereferences
/// inside are governed by the module-level lookup invariant: S-box
/// gathers index with byte-masked lanes only.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn bcrypt_lanes_impl(
    cost: u32,
    key_words: &[[u32; 18]],
    salt_words: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert!((4..=31).contains(&cost));
    debug_assert_eq!(key_words.len(), LANES);
    debug_assert_eq!(salt_words.len(), LANES);
    debug_assert_eq!(outs.len(), LANES);
    let kwv = transpose_keys(key_words);
    let swv = transpose_salts(salt_words);
    let mut state = State4 {
        p: [vdupq_n_u32(0); 18],
        s: SBoxes4([[0; 4]; 1024]),
    };
    expand_state_v(&mut state, &swv, &kwv);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state_v(&mut state, &kwv);
        expand0state_salt_v(&mut state, &swv);
    }
    // "OrpheanBeholderScryDoubt" as six broadcast words.
    let mut cdata = [
        vdupq_n_u32(0x4f72_7068),
        vdupq_n_u32(0x6561_6e42),
        vdupq_n_u32(0x6568_6f6c),
        vdupq_n_u32(0x6465_7253),
        vdupq_n_u32(0x6372_7944),
        vdupq_n_u32(0x6f75_6274),
    ];
    for _ in 0..64 {
        for pair in 0..3 {
            let (l, r) = encipher4(&state, cdata[2 * pair], cdata[2 * pair + 1]);
            cdata[2 * pair] = l;
            cdata[2 * pair + 1] = r;
        }
    }
    // Split the lanes back out: word `w` of lane `l`'s output is lane `l`
    // of `cdata[w]`, stored big-endian.
    for (w, &cv) in cdata.iter().enumerate() {
        let mut words = [0u32; 4];
        // SAFETY: `words` is a live 16-byte stack array, writable in full.
        unsafe { vst1q_u32(words.as_mut_ptr(), cv) };
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `uint32x4_t` is four `u32`s, so the `*mut u32` view
        // covers exactly the same exclusively-owned stack bytes as `state.p`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.p.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
        crate::wipe::secure_wipe_u32(state.s.0.as_flattened_mut());
        // SAFETY: same reinterpretation as above, for the six `cdata`
        // vectors: 24 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(cdata.as_mut_ptr().cast::<u32>(), 6 * 4)
        });
    }
}

/// The NEON batch kernel: [`LANES`] independent bcrypt hashes in lockstep,
/// one per 32-bit vector lane.
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Neon`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long. Resolve
/// through [`super::bcrypt_lanes_fn`]; never call directly without an
/// availability check.
///
/// # Safety
///
/// * The CPU must support NEON. It is in the aarch64 baseline, and the
///   dispatch table hands this pointer out only for
///   [`super::Backend::Neon`], whose `is_available` is the documented
///   check.
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[target_feature(enable = "neon")]
pub unsafe fn bcrypt_lanes(
    cost: u32,
    key_words: &[[u32; 18]],
    salt_words: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert!((4..=31).contains(&cost));
    debug_assert_eq!(key_words.len(), LANES);
    debug_assert_eq!(salt_words.len(), LANES);
    debug_assert_eq!(outs.len(), LANES);
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — NEON is
    // available on this CPU and the three slices are exactly 4 lanes each —
    // which is everything `bcrypt_lanes_impl` requires.
    unsafe { bcrypt_lanes_impl(cost, key_words, salt_words, outs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base64;
    use crate::eks::{expand_key_words, salt_words};

    /// Reduce a password to kernel key words exactly as `crate::core` will
    /// (mirrors the scalar tests' helper, plus the 72-byte cap
    /// `padded_key` applies: the stream cycles at `min(len + 1, 72)`).
    fn key_words_for(password: &[u8]) -> [u32; 18] {
        let mut key = [0u8; 72];
        let copied = password.len().min(72);
        key[..copied].copy_from_slice(&password[..copied]);
        expand_key_words(&key, (copied + 1).min(72))
    }

    /// Four different passwords and four different salts, one per lane,
    /// must reproduce four scalar hashes bit for bit — including the
    /// OpenBSD `U*U` vector in lane 1.
    #[test]
    fn lanes_match_scalar() {
        let vector_salt = base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.")
            .expect("test salt must decode");
        let passwords: [&[u8]; 4] = [b"", b"U*U", b"hunter2", &[0xAA; 72]];
        let kws: [[u32; 18]; 4] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; 4] = [
            salt_words(&[0x11; 16]),
            salt_words(&vector_salt),
            salt_words(&[0x33; 16]),
            salt_words(&[0x44; 16]),
        ];
        let mut outs = [[0u8; 24]; 4];
        // SAFETY: this module only compiles on aarch64, where NEON is
        // baseline; the three slices are exactly 4 lanes.
        unsafe { bcrypt_lanes(5, &kws, &sws, &mut outs) };
        for lane in 0..LANES {
            let mut expected = [0u8; 24];
            crate::eks::scalar::bcrypt_lanes(
                5,
                &kws[lane..lane + 1],
                &sws[lane..lane + 1],
                core::slice::from_mut(&mut expected),
            );
            assert_eq!(outs[lane], expected, "lane {lane} diverged from scalar");
        }
    }
}
