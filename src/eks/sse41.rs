//! x86 / x86-64 SSE4.1 EksBlowfish: four independent bcrypt hashes in
//! lockstep, one per 32-bit xmm lane.
//!
//! Ports [`super::scalar`] exactly — same OpenBSD structure, same
//! continuous salt-word stream across the P→S transition in
//! `expand_state`, same key-then-salt cost-loop order — with every 32-bit
//! operation widened to a lane-wise `__m128i` one. Scalar `wrapping_add`
//! becomes `_mm_add_epi32` (wraps, which is what Blowfish wants); every
//! XOR becomes `_mm_xor_si128`.
//!
//! # State layout: struct-of-arrays
//!
//! The four S-boxes are stored interleaved by lane — [`SBoxes4x`] — so
//! entry `(b, i)` holds all four lanes contiguously and expansion-chain
//! writes stay full-vector stores. Lookups are gathers: lane `l` of a
//! lookup reads `box_base[idx_l * 4 + l]`.
//!
//! # No gather at this ISA — two lookup spellings
//!
//! SSE4.1 has no gather instruction (`vpgatherdd` arrives with AVX2), so
//! every lookup is four scalar loads either way; the only choice is how
//! the indices cross between the vector and GPR domains:
//!
//! * [`lookup4_stack`] (active) — one `_mm_storeu_si128` of the index
//!   vector, four scalar loads, one `_mm_loadu_si128` of the results: two
//!   domain crossings per lookup, and the four loads are independent.
//! * [`lookup4_pinsr`] — one `_mm_extract_epi32` per index plus one
//!   `_mm_insert_epi32` per result: eight crossings per lookup, and the
//!   inserts chain serially (each feeds the next). Inactive, kept as a
//!   tuning alternative; checked against the active spelling in the tests.
//!
//! Measured `cargo bench --bench micro --target x86_64-apple-darwin --
//! --backend sse41 --vs-scalar`, cost 5 batch 64, under Rosetta 2 on the
//! dev host (translation-bound: a sanity check, not µarch truth — see the
//! Rosetta note): stack 810.5 hashes/s, pinsr 812.0 hashes/s. No visible
//! difference at the bcrypt level, so the simpler stack round-trip ships.
//!
//! # Rosetta note
//!
//! Rosetta 2 advertises SSE4.1 in `cpuid` (and does not advertise AVX2
//! unless it was enabled at compile time), so on the aarch64 dev host a
//! plain `cargo test --target x86_64-apple-darwin` runs this backend
//! natively translated — correctness is the gate under Rosetta, not speed.
//!
//! # Lookup invariant
//!
//! Every index vector handed to the lookups is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f4`], so every
//! lane is in `0..=255` and `idx * 4 + lane` stays inside the addressed
//! 256-entry box — a gather can never leave the 16 KiB [`SBoxes4x`]. The
//! invariant is stated once here and referenced at every unsafe
//! dereference below.

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _mm_add_epi32, _mm_and_si128, _mm_extract_epi32, _mm_insert_epi32,
    _mm_loadu_si128, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_srli_epi32,
    _mm_store_si128, _mm_storeu_si128, _mm_xor_si128,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_and_si128, _mm_extract_epi32, _mm_insert_epi32,
    _mm_loadu_si128, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_srli_epi32,
    _mm_store_si128, _mm_storeu_si128, _mm_xor_si128,
};

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call — the four 32-bit SSE lanes.
const LANES: usize = 4;

/// Four lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is the
/// `(b, i)` Blowfish S-box word of all four lanes, lane `l` at u32 offset
/// `(b * 256 + i) * 4 + l`. 16 KiB, 64-byte aligned, so every entry is
/// 16-byte aligned by construction (the entry stride divides the struct
/// alignment) and aligned vector stores are legal on every entry.
#[repr(C, align(64))]
struct SBoxes4x([[u32; 4]; 1024]);

impl SBoxes4x {
    /// Base pointer of box `b` — the address lane gathers index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 1024` stays inside the 4096-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 4) }
    }
}

/// The lockstep EksBlowfish state: 18 P-vectors plus the interleaved
/// S-boxes — the lane-wise image of scalar `State`.
struct State4x {
    p: [__m128i; 18],
    s: SBoxes4x,
}

/// Gather one S-box word per lane: lane `l` loads `box_base[idx_l * 4 + l]`.
///
/// # Safety
///
/// `box_base` must point at the first word of one 256-entry box of an
/// [`SBoxes4x`], and every lane of `idx` must be in `0..=255` — the
/// module-level lookup invariant. Then `idx_l * 4 + l < 1024` and each
/// load stays inside that box's 4 KiB.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn lookup4(box_base: *const u32, idx: __m128i) -> __m128i {
    // SAFETY: forwarded unchanged — the spellings share this contract.
    unsafe { lookup4_stack(box_base, idx) }
}

/// [`lookup4`] spelling "stack" (active — see the module-level numbers):
/// store the index vector, four scalar loads, load the results back.
///
/// # Safety
///
/// Same contract as [`lookup4`].
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn lookup4_stack(box_base: *const u32, idx: __m128i) -> __m128i {
    let mut ix = [0u32; 4];
    // SAFETY: `ix` is a live 16-byte stack array, writable in full.
    unsafe { _mm_storeu_si128(ix.as_mut_ptr().cast::<__m128i>(), idx) };
    let gathered = [
        // SAFETY: lookup invariant — `ix[l] <= 255`, so `ix[l] * 4 + l`
        // stays inside the box `box_base` points at (repeated per load).
        unsafe { *box_base.add(ix[0] as usize * 4) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[1] as usize * 4 + 1) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[2] as usize * 4 + 2) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[3] as usize * 4 + 3) },
    ];
    // SAFETY: `gathered` is a live 16-byte stack array, readable in full.
    unsafe { _mm_loadu_si128(gathered.as_ptr().cast::<__m128i>()) }
}

/// [`lookup4`] spelling "pinsr" (inactive — see the module-level numbers):
/// extract each index with `_mm_extract_epi32`, insert each loaded word
/// with `_mm_insert_epi32`.
///
/// # Safety
///
/// Same contract as [`lookup4`].
#[allow(dead_code)] // tuning alternative; checked by lookup_spellings_agree
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn lookup4_pinsr(box_base: *const u32, idx: __m128i) -> __m128i {
    // SAFETY: lookup invariant — each extracted lane is `<= 255`, so
    // `idx_l * 4 + l < 1024` stays inside the box `box_base` points at
    // (repeated per load).
    unsafe {
        let v = _mm_insert_epi32::<0>(
            _mm_setzero_si128(),
            *box_base.add(_mm_extract_epi32::<0>(idx) as u32 as usize * 4) as i32,
        );
        let v = _mm_insert_epi32::<1>(
            v,
            *box_base.add(_mm_extract_epi32::<1>(idx) as u32 as usize * 4 + 1) as i32,
        );
        let v = _mm_insert_epi32::<2>(
            v,
            *box_base.add(_mm_extract_epi32::<2>(idx) as u32 as usize * 4 + 2) as i32,
        );
        _mm_insert_epi32::<3>(
            v,
            *box_base.add(_mm_extract_epi32::<3>(idx) as u32 as usize * 4 + 3) as i32,
        )
    }
}

/// The Blowfish round function, lane-wise: `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`,
/// bytes taken MSB-first, every addition wrapping (`_mm_add_epi32` wraps,
/// which is what the scalar `wrapping_add` does). The byte masks below are
/// what make the [`lookup4`] gathers safe — see the module-level invariant.
#[target_feature(enable = "sse4.1")]
#[inline]
fn f4(s: &SBoxes4x, x: __m128i) -> __m128i {
    let byte_mask = _mm_set1_epi32(0xff);
    let a = _mm_srli_epi32::<24>(x);
    let b = _mm_and_si128(_mm_srli_epi32::<16>(x), byte_mask);
    let c = _mm_and_si128(_mm_srli_epi32::<8>(x), byte_mask);
    let d = _mm_and_si128(x, byte_mask);
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
    _mm_add_epi32(_mm_xor_si128(_mm_add_epi32(va, vb), vc), vd)
}

/// The 16 Feistel rounds of one lockstep encryption, as a macro so every
/// caller gets a *textually* inlined copy — the same reasoning as
/// `super::avx2`'s `rounds16!` (an outlined `#[target_feature]` rounds fn
/// let LLVM keep the box-base math per call). Fully unrolled swap-free
/// round pairs: the pair ends in reference post-swap orientation, so the
/// whitening lands on the exchanged registers and the final swap-undo is
/// the one `mem::swap` after the pairs (register renaming; free).
macro_rules! rounds16 {
    ($state:expr, $l:ident, $r:ident) => {{
        macro_rules! pair {
            ($i:expr) => {{
                $l = _mm_xor_si128($l, $state.p[$i]);
                $r = _mm_xor_si128($r, f4(&$state.s, $l));
                $r = _mm_xor_si128($r, $state.p[$i + 1]);
                $l = _mm_xor_si128($l, f4(&$state.s, $r));
            }};
        }
        pair!(0);
        pair!(2);
        pair!(4);
        pair!(6);
        pair!(8);
        pair!(10);
        pair!(12);
        pair!(14);
        $l = _mm_xor_si128($l, $state.p[16]);
        $r = _mm_xor_si128($r, $state.p[17]);
        core::mem::swap(&mut $l, &mut $r);
    }};
}

/// One Blowfish block encryption, four lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`.
#[target_feature(enable = "sse4.1")]
#[inline]
fn encipher4(state: &State4x, mut l: __m128i, mut r: __m128i) -> (__m128i, __m128i) {
    rounds16!(state, l, r);
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
#[target_feature(enable = "sse4.1")]
#[inline]
fn expand_state_v4(state: &mut State4x, swv: &[__m128i; 4], kwv: &[__m128i; 18]) {
    for ((p, &init), &k) in state.p.iter_mut().zip(P_INIT.iter()).zip(kwv.iter()) {
        *p = _mm_xor_si128(_mm_set1_epi32(init as i32), k);
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 4];
    }
    let (mut l, mut r) = (_mm_setzero_si128(), _mm_setzero_si128());
    let mut j = 0usize;
    for pair in 0..9 {
        l = _mm_xor_si128(l, swv[j % 4]);
        j += 1;
        r = _mm_xor_si128(r, swv[j % 4]);
        j += 1;
        rounds16!(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            l = _mm_xor_si128(l, swv[j % 4]);
            j += 1;
            r = _mm_xor_si128(r, swv[j % 4]);
            j += 1;
            rounds16!(state, l, r);
            let e = b * 256 + 2 * pair;
            // SAFETY: entries `e`/`e + 1` are 16-byte aligned by
            // construction (struct align 64, entry stride 16) and
            // exclusively owned — aligned stores of the full vectors.
            unsafe {
                _mm_store_si128(state.s.0[e].as_mut_ptr().cast::<__m128i>(), l);
                _mm_store_si128(state.s.0[e + 1].as_mut_ptr().cast::<__m128i>(), r);
            }
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v4`] does, minus the salt mixing.
#[target_feature(enable = "sse4.1")]
#[inline]
fn encrypt_zero_chain_v4(state: &mut State4x) {
    let (mut l, mut r) = (_mm_setzero_si128(), _mm_setzero_si128());
    for pair in 0..9 {
        rounds16!(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            rounds16!(state, l, r);
            let e = b * 256 + 2 * pair;
            // SAFETY: same argument as `expand_state_v4` — 16-byte aligned,
            // exclusively owned entries.
            unsafe {
                _mm_store_si128(state.s.0[e].as_mut_ptr().cast::<__m128i>(), l);
                _mm_store_si128(state.s.0[e + 1].as_mut_ptr().cast::<__m128i>(), r);
            }
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
#[target_feature(enable = "sse4.1")]
#[inline]
fn expand0state_v4(state: &mut State4x, wv: &[__m128i; 18]) {
    for (p, &w) in state.p.iter_mut().zip(wv.iter()) {
        *p = _mm_xor_si128(*p, w);
    }
    encrypt_zero_chain_v4(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words, so
/// the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "sse4.1")]
#[inline]
fn expand0state_salt_v4(state: &mut State4x, swv: &[__m128i; 4]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        *p = _mm_xor_si128(*p, swv[i & 3]);
    }
    encrypt_zero_chain_v4(state);
}

/// Transpose the four lanes' key words into 18 vectors: vector `i` holds
/// word `i` of lanes 0..=3. Runs once per group, so the plain `setr`
/// gathers below are fine — nothing here is on the cost-loop path.
#[target_feature(enable = "sse4.1")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> [__m128i; 18] {
    debug_assert_eq!(key_words.len(), LANES);
    let mut out = [_mm_setzero_si128(); 18];
    for (i, v) in out.iter_mut().enumerate() {
        *v = _mm_setr_epi32(
            key_words[0][i] as i32,
            key_words[1][i] as i32,
            key_words[2][i] as i32,
            key_words[3][i] as i32,
        );
    }
    out
}

/// Transpose the four lanes' salt words into 4 vectors; see
/// [`transpose_keys`].
#[target_feature(enable = "sse4.1")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> [__m128i; 4] {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut out = [_mm_setzero_si128(); 4];
    for (i, v) in out.iter_mut().enumerate() {
        *v = _mm_setr_epi32(
            salt_words[0][i] as i32,
            salt_words[1][i] as i32,
            salt_words[2][i] as i32,
            salt_words[3][i] as i32,
        );
    }
    out
}

/// The whole lockstep bcrypt: one key+salt expansion, `2^cost` rounds of
/// key-then-salt expansion (OpenBSD order), then 64 encryptions of the
/// "OrpheanBeholderScryDoubt" constant and a big-endian store per lane.
///
/// Every kernel function carries the same `#[target_feature(enable =
/// "sse4.1")]` scope — stdarch annotates the SSE4.1 intrinsics, and a
/// matching-scope caller both calls and inlines them without `unsafe` —
/// so the whole kernel compiles as one SSE4.1 unit behind the
/// [`bcrypt_lanes`] entry point.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer dereferences
/// inside are governed by the module-level lookup invariant: S-box
/// gathers index with byte-masked lanes only.
#[target_feature(enable = "sse4.1")]
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
    let mut state = State4x {
        p: [_mm_setzero_si128(); 18],
        s: SBoxes4x([[0; 4]; 1024]),
    };
    expand_state_v4(&mut state, &swv, &kwv);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state_v4(&mut state, &kwv);
        expand0state_salt_v4(&mut state, &swv);
    }
    // "OrpheanBeholderScryDoubt" as six broadcast words.
    let mut cdata = [
        _mm_set1_epi32(0x4f72_7068),
        _mm_set1_epi32(0x6561_6e42),
        _mm_set1_epi32(0x6568_6f6c),
        _mm_set1_epi32(0x6465_7253),
        _mm_set1_epi32(0x6372_7944),
        _mm_set1_epi32(0x6f75_6274),
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
        unsafe { _mm_storeu_si128(words.as_mut_ptr().cast::<__m128i>(), cv) };
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `__m128i` is four `u32`s, so the `*mut u32` view
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

/// The SSE4.1 batch kernel: [`LANES`] independent bcrypt hashes in
/// lockstep, one per 32-bit vector lane.
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Sse41`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long. Resolve
/// through [`super::bcrypt_lanes_fn`]; never call directly without an
/// availability check.
///
/// # Safety
///
/// * The CPU must support SSE4.1. The dispatch table hands this pointer
///   out only for [`super::Backend::Sse41`], whose `is_available` is the
///   documented check.
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[target_feature(enable = "sse4.1")]
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
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — SSE4.1 is
    // available on this CPU and the three slices are exactly 4 lanes each —
    // which is everything `bcrypt_lanes_impl` requires.
    unsafe { bcrypt_lanes_impl(cost, key_words, salt_words, outs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base64;
    use crate::eks::{Backend, expand_key_words, salt_words};

    /// Reduce a password to kernel key words exactly as `crate::core` will
    /// (mirrors the scalar tests' helper, plus the 72-byte cap
    /// `padded_key` applies: the stream cycles at `min(len + 1, 72)`).
    fn key_words_for(password: &[u8]) -> [u32; 18] {
        let mut key = [0u8; 72];
        let copied = password.len().min(72);
        key[..copied].copy_from_slice(&password[..copied]);
        expand_key_words(&key, (copied + 1).min(72))
    }

    /// Gate every kernel-driving test on real SSE4.1 availability: SSE4.1
    /// is in no x86(-64) baseline, so a plain `cargo test` on a pre-Penryn
    /// x86 host must skip, not `SIGILL`. Miri skips too — `detect()` pins
    /// Scalar there by design.
    fn sse41_available() -> bool {
        !cfg!(miri) && Backend::Sse41.is_available()
    }

    /// The inactive `_mm_insert_epi32` spelling must agree with the active
    /// [`lookup4_stack`] on every index vector — it stays in the tree as a
    /// tuning alternative, so it stays checked.
    #[test]
    fn lookup_spellings_agree() {
        if !sse41_available() {
            return;
        }
        let mut boxes = SBoxes4x([[0; 4]; 1024]);
        for (entry, words) in boxes.0.iter_mut().enumerate() {
            *words = [
                entry as u32,
                (entry as u32) ^ 0xAAAA_AAAA,
                (entry as u32).wrapping_mul(31),
                !(entry as u32),
            ];
        }
        for i in 0..256u32 {
            // SAFETY: availability checked above; the vector build has no
            // pointer contract beyond that.
            let idx = unsafe {
                _mm_setr_epi32(i as i32, (255 - i) as i32, (i ^ 0x5A) as i32, ((i * 7) & 0xFF) as i32)
            };
            // SAFETY: availability checked above; `idx` lanes are all in
            // `0..=255` by construction, and `box_base` points at a real
            // 256-entry box of `boxes` — the module-level lookup invariant.
            let (a, b) = unsafe {
                (
                    lookup4_stack(boxes.box_base((i % 4) as usize), idx),
                    lookup4_pinsr(boxes.box_base((i % 4) as usize), idx),
                )
            };
            let (mut wa, mut wb) = ([0u32; 4], [0u32; 4]);
            // SAFETY: both are live 16-byte stack arrays, writable in full.
            unsafe {
                _mm_storeu_si128(wa.as_mut_ptr().cast::<__m128i>(), a);
                _mm_storeu_si128(wb.as_mut_ptr().cast::<__m128i>(), b);
            }
            assert_eq!(wa, wb, "spellings diverged at index set {i}");
        }
    }

    /// Four different passwords and four different salts, one per lane,
    /// must reproduce four scalar hashes bit for bit — including the
    /// OpenBSD `U*U` vector in lane 1.
    #[test]
    fn lanes_match_scalar() {
        if !sse41_available() {
            return;
        }
        let vector_salt =
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let passwords: [&[u8]; 4] = [b"", b"U*U", b"hunter2", &[0xAA; 72]];
        let kws: [[u32; 18]; 4] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; 4] = [
            salt_words(&[0x11; 16]),
            salt_words(&vector_salt),
            salt_words(&[0x33; 16]),
            salt_words(&[0x44; 16]),
        ];
        let mut outs = [[0u8; 24]; 4];
        // SAFETY: availability checked above; the slices are exactly 4 lanes.
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
