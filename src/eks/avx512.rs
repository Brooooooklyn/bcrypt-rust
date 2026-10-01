//! x86-64 AVX-512F EksBlowfish: sixteen independent bcrypt hashes in
//! lockstep, one per 32-bit zmm lane.
//!
//! Ports [`super::avx2`] lane-for-lane — which in turn ports
//! [`super::scalar`] exactly — with every 256-bit `__m256i` operation
//! widened to a 512-bit `__m512i` one: eight lanes become sixteen, the
//! index shift 3 becomes 4, and the gathers become
//! `_mm512_i32gather_epi32` (`vpgatherdd zmm`). Same OpenBSD structure,
//! same continuous salt-word stream across the P→S transition in
//! `expand_state`, same key-then-salt cost-loop order.
//!
//! **AVX-512F only.** Every intrinsic below is in the foundation set —
//! nothing from AVX-512BW/DQ/VL — so `avx512f` alone is the whole
//! backend contract and `is_x86_feature_detected!("avx512f")` is the
//! complete availability check.
//!
//! # Verification status (read this)
//!
//! Unlike every sibling backend, this one has **not run on the dev
//! host**: Apple Silicon executes x86-64 only through Rosetta 2, which
//! translates AVX2 but traps AVX-512 with `SIGILL`. Correctness here
//! therefore rests on structural fidelity to `super::avx2` (itself
//! differentially tested against `super::scalar`), compilation under
//! `--target x86_64-apple-darwin`, clippy, and code review. Runtime
//! verification happens on CI hardware with AVX-512F; the README says
//! the same. The unit tests gate on a real runtime probe
//! (`avx512_runnable`, in `tests` below) and SKIP where the CPU lacks
//! the feature — they must never `SIGILL`. For the same reason, never
//! test this crate with `RUSTFLAGS="-C target-feature=+avx512f"` on a
//! host without AVX-512: `is_x86_feature_detected!` short-circuits
//! compile-time-enabled features to `true` and the probe would lie.
//!
//! # State layout: struct-of-arrays
//!
//! The four S-boxes are stored interleaved by lane — [`SBoxes16`] — so
//! entry `(b, i)` holds all sixteen lanes contiguously and expansion-
//! chain writes stay full-vector stores. Lookups are gathers: lane `l`
//! of a lookup reads `box_base[idx_l * 16 + l]`.
//!
//! # Two lookup flavors, measured not assumed
//!
//! Same shootout pattern as `super::avx2`:
//!
//! * **Gather** — [`lookup16_gather`], one `vpgatherdd` per box. Fast on
//!   Intel parts with a hardware gather.
//! * **Insert** — [`lookup16_insert`]: store the index vector, sixteen
//!   scalar loads, load the result back.
//!
//! With `std` and optimized code a one-time shootout picks per CPU (both
//! flavors hash the same fixed group at cost 4 with outputs asserted
//! identical — a mismatch is a kernel bug), cached in [`CACHED_FLAVOR`].
//! Debug builds, Miri and `no_std` take **Insert**, the minimax default:
//! on AMD Zen 4/5 the zmm `vpgatherdd` is microcoded (~81 uops per
//! gather on Zen 4 per uops.info), so the wrong Gather pick there loses
//! several-fold to sixteen scalar loads, while the wrong Insert pick on
//! Intel costs only a modest fraction. [`bcrypt_lanes`] reads the cache
//! once per call and branches once per **group** into two monomorphized
//! kernel paths (`bcrypt16::<true>` / `bcrypt16::<false>`), so no
//! per-lookup branch exists.
//!
//! # Lookup invariant
//!
//! Every index vector handed to the lookups is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f16_gather`] /
//! [`f16_insert`], so every lane is in `0..=255` and `idx * 16 + lane`
//! stays inside the addressed 256-entry box — a gather can never leave
//! the 64 KiB [`SBoxes16`]. The invariant is stated once here and
//! referenced at every unsafe dereference below.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — the `State16` expansion state, the `kwv` transposed key
//! schedule and `cdata` — before returning, best-effort like the rest of
//! the crate: compiler spills and transposition-internal temporaries are
//! not chased.

use core::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_and_si512, _mm512_i32gather_epi32, _mm512_loadu_si512,
    _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512, _mm512_slli_epi32,
    _mm512_srli_epi32, _mm512_store_si512, _mm512_storeu_si512, _mm512_xor_si512,
};
use core::sync::atomic::{AtomicU8, Ordering};

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call — the sixteen 32-bit AVX-512 lanes.
const LANES: usize = 16;

/// Sixteen lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is
/// the `(b, i)` Blowfish S-box word of all sixteen lanes, lane `l` at
/// u32 offset `(b * 256 + i) * 16 + l`. 64 KiB, 64-byte aligned, so
/// every entry is 64-byte aligned by construction (the entry stride *is*
/// the struct alignment) and aligned vector stores are legal on every
/// entry.
#[repr(C, align(64))]
struct SBoxes16([[u32; 16]; 1024]);

impl SBoxes16 {
    /// Base pointer of box `b` — the address lane gathers index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 4096` stays inside the 16384-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 16) }
    }
}

/// The lockstep EksBlowfish state: 18 P-vectors plus the interleaved
/// S-boxes — the lane-wise image of scalar `State`.
struct State16 {
    p: [__m512i; 18],
    s: SBoxes16,
}

/// Gather one S-box word per lane with the hardware gather: lane `l`
/// loads `box_base[idx_l * 16 + l]` — vindex `idx * 16 + {0..16}`,
/// scale 4, i.e. byte offset `(idx * 16 + lane) * 4`. `lane_off` is
/// hoisted by the caller (one constant per F evaluation).
///
/// Mind the argument order: the AVX-512 gather intrinsic takes `offsets`
/// FIRST and the base pointer second — the reverse of AVX2's
/// `_mm256_i32gather_epi32(base, vindex)`.
///
/// # Safety
///
/// `box_base` must point at the first word of one 256-entry box of an
/// [`SBoxes16`], and every lane of `idx` must be in `0..=255` — the
/// module-level lookup invariant. Then `idx_l * 16 + l < 4096` and the
/// gather stays inside that box's 16 KiB.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn lookup16_gather(box_base: *const i32, idx: __m512i, lane_off: __m512i) -> __m512i {
    let vindex = _mm512_add_epi32(_mm512_slli_epi32::<4>(idx), lane_off);
    // SAFETY: the contract above is forwarded from the caller; `vindex`
    // lanes are exactly the bounded `idx_l * 16 + l`. Offsets-first
    // argument order, per the note above.
    unsafe { _mm512_i32gather_epi32::<4>(vindex, box_base) }
}

/// Gather one S-box word per lane without `vpgatherdd`: store the index
/// vector to the stack, sixteen scalar loads in GPR code, load the
/// result back. The flavor that wins where gathers are microcoded.
///
/// # Safety
///
/// Same contract as [`lookup16_gather`].
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn lookup16_insert(box_base: *const u32, idx: __m512i) -> __m512i {
    let mut ix = [0u32; 16];
    // SAFETY: `ix` is a live 64-byte stack array, writable in full.
    unsafe { _mm512_storeu_si512(ix.as_mut_ptr().cast::<__m512i>(), idx) };
    let gathered = [
        // SAFETY: lookup invariant — `ix[l] <= 255`, so `ix[l] * 16 + l`
        // stays inside the box `box_base` points at (repeated per load).
        unsafe { *box_base.add(ix[0] as usize * 16) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[1] as usize * 16 + 1) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[2] as usize * 16 + 2) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[3] as usize * 16 + 3) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[4] as usize * 16 + 4) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[5] as usize * 16 + 5) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[6] as usize * 16 + 6) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[7] as usize * 16 + 7) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[8] as usize * 16 + 8) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[9] as usize * 16 + 9) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[10] as usize * 16 + 10) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[11] as usize * 16 + 11) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[12] as usize * 16 + 12) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[13] as usize * 16 + 13) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[14] as usize * 16 + 14) },
        // SAFETY: lookup invariant — see lane 0.
        unsafe { *box_base.add(ix[15] as usize * 16 + 15) },
    ];
    // SAFETY: `gathered` is a live 64-byte stack array, readable in full.
    unsafe { _mm512_loadu_si512(gathered.as_ptr().cast::<__m512i>()) }
}

/// The Blowfish round function, lane-wise, gather flavor:
/// `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`, bytes taken MSB-first, every
/// addition wrapping (`_mm512_add_epi32` wraps, which is what the scalar
/// `wrapping_add` does). The byte masks below are what make the
/// [`lookup16_gather`] gathers safe — see the module-level invariant.
#[target_feature(enable = "avx512f")]
#[inline]
fn f16_gather(s: &SBoxes16, x: __m512i) -> __m512i {
    let byte_mask = _mm512_set1_epi32(0xff);
    let a = _mm512_srli_epi32::<24>(x);
    let b = _mm512_and_si512(_mm512_srli_epi32::<16>(x), byte_mask);
    let c = _mm512_and_si512(_mm512_srli_epi32::<8>(x), byte_mask);
    let d = _mm512_and_si512(x, byte_mask);
    // Lane gather offsets `{0..16}`, hoisted out of the four lookups.
    let lane_off = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    // SAFETY: every index lane is a byte — `a` by shifting the other bytes
    // away, `b`/`c`/`d` by the `& 0xff` mask right above — and each
    // `box_base` points at a real box of `s`, so the module-level lookup
    // invariant holds at all four gathers.
    let (va, vb, vc, vd) = unsafe {
        (
            lookup16_gather(s.box_base(0).cast::<i32>(), a, lane_off),
            lookup16_gather(s.box_base(1).cast::<i32>(), b, lane_off),
            lookup16_gather(s.box_base(2).cast::<i32>(), c, lane_off),
            lookup16_gather(s.box_base(3).cast::<i32>(), d, lane_off),
        )
    };
    _mm512_add_epi32(_mm512_xor_si512(_mm512_add_epi32(va, vb), vc), vd)
}

/// [`f16_gather`], insert flavor: identical math through
/// [`lookup16_insert`] — see the module-level flavor notes.
#[target_feature(enable = "avx512f")]
#[inline]
fn f16_insert(s: &SBoxes16, x: __m512i) -> __m512i {
    let byte_mask = _mm512_set1_epi32(0xff);
    let a = _mm512_srli_epi32::<24>(x);
    let b = _mm512_and_si512(_mm512_srli_epi32::<16>(x), byte_mask);
    let c = _mm512_and_si512(_mm512_srli_epi32::<8>(x), byte_mask);
    let d = _mm512_and_si512(x, byte_mask);
    // SAFETY: same argument as `f16_gather` — byte-masked indices into
    // real boxes, so the module-level lookup invariant holds at all four
    // loads.
    let (va, vb, vc, vd) = unsafe {
        (
            lookup16_insert(s.box_base(0), a),
            lookup16_insert(s.box_base(1), b),
            lookup16_insert(s.box_base(2), c),
            lookup16_insert(s.box_base(3), d),
        )
    };
    _mm512_add_epi32(_mm512_xor_si512(_mm512_add_epi32(va, vb), vc), vd)
}

/// The flavor-dispatched F: `GATHER` is a const generic, so the branch
/// folds at monomorphization and no per-lookup branch exists in either
/// kernel path.
#[target_feature(enable = "avx512f")]
#[inline]
fn f16<const GATHER: bool>(s: &SBoxes16, x: __m512i) -> __m512i {
    if GATHER { f16_gather(s, x) } else { f16_insert(s, x) }
}

/// The 16 Feistel rounds of one lockstep encryption, as a macro so every
/// caller gets a *textually* inlined copy — the same reasoning as
/// `super::avx2`'s `rounds16!` (an outlined `#[target_feature]` rounds
/// fn let LLVM keep the box-base math per call). Fully unrolled
/// swap-free round pairs: the pair ends in reference post-swap
/// orientation, so the whitening lands on the exchanged registers and
/// the final swap-undo is the one `mem::swap` after the pairs (register
/// renaming; free).
macro_rules! rounds16 {
    ($state:expr, $gather:ident, $l:ident, $r:ident) => {{
        macro_rules! pair {
            ($i:expr) => {{
                $l = _mm512_xor_si512($l, $state.p[$i]);
                $r = _mm512_xor_si512($r, f16::<$gather>(&$state.s, $l));
                $r = _mm512_xor_si512($r, $state.p[$i + 1]);
                $l = _mm512_xor_si512($l, f16::<$gather>(&$state.s, $r));
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
        $l = _mm512_xor_si512($l, $state.p[16]);
        $r = _mm512_xor_si512($r, $state.p[17]);
        core::mem::swap(&mut $l, &mut $r);
    }};
}

/// One Blowfish block encryption, sixteen lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`.
#[target_feature(enable = "avx512f")]
#[inline]
fn encipher16<const GATHER: bool>(
    state: &State16,
    mut l: __m512i,
    mut r: __m512i,
) -> (__m512i, __m512i) {
    rounds16!(state, GATHER, l, r);
    (l, r)
}

/// Lockstep `Blowfish_expandstate`: fresh state from the pi digits with
/// the key XORed into P, then 521 encryptions mixing salt words into the
/// running block and writing each ciphertext pair back over P (9 pairs)
/// and then S (512 pairs).
///
/// The salt counter `j` is one continuous stream across **both** loops —
/// the P loop consumes 18 words, so the first S-box pair XORs `salt[2]`
/// and `salt[3]`. Faithful to scalar `expand_state`; do not "tidy" the
/// counter into the S loop.
#[target_feature(enable = "avx512f")]
#[inline]
fn expand_state_v16<const GATHER: bool>(
    state: &mut State16,
    swv: &[__m512i; 4],
    kwv: &[__m512i; 18],
) {
    for ((p, &init), &k) in state.p.iter_mut().zip(P_INIT.iter()).zip(kwv.iter()) {
        *p = _mm512_xor_si512(_mm512_set1_epi32(init as i32), k);
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 16];
    }
    let (mut l, mut r) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let mut j = 0usize;
    for pair in 0..9 {
        l = _mm512_xor_si512(l, swv[j % 4]);
        j += 1;
        r = _mm512_xor_si512(r, swv[j % 4]);
        j += 1;
        rounds16!(state, GATHER, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            l = _mm512_xor_si512(l, swv[j % 4]);
            j += 1;
            r = _mm512_xor_si512(r, swv[j % 4]);
            j += 1;
            rounds16!(state, GATHER, l, r);
            let e = b * 256 + 2 * pair;
            // SAFETY: entries `e`/`e + 1` are 64-byte aligned by
            // construction (struct align 64, entry stride 64) and
            // exclusively owned — aligned stores of the full vectors.
            unsafe {
                _mm512_store_si512(state.s.0[e].as_mut_ptr().cast::<__m512i>(), l);
                _mm512_store_si512(state.s.0[e + 1].as_mut_ptr().cast::<__m512i>(), r);
            }
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v16`] does, minus the salt mixing.
#[target_feature(enable = "avx512f")]
#[inline]
fn encrypt_zero_chain_v16<const GATHER: bool>(state: &mut State16) {
    let (mut l, mut r) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    for pair in 0..9 {
        rounds16!(state, GATHER, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            rounds16!(state, GATHER, l, r);
            let e = b * 256 + 2 * pair;
            // SAFETY: same argument as `expand_state_v16` — 64-byte
            // aligned, exclusively owned entries.
            unsafe {
                _mm512_store_si512(state.s.0[e].as_mut_ptr().cast::<__m512i>(), l);
                _mm512_store_si512(state.s.0[e + 1].as_mut_ptr().cast::<__m512i>(), r);
            }
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
#[target_feature(enable = "avx512f")]
#[inline]
fn expand0state_v16<const GATHER: bool>(state: &mut State16, wv: &[__m512i; 18]) {
    for (p, &w) in state.p.iter_mut().zip(wv.iter()) {
        *p = _mm512_xor_si512(*p, w);
    }
    encrypt_zero_chain_v16::<GATHER>(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words,
/// so the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "avx512f")]
#[inline]
fn expand0state_salt_v16<const GATHER: bool>(state: &mut State16, swv: &[__m512i; 4]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        *p = _mm512_xor_si512(*p, swv[i & 3]);
    }
    encrypt_zero_chain_v16::<GATHER>(state);
}

/// Transpose the sixteen lanes' key words into 18 vectors: vector `i`
/// holds word `i` of lanes 0..=15. Runs once per group, so the plain
/// `setr` gathers below are fine — nothing here is on the cost-loop
/// path.
#[target_feature(enable = "avx512f")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> [__m512i; 18] {
    debug_assert_eq!(key_words.len(), LANES);
    let mut out = [_mm512_setzero_si512(); 18];
    for (i, v) in out.iter_mut().enumerate() {
        *v = _mm512_setr_epi32(
            key_words[0][i] as i32,
            key_words[1][i] as i32,
            key_words[2][i] as i32,
            key_words[3][i] as i32,
            key_words[4][i] as i32,
            key_words[5][i] as i32,
            key_words[6][i] as i32,
            key_words[7][i] as i32,
            key_words[8][i] as i32,
            key_words[9][i] as i32,
            key_words[10][i] as i32,
            key_words[11][i] as i32,
            key_words[12][i] as i32,
            key_words[13][i] as i32,
            key_words[14][i] as i32,
            key_words[15][i] as i32,
        );
    }
    out
}

/// Transpose the sixteen lanes' salt words into 4 vectors; see
/// [`transpose_keys`].
#[target_feature(enable = "avx512f")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> [__m512i; 4] {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut out = [_mm512_setzero_si512(); 4];
    for (i, v) in out.iter_mut().enumerate() {
        *v = _mm512_setr_epi32(
            salt_words[0][i] as i32,
            salt_words[1][i] as i32,
            salt_words[2][i] as i32,
            salt_words[3][i] as i32,
            salt_words[4][i] as i32,
            salt_words[5][i] as i32,
            salt_words[6][i] as i32,
            salt_words[7][i] as i32,
            salt_words[8][i] as i32,
            salt_words[9][i] as i32,
            salt_words[10][i] as i32,
            salt_words[11][i] as i32,
            salt_words[12][i] as i32,
            salt_words[13][i] as i32,
            salt_words[14][i] as i32,
            salt_words[15][i] as i32,
        );
    }
    out
}

/// The whole lockstep bcrypt for one flavor: one key+salt expansion,
/// `2^cost` rounds of key-then-salt expansion (OpenBSD order), then 64
/// encryptions of the "OrpheanBeholderScryDoubt" constant and a
/// big-endian store per lane.
///
/// Every kernel function carries the same `#[target_feature(enable =
/// "avx512f")]` scope — stdarch annotates the AVX-512F intrinsics, and
/// a matching-scope caller both calls and inlines them without `unsafe`
/// — so the whole kernel compiles as one AVX-512F unit behind the
/// [`bcrypt_lanes`] entry point.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer
/// dereferences inside are governed by the module-level lookup
/// invariant: S-box gathers index with byte-masked lanes only.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn bcrypt16<const GATHER: bool>(
    cost: u32,
    key_words: &[[u32; 18]],
    salt_words: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert!((4..=31).contains(&cost));
    debug_assert_eq!(key_words.len(), LANES);
    debug_assert_eq!(salt_words.len(), LANES);
    debug_assert_eq!(outs.len(), LANES);
    // `mut` serves only the `zeroize` wipe at the bottom of this fn.
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut kwv = transpose_keys(key_words);
    let swv = transpose_salts(salt_words);
    let mut state = State16 {
        p: [_mm512_setzero_si512(); 18],
        s: SBoxes16([[0; 16]; 1024]),
    };
    expand_state_v16::<GATHER>(&mut state, &swv, &kwv);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state_v16::<GATHER>(&mut state, &kwv);
        expand0state_salt_v16::<GATHER>(&mut state, &swv);
    }
    // "OrpheanBeholderScryDoubt" as six broadcast words.
    let mut cdata = [
        _mm512_set1_epi32(0x4f72_7068),
        _mm512_set1_epi32(0x6561_6e42),
        _mm512_set1_epi32(0x6568_6f6c),
        _mm512_set1_epi32(0x6465_7253),
        _mm512_set1_epi32(0x6372_7944),
        _mm512_set1_epi32(0x6f75_6274),
    ];
    for _ in 0..64 {
        for pair in 0..3 {
            let (l, r) = encipher16::<GATHER>(&state, cdata[2 * pair], cdata[2 * pair + 1]);
            cdata[2 * pair] = l;
            cdata[2 * pair + 1] = r;
        }
    }
    // Split the lanes back out: word `w` of lane `l`'s output is lane `l`
    // of `cdata[w]`, stored big-endian.
    for (w, &cv) in cdata.iter().enumerate() {
        let mut words = [0u32; 16];
        // SAFETY: `words` is a live 64-byte stack array, writable in full.
        unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast::<__m512i>(), cv) };
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `__m512i` is sixteen `u32`s, so the `*mut u32` view
        // covers exactly the same exclusively-owned stack bytes as
        // `state.p`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.p.as_mut_ptr().cast::<u32>(), 18 * 16)
        });
        crate::wipe::secure_wipe_u32(state.s.0.as_flattened_mut());
        // SAFETY: same reinterpretation as above, for the six `cdata`
        // vectors: 96 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(cdata.as_mut_ptr().cast::<u32>(), 6 * 16)
        });
        // `kwv` is the transposed key schedule — all sixteen lanes'
        // password-derived key words — so it is wiped with the state.
        // SAFETY: same reinterpretation as above, for the eighteen `kwv`
        // vectors: 288 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwv.as_mut_ptr().cast::<u32>(), 18 * 16)
        });
    }
}

// ---------------------------------------------------------------------------
// Flavor selection
// ---------------------------------------------------------------------------

/// Which S-box lookup spelling the kernel runs — see the module-level
/// flavor notes.
#[derive(Copy, Clone)]
enum Flavor {
    /// `_mm512_i32gather_epi32` (`vpgatherdd zmm`).
    Gather,
    /// Stack round-trip: store indices, sixteen scalar loads, load back.
    Insert,
}

/// Sentinel meaning "the shootout has not run yet". Not a valid flavor.
const FLAVOR_UNINIT: u8 = 0;
const FLAVOR_GATHER: u8 = 1;
const FLAVOR_INSERT: u8 = 2;

/// Cached [`Flavor`] as a `u8`, or [`FLAVOR_UNINIT`].
static CACHED_FLAVOR: AtomicU8 = AtomicU8::new(FLAVOR_UNINIT);

const fn flavor_to_u8(flavor: Flavor) -> u8 {
    match flavor {
        Flavor::Gather => FLAVOR_GATHER,
        Flavor::Insert => FLAVOR_INSERT,
    }
}

/// Anything unknown maps to Insert — the conservative default — so the
/// cache can never produce a panic.
const fn flavor_from_u8(value: u8) -> Flavor {
    match value {
        FLAVOR_GATHER => Flavor::Gather,
        _ => Flavor::Insert,
    }
}

/// The flavor for this process: one relaxed atomic load on the hot path.
/// Not a `OnceLock` for the same reason as `super::backend`: there is no
/// data to publish, and a racing duplicate pick computes the same class
/// of answer (a wrong pick costs a few percent, never correctness — the
/// shootout asserts both flavors agree before timing anything).
#[inline]
fn flavor() -> Flavor {
    let cached = CACHED_FLAVOR.load(Ordering::Relaxed);
    if cached == FLAVOR_UNINIT {
        pick_and_cache()
    } else {
        flavor_from_u8(cached)
    }
}

/// Pick and populate the cache. Outlined so [`flavor`] stays tiny.
#[cold]
#[inline(never)]
fn pick_and_cache() -> Flavor {
    let picked = pick_flavor();
    CACHED_FLAVOR.store(flavor_to_u8(picked), Ordering::Relaxed);
    picked
}

/// With a clock and optimized code: measure both flavors and keep the
/// winner. In debug builds or under Miri, timing unoptimized/interpreted
/// intrinsics measures the codegen mode, not the µarch — and without
/// `std` there is no clock at all. All three take the minimax default
/// [`Flavor::Insert`] (module-level flavor notes).
#[cfg(all(feature = "std", not(debug_assertions), not(miri)))]
fn pick_flavor() -> Flavor {
    shootout()
}

/// See the measuring variant above.
#[cfg(not(all(feature = "std", not(debug_assertions), not(miri))))]
fn pick_flavor() -> Flavor {
    Flavor::Insert
}

/// The one-time gather-vs-insert shootout: both flavors hash the same
/// fixed deterministic 16-password group at cost 4 — correctness first
/// (identical outputs; a mismatch is a kernel bug, not a tie), then
/// three interleaved timed reps each, min per side, faster wins.
#[cfg(all(feature = "std", not(debug_assertions), not(miri)))]
fn shootout() -> Flavor {
    use std::time::Instant;

    // SplitMix64, fixed seed: deterministic inputs (not a CSPRNG, nor does
    // it need to be).
    let mut rng = 0xBC79_7A5A_F1A0_0002u64;
    let mut next_u32 = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    };
    let mut kws = [[0u32; 18]; LANES];
    let mut sws = [[0u32; 4]; LANES];
    for w in kws.as_flattened_mut().iter_mut().chain(sws.as_flattened_mut()) {
        *w = next_u32();
    }
    let mut outs = [[0u8; 24]; LANES];
    // SAFETY (this whole fn): the shootout is reached only through
    // `bcrypt_lanes`, which the dispatch table hands out solely for
    // `Backend::Avx512` on CPUs advertising AVX-512F (or the arch-gated
    // tests, which check `is_available` first); the slices are exactly
    // 16 lanes.
    unsafe { bcrypt16::<true>(4, &kws, &sws, &mut outs) };
    let reference = outs;
    unsafe { bcrypt16::<false>(4, &kws, &sws, &mut outs) };
    assert_eq!(
        outs, reference,
        "avx512 flavor shootout: gather and insert kernels diverged — a kernel bug, not timing"
    );
    let (mut best_gather, mut best_insert) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        let start = Instant::now();
        unsafe { bcrypt16::<true>(4, &kws, &sws, &mut outs) };
        best_gather = best_gather.min(start.elapsed().as_secs_f64());
        let start = Instant::now();
        unsafe { bcrypt16::<false>(4, &kws, &sws, &mut outs) };
        best_insert = best_insert.min(start.elapsed().as_secs_f64());
        core::hint::black_box(&mut outs);
    }
    if best_gather <= best_insert { Flavor::Gather } else { Flavor::Insert }
}

/// The flavor branch: resolved once per GROUP (one relaxed atomic load),
/// then one of two fully monomorphized kernel paths — no per-lookup
/// branch exists in either path.
///
/// # Safety
///
/// Forwards [`bcrypt_lanes`]' contract unchanged.
#[target_feature(enable = "avx512f")]
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
    match flavor() {
        // SAFETY: forwards this fn's contract unchanged.
        Flavor::Gather => unsafe { bcrypt16::<true>(cost, key_words, salt_words, outs) },
        // SAFETY: forwards this fn's contract unchanged.
        Flavor::Insert => unsafe { bcrypt16::<false>(cost, key_words, salt_words, outs) },
    }
}

/// The AVX-512F batch kernel: [`LANES`] independent bcrypt hashes in
/// lockstep, one per 32-bit vector lane.
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Avx512`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long.
/// Resolve through [`super::bcrypt_lanes_fn`]; never call directly
/// without an availability check.
///
/// # Safety
///
/// * The CPU must support AVX-512F. The dispatch table hands this
///   pointer out only for [`super::Backend::Avx512`], whose
///   `is_available` is the documented check.
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[target_feature(enable = "avx512f")]
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
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — AVX-512F
    // is available on this CPU and the three slices are exactly 16 lanes
    // each — which is everything `bcrypt_lanes_impl` requires.
    unsafe { bcrypt_lanes_impl(cost, key_words, salt_words, outs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base64;
    use crate::eks::{Backend, expand_key_words, salt_words};

    /// Reduce a password to kernel key words exactly as `crate::core`
    /// will (mirrors the scalar tests' helper, plus the 72-byte cap
    /// `padded_key` applies: the stream cycles at `min(len + 1, 72)`).
    fn key_words_for(password: &[u8]) -> [u32; 18] {
        let mut key = [0u8; 72];
        let copied = password.len().min(72);
        key[..copied].copy_from_slice(&password[..copied]);
        expand_key_words(&key, (copied + 1).min(72))
    }

    /// Gate every kernel-driving test on REAL runtime AVX-512F
    /// availability: the dev host runs x86-64 tests under Rosetta 2,
    /// which cannot execute AVX-512 at all, so a plain
    /// `cargo test --target x86_64-apple-darwin` must skip, not
    /// `SIGILL`. With `std` this probe is
    /// `std::arch::is_x86_feature_detected!("avx512f")` (a real cpuid
    /// check — false under Rosetta); without `std` it is
    /// `cfg!(target_feature = "avx512f")`. Miri skips too — `detect()`
    /// pins Scalar there by design.
    fn avx512_runnable() -> bool {
        !cfg!(miri) && Backend::Avx512.is_available()
    }

    /// Deterministic 16-lane key/salt words from an inline SplitMix64
    /// with a fixed seed (arbitrary u32s are valid key and salt words —
    /// they are pure XOR inputs). Reproduces on rerun.
    fn fixed_words() -> ([[u32; 18]; LANES], [[u32; 4]; LANES]) {
        let mut rng = 0x853C_49E6_748F_EA9Cu64; // splitmix64-style constant
        let mut next_u32 = move || {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as u32
        };
        let mut kws = [[0u32; 18]; LANES];
        let mut sws = [[0u32; 4]; LANES];
        for w in kws.as_flattened_mut().iter_mut().chain(sws.as_flattened_mut()) {
            *w = next_u32();
        }
        (kws, sws)
    }

    /// The gather-vs-insert correctness check: both flavors must produce
    /// byte-identical output on a fixed input. This is the test that
    /// pins the `vpgatherdd` index math against the scalar-load
    /// spelling — the shootout asserts the same invariant before it
    /// times anything.
    #[test]
    fn flavors_agree() {
        if !avx512_runnable() {
            return;
        }
        let (kws, sws) = fixed_words();
        let (mut gathered, mut inserted) = ([[0u8; 24]; LANES], [[0u8; 24]; LANES]);
        // SAFETY: availability checked above; the slices are exactly 16 lanes.
        unsafe {
            bcrypt16::<true>(4, &kws, &sws, &mut gathered);
            bcrypt16::<false>(4, &kws, &sws, &mut inserted);
        }
        assert_eq!(gathered, inserted, "gather and insert flavors diverged");
    }

    /// Sixteen different passwords and sixteen different salts, one per
    /// lane, must reproduce sixteen scalar hashes bit for bit —
    /// including the OpenBSD `U*U` vector in lane 1.
    #[test]
    fn lanes_match_scalar() {
        if !avx512_runnable() {
            return;
        }
        let vector_salt =
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let passwords: [&[u8]; 16] = [
            b"",
            b"U*U",
            b"hunter2",
            &[0xAA; 72],
            b"correct horse",
            &[0x01],
            &[0x55; 71],
            &[0xFF; 100],
            b"lane eight",
            &[0x00; 1],
            &[0x5A; 73],
            b"OrpheanBeholderScryDoubt",
            &[0x10; 17],
            "🔑 emoji key".as_bytes(),
            &[0x7F; 36],
            b"a much longer password that still fits within the seventy-two byte cap ok",
        ];
        let kws: [[u32; 18]; 16] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; 16] = [
            salt_words(&[0x11; 16]),
            salt_words(&vector_salt),
            salt_words(&[0x33; 16]),
            salt_words(&[0x44; 16]),
            salt_words(&[0x55; 16]),
            salt_words(&[0x66; 16]),
            salt_words(&[0x77; 16]),
            salt_words(&[0x88; 16]),
            salt_words(&[0x99; 16]),
            salt_words(&[0xAA; 16]),
            salt_words(&[0xBB; 16]),
            salt_words(&[0xCC; 16]),
            salt_words(&[0xDD; 16]),
            salt_words(&[0xEE; 16]),
            salt_words(&[0xFF; 16]),
            salt_words(&[0x00; 16]),
        ];
        let mut outs = [[0u8; 24]; 16];
        // SAFETY: availability checked above; the slices are exactly 16 lanes.
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

    /// The flavor pick runs (the shootout, in release+std builds) and
    /// caches a valid value; repeated reads agree.
    #[test]
    fn flavor_pick_is_cached() {
        if !avx512_runnable() {
            return;
        }
        let first = flavor();
        assert_ne!(CACHED_FLAVOR.load(Ordering::Relaxed), FLAVOR_UNINIT);
        assert_eq!(flavor_to_u8(first), CACHED_FLAVOR.load(Ordering::Relaxed));
    }
}
