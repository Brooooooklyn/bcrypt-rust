//! x86-64 AVX2 EksBlowfish, twelve lanes: eight in one ymm group plus four
//! in one xmm group, all in lockstep — one bcrypt hash per 32-bit lane.
//!
//! Ports [`super::scalar`] exactly (same OpenBSD structure as every
//! sibling), with each half a lane-wise copy of an existing kernel: the
//! ymm half is [`super::avx2`]'s 8-lane shape, the xmm half
//! [`super::sse41`]'s 4-lane shape, and the two advance through one shared
//! `rounds16x12!` per encipher. The interleave also widens the dependency
//! graph: the halves' F computations are independent, so the round loop
//! has two F chains in flight instead of one.
//!
//! # Why twelve lanes exist: the L1d wall
//!
//! Every lane carries a 4 KiB private S-box working set (struct-of-arrays,
//! below), so a group touches `lanes * 4 KiB`. Eight lanes = 32 KiB, the
//! exact L1d of Zen 3/4; sixteen lanes = 64 KiB, which is why AVX-512
//! thrashes there. Zen 5, Ice Lake and later Intel cores ship a 48 KiB
//! L1d: twelve lanes = 48 KiB is the widest AVX2 group that still fits.
//! This kernel exists for exactly those microarchitectures.
//!
//! # Self-selecting — no performance win is claimed
//!
//! This backend is never in the static fallback order. It is reachable
//! only through the measured width shootout in [`super`], and only as a
//! candidate: the shootout admits it on hosts whose cpuid leaf-4 L1 data
//! cache is at least 48 KiB, asserts its output byte-identical to the
//! other arms, and picks it only on a strict timed win. Our lab has no
//! 48 KiB-L1d x86 hardware (Zen 5 / Ice Lake+), so **no performance win
//! is claimed**: correctness is force-verified
//! (`BCRYPT_REQUIRE_BACKEND=avx2_12`) on Zen 4 — where its 32 KiB L1d
//! keeps the gate OFF, making the forced run the kernel's execution
//! evidence — and under Rosetta 2, and the kernel only ever runs where
//! the measured shootout picks it.
//!
//! # State layout: two SoA halves
//!
//! The S-boxes live in two structs — [`SBoxes8`] (32 KiB, the ymm half,
//! entry stride 8 words) and [`SBoxes4`] (16 KiB, the xmm half, stride 4)
//! — so each half keeps its sibling kernel's power-of-two stride and
//! prescaled byte-offset masks (`& 0x1fe0` / `& 0xff0`). A single 12-wide
//! SoA would need a stride-12 entry: a non-power-of-two scale no address
//! generator folds for free.
//!
//! # Two lookup flavors, no gather
//!
//! * **Insert** — [`lookup8_insert!`]/[`lookup4_insert!`]: store the
//!   prescaled byte-offset vector, scalar loads, load the result back.
//! * **Extract** — [`lookup8_extract!`]/[`lookup4_extract!`]: `vpextrd`/
//!   `pextrd` each offset lane to a GPR, scalar load, `vpinsrd`/`pinsrd`
//!   rebuild — no stack round-trip.
//!
//! The hardware gather is skipped on purpose: `vpgatherdd` is ymm-wide
//! here, microcoded on AMD — and the microarchitectures this backend
//! targets include Zen 5 — while on Intel's hardware-gather cores the
//! 16-lane AVX-512 backend is the stronger wide pick anyway. The one-time
//! flavor shootout (std + optimized + non-Miri) picks per CPU; debug,
//! `no_std` and Miri builds take **Insert**, the same minimax default as
//! [`super::avx2`].
//!
//! # Lookup invariant
//!
//! Every offset vector handed to the lookups is a prescaled byte value
//! produced inside [`f8_insert!`]/[`f8_extract!`]/[`f4_insert!`]/
//! [`f4_extract!`]: a multiple of the entry stride (32 or 16 bytes) at
//! most `255 * stride`, so `off + lane * 4` can never leave the addressed
//! 256-entry box of the addressed [`SBoxes8`]/[`SBoxes4`]. The invariant
//! is stated once here and referenced at every unsafe dereference below.
//!
//! # Rosetta note
//!
//! Under Rosetta 2 with `RUSTFLAGS="-C target-feature=+avx2"` the kernel
//! executes translated when forced (`BCRYPT_REQUIRE_BACKEND=avx2_12`);
//! Rosetta never advertises AVX-512, so detection never reaches the width
//! shootout that would self-select it. Correctness is the gate under
//! Rosetta, not speed.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — both halves of the [`State12`] expansion state, the `kwv`
//! transposed key schedules and `cdata` — before returning, best-effort
//! like the rest of the crate: compiler spills and transposition-internal
//! temporaries are not chased.
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_and_si128, _mm_extract_epi32, _mm_insert_epi32,
    _mm_loadu_si128, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_slli_epi32,
    _mm_srli_epi32, _mm_store_si128, _mm_storeu_si128, _mm_xor_si128, _mm256_add_epi32,
    _mm256_and_si256, _mm256_extract_epi32, _mm256_insert_epi32, _mm256_loadu_si256,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_slli_epi32,
    _mm256_srli_epi32, _mm256_store_si256, _mm256_storeu_si256, _mm256_xor_si256,
};
use core::sync::atomic::{AtomicU8, Ordering};

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call — the twelve 32-bit lanes (eight ymm + four
/// xmm). Must match `Backend::Avx2x12.lanes()` in `super`.
const LANES: usize = 12;

/// Eight lockstep S-boxes, interleaved by lane — the ymm half, identical
/// layout to [`super::avx2`]'s: entry `b * 256 + i` holds all eight lanes
/// of `(b, i)`, lane `l` at u32 offset `(b * 256 + i) * 8 + l`. 32 KiB,
/// 64-byte aligned, so every entry is 32-byte aligned by construction and
/// aligned vector stores are legal on every entry.
#[repr(C, align(64))]
struct SBoxes8([[u32; 8]; 1024]);

impl SBoxes8 {
    /// Base pointer of box `b` — the address lane lookups index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 2048` stays inside the 8192-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 8) }
    }
}

/// Four lockstep S-boxes, interleaved by lane — the xmm half, identical
/// layout to [`super::sse41`]'s: entry `b * 256 + i` holds all four lanes
/// of `(b, i)`, lane `l` at u32 offset `(b * 256 + i) * 4 + l`. 16 KiB,
/// 64-byte aligned, so every entry is 16-byte aligned by construction.
#[repr(C, align(64))]
struct SBoxes4([[u32; 4]; 1024]);

impl SBoxes4 {
    /// Base pointer of box `b` — the address lane lookups index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 1024` stays inside the 4096-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 4) }
    }
}

/// The lockstep EksBlowfish state: 18 P-vectors plus the interleaved
/// S-boxes, per half — the lane-wise image of scalar `State`, twice.
struct State12 {
    p8: [__m256i; 18],
    p4: [__m128i; 18],
    s8: SBoxes8,
    s4: SBoxes4,
}

/// One scalar S-box load: `$base` (a box pointer) + `$byte_off`, with a
/// zero-instruction opaque `asm!` identity on the address — the same SLP
/// blocker `super::avx2` uses, copied per-backend: uniform `base + off`
/// scalar loads are a gather pattern LLVM can re-form into a microcoded
/// `vpgather*q`, the very instruction the scalar-load flavors exist to
/// avoid. The caller wraps the invocation in `unsafe` and guarantees the
/// module-level lookup invariant (the address stays inside the addressed
/// box and is 4-aligned).
macro_rules! lane_load {
    ($base:expr, $byte_off:expr) => {{
        let mut a = $base as usize + $byte_off;
        core::arch::asm!(
            "/* {0} */",
            inout(reg) a,
            options(pure, nomem, nostack, preserves_flags),
        );
        *(a as *const u32)
    }};
}
/// Gather one S-box word per ymm lane without `vpgatherdd`, as a MACRO so
/// every call site gets a textually inlined copy (the force-inline
/// reasoning is [`super::avx2`]'s: LLVM outlined the scalar-load F as
/// `callq`s, and `#[inline(always)]` on a `#[target_feature]` fn is a hard
/// error, rust#145574): store the *prescaled* byte-offset vector to the
/// stack, eight scalar loads in GPR code, load the result back. `$off`
/// lane `l` must already hold the byte offset `idx_l * 32` (the
/// [`SBoxes8`] entry stride).
///
/// # Safety
///
/// `$box_base` must evaluate to the first word of one 256-entry box of an
/// [`SBoxes8`], and every lane of `$off` must be a multiple of 32 in
/// `0..=8160` — the module-level lookup invariant, prescaled. Then
/// `off_l + l * 4 <= 8188` is a 4-aligned byte offset inside that box's
/// 8 KiB.
macro_rules! lookup8_insert {
    ($box_base:expr, $off:expr) => {{
        let base = ($box_base).cast::<u8>();
        let mut offs = [0u32; 8];
        // SAFETY: `offs` is a live 32-byte stack array, writable in full.
        unsafe { _mm256_storeu_si256(offs.as_mut_ptr().cast::<__m256i>(), $off) };
        let gathered = [
            // SAFETY: lookup invariant (prescaled) — `offs[l]` is a
            // multiple of 32 at most 8160, so `offs[l] + l * 4` is a
            // 4-aligned byte offset inside the box `base` points at
            // (repeated per load).
            unsafe { lane_load!(base, offs[0] as usize) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[1] as usize + 4) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[2] as usize + 8) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[3] as usize + 12) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[4] as usize + 16) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[5] as usize + 20) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[6] as usize + 24) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[7] as usize + 28) },
        ];
        // SAFETY: `gathered` is a live 32-byte stack array, readable in
        // full.
        unsafe { _mm256_loadu_si256(gathered.as_ptr().cast::<__m256i>()) }
    }};
}

/// [`lookup8_insert!`] without the stack round-trip, as a MACRO (same
/// force-inline reasoning): `vpextrd` each prescaled offset lane to a
/// GPR, scalar load, `vpinsrd` rebuild into a vector register.
///
/// # Safety
///
/// Same contract as [`lookup8_insert!`].
macro_rules! lookup8_extract {
    ($box_base:expr, $off:expr) => {{
        let base = ($box_base).cast::<u8>();
        let off = $off;
        // SAFETY: lookup invariant (prescaled), repeated per lane — every
        // extracted lane is a multiple of 32 at most 8160, so `+ l * 4`
        // is a 4-aligned byte offset inside the box `base` points at.
        let g0 = unsafe { lane_load!(base, _mm256_extract_epi32::<0>(off) as usize) };
        // SAFETY: lookup invariant — see lane 0.
        let g1 = unsafe { lane_load!(base, _mm256_extract_epi32::<1>(off) as usize + 4) };
        // SAFETY: lookup invariant — see lane 0.
        let g2 = unsafe { lane_load!(base, _mm256_extract_epi32::<2>(off) as usize + 8) };
        // SAFETY: lookup invariant — see lane 0.
        let g3 = unsafe { lane_load!(base, _mm256_extract_epi32::<3>(off) as usize + 12) };
        // SAFETY: lookup invariant — see lane 0.
        let g4 = unsafe { lane_load!(base, _mm256_extract_epi32::<4>(off) as usize + 16) };
        // SAFETY: lookup invariant — see lane 0.
        let g5 = unsafe { lane_load!(base, _mm256_extract_epi32::<5>(off) as usize + 20) };
        // SAFETY: lookup invariant — see lane 0.
        let g6 = unsafe { lane_load!(base, _mm256_extract_epi32::<6>(off) as usize + 24) };
        // SAFETY: lookup invariant — see lane 0.
        let g7 = unsafe { lane_load!(base, _mm256_extract_epi32::<7>(off) as usize + 28) };
        let mut v = _mm256_setzero_si256();
        v = _mm256_insert_epi32::<0>(v, g0 as i32);
        v = _mm256_insert_epi32::<1>(v, g1 as i32);
        v = _mm256_insert_epi32::<2>(v, g2 as i32);
        v = _mm256_insert_epi32::<3>(v, g3 as i32);
        v = _mm256_insert_epi32::<4>(v, g4 as i32);
        v = _mm256_insert_epi32::<5>(v, g5 as i32);
        v = _mm256_insert_epi32::<6>(v, g6 as i32);
        _mm256_insert_epi32::<7>(v, g7 as i32)
    }};
}
/// [`lookup8_insert!`] for the xmm half: store the *prescaled* byte-offset
/// vector, four scalar loads, load the result back. `$off` lane `l` must
/// already hold the byte offset `idx_l * 16` (the [`SBoxes4`] entry
/// stride). A macro for the same force-inline reason.
///
/// # Safety
///
/// `$box_base` must evaluate to the first word of one 256-entry box of an
/// [`SBoxes4`], and every lane of `$off` must be a multiple of 16 in
/// `0..=4080` — the module-level lookup invariant, prescaled. Then
/// `off_l + l * 4 <= 4092` is a 4-aligned byte offset inside that box's
/// 4 KiB.
macro_rules! lookup4_insert {
    ($box_base:expr, $off:expr) => {{
        let base = ($box_base).cast::<u8>();
        let mut offs = [0u32; 4];
        // SAFETY: `offs` is a live 16-byte stack array, writable in full.
        unsafe { _mm_storeu_si128(offs.as_mut_ptr().cast::<__m128i>(), $off) };
        let gathered = [
            // SAFETY: lookup invariant (prescaled) — `offs[l]` is a
            // multiple of 16 at most 4080, so `offs[l] + l * 4` is a
            // 4-aligned byte offset inside the box `base` points at
            // (repeated per load).
            unsafe { lane_load!(base, offs[0] as usize) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[1] as usize + 4) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[2] as usize + 8) },
            // SAFETY: lookup invariant — see lane 0.
            unsafe { lane_load!(base, offs[3] as usize + 12) },
        ];
        // SAFETY: `gathered` is a live 16-byte stack array, readable in
        // full.
        unsafe { _mm_loadu_si128(gathered.as_ptr().cast::<__m128i>()) }
    }};
}

/// [`lookup4_insert!`] without the stack round-trip, as a MACRO (same
/// force-inline reasoning): `pextrd` each prescaled offset lane to a GPR,
/// scalar load, `pinsrd` rebuild.
///
/// # Safety
///
/// Same contract as [`lookup4_insert!`].
macro_rules! lookup4_extract {
    ($box_base:expr, $off:expr) => {{
        let base = ($box_base).cast::<u8>();
        let off = $off;
        // SAFETY: lookup invariant (prescaled), repeated per lane — every
        // extracted lane is a multiple of 16 at most 4080, so `+ l * 4`
        // is a 4-aligned byte offset inside the box `base` points at.
        let g0 = unsafe { lane_load!(base, _mm_extract_epi32::<0>(off) as usize) };
        // SAFETY: lookup invariant — see lane 0.
        let g1 = unsafe { lane_load!(base, _mm_extract_epi32::<1>(off) as usize + 4) };
        // SAFETY: lookup invariant — see lane 0.
        let g2 = unsafe { lane_load!(base, _mm_extract_epi32::<2>(off) as usize + 8) };
        // SAFETY: lookup invariant — see lane 0.
        let g3 = unsafe { lane_load!(base, _mm_extract_epi32::<3>(off) as usize + 12) };
        let mut v = _mm_setzero_si128();
        v = _mm_insert_epi32::<0>(v, g0 as i32);
        v = _mm_insert_epi32::<1>(v, g1 as i32);
        v = _mm_insert_epi32::<2>(v, g2 as i32);
        _mm_insert_epi32::<3>(v, g3 as i32)
    }};
}
/// The Blowfish round function on the ymm half, insert flavor, as a MACRO
/// — textually inlined at every call site for the same reason
/// [`lookup8_insert!`] is a macro. Identical math to [`super::avx2`]'s F:
/// `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`, bytes MSB-first, wrapping adds.
/// The byte extraction folds the ×32 stride into the mask
/// (`0x1fe0 = 0xff << 5`), so the lookups receive finished byte offsets.
macro_rules! f8_insert {
    ($s:expr, $x:expr) => {{
        let s = $s;
        let x = $x;
        let stride_mask = _mm256_set1_epi32(0x1fe0);
        let a = _mm256_and_si256(_mm256_srli_epi32::<19>(x), stride_mask);
        let b = _mm256_and_si256(_mm256_srli_epi32::<11>(x), stride_mask);
        let c = _mm256_and_si256(_mm256_srli_epi32::<3>(x), stride_mask);
        let d = _mm256_and_si256(_mm256_slli_epi32::<5>(x), stride_mask);
        // Every offset lane is a byte value shifted left by 5 — a
        // multiple of 32 at most 8160 — and each `box_base` points at a
        // real box of `s`, so the module-level lookup invariant holds
        // (prescaled) at all four lookups; that is the contract
        // `lookup8_insert!`'s internal unsafe blocks rely on.
        let (va, vb, vc, vd) = (
            lookup8_insert!(s.box_base(0), a),
            lookup8_insert!(s.box_base(1), b),
            lookup8_insert!(s.box_base(2), c),
            lookup8_insert!(s.box_base(3), d),
        );
        _mm256_add_epi32(_mm256_xor_si256(_mm256_add_epi32(va, vb), vc), vd)
    }};
}

/// [`f8_insert!`], extract flavor: identical math through
/// [`lookup8_extract!`] — same prescaled extraction, no stack round-trip.
/// A macro for the same force-inline reason.
macro_rules! f8_extract {
    ($s:expr, $x:expr) => {{
        let s = $s;
        let x = $x;
        let stride_mask = _mm256_set1_epi32(0x1fe0);
        let a = _mm256_and_si256(_mm256_srli_epi32::<19>(x), stride_mask);
        let b = _mm256_and_si256(_mm256_srli_epi32::<11>(x), stride_mask);
        let c = _mm256_and_si256(_mm256_srli_epi32::<3>(x), stride_mask);
        let d = _mm256_and_si256(_mm256_slli_epi32::<5>(x), stride_mask);
        // Same argument as `f8_insert!` — prescaled byte offsets into
        // real boxes, so the module-level lookup invariant holds at all
        // four lookups; that is the contract `lookup8_extract!`'s
        // internal unsafe blocks rely on.
        let (va, vb, vc, vd) = (
            lookup8_extract!(s.box_base(0), a),
            lookup8_extract!(s.box_base(1), b),
            lookup8_extract!(s.box_base(2), c),
            lookup8_extract!(s.box_base(3), d),
        );
        _mm256_add_epi32(_mm256_xor_si256(_mm256_add_epi32(va, vb), vc), vd)
    }};
}

/// [`f8_insert!`] on the xmm half: identical math with the ×16 stride
/// folded into the mask (`0xff0 = 0xff << 4`), so `(x >> 20) & 0xff0` is
/// `(x >> 24) * 16`, and so on.
macro_rules! f4_insert {
    ($s:expr, $x:expr) => {{
        let s = $s;
        let x = $x;
        let stride_mask = _mm_set1_epi32(0xff0);
        let a = _mm_and_si128(_mm_srli_epi32::<20>(x), stride_mask);
        let b = _mm_and_si128(_mm_srli_epi32::<12>(x), stride_mask);
        let c = _mm_and_si128(_mm_srli_epi32::<4>(x), stride_mask);
        let d = _mm_and_si128(_mm_slli_epi32::<4>(x), stride_mask);
        // Every offset lane is a byte value shifted left by 4 — a
        // multiple of 16 at most 4080 — and each `box_base` points at a
        // real box of `s`, so the module-level lookup invariant holds
        // (prescaled) at all four lookups; that is the contract
        // `lookup4_insert!`'s internal unsafe blocks rely on.
        let (va, vb, vc, vd) = (
            lookup4_insert!(s.box_base(0), a),
            lookup4_insert!(s.box_base(1), b),
            lookup4_insert!(s.box_base(2), c),
            lookup4_insert!(s.box_base(3), d),
        );
        _mm_add_epi32(_mm_xor_si128(_mm_add_epi32(va, vb), vc), vd)
    }};
}

/// [`f4_insert!`], extract flavor: identical math through
/// [`lookup4_extract!`]. A macro for the same force-inline reason.
macro_rules! f4_extract {
    ($s:expr, $x:expr) => {{
        let s = $s;
        let x = $x;
        let stride_mask = _mm_set1_epi32(0xff0);
        let a = _mm_and_si128(_mm_srli_epi32::<20>(x), stride_mask);
        let b = _mm_and_si128(_mm_srli_epi32::<12>(x), stride_mask);
        let c = _mm_and_si128(_mm_srli_epi32::<4>(x), stride_mask);
        let d = _mm_and_si128(_mm_slli_epi32::<4>(x), stride_mask);
        // Same argument as `f4_insert!` — prescaled byte offsets into
        // real boxes, so the module-level lookup invariant holds at all
        // four lookups; that is the contract `lookup4_extract!`'s
        // internal unsafe blocks rely on.
        let (va, vb, vc, vd) = (
            lookup4_extract!(s.box_base(0), a),
            lookup4_extract!(s.box_base(1), b),
            lookup4_extract!(s.box_base(2), c),
            lookup4_extract!(s.box_base(3), d),
        );
        _mm_add_epi32(_mm_xor_si128(_mm_add_epi32(va, vb), vc), vd)
    }};
}

/// The flavor-dispatched F per half, as MACROs (see [`f8_insert!`]).
/// `$flavor` is the caller's const generic ([`FLAVOR_INSERT`] /
/// [`FLAVOR_EXTRACT`]), so the branch folds at monomorphization and no
/// per-lookup branch exists in any kernel path.
macro_rules! f8 {
    ($flavor:ident, $s:expr, $x:expr) => {{
        if $flavor == FLAVOR_EXTRACT {
            f8_extract!($s, $x)
        } else {
            f8_insert!($s, $x)
        }
    }};
}

/// [`f8!`] for the xmm half.
macro_rules! f4 {
    ($flavor:ident, $s:expr, $x:expr) => {{
        if $flavor == FLAVOR_EXTRACT {
            f4_extract!($s, $x)
        } else {
            f4_insert!($s, $x)
        }
    }};
}
/// The 16 Feistel rounds of one lockstep encryption on BOTH halves, as a
/// macro so every caller gets a *textually* inlined copy — the same
/// reasoning as `super::avx2`'s `rounds16!`. Fully unrolled swap-free
/// round pairs; each half's F is independent of the other's, so the two
/// chains interleave in the scheduler (register renaming; the swap-undo
/// `mem::swap`s are free).
macro_rules! rounds16x12 {
    ($state:expr, $flavor:ident, $l8:ident, $r8:ident, $l4:ident, $r4:ident) => {{
        macro_rules! pair {
            ($i:expr) => {{
                $l8 = _mm256_xor_si256($l8, $state.p8[$i]);
                $l4 = _mm_xor_si128($l4, $state.p4[$i]);
                $r8 = _mm256_xor_si256($r8, f8!($flavor, &$state.s8, $l8));
                $r4 = _mm_xor_si128($r4, f4!($flavor, &$state.s4, $l4));
                $r8 = _mm256_xor_si256($r8, $state.p8[$i + 1]);
                $r4 = _mm_xor_si128($r4, $state.p4[$i + 1]);
                $l8 = _mm256_xor_si256($l8, f8!($flavor, &$state.s8, $r8));
                $l4 = _mm_xor_si128($l4, f4!($flavor, &$state.s4, $r4));
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
        $l8 = _mm256_xor_si256($l8, $state.p8[16]);
        $l4 = _mm_xor_si128($l4, $state.p4[16]);
        $r8 = _mm256_xor_si256($r8, $state.p8[17]);
        $r4 = _mm_xor_si128($r4, $state.p4[17]);
        core::mem::swap(&mut $l8, &mut $r8);
        core::mem::swap(&mut $l4, &mut $r4);
    }};
}

/// One Blowfish block encryption, twelve lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`, per half.
#[target_feature(enable = "avx2")]
#[inline]
fn encipher12<const FLAVOR: u8>(
    state: &State12,
    mut l8: __m256i,
    mut r8: __m256i,
    mut l4: __m128i,
    mut r4: __m128i,
) -> (__m256i, __m256i, __m128i, __m128i) {
    rounds16x12!(state, FLAVOR, l8, r8, l4, r4);
    (l8, r8, l4, r4)
}

/// Lockstep `Blowfish_expandstate`: fresh state from the pi digits with the
/// key XORed into P, then 521 encryptions mixing salt words into the
/// running block and writing each ciphertext pair back over P (9 pairs)
/// and then S (512 pairs) — both halves at once.
///
/// The salt counter `j` is one continuous stream across **both** loops —
/// the P loop consumes 18 words, so the first S-box pair XORs `salt[2]`
/// and `salt[3]`. Faithful to scalar `expand_state`; do not "tidy" the
/// counter into the S loop.
#[target_feature(enable = "avx2")]
#[inline]
fn expand_state_v12<const FLAVOR: u8>(
    state: &mut State12,
    swv8: &[__m256i; 4],
    swv4: &[__m128i; 4],
    kwv8: &[__m256i; 18],
    kwv4: &[__m128i; 18],
) {
    for ((p, &init), &k) in state.p8.iter_mut().zip(P_INIT.iter()).zip(kwv8.iter()) {
        *p = _mm256_xor_si256(_mm256_set1_epi32(init as i32), k);
    }
    for ((p, &init), &k) in state.p4.iter_mut().zip(P_INIT.iter()).zip(kwv4.iter()) {
        *p = _mm_xor_si128(_mm_set1_epi32(init as i32), k);
    }
    for (entry, &init) in state.s8.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 8];
    }
    for (entry, &init) in state.s4.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 4];
    }
    let (mut l8, mut r8) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let (mut l4, mut r4) = (_mm_setzero_si128(), _mm_setzero_si128());
    let mut j = 0usize;
    for pair in 0..9 {
        l8 = _mm256_xor_si256(l8, swv8[j % 4]);
        l4 = _mm_xor_si128(l4, swv4[j % 4]);
        j += 1;
        r8 = _mm256_xor_si256(r8, swv8[j % 4]);
        r4 = _mm_xor_si128(r4, swv4[j % 4]);
        j += 1;
        rounds16x12!(state, FLAVOR, l8, r8, l4, r4);
        state.p8[2 * pair] = l8;
        state.p4[2 * pair] = l4;
        state.p8[2 * pair + 1] = r8;
        state.p4[2 * pair + 1] = r4;
    }
    for b in 0..4 {
        for pair in 0..128 {
            l8 = _mm256_xor_si256(l8, swv8[j % 4]);
            l4 = _mm_xor_si128(l4, swv4[j % 4]);
            j += 1;
            r8 = _mm256_xor_si256(r8, swv8[j % 4]);
            r4 = _mm_xor_si128(r4, swv4[j % 4]);
            j += 1;
            rounds16x12!(state, FLAVOR, l8, r8, l4, r4);
            let e = b * 256 + 2 * pair;
            // SAFETY: entries `e`/`e + 1` are 32-byte aligned in `s8` and
            // 16-byte aligned in `s4` by construction (struct align 64,
            // entry strides 32/16) and exclusively owned — aligned stores
            // of the full vectors.
            unsafe {
                _mm256_store_si256(state.s8.0[e].as_mut_ptr().cast::<__m256i>(), l8);
                _mm256_store_si256(state.s8.0[e + 1].as_mut_ptr().cast::<__m256i>(), r8);
                _mm_store_si128(state.s4.0[e].as_mut_ptr().cast::<__m128i>(), l4);
                _mm_store_si128(state.s4.0[e + 1].as_mut_ptr().cast::<__m128i>(), r4);
            }
        }
    }
}
/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v12`] does, minus the salt mixing.
#[target_feature(enable = "avx2")]
#[inline]
fn encrypt_zero_chain_v12<const FLAVOR: u8>(state: &mut State12) {
    let (mut l8, mut r8) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let (mut l4, mut r4) = (_mm_setzero_si128(), _mm_setzero_si128());
    for pair in 0..9 {
        rounds16x12!(state, FLAVOR, l8, r8, l4, r4);
        state.p8[2 * pair] = l8;
        state.p4[2 * pair] = l4;
        state.p8[2 * pair + 1] = r8;
        state.p4[2 * pair + 1] = r4;
    }
    for b in 0..4 {
        for pair in 0..128 {
            rounds16x12!(state, FLAVOR, l8, r8, l4, r4);
            let e = b * 256 + 2 * pair;
            // SAFETY: same argument as `expand_state_v12` — aligned,
            // exclusively owned entries.
            unsafe {
                _mm256_store_si256(state.s8.0[e].as_mut_ptr().cast::<__m256i>(), l8);
                _mm256_store_si256(state.s8.0[e + 1].as_mut_ptr().cast::<__m256i>(), r8);
                _mm_store_si128(state.s4.0[e].as_mut_ptr().cast::<__m128i>(), l4);
                _mm_store_si128(state.s4.0[e + 1].as_mut_ptr().cast::<__m128i>(), r4);
            }
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
#[target_feature(enable = "avx2")]
#[inline]
fn expand0state_v12<const FLAVOR: u8>(
    state: &mut State12,
    kwv8: &[__m256i; 18],
    kwv4: &[__m128i; 18],
) {
    for (p, &w) in state.p8.iter_mut().zip(kwv8.iter()) {
        *p = _mm256_xor_si256(*p, w);
    }
    for (p, &w) in state.p4.iter_mut().zip(kwv4.iter()) {
        *p = _mm_xor_si128(*p, w);
    }
    encrypt_zero_chain_v12::<FLAVOR>(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words, so
/// the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "avx2")]
#[inline]
fn expand0state_salt_v12<const FLAVOR: u8>(
    state: &mut State12,
    swv8: &[__m256i; 4],
    swv4: &[__m128i; 4],
) {
    for (i, p) in state.p8.iter_mut().enumerate() {
        *p = _mm256_xor_si256(*p, swv8[i & 3]);
    }
    for (i, p) in state.p4.iter_mut().enumerate() {
        *p = _mm_xor_si128(*p, swv4[i & 3]);
    }
    encrypt_zero_chain_v12::<FLAVOR>(state);
}

/// Transpose the twelve lanes' key words into 18 + 18 vectors: lanes
/// 0..=7 into ymm vectors, lanes 8..=11 into xmm vectors. Runs once per
/// group, so the plain `setr` gathers below are fine — nothing here is on
/// the cost-loop path.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> ([__m256i; 18], [__m128i; 18]) {
    debug_assert_eq!(key_words.len(), LANES);
    let mut out8 = [_mm256_setzero_si256(); 18];
    for (i, v) in out8.iter_mut().enumerate() {
        *v = _mm256_setr_epi32(
            key_words[0][i] as i32,
            key_words[1][i] as i32,
            key_words[2][i] as i32,
            key_words[3][i] as i32,
            key_words[4][i] as i32,
            key_words[5][i] as i32,
            key_words[6][i] as i32,
            key_words[7][i] as i32,
        );
    }
    let mut out4 = [_mm_setzero_si128(); 18];
    for (i, v) in out4.iter_mut().enumerate() {
        *v = _mm_setr_epi32(
            key_words[8][i] as i32,
            key_words[9][i] as i32,
            key_words[10][i] as i32,
            key_words[11][i] as i32,
        );
    }
    (out8, out4)
}

/// Transpose the twelve lanes' salt words into 4 + 4 vectors; see
/// [`transpose_keys`].
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> ([__m256i; 4], [__m128i; 4]) {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut out8 = [_mm256_setzero_si256(); 4];
    for (i, v) in out8.iter_mut().enumerate() {
        *v = _mm256_setr_epi32(
            salt_words[0][i] as i32,
            salt_words[1][i] as i32,
            salt_words[2][i] as i32,
            salt_words[3][i] as i32,
            salt_words[4][i] as i32,
            salt_words[5][i] as i32,
            salt_words[6][i] as i32,
            salt_words[7][i] as i32,
        );
    }
    let mut out4 = [_mm_setzero_si128(); 4];
    for (i, v) in out4.iter_mut().enumerate() {
        *v = _mm_setr_epi32(
            salt_words[8][i] as i32,
            salt_words[9][i] as i32,
            salt_words[10][i] as i32,
            salt_words[11][i] as i32,
        );
    }
    (out8, out4)
}
/// The whole lockstep bcrypt for one flavor: one key+salt expansion,
/// `2^cost` rounds of key-then-salt expansion (OpenBSD order), then 64
/// encryptions of the "OrpheanBeholderScryDoubt" constant and a big-endian
/// store per lane.
///
/// Every kernel function carries the same `#[target_feature(enable =
/// "avx2")]` scope — stdarch annotates the AVX2 intrinsics (and the AVX2
/// feature subsumes the SSE4.1 ones the xmm half uses), so the whole
/// kernel compiles as one AVX2 unit behind the [`bcrypt_lanes`] entry
/// point.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer dereferences
/// inside are governed by the module-level lookup invariant: S-box
/// lookups index with byte-masked lanes only.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bcrypt12<const FLAVOR: u8>(
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
    let (mut kwv8, mut kwv4) = transpose_keys(key_words);
    let (swv8, swv4) = transpose_salts(salt_words);
    let mut state = State12 {
        p8: [_mm256_setzero_si256(); 18],
        p4: [_mm_setzero_si128(); 18],
        s8: SBoxes8([[0; 8]; 1024]),
        s4: SBoxes4([[0; 4]; 1024]),
    };
    expand_state_v12::<FLAVOR>(&mut state, &swv8, &swv4, &kwv8, &kwv4);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state_v12::<FLAVOR>(&mut state, &kwv8, &kwv4);
        expand0state_salt_v12::<FLAVOR>(&mut state, &swv8, &swv4);
    }
    // "OrpheanBeholderScryDoubt" as six broadcast words, per half.
    let mut cdata8 = [
        _mm256_set1_epi32(0x4f72_7068),
        _mm256_set1_epi32(0x6561_6e42),
        _mm256_set1_epi32(0x6568_6f6c),
        _mm256_set1_epi32(0x6465_7253),
        _mm256_set1_epi32(0x6372_7944),
        _mm256_set1_epi32(0x6f75_6274),
    ];
    let mut cdata4 = [
        _mm_set1_epi32(0x4f72_7068),
        _mm_set1_epi32(0x6561_6e42),
        _mm_set1_epi32(0x6568_6f6c),
        _mm_set1_epi32(0x6465_7253),
        _mm_set1_epi32(0x6372_7944),
        _mm_set1_epi32(0x6f75_6274),
    ];
    for _ in 0..64 {
        for pair in 0..3 {
            let (l8, r8, l4, r4) = encipher12::<FLAVOR>(
                &state,
                cdata8[2 * pair],
                cdata8[2 * pair + 1],
                cdata4[2 * pair],
                cdata4[2 * pair + 1],
            );
            cdata8[2 * pair] = l8;
            cdata8[2 * pair + 1] = r8;
            cdata4[2 * pair] = l4;
            cdata4[2 * pair + 1] = r4;
        }
    }
    // Split the lanes back out: word `w` of lane `l`'s output is lane `l`
    // of `cdata8[w]` (lanes 0..=7) or `cdata4[w]` (lanes 8..=11), stored
    // big-endian.
    for (w, &cv) in cdata8.iter().enumerate() {
        let mut words = [0u32; 8];
        // SAFETY: `words` is a live 32-byte stack array, writable in full.
        unsafe { _mm256_storeu_si256(words.as_mut_ptr().cast::<__m256i>(), cv) };
        for (lane, out) in outs[..8].iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    for (w, &cv) in cdata4.iter().enumerate() {
        let mut words = [0u32; 4];
        // SAFETY: `words` is a live 16-byte stack array, writable in full.
        unsafe { _mm_storeu_si128(words.as_mut_ptr().cast::<__m128i>(), cv) };
        for (lane, out) in outs[8..].iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `__m256i` is eight `u32`s and a `__m128i` four, so the
        // `*mut u32` views cover exactly the same exclusively-owned stack
        // bytes as `state.p8`/`state.p4`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.p8.as_mut_ptr().cast::<u32>(), 18 * 8)
        });
        // SAFETY: same reinterpretation as above, for `state.p4`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.p4.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
        crate::wipe::secure_wipe_u32(state.s8.0.as_flattened_mut());
        crate::wipe::secure_wipe_u32(state.s4.0.as_flattened_mut());
        // SAFETY: same reinterpretation as above, for the six `cdata8`
        // vectors: 48 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(cdata8.as_mut_ptr().cast::<u32>(), 6 * 8)
        });
        // SAFETY: same reinterpretation as above, for the six `cdata4`
        // vectors: 24 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(cdata4.as_mut_ptr().cast::<u32>(), 6 * 4)
        });
        // `kwv8`/`kwv4` are the transposed key schedule — all twelve
        // lanes' password-derived key words — so they are wiped with the
        // state.
        // SAFETY: same reinterpretation as above, for the eighteen `kwv8`
        // vectors: 144 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwv8.as_mut_ptr().cast::<u32>(), 18 * 8)
        });
        // SAFETY: same reinterpretation as above, for the eighteen `kwv4`
        // vectors: 72 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwv4.as_mut_ptr().cast::<u32>(), 18 * 4)
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
    /// Stack round-trip: store offsets, scalar loads, load back.
    Insert,
    /// `vpextrd`/`pextrd` lane → GPR → scalar load → `vpinsrd`/`pinsrd`
    /// rebuild.
    Extract,
}

/// Sentinel meaning "the shootout has not run yet". Not a valid flavor.
const FLAVOR_UNINIT: u8 = 0;
// The values match `super::avx2`'s convention (INSERT = 2, EXTRACT = 3);
// the gather flavor is deliberately absent here (module-level flavor
// notes), so value 1 simply never occurs.
const FLAVOR_INSERT: u8 = 2;
const FLAVOR_EXTRACT: u8 = 3;

/// Cached [`Flavor`] as a `u8`, or [`FLAVOR_UNINIT`].
static CACHED_FLAVOR: AtomicU8 = AtomicU8::new(FLAVOR_UNINIT);

const fn flavor_to_u8(flavor: Flavor) -> u8 {
    match flavor {
        Flavor::Insert => FLAVOR_INSERT,
        Flavor::Extract => FLAVOR_EXTRACT,
    }
}

/// Anything unknown maps to Insert — the conservative default — so the
/// cache can never produce a panic.
const fn flavor_from_u8(value: u8) -> Flavor {
    match value {
        FLAVOR_EXTRACT => Flavor::Extract,
        _ => Flavor::Insert,
    }
}

/// The flavor for this process: one relaxed atomic load on the hot path.
/// Not a `OnceLock` for the same reason as `super::backend`: there is no
/// data to publish, and a racing duplicate pick computes the same class of
/// answer (a wrong pick costs a few percent, never correctness — the
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

/// The one-time flavor shootout: both flavors hash the same fixed
/// deterministic 12-password group at cost 4 — correctness first
/// (identical outputs; a mismatch is a kernel bug, not a tie), then three
/// interleaved timed reps each, min per side, fastest wins.
#[cfg(all(feature = "std", not(debug_assertions), not(miri)))]
fn shootout() -> Flavor {
    use std::time::Instant;

    // SplitMix64, fixed seed: deterministic inputs (not a CSPRNG, nor does
    // it need to be).
    let mut rng = 0xBC79_7A5A_F1A0_0004u64;
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
    // `Backend::Avx2x12` on CPUs advertising AVX2 (or the arch-gated tests,
    // which check `is_available` first); the slices are exactly 12 lanes.
    // SAFETY: per the fn-level comment above — dispatch hands this fn out
    // only for AVX2-capable CPUs; the slices are exactly 12 lanes.
    unsafe { bcrypt12::<FLAVOR_INSERT>(4, &kws, &sws, &mut outs) };
    let reference = outs;
    // SAFETY: per the fn-level comment above — dispatch hands this fn out
    // only for AVX2-capable CPUs; the slices are exactly 12 lanes.
    unsafe { bcrypt12::<FLAVOR_EXTRACT>(4, &kws, &sws, &mut outs) };
    assert_eq!(
        outs, reference,
        "avx2x12 flavor shootout: insert and extract kernels diverged — a kernel bug, not timing"
    );
    let (mut best_insert, mut best_extract) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        let start = Instant::now();
        // SAFETY: per the fn-level comment above — dispatch hands this fn
        // out only for AVX2-capable CPUs; the slices are exactly 12 lanes.
        unsafe { bcrypt12::<FLAVOR_INSERT>(4, &kws, &sws, &mut outs) };
        best_insert = best_insert.min(start.elapsed().as_secs_f64());
        let start = Instant::now();
        // SAFETY: per the fn-level comment above — dispatch hands this fn
        // out only for AVX2-capable CPUs; the slices are exactly 12 lanes.
        unsafe { bcrypt12::<FLAVOR_EXTRACT>(4, &kws, &sws, &mut outs) };
        best_extract = best_extract.min(start.elapsed().as_secs_f64());
        core::hint::black_box(&mut outs);
    }
    if best_insert <= best_extract {
        Flavor::Insert
    } else {
        Flavor::Extract
    }
}
/// The flavor branch: resolved once per GROUP (one relaxed atomic load),
/// then one of two fully monomorphized kernel paths — no per-lookup
/// branch exists in any path.
///
/// # Safety
///
/// Forwards [`bcrypt_lanes`]' contract unchanged.
#[target_feature(enable = "avx2")]
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
        Flavor::Insert => unsafe { bcrypt12::<FLAVOR_INSERT>(cost, key_words, salt_words, outs) },
        // SAFETY: forwards this fn's contract unchanged.
        Flavor::Extract => unsafe { bcrypt12::<FLAVOR_EXTRACT>(cost, key_words, salt_words, outs) },
    }
}

/// The AVX2x12 batch kernel: [`LANES`] independent bcrypt hashes in
/// lockstep, one per 32-bit vector lane (eight ymm + four xmm).
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Avx2x12`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long. Resolve
/// through [`super::bcrypt_lanes_fn`]; never call directly without an
/// availability check.
///
/// # Safety
///
/// * The CPU must support AVX2. The dispatch table hands this pointer out
///   only for [`super::Backend::Avx2x12`], whose `is_available` is the
///   documented check (the 48 KiB-L1d gate in `super` decides only
///   whether detection *offers* this backend, never whether it may run).
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[target_feature(enable = "avx2")]
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
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — AVX2 is
    // available on this CPU and the three slices are exactly 12 lanes
    // each — which is everything `bcrypt_lanes_impl` requires.
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

    /// Gate every kernel-driving test on real AVX2 availability: the 48
    /// KiB-L1d gate in `super` decides only whether detection *offers*
    /// this backend — the kernel itself is plain AVX2 code, so any
    /// AVX2-capable host (Zen 4 included) executes it when asked. Miri
    /// skips too — `detect()` pins Scalar there by design.
    fn avx2x12_available() -> bool {
        !cfg!(miri) && Backend::Avx2x12.is_available()
    }

    /// Deterministic 12-lane key/salt words from an inline SplitMix64 with
    /// a fixed seed (arbitrary u32s are valid key and salt words — they
    /// are pure XOR inputs). Reproduces on rerun.
    fn fixed_words() -> ([[u32; 18]; LANES], [[u32; 4]; LANES]) {
        let mut rng = 0x853C_49E6_748F_EA9Cu64; // splitmix64 constant
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

    /// The flavor correctness check: both flavors must produce
    /// byte-identical output on a fixed input — the shootout asserts the
    /// same invariant before it times anything.
    #[test]
    fn flavors_agree() {
        if !avx2x12_available() {
            return;
        }
        let (kws, sws) = fixed_words();
        let (mut inserted, mut extracted) = ([[0u8; 24]; LANES], [[0u8; 24]; LANES]);
        // SAFETY: availability checked above; the slices are exactly 12
        // lanes.
        unsafe {
            bcrypt12::<FLAVOR_INSERT>(4, &kws, &sws, &mut inserted);
            bcrypt12::<FLAVOR_EXTRACT>(4, &kws, &sws, &mut extracted);
        }
        assert_eq!(inserted, extracted, "insert and extract flavors diverged");
    }

    /// Twelve different passwords and twelve different salts, one per
    /// lane, must reproduce twelve scalar hashes bit for bit — including
    /// the OpenBSD `U*U` vector in lane 1 and the 71/72-byte password
    /// boundary in lanes 6/9.
    #[test]
    fn lanes_match_scalar() {
        if !avx2x12_available() {
            return;
        }
        let vector_salt =
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let passwords: [&[u8]; 12] = [
            b"",
            b"U*U",
            b"hunter2",
            &[0xAA; 72],
            b"correct horse",
            &[0x01],
            &[0x55; 71],
            &[0xFF; 100],
            b"lane eight",
            &[0x5A; 72],
            &[0xFE; 2],
            b"0123456789",
        ];
        let kws: [[u32; 18]; 12] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; 12] = [
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
        ];
        let mut outs = [[0u8; 24]; 12];
        // SAFETY: availability checked above; the slices are exactly 12
        // lanes.
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
        if !avx2x12_available() {
            return;
        }
        let first = flavor();
        assert_ne!(CACHED_FLAVOR.load(Ordering::Relaxed), FLAVOR_UNINIT);
        assert_eq!(flavor_to_u8(first), CACHED_FLAVOR.load(Ordering::Relaxed));
    }
}
