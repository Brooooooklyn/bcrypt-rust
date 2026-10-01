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
//! entry `(b, i)` holds all four lanes contiguously and expansion-chain
//! writes stay four plain word stores. Lookups are gathers: lane `l` of a
//! lookup reads `box_base[idx_l * 4 + l]`. NEON has no gather instruction,
//! so every gather is four scalar loads — see the [`lookup4`] spellings.
//!
//! # Tuning summary (M5 Max, `cargo bench --bench micro`, cost 5 batch 64)
//!
//! The kernel is ~34k serial Blowfish encryptions per hash; the expansion
//! S-box loops (512 of 521 encryptions per chain) dominate. Measured
//! hashes/s, scalar always ~820:
//!
//! * 862 — [`lookup4`] spellings (a) stack array and (b) `vld1q_lane`
//!   chain tie: LLVM lowers (a) to (b)'s loads. Baseline ratio 1.05×.
//! * 895 — spelling (c), prescaled per-lane byte offsets.
//! * 1097 — [`f4_split`]: whole F per lane in GPRs (`ldr [base, w, uxtw
//!   #2]` gathers), macro-inlined + unrolled rounds, u16 extracts.
//! * 2253 — `rounds16_scalar!`: lanes live in eight GPRs through the 16
//!   rounds; no vector register is touched inside the round loop, so the
//!   insert→extract round trip leaves the dependency chain. Ratio 2.7–2.9×.
//!
//! # Lookup invariant
//!
//! Every index vector handed to [`lookup4`] is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f4`], so every
//! lane is in `0..=255` and `idx * 4 + lane` stays inside the addressed
//! 256-entry box — a gather can never leave the 16 KiB [`SBoxes4`]. The
//! invariant is stated once here and referenced at every unsafe
//! dereference below.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — the `State4` expansion state, the `kwv` transposed key
//! schedule and `cdata` — before returning, best-effort like the rest of
//! the crate: per-expansion scalar-rounds scratch (the `pw` P-array
//! snapshot, the lane registers) and compiler spills are not chased.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_u32, veorq_u32, vgetq_lane_u16, vgetq_lane_u32,
    vld1q_lane_u32, vld1q_u32, vreinterpretq_u16_u32, vsetq_lane_u32, vshlq_n_u32, vshrq_n_u32,
    vst1q_u32,
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
}

/// The lockstep EksBlowfish state: 18 P-vectors plus the interleaved
/// S-boxes — the lane-wise image of scalar `State`.
struct State4 {
    p: [uint32x4_t; 18],
    s: SBoxes4,
}

/// Lane byte offsets: lane `l` gathers at `box_base + idx_l * 16 + l * 4`
/// (the entry stride is 16 bytes — four u32 lanes per entry).
const LANE_BYTE_OFFSETS: [u32; 4] = [0, 4, 8, 12];

/// Gather one S-box word per lane: lane `l` loads `box_base[idx_l * 4 + l]`.
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
    // SAFETY: forwarded unchanged — the spellings share this contract.
    unsafe { lookup4_scaled(box_base, idx) }
}

/// [`lookup4`] spelling (a): extract the four indices, four scalar loads
/// through a stack array, reassemble with `vld1q_u32`. Inactive: LLVM
/// rewrites the round-trip into `ldr s` + three `ld1` lane inserts anyway.
#[allow(dead_code)] // kept for re-tuning on other cores; see lookup4
#[target_feature(enable = "neon")]
#[inline]
unsafe fn lookup4_stack(box_base: *const u32, idx: uint32x4_t) -> uint32x4_t {
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

/// [`lookup4`] spelling (b): a zeroed vector and four `vld1q_lane_u32`
/// inserts — one load per lane straight into its slot, no stack array.
/// Inactive: identical to (a) after LLVM lowering, and both lose to the
/// prescaled gather (the numbers live at [`lookup4_scaled`]).
#[allow(dead_code)] // kept for re-tuning on other cores; see lookup4
#[target_feature(enable = "neon")]
#[inline]
unsafe fn lookup4_lane(box_base: *const u32, idx: uint32x4_t) -> uint32x4_t {
    let v = vdupq_n_u32(0);
    // SAFETY: lookup invariant — every lane of `idx` is `<= 255`, so each
    // `idx_l * 4 + l < 1024` stays inside the box `box_base` points at.
    unsafe {
        let v = vld1q_lane_u32::<0>(box_base.add(vgetq_lane_u32::<0>(idx) as usize * 4), v);
        let v = vld1q_lane_u32::<1>(box_base.add(vgetq_lane_u32::<1>(idx) as usize * 4 + 1), v);
        let v = vld1q_lane_u32::<2>(box_base.add(vgetq_lane_u32::<2>(idx) as usize * 4 + 2), v);
        vld1q_lane_u32::<3>(box_base.add(vgetq_lane_u32::<3>(idx) as usize * 4 + 3), v)
    }
}

/// [`lookup4`] spelling (c), active: scale the index vector to *byte*
/// offsets lane-wise — `idx * 16 + {0,4,8,12}` per lane — so the extracted
/// GPR value is the finished in-box offset and each gather is one
/// register-offset `ld1`. Spellings (a)/(b) make the GPR side recompute
/// that: one `lsl` plus two `add`s per lane (LLVM lowers the stack array
/// of (a) to (b)'s loads, so they tie). Measured on M5 Max,
/// `cargo bench --bench micro -- --vs-scalar`, cost 5 batch 64:
/// (a) 862.5 hashes/s, (b) 862.5, (c) 894.7.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn lookup4_scaled(box_base: *const u32, idx: uint32x4_t) -> uint32x4_t {
    // SAFETY: `LANE_BYTE_OFFSETS` is a live 16-byte array, readable in full.
    let lane_off = unsafe { vld1q_u32(LANE_BYTE_OFFSETS.as_ptr()) };
    let off = vaddq_u32(vshlq_n_u32::<4>(idx), lane_off);
    let byte_base = box_base.cast::<u8>();
    let v = vdupq_n_u32(0);
    // SAFETY: lookup invariant — `idx_l <= 255`, so the byte offset
    // `idx_l * 16 + 4 * l <= 0xFFC` stays inside the box's 4 KiB, and the
    // u32 read at that 4-byte-aligned offset is one box word.
    unsafe {
        let v = vld1q_lane_u32::<0>(byte_base.add(vgetq_lane_u32::<0>(off) as usize).cast(), v);
        let v = vld1q_lane_u32::<1>(byte_base.add(vgetq_lane_u32::<1>(off) as usize).cast(), v);
        let v = vld1q_lane_u32::<2>(byte_base.add(vgetq_lane_u32::<2>(off) as usize).cast(), v);
        vld1q_lane_u32::<3>(byte_base.add(vgetq_lane_u32::<3>(off) as usize).cast(), v)
    }
}

/// The Blowfish round function, lane-wise: `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`,
/// bytes taken MSB-first, every addition wrapping (`vaddq_u32` wraps, which
/// is what the scalar `wrapping_add` does). The byte masks below are what
/// make the [`lookup4`] gathers safe — see the module-level invariant.
///
/// This is the pure-SIMD F: inactive (the P-loops and cdata loop run
/// [`f4_split`], the S-box loops run `rounds16_scalar!` — see the module
/// "Tuning summary"), kept as the reference datapath the `lookup4`
/// spellings hang off, and checked against `f4_split` in the tests.
#[allow(dead_code)] // tested (f4_variants_agree); see the tuning summary
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

/// [`f4`] spelling "split": extract each lane's word to the GPR side once,
/// split the bytes there, run the four gathers and the `((a + b) ^ c) + d`
/// combine in scalar code, and insert the results with `vsetq_lane_u32`.
/// Trades SIMD ops for GPR ALU ops, which issue on different pipes.
///
/// The word reaches the GPR side as two `umov.h` 16-bit extracts (lo =
/// bytes d,c; hi = bytes b,a) rather than one 32-bit `mov.s`: each half
/// then splits with one `ubfiz`/`ubfx` apiece, moving work off the
/// saturated GPR-ALU pipes onto the SIMD-move pipes. Measured whole-kernel
/// with the rounds macro-inlined and unrolled: 1097 hashes/s vs 895 for
/// the SIMD [`f4`] datapath (M5 Max, cost 5 batch 64).
#[target_feature(enable = "neon")]
#[inline]
fn f4_split(s: &SBoxes4, x: uint32x4_t) -> uint32x4_t {
    /// Per-lane scalar F from the two half-word extracts.
    #[inline]
    fn halves(s: &SBoxes4, lo: u32, hi: u32, lane: usize) -> u32 {
        let (a, b) = ((hi >> 8) as usize, (hi & 0xff) as usize);
        let (c, d) = ((lo >> 8) as usize, (lo & 0xff) as usize);
        // SAFETY: lookup invariant — `a`/`b`/`c`/`d` are bytes by the masks
        // above, so `idx * 4 + lane < 1024` stays inside each box.
        unsafe {
            let va = *s.box_base(0).add(a * 4 + lane);
            let vb = *s.box_base(1).add(b * 4 + lane);
            let vc = *s.box_base(2).add(c * 4 + lane);
            let vd = *s.box_base(3).add(d * 4 + lane);
            (va.wrapping_add(vb) ^ vc).wrapping_add(vd)
        }
    }
    let x16 = vreinterpretq_u16_u32(x);
    // Every `halves` call site upholds the module-level lookup invariant
    // (its byte masks bound each index); the extract/insert intrinsics are
    // safe aarch64 NEON, so no `unsafe` block is needed here.
    let v = vsetq_lane_u32::<0>(
        halves(s, vgetq_lane_u16::<0>(x16).into(), vgetq_lane_u16::<1>(x16).into(), 0),
        vdupq_n_u32(0),
    );
    let v = vsetq_lane_u32::<1>(
        halves(s, vgetq_lane_u16::<2>(x16).into(), vgetq_lane_u16::<3>(x16).into(), 1),
        v,
    );
    let v = vsetq_lane_u32::<2>(
        halves(s, vgetq_lane_u16::<4>(x16).into(), vgetq_lane_u16::<5>(x16).into(), 2),
        v,
    );
    vsetq_lane_u32::<3>(
        halves(s, vgetq_lane_u16::<6>(x16).into(), vgetq_lane_u16::<7>(x16).into(), 3),
        v,
    )
}

/// One lane of the Blowfish round function in scalar code:
/// `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`, bytes MSB-first, wrapping adds.
/// Pure GPR code, so it carries no `#[target_feature]` and inlines
/// anywhere. This is the datapath of `rounds16_scalar!`.
#[inline]
unsafe fn f_word(s: &SBoxes4, w: u32, lane: usize) -> u32 {
    let a = (w >> 24) as usize;
    let b = ((w >> 16) & 0xff) as usize;
    let c = ((w >> 8) & 0xff) as usize;
    let d = (w & 0xff) as usize;
    // SAFETY: lookup invariant — `a`/`b`/`c`/`d` are bytes by the masks
    // above and `lane < 4` at every call site, so `idx * 4 + lane < 1024`
    // stays inside each box.
    unsafe {
        let va = *s.box_base(0).add(a * 4 + lane);
        let vb = *s.box_base(1).add(b * 4 + lane);
        let vc = *s.box_base(2).add(c * 4 + lane);
        let vd = *s.box_base(3).add(d * 4 + lane);
        (va.wrapping_add(vb) ^ vc).wrapping_add(vd)
    }
}

/// The P-array as per-lane scalar words: `pw[i][lane]`. The expansion
/// S-loops run with a frozen P (the chain's P writes all happen in the
/// P-loop), so the rounds can read P from this stack copy — four plain
/// `ldr`s per round instead of one vector load plus four lane extracts.
#[inline]
fn p_words(p: &[uint32x4_t; 18]) -> [[u32; 4]; 18] {
    let mut pw = [[0u32; 4]; 18];
    for (w, v) in pw.iter_mut().zip(p.iter()) {
        // SAFETY: `w` is a live 16-byte stack array, writable in full.
        unsafe { vst1q_u32(w.as_mut_ptr(), *v) };
    }
    pw
}

/// The four lanes of a vector as a plain word array (once per chain —
/// the scalar rounds keep lanes in GPRs across the whole S-box loop).
#[inline]
fn lanes_of(v: uint32x4_t) -> [u32; 4] {
    let mut w = [0u32; 4];
    // SAFETY: `w` is a live 16-byte stack array, writable in full.
    unsafe { vst1q_u32(w.as_mut_ptr(), v) };
    w
}

/// The 16 Feistel rounds with all four lanes in GPR word arrays: no vector
/// register is touched between the extracts before it and the stores after
/// it, so the round-to-round dependency chain is pure scalar code and the
/// four lanes' chains overlap in the out-of-order window. Splits the
/// difference between the two above: vector rounds pay a serial
/// insert→extract round trip per round; scalar rounds pay four stack
/// `ldr`s for P per round. Measured whole-kernel: 1097 hashes/s with
/// vector rounds in the S-box loops, 2253 with scalar rounds (M5 Max,
/// cost 5 batch 64).
macro_rules! rounds16_scalar {
    ($s:expr, $pw:expr,
     $l0:ident, $l1:ident, $l2:ident, $l3:ident,
     $r0:ident, $r1:ident, $r2:ident, $r3:ident) => {{
        // One half-round for one lane: `x ^= P[i]`, `y ^= F(x)`. Eight
        // separate variables, not two arrays, so the lane state is
        // guaranteed register-resident (array elements spilled to the
        // stack under the pressure of the 16 hoisted box bases). The
        // `black_box` cuts the four lanes' isomorphic instruction trees:
        // without it LLVM's SLP vectorizer rebuilds a vector from the lane
        // results and re-extracts it on the next round — an
        // insert→extract round trip on the per-round dependency chain that
        // this spelling exists to remove. Measured: 1073 hashes/s without
        // `black_box` (SLP rebuilt the vectors), 1326 with it but lanes in
        // arrays (they spilled to the stack), 2253 with it and lanes as
        // eight named variables. `black_box` compiles to zero
        // instructions; the value is already in a GPR.
        macro_rules! half {
            ($x:ident, $y:ident, $i:expr, $lane:expr) => {{
                $x ^= $pw[$i][$lane];
                // SAFETY: `$lane < 4`; the byte masks inside `f_word`
                // bound every index — the module-level lookup invariant.
                $y ^= core::hint::black_box(unsafe { f_word($s, $x, $lane) });
            }};
        }
        for pair in 0..8 {
            half!($l0, $r0, 2 * pair, 0);
            half!($l1, $r1, 2 * pair, 1);
            half!($l2, $r2, 2 * pair, 2);
            half!($l3, $r3, 2 * pair, 3);
            half!($r0, $l0, 2 * pair + 1, 0);
            half!($r1, $l1, 2 * pair + 1, 1);
            half!($r2, $l2, 2 * pair + 1, 2);
            half!($r3, $l3, 2 * pair + 1, 3);
        }
        // Whitening: the pair loop ends in reference post-swap orientation,
        // so the undo-swap after it exchanges each pair of registers.
        $l0 ^= $pw[16][0];
        $l1 ^= $pw[16][1];
        $l2 ^= $pw[16][2];
        $l3 ^= $pw[16][3];
        $r0 ^= $pw[17][0];
        $r1 ^= $pw[17][1];
        $r2 ^= $pw[17][2];
        $r3 ^= $pw[17][3];
        core::mem::swap(&mut $l0, &mut $r0);
        core::mem::swap(&mut $l1, &mut $r1);
        core::mem::swap(&mut $l2, &mut $r2);
        core::mem::swap(&mut $l3, &mut $r3);
    }};
}

/// The 16 Feistel rounds of one lockstep encryption, as a macro so every
/// caller gets a *textually* inlined copy. `#[inline]` on the equivalent
/// `#[target_feature]` fn let LLVM keep it out-of-line — one call per
/// encryption, each re-running the ~30-instruction prologue that computes
/// the 16 lane-adjusted box bases — and `#[inline(always)]` is rejected on
/// `#[target_feature]` fns. Inlined per caller, the bases hoist out of the
/// 521-encryption loops: 1030 hashes/s outlined, 1064 inlined (M5 Max,
/// cost 5 batch 64, [`f4_split`] datapath).
///
/// Written as eight swap-free round pairs: pairing two rounds lets the
/// roles of `$l` and `$r` alternate between halves, so no `mem::swap` (two
/// register moves per round in the rolled-loop codegen) ever exists inside
/// the loop. The whitening lands on the exchanged registers — the pair
/// loop ends in reference post-swap orientation, so the final swap-undo is
/// the one `mem::swap` after the loop (register renaming; free).
macro_rules! rounds16 {
    ($state:expr, $l:ident, $r:ident) => {{
        // Fully unrolled: the rolled loop cost a counter compare/branch per
        // pair and, worse, kept LLVM from scheduling one round's address
        // math into the previous round's load latency (1064 rolled, 1073
        // unrolled — small, but free).
        macro_rules! pair {
            ($i:expr) => {{
                $l = veorq_u32($l, $state.p[$i]);
                $r = veorq_u32($r, f4_split(&$state.s, $l));
                $r = veorq_u32($r, $state.p[$i + 1]);
                $l = veorq_u32($l, f4_split(&$state.s, $r));
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
        $l = veorq_u32($l, $state.p[16]);
        $r = veorq_u32($r, $state.p[17]);
        core::mem::swap(&mut $l, &mut $r);
    }};
}

/// One Blowfish block encryption, four lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`.
#[target_feature(enable = "neon")]
#[inline]
fn encipher4(state: &State4, mut l: uint32x4_t, mut r: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
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
        rounds16!(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    // Same switch as `encrypt_zero_chain_v`: P is frozen during the S-box
    // loop, so the salt stream and the lanes all go scalar with it.
    let pw = p_words(&state.p);
    let sww: [[u32; 4]; 4] = core::array::from_fn(|i| lanes_of(swv[i]));
    let [mut l0, mut l1, mut l2, mut l3] = lanes_of(l);
    let [mut r0, mut r1, mut r2, mut r3] = lanes_of(r);
    for b in 0..4 {
        for pair in 0..128 {
            l0 ^= sww[j % 4][0];
            l1 ^= sww[j % 4][1];
            l2 ^= sww[j % 4][2];
            l3 ^= sww[j % 4][3];
            j += 1;
            r0 ^= sww[j % 4][0];
            r1 ^= sww[j % 4][1];
            r2 ^= sww[j % 4][2];
            r3 ^= sww[j % 4][3];
            j += 1;
            rounds16_scalar!(&state.s, &pw, l0, l1, l2, l3, r0, r1, r2, r3);
            let e = b * 256 + 2 * pair;
            state.s.0[e] = [l0, l1, l2, l3];
            state.s.0[e + 1] = [r0, r1, r2, r3];
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
        rounds16!(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    // P is frozen from here on: hoist it to scalar words and run the S-box
    // loop with all four lanes in GPRs — see `rounds16_scalar`.
    let pw = p_words(&state.p);
    let [mut l0, mut l1, mut l2, mut l3] = lanes_of(l);
    let [mut r0, mut r1, mut r2, mut r3] = lanes_of(r);
    for b in 0..4 {
        for pair in 0..128 {
            rounds16_scalar!(&state.s, &pw, l0, l1, l2, l3, r0, r1, r2, r3);
            let e = b * 256 + 2 * pair;
            state.s.0[e] = [l0, l1, l2, l3];
            state.s.0[e + 1] = [r0, r1, r2, r3];
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
    // `mut` serves only the `zeroize` wipe at the bottom of this fn.
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut kwv = transpose_keys(key_words);
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
        // `kwv` is the transposed key schedule — all four lanes'
        // password-derived key words — so it is wiped with the state.
        // SAFETY: same reinterpretation as above, for the eighteen `kwv`
        // vectors: 72 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwv.as_mut_ptr().cast::<u32>(), 18 * 4)
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

    /// The inactive lookup spellings must agree with the active
    /// [`lookup4_scaled`] on every index vector — they stay in the tree as
    /// tuning alternatives, so they stay checked.
    #[test]
    fn lookup_spellings_agree() {
        let mut boxes = SBoxes4([[0; 4]; 1024]);
        for (entry, words) in boxes.0.iter_mut().enumerate() {
            *words = [
                entry as u32,
                (entry as u32) ^ 0xAAAA_AAAA,
                (entry as u32).wrapping_mul(31),
                !(entry as u32),
            ];
        }
        for i in 0..256u32 {
            // SAFETY: the 4-word stack array is a live 16-byte read source.
            let idx = unsafe { vld1q_u32([i, 255 - i, i ^ 0x5A, (i * 7) & 0xFF].as_ptr()) };
            // SAFETY: `idx` lanes are all in `0..=255` by construction, and
            // each `box_base` points at a real 256-entry box of `boxes` —
            // the module-level lookup invariant.
            let (a, b, c) = unsafe {
                (
                    lookup4_stack(boxes.box_base((i % 4) as usize), idx),
                    lookup4_lane(boxes.box_base((i % 4) as usize), idx),
                    lookup4_scaled(boxes.box_base((i % 4) as usize), idx),
                )
            };
            let (mut wa, mut wb, mut wc) = ([0u32; 4], [0u32; 4], [0u32; 4]);
            // SAFETY: all three are live 16-byte stack arrays, writable in full.
            unsafe {
                vst1q_u32(wa.as_mut_ptr(), a);
                vst1q_u32(wb.as_mut_ptr(), b);
                vst1q_u32(wc.as_mut_ptr(), c);
            }
            assert_eq!(wa, wb, "spellings (a)/(b) diverged at index set {i}");
            assert_eq!(wa, wc, "spellings (a)/(c) diverged at index set {i}");
        }
    }

    /// The inactive SIMD [`f4`] must agree with the active [`f4_split`] —
    /// it is the reference datapath the `lookup4` spellings hang off, so
    /// it stays checked against the code that actually ships in the loops.
    #[test]
    fn f4_variants_agree() {
        let mut boxes = SBoxes4([[0; 4]; 1024]);
        let mut state = 0x853C_49E6_748F_EA9Bu64; // splitmix64 constant
        for entry in boxes.0.iter_mut().flatten() {
            // Deterministic pseudo-random fill (SplitMix64 steps).
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *entry = (z ^ (z >> 31)) as u32;
        }
        for i in 0..64u32 {
            let x = [i, !i, i.wrapping_mul(0x0101_0101), i.rotate_right(8) | 0x5A5A];
            // SAFETY: `xv` is a live 16-byte stack array, readable in full.
            let xv = unsafe { vld1q_u32(x.as_ptr()) };
            // SAFETY: this module only compiles on aarch64, where NEON is
            // baseline — the same scope argument as `lanes_match_scalar`.
            let (a, b) = unsafe { (f4(&boxes, xv), f4_split(&boxes, xv)) };
            let (mut wa, mut wb) = ([0u32; 4], [0u32; 4]);
            // SAFETY: both are live 16-byte stack arrays, writable in full.
            unsafe {
                vst1q_u32(wa.as_mut_ptr(), a);
                vst1q_u32(wb.as_mut_ptr(), b);
            }
            assert_eq!(wa, wb, "f4 variants diverged at input {i}");
        }
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
