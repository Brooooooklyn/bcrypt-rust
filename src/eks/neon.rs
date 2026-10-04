//! AArch64 batch EksBlowfish: eight independent bcrypt hashes in
//! lockstep, every lane living in general-purpose registers.
//!
//! Despite the backend name, this kernel uses **no NEON instructions**.
//! The hot loop is data-dependent 32-bit gathers into per-lane S-boxes;
//! NEON has no gather, and a vector-resident round pays a UMOV→LDR→INS
//! domain crossing (~16 cycles, three pipe classes) per lookup. Measured
//! on M5 Max (cost 5, batch 64): vector rounds 1097 hashes/s, the same
//! rounds with lanes in GPRs 2253 — and John the Ripper, whose design
//! the GPR-resident round loop is, ships no SIMD bcrypt anywhere. What
//! vector plumbing this file once had (a `uint32x4_t` expansion datapath
//! feeding four SIMD lanes) existed only to construct a four-lane
//! interleave; widening past four lanes removes it, because the per-lane
//! key/salt words arrive as plain word arrays and every datapath is
//! scalar code.
//!
//! # State layout: struct-of-arrays, stride 8
//!
//! The four S-boxes are interleaved by lane — [`SBoxes`] — entry
//! `(b, i)` holding all lanes contiguously, so expansion-chain writes
//! stay a few plain word stores and a lookup gathers lane `l` at
//! `box_base + idx * STRIDE + lane`. `STRIDE` is 8, matching the eight
//! lanes, so entries carry no padding: `idx * STRIDE` keeps its low
//! three bits clear, the lane folds into the index with one `orr` (see
//! [`gather_idx`]), and the 32 KiB working set is a quarter of the
//! 128 KiB L1d. (One scalar region per lane — AoS — was measured and
//! lost: 2504-2563 vs 2963-2967 hashes/s full-group at six lanes.)
//!
//! # Tuning summary (M5 Max, `cargo bench --bench micro`, cost 5)
//!
//! Measured hashes/s, scalar always ~790-840:
//!
//! * 2253 — four lanes, lanes 2/3 reloading gather bases from the stack
//!   every half-round (16 hoisted lane-adjusted box bases spilled).
//! * 2463-2492 — the `black_box` anti-SLP barrier replaced by the
//!   zero-cost register-identity [`opaque`] (it had been a store+reload
//!   through the stack on the per-lane Feistel chain, ~64 forwards per
//!   encipher).
//! * 2267-2298 — four lanes with the lane folded into the gather index
//!   ([`gather_idx`]): the fold costs one `orr` per gather on the
//!   address chain — a small loss at four lanes, kept because it is what
//!   makes six and eight lanes register-possible at all.
//! * 2963-2974 — six lanes (batch 6/12/24 full groups; 2624-2645 at
//!   batch 16, which pads a 4-of-6 tail group). ~25 instructions per
//!   half-round, IPC ~8.9 — issue-bound.
//! * 3128-3217 — eight lanes (batch 8/16/32), ratio ~3.7-3.85x: 345
//!   instructions per pair-iteration with 14 spill-traffic ops (one
//!   shared base-pointer slot reloaded ~7x, two lane words). Register
//!   pressure stops the widening here — the loop runs at IPC ~8.2 of a
//!   ~10-wide core, still issue-bound rather than load-bound (80 loads
//!   per pair-iteration = 26.7c against a 42c measured pair time).
//! * +1.2-2.0% — the 16 per-iteration P-row loads paired into 8-byte
//!   loads feeding two lanes each (one `lsr` per pair, so the
//!   instruction count is flat but the load count drops 92→82 in the
//!   S-loop instance). Measured as a 10-pair interleaved A/B, batch 16
//!   medians 3150 vs 3110 hashes/s. Nearby instruction-diet attempts
//!   that all measured ≤0 or below the 1% bar and were reverted: full
//!   unroll of the pair loop (−3%, half the spill traffic but a 2.9 KB
//!   loop body), four mutually-opaque box base pointers (−4% of
//!   instructions, −50% of spills, +0.1%), gather issue order (−0.8%).
//!
//! # Lookup invariant
//!
//! Every index handed to [`gather_idx`] is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f_word`], so
//! every byte is in `0..=255` and `byte * STRIDE | lane < 256 * STRIDE`
//! stays inside the addressed 256-entry box — a gather can never leave
//! the 32 KiB [`SBoxes`]. The invariant is stated once here and
//! referenced at every unsafe dereference below.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — the `State` expansion state, the `kw` key-word copy and
//! `cdata` — before returning, best-effort like the rest of the crate:
//! per-expansion scratch (the lane registers) and compiler spills are
//! not chased.

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call. Must match `Backend::Neon.lanes()` in
/// `super` (compile-checked right below): the interleave writes its lane
/// variables out for exactly this width.
const LANES: usize = 8;

const _: () = assert!(
    LANES == super::Backend::Neon.lanes(),
    "neon LANES and Backend::Neon.lanes() disagree"
);

/// S-box entry stride in words — a power of two so the lane folds into a
/// gather index with one `orr` (see [`gather_idx`]). Eight, matching
/// [`LANES`]; a six-lane shape used the same stride with two padding
/// words per entry (never read). 32 KiB either way.
const STRIDE: usize = 8;

/// The lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is the
/// `(b, i)` Blowfish S-box word of all lanes, lane `l` at u32 offset
/// `(b * 256 + i) * STRIDE + l`. SoA beat one-scalar-region-per-lane
/// (AoS) head to head at six lanes — 2963-2967 vs 2504-2563 hashes/s
/// full-group: AoS cuts the per-gather extract work, but its scattered
/// per-lane expansion stores dropped the loop from IPC ~8.9 to ~5.2.
/// 32 KiB, 64-byte aligned.
#[repr(C, align(64))]
struct SBoxes([[u32; STRIDE]; 1024]);

/// The lockstep EksBlowfish state: 18 P-rows plus the interleaved
/// S-boxes — the lane-wise image of scalar `State`. P rows are
/// `STRIDE`-word too, so `p[i][lane]` compiles to one immediate-offset
/// load per half-round.
struct State {
    p: [[u32; STRIDE]; 18],
    s: SBoxes,
}

/// Anti-SLP barrier at zero codegen cost: an empty `asm!` template with
/// an `inout(reg)` operand is opaque to LLVM's SLP vectorizer (an unknown
/// operation on the value) yet emits no instructions and touches no
/// memory — the value is already in a GPR. Probe-verified on rustc 1.98:
/// `core::hint::black_box(u32)` is NOT this cheap — it lowers to a
/// store+reload through a stack slot (see `rounds16_scalar!`).
#[inline(always)]
fn opaque(v: u32) -> u32 {
    let mut v = v;
    // SAFETY: register-only identity — no memory access, no clobbers.
    unsafe { core::arch::asm!("/* {0:w} */", inout(reg) v) };
    v
}

/// The gather index for one S-box lookup: `byte * STRIDE | lane`
/// (`byte * STRIDE` has its low three bits clear). The `asm!` barrier is
/// what keeps the lane in the *index*: without it LLVM reassociates the
/// constant lane offset into the pointer, and every spelling then
/// loses — either `4 * LANES` lane-adjusted box bases are hoisted and
/// spill (the tail lanes reloading theirs from the stack every
/// half-round), or only the box-0 bases survive and boxes 1-3 are
/// reached by an extended-register `add` + displacement `ldr` per
/// gather (2-cycle, 2-issue, on the address path — measured 1794-1820
/// hashes/s vs 2463-2492 for the spilling spelling at four lanes).
/// Behind the barrier the address stays `box_base + idx * 4` with no
/// additive constant, so isel emits one `ldr [xBase, wIdx, uxtw #2]`
/// per gather against four shared box bases. The options keep the
/// barrier scheduling-transparent: loads and ALU ops move freely
/// across it.
#[inline(always)]
fn gather_idx(byte: u32, lane: u32) -> usize {
    let mut i = (byte * STRIDE as u32) | lane;
    // SAFETY: register-only identity — no memory access, no flags touched.
    unsafe {
        core::arch::asm!("/* {0:w} */", inout(reg) i, options(preserves_flags, nostack, nomem))
    };
    i as usize
}

/// One lane of the Blowfish round function in scalar code:
/// `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`, bytes MSB-first, wrapping adds.
/// This is the datapath of `rounds16_scalar!`.
#[inline]
unsafe fn f_word(s: &SBoxes, w: u32, lane: u32) -> u32 {
    let (a, b, c, d) = (w >> 24, (w >> 16) & 0xff, (w >> 8) & 0xff, w & 0xff);
    // SAFETY: lookup invariant — `a`/`b`/`c`/`d` are bytes by the masks
    // above and `lane < STRIDE` at every call site, so each
    // `gather_idx = byte * STRIDE | lane < 256 * STRIDE` stays inside the
    // addressed 256-entry box.
    unsafe {
        let base = s.0.as_ptr().cast::<u32>();
        let va = *base.add(gather_idx(a, lane));
        let vb = *base.add(256 * STRIDE).add(gather_idx(b, lane));
        let vc = *base.add(2 * 256 * STRIDE).add(gather_idx(c, lane));
        let vd = *base.add(3 * 256 * STRIDE).add(gather_idx(d, lane));
        (va.wrapping_add(vb) ^ vc).wrapping_add(vd)
    }
}

/// The 16 Feistel rounds with all lanes in GPRs: no vector register is
/// touched between the first salt/P XOR and the store after it, so the
/// round-to-round dependency chains are pure scalar code and the lanes'
/// chains overlap in the out-of-order window. Written as eight swap-free
/// round pairs: pairing two rounds lets the roles of `l` and `r`
/// alternate between halves, so no `mem::swap` exists inside the loop.
/// The whitening lands on the exchanged registers — the pair loop ends
/// in reference post-swap orientation, so the swap-undo is the one
/// `mem::swap` per lane after the loop (register renaming; free).
macro_rules! rounds16_scalar {
    ($s:expr, $pw:expr, $((($la:ident, $ra:ident), ($lb:ident, $rb:ident), $lp:literal)),+ $(,)?) => {{
        // One half-round for two adjacent lanes: `x ^= P[i]`, `y ^= F(x)`.
        // The P words of lanes `2*lp` and `2*lp+1` are adjacent in a P
        // row, so one 8-byte load feeds both lanes' XOR (`from_le` keeps
        // the lane order correct regardless of target endianness). Each
        // lane's state is a pair of named variables, not array elements,
        // so it is guaranteed register-resident. The `opaque` call cuts
        // the lanes' isomorphic instruction trees: without it LLVM's SLP
        // vectorizer rebuilds vectors from the lane results and
        // re-extracts them on the next round — an insert→extract round
        // trip on the per-round dependency chain that this spelling
        // exists to remove. (`black_box` used to play this role; on
        // rustc 1.98 it lowers to a store+reload through a stack slot —
        // ~64 store→load forwards per encipher on the per-lane chains.
        // `opaque` is the same barrier, zero instructions.)
        macro_rules! half2 {
            ($xa:ident, $ya:ident, $xb:ident, $yb:ident, $i:expr, $ln:literal) => {{
                // SAFETY: reads 8 bytes inside row `$pw[$i]` (a `[u32;
                // STRIDE]`, `STRIDE >= 2`), alignment handled by
                // `read_unaligned`.
                let p2 = unsafe { ($pw[$i].as_ptr().cast::<u64>()).add($ln).read_unaligned() };
                let p2 = u64::from_le(p2);
                $xa ^= p2 as u32;
                $xb ^= (p2 >> 32) as u32;
                // SAFETY: lane `2*$ln < STRIDE`; the byte masks inside
                // `f_word` bound every index — the module-level lookup
                // invariant.
                $ya ^= opaque(unsafe { f_word($s, $xa, 2 * $ln) });
                // SAFETY: lane `2*$ln+1 < STRIDE`; same lookup invariant.
                $yb ^= opaque(unsafe { f_word($s, $xb, 2 * $ln + 1) });
            }};
        }
        for pair in 0..8 {
            $(half2!($la, $ra, $lb, $rb, 2 * pair, $lp);)+
            $(half2!($ra, $la, $rb, $lb, 2 * pair + 1, $lp);)+
        }
        $(
            $la ^= $pw[16][2 * $lp];
            $lb ^= $pw[16][2 * $lp + 1];
            $ra ^= $pw[17][2 * $lp];
            $rb ^= $pw[17][2 * $lp + 1];
            core::mem::swap(&mut $la, &mut $ra);
            core::mem::swap(&mut $lb, &mut $rb);
        )+
    }};
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
fn expand_state(state: &mut State, sw: &[[u32; STRIDE]; 4], kw: &[[u32; 18]; LANES]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        for (w, k) in p[..LANES].iter_mut().zip(kw.iter()) {
            *w = P_INIT[i] ^ k[i];
        }
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        entry[..LANES].fill(init);
    }
    let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6, mut l7) = (0, 0, 0, 0, 0, 0, 0, 0);
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) = (0, 0, 0, 0, 0, 0, 0, 0);
    let mut j = 0usize;
    for pair in 0..9 {
        l0 ^= sw[j % 4][0];
        l1 ^= sw[j % 4][1];
        l2 ^= sw[j % 4][2];
        l3 ^= sw[j % 4][3];
        l4 ^= sw[j % 4][4];
        l5 ^= sw[j % 4][5];
        l6 ^= sw[j % 4][6];
        l7 ^= sw[j % 4][7];
        j += 1;
        r0 ^= sw[j % 4][0];
        r1 ^= sw[j % 4][1];
        r2 ^= sw[j % 4][2];
        r3 ^= sw[j % 4][3];
        r4 ^= sw[j % 4][4];
        r5 ^= sw[j % 4][5];
        r6 ^= sw[j % 4][6];
        r7 ^= sw[j % 4][7];
        j += 1;
        rounds16_scalar!(
            &state.s, &state.p,
            ((l0, r0), (l1, r1), 0), ((l2, r2), (l3, r3), 1),
            ((l4, r4), (l5, r5), 2), ((l6, r6), (l7, r7), 3),
        );
        state.p[2 * pair][..LANES].copy_from_slice(&[l0, l1, l2, l3, l4, l5, l6, l7]);
        state.p[2 * pair + 1][..LANES].copy_from_slice(&[r0, r1, r2, r3, r4, r5, r6, r7]);
    }
    for b in 0..4 {
        for pair in 0..128 {
            l0 ^= sw[j % 4][0];
            l1 ^= sw[j % 4][1];
            l2 ^= sw[j % 4][2];
            l3 ^= sw[j % 4][3];
            l4 ^= sw[j % 4][4];
            l5 ^= sw[j % 4][5];
            l6 ^= sw[j % 4][6];
            l7 ^= sw[j % 4][7];
            j += 1;
            r0 ^= sw[j % 4][0];
            r1 ^= sw[j % 4][1];
            r2 ^= sw[j % 4][2];
            r3 ^= sw[j % 4][3];
            r4 ^= sw[j % 4][4];
            r5 ^= sw[j % 4][5];
            r6 ^= sw[j % 4][6];
            r7 ^= sw[j % 4][7];
            j += 1;
            rounds16_scalar!(
                &state.s, &state.p,
                ((l0, r0), (l1, r1), 0), ((l2, r2), (l3, r3), 1),
                ((l4, r4), (l5, r5), 2), ((l6, r6), (l7, r7), 3),
            );
            let e = b * 256 + 2 * pair;
            state.s.0[e][..LANES].copy_from_slice(&[l0, l1, l2, l3, l4, l5, l6, l7]);
            state.s.0[e + 1][..LANES].copy_from_slice(&[r0, r1, r2, r3, r4, r5, r6, r7]);
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state`] does, minus the salt mixing.
fn encrypt_zero_chain(state: &mut State) {
    let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6, mut l7) = (0, 0, 0, 0, 0, 0, 0, 0);
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) = (0, 0, 0, 0, 0, 0, 0, 0);
    for pair in 0..9 {
        rounds16_scalar!(
            &state.s, &state.p,
            ((l0, r0), (l1, r1), 0), ((l2, r2), (l3, r3), 1),
            ((l4, r4), (l5, r5), 2), ((l6, r6), (l7, r7), 3),
        );
        state.p[2 * pair][..LANES].copy_from_slice(&[l0, l1, l2, l3, l4, l5, l6, l7]);
        state.p[2 * pair + 1][..LANES].copy_from_slice(&[r0, r1, r2, r3, r4, r5, r6, r7]);
    }
    for b in 0..4 {
        for pair in 0..128 {
            rounds16_scalar!(
                &state.s, &state.p,
                ((l0, r0), (l1, r1), 0), ((l2, r2), (l3, r3), 1),
                ((l4, r4), (l5, r5), 2), ((l6, r6), (l7, r7), 3),
            );
            let e = b * 256 + 2 * pair;
            state.s.0[e][..LANES].copy_from_slice(&[l0, l1, l2, l3, l4, l5, l6, l7]);
            state.s.0[e + 1][..LANES].copy_from_slice(&[r0, r1, r2, r3, r4, r5, r6, r7]);
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
fn expand0state(state: &mut State, kw: &[[u32; 18]; LANES]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        for (w, k) in p[..LANES].iter_mut().zip(kw.iter()) {
            *w ^= k[i];
        }
    }
    encrypt_zero_chain(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words,
/// so the P XOR cycles it (`i & 3`), then the same zero chain.
fn expand0state_salt(state: &mut State, swk: &[[u32; 4]; LANES]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        for (w, s) in p[..LANES].iter_mut().zip(swk.iter()) {
            *w ^= s[i & 3];
        }
    }
    encrypt_zero_chain(state);
}

/// The whole lockstep bcrypt: one key+salt expansion, `2^cost` rounds of
/// key-then-salt expansion (OpenBSD order), then 64 encryptions of the
/// "OrpheanBeholderScryDoubt" constant and a big-endian store per lane.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer
/// dereferences inside are governed by the module-level lookup
/// invariant: S-box gathers index with byte-masked lanes only.
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
    // Per-lane working copies: `kw` is read every cost-loop iteration and
    // `swk` every salt expansion — stack arrays keep those reads
    // bounds-check-free. `sw` transposes the salt into `sw[word][lane]`
    // rows for the expansion-chain XORs. `kw`'s `mut` serves only the
    // `zeroize` wipe at the bottom of this fn.
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut kw: [[u32; 18]; LANES] = core::array::from_fn(|l| key_words[l]);
    let swk: [[u32; 4]; LANES] = core::array::from_fn(|l| salt_words[l]);
    let mut sw = [[0u32; STRIDE]; 4];
    for (w, row) in sw.iter_mut().enumerate() {
        for (slot, s) in row[..LANES].iter_mut().zip(salt_words.iter()) {
            *slot = s[w];
        }
    }
    let mut state = State {
        p: [[0; STRIDE]; 18],
        s: SBoxes([[0; STRIDE]; 1024]),
    };
    expand_state(&mut state, &sw, &kw);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state(&mut state, &kw);
        expand0state_salt(&mut state, &swk);
    }
    // "OrpheanBeholderScryDoubt" as six per-lane rows of one broadcast
    // word each.
    let mut cdata = [[0u32; STRIDE]; 6];
    for (row, init) in cdata.iter_mut().zip([
        0x4f72_7068,
        0x6561_6e42,
        0x6568_6f6c,
        0x6465_7253,
        0x6372_7944,
        0x6f75_6274,
    ]) {
        row[..LANES].fill(init);
    }
    for _ in 0..64 {
        for pair in 0..3 {
            let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6, mut l7) = (
                cdata[2 * pair][0],
                cdata[2 * pair][1],
                cdata[2 * pair][2],
                cdata[2 * pair][3],
                cdata[2 * pair][4],
                cdata[2 * pair][5],
                cdata[2 * pair][6],
                cdata[2 * pair][7],
            );
            let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) = (
                cdata[2 * pair + 1][0],
                cdata[2 * pair + 1][1],
                cdata[2 * pair + 1][2],
                cdata[2 * pair + 1][3],
                cdata[2 * pair + 1][4],
                cdata[2 * pair + 1][5],
                cdata[2 * pair + 1][6],
                cdata[2 * pair + 1][7],
            );
            rounds16_scalar!(
                &state.s, &state.p,
                ((l0, r0), (l1, r1), 0), ((l2, r2), (l3, r3), 1),
                ((l4, r4), (l5, r5), 2), ((l6, r6), (l7, r7), 3),
            );
            cdata[2 * pair][..LANES].copy_from_slice(&[l0, l1, l2, l3, l4, l5, l6, l7]);
            cdata[2 * pair + 1][..LANES].copy_from_slice(&[r0, r1, r2, r3, r4, r5, r6, r7]);
        }
    }
    // Split the lanes back out: word `w` of lane `l`'s output is
    // `cdata[w][l]`, stored big-endian.
    for (w, row) in cdata.iter().enumerate() {
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&row[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        crate::wipe::secure_wipe_u32(state.p.as_flattened_mut());
        crate::wipe::secure_wipe_u32(state.s.0.as_flattened_mut());
        crate::wipe::secure_wipe_u32(cdata.as_flattened_mut());
        // `kw` holds every lane's password-derived key words, so it is
        // wiped with the state. (`swk`/`sw` are salt — public.)
        crate::wipe::secure_wipe_u32(kw.as_flattened_mut());
    }
}

/// The NEON batch kernel: [`LANES`] independent bcrypt hashes in
/// lockstep. (The name is historical — see the module docs: the fastest
/// bcrypt shape on AArch64 is a pure GPR interleave.)
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Neon`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long.
/// Resolve through [`super::bcrypt_lanes_fn`]; never call directly
/// without an availability check.
///
/// # Safety
///
/// * The dispatch table hands this pointer out only for
///   [`super::Backend::Neon`], whose `is_available` is the documented
///   check.
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[inline]
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
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — the three
    // slices are exactly `LANES` lanes each — which is everything
    // `bcrypt_lanes_impl` requires.
    unsafe { bcrypt_lanes_impl(cost, key_words, salt_words, outs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base64;
    use crate::eks::{expand_key_words, salt_words};

    /// Reduce a password to kernel key words exactly as `crate::core`
    /// will (mirrors the scalar tests' helper, plus the 72-byte cap
    /// `padded_key` applies: the stream cycles at `min(len + 1, 72)`).
    fn key_words_for(password: &[u8]) -> [u32; 18] {
        let mut key = [0u8; 72];
        let copied = password.len().min(72);
        key[..copied].copy_from_slice(&password[..copied]);
        expand_key_words(&key, (copied + 1).min(72))
    }

    /// Different passwords and different salts, one per lane, must
    /// reproduce `LANES` scalar hashes bit for bit — including the
    /// OpenBSD `U*U` vector in lane 1.
    // Miri cannot interpret NEON intrinsics; production paths never reach
    // this kernel under Miri (detect pins Scalar), only this direct test
    // would.
    #[cfg(not(miri))]
    #[test]
    fn lanes_match_scalar() {
        let vector_salt =
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let passwords: [&[u8]; LANES] = [
            b"",
            b"U*U",
            b"hunter2",
            &[0xAA; 72],
            b"bcrypt",
            &[0x55; 71],
            b"OrpheanBeholderScryDoubt",
            &[0x01],
        ];
        let kws: [[u32; 18]; LANES] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; LANES] = [
            salt_words(&[0x11; 16]),
            salt_words(&vector_salt),
            salt_words(&[0x33; 16]),
            salt_words(&[0x44; 16]),
            salt_words(&[0x55; 16]),
            salt_words(&[0x66; 16]),
            salt_words(&[0x77; 16]),
            salt_words(&[0x88; 16]),
        ];
        let mut outs = [[0u8; 24]; LANES];
        // SAFETY: this module only compiles on aarch64, where NEON is
        // baseline; the three slices are exactly `LANES` lanes.
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
