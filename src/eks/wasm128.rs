//! wasm32 SIMD128 EksBlowfish: eight independent bcrypt hashes per kernel
//! call, run as two interleaved four-lane `v128` states — "X2": lanes
//! 0..=3 form state A, lanes 4..=7 state B.
//!
//! Ports [`super::scalar`] exactly — same OpenBSD structure, same
//! continuous salt-word stream across the P→S transition in
//! `expand_state`, same key-then-salt cost-loop order — with every 32-bit
//! operation widened to a lane-wise `v128` one, two states in flight at
//! once. Scalar `wrapping_add` becomes `i32x4_add` (wraps, which is what
//! Blowfish wants); every XOR becomes `v128_xor`.
//!
//! # Why X2: the single-state 4-lane shape is at its ceiling
//!
//! The 4-lane lockstep kernel measured 1.21× over scalar-in-wasm
//! (wasmtime 48, M5 Max, cost 4) — already ~90–95% of the ceiling for
//! that shape: natively, a 4-lane AVX2 bcrypt *with a real hardware
//! gather* reaches only 1.33× (pbcrypt). The one shape with native
//! evidence past that ceiling is interleaving independent dependency
//! chains — John the Ripper's X2/X3 instruction-interleaved scalar beat
//! its own AVX2-gather kernel — so the round loop below interleaves A and
//! B at the F() level: B's gathers issue while A's store-forward
//! crossings resolve, and vice versa. Measured (wasmtime 48, M5 Max,
//! cost 4): 1.21× → ~1.58× over scalar-in-wasm — but only with the
//! lookup rebuild spelled as independent stores (see "No gather" below);
//! with LLVM's lane-load fusion the same X2 kernel merely ties the
//! 4-lane one.
//!
//! # Selection is compile-time, and why that is sound here
//!
//! WebAssembly has no portable runtime feature detection a module can
//! survive: one containing SIMD128 instructions fails *validation* on an
//! engine that lacks them, so the deployment story is a compile-time
//! choice — build with `-C target-feature=+simd128` for engines that have
//! it (every current browser, wasmtime, and Node; part of the web baseline
//! since 2021). This module therefore exists exactly when
//! `cfg(all(target_arch = "wasm32", target_feature = "simd128"))` held at
//! compile time, and [`super::have_wasm_simd128`] answers from that same
//! cfg — the crate can never dispatch to an instruction the engine lacks.
//!
//! # State layout: struct-of-arrays, eight lanes wide
//!
//! The four S-boxes are stored interleaved by lane — [`SBoxes8x`] — so
//! entry `(b, i)` holds all eight lanes contiguously (A in the low 16
//! bytes, B in the high) and expansion-chain writes stay full-vector
//! stores. Lookups are gathers: lane `l` of a lookup reads
//! `box_base[idx_l * 8 + l]`. The entry stride is a power of two
//! (32 bytes), so `idx * 8 + l` is one shift and an add. 32 KiB of S-box
//! per kernel: M-series' 128 KiB L1d holds it easily; 32–48 KiB x86 L1d
//! is why the width caps at X2 rather than going wider.
//!
//! # No gather at this ISA — and LLVM's lane fusion must be declined
//!
//! wasm SIMD128 has no gather instruction, so every lookup is eight
//! scalar loads; the only question is how values cross between the
//! vector and scalar domains. Given the plain "stack round-trip"
//! spelling (store the index vectors, scalar loads, load the results
//! back), LLVM's wasm backend rewrites both crossings: indices become
//! `i32x4.extract_lane` (fine — independent `umov`s) but the S-box loads
//! get fused into `v128.load32_lane` chains, where each lane-inserting
//! load depends on the previous one through the destination register —
//! serializing the one part of the kernel that must stay parallel.
//! [`lookup8`] therefore pins the rebuild with volatile writes to its
//! own stack array: wasm has no volatile memory ops, so engines receive
//! plain stores and loads — the qualifier only opts this one spot out of
//! LLVM's fusion. Measured (wasmtime 48, M5 Max, cost 4, batch 8, ratio
//! vs scalar-in-wasm):
//!
//! | kernel      | LLVM-fused rebuild | pinned rebuild |
//! |-------------|--------------------|----------------|
//! | 4-lane      | 1.21× (~1550 h/s)  | 1.00× (~1225)  |
//! | 8-lane (X2) | 1.22× (~1575)      | ~1.58× (~1920–2025) |
//!
//! The pinned spelling only wins with a second dependency chain to hide
//! its store-forward latency — which is exactly what X2 provides.
//!
//! # Lookup invariant
//!
//! Every index handed to [`lookup8`] is a *pre-scaled* byte value —
//! `idx * 8 + lane` per lane, where `idx` is a `>> 24` or `& 0xff` of a
//! `u32` lane produced inside [`f8`] — so every lane is in `0..=2047`
//! and stays inside the addressed 256-entry box: a gather can never
//! leave the 32 KiB [`SBoxes8x`]. The invariant is stated once here and
//! referenced at every unsafe dereference below.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — the [`State8x`] expansion state (both P-arrays and the
//! S-boxes), the `kwa`/`kwb` transposed key schedules and `ca`/`cb` —
//! before returning, best-effort like the rest of the crate: compiler
//! spills and transposition-internal temporaries are not chased.

use core::arch::wasm32::{
    i32x4_add, i32x4_shl, u32x4_shr, u32x4_splat, v128, v128_and, v128_load, v128_store, v128_xor,
};

use crate::consts::{P_INIT, S_INIT};

/// Per-lane gather offsets inside one [`SBoxes8x`] entry: state A occupies
/// the low half of each 8-word entry, state B the high half.
static LANE_OFFSETS_A: [u32; 4] = [0, 1, 2, 3];
static LANE_OFFSETS_B: [u32; 4] = [4, 5, 6, 7];

/// Passwords per kernel call: two interleaved 4-lane `v128` states —
/// lanes 0..=3 are state A, lanes 4..=7 state B.
pub(crate) const LANES: usize = 8;

/// Eight lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is
/// the `(b, i)` Blowfish S-box word of all eight lanes, lane `l` at u32
/// offset `(b * 256 + i) * 8 + l` — the A state in the low 16 bytes, B in
/// the high. 32 KiB, 64-byte aligned, so each 32-byte entry and each
/// 16-byte half of it are aligned by construction (the entry stride
/// divides the struct alignment).
#[repr(C, align(64))]
struct SBoxes8x([[u32; 8]; 1024]);

impl SBoxes8x {
    /// Base pointer of box `b` — the address lane gathers index from.
    #[inline(always)]
    fn box_base(&self, b: usize) -> *const u32 {
        debug_assert!(b < 4);
        // SAFETY: `b < 4`, so `b * 2048` stays inside the 8192-word array.
        unsafe { self.0.as_ptr().cast::<u32>().add(b * 256 * 8) }
    }
}

/// The X2 lockstep EksBlowfish state: two lane-wise images of scalar
/// `State`'s P-array (A for lanes 0..=3, B for lanes 4..=7) plus the
/// interleaved S-boxes both halves share.
struct State8x {
    pa: [v128; 18],
    pb: [v128; 18],
    s: SBoxes8x,
}

/// Gather one S-box word per lane of both half-states at once, from
/// *pre-scaled* index vectors: lane `l` of `sa`/`sb` must already hold
/// `idx * 8 + l` / `idx * 8 + 4 + l`, so each lane's address is one
/// scalar add of `box_base` (scaling in the vector domain costs two
/// vector ops per byte position instead of a shift and two adds per
/// extracted lane). Stack round-trip spelling (the only one — see the
/// module docs): store both index vectors side by side, eight scalar
/// loads with A's and B's alternating so B's independent chain issues
/// while A's store-forward resolves, load both result vectors back.
///
/// # Safety
///
/// `box_base` must point at the first word of one 256-entry box of an
/// [`SBoxes8x`], and lane `l` of `sa`/`sb` must be `idx * 8 + l` /
/// `idx * 8 + 4 + l` with `idx <= 255` — the module-level lookup
/// invariant, pre-scaled. Then every lane is `<= 2047` and each load
/// stays inside that box's 8 KiB.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn lookup8(box_base: *const u32, sa: v128, sb: v128) -> (v128, v128) {
    let mut ix = [0u32; 8];
    // SAFETY: `ix` is a live 32-byte stack array; the two stores cover
    // its bytes 0..16 and 16..32 in full.
    unsafe {
        v128_store(ix.as_mut_ptr().cast::<v128>(), sa);
        v128_store(ix.as_mut_ptr().add(4).cast::<v128>(), sb);
    }
    // SAFETY: lookup invariant — `ix[l] <= 2047`, so each load stays
    // inside the 2048-word box `box_base` points at.
    let a0 = unsafe { *box_base.add(ix[0] as usize) };
    // SAFETY: lookup invariant — see a0.
    let b0 = unsafe { *box_base.add(ix[4] as usize) };
    // SAFETY: lookup invariant — see a0.
    let a1 = unsafe { *box_base.add(ix[1] as usize) };
    // SAFETY: lookup invariant — see a0.
    let b1 = unsafe { *box_base.add(ix[5] as usize) };
    // SAFETY: lookup invariant — see a0.
    let a2 = unsafe { *box_base.add(ix[2] as usize) };
    // SAFETY: lookup invariant — see a0.
    let b2 = unsafe { *box_base.add(ix[6] as usize) };
    // SAFETY: lookup invariant — see a0.
    let a3 = unsafe { *box_base.add(ix[3] as usize) };
    // SAFETY: lookup invariant — see a0.
    let b3 = unsafe { *box_base.add(ix[7] as usize) };
    // The volatile writes pin the rebuild to the stack spelling: wasm
    // has no volatile memory ops, so engines see plain stores and loads
    // — the qualifier only stops LLVM's wasm backend from fusing the
    // eight S-box loads into `v128.load32_lane` chains. Those chains are
    // serial (each lane-inserting load reads and writes the destination
    // register), and with them this kernel measured ~1575 h/s where the
    // independent-store spelling measures ~1920-2025 (wasmtime 48,
    // M5 Max, cost 4, batch 8).
    //
    // SAFETY: `gathered` is a live 32-byte stack array, exclusively
    // owned, written in full before the two reads below.
    let mut gathered = [0u32; 8];
    // SAFETY: as above — `gathered` is exclusively-owned stack memory,
    // valid for writes of all eight words and then full reads.
    unsafe {
        core::ptr::write_volatile(&mut gathered[0], a0);
        core::ptr::write_volatile(&mut gathered[1], a1);
        core::ptr::write_volatile(&mut gathered[2], a2);
        core::ptr::write_volatile(&mut gathered[3], a3);
        core::ptr::write_volatile(&mut gathered[4], b0);
        core::ptr::write_volatile(&mut gathered[5], b1);
        core::ptr::write_volatile(&mut gathered[6], b2);
        core::ptr::write_volatile(&mut gathered[7], b3);
        (
            core::ptr::read_volatile(gathered.as_ptr().cast::<v128>()),
            core::ptr::read_volatile(gathered.as_ptr().add(4).cast::<v128>()),
        )
    }
}

/// The Blowfish round function on both half-states at once:
/// `((S0[a] + S1[b]) ^ S2[c]) + S3[d]` per state, bytes taken MSB-first,
/// every addition wrapping (`i32x4_add` wraps, which is what the scalar
/// `wrapping_add` does). Running A and B through one body is the X2
/// latency hiding: B's four gathers are independent of A's, so they
/// issue while A's store-forward crossings resolve. The byte masks below
/// are what make the [`lookup8`] gathers safe — see the module-level
/// invariant.
#[target_feature(enable = "simd128")]
#[inline]
fn f8(s: &SBoxes8x, xa: v128, xb: v128) -> (v128, v128) {
    let byte_mask = u32x4_splat(0xff);
    // SAFETY: both statics are live 16-byte arrays, readable in full.
    let (off_a, off_b) = unsafe {
        (
            v128_load(LANE_OFFSETS_A.as_ptr().cast::<v128>()),
            v128_load(LANE_OFFSETS_B.as_ptr().cast::<v128>()),
        )
    };
    // Byte extraction plus the `idx * 8 + lane` pre-scale, all in the
    // vector domain (two vector ops per byte position per state).
    let aa = i32x4_add(i32x4_shl(u32x4_shr(xa, 24), 3), off_a);
    let ab = i32x4_add(i32x4_shl(u32x4_shr(xb, 24), 3), off_b);
    let ba = i32x4_add(i32x4_shl(v128_and(u32x4_shr(xa, 16), byte_mask), 3), off_a);
    let bb = i32x4_add(i32x4_shl(v128_and(u32x4_shr(xb, 16), byte_mask), 3), off_b);
    let ca = i32x4_add(i32x4_shl(v128_and(u32x4_shr(xa, 8), byte_mask), 3), off_a);
    let cb = i32x4_add(i32x4_shl(v128_and(u32x4_shr(xb, 8), byte_mask), 3), off_b);
    let da = i32x4_add(i32x4_shl(v128_and(xa, byte_mask), 3), off_a);
    let db = i32x4_add(i32x4_shl(v128_and(xb, byte_mask), 3), off_b);
    // SAFETY: every index lane is a pre-scaled byte (`<= 2047`) — the
    // `>> 24`/`& 0xff` masks plus the `* 8 + lane` scale right above —
    // and each `box_base` points at a real box of `s`, so the
    // module-level lookup invariant holds at all eight gathers.
    let (vaa, vab) = unsafe { lookup8(s.box_base(0), aa, ab) };
    // SAFETY: lookup invariant — see above.
    let (vba, vbb) = unsafe { lookup8(s.box_base(1), ba, bb) };
    // SAFETY: lookup invariant — see above.
    let (vca, vcb) = unsafe { lookup8(s.box_base(2), ca, cb) };
    // SAFETY: lookup invariant — see above.
    let (vda, vdb) = unsafe { lookup8(s.box_base(3), da, db) };
    (
        i32x4_add(v128_xor(i32x4_add(vaa, vba), vca), vda),
        i32x4_add(v128_xor(i32x4_add(vab, vbb), vcb), vdb),
    )
}

/// The 16 Feistel rounds of one X2 lockstep encryption, as a macro so
/// every caller gets a *textually* inlined copy — the same reasoning as
/// `super::sse41`'s `rounds16!` (an outlined `#[target_feature]` rounds
/// fn let LLVM keep the box-base math per call; and `#[inline(always)]`
/// combined with `#[target_feature]` is a hard error, so the macro route
/// is the only force-inline one). Fully unrolled swap-free round pairs,
/// A and B interleaved statement by statement: each pair ends in
/// reference post-swap orientation, so the whitening lands on the
/// exchanged registers and the final swap-undo is the one `mem::swap`
/// per half after the pairs (register renaming; free).
macro_rules! rounds16x2 {
    ($state:expr, $la:ident, $ra:ident, $lb:ident, $rb:ident) => {{
        macro_rules! pair {
            ($i:expr) => {{
                $la = v128_xor($la, $state.pa[$i]);
                $lb = v128_xor($lb, $state.pb[$i]);
                let (fa, fb) = f8(&$state.s, $la, $lb);
                $ra = v128_xor($ra, fa);
                $rb = v128_xor($rb, fb);
                $ra = v128_xor($ra, $state.pa[$i + 1]);
                $rb = v128_xor($rb, $state.pb[$i + 1]);
                let (fa, fb) = f8(&$state.s, $ra, $rb);
                $la = v128_xor($la, fa);
                $lb = v128_xor($lb, fb);
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
        $la = v128_xor($la, $state.pa[16]);
        $lb = v128_xor($lb, $state.pb[16]);
        $ra = v128_xor($ra, $state.pa[17]);
        $rb = v128_xor($rb, $state.pb[17]);
        core::mem::swap(&mut $la, &mut $ra);
        core::mem::swap(&mut $lb, &mut $rb);
    }};
}

/// One Blowfish block encryption, eight lanes as two interleaved
/// four-lane states: 16 Feistel rounds with the final swap undone, the
/// output halves whitened by P[16]/P[17] — identical control flow to
/// scalar `encipher`, twice.
#[target_feature(enable = "simd128")]
#[inline]
fn encipher8(
    state: &State8x,
    mut la: v128,
    mut ra: v128,
    mut lb: v128,
    mut rb: v128,
) -> (v128, v128, v128, v128) {
    rounds16x2!(state, la, ra, lb, rb);
    (la, ra, lb, rb)
}

/// Lockstep `Blowfish_expandstate` on both half-states: fresh state from
/// the pi digits with each half's key XORed into its P-array, then 521
/// encryptions mixing salt words into the running blocks and writing
/// each ciphertext pair back over P (9 pairs) and then S (512 pairs).
///
/// The salt counter `j` is one continuous stream across **both** loops —
/// the P loop consumes 18 words, so the first S-box pair XORs `salt[2]`
/// and `salt[3]` — and both halves consume the same stream positions,
/// each with its own transposed salt vectors. Faithful to scalar
/// `expand_state`; do not "tidy" the counter into the S loop.
#[target_feature(enable = "simd128")]
#[inline]
fn expand_state_v8(
    state: &mut State8x,
    swa: &[v128; 4],
    swb: &[v128; 4],
    kwa: &[v128; 18],
    kwb: &[v128; 18],
) {
    for (i, &init) in P_INIT.iter().enumerate() {
        state.pa[i] = v128_xor(u32x4_splat(init), kwa[i]);
        state.pb[i] = v128_xor(u32x4_splat(init), kwb[i]);
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 8];
    }
    let zero = u32x4_splat(0);
    let (mut la, mut ra, mut lb, mut rb) = (zero, zero, zero, zero);
    let mut j = 0usize;
    for pair in 0..9 {
        la = v128_xor(la, swa[j % 4]);
        lb = v128_xor(lb, swb[j % 4]);
        j += 1;
        ra = v128_xor(ra, swa[j % 4]);
        rb = v128_xor(rb, swb[j % 4]);
        j += 1;
        rounds16x2!(state, la, ra, lb, rb);
        state.pa[2 * pair] = la;
        state.pa[2 * pair + 1] = ra;
        state.pb[2 * pair] = lb;
        state.pb[2 * pair + 1] = rb;
    }
    for b in 0..4 {
        for pair in 0..128 {
            la = v128_xor(la, swa[j % 4]);
            lb = v128_xor(lb, swb[j % 4]);
            j += 1;
            ra = v128_xor(ra, swa[j % 4]);
            rb = v128_xor(rb, swb[j % 4]);
            j += 1;
            rounds16x2!(state, la, ra, lb, rb);
            let e = b * 256 + 2 * pair;
            // SAFETY: both 16-byte halves of entries `e`/`e + 1` are
            // 16-byte aligned by construction (struct align 64, entry
            // stride 32) and exclusively owned — each `v128_store`
            // writes a full half.
            unsafe {
                let e0 = state.s.0[e].as_mut_ptr();
                v128_store(e0.cast::<v128>(), la);
                v128_store(e0.add(4).cast::<v128>(), lb);
                let e1 = state.s.0[e + 1].as_mut_ptr();
                v128_store(e1.cast::<v128>(), ra);
                v128_store(e1.add(4).cast::<v128>(), rb);
            }
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v8`] does, minus the salt mixing.
#[target_feature(enable = "simd128")]
#[inline]
fn encrypt_zero_chain_v8(state: &mut State8x) {
    let zero = u32x4_splat(0);
    let (mut la, mut ra, mut lb, mut rb) = (zero, zero, zero, zero);
    for pair in 0..9 {
        rounds16x2!(state, la, ra, lb, rb);
        state.pa[2 * pair] = la;
        state.pa[2 * pair + 1] = ra;
        state.pb[2 * pair] = lb;
        state.pb[2 * pair + 1] = rb;
    }
    for b in 0..4 {
        for pair in 0..128 {
            rounds16x2!(state, la, ra, lb, rb);
            let e = b * 256 + 2 * pair;
            // SAFETY: same argument as `expand_state_v8` — aligned,
            // exclusively owned entry halves.
            unsafe {
                let e0 = state.s.0[e].as_mut_ptr();
                v128_store(e0.cast::<v128>(), la);
                v128_store(e0.add(4).cast::<v128>(), lb);
                let e1 = state.s.0[e + 1].as_mut_ptr();
                v128_store(e1.cast::<v128>(), ra);
                v128_store(e1.add(4).cast::<v128>(), rb);
            }
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR each half's password words
/// into its P-array, then run the zero chain.
#[target_feature(enable = "simd128")]
#[inline]
fn expand0state_v8(state: &mut State8x, kwa: &[v128; 18], kwb: &[v128; 18]) {
    for ((pa, pb), (&ka, &kb)) in state
        .pa
        .iter_mut()
        .zip(state.pb.iter_mut())
        .zip(kwa.iter().zip(kwb.iter()))
    {
        *pa = v128_xor(*pa, ka);
        *pb = v128_xor(*pb, kb);
    }
    encrypt_zero_chain_v8(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words,
/// so the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "simd128")]
#[inline]
fn expand0state_salt_v8(state: &mut State8x, swa: &[v128; 4], swb: &[v128; 4]) {
    for (i, (pa, pb)) in state.pa.iter_mut().zip(state.pb.iter_mut()).enumerate() {
        *pa = v128_xor(*pa, swa[i & 3]);
        *pb = v128_xor(*pb, swb[i & 3]);
    }
    encrypt_zero_chain_v8(state);
}

/// Transpose the eight lanes' key words into the two 18-vector
/// schedules: A vector `i` holds word `i` of lanes 0..=3, B vector `i`
/// word `i` of lanes 4..=7. wasm has no `setr`-style intrinsic, so each
/// vector is a plain 16-byte stack load — fine here, as nothing in this
/// function is on the cost-loop path.
#[target_feature(enable = "simd128")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> ([v128; 18], [v128; 18]) {
    debug_assert_eq!(key_words.len(), LANES);
    let mut a = [u32x4_splat(0); 18];
    let mut b = [u32x4_splat(0); 18];
    for (i, (va, vb)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
        let wa = [
            key_words[0][i],
            key_words[1][i],
            key_words[2][i],
            key_words[3][i],
        ];
        let wb = [
            key_words[4][i],
            key_words[5][i],
            key_words[6][i],
            key_words[7][i],
        ];
        // SAFETY: `wa`/`wb` are live 16-byte stack arrays, readable in
        // full.
        unsafe {
            *va = v128_load(wa.as_ptr().cast::<v128>());
            *vb = v128_load(wb.as_ptr().cast::<v128>());
        }
    }
    (a, b)
}

/// Transpose the eight lanes' salt words into the two 4-vector sets; see
/// [`transpose_keys`].
#[target_feature(enable = "simd128")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> ([v128; 4], [v128; 4]) {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut a = [u32x4_splat(0); 4];
    let mut b = [u32x4_splat(0); 4];
    for (i, (va, vb)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
        let wa = [
            salt_words[0][i],
            salt_words[1][i],
            salt_words[2][i],
            salt_words[3][i],
        ];
        let wb = [
            salt_words[4][i],
            salt_words[5][i],
            salt_words[6][i],
            salt_words[7][i],
        ];
        // SAFETY: `wa`/`wb` are live 16-byte stack arrays, readable in
        // full.
        unsafe {
            *va = v128_load(wa.as_ptr().cast::<v128>());
            *vb = v128_load(wb.as_ptr().cast::<v128>());
        }
    }
    (a, b)
}

/// The whole lockstep bcrypt: one key+salt expansion, `2^cost` rounds of
/// key-then-salt expansion (OpenBSD order), then 64 encryptions of the
/// "OrpheanBeholderScryDoubt" constant and a big-endian store per lane —
/// for both half-states at once.
///
/// Every kernel function carries the same `#[target_feature(enable =
/// "simd128")]` scope — stdarch annotates the SIMD128 intrinsics, and a
/// matching-scope caller both calls and inlines them — so the whole
/// kernel compiles as one SIMD128 unit behind the [`bcrypt_lanes`] entry
/// point.
///
/// # Safety
///
/// The same contract as [`bcrypt_lanes`]: all three slices are exactly
/// [`LANES`] long and `outs` is not aliased. The raw-pointer dereferences
/// inside are governed by the module-level lookup invariant: S-box
/// gathers index with byte-masked lanes only.
#[target_feature(enable = "simd128")]
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
    let (mut kwa, mut kwb) = transpose_keys(key_words);
    let (swa, swb) = transpose_salts(salt_words);
    let mut state = State8x {
        pa: [u32x4_splat(0); 18],
        pb: [u32x4_splat(0); 18],
        s: SBoxes8x([[0; 8]; 1024]),
    };
    expand_state_v8(&mut state, &swa, &swb, &kwa, &kwb);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state_v8(&mut state, &kwa, &kwb);
        expand0state_salt_v8(&mut state, &swa, &swb);
    }
    // "OrpheanBeholderScryDoubt" as six broadcast words, per half-state.
    let mut ca = [
        u32x4_splat(0x4f72_7068),
        u32x4_splat(0x6561_6e42),
        u32x4_splat(0x6568_6f6c),
        u32x4_splat(0x6465_7253),
        u32x4_splat(0x6372_7944),
        u32x4_splat(0x6f75_6274),
    ];
    let mut cb = ca;
    for _ in 0..64 {
        for pair in 0..3 {
            let (la, ra, lb, rb) = encipher8(
                &state,
                ca[2 * pair],
                ca[2 * pair + 1],
                cb[2 * pair],
                cb[2 * pair + 1],
            );
            ca[2 * pair] = la;
            ca[2 * pair + 1] = ra;
            cb[2 * pair] = lb;
            cb[2 * pair + 1] = rb;
        }
    }
    // Split the lanes back out: word `w` of lane `l`'s output is lane `l`
    // of the A image for lanes 0..=3, lane `l - 4` of the B image for
    // lanes 4..=7 — one side-by-side store, eight big-endian word stores.
    for w in 0..6 {
        let mut words = [0u32; 8];
        // SAFETY: `words` is a live 32-byte stack array; the two stores
        // cover its bytes 0..16 and 16..32 in full.
        unsafe {
            v128_store(words.as_mut_ptr().cast::<v128>(), ca[w]);
            v128_store(words.as_mut_ptr().add(4).cast::<v128>(), cb[w]);
        }
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `v128` is four `u32`s, so the `*mut u32` view covers
        // exactly the same exclusively-owned stack bytes as `state.pa`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.pa.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
        // SAFETY: same reinterpretation as above, for `state.pb`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(state.pb.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
        crate::wipe::secure_wipe_u32(state.s.0.as_flattened_mut());
        // SAFETY: same reinterpretation as above, for the six `ca`
        // vectors: 24 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(ca.as_mut_ptr().cast::<u32>(), 6 * 4)
        });
        // SAFETY: same reinterpretation as above, for `cb`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(cb.as_mut_ptr().cast::<u32>(), 6 * 4)
        });
        // `kwa`/`kwb` are the transposed key schedule — all eight lanes'
        // password-derived key words — so they are wiped with the state.
        // SAFETY: same reinterpretation as above, for the eighteen `kwa`
        // vectors: 72 exclusively-owned stack words, valid for writes.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwa.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
        // SAFETY: same reinterpretation as above, for `kwb`.
        crate::wipe::secure_wipe_u32(unsafe {
            core::slice::from_raw_parts_mut(kwb.as_mut_ptr().cast::<u32>(), 18 * 4)
        });
    }
}

/// The SIMD128 batch kernel: [`LANES`] independent bcrypt hashes as two
/// interleaved four-lane `v128` states (X2 — see the module docs).
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Wasm128`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are exactly [`LANES`] long. Resolve
/// through [`super::bcrypt_lanes_fn`]; never call directly without an
/// availability check.
///
/// # Safety
///
/// * The engine must support SIMD128. The dispatch table hands this
///   pointer out only for [`super::Backend::Wasm128`], whose
///   `is_available` is the documented check — and this module only exists
///   when the crate was compiled with `+simd128`, which is precisely that
///   contract.
/// * `key_words`, `salt_words` and `outs` must each be exactly [`LANES`]
///   long (debug-asserted at entry), and `outs` must not be aliased for
///   the duration of the call.
#[target_feature(enable = "simd128")]
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
    // SAFETY: the caller upholds the `BcryptLanesFn` contract — SIMD128 is
    // available in this engine and the three slices are exactly 8 lanes
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

    /// Gate every kernel-driving test on real SIMD128 availability. This
    /// module only exists when the crate was compiled with `+simd128`, so
    /// `is_available` is the compile-time answer — but Miri pins Scalar by
    /// design, so skip there too rather than interpret intrinsics Miri
    /// does not implement.
    fn wasm128_available() -> bool {
        !cfg!(miri) && Backend::Wasm128.is_available()
    }

    /// The dispatch table must hand out this kernel for
    /// [`Backend::Wasm128`], not the scalar fallback — the failure mode
    /// where the module exists but its arm was never flipped.
    #[test]
    fn dispatch_resolves_the_real_kernel() {
        if !wasm128_available() {
            return;
        }
        let wasm = crate::eks::bcrypt_lanes_fn(Backend::Wasm128);
        let scalar = crate::eks::bcrypt_lanes_fn(Backend::Scalar);
        assert!(
            !core::ptr::fn_addr_eq(wasm, scalar),
            "Backend::Wasm128 still resolves to the scalar kernel"
        );
    }

    /// Eight different passwords and eight different salts, one per lane,
    /// must reproduce eight scalar hashes bit for bit — including the
    /// OpenBSD `U*U` vector in lane 1 and the A/B half boundary (lanes 3
    /// and 4 must not leak into each other).
    #[test]
    fn lanes_match_scalar() {
        if !wasm128_available() {
            return;
        }
        let vector_salt =
            base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let passwords: [&[u8]; 8] = [
            b"",
            b"U*U",
            b"hunter2",
            &[0xAA; 72],
            b"correct horse battery staple",
            &[0x00; 1],
            b"\xff\xfe\xfd binary",
            b"a",
        ];
        let kws: [[u32; 18]; 8] = core::array::from_fn(|i| key_words_for(passwords[i]));
        let sws: [[u32; 4]; 8] = [
            salt_words(&[0x11; 16]),
            salt_words(&vector_salt),
            salt_words(&[0x33; 16]),
            salt_words(&[0x44; 16]),
            salt_words(&[0x55; 16]),
            salt_words(&[0x66; 16]),
            salt_words(&[0x77; 16]),
            salt_words(&[0x88; 16]),
        ];
        let mut outs = [[0u8; 24]; 8];
        // SAFETY: availability checked above; the slices are exactly 8 lanes.
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
