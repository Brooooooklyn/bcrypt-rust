//! wasm32 SIMD128 EksBlowfish: four independent bcrypt hashes in lockstep,
//! one per 32-bit `v128` lane.
//!
//! Ports [`super::scalar`] exactly — same OpenBSD structure, same
//! continuous salt-word stream across the P→S transition in
//! `expand_state`, same key-then-salt cost-loop order — with every 32-bit
//! operation widened to a lane-wise `v128` one. Scalar `wrapping_add`
//! becomes `i32x4_add` (wraps, which is what Blowfish wants); every XOR
//! becomes `v128_xor`.
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
//! # State layout: struct-of-arrays
//!
//! The four S-boxes are stored interleaved by lane — [`SBoxes4x`] — so
//! entry `(b, i)` holds all four lanes contiguously and expansion-chain
//! writes stay full-vector stores. Lookups are gathers: lane `l` of a
//! lookup reads `box_base[idx_l * 4 + l]`.
//!
//! # No gather at this ISA — one lookup spelling
//!
//! wasm SIMD128 has no gather instruction, so every lookup is four scalar
//! loads; the only question is how the indices cross between the vector
//! and scalar domains. wasm has no extract-to-memory penalty to weigh
//! against insert chains, so there is one spelling, the same "stack
//! round-trip" `super::sse41` ships: store the index vector, four scalar
//! loads, load the results back. The four loads are independent and the
//! engine's store-to-load forwarding handles the crossings.
//!
//! # Lookup invariant
//!
//! Every index vector handed to [`lookup4`] is a byte-masked value (a
//! `>> 24` or `& 0xff` of a `u32` lane) produced inside [`f4`], so every
//! lane is in `0..=255` and `idx * 4 + lane` stays inside the addressed
//! 256-entry box — a gather can never leave the 16 KiB [`SBoxes4x`]. The
//! invariant is stated once here and referenced at every unsafe
//! dereference below.
//!
//! # Zeroization
//!
//! With the `zeroize` feature the kernel wipes its named key-material
//! buffers — the `State4x` expansion state, the `kwv` transposed key
//! schedule and `cdata` — before returning, best-effort like the rest of
//! the crate: compiler spills and transposition-internal temporaries are
//! not chased.

use core::arch::wasm32::{
    i32x4_add, u32x4_shr, u32x4_splat, v128, v128_and, v128_load, v128_store, v128_xor,
};

use crate::consts::{P_INIT, S_INIT};

/// Passwords per kernel call — the four 32-bit `v128` lanes.
const LANES: usize = 4;

/// Four lockstep S-boxes, interleaved by lane: entry `b * 256 + i` is the
/// `(b, i)` Blowfish S-box word of all four lanes, lane `l` at u32 offset
/// `(b * 256 + i) * 4 + l`. 16 KiB, 64-byte aligned, so every entry is
/// 16-byte aligned by construction (the entry stride divides the struct
/// alignment).
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
    p: [v128; 18],
    s: SBoxes4x,
}

/// Gather one S-box word per lane: lane `l` loads `box_base[idx_l * 4 + l]`.
/// Stack round-trip spelling (the only one — see the module docs): store
/// the index vector, four scalar loads, load the results back.
///
/// # Safety
///
/// `box_base` must point at the first word of one 256-entry box of an
/// [`SBoxes4x`], and every lane of `idx` must be in `0..=255` — the
/// module-level lookup invariant. Then `idx_l * 4 + l < 1024` and each
/// load stays inside that box's 4 KiB.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn lookup4(box_base: *const u32, idx: v128) -> v128 {
    let mut ix = [0u32; 4];
    // SAFETY: `ix` is a live 16-byte stack array, writable in full.
    unsafe { v128_store(ix.as_mut_ptr().cast::<v128>(), idx) };
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
    unsafe { v128_load(gathered.as_ptr().cast::<v128>()) }
}

/// The Blowfish round function, lane-wise: `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`,
/// bytes taken MSB-first, every addition wrapping (`i32x4_add` wraps,
/// which is what the scalar `wrapping_add` does). The byte masks below are
/// what make the [`lookup4`] gathers safe — see the module-level invariant.
#[target_feature(enable = "simd128")]
#[inline]
fn f4(s: &SBoxes4x, x: v128) -> v128 {
    let byte_mask = u32x4_splat(0xff);
    let a = u32x4_shr(x, 24);
    let b = v128_and(u32x4_shr(x, 16), byte_mask);
    let c = v128_and(u32x4_shr(x, 8), byte_mask);
    let d = v128_and(x, byte_mask);
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
    i32x4_add(v128_xor(i32x4_add(va, vb), vc), vd)
}

/// The 16 Feistel rounds of one lockstep encryption, as a macro so every
/// caller gets a *textually* inlined copy — the same reasoning as
/// `super::sse41`'s `rounds16!` (an outlined `#[target_feature]` rounds fn
/// let LLVM keep the box-base math per call). Fully unrolled swap-free
/// round pairs: the pair ends in reference post-swap orientation, so the
/// whitening lands on the exchanged registers and the final swap-undo is
/// the one `mem::swap` after the pairs (register renaming; free).
macro_rules! rounds16 {
    ($state:expr, $l:ident, $r:ident) => {{
        macro_rules! pair {
            ($i:expr) => {{
                $l = v128_xor($l, $state.p[$i]);
                $r = v128_xor($r, f4(&$state.s, $l));
                $r = v128_xor($r, $state.p[$i + 1]);
                $l = v128_xor($l, f4(&$state.s, $r));
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
        $l = v128_xor($l, $state.p[16]);
        $r = v128_xor($r, $state.p[17]);
        core::mem::swap(&mut $l, &mut $r);
    }};
}

/// One Blowfish block encryption, four lanes in lockstep: 16 Feistel
/// rounds with the final swap undone, the output halves whitened by
/// P[16]/P[17] — identical control flow to scalar `encipher`.
#[target_feature(enable = "simd128")]
#[inline]
fn encipher4(state: &State4x, mut l: v128, mut r: v128) -> (v128, v128) {
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
#[target_feature(enable = "simd128")]
#[inline]
fn expand_state_v4(state: &mut State4x, swv: &[v128; 4], kwv: &[v128; 18]) {
    for ((p, &init), &k) in state.p.iter_mut().zip(P_INIT.iter()).zip(kwv.iter()) {
        *p = v128_xor(u32x4_splat(init), k);
    }
    for (entry, &init) in state.s.0.iter_mut().zip(S_INIT.as_flattened().iter()) {
        *entry = [init; 4];
    }
    let zero = u32x4_splat(0);
    let (mut l, mut r) = (zero, zero);
    let mut j = 0usize;
    for pair in 0..9 {
        l = v128_xor(l, swv[j % 4]);
        j += 1;
        r = v128_xor(r, swv[j % 4]);
        j += 1;
        rounds16!(state, l, r);
        state.p[2 * pair] = l;
        state.p[2 * pair + 1] = r;
    }
    for b in 0..4 {
        for pair in 0..128 {
            l = v128_xor(l, swv[j % 4]);
            j += 1;
            r = v128_xor(r, swv[j % 4]);
            j += 1;
            rounds16!(state, l, r);
            let e = b * 256 + 2 * pair;
            // SAFETY: entries `e`/`e + 1` are 16-byte aligned by
            // construction (struct align 64, entry stride 16) and
            // exclusively owned — `v128_store` writes the full vectors.
            unsafe {
                v128_store(state.s.0[e].as_mut_ptr().cast::<v128>(), l);
                v128_store(state.s.0[e + 1].as_mut_ptr().cast::<v128>(), r);
            }
        }
    }
}

/// The 521-encryption zero chain shared by both lockstep `expand0state`
/// variants: overwrite P (9 pairs) then S (512 pairs) exactly as
/// [`expand_state_v4`] does, minus the salt mixing.
#[target_feature(enable = "simd128")]
#[inline]
fn encrypt_zero_chain_v4(state: &mut State4x) {
    let zero = u32x4_splat(0);
    let (mut l, mut r) = (zero, zero);
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
                v128_store(state.s.0[e].as_mut_ptr().cast::<v128>(), l);
                v128_store(state.s.0[e + 1].as_mut_ptr().cast::<v128>(), r);
            }
        }
    }
}

/// Lockstep `Blowfish_expand0state(key)`: XOR the password words into P,
/// then run the zero chain.
#[target_feature(enable = "simd128")]
#[inline]
fn expand0state_v4(state: &mut State4x, wv: &[v128; 18]) {
    for (p, &w) in state.p.iter_mut().zip(wv.iter()) {
        *p = v128_xor(*p, w);
    }
    encrypt_zero_chain_v4(state);
}

/// Lockstep `Blowfish_expand0state(salt)`: the salt is exactly 4 words, so
/// the P XOR cycles it (`i & 3`), then the same zero chain.
#[target_feature(enable = "simd128")]
#[inline]
fn expand0state_salt_v4(state: &mut State4x, swv: &[v128; 4]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        *p = v128_xor(*p, swv[i & 3]);
    }
    encrypt_zero_chain_v4(state);
}

/// Transpose the four lanes' key words into 18 vectors: vector `i` holds
/// word `i` of lanes 0..=3. wasm has no `setr`-style intrinsic, so each
/// vector is a plain 16-byte stack load — fine here, as nothing in this
/// function is on the cost-loop path.
#[target_feature(enable = "simd128")]
#[inline]
fn transpose_keys(key_words: &[[u32; 18]]) -> [v128; 18] {
    debug_assert_eq!(key_words.len(), LANES);
    let mut out = [u32x4_splat(0); 18];
    for (i, v) in out.iter_mut().enumerate() {
        let words = [
            key_words[0][i],
            key_words[1][i],
            key_words[2][i],
            key_words[3][i],
        ];
        // SAFETY: `words` is a live 16-byte stack array, readable in full.
        *v = unsafe { v128_load(words.as_ptr().cast::<v128>()) };
    }
    out
}

/// Transpose the four lanes' salt words into 4 vectors; see
/// [`transpose_keys`].
#[target_feature(enable = "simd128")]
#[inline]
fn transpose_salts(salt_words: &[[u32; 4]]) -> [v128; 4] {
    debug_assert_eq!(salt_words.len(), LANES);
    let mut out = [u32x4_splat(0); 4];
    for (i, v) in out.iter_mut().enumerate() {
        let words = [
            salt_words[0][i],
            salt_words[1][i],
            salt_words[2][i],
            salt_words[3][i],
        ];
        // SAFETY: `words` is a live 16-byte stack array, readable in full.
        *v = unsafe { v128_load(words.as_ptr().cast::<v128>()) };
    }
    out
}

/// The whole lockstep bcrypt: one key+salt expansion, `2^cost` rounds of
/// key-then-salt expansion (OpenBSD order), then 64 encryptions of the
/// "OrpheanBeholderScryDoubt" constant and a big-endian store per lane.
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
    let mut kwv = transpose_keys(key_words);
    let swv = transpose_salts(salt_words);
    let mut state = State4x {
        p: [u32x4_splat(0); 18],
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
        u32x4_splat(0x4f72_7068),
        u32x4_splat(0x6561_6e42),
        u32x4_splat(0x6568_6f6c),
        u32x4_splat(0x6465_7253),
        u32x4_splat(0x6372_7944),
        u32x4_splat(0x6f75_6274),
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
        unsafe { v128_store(words.as_mut_ptr().cast::<v128>(), cv) };
        for (lane, out) in outs.iter_mut().enumerate() {
            out[4 * w..4 * w + 4].copy_from_slice(&words[lane].to_be_bytes());
        }
    }
    #[cfg(feature = "zeroize")]
    {
        // SAFETY: a `v128` is four `u32`s, so the `*mut u32` view
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

/// The SIMD128 batch kernel: [`LANES`] independent bcrypt hashes in
/// lockstep, one per 32-bit vector lane.
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
    // available in this engine and the three slices are exactly 4 lanes
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

    /// Four different passwords and four different salts, one per lane,
    /// must reproduce four scalar hashes bit for bit — including the
    /// OpenBSD `U*U` vector in lane 1.
    #[test]
    fn lanes_match_scalar() {
        if !wasm128_available() {
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
