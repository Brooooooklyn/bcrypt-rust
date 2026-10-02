//! EksBlowfish backend selection: one batch kernel per instruction set, chosen
//! by **runtime** CPU feature detection.
//!
//! bcrypt's cost loop is `2^cost` iterations of two key expansions, each 521
//! strictly sequential Blowfish encryptions — one hash cannot be parallelized.
//! The SIMD backends therefore hash N *independent* passwords in lockstep, one
//! per vector lane, and are reached only through the batch entry points in
//! `crate::core`. A single hash always runs on [`Backend::Scalar`].
//!
//! # Kernel contract
//!
//! Every backend exposes one `bcrypt_lanes` with the [`BcryptLanesFn`]
//! signature, hashing exactly [`Backend::lanes`] passwords per call. Inputs are
//! pre-digested by the caller (see `crate::core`):
//!
//! * `key_words[lane][i]` — the password's i-th cyclical big-endian key word
//!   (the 72-byte-padded, NUL-terminated bcrypt key stream reduced to the 18
//!   words the P-array XOR actually consumes);
//! * `salt_words[lane][i]` — the lane's 16-byte salt as 4 big-endian words;
//! * `outs[lane]` — receives the 24-byte ciphertext (of which callers encode 23).
//!
//! Pre-computing the words in safe scalar code keeps the byte-cycling logic in
//! exactly one tested place; the kernels are pure lockstep vector code.
//!
//! # Cost model
//!
//! * Detection runs at most once per process. The result is cached in a
//!   [`AtomicU8`] read with [`Ordering::Relaxed`]. The initialisation race is
//!   benign: every thread computes the same answer, so a duplicated `detect()`
//!   only wastes a `cpuid`. (The one exception: the width shootout below can
//!   make a racing duplicate *measure* twice; the cache is single-assignment,
//!   so all threads still settle on one pick.)
//! * A batch call resolves the function pointer **once**, before chunking into
//!   lane-width groups (see `crate::core`). Nothing dispatches inside the
//!   per-group loop.
//! * Per batch: one relaxed load plus one compare. Per group: one indirect
//!   call. A group is `lanes` full bcrypt hashes — expensive at any realistic
//!   cost — so dispatch overhead is nil.
//!
//! # The x86-64 width shootout (AVX2 vs AVX-512)
//!
//! Wider is not automatically faster for bcrypt. Every lane carries a 4 KiB
//! mutable S-box working set, so the 16-lane AVX-512 kernel's struct-of-
//! arrays touches 64 KiB per group — twice the 32 KiB L1d of Zen 3/4, whose
//! 512-bit ops are also double-pumped — while Sapphire Rapids has a 6.0c zmm
//! hardware gather and 48 KiB L1d, where the wide kernel pulls clearly ahead
//! (`docs/research/research-x86-microarch.md`, §4–7). The measured spread on
//! this crate confirms both sides: Zen 4 is at parity (avx2 888.6 vs avx512
//! 889.2 hashes/s, cost 5 batch 16, rustc 1.99) while SPR-class cores favor
//! AVX-512 decisively. No vendor table captures that split — Zen 5, mobile
//! Zen 5 and Turin Dense all differ again — so [`detect`] settles it by
//! *measurement*: with `std`, optimized code and no Miri, a CPU advertising
//! both AVX2 and AVX-512F runs a one-time width shootout — the same fixed
//! 32-password batch at cost 4 through both backends' normal kernels (which
//! also settles each backend's own flavor shootout), outputs asserted
//! byte-identical before anything is timed (a mismatch is a kernel bug, not
//! a tie), three interleaved timed reps each, min wins. The winner is cached
//! in `CACHED_WIDTH`; a tie keeps the static order. The one-time cost is
//! measured, not theoretical: ~0.4 s on the shared 4-vCPU Zen 4 sandbox
//! (a cost-4 hash there is ~0.55 ms, and the batch is hashed eight times per
//! side — once for the assert, three times for the clock — plus each
//! backend's own flavor shootout; the winner's would be paid by the first
//! batch call anyway). Debug, `no_std` and Miri builds never time anything —
//! they keep the static `Avx512`-first order.
//!
//! # `no_std`
//!
//! Runtime detection needs `std`. Without the `std` feature, [`detect`] falls
//! back to compile-time `cfg(target_feature = ...)` and then to
//! [`Backend::Scalar`].
//!
//! # Testing under Rosetta on aarch64-apple-darwin
//!
//! `is_x86_feature_detected!` expands to
//! `cfg!(target_feature = "...") || runtime_cpuid_check()`, so a compile-time
//! `target_feature` short-circuits it to `true`. That matters here because
//! Rosetta 2 *executes* AVX2 but does not advertise it in `cpuid`:
//!
//! * `cargo test --target x86_64-apple-darwin` — [`detect`] stops at
//!   [`Backend::Sse41`] and `Backend::Avx2.is_available()` is `false`.
//! * `RUSTFLAGS="-C target-feature=+avx2" cargo test --target x86_64-apple-darwin`
//!   — [`detect`] returns [`Backend::Avx2`] and it really runs.
//! * Never add `+avx512f`: `is_available()` would then report `true` while the
//!   instruction itself traps with `SIGILL` (Rosetta has no AVX-512).

use core::sync::atomic::{AtomicU8, Ordering};

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse41;

// WebAssembly has no runtime feature detection a module can survive (SIMD
// instructions fail validation on engines that lack them), so the module
// exists exactly when `cfg(all(target_arch = "wasm32", target_feature =
// "simd128"))` held at compile time — the same condition
// `have_wasm_simd128` answers from.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm128;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Which `bcrypt_lanes` implementation to use.
///
/// A variant is added every time another ISA is ported, so this is
/// `#[non_exhaustive]` — write a `_` arm downstream. [`Backend::ALL`] is a
/// slice for the same reason: its length must not be part of the API.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum Backend {
    /// Portable scalar code, one lane. Always available.
    Scalar = 0,
    /// AArch64 NEON, 8 lanes.
    Neon = 1,
    /// x86 / x86-64 SSE4.1 (`pinsrd`/`pextrd` for lane construction), 4 lanes.
    Sse41 = 2,
    /// x86-64 AVX2, 8 lanes. Two internal flavours (hardware gather vs
    /// load+insert); a one-time shootout picks per CPU — AMD Zen gathers are
    /// microcoded and lose.
    Avx2 = 3,
    /// x86-64 AVX-512F, 16 lanes. Same flavour shootout as AVX2.
    Avx512 = 4,
    /// wasm32 fixed-width SIMD128, 8 lanes as two interleaved 4-lane
    /// `v128` states (X2). Compile-time selected.
    Wasm128 = 5,
}

/// The signature every backend's batch kernel has.
///
/// `key_words`, `salt_words` and `outs` must all have length
/// [`Backend::lanes`] for the backend the pointer was resolved for.
///
/// # Safety
///
/// Calling one of these requires:
///
/// * the CPU to support the backend's instruction set — check
///   [`Backend::is_available`], or get the pointer from [`bcrypt_lanes_fn`]
///   fed by [`backend`];
/// * the three slices to be valid and of exactly the backend's lane count;
/// * no other thread to be writing `outs` at the same time.
pub type BcryptLanesFn = unsafe fn(u32, &[[u32; 18]], &[[u32; 4]], &mut [[u8; 24]]);

impl Backend {
    /// Every backend, in ascending preference order.
    pub const ALL: &'static [Backend] = &[
        Backend::Scalar,
        Backend::Neon,
        Backend::Sse41,
        Backend::Avx2,
        Backend::Avx512,
        Backend::Wasm128,
    ];

    /// Short lowercase name, handy for bench ids and test output.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Neon => "neon",
            Backend::Sse41 => "sse41",
            Backend::Avx2 => "avx2",
            Backend::Avx512 => "avx512",
            Backend::Wasm128 => "wasm128",
        }
    }

    /// Passwords hashed per kernel call — the vector width in 32-bit lanes.
    #[inline]
    #[must_use]
    pub const fn lanes(self) -> usize {
        match self {
            Backend::Scalar => 1,
            Backend::Neon => 8,
            Backend::Sse41 => 4,
            Backend::Avx2 => 8,
            Backend::Avx512 => 16,
            // The kernel's lane count lives next to the kernel; the
            // module only exists on simd128-enabled wasm32, so off-arch
            // builds keep the constant inline (the backend is never
            // available there — the value is dead).
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Wasm128 => wasm128::LANES,
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            Backend::Wasm128 => 8,
        }
    }

    /// Whether this CPU can execute this backend *right now*.
    ///
    /// Unlike `detect`, this asks about one specific backend, so tests and
    /// benches can loop over [`Backend::ALL`] and skip what the host cannot run.
    #[inline]
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Neon => have_neon(),
            Backend::Sse41 => have_sse41(),
            Backend::Avx2 => have_avx2(),
            Backend::Avx512 => have_avx512f(),
            Backend::Wasm128 => have_wasm_simd128(),
        }
    }

    #[inline]
    const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Total inverse of [`Backend::to_u8`]; anything unknown maps to
    /// [`Backend::Scalar`] so the cache can never produce a panic.
    #[inline]
    const fn from_u8(value: u8) -> Backend {
        match value {
            1 => Backend::Neon,
            2 => Backend::Sse41,
            3 => Backend::Avx2,
            4 => Backend::Avx512,
            5 => Backend::Wasm128,
            _ => Backend::Scalar,
        }
    }
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Per-feature probes
// ---------------------------------------------------------------------------
//
// Each probe is defined twice under mutually exclusive `cfg`s, so exactly one
// definition ever exists and there is no dead or unreachable code. With `std`
// the probe is a real runtime check; without it, it degrades to the
// compile-time `target_feature` cfg.

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn have_avx512f() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
fn have_avx512f() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn have_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
fn have_avx2() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn have_sse41() -> bool {
    std::arch::is_x86_feature_detected!("sse4.1")
}
#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
#[inline]
fn have_sse41() -> bool {
    cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse4.1"
    ))
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[inline]
fn have_neon() -> bool {
    // NEON (Advanced SIMD) is in the architectural baseline Apple and
    // Windows guarantee for aarch64; probing the OS for it would be a
    // formality, so on those platforms the answer is compile-time.
    #[cfg(any(target_vendor = "apple", target_os = "windows"))]
    {
        true
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "windows")))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
}
#[cfg(not(all(feature = "std", target_arch = "aarch64")))]
#[inline]
fn have_neon() -> bool {
    cfg!(all(target_arch = "aarch64", target_feature = "neon"))
}

/// wasm32 SIMD128. There is no runtime probe a wasm module can survive
/// (SIMD instructions fail validation where unsupported), so the answer is
/// purely compile-time: it is `true` exactly when the crate was built with
/// `-C target-feature=+simd128`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn have_wasm_simd128() -> bool {
    true
}
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn have_wasm_simd128() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Detection and caching
// ---------------------------------------------------------------------------

/// Sentinel meaning "detection has not run yet". Not a valid [`Backend`] value.
const UNINIT: u8 = 0xFF;

/// Cached [`Backend`] as a `u8`, or [`UNINIT`].
static CACHED_BACKEND: AtomicU8 = AtomicU8::new(UNINIT);

/// Run the feature cascade and return the best backend for this CPU.
///
/// Preference order: `Avx512 > Avx2 > Sse41` on x86(-64), `Neon` on AArch64,
/// `Wasm128` on simd128-enabled wasm32, `Scalar` everywhere else. The probes
/// are arch-gated, so the single cascade below cannot pick an off-arch backend.
/// The one exception to the static order is the module-level width shootout:
/// where it applies it can demote `Avx512` to `Avx2` — by measurement.
///
/// The cascade itself always re-runs (a few `cpuid` reads, each cached by the
/// standard library); the one expensive decision — the width shootout, when
/// it applies — is cached after the first call, so a repeated `detect()`
/// stays cheap. Use [`backend`] for the fully cached value.
#[must_use]
pub fn detect() -> Backend {
    if cfg!(miri) {
        // Miri interprets the crate: its intrinsic support stops around
        // SSE2, and wall-clock flavour shootouts mean nothing under an
        // interpreter. Scalar is the one backend that behaves identically
        // on every host Miri runs on, which is also what makes the Miri CI
        // job arch-independent.
        Backend::Scalar
    } else if have_avx512f() {
        // Both x86-64 SIMD widths exist; which is faster is a µarch
        // question the width shootout answers where it can (module docs).
        width_pick()
    } else if have_avx2() {
        Backend::Avx2
    } else if have_sse41() {
        Backend::Sse41
    } else if have_neon() {
        Backend::Neon
    } else if have_wasm_simd128() {
        Backend::Wasm128
    } else {
        Backend::Scalar
    }
}

/// Detect and populate the cache. Outlined so [`backend`] stays tiny.
#[cold]
#[inline(never)]
fn detect_and_cache() -> Backend {
    let detected = detect();
    // Relaxed is enough: the value is a plain `u8` with no associated data, and
    // every thread that races here computes the same answer.
    CACHED_BACKEND.store(detected.to_u8(), Ordering::Relaxed);
    detected
}

/// The backend for this process: one relaxed atomic load on the hot path.
///
/// Deliberately not a `OnceLock` — acquire ordering would buy nothing here
/// (there is no data to publish) and `OnceLock` needs `std`.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    let cached = CACHED_BACKEND.load(Ordering::Relaxed);
    if cached == UNINIT {
        detect_and_cache()
    } else {
        Backend::from_u8(cached)
    }
}

// ---------------------------------------------------------------------------
// Width shootout: AVX2 vs AVX-512 (x86-64, std, optimized, non-Miri only)
// ---------------------------------------------------------------------------

/// Cached width-shootout winner as a `u8`, or [`UNINIT`] — the same sentinel
/// protocol as [`CACHED_BACKEND`]. Only ever holds [`Backend::Avx2`] or
/// [`Backend::Avx512`]; single-assignment (see [`width_shootout_and_cache`]).
#[cfg(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri)))]
static CACHED_WIDTH: AtomicU8 = AtomicU8::new(UNINIT);

/// The answer to "AVX-512F advertised": which width actually wins on *this*
/// CPU. With a clock and optimized code a one-time shootout decides (module
/// docs); it needs AVX2 as the alternative, and without it the static order
/// stands. Reached only from [`detect`]'s `have_avx512f()` arm, so AVX-512F
/// is known available here.
#[cfg(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri)))]
fn width_pick() -> Backend {
    if !have_avx2() {
        return Backend::Avx512;
    }
    let cached = CACHED_WIDTH.load(Ordering::Relaxed);
    if cached != UNINIT {
        return Backend::from_u8(cached);
    }
    width_shootout_and_cache()
}

/// See the measuring variant above. Debug builds, `no_std` and Miri keep the
/// static AVX-512-first order: timing unoptimized/interpreted intrinsics
/// measures the codegen mode, not the µarch — and without `std` there is no
/// clock at all (the flavor shootouts make the same trade).
#[cfg(not(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri))))]
fn width_pick() -> Backend {
    Backend::Avx512
}

/// Run the shootout and publish the winner. `compare_exchange` makes the
/// cache single-assignment: a racing duplicate measures the same class of
/// answer, but on parity machines two measurements can flip, and every
/// thread must agree with the value [`detect_and_cache`] may already have
/// published. First store wins; losers take the established pick.
#[cfg(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri)))]
#[cold]
#[inline(never)]
fn width_shootout_and_cache() -> Backend {
    let picked = width_shootout();
    match CACHED_WIDTH.compare_exchange(UNINIT, picked.to_u8(), Ordering::Relaxed, Ordering::Relaxed)
    {
        Ok(_) => picked,
        Err(established) => Backend::from_u8(established),
    }
}

/// The one-time width shootout: both x86-64 SIMD backends hash the same fixed
/// deterministic 32-password batch at cost 4 — correctness first
/// (byte-identical outputs; a mismatch is a kernel bug, not a tie), then
/// three interleaved timed reps each, min per side, faster wins. A tie keeps
/// the static order (AVX-512). The batch is 32 items — four AVX2 groups /
/// two AVX-512 groups, a multiple of both lane counts, so no padded tail
/// distorts either side and both backends hash the same 32 passwords, making
/// raw elapsed time a per-hash comparison.
#[cfg(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri)))]
fn width_shootout() -> Backend {
    use std::time::Instant;

    // SplitMix64, fixed seed: deterministic inputs (not a CSPRNG, nor does
    // it need to be).
    let mut rng = 0xBC79_7A5A_F1A0_0003u64;
    let mut next_u32 = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    };
    const ITEMS: usize = 32;
    let mut kws = [[0u32; 18]; ITEMS];
    let mut sws = [[0u32; 4]; ITEMS];
    for w in kws.as_flattened_mut().iter_mut().chain(sws.as_flattened_mut()) {
        *w = next_u32();
    }

    // One backend's kernel over the whole batch, in lane-width groups — the
    // same dispatch `crate::core`'s chunk loop performs.
    fn run_batch(
        kernel: BcryptLanesFn,
        lanes: usize,
        kws: &[[u32; 18]; ITEMS],
        sws: &[[u32; 4]; ITEMS],
        outs: &mut [[u8; 24]; ITEMS],
    ) {
        let mut base = 0;
        while base < ITEMS {
            // SAFETY: `width_shootout` runs only when both AVX2 and AVX-512F
            // are advertised (`width_pick`'s gate), so both kernels target
            // this CPU's instruction set. `ITEMS` is a multiple of both lane
            // counts, so every group slice is exactly `lanes` long, and each
            // `outs` group is exclusively owned by this frame.
            unsafe {
                kernel(
                    4,
                    &kws[base..base + lanes],
                    &sws[base..base + lanes],
                    &mut outs[base..base + lanes],
                )
            };
            base += lanes;
        }
    }

    let avx2 = bcrypt_lanes_fn(Backend::Avx2);
    let avx512 = bcrypt_lanes_fn(Backend::Avx512);
    let mut outs_avx2 = [[0u8; 24]; ITEMS];
    let mut outs_avx512 = [[0u8; 24]; ITEMS];
    // Correctness before timing: these first calls also run each backend's
    // own flavor shootout, so both kernels are fully warmed before the clock
    // starts.
    run_batch(avx2, Backend::Avx2.lanes(), &kws, &sws, &mut outs_avx2);
    run_batch(avx512, Backend::Avx512.lanes(), &kws, &sws, &mut outs_avx512);
    assert_eq!(
        outs_avx2, outs_avx512,
        "width shootout: avx2 and avx512 kernels diverged — a kernel bug, not timing"
    );
    let (mut best_avx2, mut best_avx512) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        let start = Instant::now();
        run_batch(avx2, Backend::Avx2.lanes(), &kws, &sws, &mut outs_avx2);
        best_avx2 = best_avx2.min(start.elapsed().as_secs_f64());
        let start = Instant::now();
        run_batch(avx512, Backend::Avx512.lanes(), &kws, &sws, &mut outs_avx512);
        best_avx512 = best_avx512.min(start.elapsed().as_secs_f64());
        core::hint::black_box(&mut outs_avx2);
        core::hint::black_box(&mut outs_avx512);
    }
    if best_avx2 < best_avx512 {
        Backend::Avx2
    } else {
        Backend::Avx512
    }
}

/// The `bcrypt_lanes` implementation for `backend`.
///
/// Resolve this **once per batch call**, outside every loop.
///
/// On an architecture that has no module for the requested backend, this
/// returns the scalar implementation rather than failing to compile, so tests
/// can iterate over [`Backend::ALL`] on any host. It does **not** check
/// availability: a pointer for a backend this CPU lacks will fault when called.
/// Use [`Backend::is_available`] first, or take the value from [`backend`].
#[must_use]
pub fn bcrypt_lanes_fn(backend: Backend) -> BcryptLanesFn {
    // On-arch Neon, Sse41, Avx2, Avx512 and (compile-time-gated) Wasm128
    // run their real kernels; every off-arch arm resolves to scalar.
    match backend {
        Backend::Scalar => scalar::bcrypt_lanes,
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => neon::bcrypt_lanes,
        #[cfg(not(target_arch = "aarch64"))]
        Backend::Neon => scalar::bcrypt_lanes,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse41 => sse41::bcrypt_lanes,
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::Sse41 => scalar::bcrypt_lanes,
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => avx2::bcrypt_lanes,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx2 => scalar::bcrypt_lanes,
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => avx512::bcrypt_lanes,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 => scalar::bcrypt_lanes,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Backend::Wasm128 => wasm128::bcrypt_lanes,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        Backend::Wasm128 => scalar::bcrypt_lanes,
    }
}

/// Reduce a padded bcrypt key stream to the 18 cyclical big-endian words the
/// P-array XOR consumes — OpenBSD `Blowfish_stream2word` applied 18 times.
///
/// `key` is the 72-byte buffer and `key_len` the cycling length (1..=72):
/// word `i` is `key[j]<<24 | key[j+1]<<16 | key[j+2]<<8 | key[j+3]` with each
/// index taken mod `key_len`. Only 18 words (72 bytes) are ever read, which is
/// why a ≥72-byte password's appended NUL never participates.
#[must_use]
pub(crate) fn expand_key_words(key: &[u8; 72], key_len: usize) -> [u32; 18] {
    debug_assert!((1..=72).contains(&key_len));
    let mut words = [0u32; 18];
    let mut j = 0usize;
    for w in &mut words {
        let mut word = 0u32;
        for _ in 0..4 {
            if j >= key_len {
                j = 0;
            }
            word = (word << 8) | u32::from(key[j]);
            j += 1;
        }
        *w = word;
    }
    words
}

/// The 16-byte salt as 4 big-endian words, matching `stream2word` on a
/// 16-byte input (which never wraps: 16 is a multiple of 4).
#[must_use]
pub(crate) fn salt_words(salt: &[u8; 16]) -> [u32; 4] {
    let mut words = [0u32; 4];
    for (w, chunk) in words.iter_mut().zip(salt.chunks_exact(4)) {
        *w = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_u8_round_trip() {
        for &b in Backend::ALL {
            assert_eq!(Backend::from_u8(b.to_u8()), b);
        }
        // The sentinel and anything unknown must degrade, never panic.
        assert_eq!(Backend::from_u8(UNINIT), Backend::Scalar);
        assert_eq!(Backend::from_u8(200), Backend::Scalar);
    }

    #[test]
    fn cache_agrees_with_detect() {
        let first = backend();
        assert_eq!(first, detect());
        // Second call takes the cached path.
        assert_eq!(backend(), first);
        assert_ne!(CACHED_BACKEND.load(Ordering::Relaxed), UNINIT);
    }

    /// Where both x86-64 SIMD backends exist, detection's pick is one of the
    /// two and stable across calls — the width shootout's winner is cached.
    /// Hosts without both widths (including Miri, which pins Scalar, and
    /// Rosetta, which lacks AVX-512) have no pick to make: skip.
    #[test]
    fn width_pick_is_stable_and_cached() {
        if cfg!(miri) || !(Backend::Avx2.is_available() && Backend::Avx512.is_available()) {
            return;
        }
        let first = detect();
        assert!(matches!(first, Backend::Avx2 | Backend::Avx512));
        // Second call reads the cached pick: same answer, and where the
        // shootout is compiled in at all its cache is populated.
        assert_eq!(detect(), first);
        #[cfg(all(feature = "std", target_arch = "x86_64", not(debug_assertions), not(miri)))]
        assert_ne!(CACHED_WIDTH.load(Ordering::Relaxed), UNINIT);
    }

    #[test]
    fn detected_backend_is_available() {
        assert!(detect().is_available());
        assert!(Backend::Scalar.is_available());
    }

    #[test]
    fn detection_respects_the_architecture() {
        if cfg!(target_arch = "aarch64") {
            assert!(!have_sse41());
            assert!(!have_avx2());
            assert!(!have_avx512f());
            // NEON is baseline on aarch64-apple-darwin.
            if cfg!(target_vendor = "apple") {
                assert_eq!(detect(), Backend::Neon);
            } else {
                assert!(matches!(detect(), Backend::Neon | Backend::Scalar));
            }
        }
        if cfg!(target_arch = "x86_64") {
            assert!(!have_neon());
            assert!(matches!(
                detect(),
                Backend::Sse41 | Backend::Avx2 | Backend::Avx512 | Backend::Scalar
            ));
        }
        if cfg!(target_arch = "wasm32") {
            assert!(!have_sse41());
            assert!(!have_avx2());
            assert!(!have_avx512f());
            assert!(!have_neon());
            if cfg!(target_feature = "simd128") {
                assert_eq!(detect(), Backend::Wasm128);
            } else {
                assert_eq!(detect(), Backend::Scalar);
            }
        }
    }

    #[test]
    fn every_backend_resolves_to_a_function() {
        for &b in Backend::ALL {
            let f = bcrypt_lanes_fn(b);
            let scalar = bcrypt_lanes_fn(Backend::Scalar);
            if b == Backend::Scalar {
                assert!(core::ptr::fn_addr_eq(f, scalar));
            }
        }
    }

    #[test]
    fn expand_key_words_cycles_big_endian() {
        // "a" + NUL: key_len 2, words cycle over bytes [0x61, 0x00].
        let mut key = [0u8; 72];
        key[0] = b'a';
        let w = expand_key_words(&key, 2);
        assert_eq!(w[0], 0x61006100);
        assert_eq!(w[17], 0x61006100);
        // A full-length key: word i reads bytes 4i..4i+3, no wrap.
        let key72: [u8; 72] = core::array::from_fn(|i| i as u8);
        let w = expand_key_words(&key72, 72);
        assert_eq!(w[0], 0x00010203);
        assert_eq!(w[17], 0x44454647);
        // key_len 71 wraps inside word 17: bytes 68,69,70, then byte 0.
        let w = expand_key_words(&key72, 71);
        assert_eq!(w[17], 0x44454600);
    }

    #[test]
    fn salt_words_are_big_endian() {
        let salt: [u8; 16] = core::array::from_fn(|i| (i + 1) as u8);
        assert_eq!(
            salt_words(&salt),
            [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10]
        );
    }
}
