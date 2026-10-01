# argon2-rust Architecture Report

Reference codebase: `/Users/brooklyn/workspace/github/argon2-rust` (crate `argon2-rust` v1.1.0).
Purpose: blueprint for a new crate (`bcrypt-rust`) that mirrors this architecture exactly.

Line references are to the files as read on 2026-10-01.

---

## 1. `Cargo.toml` — manifest conventions

Full path: `/Users/brooklyn/workspace/github/argon2-rust/Cargo.toml`

### Package metadata
- `edition = "2024"`, `rust-version = "1.89"` (Cargo.toml:7). The MSRV is set by exactly one thing and the comment says so: `stdarch_x86_avx512` (AVX-512 intrinsics + the `avx512f` target feature) stabilized in 1.89. "Verified, not guessed — the `msrv` job in ci.yml builds with exactly this toolchain on x86_64 and aarch64."
- `license = "MIT"`, `categories = ["cryptography", "no-std"]`, `keywords = ["argon2", "password-hash", "kdf", "simd", "crypto"]`.
- `exclude = ["/.github", "/benches", "/tests", "/fuzz", "/vm.sh", "/renovate.json"]` (line 17) — the published `.crate` ships only `src/`, `Cargo.toml`, README/LICENSE/NOTICE/CHANGELOG.
- docs.rs: `[package.metadata.docs.rs] all-features = true`, `rustdoc-args = ["--cfg", "docsrs"]` (lines 25-27).
- `autobenches = false` (line 22) with the reason in a comment: `benches/support/` holds a shared module that both benches `#[path]`-include, and auto-discovery would misinterpret it.

### Features (lines 31-55)
```toml
[features]
default = ["std", "parallel", "zeroize-memory"]

# Enables runtime CPU feature detection (needs `std::arch::is_*_feature_detected!`).
# Without it, backend selection falls back to compile-time `target_feature` cfgs.
# Also turns on `memchr/std` so PHC field scans use its runtime SIMD cascade.
std = ["memchr/std"]

# Multi-threaded `fill_memory_blocks` via `std::thread::scope`. Implies `std`.
parallel = ["std"]

# Securely wipe internal buffers (mirrors `FLAG_clear_internal_memory` in core.c).
zeroize-memory = []

# Internal benchmark/control for a reusable `bumpalo::Bump` inside `Workspace`. ...
bump-alloc = ["dep:bumpalo"]

# Exposes `argon2_rust::__internal` so tests/benches can drive individual
# backends. Not part of the stable API; enabled for dev builds via the
# self-dev-dependency below.
internal-api = []
```
Feature naming is kebab-case. `std` is the switch between runtime detection and compile-time `target_feature` fallback — this is the single most important feature interaction.

### Dependencies
- One mandatory dependency: `memchr = { version = "2.8.3", default-features = false }` (line 62), justified in a comment (don't maintain a second SIMD memchr; `default-features = false` keeps `no_std`).
- One optional dependency: `bumpalo` (default-features = false, optional), off by default with a long *measured* justification comment (lines 65-93) — the pattern is "document why a dependency does or doesn't exist, with numbers".

### Self-dev-dependency — the `internal-api` wiring (lines 98-100)
```toml
[dev-dependencies]
# Self-dev-dependency: makes `cargo test` / `cargo bench` build the lib with
# `internal-api` on, without leaking it into a plain `cargo build`.
# `default-features = false` keeps `cargo test --no-default-features` honest.
argon2-rust = { path = ".", default-features = false, features = ["internal-api"] }
```
This is how every test/bench gets `__internal` without the feature ever being on for downstream builds.

### Target-gated dev-dependencies (lines 102-107)
```toml
[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]
codspeed-criterion-compat = "5.0.1"
criterion = { version = "0.8", features = ["html_reports"] }
```
Criterion doesn't build for wasi; wasm runs only the lib + integration tests.

### Explicit bench/test targets (lines 109-159)
Six `[[bench]]` blocks, all `harness = false`: `argon2` (Criterion sweep), `codspeed` (CI regression net, "small, hermetic and single-threaded on purpose"), `micro` (plain `fn main()` fast-iteration timer), `blake2b`, `base64`, `base64_shootout` (dependency-free paired scalar/SIMD harness that also runs under Wasmtime). One custom `[[test]]`:
```toml
# Resident-set-size checks ... are a property of a *process* ...
# `harness = false` makes this file its own `fn main()` in its own process, so
# the measurement is contamination-free by construction rather than by slack.
[[test]]
name = "rss_isolation"
harness = false
```

### Profiles (lines 161-172)
```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
# `panic` deliberately left at the default: the library must not panic on any
# input reachable through the public API, so `abort` would buy nothing and it
# would break `cargo test`.

[profile.bench]
opt-level = 3
lto = "thin"
codegen-units = 1
```

---

## 2. `build.rs` — custom cfgs

`/Users/brooklyn/workspace/github/argon2-rust/build.rs`, 21 lines, does exactly one thing:

```rust
fn main() {
    println!("cargo:rustc-check-cfg=cfg(wasi_threadless)");
    let target = std::env::var("TARGET").expect("TARGET is always set for a build script");
    // wasm32-wasip1 (no threads) and wasm32-unknown-unknown (no std, so no
    // threads either). wasm32-wasip1-threads is deliberately NOT matched:
    // std::thread works there.
    if target == "wasm32-wasip1" || target == "wasm32-unknown-unknown" {
        println!("cargo:rustc-cfg=wasi_threadless");
    }
}
```
Rationale in the module docs: `cfg!(target_feature = "atomics")` cannot discriminate because `atomics` is an unstable wasm target feature that stable rustc never reports; the triple name is the only stable signal. Note the `rustc-check-cfg` line — every custom cfg is declared. (One deliberate exception: test-only cfgs like `argon2_force_avx2` are set out of band via RUSTFLAGS and the *modules* `#![allow(unexpected_cfgs)]` in their `#[cfg(test)]` blocks, e.g. `src/fill_block/avx2.rs:860`.)

---

## 3. `src/lib.rs` — public API surface and crate wiring

`/Users/brooklyn/workspace/github/argon2-rust/src/lib.rs` (391 lines).

### no_std / alloc setup (lines 196-218)
```rust
#![no_std]
#![warn(missing_docs)]
#![warn(unnameable_types)]          // regression guard: Hasher once shipped unnameable
#![warn(clippy::undocumented_unsafe_blocks)]

// NOTE FOR EVERY CONTRIBUTOR: this crate has a module named `core`, which
// shadows the `core` crate *in this root module only*. Inside `src/lib.rs`
// always write `::core::...`. Submodules are unaffected.
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;
```
The crate is `#![no_std]` + `alloc` always; `std` is opt-in and only gates detection/threading/entropy.

### Module organization (lines 219-248)
```rust
pub mod error;
pub mod params;

macro_rules! private_modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg_attr(not(feature = "internal-api"), allow(dead_code))]
            mod $name;
        )*
    };
}

private_modules!(base64, blake2b, block, core, encoding, fill_block, memory);

#[cfg(feature = "std")]
mod random;
```
The `private_modules!` macro is a deliberate alternative to a blanket `allow(dead_code)`: with `internal-api` OFF the internals are unreachable (allow), with it ON the compiler still reports genuinely dead code.

### Re-exports (lines 250-263)
```rust
pub use crate::core::{Argon2, BOUNDED_MAX_SALT_LEN, Hasher, constant_time_eq};
pub use crate::base64::Base64Backend;
pub use crate::blake2b::{Blake2bBackend, blake2b, blake2b_long};
#[cfg(feature = "std")]
pub use crate::core::RANDOM_SALT_LEN;
pub use crate::encoding::{
    Decoded, decode_base64, decode_phc, decode_string, encode_base64, encode_string,
    encode_string_alloc, encoded_len, from_base64, to_base64,
};
pub use crate::error::Error;
pub use crate::fill_block::Backend;
pub use crate::params::{Algorithm, Params, Version};
```

### Diagnostic accessors (lines 268-307)
Three `#[inline] #[must_use]` functions — `detected_backend()`, `detected_base64_backend()`, `detected_blake2b_backend()` — each a thin wrapper over the private module's cached `backend()` fn, each with a runnable doctest.

### The `__internal` module (lines 309-390)
```rust
/// Unstable internals, exposed for this crate's own tests and benches.
///
/// Gated behind the non-default `internal-api` feature. **No stability
/// guarantees**: anything here can change in a patch release.
///
/// # Soundness
///
/// Unstable is not the same as unsound. Every entry point here that takes an
/// explicit [`Backend`] or `Blake2bBackend` — including
/// `fill_memory_blocks_traced`, `hash_traced`, `hash_with_backend`,
/// `blake2b_with_backend`, and `blake2b_long_with_backend` — is an `unsafe fn`,
/// and so is each backend's low-level entry point. They dispatch to a
/// `#[target_feature(enable = ...)]` function, so running one whose feature
/// this CPU lacks is undefined behaviour (`SIGILL` in practice) ...
///
/// The safe entry points — [`Argon2`], [`detected_backend`], `blake2b`,
/// `blake2b_long`, and `fill_memory_blocks` — never let a caller name the
/// backend. They take it from the corresponding cached runtime cascade, which
/// by construction only ever names a backend this CPU advertises. That is the
/// whole reason they can be safe ...
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __internal {
    pub use crate::base64::{Base64Backend, base64_backend, detect_base64_backend};
    pub use crate::blake2b::{ BLOCKBYTES, Blake2b, Blake2bBackend, IV, KEYBYTES, OUTBYTES,
        PERSONALBYTES, SALTBYTES, blake2b, blake2b_backend, blake2b_long,
        blake2b_long_with_backend, blake2b_with_backend, detect_blake2b_backend };
    pub use crate::block::{Block, Instance, Position};
    pub use crate::core::{ PassTrace, constant_time_eq, fill_first_blocks, fill_memory_blocks,
        fill_memory_blocks_traced, finalize, hash_traced, hash_with_backend, index_alpha,
        initial_hash };
    pub use crate::encoding::{ ... to_base64_with_backend, ... };
    pub use crate::fill_block::{Backend, FillSegmentFn, backend, detect, fill_segment_fn};
    #[cfg(feature = "std")]
    pub use crate::memory::audit;
    pub use crate::memory::{ ARENA_ALIGN, Arena, ArenaGuard, Workspace, clear_internal_memory,
        ... secure_wipe, ... };
    pub use crate::params::validate_inputs;

    #[cfg(feature = "bump-alloc")]
    pub use ::bumpalo;

    /// Each backend's `fill_segment`, reachable directly so a differential test
    /// can pit two backends against each other on the same arena.
    pub mod backends {
        pub use crate::fill_block::scalar;
        #[cfg(target_arch = "aarch64")]
        pub use crate::fill_block::neon;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub use crate::fill_block::sse2;
        #[cfg(target_arch = "x86_64")]
        pub use crate::fill_block::avx2;
        #[cfg(target_arch = "x86_64")]
        pub use crate::fill_block::avx512;
    }
}
```
So: backend-forcing hooks reach tests via (a) `unsafe fn hash_with_backend` / `fill_memory_blocks_traced` / `*_with_backend`, (b) `fill_segment_fn(Backend) -> FillSegmentFn`, and (c) the `__internal::backends` module exposing each ISA module's `fill_segment` directly, arch-gated.

---

## 4. `src/fill_block/` — THE CORE PATTERN

### 4.1 `mod.rs` — dispatch (565 lines, read in full)

`/Users/brooklyn/workspace/github/argon2-rust/src/fill_block/mod.rs`

#### Module docs — cost model and the `#[target_feature]` placement rule (lines 1-47)
```
//! * Detection runs at most once per process. The result is cached in a
//!   [`AtomicU8`] read with [`Ordering::Relaxed`]. The initialisation race is
//!   benign: every thread computes the same answer, so a duplicated `detect()`
//!   only wastes a `cpuid`.
//! * A hash call resolves the function pointer **once**, before entering the
//!   pass/slice/lane loops (see `core::fill_memory_blocks`). Nothing detects or
//!   dispatches inside the per-block loop.
//! * Per hash: one relaxed load plus one compare. Per segment: one indirect
//!   call. A segment is `segment_length` blocks — thousands at any realistic
//!   `m_cost` — so the per-block overhead is nil.
...
//! # Why `#[target_feature]` sits on `fill_segment`
//!
//! LLVM will not inline a callee with a higher target-feature set into a caller
//! with a lower one. Putting the attribute on the whole `fill_segment` lets
//! `fill_block` inline into it, which is what preserves the `src/opt.c`
//! optimisation of keeping the 1 KiB `state` in registers across loop
//! iterations ... Do **not** move the attribute down onto `fill_block` or onto
//! individual intrinsics.
```

#### Module cfg gating (lines 51-70)
```rust
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

// WebAssembly has no runtime feature detection a module can survive (SIMD
// instructions fail validation on engines that lack them), so the module
// exists exactly when the engine contract was given at compile time.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm128;
```
Modules are `pub` (not `pub(crate)`) so that `__internal::backends` can re-export them; they're unreachable without `internal-api`.

#### The `Backend` enum (lines 82-97)
```rust
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum Backend {
    /// Portable scalar code. Always available.
    Scalar = 0,
    /// AArch64 NEON.
    Neon = 1,
    /// x86 / x86-64 SSE2.
    Sse2 = 2,
    /// x86-64 AVX2.
    Avx2 = 3,
    /// x86-64 AVX-512F.
    Avx512 = 4,
    /// wasm32 fixed-width SIMD128. Compile-time selected; see `wasm128`.
    Wasm128 = 5,
}
```
- `#[non_exhaustive]` + `Backend::ALL: &'static [Backend]` (lines 115-121) so adding a variant is not breaking.
- `name() -> &'static str` lowercase names for bench ids (lines 125-135).
- `is_available()` (lines 142-152) — per-backend runtime probe, for tests/benches to loop `Backend::ALL` and skip.
- `to_u8` / `from_u8` (lines 153-170): `from_u8` is a *total* inverse — "anything unknown maps to `Backend::Scalar` so the cache can never produce a panic".
- `Display` via `name()`.

#### The function-pointer type and its contract (lines 99-110)
```rust
/// The signature every backend's `fill_segment` has.
///
/// # Safety
///
/// Calling one of these requires:
///
/// * the CPU to support the backend's instruction set — check
///   [`Backend::is_available`], or get the pointer from [`backend`];
/// * `instance`'s arena pointer to be valid for `instance.memory_len()` blocks;
/// * `position` to be in range, and no other thread to be writing the segment
///   `(position.lane, position.slice)` at the same time.
pub type FillSegmentFn = unsafe fn(&Instance, Position);
```

#### Per-feature probes — defined twice under mutually exclusive cfgs (lines 184-258)
```rust
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
```
Same shape for `have_avx2`, `have_sse2` (`any(x86, x86_64)`), `have_neon`, `have_wasm_simd128`. Two special cases:

- NEON under `std` on aarch64 (lines 226-239): compile-time `true` on `target_vendor = "apple"` or `target_os = "windows"` (NEON is in their guaranteed baseline), real `is_aarch64_feature_detected!("neon")` elsewhere.
- wasm SIMD128 (lines 241-253): purely compile-time both ways — "There is no runtime probe a wasm module can survive".

#### The aarch64 runtime shootout (lines 262-378)
`neon_wins_here()` exists because "has NEON" ≠ "NEON is fastest" on unknown microarchitectures (Neoverse N1: NEON fill is *slower* than scalar, measured 467 vs 331 ns/block). Compiled only under `all(feature = "std", target_arch = "aarch64", not(miri), not(debug_assertions), not(apple/windows))`; it fills a 1 MiB single-lane instance with both backends, finely interleaved, best of 6 reps, ~4 ms once per process. Everywhere else a second definition returns `true`. This is the pattern for "detection that cpuid cannot answer".

#### detect / cache / resolve (lines 383-487) — THE runtime-dispatch pattern
```rust
/// Sentinel meaning "detection has not run yet". Not a valid [`Backend`] value.
const UNINIT: u8 = 0xFF;

/// Cached [`Backend`] as a `u8`, or [`UNINIT`].
static CACHED_BACKEND: AtomicU8 = AtomicU8::new(UNINIT);

#[must_use]
pub fn detect() -> Backend {
    if cfg!(miri) {
        // Miri interprets the crate: its intrinsic support stops around
        // SSE2 ... Scalar is the one backend that behaves identically
        // on every host Miri runs on ...
        Backend::Scalar
    } else if have_avx512f() {
        Backend::Avx512
    } else if have_avx2() {
        Backend::Avx2
    } else if have_sse2() {
        Backend::Sse2
    } else if have_neon() && neon_wins_here() {
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

/// The `fill_segment` implementation for `backend`.
///
/// Resolve this **once per hash call**, outside every loop. ...
/// It does **not** check availability: a pointer for a backend this CPU lacks
/// will fault when called. Use [`Backend::is_available`] first, or take the
/// value from [`backend`].
#[must_use]
pub fn fill_segment_fn(backend: Backend) -> FillSegmentFn {
    match backend {
        Backend::Scalar => scalar::fill_segment,

        #[cfg(target_arch = "aarch64")]
        Backend::Neon => neon::fill_segment,
        #[cfg(not(target_arch = "aarch64"))]
        Backend::Neon => scalar::fill_segment,

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => sse2::fill_segment,
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::Sse2 => scalar::fill_segment,

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => avx2::fill_segment,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx2 => scalar::fill_segment,

        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => avx512::fill_segment,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 => scalar::fill_segment,

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Backend::Wasm128 => wasm128::fill_segment,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        Backend::Wasm128 => scalar::fill_segment,
    }
}
```
Key properties to replicate:
1. `AtomicU8` + sentinel `0xFF`, `Relaxed` load/store, benign init race (every thread computes the same answer). Not `OnceLock` (needs `std`).
2. `#[cold] #[inline(never)]` on the populate path so the hot accessor stays tiny.
3. `detect()` re-runs detection (uncached) for tests; `backend()` is the cached value; `cfg!(miri)` short-circuits to Scalar inside `detect()`.
4. Off-arch match arms fall back to `scalar::fill_segment` so tests can iterate `Backend::ALL` on any host; the resolver never checks availability — that's the caller's unsafe obligation.

#### mod.rs unit tests (lines 489-564)
`backend_u8_round_trip` (sentinel and garbage map to Scalar), `cache_agrees_with_detect`, `detected_backend_is_available`, `detection_respects_the_architecture` (per-arch truth table, including wasm `simd128` on/off), `every_backend_resolves_to_a_function` (`core::ptr::fn_addr_eq`).

### 4.2 Backend module anatomy — `scalar.rs` (1081 lines)

`/Users/brooklyn/workspace/github/argon2-rust/src/fill_block/scalar.rs`

- Module header names the exact C files and functions it ports: `ref.c` (`fill_block`, `next_addresses`, `fill_segment`) and `blamka-round-ref.h` (`fBlaMka`, `G`, `BLAKE2_ROUND_NOMSG`). "the index expressions are the C's index expressions".
- Every C-level fact is spelled out: unsigned wraparound → `wrapping_*` "otherwise the function panics in debug builds" (line 35-38); C snippets quoted in doc comments.
- Pure safe Rust for the primitive layer: `pub const fn f_blamka`, `pub const fn g_values`, `pub fn blake2_round_nomsg` on `[u64; 16]`, `#[inline(always)]`.
- `pub fn fill_block(prev: &Block, ref_: &Block, next: &mut Block, with_xor: bool)` — safe, references only; the aliasing case (`next_addresses` needs ref == next) is handled by copying the block first since `Block` is `Copy` (lines 138-145, 211-218).
- The one `pub unsafe fn fill_segment(instance: &Instance, position: Position)` (line 233) with a `# Safety` section restating `FillSegmentFn`'s contract and the concrete invariants that keep offsets in bounds. It guards degenerate instances (`lane_length == 0 || lanes == 0`) instead of the C's NULL check, because "this library must not panic".
- Extensive `#[cfg(test)] mod tests`: expected values were produced by compiling the untouched C into a harness (the exact `cc` command is in a comment, lines 370-379), plus property tests (all-zero stays zero, symmetry, xor-fold identity, row-round diffusion) and a 60-case `fill_segment_matches_the_c_reference_across_the_grid` table driving the whole pass/slice/lane grid and comparing an arena digest.

### 4.3 Backend module anatomy — `avx2.rs` (1309 lines)

`/Users/brooklyn/workspace/github/argon2-rust/src/fill_block/avx2.rs`

Structure (representative of all SIMD backends):
1. **Long module doc**: provenance with C file/line citations (`opt.c:94-97`, `blamka-round-opt.h:186`), layout explanation, the traps (macro argument orders), the `#[target_feature]` placement rule repeated, and a "Testing this backend on aarch64-apple-darwin — measured, not assumed" section with probe output.
2. `use core::arch::x86_64::*;` — **raw intrinsics**, no wrapper crate. Shuffle tables are plain `const R24: [u8; 32]` byte arrays loaded via `_mm256_loadu_si256` (lines 116-128) so lane order is unambiguous.
3. Small `#[inline(always)] unsafe fn` helpers (`rotr32/24/16/63`, `muladd`, `g1`, `g2`, `diagonalize_1/2`, `undiagonalize_1/2`, `blake2_round_1/2`), each with a one-line `// SAFETY:` comment naming the intrinsics' feature (lines 131-477). The C macros become fns taking `&mut __m256i` or by-value with `[__m256i; 8]` returns.
4. `fill_block` (lines 505-616): `#[inline(always)] unsafe fn fill_block(state: &mut [__m256i; HWORDS_IN_BLOCK], ref_block: *const Block, next_block: *mut Block, with_xor: bool)` — **raw pointers**, explicitly because `next_addresses` calls it with `ref == next` and that must not form a `&`/`&mut` pair. `block_xy` is `[MaybeUninit<__m256i>; 32]` with a *measured* justification (an initialised array costs a real 1 KiB memset per block; instruction counts quoted, lines 514-534).
5. `next_addresses` (lines 636-667): same raw-pointer shape.
6. `fill_segment_impl` (lines 679-833): `#[inline(always)] unsafe fn`, and the attribute is documented as load-bearing: "it is what puts this body ... inside [`fill_segment`], which declares `target_feature(enable = "avx2")`" (line 670-674). Body mirrors `opt.c:174-283` line by line with `opt.c:` citations, `wrapping_*` arithmetic, and SAFETY comments on every raw-pointer block access.
7. The entry point (lines 838-853):
```rust
/// AVX2 `fill_segment()`.
///
/// # Safety
///
/// The CPU must support AVX2 — check
/// [`crate::fill_block::Backend::is_available`] or take the pointer from
/// [`crate::fill_block::backend`]. All the requirements of
/// [`crate::fill_block::FillSegmentFn`] apply.
#[target_feature(enable = "avx2")]
pub unsafe fn fill_segment(instance: &Instance, position: Position) {
    // SAFETY: AVX2 is this function's declared feature, so a caller reaching
    // here without it is already unsound; the rest is `FillSegmentFn`'s
    // contract, which the caller upholds.
    unsafe { fill_segment_impl(instance, position) }
}
```
8. **Tests** (lines 855+): share `super::super::sse2::test_support::{...}` helpers (the shared x86 test-support module lives in sse2.rs because its cfg is the widest, sse2.rs:946-960). Key devices:
   - `skip_unless_available(Backend::Avx2)` — tests silently skip when the host can't run the ISA;
   - `const FORCE_UNDETECTED_AVX2: bool = cfg!(argon2_force_avx2);` (line 919) — an **out-of-band cfg, never a Cargo feature**, documented as a deliberate test-only detection bypass for Rosetta (which executes AVX2 but hides it from cpuid). Invocation: `RUSTFLAGS="--cfg argon2_force_avx2" cargo test --target x86_64-apple-darwin --features internal-api avx2`. The module has `#![allow(unexpected_cfgs)]` in its test mod (line 860).
   - `ARGON2_REQUIRE_BACKEND=avx2` env var turns skips into failures so a CI gate can prove the tests ran (referenced in avx2.rs:927, implemented in sse2.rs:1296).
   - Equivalence: `fill_block_matches_scalar_over_2048_triples` — a `#[target_feature(enable = "avx2")] unsafe fn fill_block_blocks` adapter that runs one block through the AVX2 kernel and compares against `scalar::fill_block` over a XorShift64Star-driven corpus; plus `check_official_vectors`, aliasing tests, all-zero tests.

### 4.4 `avx512.rs` (1788 lines)

Same anatomy with two extras:
- `#[target_feature(enable = "avx512f")] pub unsafe fn fill_segment` (line 1071-1072).
- A **"Verification status — measured, not assumed"** table in the module docs (lines 61-114): the backend is compile-verified only on the dev host (Rosetta SIGILLs on AVX-512, proven with a disassembly-checked probe); every test checks `is_available()` first; `ARGON2_REQUIRE_BACKEND=avx512` converts skips to failures; lane-permutation unit tests are written in plain integer arithmetic so they *do* run on the host. This is the pattern for "a backend you cannot execute locally".

### 4.5 `sse2.rs` (1922 lines) — sub-ISA runtime dispatch inside one backend

- Genuine SSE2-only implementation *and* an SSSE3 spelling of the same rounds, selected by **a second, module-local cached probe**: `static SSSE3_CACHE: AtomicU8` with sentinel `0xFF`, `#[cold] #[inline(never)] probe_and_cache_ssse3`, `have_ssse3()` (lines 104-152). Without `std` the probe degrades to `cfg!(target_feature = "ssse3")`.
- The split is a `const SSSE3: bool` generic on `#[inline(always)]` helpers so each entry point gets a fully-inlined specialized copy (module docs lines 56-59).
- Three entry points (lines 894-940):
```rust
#[target_feature(enable = "sse2")]
pub unsafe fn fill_segment(instance: &Instance, position: Position) {
    if have_ssse3() {
        unsafe { fill_segment_ssse3(instance, position) }
    } else {
        unsafe { fill_segment_sse2_only(instance, position) }
    }
}

#[target_feature(enable = "sse2")]
pub unsafe fn fill_segment_sse2_only(...) { unsafe { fill_segment_impl::<false>(...) } }

#[target_feature(enable = "sse2,ssse3")]
pub unsafe fn fill_segment_ssse3(...) { unsafe { fill_segment_impl::<true>(...) } }
```
The dispatch cost is one relaxed load + one branch *per segment*, verified by disassembly (24 instructions for the trampoline vs 2122 for the body — the docs quote the counts and the exact `cargo rustc --release --emit asm` command used to check them, lines 61-78).
- `#[cfg(test)] pub(crate) mod test_support` (line 952) — shared harness (`assert_fill_block_matches_scalar`, `check_official_vectors`, `skip_unless_available`, XorShift64Star) used by sse2/avx2/avx512 tests.

### 4.6 `neon.rs` (3659 lines)

- Documents that there is **no NEON code in the C reference** — the backend is derived from the SSE2/SSSE3 shape, with an intrinsic-correspondence table in the module docs (lines 16-31) and the NEON-specific `UZP1`+`UMULL`/`UMULL2` pairing trick for `fBlaMka`.
- Tuning knobs are exposed as extra `#[doc(hidden)]` entry points for tests/benches only:
```rust
#[target_feature(enable = "neon")]
pub unsafe fn fill_segment(instance: &Instance, position: Position) { ... }

#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
#[target_feature(enable = "neon")]
pub unsafe fn fill_segment_variant<const TBL: bool, const ML: u8, const X32: bool, const V: u8>(...)
```
plus `fill_block_isolated` for differential testing against scalar. Note the pattern: const-generic tuning parameters on the `#[inline(always)]` impl, public entry point instantiates the measured winners.

### 4.7 `wasm128.rs` (507 lines)

- Exists only under `cfg(all(target_arch = "wasm32", target_feature = "simd128"))` — selection is compile-time because an engine without SIMD128 fails module validation.
- `#![allow(unsafe_op_in_unsafe_fn)]` at module level (line 36) — deliberate per-module relaxation.
- `macro_rules! g1/g2/round8` instead of fns (wasm intrinsics are safe in core::arch::wasm32, so the macro layer keeps the code shape close to the C macros); `#[target_feature(enable = "simd128")]` on `fill_block`, `next_addresses`, `pub unsafe fn fill_segment` (lines 200, 259, 281).

### 4.8 Safety invariants documented (cross-cutting)

- `FillSegmentFn` (mod.rs:99-110): CPU feature + arena validity + position in range + no concurrent writer to the same segment.
- `Instance` (`src/block.rs:210-260`): borrows the arena as `*mut Block` so the signature can be `unsafe fn(&Instance, Position)` shared across lanes; `unsafe fn new(memory, memory_len, ...)` contract is `memory_len == memory_blocks as usize`; `unsafe fn block/block_mut/block_ptr` with per-method contracts; deliberately not Send/Sync ("sharing it across lanes is `core`'s job, behind its own newtype"). Aliasing note: `prev`/`ref` may coincide, `curr` never does; `next_addresses` copies instead of aliasing.
- Every backend `fill_segment` restates: "The CPU must support X — check `Backend::is_available` or take the pointer from `backend()`".
- `unsafe` blocks carry `// SAFETY:` comments (`#![warn(clippy::undocumented_unsafe_blocks)]` enforces).

---

## 5. `src/blake2b.rs` + `src/blake2b/{avx2,avx512,sse41}.rs`

`/Users/brooklyn/workspace/github/argon2-rust/src/blake2b.rs` (1404 lines) — same dispatch pattern, with notable variations:

- **Enum with cfg'd variants** (lines 158-169): `Blake2bBackend::{Scalar, Sse41, Avx2, Avx512}` where the three SIMD variants exist only under `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`. `ALL` is defined twice (per-arch). `from_u8` is cfg'd likewise. So the enum's *shape* is platform-dependent here (unlike `Backend` which is uniform).
- Probes `have_sse41/have_avx2/have_avx512vl` follow the same twice-defined std/cfg pattern (lines 230-331). AVX-512 requires `avx2 && avx512f && avx512vl`.
- **Rosetta carve-out** (lines 243-299): `prefer_sse41()` on `x86_64-apple-darwin` calls `sysctlbyname("sysctl.proc_translated")` (hand-declared extern) and declines to *auto-select* SSE4.1 under translation (measured 2× slower than scalar), while keeping the backend executable for forced tests.
- Same cache: `static DETECTED_BACKEND: AtomicU8` (x86 only), `detect_blake2b_backend()` (Miri → Scalar via `#[cfg(miri)] return`), `detect_and_cache_blake2b_backend` `#[cold]`, `blake2b_backend()`; on non-x86 `blake2b_backend()` is a `const fn` returning Scalar (lines 393-399).
- **Zero-cost non-x86 fallback** (lines 401-436): on x86 `type CompressFn = unsafe fn(...)`; elsewhere a zero-sized `struct ScalarCompress`, "so ARM must not pay an indirect call for an x86-only optimization".
- `fn compress_fn(backend) -> CompressFn` maps backends to `sse41::compress` / `avx2::compress` / `avx512::compress`.
- Backend modules: `sse41.rs` has `#[target_feature(enable = "sse4.1")] pub(super) unsafe fn compress(h, t, f, words)` (line 269); `avx2.rs` the same with `"avx2"` (line 140); `avx512.rs` declares only `"avx2"` and uses an **inline-asm `vprorq`** for the rotate, with the SAFETY comment stating the dispatcher requires AVX-512F+VL before exposing the pointer (lines 73-101) — a documented, deliberate pattern for using an instruction rustc can't name in the target_feature list.
- Unsafe API: `Blake2b::new_with_backend`, `blake2b_with_backend`, `blake2b_long_with_backend` are all `pub unsafe fn` with "`backend` must be executable on the current CPU, as reported by `Blake2bBackend::is_available`" (lines 526, 748, 792). Safe one-shots `blake2b`/`blake2b_long` and `Blake2b::new` use the cached cascade.
- `Blake2b` state wipes itself on Drop (`clear_internal_memory`); `finalize` takes `self` by value so state reuse is a compile error (the C's reuse guards are kept anyway for statement parity).

## 6. `src/base64.rs` + `src/base64/{x86,neon,wasm128}.rs`

`/Users/brooklyn/workspace/github/argon2-rust/src/base64.rs` (380 lines):

- `Base64Backend::{Scalar, Neon, Ssse3, Avx2, Wasm128}` — `#[repr(u8)]`, per-arch `ALL` consts (aarch64 / x86 / wasm128 / fallback), `name()`, `is_available()` where `Avx2` requires `have_avx2() && have_ssse3()` because the AVX2 kernel falls through to the SSSE3 kernel for the tail (lines 88-99).
- Identical probe + `AtomicU8` + sentinel + `detect_base64_backend()`/`base64_backend()` cache (lines 170-209). Miri → Scalar.
- Different dispatch *shape*: instead of a function-pointer table, two `#[inline] pub unsafe fn encode_prefix/decode_prefix(backend, ...)` (lines 259-359) `match` on the backend and call into `neon::encode`, `x86::encode_ssse3`, `x86::encode_avx2`, `wasm128::encode` — off-arch arms return `(0, 0)`. SIMD handles whole vector blocks; the scalar loop always handles tails and exact error positioning.
- Per-arch `MIN_ENCODE_LEN`/`MIN_DECODE_LEN` consts keep dispatch cost off short inputs; `usize::MAX` where no SIMD exists (lines 211-245).
- Backend entry points: `base64/x86.rs` has `#[target_feature(enable = "ssse3")]` and `#[target_feature(enable = "avx2,ssse3")]` fns (lines 59, 129, 245, 366); `neon.rs` uses `"neon"`; `wasm128.rs` compiles only under the simd128 contract.

---

## 7. Core modules

### 7.1 `src/core.rs` (4757 lines) — API design
- `pub struct Argon2 { algorithm, version, params }` — `Copy`, `const fn new` cannot fail because `Params` is pre-validated (lines 1118-1133). Rich method matrix: `hash_into`/`hash`/`hash_encoded` (+`_with_ad`), `verify`/`verify_encoded`/`verify_encoded_bounded` (+`_with_ad`), `hash_password*` convenience spellings (`RANDOM_SALT_LEN = 16`, `BOUNDED_MAX_SALT_LEN = 1024`). `verify_encoded_bounded` takes a ceiling `Params` to reject hostile PHC costs before allocating (DoS guard).
- `Hasher` (line 1974): pooled arena reuse via `Workspace`; `Send` but deliberately not `Sync` with a **`compile_fail` doctest** proving it (core.rs:1944-1950); `reserve()/reserved_blocks()/clear()`; every method mirrors `Argon2`.
- Free functions: `initial_hash`, `fill_first_blocks`, `finalize`, `constant_time_eq`, and `fill_memory_blocks(instance)` — the safe dispatch point:
```rust
pub fn fill_memory_blocks(instance: &Instance) -> Result<(), Error> {
    // SAFETY: `backend()` is the cached result of the `is_*_feature_detected!`
    // cascade in `fill_block::detect`, so this CPU can execute it. ...
    unsafe { fill_memory_blocks_traced(instance, crate::fill_block::backend(), None) }
}
```
- `pub unsafe fn fill_memory_blocks_traced(instance, backend, trace)` (line 439): resolves `let fill = fill_segment_fn(backend);` **once before every loop**, runs `std::thread::scope` pooled fill when `parallel && threads > 1 && lanes > 1` (one scope for the whole fill, barrier per slice), otherwise the Miri-checkable single-threaded `fill_slice_st`. `trace: Option<PassTrace>` is the genkat `internal_kat` hook (`pub type PassTrace<'a> = &'a mut dyn FnMut(u32, &[Block])`).
- `pub unsafe fn hash_with_backend(...)` (line 3015, gated `internal-api`): the docs contain the **paired doctest / compile_fail doctest** pattern (lines 2963-3004):
```rust
/// Guarded, and therefore fine:
/// ```
/// for &backend in Backend::ALL {
///     if !backend.is_available() { continue; }
///     // SAFETY: `is_available()` just said this CPU can execute `backend`.
///     unsafe { hash_with_backend(...).unwrap(); }
/// }
/// ```
/// The **same snippet with the `unsafe` block deleted** must not compile ...
/// ```compile_fail
/// for &backend in Backend::ALL {
///     if !backend.is_available() { continue; }
///     hash_with_backend(...).unwrap();   // ERROR: unsafe fn
/// }
/// ```
```
"Keep these two in sync — the pair is the regression test, and the runnable one above is what proves the failing one fails for the right reason." The other `compile_fail` doctests prove `Hasher`/`Workspace` are not `Sync` (memory.rs:1143-1147, core.rs:1944).
- `index_alpha` (line 98) documents the two classic port traps (deliberate wrapping subtraction at `index == 0`; exact integer widths) with C citations.

### 7.2 `src/error.rs` (319 lines) — C-parity error codes
- `#[repr(i32)] #[non_exhaustive] pub enum Error` — **discriminants are the C codes**: `OutputPtrNull = -1` … `VerifyMismatch = -35`; crate-specific codes sit below `MIN_C_CODE` (`OsRandom = -100`) so they can never collide with a future C code. `ARGON2_OK` is absent (success is `Ok(())`).
- `pub const MIN_C_CODE: i32 = -35; MAX_C_CODE: i32 = -1;`
- `as_c_code()` / `from_c_code()` (total inverse; `None` for 0 and unknowns), `message()` byte-identical to `argon2_error_message()`. Unit tests round-trip every code in range and pin that crate-specific codes stay below the C range.
- Implements `core::error::Error` (works in no_std), `Display` via `message()`.

### 7.3 `src/params.rs` (1480 lines)
- All limit consts mirrored from `argon2.h`/`core.h` (`MIN_LANES`…`MAX_SECRET`, `BLOCK_SIZE`, `QWORDS_IN_BLOCK`, `ADDRESSES_IN_BLOCK`, …).
- **Typed units**: `pub struct Memory(u64)` with `const fn kib/mib/gib` (saturating, never panic, validate nothing) and `pub struct TagLen(u64)` with `const fn bytes` (lines 103-165).
- `Algorithm` (`#[repr(u32)]`, `Argon2d=0/Argon2i=1/Argon2id=2`, `Default`) and `Version` (`V0x10=0x10`, `V0x13=0x13`) with `as_u32/from_u32/as_str/ALL` consts.
- `pub const fn validate_inputs(out_len, pwd_len, salt_len, secret_len, ad_len, m_cost, t_cost, lanes, threads)` — the C's `validate_inputs()` **in the exact same check order**, documented as load-bearing for differential error-code parity; uses `wrapping_mul` for `8 * lanes` because the C computes it in `uint32_t` before range-checking lanes.
- `Params` (private `u32` fields, always valid), `ParamsBuilder` with by-value `const fn` setters (usable in `const` items), `threads: Option<u32>` where `None` tracks `lanes`, `DEFAULT` = OWASP profile, `build()` checks tag length before memory (the C's order) then delegates to `validate_inputs`; `build_or_panic()` is the single intentional panic in the crate (for const evaluation). Presets: `Params::OWASP`, `RFC9106_HIGH_MEMORY`, `RFC9106_LOW_MEMORY`.

### 7.4 `src/encoding.rs` (2395 lines)
- Ports `encoding.c`: `to_base64`/`from_base64` (caller buffers, `Result<usize>`/`Result<(usize,usize)>`), allocating `encode_base64`/`decode_base64`, `encode_string`/`encode_string_alloc`, `decode_string` (C parity), `decode_phc` (node-argon2 field order tolerance), `Decoded` struct, `b64_len`/`num_len`/`encoded_len` const fns.
- Branch-free character classification mirroring the C's `EQ/GT/GE/LT/LE` macros (side-channel rationale documented).
- A "Known divergences from the C reference" section documents every deliberate difference with measurements (unrepresentable versions, embedded NULs, signed-char bug).

### 7.5 `src/memory.rs` (2211 lines) — arena, mmap, zeroize
- `pub const ARENA_ALIGN: usize = 64` (cache-line / AVX-512).
- **Linux mmap path without libc**: `#[cfg(all(feature = "std", target_os = "linux"))] mod os` hand-declares `mmap/munmap/madvise/sysconf` externs (lines 380-496) — "not a crate dependency: std already links the system libc". `map_aligned` over-maps by 2 MiB, trims head/tail with `munmap`; `MADV_HUGEPAGE` hint; `MMAP_THRESHOLD = 2 MiB`. `MAP_ANONYMOUS` zero-fill is what makes the arena "initialised **and** zero with no user-space write" — the docs explain why safe `&[Block]` accessors require a mapping rather than `alloc` + a promise.
- `Arena` (line 677): `NonNull<Block>`, visible `len` vs `capacity`, a `zeroed: bool` conservative invariant ("being wrong in the `true` direction would skip a security wipe"), `workers` for the threaded wipe, `Backing::{Heap, Mapped{base,len,huge}}`. `unsafe impl Send`, deliberately not `Sync`. `Drop` wipes (feature-gated) then frees/unmaps.
- `secure_wipe_raw` (line 253): `write_bytes` + empty `asm!` block taking the pointer as input (the glibc `explicit_bzero` construction, *defined* to read/write arbitrary memory) + `compiler_fence`; volatile-loop fallback where `asm!` is unavailable (`HAVE_ASM_BARRIER` const incl. `not(miri)`). Threaded wipe above `WIPE_THREAD_THRESHOLD = 64 MiB`. `secure_wipe*` always wipe; `clear_internal_memory*` compile to nothing without the `zeroize-memory` feature (C's `FLAG_clear_internal_memory`).
- `Workspace` (line 1184): parks one `Arena` between hashes; `acquire() -> ArenaGuard` (Drop returns the arena, **wipe on release, never on acquire**), `reserve`, `release`, `clear`; optional reused `bumpalo::Bump` under `bump-alloc`. Send-but-not-Sync with the compile_fail doctest.
- `#[cfg(all(feature = "internal-api", feature = "std"))] pub mod audit` (line 1036): thread-local release observation hook (`watch/released/released_dirty/is_dirty`) called from `Arena::drop`, existing *because* the mmap backing is invisible to a `GlobalAlloc` spy.

### 7.6 `src/random.rs` (421 lines) — zero-dependency OS CSPRNG
- `pub(crate) fn os_random(buf) -> Result<(), Error>`, std-only.
- Per-platform hand-declared FFI with a documented decision table (lines 9-29): Linux/Android → raw `syscall(SYS_getrandom)` with a **per-arch allowlist of syscall numbers** (verified by executing under QEMU; x32 excluded because a wrong number "does not fail safe"), fallback `/dev/urandom` on *any* errno; macOS/OpenBSD `getentropy`; iOS/tvOS/watchOS/visionOS `CCRandomGenerateBytes`; Windows `ProcessPrng` (not BCryptGenRandom — registry read + sandbox hang, cited); WASI p1 `random_get`; everything else returns `Error::OsRandom` rather than failing the build ("it took hashing down with it" when `compile_error!` was tried).

---

## 8. Tests (`/tests/`)

All integration tests import `argon2_rust::__internal::*` — available because of the self-dev-dependency. Common invocation line in headers: `cargo test --test <name> --features internal-api --release`.

### 8.1 `kat.rs` (795 lines) — KAT trace replay
- Replays the six golden files in `phc-winner-argon2/kats/` (output of `genkat.c` built with `-DGENKAT`): 9-line header with H0, a full arena dump after every pass, final tag; 12304 lines each, reproduced byte-for-byte including genkat's **trailing space** per line.
- Uses `__internal::hash_traced(backend, ..., trace: PassTrace)`; the trace closure re-formats the arena exactly as `internal_kat` does.
- Runs **once per backend the CPU can execute** (`runnable_backends()` from `Backend::is_available()`); `every_runnable_backend_is_kat_checked` prints and pins the list so it can't silently shrink to `[Scalar]`.
- `a_pooled_hasher_reproduces_every_kat_tag` runs all six files through one reused `Hasher` in rotating orders.
- `forced_avx2_reproduces_every_golden_file` (`#[ignore]`, x86-64 only) pairs with the `argon2_force_avx2` cfg for Rosetta.
- File access: `env!("CARGO_MANIFEST_DIR")` to find `phc-winner-argon2/kats/`.

### 8.2 `differential.rs` (1701 lines) — live differential vs compiled C reference
**Shells out; does not link or dlopen** (module docs lines 4-20 explain why: linking would need build.rs + a feature, "neither file belongs to this test's owner").
1. `make -C phc-winner-argon2 OPTTARGET=none libs` builds `libargon2.a` once per process (skipped if present). `OPTTARGET=none` forces the *scalar* `ref.c` build — "the thing this port must agree with".
2. A `HARNESS_C` string (a stdin/stdout line-protocol driver around `argon2_ctx()`, chosen because it's the entry point that takes `secret`/`ad` and matches `validate_inputs` ordering) is written to `target/differential-harness/harness.c` and compiled with `cc -O2 -I include harness.c libargon2.a -pthread` (`CC` env var respected), with mtime-based rebuild skipping and a `OnceLock` so it happens once per test process.
3. One harness process serves a whole sweep (`OK <taghex>` / `ERR <code>`).
4. Every batch is answered by C once, then replayed against every `Driver`: `PublicApi`, `Pooled` (one `Hasher` across a whole batch — adversarial arena reuse), and `Forced(backend)` for each `Backend::is_available()`. Comparison is the **whole `Result`**: identical tags *or* identical numeric error codes (`Error::as_c_code` vs C's `int`). Batches assert floors on both success and error counts so a degenerate batch fails.
5. Deterministic: SplitMix64 with a constant `SEED`; failure messages carry the full case for reproduction.
6. Path root: `PathBuf::from(env!("CARGO_MANIFEST_DIR"))` (line 289-291); the C tree is expected at `$CARGO_MANIFEST_DIR/phc-winner-argon2` (an independent clone, gitignored, pinned commit in CI).

### 8.3 `vectors.rs` (1722 lines)
- Every `hashtest(...)` call in the C's `src/test.c` transcribed (machine-parsed, each entry carries the `test.c` line number); one `#[test]` per vector; the two 1 GiB `TEST_LARGE_RAM` vectors are `#[ignore]`d exactly like the C. Asserts tag hex, PHC string (only when `version == ARGON2_VERSION_NUMBER`, matching `test.c:55`), `argon2_verify` acceptance both ways, plus extras: `threads_do_not_change_the_tag`, reused-arena replay in descending m_cost order.

### 8.4 `allocation_audit.rs` (1265 lines) — allocator spy
- A `#[global_allocator]` `Spy` forwarding to `System`, with **thread-local** arming (`WATCH_SIZE`, `FREED`, `FREED_DIRTY`, `LIVE_BYTES`, `LIVE_CHUNKS` — `const`-initialized `Cell`s so the allocator can't re-enter). At `dealloc` it reads the region back and counts non-zero bytes — so the secure wipe is verified in `--release` where a non-volatile wipe would be elided.
- Every wipe check has a **control** that fails when the wipe is absent ("a green result cannot come from a test that never looked").
- Leak checks read the thread-local allocation *balance* (exact, byte-for-byte), not RSS.
- Uses `__internal::audit` (the `Arena::drop` hook) for the mmap backing the spy can't see. Miri-scaled parameter consts; this is the suite `cargo miri test --release --test allocation_audit` runs.

### 8.5 `rss_isolation.rs` (292 lines)
- `harness = false` — its own `fn main()` process, because RSS is a process property and libtest runs tests concurrently in one process (the docs describe the deterministic contamination failure that forced the move). Samples `ps -o rss=`, collects failures in a `Report` rather than panicking on first, asserts pooled hashing doesn't grow RSS with tight budgets (4 MiB budget vs 614400 KiB leak signal).

### 8.6 `reuse.rs` (1030 lines)
- Public-API-only verification of arena reuse: the standard is **byte-identity with the one-shot API** across sequences (long fixed run; A,B,A interleave; shrink/regrow; predecessor-cannot-change-the-next-tag). Deterministic SplitMix64. Also documents the `unnameable_types` regression story.

### 8.7 `readme.rs` (134 lines)
- README rust blocks transcribed and run; `transcription_still_matches_the_readme` re-extracts every ```rust block from README.md and asserts each line is present verbatim — drift guard.

---

## 9. Benches (`/benches/`)

### 9.1 Conventions
- All `harness = false`; Criterion via `criterion_group!/criterion_main!` or a hand-rolled `Criterion::default()` in `fn main` (`blake2b.rs`), or fully bespoke `fn main` (`micro.rs`, `base64_shootout.rs`).
- Backend sweeps always: `for &b in Backend::ALL { if !b.is_available() { continue; } ... }` and forced calls are `unsafe { ... }` with `// SAFETY: unavailable backends were skipped above`.
- Dispatch is always resolved/warmed before the measured closure (codspeed.rs:66-70 comment: "Resolve runtime dispatch before CodSpeed enters the measured closure").

### 9.2 `argon2.rs` (1933 lines)
Groups: `grid` (every available backend × m/t/p sweep), `variants`, `parallel`, `reuse`, `reuse_alloc`, `encoded`, `bump` (only with `bump-alloc`), `dispatch` (cost of detection itself), `vs_c`. Env-var controls: `ARGON2_BENCH_SUMMARY=0`, `ARGON2_BENCH_REUSE_REPS`, `ARGON2_BENCH_REUSE_ONLY`, `ARGON2_BENCH_REUSE_BIG`, `ARGON2_BENCH_DIST`, `ARGON2_BENCH_FORCE=avx2` (force a backend past detection; refuses avx512), `ARGON2_BENCH_SUMMARY_BACKEND=avx2` (matched-ISA rows vs C). The module docs carry full measured result tables and an explicit Rosetta caveat ("execution proof, not speed").

**Loading the C reference** (lines 589-640): a private `mod cref` hand-declares
```rust
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;
pub const CANDIDATES: &[&str] = &[
    concat!(env!("CARGO_MANIFEST_DIR"), "/phc-winner-argon2/libargon2.1.dylib\0"),
    concat!(env!("CARGO_MANIFEST_DIR"), "/phc-winner-argon2/libargon2.so.1\0"),
];
```
`argon2_hash` is typed as `unsafe extern "C" fn(t_cost, m_cost, parallelism, pwd, pwdlen, salt, saltlen, out, outlen, ty, version) -> c_int`; the handle is deliberately leaked. No build.rs, no dependency — "dlopen lives in libSystem on macOS and libc on modern glibc, both of which every Rust binary already links". `micro.rs` duplicates this `cref` mod (lines 878-940).

### 9.3 `support/cref_isa.rs` (311 lines) — probing the C build's ISA
Shared via `#[path = "support/cref_isa.rs"] mod cref_isa;` in `argon2.rs` and `micro.rs`. Exists because a bench once printed "(scalar)" next to timings taken against an AVX-512 C build — "a label must never be a guess". It:
1. Parses the ELF64 header to find `.symtab`/`.strtab`, locates the `fill_block` symbol (static, so `.symtab` not `.dynsym`), and scans exactly its bytes (Mach-O: whole-file fallback, labelled as suspect).
2. Counts encoding markers: EVEX prefixes (AVX-512), VEX.L=1 (256-bit), `pshufb` opcode (SSSE3), `pmuludq`/`vpmuludq` (separates hand-written `opt.c` — exactly 32 — from auto-vectorized `ref.c`).
3. Cross-checks which of `src/opt.o`/`src/ref.o` exists next to the library.
4. `CrefIsa::describe()` prints the raw counts alongside the classification so a reader can verify it.

### 9.4 `micro.rs` (1283 lines)
Seconds-scale iteration harness: `--backend/--alg/--m/--t/--p/--r/--w` args, modes `fill` (default), `hash`, `seg`, `decompose` (least-squares split of fixed vs per-pass cost — "start with decompose"), `stages`, `memory`; `--vs-c` interleaves `argon2_hash` from the dlopened C rep-by-rep and prints ratios with the `cref_isa` header line; `--sweep` runs a 12-cell matrix. Drives `__internal::{fill_segment_fn, fill_memory_blocks_traced, hash_with_backend, initial_hash, fill_first_blocks, finalize, Arena, Instance, Position}`. Prints `ns/block med/min/max/spread` with the spread as the honesty check.

### 9.5 `codspeed.rs` (293 lines)
The CI regression net: `codspeed-criterion-compat`, deliberately small, hermetic, single-threaded (simulation instrument; the crate's spin barrier would measure scheduling artifacts). Public API only (`hash_into` per algorithm, PHC codec via `__internal::{decode_string, encode_string_alloc}`).

### 9.6 `blake2b.rs` (109 lines) / `base64.rs` (59 lines)
Criterion groups over the three shapes the algorithm actually uses (`blake2b/short_72_to_64`, `expand_72_to_1024`, `finalize_1024_to_32`; base64 encode/decode at lengths `[8, 16, 32, 256, 4096]` — production sizes included so a SIMD backend "cannot look good only on bulk data it never sees"). Both force each backend with `*_with_backend` after `is_available()`.

### 9.7 `base64_shootout.rs` (283 lines)
Dependency-free paired scalar/SIMD timing (no Criterion) so the identical binary runs natively, under `wasmtime` (`CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime RUSTFLAGS='-C target-feature=+simd128'`), and in minimal containers. Reports `auto` (production cached dispatch) plus each named backend; every backend is verified against scalar before being timed.

---

## 10. `fuzz/`

- Standalone workspace (`[workspace]` in `fuzz/Cargo.toml` — "this crate must not leak into the library's builds"), `package.metadata.cargo-fuzz = true`, dep `libfuzzer-sys = "0.4"`, path dep on `..`. Two `[[bin]]` targets with `test = false, doc = false`.
- `fuzz_targets/phc_decode.rs` (40 lines): fuzzes `Argon2::verify_encoded` over all three algorithms; property is "never panics". Has a cheap `looks_expensive` pre-filter rejecting strings with `m=` > 8192 so a valid-and-large string doesn't cost a full hash per iteration.
- `fuzz_targets/tiny_hash.rs` (60 lines): derives `Params` from fuzzer bytes (lanes 1..=4, threads 1..=8, m up to ~64 KiB, t 1..=3, outlen 4..=64, algorithm, version) and runs the whole pipeline; property: accepted params hash without panic, rejected fail cleanly. Uses fallible `build()` deliberately ("never panics" rules out `build_or_panic`).
- `corpus/` and `artifacts/` are checked in per target.

---

## 11. CI (`.github/workflows/`)

### `ci.yml` (404 lines)
- Env: `PHC_COMMIT: f57e61e...` — the vendored C reference is an **independent git clone** (gitignored, not a submodule), pinned to a commit; cloned in the jobs that need it.
- `test`: matrix `ubuntu-latest`, `ubuntu-24.04-arm` (the only NEON + Linux-aarch64-mmap coverage), `macos-latest`. Steps: clone phc-winner-argon2, then `cargo test --release`, `cargo test` (debug), `cargo test --release --no-default-features`, `cargo test --release --all-features`.
- `windows`: `x86_64-pc-windows-msvc` and `i686-pc-windows-msvc` (32-bit catches `extern "system"` vs `extern "C"` decoration bugs; run under WOW64). Build + `--lib` tests + `--test vectors` (the one that calls ProcessPrng).
- `no-std`: `cargo build --release --no-default-features --target thumbv7em-none-eabi`.
- `docs`: `cargo doc --no-deps` and `--all-features` with `RUSTDOCFLAGS: -D warnings`.
- `msrv`: reads `rust-version` out of `Cargo.toml` with `sed` (single source of truth), installs that exact toolchain via `dtolnay/rust-toolchain@master`, on **both** `ubuntu-latest` (x86_64 — required because avx512.rs is cfg'd out elsewhere) and `ubuntu-24.04-arm` (NEON). `cargo build --release --all-features` (real codegen, not just check) + `cargo check --no-default-features` (dev-deps are not part of the MSRV promise). No `--ignore-rust-version`.
- `entropy-target-matrix`: `cargo check --release` for `wasm32-unknown-unknown`, `wasm32-wasip1`, `wasm32-wasip2` (random.rs must never make a target uncompilable).
- `old-glibc`: docker `manylinux2014_x86_64` — builds and *runs* `cargo test --release --test vectors` to prove the raw-syscall entropy path links on glibc 2.17.
- `wasm`: matrix over wasip1+simd128, wasip1-scalar, wasip1-threads (wasmtime pinned `@38.0.4`); `wasmtime run --dir=/` runner; `RUSTFLAGS: -C target-feature=±simd128`; runs `--lib` + kat/vectors/reuse/allocation_audit suites.
- `benches-compile`: `cargo build --release --benches --features internal-api`.
- `miri`: nightly + miri on ubuntu + macos, `cargo miri test --release --test allocation_audit` only (scoped because it has MIRI-scaled sizes; `detect()` returns Scalar under Miri making the matrix meaningful).
- `asan`: nightly `-Zsanitizer=address` on x86_64-linux and aarch64-macOS, full `--lib --tests` suite including the live-C differential (doctests excluded — rustdoc doesn't link the sanitizer runtime).
- `fuzz`: `cargo-fuzz` on ubuntu (explicit `--target x86_64-unknown-linux-gnu`) and macos (`aarch64-apple-darwin`), 5 min per target.
- `done`: aggregate gate with `if: always()` and explicit failure propagation — the single required status check for branch protection.

### `codspeed.yml`
CodSpeed on PRs: `cargo codspeed build/run --bench codspeed`, `CodSpeedHQ/action@v5` with `mode: simulation`, OIDC (`id-token: write`, no token).

### `release.yml`
release-plz: release PR from conventional commits on every main push; publish on merge via crates.io **Trusted Publishing** (OIDC; repository/workflow/`environment: release` must all match; crate is `trustpub_only`).

---

## 12. `docs/superpowers/` process conventions

- `plans/2026-08-10-params-api.md`: date-prefixed implementation plans consumed by agentic workers ("REQUIRED SUB-SKILL: superpowers:subagent-driven-development"), checkbox-tracked tasks (`- [ ]`), a **Global Constraints** section (MSRV 1.89, zero mandatory deps, no_std + alloc, "no public fallible path may panic", "check order must match the C", fixed vocabulary, doc-density requirements, the docs CI gate rule), a file-structure table, and per-task "write the failing tests first" steps.
- `specs/2026-08-10-params-api-design.md`: date-prefixed design docs with Status header, problem statement with code examples, an ecosystem evidence table, a decisions table (choice + cut alternatives), then the exact new API surface in Rust signatures.
- `specs/2026-08-12-perf-svg-charts-design.md`: same shape for README perf charts (scope table, visual system, non-goals).

---

## 13. Cross-cutting conventions worth mirroring

- **Comment style**: short, factual, cite the C file/line for ported logic; every measurement claim carries the numbers and the host; "measured, not assumed" is a recurring header; rejected alternatives are documented with the measurement that rejected them.
- **Vocabulary**: module per ISA named after the ISA (`avx2.rs`, `avx512.rs`, `sse2.rs`, `neon.rs`, `wasm128.rs`, `sse41.rs`); backend enum variants CamelCase (`Sse2`, `Avx512`, `Wasm128`); `name()` lowercase for bench ids; features kebab-case; env vars `ARGON2_*`; out-of-band test cfg `argon2_force_avx2` (declared per-module, never in Cargo features).
- **Lints**: `#![warn(missing_docs)]`, `#![warn(unnameable_types)]`, `#![warn(clippy::undocumented_unsafe_blocks)]`; per-scoped `allow`s with reasons instead of blanket allows (`private_modules!`, `#[allow(clippy::too_many_arguments)]` where arity mirrors a C macro, etc.).
- **SemVer policy** (README + lib.rs): default-feature public API is SemVer-covered from 1.0.0; `internal-api`/`__internal` explicitly not; MSRV may rise on minor releases with a documented reason.
- **No-panic rule**: no fallible public path panics; the single exception (`build_or_panic`) exists to turn bad consts into compile errors.
- `Cargo.lock` is committed. `NOTICE` + `LICENSE` (MIT) present. `renovate.json` for dep updates. `SECURITY.md`.
