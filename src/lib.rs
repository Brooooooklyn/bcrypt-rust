//! A pure-Rust implementation of [bcrypt](https://www.usenix.org/legacy/events/usenix99/provos/provos.pdf),
//! the Blowfish-based password hash by Provos and Mazières (USENIX 1999),
//! with runtime-dispatched SIMD backends for **batch** hashing.
//!
//! bcrypt hashes a password with a per-password salt and a tunable cost
//! (`2^cost` iterations of the EksBlowfish key-expansion loop) and is
//! wire-compatible with every implementation that speaks the `$2*$`
//! hash-string format: `$2v$cc$<22-char salt><31-char hash>`, exactly 60
//! ASCII bytes.
//!
//! # What SIMD does — and does not — do here
//!
//! bcrypt's cost loop is strictly sequential: **one hash cannot be
//! parallelized**. The SIMD backends (chosen by runtime CPU detection, see
//! [`detected_backend`]) therefore speed up **batch throughput only**, by
//! interleaving N *independent* hashes in lockstep, one per vector lane.
//! Single-hash latency is unchanged, and [`hash`] / [`verify`] always run
//! the scalar kernel — they never see a [`Backend`] value at all.
//!
//! Measured batch gains (`benches/micro.rs` — every timed iteration
//! asserted byte-identical to scalar). Apple M5 Max (aarch64), cost 5:
//!
//! | backend | lanes | batch | hashes/s | vs scalar |
//! |---|---|---|---|---|
//! | scalar | 1 | 16 | 790.4 | 1.00 |
//! | NEON | 8 | 16 | 3187.1 | 4.03× |
//! | wasm128 (under wasmtime 48, cost 4) | 8 | 8 | 2006.8 | 1.56× |
//!
//! AMD EPYC Zen 4 (x86_64, 4 vCPU), cost 5, batch 16: scalar 568.8 h/s,
//! SSE4.1 1.11×, AVX2 1.58–1.67×, AVX-512 1.55–1.64× (parity on Zen 4; a
//! one-time runtime width shootout picks per host — see the README).
//! Batch-16 speedup vs the `bcrypt` crate: 4.32× (M5 Max) / 1.94× (Zen 4);
//! full tables and the `parallel` numbers are in the README.
//!
//! The batch entry points are [`bcrypt_many`], [`hash_many`],
//! [`hash_many_with_salts`] and [`verify_many`].
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "std")]
//! # fn run() -> Result<(), bcrypt_rust::BcryptError> {
//! // Cost 4 keeps the doctest fast; use `DEFAULT_COST` (12) or higher —
//! // whatever your latency budget tolerates — in real code.
//! let hash = bcrypt_rust::hash(b"hunter2", 4)?;
//! assert_eq!(hash.len(), 60);
//! assert!(bcrypt_rust::verify(b"hunter2", &hash)?);
//! assert!(!bcrypt_rust::verify(b"hunter3", &hash)?);
//! #     Ok(())
//! # }
//! # #[cfg(not(feature = "std"))]
//! # fn run() -> Result<(), bcrypt_rust::BcryptError> { Ok(()) }
//! # run().unwrap();
//! ```
//!
//! # Batch hashing
//!
//! ```
//! # #[cfg(feature = "std")]
//! # fn run() -> Result<(), bcrypt_rust::BcryptError> {
//! let passwords: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
//! let hashes = bcrypt_rust::hash_many(&passwords, 4)?;
//! assert_eq!(hashes.len(), 3);
//! for (password, hash) in passwords.iter().zip(&hashes) {
//!     assert!(bcrypt_rust::verify(password, hash)?);
//! }
//! #     Ok(())
//! # }
//! # #[cfg(not(feature = "std"))]
//! # fn run() -> Result<(), bcrypt_rust::BcryptError> { Ok(()) }
//! # run().unwrap();
//! ```
//!
//! # Features
//!
//! | feature | default | what it gates |
//! |---|---|---|
//! | `std` | ✓ | runtime CPU feature detection; OS-entropy salts for [`hash`], [`hash_bytes`], [`hash_with_result`], [`hash_many`] |
//! | `alloc` | via `std` | `String`/`Vec` APIs: [`bcrypt_many`], [`hash_many_with_salts`], [`verify_many`], [`HashParts::get_salt`], [`HashParts::format_for_version`] |
//! | `zeroize` | ✓ | securely wipe internal buffers that held key material (padded passwords, key words, lane scratch) so `-O3` cannot elide the erasure |
//! | `parallel` | — | split batches of ≥ 2 × available cores across `std::thread::scope` workers; a throughput knob for large batches only — output is byte-identical and the sequential loop runs below the threshold |
//! | `internal-api` | — | exposes `__internal` for this crate's own tests and benches; not stable |
//!
//! The crate is `#![no_std]` and stays that way with every feature enabled.
//! With `alloc` but no `std`, everything works except the random-salt
//! conveniences and runtime detection (backend selection falls back to
//! compile-time `target_feature` cfgs, then scalar). With neither, the
//! byte-oriented core — [`bcrypt`], [`hash_with_salt`],
//! [`hash_with_salt_bytes`], [`verify`], [`HashParts`] parsing/formatting —
//! is fully functional.
//!
//! # `bcrypt`-crate compatibility
//!
//! The API mirrors the [`bcrypt` crate](https://crates.io/crates/bcrypt)
//! (0.19): same names and semantics — NUL-then-truncate-at-72 key
//! preparation, all four `$2*$` prefixes accepted on parse, `$2b$` emitted on
//! format, [`HashParts`] with the same accessors, the same [`BcryptError`]
//! variant set. Two deliberate departures, both additive or internal:
//!
//! * [`BcryptError::Rand`] carries this crate's own [`EntropyError`] — OS
//!   entropy is hand-declared per platform, so the crate has **zero
//!   dependencies**.
//! * Additions the `bcrypt` crate does not have: the batch API above and
//!   [`HashParts::get_hash`] (the raw 23-byte payload).
//!
//! # Variant semantics
//!
//! Every accepted prefix — `$2a$`, `$2b$`, `$2x$`, `$2y$` — computes the
//! same, correct algorithm; the version is a parse/format label, and
//! formatting always emits `$2b$` unless told otherwise. The historical bug
//! emulations are deliberately **not** reproduced:
//!
//! * crypt_blowfish's `$2x$` sign-extension bug (CVE-2011-2481 era);
//! * crypt_blowfish's `$2a$` 8-bit "safety" XOR;
//! * OpenBSD `$2a$`'s `len + 1` byte wraparound for >72-byte passwords.
//!
//! See [the design doc, §3](https://github.com/Brooooooklyn/bcrypt-rust/blob/main/docs/superpowers/specs/2026-10-01-bcrypt-rust-design.md#3-variant-semantics-deliberate-documented)
//! for the full rationale.
//!
//! # Panics
//!
//! No fallible public path panics: hashing, verifying, parsing and batch
//! length validation all report failure as a [`BcryptError`].
//! [`verify_many`] returns per-item `Result`s precisely so that one
//! malformed string in a batch stays a value, not a panic.

#![no_std]
#![warn(missing_docs)]
// A `pub` item that a downstream crate cannot *name* is only half-public: it
// works in `let` bindings and nowhere else — not in a struct field, a
// function signature, or a `Vec`. `HashParts` (returned by
// `hash_with_result`) and `Backend` (returned by `detected_backend`) are
// exactly the items a caller needs to store; this lint is the regression
// guard.
#![warn(unnameable_types)]
#![warn(clippy::undocumented_unsafe_blocks)]
// `__internal` re-exports `eks::scalar` (a module) so tests can drive the
// scalar kernel directly. Its module doc links to private helpers
// (`expand_state`), which rustdoc flags once the module is publicly
// reachable; the link is correct under `--document-private-items` and the
// module is off-limits to edit, so allow it here rather than churn the docs.
#![allow(rustdoc::private_intra_doc_links)]
// clippy suggests `as_chunks`/`as_chunks_mut` (stable since 1.88, so usable at
// the pinned MSRV 1.89 — this lint fires on stable clippy). Rewriting the
// verified base64 codec for it is style-only churn, so the lint is allowed
// crate-wide instead.
#![allow(clippy::chunks_exact_to_as_chunks)]

// NOTE FOR EVERY CONTRIBUTOR: this crate has a module named `core`, which
// shadows the `core` crate *in this root module only*. Inside `src/lib.rs`
// always write `::core::...`. Submodules are unaffected — bare `core::` there
// still means the `core` crate.

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod error;

// These modules are private, and a good deal of what they expose escapes the
// crate only through `__internal` (below), which tests and benches enable. In
// a plain build those items are legitimately unreachable, so `dead_code`
// would fire on all of them.
//
// Rather than blanket-allowing `dead_code` — which would also hide code that
// is dead by mistake — the allow is tied to `internal-api` being OFF. With
// the feature ON, `__internal` re-exports the intended surface, so anything
// the compiler still calls dead really is dead and gets reported.
macro_rules! private_modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg_attr(not(feature = "internal-api"), allow(dead_code))]
            mod $name;
        )*
    };
}

private_modules!(base64, consts, core, encoding, eks, wipe);

// OS entropy for the convenience salt API; needs std for the syscall and the
// /dev/urandom fallback. Declared per-platform inside the module.
//
// Deliberately not in `private_modules!`: everything here is reachable from
// the random-salt entry points (`hash`, `hash_many`, …) on every `std`
// build, so it needs no `dead_code` allow, and should not have one hiding a
// future mistake.
#[cfg(feature = "std")]
mod random;

pub use crate::core::{
    DEFAULT_COST, MAX_COST, MIN_COST, bcrypt, hash_with_salt, hash_with_salt_bytes,
    non_truncating_hash_with_salt, non_truncating_hash_with_salt_bytes,
    non_truncating_verify, verify,
};
#[cfg(feature = "alloc")]
pub use crate::core::{bcrypt_many, hash_many_with_salts, verify_many};
#[cfg(feature = "std")]
pub use crate::core::{
    hash, hash_bytes, hash_many, hash_with_result, non_truncating_hash,
    non_truncating_hash_bytes, non_truncating_hash_with_result,
};
pub use crate::eks::Backend;
pub use crate::encoding::{HashParts, Version};
pub use crate::error::{BcryptError, BcryptResult, EntropyError};

/// The [`Backend`] this CPU resolved to, cached after the first call.
///
/// Diagnostic only — the hashing entry points resolve it for you. Remember
/// it only speaks for *batch* throughput: [`hash`] and [`verify`] always run
/// scalar (see the crate-level "What SIMD does — and does not — do here").
///
/// ```
/// println!("bcrypt backend: {}", bcrypt_rust::detected_backend());
/// ```
#[inline]
#[must_use]
pub fn detected_backend() -> Backend {
    crate::eks::backend()
}

#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __internal {
    //! Unstable internals, exposed for this crate's own tests and benches.
    //!
    //! Gated behind the non-default `internal-api` feature. **No stability
    //! guarantees**: anything here can change in a patch release.
    //!
    //! # Soundness
    //!
    //! Unstable is not the same as unsound. Every entry point here that takes
    //! an explicit [`Backend`] — the [`BcryptLanesFn`] pointers handed out by
    //! [`bcrypt_lanes_fn`] — is an `unsafe fn`: it dispatches to a
    //! `#[target_feature(enable = ...)]` kernel, so running one whose feature
    //! this CPU lacks is undefined behaviour (`SIGILL` in practice), and only
    //! the caller can rule that out. [`Backend::is_available`] is the
    //! portable check.
    //!
    //! The safe public API — [`hash`](crate::hash), [`verify`](crate::verify),
    //! the batch functions, [`detected_backend`](crate::detected_backend) —
    //! never lets a caller name a backend. It takes the backend from the
    //! cached runtime cascade in [`backend`], which by construction only ever
    //! names a backend this CPU advertises. That is the whole reason it can
    //! be safe, and why turning on `internal-api` cannot make a
    //! `#![forbid(unsafe_code)]` program reachable by UB.

    pub use crate::core::constant_time_eq;
    #[cfg(feature = "alloc")]
    pub use crate::core::bcrypt_many_with_backend;
    pub use crate::eks::scalar;
    pub use crate::eks::{Backend, BcryptLanesFn, backend, bcrypt_lanes_fn, detect};
    pub use crate::encoding::HashParts;

    // The base64 codec and wipe helpers are `pub(crate)` in their modules,
    // so they cannot be re-exported; these thin inline wrappers are the
    // same functions by another name.
    /// Encode a 16-byte salt to its 22-char bcrypt-base64 form.
    #[inline]
    #[must_use]
    pub fn encode_16(bytes: &[u8; 16]) -> [u8; 22] {
        crate::base64::encode_16(bytes)
    }

    /// Encode the 23 hash bytes to their 31-char bcrypt-base64 form.
    #[inline]
    #[must_use]
    pub fn encode_23(bytes: &[u8; 23]) -> [u8; 31] {
        crate::base64::encode_23(bytes)
    }

    // The unit error is the internal codec's exact contract (the only
    // failure is "bad char or length", with nothing to say about it); these
    // wrappers exist to give tests that exact signature.
    /// Decode exactly 22 bcrypt-base64 chars back to the 16-byte salt.
    #[inline]
    #[allow(clippy::result_unit_err)]
    pub fn decode_16(s: &[u8]) -> Result<[u8; 16], ()> {
        crate::base64::decode_16(s)
    }

    /// Decode exactly 31 bcrypt-base64 chars back to the 23 hash bytes.
    #[inline]
    #[allow(clippy::result_unit_err)]
    pub fn decode_23(s: &[u8]) -> Result<[u8; 23], ()> {
        crate::base64::decode_23(s)
    }

    /// Overwrite `xs` with zeros in a way the optimizer cannot elide.
    #[inline]
    pub fn secure_wipe_bytes(xs: &mut [u8]) {
        crate::wipe::secure_wipe_bytes(xs);
    }

    /// Overwrite `xs` with zeros in a way the optimizer cannot elide.
    #[inline]
    pub fn secure_wipe_u32(xs: &mut [u32]) {
        crate::wipe::secure_wipe_u32(xs);
    }

    /// Each backend's `bcrypt_lanes`, reachable directly so a differential
    /// test can pit two backends against each other on the same inputs.
    pub mod backends {
        pub use crate::eks::scalar;
        #[cfg(target_arch = "aarch64")]
        pub use crate::eks::neon;
        #[cfg(target_arch = "x86_64")]
        pub use crate::eks::avx2;
        #[cfg(target_arch = "x86_64")]
        pub use crate::eks::avx512;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub use crate::eks::sse41;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        pub use crate::eks::wasm128;
    }
}
