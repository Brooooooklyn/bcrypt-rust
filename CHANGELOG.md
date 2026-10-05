# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/Brooooooklyn/bcrypt-rust/releases/tag/v1.0.0) - 2026-10-05

First stable release. (0.1.0 was the crates.io bootstrap publish.)

### Added

- Pure-Rust bcrypt (`$2a$`/`$2b$`/`$2y$`): zero dependencies, `#![no_std]`
  (default `std` feature; `alloc` for owned APIs), MSRV 1.89, edition 2024.
- Batch-first API — `bcrypt_many`, `hash_many`, `verify_many` — with
  runtime-dispatched SIMD kernels measured per host: AVX-512 / AVX2 /
  AVX2x12 (48 KiB L1d parts) / SSE4.1 on x86-64, 2x8 NEON on aarch64,
  two-state v128 on wasm simd128, scalar elsewhere.
- SIMD bcrypt base64 codec on the same targets, byte-exact against the
  scalar reference with the decode reject set pinned exhaustively.
- `BCRYPT_FORCE_BACKEND` (std) pins the dispatch for benchmarks and
  debugging.

### Performance

- Nibble-LUT base64 decode on all SIMD backends: −22…−32% decode time on
  aarch64 NEON, −29% on Zen 4 AVX2.

### Fixed

- Packaging: `tests/` and `benches/` now ship in the crate; `cargo
  package` emits zero warnings and the packaged crate's test suite runs.
