# bcrypt-rust design: pure-Rust bcrypt with runtime-dispatched SIMD

Date: 2026-10-01. Status: approved-by-user-brief ("a full SIMD optimized bcrypt implementation
like argon2"). Mirrors the architecture of `/Users/brooklyn/workspace/github/argon2-rust`
(v1.1.0). Research inputs: `/tmp/argon2-arch-report.md`, `/tmp/bcrypt-simd-research.md`,
`/tmp/rust-bcrypt-crates.md`.

## 1. The one fact that shapes everything

A single bcrypt hash **cannot** be SIMD-parallelized: the cost loop is `2^cost` iterations of
two EksBlowfish key expansions, each 521 strictly sequential Blowfish encryptions. The only
viable SIMD strategy is **multi-lane interleaving of independent hashes** (John the Ripper,
hashcat, cat-j/pbcrypt all do this). Therefore:

- The SIMD backends are reached only through **batch entry points** (`*_many`).
- Single-hash calls use the scalar backend. Their latency is unchanged.
- Expected batch throughput gains per core (measured literature, §9): AVX2-Intel ~2.5–3x,
  AVX-512 ~3–6x, AVX2-AMD ~2–2.5x, NEON/wasm ~1.5–2.5x. Honest README numbers only.

## 2. Algorithm (OpenBSD semantics, the de-facto standard)

- Blowfish: P[18], S[4][256] u32, init = π hex digits (transcribed from public-domain
  Openwall crypt_blowfish 1.3 `crypt_blowfish.c`; cross-checked against OpenBSD `blowfish.c`).
- `stream2word`: big-endian, cyclical mod input length.
- F(x) = `((S0[x>>24] + S1[(x>>16)&0xff]) ^ S2[(x>>8)&0xff]) + S3[x&0xff]`, wrapping adds.
- Encipher: 16 Feistel rounds, `Xl ^= P[i]; Xr ^= F(Xl); swap`, undo last swap,
  `Xr ^= P[16]; Xl ^= P[17]`.
- `expandstate(salt, key)`: init state, XOR key words into P (18 words max — a 72-byte
  password's trailing NUL is never read, so `$2b$` `min(72)+NUL` ≡ `min(72, len+1)`),
  then chain-encrypt zeros XORing salt words, overwriting P then S.
- `expand0state(data)`: XOR data words into P, chain-encrypt zeros, overwrite P then S.
- bcrypt core: `expandstate(salt16, key)`; `2^cost × { expand0state(key); expand0state(salt) }`
  (OpenBSD order — key first, salt second; the USENIX paper prints it backwards);
  then 64 × ECB re-encryption of `"OrpheanBeholderScryDoubt"` =
  `{0x4f727068,0x65616e42,0x65686f6c,0x64657253,0x63727944,0x6f756274}`; store 6 words BE;
  **encode 23 of 24 bytes** (drop `cdata[5] & 0xff`).
- Cost range 4..=31, default 12.
- bcrypt base64: alphabet `./A-Za-z0-9`, **MSB-first packing identical to RFC 4648**, no
  padding, partial final groups emitted (16 B → 22 chars, 23 B → 31 chars; last salt char ∈
  {. O e u}, last hash char index ≡ 0 mod 4). Scalar implementation only — the payload is at
  most 23 bytes, so a SIMD codec could never pay for itself; this differs from argon2-rust,
  whose PHC strings carry arbitrary-length fields.

## 3. Variant semantics (deliberate, documented)

Every accepted prefix (`$2a$`, `$2b$`, `$2x$`, `$2y$`) computes the **same, correct** `$2b$`
algorithm; versions are parse/format labels. This is exactly the `bcrypt` crate's (Keats)
behavior. Deliberately NOT implemented:

- crypt_blowfish's `$2x$` sign-extension bug emulation (CVE-2011-2481 era);
- crypt_blowfish's `$2a$` 8-bit "safety" XOR;
- OpenBSD `$2a$`'s `(u8)(len+1)` wraparound for >72-byte passwords.

The Openwall buggy vectors are kept in an expected-divergent table that must NOT match.

## 4. API surface — drop-in superset of `bcrypt` 0.19.3

Same names/signatures: `DEFAULT_COST=12`, `hash`, `verify`, `hash_with_salt`,
`hash_with_result`, `hash_bytes`, `hash_with_salt_bytes`, the `non_truncating_*` family,
`HashParts` (`get_cost`, `get_salt`, `get_salt_raw`, `format_for_version`,
`write_for_version`, `FromStr`, `Display` → `$2b$`), `Version::{TwoA,TwoX,TwoY,TwoB}`,
`BcryptError::{CostNotAllowed,InvalidHash,Rand,Truncation}`, `BcryptResult`, raw
`bcrypt(cost, salt, password) -> [u8; 24]`.

New (the SIMD payoff):

- `bcrypt_many(cost, passwords: &[&[u8]], salts: &[[u8; 16]]) -> Vec<[u8; 24]>` — raw batch.
- `hash_many(passwords: &[&[u8]], cost) -> Vec<BcryptResult<String>>` — random salt per item.
- `hash_many_with_salts(passwords, salts, cost) -> Vec<BcryptResult<HashParts>>`.
- `verify_many(passwords: &[&[u8]], hashes: &[&str]) -> Vec<BcryptResult<bool>>` — parses,
  groups by cost internally so one `verify_many` call handles mixed-cost inputs.
- `detected_backend() -> Backend` diagnostic.

Batch execution model: inputs are chunked into lane-width groups; a tail group shorter than
the lane count is padded by repeating lane 0 (duplicate output discarded). Lockstep across
lanes; one cost per kernel launch. Every lane position is byte-compared against scalar in
tests, including tail lanes.

## 5. SIMD kernel architecture

Per lane-independent state, struct-of-arrays:

- `Pv[18]` — 18 vectors of N u32 lanes.
- `Sv[4][256][N]` — S-boxes transposed: entry `(box, idx, lane)` at byte offset
  `((box*256 + idx)*N + lane)*4`, 64-byte-aligned. Vector stores during key schedule;
  per-F lookups are 4 gathers or 4×(N scalar loads + vector rebuild).

Working set ≈ 4.1 KiB/lane: 4 lanes ≈ 16 KiB, 8 lanes ≈ 33 KiB, 16 lanes ≈ 66 KiB
(L2-resident; accepted, documented).

Backend matrix and lookup strategy:

| Backend | lanes | S-box lookup | notes |
|---|---|---|---|
| Scalar | 1 | direct | also runs batch sequentially; safe Rust |
| Sse41 | 4 | scalar loads + rebuild (SSE2-only spelling kept for baseline) | x86 baseline |
| Avx2 | 8 | `vpgatherdd` ymm **or** loads+insert — flavor chosen per CPU | AMD Zen gathers are microcoded |
| Avx512 | 16 | `vpgatherdd` zmm **or** loads+insert | ditto |
| Neon | 4 | `ldr`+`ins` (no gather on Apple/Neoverse-N) | |
| Wasm128 | 4 | `v128.load32_lane`/`replace_lane` | compile-time selected |

Gather-vs-insert is a **measured** choice, not a vendor-string guess: on first AVX2/AVX-512
batch use, a one-time shootout (~1–2 ms, both flavors on the same 64-batch at cost 4, min of
3 reps each, interleaved) caches the winner in an `AtomicU8`. This mirrors argon2-rust's
`neon_wins_here` pattern. Both flavors are required to produce identical output (tested).

Key-schedule precomputations per batch (JtR/pbcrypt tricks): per-lane `key_words_v[18]`
(salt-independent), `P_init_key_v[18] = P_pi ^ key_words_v`; salt XOR is
`Pv[i] ^= salt_words_v[i & 3]`.

## 6. Dispatch — copied from argon2-rust

- `#[repr(u8)] #[non_exhaustive] enum Backend { Scalar=0, Neon=1, Sse41=2, Avx2=3, Avx512=4,
  Wasm128=5 }` with `ALL`, `name()`, `is_available()`, total `from_u8` (unknown → Scalar).
- `static CACHED_BACKEND: AtomicU8` with `0xFF` sentinel, relaxed, benign race,
  `#[cold] detect_and_cache()`, `detect()` uncached for tests, `cfg!(miri)` → Scalar.
- Per-feature probes written twice under mutually exclusive cfgs (`std` runtime detection vs
  compile-time `cfg!(target_feature)`); wasm is compile-time only.
- One kernel-pointer resolve per batch call; the batch API resolves once, then runs all
  groups. Safe API never names a backend; explicit-backend entry points are `unsafe fn` with
  paired runnable/`compile_fail` doctests, exposed only under `internal-api` via `__internal`.
- Cascade order: Avx512 → Avx2 → Sse41 → Neon → Wasm128 → Scalar (arch-gated probes).
- Unlike argon2-rust there is **no aarch64 scalar-vs-NEON shootout**: bcrypt NEON is
  load+insert, not a differently-scheduled pipeline, and 4 lanes × 4 KiB is L1-resident on
  every aarch64; expected to win everywhere. If a regression host appears, add the shootout
  then (documented as a deferred decision).

## 7. Crate skeleton (mirrors argon2-rust conventions)

```
Cargo.toml   edition 2024, rust-version 1.89 (AVX-512 stdarch), autobenches=false,
             harness=false benches, self-dev-dep for internal-api, profiles
             opt-level=3/lto=thin/codegen-units=1
build.rs     wasi_threadless cfg (copied)
src/lib.rs   #![no_std] + alloc, lints (missing_docs, unnameable_types,
             undocumented_unsafe_blocks), private_modules! macro, __internal
src/consts.rs    π P/S tables
src/base64.rs    bcrypt alphabet codec (scalar)
src/encoding.rs  HashParts, Version, $2*$ parse/format (strict 60-ASCII)
src/error.rs     BcryptError superset
src/core.rs      single-hash API + batch orchestration (chunk/pad/verify-grouping)
src/random.rs    OS CSPRNG salt generation (per-platform hand-declared, from argon2-rust)
src/eks/mod.rs   Backend, dispatch, flavor shootout
src/eks/{scalar,sse41,avx2,avx512,neon,wasm128}.rs
crypt_blowfish/  vendored Openwall 1.3 (public domain) for differential tests/benches
tests/{vectors,backends,differential,reuse}.rs
benches/{bcrypt,codspeed,micro}.rs + benches/support/cref_check.rs
fuzz/            phc-like decode + tiny-hash targets
```

Features: `default = ["std", "zeroize"]`; `std` = runtime detection + OS salt RNG;
`alloc` = no_std string API; `zeroize` = wipe lane states and password buffers;
`parallel` (non-default, implies `std`) = `std::thread::scope` split of large batches across
cores; `internal-api` = `__internal`. Rationale for `parallel` non-default: batch sizes under
one second rarely repay thread spawn; measure before defaulting.

MSRV 1.89, enforced by CI job reading `rust-version` from the manifest.

## 8. Verification strategy

1. `tests/vectors.rs` — curated table (61 positive vectors): Openwall `wrapper.c` suite
   (28, incl. empty password, exactly-72-byte, >72 truncation, 8-bit `\xaa`/`\xff` patterns,
   special salts), jBCrypt 20 (costs 6–12), rust-bcrypt cross-impl 8 (pyca/node/Go), 2
   UTF-8 cross-verified, invalid-settings error cases (`$2a$03`, `$2a$32`, `$2c$`, `$2z$`,
   `$2\`$`, `$2{$`), and the expected-divergent buggy-variant table (must NOT match).
   Every vector runs on scalar and every `is_available()` backend.
2. `tests/backends.rs` — randomized cross-backend equality: SplitMix64-seeded
   (password, salt, cost 4–6) tuples, scalar vs each available backend, incl. batch sizes
   1..=2*lanes+1 to exercise tail padding.
3. `tests/differential.rs` — shells out: builds vendored crypt_blowfish as a shared lib with
   `cc`, dlopens it (hand-declared, leaked handle, as argon2-rust does), compares full hash
   strings over randomized tuples. Restrictions: passwords without NUL bytes (C API is
   `char*`), 7-bit passwords under `$2a$` only; 8-bit compared under `$2b$`.
4. Doctests for the public API; `compile_fail` doctests pin the unsafe backend boundary.
5. Fuzz: `decode` (never panics on arbitrary bytes) and `tiny_hash` (cost pre-filtered,
   never panics) targets, cargo-fuzz compatible.
6. Miri + ASan CI jobs (CI file; not run locally).

## 9. Benches and honesty rules

- `benches/bcrypt.rs` — Criterion: single-hash scalar; batch throughput per backend × batch
  sizes {1, 8, 64, 256}; vs crypt_blowfish C via dlopen; vs `bcrypt` crate.
- `benches/micro.rs` — fn-main seconds-scale harness, env controls (`BCRYPT_BENCH_BACKEND`,
  `--vs-c`), tag equality asserted every rep (argon2-rust rules).
- `benches/codspeed.rs` — small hermetic CI regression net.
- Numbers in the README must be measured on the named host. M5 Max gives NEON natively;
  AVX2 numbers under Rosetta are reported as Rosetta numbers or omitted. AVX-512 numbers
  only when run on AVX-512 hardware (CI/cloud); never extrapolated.
- Expected ranges (from research §9) go in the README as *expectations*, next to measured
  reality.

## 10. Known limits

- AVX-512 cannot be executed on the development host (Apple M5 Max; Rosetta lacks AVX-512).
  The backend is compile-checked, unit-reviewed, and marked runtime-unverified until CI
  hardware runs it; the dispatch cascade never selects it on such hosts.
- wasm128 is compile-checked; executed only if a wasmtime runtime is available locally.
- Single-hash latency is unchanged by SIMD (batch-only speedup). This is stated in the
  crate-level docs so no user mistakes it.
