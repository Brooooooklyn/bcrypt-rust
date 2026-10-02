# bcrypt-rust

Pure-Rust [bcrypt](https://www.usenix.org/legacy/events/usenix99/provos/provos.pdf)
(Provos & Mazières, USENIX 1999) with runtime-dispatched SIMD backends for
**batch** hashing. **Zero dependencies**, `#![no_std]`-capable, MSRV 1.89.

Wire-compatible with every implementation that speaks the `$2*$` hash-string
format (`$2v$cc$<22-char salt><31-char hash>`, exactly 60 ASCII bytes), and
API-compatible with the [`bcrypt`](https://crates.io/crates/bcrypt) crate
(0.19): same names, same semantics, same `BcryptError` variant set.

## Security and audit status

This crate implements security-sensitive cryptographic code and has **not**
received an independent third-party audit. Its defense-in-depth is the
verification suite described below. See [SECURITY.md](SECURITY.md) for the
support policy and how to report a vulnerability privately.

## What SIMD does — and does not — do here

bcrypt's cost loop is strictly sequential: **one hash cannot be
parallelized**. The SIMD backends therefore speed up **batch throughput
only**, by interleaving N *independent* hashes in lockstep, one per vector
lane. Single-hash latency is unchanged — `hash` / `verify` always run the
scalar kernel.

## Performance

Measured via `benches/micro.rs` (correctness-gated: every timed iteration's
output is asserted byte-identical to the scalar batch, so a divergent kernel
aborts instead of posting a fast-but-wrong number).

**Apple M5 Max** (aarch64), rustc 1.98.0, cost 5:

| backend | lanes | batch | hashes/s | vs scalar |
|---|---|---|---|---|
| scalar | 1 | 16 | 835.4 | 1.00 |
| NEON | 8 | 8 | 3174.7 | **3.90×** |
| NEON | 8 | 16 | 3193.8 | 3.82× |
| NEON | 8 | 32 | 3190.7 | 3.80× |
| wasm128 (under wasmtime 48, cost 4) | 8 | 8 | 2006.8 | 1.56× |

The NEON backend is an 8-lane scalar-GPR interleave (John the Ripper's
design: vector registers appear only at pack/unpack; the name is
historical). It replaced a 4-lane vector kernel after measurement
(2.78× → 3.82×). The wasm128 backend likewise runs two interleaved
4-lane states (1.21× → 1.56×).

**AMD EPYC Zen 4** (x86_64, Cloudflare sandbox, 4 vCPU), rustc 1.99.0, cost 5,
batch 16 (ranges span the ±10% drift between sandbox windows):

| backend | lanes | hashes/s | vs scalar |
|---|---|---|---|
| scalar | 1 | 568.8 | 1.00 |
| SSE4.1 | 4 | 631.4 | 1.11× |
| AVX2 | 8 | 896.8–950.2 | **1.58–1.67×** |
| AVX-512 | 16 | 883.9–932.5 | 1.55–1.64× |

AVX2 and AVX-512 sit at parity on Zen 4 (the width shootout picked AVX2 in
this window): AVX-512's SoA S-boxes are 64 KiB against Zen 4's 32 KiB L1d,
and Zen 4 double-pumps 512-bit ops anyway. On Sapphire Rapids — the one
µarch with a fast zmm gather (6 cycles) — the wide kernel pulls ahead, so
runtime dispatch answers the width question by measurement (one-time
shootout; see "Dispatch notes"). Both x86-64 backends also shoot out three
S-box lookup flavors (gather/insert/extract) per process the same way;
prescaled byte offsets plus an opaque-asm blocker that stops LLVM from
re-forming microcoded gathers took the AVX-512 path from 383 h/s to
parity.

Single-hash latency, cost 4: **646 µs** (criterion, M5 Max). Throughput
scales with `2^cost`, so cost 12 runs 128× slower per hash than cost 5.

### vs the `bcrypt` crate (0.19.3)

Same machine runs of `benches/vs_bcrypt.rs` (criterion; the sanity gate
asserts both crates produce byte-identical strings before timing). The
incumbent has no batch API, so its batch arm is the sequential `hash` loop —
the pattern real callers use today.

**Apple M5 Max:**

| workload | `bcrypt` | `bcrypt-rust` | speedup |
|---|---|---|---|
| single hash, cost 4 | 741 µs | 662 µs | 1.12× |
| single hash, cost 12 | 174.7 ms | 161.2 ms | 1.08× |
| batch 16, cost 4 | 1429 h/s | 6134 h/s | **4.29×** |
| batch 16, cost 12 | 5.79 h/s | 25.30 h/s | **4.37×** |
| verify, cost 4 | 718 µs | 666 µs | 1.08× |

**AMD EPYC Zen 4, 4 vCPU** (batch arms re-measured after the x86 work;
±10% sandbox drift applies):

| workload | `bcrypt` | `bcrypt-rust` | speedup |
|---|---|---|---|
| single hash, cost 4 | 972 µs | 899 µs | 1.08× |
| single hash, cost 12 | 239.4 ms | 240.2 ms | 1.00× |
| batch 16, cost 4 | 897 h/s | 1737 h/s | **1.94×** |
| batch 16, cost 12 | 3.69 h/s | 7.27 h/s | **1.97×** |
| verify, cost 4 | 977 µs | 998 µs | 0.98× |

### `parallel` feature (batch 64, Zen 4, 4 vCPU, cost 5)

| configuration | hashes/s | vs 1-core scalar |
|---|---|---|
| scalar | 565.5 | 1.00 |
| AVX2 | 910.6 | 1.61× |
| scalar + `parallel` | 1896.0 | 3.35× |
| AVX2 + `parallel` | 2753.2 | **4.87×** |

Chunks are lane-group-aligned, so on AVX-512 the parallel split engages only
when each worker gets at least one full 16-lane group (`items ≥ cores × 16`);
a batch of 16 on 4 cores stays single-worker by design.

### Per-backend verification status

| backend | lanes | targets | execution evidence |
|---|---|---|---|
| scalar | 1 | all | everywhere; the reference every other backend is checked against |
| NEON | 8 | aarch64 | native: full suite + the measurements above (M5 Max) |
| SSE4.1 | 4 | x86, x86_64 | full suite under Rosetta 2 **and** under QEMU TCG (Debian 12, real cpuid) |
| AVX2 | 8 | x86_64 | full suite under Rosetta 2 (`+avx2`) and QEMU TCG; the gather/insert/extract shootout asserts all three lookup flavors byte-identical before timing |
| AVX-512 | 16 | x86_64 | full suite on real Zen 4 hardware (Cloudflare sandbox) with `BCRYPT_REQUIRE_BACKEND=avx512` — byte-exact vs scalar, every other backend, and crypt_blowfish C; CI **requires** it on any runner advertising `avx512f` (macOS cannot execute it locally: Rosetta SIGILLs, QEMU TCG has no AVX-512 emulation) |
| wasm128 | 8 | wasm32 | full suite + micro bench under wasmtime (`+simd128`), scalar-fallback leg without it |

### Dispatch notes

Which of AVX2 and AVX-512 wins is a µarch question (working-set and
double-pump reasons above on Zen; the fast zmm hardware gather on Sapphire
Rapids), so `detect()` answers it by measurement: on `std` release builds, a
host advertising both runs a one-time width shootout — a cost-4, 32-item
batch through both backends' kernels, outputs asserted byte-identical before
anything is timed, three interleaved reps each, min wins — and caches the
winner for the process. Debug, `no_std` and Miri builds keep the static
AVX-512-first order. To override the pick, force the backend through
`__internal::bcrypt_many_with_backend` (`internal-api` feature) or run the
two explicitly and pick per machine, as `benches/micro.rs` does.

## Quick start

```rust
// Cost 4 keeps the example fast; use DEFAULT_COST (12) or higher in real code.
let hash = bcrypt_rust::hash(b"hunter2", 4)?;
assert_eq!(hash.len(), 60);
assert!(bcrypt_rust::verify(b"hunter2", &hash)?);
assert!(!bcrypt_rust::verify(b"hunter3", &hash)?);
```

Batch hashing (where the SIMD backends engage):

```rust
let passwords: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
let hashes = bcrypt_rust::hash_many(&passwords, 4)?;
for (password, hash) in passwords.iter().zip(&hashes) {
    assert!(bcrypt_rust::verify(password, hash)?);
}
```

## How the dispatch works

```
bcrypt_many / hash_many / hash_many_with_salts / verify_many
                    │
                    ▼
              detect()            cached in an AtomicU8; #[cold] probe
                    │
      ┌─────────────┼──────────────────────────────┐
      ▼             ▼                              ▼
 runtime cpuid   compile-time cfg               scalar
 (std builds)    (no_std / miri)                (fallback)
      │
      ▼
 avx512 ×16 ─ avx2 ×8 ─ sse41 ×4 ─ neon ×8 ─ wasm128 ×8 ─ scalar ×1
      │
      ▼
 batch split into lane groups; short groups padded, padding discarded
      │
      ▼
 per-item outputs — byte-identical on every backend (asserted in tests)
```

With the `parallel` feature, large batches (≥ 2 × available cores) are
additionally split across `std::thread::scope` workers on lane-group
boundaries — output stays byte-identical to the sequential loop.

## Feature flags

| feature | default | what it gates |
|---|---|---|
| `std` | ✓ | runtime CPU detection; OS-entropy salts for `hash`, `hash_many`, … |
| `alloc` | via `std` | `String`/`Vec` APIs: `bcrypt_many`, `hash_many_with_salts`, `verify_many`, `HashParts::format_for_version` |
| `zeroize` | ✓ | wipe internal buffers that held key material so `-O3` cannot elide it |
| `parallel` | — | thread-scoped batch splitting; throughput knob for large batches only |
| `internal-api` | — | exposes `__internal` for this crate's tests/benches; not stable |

With `alloc` but no `std`, everything works except the random-salt
conveniences and runtime detection (compile-time `target_feature` cfgs, then
scalar). With neither, the byte-oriented core (`bcrypt`, `hash_with_salt`,
`verify`, `HashParts` parsing/formatting) is fully functional.

## Semantics and compatibility

* Key preparation: password is NUL-appended then truncated at 72 bytes
  (`min(len + 1, 72)` key schedule bytes) — the OpenBSD/`bcrypt`-crate
  semantics.
* All four `$2a$`/`$2b$/`$2x$`/`$2y$` prefixes are accepted on parse and
  hashed with `$2b$` (correct-algorithm) semantics; `$2b$` is emitted on
  format.
* Two crypt_blowfish bug-compatibilities are deliberately **not** reproduced:
  the `$2x$` sign-extension bug and the `$2a$` 8-bit "safety XOR". The pinned
  rows, including the three survey rows reclassified against the vendored C
  source, live in `tests/vectors.rs`.

## Verification

* **Authoritative vectors** — 61 rows transcribed from jBCrypt, Openwall
  `wrapper.c`, the `bcrypt` crate, and pyca/bcrypt, plus divergent rows with
  per-row `must_verify` expectations (`tests/vectors.rs`).
* **Cross-backend differential** — every backend on the host must produce
  byte-identical output to scalar across lane-boundary batch sizes
  (0–33), costs 4–6, and the 71/72/73 password boundary
  (`tests/backends.rs`). `BCRYPT_REQUIRE_BACKEND=<name>` turns a skip into a
  failure for CI.
* **Differential vs reference C** — 360 seeded cases compared against the
  real `crypt_rn` of the vendored [crypt_blowfish](https://www.openwall.com/crypt/)
  1.3 (public domain), replayed through every available backend, including
  accept/reject parity on malformed settings (`tests/differential.rs`).
* **Fuzzing** — two `cargo-fuzz` targets: `decode` (the `$2*$` parser must
  never panic) and `tiny_hash` (cost validation + the whole pipeline at
  fuzzer-derived tiny parameters).
* **CI** — Linux/macOS/Windows, no-std check, docs, MSRV 1.89, Miri,
  ASAN/LSAN, wasm (simd128 + scalar fallback), entropy-target matrix,
  old-glibc, bench-compile, and a 5-minute fuzz leg per target.

## Reproducing the benchmarks

```console
# correctness-gated micro harness, per-backend, prints hashes/s and ratio
cargo bench --bench micro -- --backend all --batch 16 --cost 5 --iters 100 --vs-scalar

# criterion suite (single-hash latency, per-backend batch, verify)
cargo bench --bench bcrypt

# the CI regression net (CodSpeed)
cargo bench --bench codspeed

# wasm, under wasmtime
RUSTFLAGS="-C target-feature=+simd128" cargo test --release --target wasm32-wasip1
RUSTFLAGS="-C target-feature=+simd128" cargo bench --target wasm32-wasip1 --bench micro -- --backend all --batch 8 --cost 4 --iters 50 --vs-scalar
```

## MSRV

Rust **1.89**, pinned by exactly one thing: `stdarch_x86_avx512` (the AVX-512
intrinsics and the `avx512f` target feature) stabilized there. The CI `msrv`
job enforces it; raise it only with a reason.

## License

MIT. The `crypt_blowfish/` directory contains Openwall's crypt_blowfish 1.3
(public domain), vendored for differential testing only — it is never linked
into the crate.
