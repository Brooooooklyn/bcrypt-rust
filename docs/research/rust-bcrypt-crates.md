# Rust bcrypt ecosystem survey

Research date: 2026-10-01. Purpose: inform a new Rust bcrypt crate so it can (a) match the
de-facto standard API surface and (b) reuse authoritative test vectors / a reference C
implementation for differential testing.

All sources were downloaded and read directly (crate tarballs from static.crates.io,
git clone of Keats/rust-bcrypt at tag v0.19.3, Openwall crypt_blowfish-1.3 tarball,
jBCrypt TestBCrypt.java). Claims below were verified against source, and 15 vectors were
cross-checked by executing both `crypt_blowfish` (self-test on macOS arm64) and the Rust
`bcrypt` crate (0.19.3, release build).

---

## 1. The `bcrypt` crate (de-facto standard)

| Field | Value |
|---|---|
| crates.io name | `bcrypt` |
| Current version | **0.19.3** (released 2026-07-27; prior: 0.19.2, 0.19.1, 0.19.0, 0.18.0, 0.17.1) |
| Downloads | ~19.6 M total, ~6.0 M recent |
| Owner / repo | Vincent Prouillet — https://github.com/Keats/rust-bcrypt (tag v0.19.3) |
| Maintenance | Badge: "passively-maintained" |
| License | **MIT** |
| MSRV | 1.85.0 (per README; edition 2024; `rust-version` not set in manifest). 0.17.x was edition 2021, MSRV 1.63. |
| Unsafe | `#![forbid(unsafe_code)]` |
| Deps (0.19.3) | `blowfish` 0.10 (feature `bcrypt`), `base64` 0.23 (no default features), `getrandom` 0.4 (optional), `subtle` 2.4.1, `zeroize` 1.5.4 (optional) |

### Public API (0.19.3, quoted from `src/lib.rs`)

```rust
pub const DEFAULT_COST: u32 = 12;
// private: MIN_COST = 4, MAX_COST = 31
pub const BASE_64: GeneralPurpose = GeneralPurpose::new(&BCRYPT, NO_PAD);

pub type BcryptResult<T> = Result<T, BcryptError>;

pub enum BcryptError {
    CostNotAllowed(u32),                  // cost outside 4..=31
    InvalidHash(&'static str),            // malformed hash (cfg alloc|std)
    Rand(getrandom::Error),               // salt generation (cfg alloc|std)
    Truncation(usize),                    // non_truncating_* only; payload = input len incl. NUL
}

#[derive(Clone, Debug)]
pub enum Version { TwoA, TwoX, TwoY, TwoB }   // Display -> "2a"/"2x"/"2y"/"2b"

#[derive(Debug, PartialEq, Eq)]
pub struct HashParts { /* private: cost: u32, salt: [u8; 16], hash: [u8; 23] */ }
impl HashParts {
    pub fn get_cost(&self) -> u32;
    pub fn get_salt(&self) -> String;           // base64-encoded 22-char salt
    pub fn get_salt_raw(&self) -> [u8; 16];
    pub fn format_for_version(&self, version: Version) -> String;
    pub fn write_for_version<W: fmt::Write>(&self, version: Version, w: &mut W) -> fmt::Result;
    // private fn format(&self) -> [u8; 60]  (2b)
}
impl FromStr for HashParts { type Err = BcryptError; ... }   // strict 60-ASCII-byte parser
impl fmt::Display for HashParts { ... }                      // always $2b$

pub fn hash<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<String>;
pub fn verify<P: AsRef<[u8]>>(password: P, hash: &str) -> BcryptResult<bool>;
pub fn hash_with_salt<P: AsRef<[u8]>>(password: P, cost: u32, salt: [u8; 16]) -> BcryptResult<HashParts>;
pub fn hash_with_result<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<HashParts>;
// stack-buffer variants (new in 0.19):
pub fn hash_bytes<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<[u8; 60]>;
pub fn hash_with_salt_bytes<P: AsRef<[u8]>>(password: P, cost: u32, salt: [u8; 16]) -> BcryptResult<[u8; 60]>;
// strict 72-byte enforcement family:
pub fn non_truncating_hash<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<String>;
pub fn non_truncating_hash_with_result<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<HashParts>;
pub fn non_truncating_hash_with_salt<P: AsRef<[u8]>>(password: P, cost: u32, salt: [u8; 16]) -> BcryptResult<HashParts>;
pub fn non_truncating_hash_bytes<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<[u8; 60]>;
pub fn non_truncating_hash_with_salt_bytes<P: AsRef<[u8]>>(password: P, cost: u32, salt: [u8; 16]) -> BcryptResult<[u8; 60]>;
pub fn non_truncating_verify<P: AsRef<[u8]>>(password: P, hash: &str) -> BcryptResult<bool>;
// raw core, re-exported from a private module:
pub fn bcrypt(cost: u32, salt: [u8; 16], password: &[u8]) -> [u8; 24];
```

Behavioral contract worth copying exactly:

- Password is NUL-terminated, then truncated at 72 bytes (`copy_len = min(len,72)`, then
  `used = min(copy_len+1, 72)`); truncating functions silently truncate, `non_truncating_*`
  return `BcryptError::Truncation(len+1)` when `len >= 72`.
- Salt is 16 random bytes from `getrandom::fill`.
- Output hash string: `$<ver>$<cost:02>$<22 base64 salt><31 base64 hash>` (60 ASCII bytes;
  23 of 24 ciphertext bytes encoded).
- `verify` decodes both hashes and compares with `subtle::ConstantTimeEq` on the 23 raw bytes.
- All four prefixes `$2a$/$2b$/$2x$/$2y$` parse; all compute the *same, correct* algorithm
  (`$2x$` is just a formatting/parsing alias — no sign-extension bug emulation).
- `$2b$` is the default output version since 0.6.0.

### Base64

Not custom code (in-crate base64 was removed in bcrypt 0.8.0). It uses the `base64` crate's
**bcrypt alphabet** `./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`
(verified in base64-0.23.1 `src/alphabet.rs:273` — `pub const BCRYPT`), no padding:
`pub const BASE_64: GeneralPurpose = GeneralPurpose::new(&BCRYPT, NO_PAD);`

### Blowfish source

**Not vendored** — a dependency on the RustCrypto `blowfish` crate with `features = ["bcrypt"]`.
(History: bcrypt 0.1.x used `rust-crypto`'s blowfish module; 0.2.0 switched to the standalone
`blowfish` crate; 0.10.0 → blowfish 0.8; 0.12–0.17 → 0.9; 0.19 → 0.10.) The whole core is the
40-line `src/bcrypt.rs`: `bc_init_state()` → `salted_expand_key(salt, key)` →
`2^cost × {bc_expand_key(key); bc_expand_key(salt)}` → 64× `bc_encrypt` on
"OrpheanBeholderScryDoubt" (6 u32s), output big-endian.

### Feature flags

```
default = ["std", "zeroize"]
std     = ["getrandom/std", "base64/std"]
alloc   = ["base64/alloc", "getrandom"]   # no_std + alloc gives full API; no alloc => only raw bcrypt()
zeroize = []                              # zeroizes the password buffer after hashing
```

`no_std` supported; with neither `std` nor `alloc` only the raw `bcrypt()` function is usable.
Old `js` feature removed in 0.17.0.

### Test vectors inside the crate

Tests are **inline** (`src/lib.rs mod tests`, `src/bcrypt.rs mod tests`) — the crates.io
tarball ships them (it includes `src/**/*`); the GitHub repo additionally has `benches/`,
`examples/`, `fuzz/` (hash + verify fuzz targets). No `tests/` directory.

Fixed vectors in `src/lib.rs` (verify-style, provenance noted in comments):

| Password | Hash | Provenance |
|---|---|---|
| `password` | `$2a$04$UuTkLRZZ6QofpDOlMz32MuuxEHA43WOemOYHPz6.SjsVsyO1tDU96` | online tool |
| `correctbatteryhorsestapler` | `$2b$04$EGdrhbKUv8Oc9vGiXX0HQOxSg445d458Muh7DAHskb6QbtCvdxcie` | Python (pyca/bcrypt) |
| `correctbatteryhorsestapler` | `$2a$04$n4Uy0eSnMfvnESYL.bLwuuj0U/ETSsoTpRT9GVk5bektyVVa5xnIi` | node |
| 32 binary bytes `[29,225,195,167,…]` (high bytes) | `$2a$04$tjARW6ZON3PhrAIRW2LG/u9aDw5eFdstYLR8nFCNaOQmsH9XD23w.` | Go x/crypto |
| `"x" × 100` (truncation) | `$2a$05$......................YgIDy4hFBdVlc/6LHnD9mX488r9cLd2` | pyca/bcrypt |
| `My S3cre7 P@55w0rd!` + raw salt `[38,113,212,…,18]` | `$2b$05$HlFShUxTu4ZHHfOLJwfmCeDj/kuKFKboanXtDJXxCC7aIPTUgxNDe` | crate's own hash_with_salt |
| `hunter2` + zero salt, cost 12 | `$2a$12$…21jzCB1r6pN6rp5O2Ev0ejjTAboskKm` (and 2b/2x/2y variants) | formatting test |
| `hunter2`, cost 4 round-trip | `$2b$04$…` | self-generated |

Plus: `split_hash`/`HashParts` parsing on `$2y$12$L6Bc/…`, malformed-hash error cases,
NUL-byte behavioral matrix (`"\0"`, `"a\0"`, …), regression tests for non-ASCII in hash
strings (#62), and 2 quickcheck property tests. `src/bcrypt.rs` has 3 **raw byte-level**
vectors (salt+password bytes → 23 output bytes) "unbase64ed from jBCrypt" (empty, `a`,
`abcdefghijklmnopqrstuvwxyz`).

Coverage gaps in the crate's own tests: no fixed empty-password vector, no fixed UTF-8
password vector, no `$2y$` compute vector with known password, no high-cost (>12) vector.

---

## 2. RustCrypto `blowfish` crate (0.10.0)

Repo: https://github.com/RustCrypto/block-ciphers · License: MIT OR Apache-2.0 · MSRV 1.85 ·
edition 2024 · `#![no_std]`, `#![deny(unsafe_code)]` · deps: `cipher` 0.5, `byteorder`.

**Yes — it has bcrypt-specific support** behind `features = ["bcrypt"]` (which is exactly what
the `bcrypt` crate enables). From `src/lib.rs`:

```rust
#[cfg(feature = "bcrypt")]
impl Blowfish<BE> {
    pub fn salted_expand_key(&mut self, salt: &[u8], key: &[u8]);  // eksblowkey with salt XOR
    pub fn bc_init_state() -> Blowfish<BE>;                        // digits-of-pi initial state
    pub fn bc_encrypt(&self, lr: [u32; 2]) -> [u32; 2];            // raw 64-bit block encrypt on u32 pair
    pub fn bc_expand_key(&mut self, key: &[u8]);                   // standard Blowfish expand_key
}
```

There is no separate `BcryptState` type — the bcrypt API hangs off `Blowfish<BE>` itself
(0.9/0.10; older docs sometimes called it `Blowfish::bc_*`). A bcrypt implementation builds on
exactly these four functions (see the 40-line `bcrypt.rs` quoted above). `salted_expand_key`
XORs the key into P with `next_u32_wrap` (cyclic big-endian byte reads) and XORs salt into the
running ciphertext — this is the OpenBSD `Blowfish_expandstate`.

Performance posture (relevant if the new crate re-implements the core for SIMD):

- **Scalar only.** `ParBlocksSize = U1`; no SIMD paths, no `unsafe`, no arch-specific code.
- S-boxes are **static const tables** (`const S: [[u32; 256]; 4]`, `const P: [u32; 18]`, 4.1 KB
  in `src/consts.rs`) copied into the state struct on init — they are *not* inlined into code;
  each `round_function` does 4 data-dependent `u32` table loads + `((a+b)^c)+d`.
- `encrypt()` is hand-structured as 8 loop iterations (2 Feistel rounds each) + final P XORs.
- Endian-aware: generic over `byteorder::ByteOrder` — `Blowfish<BE>` (default, what bcrypt
  needs) and `BlowfishLE`; block I/O uses `T::read_u32_into`/`write_u32_into`.
- `#[inline]` only on the `cipher`-trait backends; the bcrypt path calls `encrypt` directly.
- Optional `zeroize` wipes S+P on `Drop`.

## 3. Other bcrypt-related Rust crates

- **`bcrypt-pbkdf` 0.11.0** (RustCrypto/password-hashes, MIT/Apache-2.0, MSRV 1.85, no_std,
  ~16.8 M downloads): the OpenSSH `bcrypt_pbkdf` KDF used by encrypted SSH private keys — a
  *different construction* (PBKDF2-style over bcrypt "Bhash" with seed
  "OxychromaticBlowfishSwatDynamite", SHA-512), **not** `$2*$` password hashing. Also builds on
  `blowfish` with the `bcrypt` feature (+ `sha2` 0.11, `pbkdf2` 0.13). API:
  `bcrypt_pbkdf(passphrase, salt, rounds, output)` and `bcrypt_pbkdf_with_memory` (heapless).
- **`pwhash` 1.0.0** (inejge/pwhash, MIT): old collection of Unix crypt hashes; its `bcrypt`
  module wraps `blowfish` 0.7 with `BcryptSetup`/`BcryptVariant` (2a/2b/2y) and `hash`/`verify`.
  Effectively unmaintained (rand 0.8-era deps); the ecosystem converged on Keats' `bcrypt`.
- **`rust-crypto` 0.2.36**: historical; its blowfish module was the pre-0.2.0 source of
  rust-bcrypt's core. Do not use.
- **`ssh-key`**: uses `bcrypt-pbkdf` for encrypted private keys (main downstream consumer).
- Nothing else notable: crates.io search for "bcrypt" by downloads surfaces only the above
  plus unrelated crates; `bcrust` and `blowfish-simd` do **not** exist on crates.io (404).

## 4. SIMD / parallel Rust bcrypt

**None exists as a published crate.** Searches (`bcrust`, `blowfish-simd`, "rust bcrypt
avx2/simd", GitHub) find no vectorized eksblowfish in Rust; `Keats/rust-bcrypt` forbids unsafe
and the RustCrypto blowfish is scalar-only.

What exists elsewhere (approach reference for a future SIMD crate):

- **pbcrypt** (https://github.com/cat-j/pbcrypt, C + NASM, academic, $2b$-only cracker):
  hashes *N independent candidate passwords* in SIMD lanes — 4-way SSE4 / 8-way AVX2
  (`vpgatherdd` for the data-dependent S-box lookups, P-array preloaded in YMM, cache-aligned
  code, unrolled loops). +33% (4-way) / +175% (8-way) hashes/sec vs scalar. Validated against
  OpenBSD bcrypt.
- **John the Ripper / hashcat**: same cross-candidate batching approach (bcrypt's serial,
  data-dependent key schedule defeats intra-hash SIMD; you vectorize across instances).
- Rust crackers (e.g. brutecraber) just call `bcrypt::verify` under rayon — thread-level
  parallelism only.

## 5. Openwall crypt_blowfish (reference C for differential testing)

Fetched from https://www.openwall.com/crypt/ → `crypt_blowfish-1.3.tar.gz`.

| Field | Value |
|---|---|
| Current version | **1.3** (2014-07-07; adds `$2b$`. 1.1 fixed CVE-2011-2483 8-bit handling + added 8-bit test vectors; 1.2 added `$2y$`) |
| License | **Public domain** ("No copyright is claimed, and the software is hereby placed in the public domain", with a fallback permissive "redistribution permitted" clause) — confirmed in every file header and on the web page. Safe to vendor. |
| Author | Solar Designer; code comes from John the Ripper |

File list (tarball):

```
crypt_blowfish.c   32 KB — THE implementation: BF_* eksblowfish core + _crypt_blowfish_rn
crypt_blowfish.h         — internal API decls (_crypt_blowfish_rn, _crypt_output_magic, _crypt_gensalt_blowfish_rn)
wrapper.c          15 KB — crypt(3)-compatible wrappers + #ifdef TEST self-test with the test arrays
crypt_gensalt.c/h        — salt setting-string generation
ow-crypt.h               — public API header (use this one)
crypt.h                  — glibc glue (not needed standalone)
x86.S                    — optional x86-32 asm (assembles to empty object off-x86)
Makefile, crypt.3, README, PERFORMANCE, LINKS, glibc-*.diff
```

- The `$2a$/$2b$` (and `$2x$/$2y$) implementation is in **`crypt_blowfish.c`**; the test
  vectors are in **`wrapper.c`** under `#ifdef TEST` (`static const char *tests[][3]`).
- **Builds standalone easily on macOS/Linux**: `make check` worked unmodified on macOS arm64
  (Apple clang; only 3 harmless null-pointer-arithmetic warnings; x86.S produces an empty
  object; self-test passed and benchmarked ~459 c/s at cost 5). As a library: compile
  `crypt_blowfish.c crypt_gensalt.c wrapper.c x86.S` and include `ow-crypt.h`.
- **C API** (from `ow-crypt.h`):

```c
char *crypt   (const char *key, const char *setting);                       // static buffer
char *crypt_r (const char *key, const char *setting, void *data);
char *crypt_rn(const char *key, const char *setting, void *data, int size); // reentrant, caller buffer
char *crypt_ra(const char *key, const char *setting, void **data, int *size); // reentrant, auto-alloc (realloc)
char *crypt_gensalt   (const char *prefix, unsigned long count, const char *input, int size);
char *crypt_gensalt_rn(..., char *output, int output_size);
char *crypt_gensalt_ra(const char *prefix, unsigned long count, const char *input, int size);
// direct bcrypt entry (crypt_blowfish.h):
char *_crypt_blowfish_rn(const char *key, const char *setting, char *output, int size);
```

  For "compute a bcrypt hash given password + settings string", `crypt_rn(key, setting, buf,
  61)` (CRYPT_OUTPUT_SIZE = 7+22+31+1) or `crypt_ra` are the right calls; `setting` may be a
  full hash (verify-style) or `"$2b$05$<22 salt chars>"`. `key` is a NUL-terminated C string
  (≤ 72 chars used), so NUL-byte passwords are untestable through this API — note for
  differential harness design.

- **Prefix semantics — critical for differential testing**: `$2y$`/`$2b$` = correct algorithm
  (OpenBSD-compatible). `$2x$` = deliberately *buggy* sign-extension emulation (CVE-2011-2483
  era). `$2a$` = bug-compatible with pre-1.1 crypt_blowfish for passwords containing bytes ≥
  0x80 (kept so old buggy hashes still verify; correct for 7-bit passwords). The Rust `bcrypt`
  crate implements only the correct algorithm for all prefixes. **Verified divergence**:
  Openwall vector `$2a$05$/OK…nqd1wy.pTMdcvrRWxyiGL2eMz.2a85.` for password `\xff\xff\xa3`
  returns `false` under Rust bcrypt 0.19.3, while the `$2y$`/`$2b$` siblings verify. The
  `$2x$`/8-bit-`$2a$` vectors must be flagged expected-divergent in a cross-implementation
  test suite.

## 6. Consolidated authoritative test vectors

Legend — sources: **JB** = jBCrypt `TestBCrypt.java` (djmdjm/jBCrypt, ISC-style license, 20
vectors), **OW** = Openwall `crypt_blowfish-1.3/wrapper.c` tests array (public domain), **RB**
= rust-bcrypt crate tests (MIT), **X** = generated for this survey with pyca/bcrypt (Python,
Apache-2.0, OpenBSD-derived C) and cross-verified against rust-bcrypt 0.19.3 (all X and the
spot-checked OW/JB/RB vectors passed `bcrypt::verify`).

All "expected" strings are full 60-char hashes; the salt/settings string is the first 29
chars of the expected hash (prefix + cost + 22-char salt).

| # | Password | Settings / salt | Expected full hash | Src | Notes |
|---|---|---|---|---|---|
| 1 | (empty) | `$2a$06$DCq7YPn5Rq63x1Lad4cll.` | `$2a$06$DCq7YPn5Rq63x1Lad4cll.TV4S6ytwfsfvkgY8jIucDrjc8deX1s.` | JB | empty pw |
| 2 | (empty) | `$2a$08$HqWuK6/Ng6sg9gQzbLrgb.` | `$2a$08$HqWuK6/Ng6sg9gQzbLrgb.Tl.ZHfXLhvt/SgVyWhQqgqcZ7ZuUtye` | JB | |
| 3 | (empty) | `$2a$10$k1wbIrmNyFAPwPVPSVa/ze` | `$2a$10$k1wbIrmNyFAPwPVPSVa/zecw2BCEnBwVS2GbrmgzxFUOqW9dk4TCW` | JB | |
| 4 | (empty) | `$2a$12$k42ZFHFWqBp3vWli.nIn8u` | `$2a$12$k42ZFHFWqBp3vWli.nIn8uYyIkbvYRvodzbfbK18SSsY.CsIQPlxO` | JB | high cost 12 |
| 5 | `a` | `$2a$06$m0CrhHm10qJ3lXRY.5zDGO` | `$2a$06$m0CrhHm10qJ3lXRY.5zDGO3rS2KdeeWLuGmsfGlMfOxih58VYVfxe` | JB | |
| 6 | `a` | `$2a$08$cfcvVd2aQ8CMvoMpP2EBfe` | `$2a$08$cfcvVd2aQ8CMvoMpP2EBfeodLEkkFJ9umNEfPD18.hUF62qqlC/V.` | JB | |
| 7 | `a` | `$2a$10$k87L/MF28Q673VKh8/cPi.` | `$2a$10$k87L/MF28Q673VKh8/cPi.SUl7MU/rWuSiIDDFayrKk/1tBsSQu4u` | JB | |
| 8 | `a` | `$2a$12$8NJH3LsPrANStV6XtBakCe` | `$2a$12$8NJH3LsPrANStV6XtBakCez0cKHXVxmvxIlcz785vxAIZrihHZpeS` | JB | |
| 9 | `abc` | `$2a$06$If6bvum7DFjUnE9p2uDeDu` | `$2a$06$If6bvum7DFjUnE9p2uDeDu0YHzrHM6tf.iqN8.yx.jNN1ILEf7h0i` | JB | |
| 10 | `abc` | `$2a$08$Ro0CUfOqk6cXEKf3dyaM7O` | `$2a$08$Ro0CUfOqk6cXEKf3dyaM7OhSCvnwM9s4wIX9JeLapehKK5YdLxKcm` | JB | |
| 11 | `abc` | `$2a$10$WvvTPHKwdBJ3uk0Z37EMR.` | `$2a$10$WvvTPHKwdBJ3uk0Z37EMR.hLA2W6N9AEBhEgrAOljy2Ae5MtaSIUi` | JB | |
| 12 | `abc` | `$2a$12$EXRkfkdmXn2gzds2SSitu.` | `$2a$12$EXRkfkdmXn2gzds2SSitu.MW9.gAVqa9eLS1//RYtYCmB1eLHg.9q` | JB | |
| 13 | `abcdefghijklmnopqrstuvwxyz` | `$2a$06$.rCVZVOThsIa97pEDOxvGu` | `$2a$06$.rCVZVOThsIa97pEDOxvGuRRgzG64bvtJ0938xuqzv18d3ZpQhstC` | JB | |
| 14 | `abcdefghijklmnopqrstuvwxyz` | `$2a$08$aTsUwsyowQuzRrDqFflhge` | `$2a$08$aTsUwsyowQuzRrDqFflhgekJ8d9/7Z3GV3UcgvzQW3J5zMyrTvlz.` | JB | |
| 15 | `abcdefghijklmnopqrstuvwxyz` | `$2a$10$fVH8e28OQRj9tqiDXs1e1u` | `$2a$10$fVH8e28OQRj9tqiDXs1e1uxpsjN0c7II7YPKXua2NAKYvM6iQk7dq` | JB | |
| 16 | `abcdefghijklmnopqrstuvwxyz` | `$2a$12$D4G5f18o7aMMfwasBL7Gpu` | `$2a$12$D4G5f18o7aMMfwasBL7GpuQWuP3pkrZrOAnqP.bmezbMng.QwJ/pG` | JB | |
| 17 | `~!@#$%^&*()      ~!@#$%^&*()PNBFRD` | `$2a$06$fPIsBO8qRqkjj273rfaOI.` | `$2a$06$fPIsBO8qRqkjj273rfaOI.HtSV9jLDpTbZn782DC6/t7qT67P6FfO` | JB | symbols/spaces |
| 18 | `~!@#$%^&*()      ~!@#$%^&*()PNBFRD` | `$2a$08$Eq2r4G/76Wv39MzSX262hu` | `$2a$08$Eq2r4G/76Wv39MzSX262huzPz612MZiYHVUJe/OcOql2jo4.9UxTW` | JB | |
| 19 | `~!@#$%^&*()      ~!@#$%^&*()PNBFRD` | `$2a$10$LgfYWkbzEvQ4JakH7rOvHe` | `$2a$10$LgfYWkbzEvQ4JakH7rOvHe0y8pHKF9OaFgwUZ2q7W2FFZmZzJYlfS` | JB | |
| 20 | `~!@#$%^&*()      ~!@#$%^&*()PNBFRD` | `$2a$12$WApznUOJfkEGSmYRfnkrPO` | `$2a$12$WApznUOJfkEGSmYRfnkrPOr466oFDCaj4b6HY3EXGvfxm43seyhgC` | JB | |
| 21 | `U*U` | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.` | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW` | OW | |
| 22 | `U*U*` | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.` | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.VGOzA784oUp/Z0DY336zx7pLYAy0lwK` | OW | |
| 23 | `U*U*U` | `$2a$05$XXXXXXXXXXXXXXXXXXXXXO` | `$2a$05$XXXXXXXXXXXXXXXXXXXXXOAcXxm9kjPGEMsLznoKqmqw7tc8WCx4a` | OW | |
| 24 | `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789chars after 72 are ignored` | `$2a$05$abcdefghijklmnopqrstuu` | `$2a$05$abcdefghijklmnopqrstuu5s2v8.iXieOjg/.AySBTTZIIVFJeBui` | OW | >72 chars, truncation |
| 25 | (empty) | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.` | `$2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy` | OW | empty pw, cost 5 |
| 26 | `\xaa`×72 + `chars after 72 are ignored as usual` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.swQOIzjOiJ9GHEPuhEkvqrUyvWhEMx6` | OW | 72×0xAA + truncation |
| 27 | `\xaa\x55`×36 (exactly 72 bytes) | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.R9xrDjiycxMbQE2bp.vgqlYpW5wx2yy` | OW | 72-byte edge, high bytes |
| 28 | `\x55\xaa\xff`×24 (exactly 72 bytes) | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.9tQZzcJfm3uj2NvJ/n5xkhpqLrMpWCe` | OW | 72-byte edge, contains 0xFF |
| 29 | `\xff\xff\xa3` | `$2b$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2b$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e` | OW | $2b$, 0xFF bytes |
| 30 | `\xff\xff\xa3` | `$2y$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2y$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e` | OW | $2y$ == $2b$ |
| 31 | `\xa3` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq` | OW | 8-bit, correct alg |
| 32 | `\xa3` | `$2y$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2y$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq` | OW | $2y$ == $2a$ here |
| 33 | `\xff\xa3` + `34` + `\xff\xff\xff\xa3` + `345` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.ZC1JEJ8Z4gPfpe1JOr/oyPXTWl9EFd.` | OW | mixed 8-bit |
| 34 | `hunter2` | `$2b$12$......................` (zero salt) | `$2b$12$......................21jzCB1r6pN6rp5O2Ev0ejjTAboskKm` | RB | cost 12; 2a/2x/2y = same body |
| 35 | `My S3cre7 P@55w0rd!` | salt bytes `[38,113,212,141,108,213,195,166,201,38,20,13,47,40,104,18]`, cost 5 | `$2b$05$HlFShUxTu4ZHHfOLJwfmCeDj/kuKFKboanXtDJXxCC7aIPTUgxNDe` | RB | hash_with_salt |
| 36 | `"x"×100` | `$2a$05$......................` | `$2a$05$......................YgIDy4hFBdVlc/6LHnD9mX488r9cLd2` | RB | truncation (from pyca) |
| 37 | `correctbatteryhorsestapler` | `$2b$04$EGdrhbKUv8Oc9vGiXX0HQO` | `$2b$04$EGdrhbKUv8Oc9vGiXX0HQOxSg445d458Muh7DAHskb6QbtCvdxcie` | RB | cost 4, pyca |
| 38 | `correctbatteryhorsestapler` | `$2a$04$n4Uy0eSnMfvnESYL.bLwuu` | `$2a$04$n4Uy0eSnMfvnESYL.bLwuuj0U/ETSsoTpRT9GVk5bektyVVa5xnIi` | RB | node |
| 39 | bytes `[29,225,195,167,223,236,85,195,114,227,7,0,209,239,189,24,51,105,124,168,151,75,144,64,198,197,196,4,241,97,110,135]` | `$2a$04$tjARW6ZON3PhrAIRW2LG/u` | `$2a$04$tjARW6ZON3PhrAIRW2LG/u9aDw5eFdstYLR8nFCNaOQmsH9XD23w.` | RB | high bytes, Go x/crypto |
| 40 | `★★★★★★★★` (U+2605 ×8, 24 UTF-8 bytes) | `$2a$05$......................` | `$2a$05$......................CVh3qAKQwo3AyWm2sH24x.4W0jOiobK` | X | UTF-8 pw, pyca×Rust verified |
| 41 | `★★★★★★★★` | `$2b$05$CCCCCCCCCCCCCCCCCCCCC.` | `$2b$05$CCCCCCCCCCCCCCCCCCCCC.GKRlRE2yXEq.CPNmj6AdW2OEOSzU/GW` | X | UTF-8 pw, $2b$ |
| 42 | `hunter2`×10 + `ab` (exactly 72 bytes) | `$2a$05$......................` | `$2a$05$......................VbcD.3tcy/UErGUKxMqc6T88xuK4rWq` | X | 72-char ASCII edge |

**Expected-divergent vectors (buggy sign-extension emulation — keep in a separate table; a
correct implementation must NOT match these):**

| # | Password | Expected hash (crypt_blowfish) | Src | Why divergent |
|---|---|---|---|---|
| 43 | `\xa3` | `$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e` | OW | $2x$ = buggy alg; collides with correct hash of `\xff\xff\xa3` |
| 44 | `\xff\xff\xa3` | `$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e` | OW | buggy |
| 45 | `\xff\xff\xa3` | `$2a$05$/OK.fbVrR/bpIqNJ5ianF.nqd1wy.pTMdcvrRWxyiGL2eMz.2a85.` | OW | $2a$ 8-bit bug-compat mode (verified: Rust bcrypt returns false) |
| 46 | `1\xa3` + `345` | `$2x$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi` | OW | buggy |
| 47 | `\xff\xa3` + `345` | `$2y$05$/OK.fbVrR/bpIqNJ5ianF.nRht2l/HRhr6zmCp9vYUvvsqynflf9e` | OW | correct alg (matches Rust); its $2a$/$2x$ siblings are buggy |
| 48 | `\xd1\x91` | `$2x$05$6bNw2HLQYeqHYyBfLMsv/OiwqTymGIGzFsA4hOTWebfehXHNprcAS` | OW | buggy |

Openwall's wrapper.c also has 7 invalid-setting cases (`$2a$03$…`, `$2a$32$…`, `$2c$…`,
`$2z$…`, `` $2`$ ``, `$2{$…` → must fail with EINVAL; `*0`/`*1` magic for failure returns) —
worth mirroring as error-path tests.

## 7. Recommendations

**API surface for a drop-in superset of `bcrypt` 0.19.3** (match these names/signatures
exactly, then extend):

1. `DEFAULT_COST: u32 = 12`; cost range 4..=31.
2. `hash`, `verify`, `hash_with_salt`, `hash_with_result`, `hash_bytes`,
   `hash_with_salt_bytes`, and the full `non_truncating_*` family + `non_truncating_verify`.
3. `HashParts { get_cost, get_salt, get_salt_raw, format_for_version, write_for_version }`
   + `FromStr` + `Display` ($2b$); `Version::{TwoA,TwoX,TwoY,TwoB}`.
4. `BcryptError::{CostNotAllowed(u32), InvalidHash(&'static str), Rand, Truncation(usize)}`
   + `BcryptResult<T>`. (If you want 0.17.x compat too, that line had extra variants
   `Io/InvalidCost/InvalidPrefix/InvalidSaltLen/InvalidBase64` — the 0.19 surface is the
   current standard.)
5. Raw `pub fn bcrypt(cost: u32, salt: [u8; 16], password: &[u8]) -> [u8; 24]`.
6. bcrypt-base64 alphabet `./A–Za–z0–9`, no padding; `subtle` constant-time compare in
   verify; NUL-terminate-then-truncate-at-72 semantics; `$2b$` default output; parse all
   four prefixes with a strict 60-ASCII-byte check; compute only the correct algorithm
   ($2x$ as parse/format alias).
7. Features mirroring: `default = ["std","zeroize"]`, `alloc` for no_std, `zeroize`
   optional. Sensible supersets: `rayon`/`parallel` batch-verify API, `std::arch` SIMD core.

**Reference C to vendor for differential testing**: Openwall **crypt_blowfish 1.3** —
public domain (no license friction at all). Vendor `crypt_blowfish.c`, `crypt_gensalt.c`,
`wrapper.c`, headers; skip `x86.S` except on x86. It compiled and self-tested unmodified on
macOS arm64 with `make check`. Call `crypt_rn(key, setting, buf, 61)` / `crypt_ra` from a
tiny FFI shim. Caveats: password is a C string (no NUL-byte differential tests), and its
`$2a$`-8-bit/`$2x$ outputs intentionally emulate the pre-2011 sign-extension bug — restrict
differential comparison to 7-bit passwords under `$2a$` and to `$2b$`/`$2y$ for 8-bit inputs.

**Best test-vector sources** (in order of value):

1. `crypt_blowfish-1.3/wrapper.c` `tests[][3]` — 28 positive vectors: the only authoritative
   source covering 8-bit/0xFF passwords, exactly-72-byte and >72-byte passwords, empty
   password, and all four prefixes, plus invalid-setting error cases. Public domain.
2. jBCrypt `TestBCrypt.java` (`test_vectors[][]`) — 20 clean (password, settings, hash)
   triples at costs 06/08/10/12, including 4 empty-password vectors. ISC-style license.
   URL: https://raw.githubusercontent.com/djmdjm/jBCrypt/master/test/org/mindrot/jbcrypt/TestBCrypt.java
3. `bcrypt` crate `src/lib.rs` tests — 8 cross-implementation vectors (pyca/node/Go/online)
   validating interop, plus the fixed-salt `hash_with_salt` vector and zero-salt
   version-formatting vectors; `src/bcrypt.rs` has 3 raw byte-level vectors for testing the
   core without base64.
4. For UTF-8 (no authoritative fixed vector exists anywhere): the two pyca×Rust
   cross-verified vectors in §6 rows 40–41 are reproducible with
   `python3 -c 'import bcrypt; print(bcrypt.hashpw("★"*8 …encode(), b"$2a$05$......................"))'`.
