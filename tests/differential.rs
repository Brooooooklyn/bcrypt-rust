//! Randomised differential testing of this crate against the vendored
//! reference C implementation in `crypt_blowfish/` (Openwall crypt_blowfish
//! 1.3, public domain — the implementation OpenBSD's bcrypt format is judged
//! against, self-tests and all).
//!
//! # How the C side is driven
//!
//! **This test shells out; it does not link.** Linking would need a
//! `build.rs` emitting `cargo:rustc-link-search` plus a feature to gate it,
//! and neither file belongs to this test's owner. Instead:
//!
//! 1. [`HARNESS_C`] below is written to `target/differential-harness/harness.c`.
//! 2. A C compiler (`$CC`, else the first of `cc`, `gcc`, `clang` that runs)
//!    compiles it together with `crypt_blowfish.c`, `crypt_gensalt.c` and
//!    `wrapper.c` — the portable half of the vendored Makefile's `CRYPT_OBJS`
//!    (`x86.S` is `__i386__`-only; `BF_ASM` is 0 on x86_64 and aarch64). Each
//!    object and the final binary are rebuilt only when newer-than checks say
//!    the inputs changed. `wrapper.c` is what defines `crypt_rn`, the entry
//!    point the harness calls — hashing through `crypt_blowfish.o` alone would
//!    mean calling the private `_crypt_blowfish_rn` instead.
//! 3. The harness speaks a line protocol on stdin/stdout, so one process
//!    serves a whole batch. Spawning ~350 processes would dominate the
//!    runtime; spawning one per batch does not.
//!
//! If no C compiler can be spawned, every test in this file prints a skip
//! line and passes (a Windows or minimal container dev box must not go red —
//! CI's `windows` and `wasm` legs never select this test) — unless
//! `BCRYPT_REQUIRE_C_HARNESS` is set, which turns even that case into a
//! panic naming the variable. Like `BCRYPT_REQUIRE_BACKEND` in
//! `tests/backends.rs`, the variable exists so CI legs with a guaranteed
//! compiler (the `test` and `asan` jobs set it) can never silently skip
//! the comparison. That no-compiler skip is the ONLY skip: a compiler that
//! exists but fails to build the vendored C, a failed mkdir/write, or a
//! harness that misbehaves at runtime panics — on the Linux/macOS CI legs
//! that run this test a broken C build is a real failure and must fail.
//!
//! # Wire protocol
//!
//! One request per line, fields separated by single spaces:
//!
//! ```text
//! HASH <setting> <pwd_hex>
//! ```
//!
//! `<setting>` is the 29-char `$2v$cc$<22 salt chars>` string (malformed
//! variants allowed — the error-parity batch exists for them), `<pwd_hex>`
//! is the lowercase hex of the password bytes, nonempty. The harness
//! hex-decodes the password, NUL-terminates it, and calls
//! `crypt_rn(pwd, setting, &data, sizeof data)` with a 61-byte output area
//! (`CRYPT_OUTPUT_SIZE`). The response is `OK <60-char hash string>` or
//! `ERR null` (`crypt_rn` returned `NULL`). There is no empty-password
//! spelling in v1 of the protocol; empty-password behaviour is pinned in
//! `tests/vectors.rs` instead.
//!
//! All bcrypt-base64 is on the Rust side: settings are built with
//! `__internal::encode_16`, the crate's own encoder, so the C only ever
//! validates settings by hashing with them — it never has to be trusted to
//! produce a setting. (The one exception is the salt-canonicalisation batch
//! below, which *deliberately* corrupts the spare low bits of the 22nd salt
//! char; it picks the sibling char from a private copy of the 64-char
//! alphabet, because the crate exposes the codec, not the alphabet.)
//!
//! # What is compared
//!
//! Per case, where the C hashed: (a) the full 60-char string, byte for byte;
//! (b) `verify(pwd, &c_hash) == Ok(true)`; (c) `verify(pwd, &tampered) ==
//! Ok(false)` for one deterministic single-char tamper inside the hash
//! payload's fully-significant bits; (d) the batch path — `bcrypt_many` and
//! `__internal::bcrypt_many_with_backend` — re-encoded per item, again byte
//! for byte. Where the C rejected (`NULL`): the Rust side must reject too
//! (`verify` / `HashParts::from_str` → `Err`). Accept/reject parity only —
//! the C has no error codes worth mirroring, just `NULL` and `errno`.
//!
//! (b) and (c) run under [`Driver::PublicApi`] only: `verify` always runs the
//! scalar kernel by design, so replaying it per driver would re-pay scalar
//! time to prove the same thing.
//!
//! # Drivers
//!
//! Each batch is answered by the C **once**, then replayed against every
//! driver this host can run: [`Driver::PublicApi`] (`hash_with_salt` +
//! `verify`), [`Driver::PublicBatch`] (`bcrypt_many`, the detected backend),
//! and [`Driver::Forced`] per `Backend::ALL` where `is_available()`. The
//! argon2-rust template this file mirrors has a pooled-arena driver instead
//! of `PublicBatch`; this crate has no arena to reuse, and `bcrypt_many` is
//! the public batch path comparison (d) names.
//!
//! # Coverage
//!
//! Counts below are per batch; the hashed cases are replayed against every
//! driver (4 on aarch64-apple-darwin: public-api, public-batch\[neon\],
//! forced\[scalar\], forced\[neon\]; more on x86_64 hosts with AVX), and the
//! public-api driver additionally runs the verify/tamper pair per hashed
//! case. Every batch asserts floors on both counts.
//!
//! | batch | cases | hashed | rejected |
//! |---|---|---|---|
//! | [`the_c_harness_is_really_running_bcrypt`] | 3 | 2 | 1 |
//! | [`length_boundary_grid`] | 60 | 60 | 0 |
//! | [`randomised_sweep`] | 256 | 256 | 0 |
//! | [`expensive_costs`] | 12 | 12 | 0 |
//! | [`error_parity_malformed_settings`] | 15 | 2 | 13 |
//! | [`salt_spelling_canonicalises_identically`] | 12 | 12 | 0 |
//! | [`the_comparison_catches_a_wrong_answer`] (baseline) | 2 | 1 | 1 |
//! | **default run** | **360** | **345** | **15** |
//!
//! # Documented divergences, excluded by construction
//!
//! * **NUL password bytes.** `crypt_rn` takes a NUL-terminated C string, so
//!   an interior `0x00` truncates the C's key stream; this crate's
//!   `padded_key` embeds it. Intentional semantic divergence — generated
//!   passwords map `0x00` to `0x01`.
//! * **`$2a$` with high-bit passwords.** crypt_blowfish's anti-collision
//!   "safety XOR" is a deliberate deviation this crate does not reproduce
//!   (pinned in `tests/vectors.rs`). Sweeps use `$2b$`/`$2y$` only; the one
//!   `$2a$` case is the ASCII `U*U` sanity vector, where the XOR cannot
//!   trigger.
//! * **`$2x$`.** The sign-extension bug emulation is likewise not
//!   reproduced, so no case uses it.
//! * **Costs 9..=31** appear nowhere as accept cases (runtime); cost 31 is
//!   not even hashed once. Rejection of out-of-range costs is checked
//!   *before* any hashing on both sides, which is what makes the
//!   error-parity batch free.
//!
//! # Determinism
//!
//! Every generated byte comes from a SplitMix64 seeded with [`SEED`] (with a
//! per-batch XOR salt). No `rand`, no clock, no thread-order dependence.
//! Failure messages carry the full case, so a failure reproduces by reading
//! it. Total runtime of `cargo test --test differential --release` is a few
//! seconds; the C is always compiled `-O2`, so the debug run's only slow
//! half is this crate itself.

#![cfg(all(feature = "alloc", feature = "internal-api"))]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use bcrypt_rust::__internal::{bcrypt_many_with_backend, decode_16, encode_16, encode_23};
use bcrypt_rust::{Backend, HashParts, Version, bcrypt_many, detected_backend, hash_with_salt, verify};

/// The one constant the whole sweep hangs off. Change it to explore a
/// different corner of the space; failures stay reproducible either way.
const SEED: u64 = 0x_BC27_2015_D1FF_E001;

// ---------------------------------------------------------------------------
// The C harness
// ---------------------------------------------------------------------------

/// Source of the C driver. Written to `target/` and compiled at test time.
///
/// Kept deliberately dumb: parse, call `crypt_rn`, print. The output area is
/// exactly `CRYPT_OUTPUT_SIZE` (7 + 22 + 31 + 1 = 61) bytes, the size
/// `_crypt_blowfish_rn` requires; on rejection `crypt_rn` returns `NULL` and
/// the answer is `ERR null` — the `*0`/`*1` failure magic it leaves in the
/// buffer is deliberately not surfaced, so the Rust side can never be
/// compared against an artifact instead of a verdict.
const HARNESS_C: &str = r#"/* GENERATED by tests/differential.rs - do not edit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ow-crypt.h"

#define LINE_CAP 4096
#define OUT_SIZE (7 + 22 + 31 + 1)

static char line[LINE_CAP];

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(void) {
    while (fgets(line, (int)sizeof line, stdin) != NULL) {
        char *cmd, *setting, *pwdhex, *spare;
        char data[OUT_SIZE];
        unsigned char pwd[256 + 1];
        size_t n, i;
        char *out;

        if (strchr(line, '\n') == NULL && !feof(stdin)) {
            printf("BAD line-too-long\n");
            fflush(stdout);
            return 1;
        }

        cmd = strtok(line, " \t\r\n");
        if (cmd == NULL) continue; /* blank line */
        setting = strtok(NULL, " \t\r\n");
        pwdhex = strtok(NULL, " \t\r\n");
        spare = strtok(NULL, " \t\r\n");
        if (strcmp(cmd, "HASH") != 0 || setting == NULL || pwdhex == NULL ||
            spare != NULL) {
            printf("BAD request\n");
            fflush(stdout);
            return 1;
        }

        n = strlen(pwdhex);
        if (n == 0 || n % 2 != 0 || n / 2 > 256) {
            printf("BAD pwd-hex\n");
            fflush(stdout);
            return 1;
        }
        for (i = 0; i < n / 2; ++i) {
            int hi = hexval((unsigned char)pwdhex[2 * i]);
            int lo = hexval((unsigned char)pwdhex[2 * i + 1]);
            if (hi < 0 || lo < 0) {
                printf("BAD pwd-hex\n");
                fflush(stdout);
                return 1;
            }
            pwd[i] = (unsigned char)((hi << 4) | lo);
        }
        pwd[n / 2] = '\0'; /* crypt_rn takes a NUL-terminated C string */

        out = crypt_rn((const char *)pwd, setting, data, (int)sizeof data);
        if (out != NULL)
            printf("OK %s\n", out);
        else
            printf("ERR null\n");
        fflush(stdout);
    }
    return 0;
}
"#;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The vendored C files the harness needs, relative to `crypt_blowfish/`.
/// `crypt_rn` lives in `wrapper.c`; it in turn references the gensalt half,
/// hence all three.
const C_SOURCES: [&str; 3] = ["crypt_blowfish.c", "crypt_gensalt.c", "wrapper.c"];
/// Headers whose mtime should force a rebuild. (`crypt.h` is used only by the
/// glibc-in-tree build, which this never is.)
const C_HEADERS: [&str; 3] = ["crypt_blowfish.h", "crypt_gensalt.h", "ow-crypt.h"];

/// The only failure [`build_harness`] may return: no C compiler exists on
/// this host. Every other failure — a missing vendored tree, a failed
/// mkdir/write, a compile or link error — is a real problem with the test
/// setup, not an environment to tolerate, so it panics with the details
/// instead of propagating.
struct NoCompiler;

impl fmt::Display for NoCompiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("none of $CC/cc/gcc/clang answered `--version`")
    }
}

/// Build (once per process) and return the harness executable, or `None`
/// when no C compiler exists on this host. A missing vendored tree or a
/// compiler that runs but fails is a panic, not a skip: CI's `test` and
/// `asan` legs run this file exactly where `cc` is guaranteed, and they set
/// `BCRYPT_REQUIRE_C_HARNESS=1`, which turns even the no-compiler case into
/// a panic — there, a skip can only mean a broken runner. The env check
/// lives here, outside the `OnceLock`, so it is re-read on every call
/// rather than frozen by the cached build result.
fn harness() -> Option<&'static Path> {
    static BUILT: OnceLock<Result<PathBuf, NoCompiler>> = OnceLock::new();
    match BUILT.get_or_init(build_harness) {
        Ok(path) => Some(path.as_path()),
        Err(why) => {
            assert!(
                std::env::var_os("BCRYPT_REQUIRE_C_HARNESS").is_none(),
                "differential: BCRYPT_REQUIRE_C_HARNESS is set but no usable C compiler: {why}"
            );
            eprintln!("differential: skipping, no usable C compiler: {why}");
            None
        }
    }
}

/// The first compiler of `$CC` (if set) or `cc`/`gcc`/`clang` that answers
/// `--version` successfully. Requiring a successful exit, not just a
/// successful spawn, keeps MSVC's `cl.exe` (which errors on `--version`)
/// from being selected and then failing the real compile.
fn find_compiler() -> Option<String> {
    let candidates: Vec<String> = match std::env::var("CC") {
        Ok(cc) => vec![cc],
        Err(_) => ["cc", "gcc", "clang"].map(str::to_string).to_vec(),
    };
    candidates.into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

fn run(what: &str, command: &mut Command) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|e| format!("{what}: failed to spawn: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{what}: exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    ))
}

fn build_harness() -> Result<PathBuf, NoCompiler> {
    let root = manifest_dir();
    let c_dir = root.join("crypt_blowfish");
    for file in C_SOURCES.iter().chain(C_HEADERS.iter()) {
        assert!(
            c_dir.join(file).is_file(),
            "the vendored crypt_blowfish tree is missing {file} at {}",
            c_dir.display()
        );
    }
    // The only tolerated absence — everything below this point panics on
    // failure; see `NoCompiler`.
    let Some(cc) = find_compiler() else {
        return Err(NoCompiler);
    };

    let build_dir = root.join("target").join("differential-harness");
    fs::create_dir_all(&build_dir)
        .unwrap_or_else(|e| panic!("mkdir {}: {e}", build_dir.display()));

    let source = build_dir.join("harness.c");
    let stale = fs::read_to_string(&source)
        .map(|s| s != HARNESS_C)
        .unwrap_or(true);
    if stale {
        fs::write(&source, HARNESS_C)
            .unwrap_or_else(|e| panic!("write {}: {e}", source.display()));
    }

    // One object per vendored source, compiled next to the harness (never
    // into the vendored tree), each only when stale.
    let mut objects = Vec::new();
    for c in C_SOURCES {
        let src = c_dir.join(c);
        let obj = build_dir.join(c.replace(".c", ".o"));
        let headers: Vec<PathBuf> = C_HEADERS.iter().map(|h| c_dir.join(h)).collect();
        let mut inputs: Vec<&Path> = vec![src.as_path()];
        inputs.extend(headers.iter().map(PathBuf::as_path));
        if needs_rebuild(&obj, &inputs) {
            run(
                &format!("{cc} -c {c}"),
                Command::new(&cc)
                    .arg("-O2")
                    .arg("-Wall")
                    .arg("-c")
                    .arg(&src)
                    .arg("-o")
                    .arg(&obj),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }
        objects.push(obj);
    }

    let exe = build_dir.join(format!("harness{}", std::env::consts::EXE_SUFFIX));
    let mut inputs: Vec<&Path> = vec![source.as_path()];
    inputs.extend(objects.iter().map(PathBuf::as_path));
    if !needs_rebuild(&exe, &inputs) {
        return Ok(exe);
    }
    let mut link = Command::new(&cc);
    link.arg("-O2")
        .arg("-Wall")
        .arg("-I")
        .arg(&c_dir)
        .arg(&source);
    for obj in &objects {
        link.arg(obj);
    }
    link.arg("-o").arg(&exe);
    run(&format!("{cc} harness.c *.o"), &mut link).unwrap_or_else(|e| panic!("{e}"));

    Ok(exe)
}

fn needs_rebuild(target: &Path, inputs: &[&Path]) -> bool {
    let Ok(target_time) = fs::metadata(target).and_then(|m| m.modified()) else {
        return true;
    };
    inputs.iter().any(|input| {
        fs::metadata(input)
            .and_then(|m| m.modified())
            .map(|t| t > target_time)
            .unwrap_or(true)
    })
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// What both sides are expected to do with a case.
#[derive(Clone, PartialEq, Eq)]
enum CaseKind {
    /// Both sides hash; the strings are compared byte for byte. `version` is
    /// only ever `TwoB`/`TwoY` except for the ASCII `$2a$` sanity vector.
    Hash {
        version: Version,
        cost: u32,
        salt: [u8; 16],
    },
    /// Both sides reject: the C answers `ERR null` for `setting`, and the
    /// crate answers `Err` for `rust_input`. Accept/reject parity only.
    Reject { rust_input: String },
}

#[derive(Clone, PartialEq, Eq)]
struct Case {
    /// Sent to the C verbatim. The canonical 29-char setting, except in the
    /// salt-canonicalisation batch, whose 22nd salt char is deliberately
    /// non-canonical (both sides must still emit the canonical spelling).
    setting: String,
    pwd: Vec<u8>,
    kind: CaseKind,
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "setting {:?} pwd[{}] {}", self.setting, self.pwd.len(), hex(&self.pwd))
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The canonical 29-char setting for these parts, built with the crate's
/// own encoder — the C never has to be trusted to produce a setting.
fn canonical_setting(version: Version, cost: u32, salt: &[u8; 16]) -> String {
    let salt_b64 = encode_16(salt);
    format!(
        "${version}${cost:02}${}",
        std::str::from_utf8(&salt_b64).expect("bcrypt base64 is ASCII")
    )
}

impl Case {
    /// A case both sides must hash, with the canonical setting.
    fn hash(version: Version, cost: u32, salt: [u8; 16], pwd: Vec<u8>) -> Case {
        Case {
            setting: canonical_setting(version, cost, &salt),
            pwd,
            kind: CaseKind::Hash { version, cost, salt },
        }
    }

    /// A case both sides must reject. `setting` goes to the C; `rust_input`
    /// is the (usually 60-char) candidate the crate's parser must refuse.
    fn reject(setting: &str, rust_input: &str, pwd: Vec<u8>) -> Case {
        Case {
            setting: setting.to_string(),
            pwd,
            kind: CaseKind::Reject {
                rust_input: rust_input.to_string(),
            },
        }
    }

    /// One request line for the harness.
    fn request(&self) -> String {
        format!("HASH {} {}\n", self.setting, hex(&self.pwd))
    }

    /// The identity of a case for de-duplication purposes: what the C sees.
    fn key(&self) -> String {
        self.request()
    }

    /// The 60-char string the C must print for a `Hash` case with raw
    /// ciphertext `out` — and what every Rust driver must produce. Built
    /// from the *parts*, so a non-canonical `setting` still expects the
    /// canonical spelling in the output.
    fn expected(&self, out: &[u8; 24]) -> String {
        let CaseKind::Hash { version, cost, salt } = &self.kind else {
            unreachable!("expected() is only called on hashing cases")
        };
        let mut payload = [0u8; 23];
        payload.copy_from_slice(&out[..23]);
        let salt_b64 = encode_16(salt);
        let hash_b64 = encode_23(&payload);
        format!(
            "${version}${cost:02}${}{}",
            std::str::from_utf8(&salt_b64).expect("bcrypt base64 is ASCII"),
            std::str::from_utf8(&hash_b64).expect("bcrypt base64 is ASCII"),
        )
    }
}

/// One deterministic single-char tamper of a 60-char hash string, inside
/// the hash payload's first 30 chars. Positions 0..=29 of the 31-char field
/// are fully significant (the 31st char carries 2 spare low bits the
/// decoder discards, and a salt-char flip can likewise decode away), so an
/// alphabet swap there must change the decoded payload — and only it.
fn tampered(c_hash: &str) -> String {
    let bytes = c_hash.as_bytes();
    assert_eq!(bytes.len(), 60, "tampered() expects a full hash string");
    let pos = 29 + bytes.iter().map(|&b| usize::from(b)).sum::<usize>() % 30;
    let mut out = bytes.to_vec();
    out[pos] = if out[pos] == b'A' { b'B' } else { b'A' };
    String::from_utf8(out).expect("the tamper writes ASCII")
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Which Rust entry point produces the answer for a case.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Driver {
    /// `hash_with_salt` + `verify` — the single-hash public API, which by
    /// design always runs the scalar kernel. This driver also owns
    /// comparisons (b) and (c): `verify` is driver-independent, so checking
    /// it once here instead of per driver saves two scalar hashes per case
    /// per extra driver.
    PublicApi,
    /// `bcrypt_many` — the public batch API on whatever backend runtime
    /// detection picked, re-encoded per item and compared byte for byte.
    PublicBatch,
    /// `__internal::bcrypt_many_with_backend` — one specific backend,
    /// detection bypassed, so the lanes a batch normally never uses get
    /// swept too.
    Forced(Backend),
}

impl fmt::Debug for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Driver::PublicApi => write!(f, "public-api"),
            Driver::PublicBatch => write!(f, "public-batch[{}]", detected_backend()),
            Driver::Forced(backend) => write!(f, "forced[{backend}]"),
        }
    }
}

/// Both public paths plus every backend this CPU can execute. Driven off
/// `Backend::is_available`, so it widens by itself on a host with AVX2 or
/// AVX-512 and never returns one that would fault.
fn drivers() -> Vec<Driver> {
    let mut drivers = vec![Driver::PublicApi, Driver::PublicBatch];
    drivers.extend(
        Backend::ALL
            .iter()
            .copied()
            .filter(|b| b.is_available())
            .map(Driver::Forced),
    );
    drivers
}

/// What the C harness answered.
#[derive(Debug, PartialEq, Eq)]
enum CResult {
    Hashed(String),
    Rejected,
}

/// Feed a whole batch of cases through one harness process.
///
/// A dedicated writer thread pushes stdin while this thread drains stdout:
/// a batch is far larger than a pipe buffer, so writing everything first
/// would deadlock the moment the child's replies filled its own pipe.
fn run_c(cases: &[Case]) -> Vec<CResult> {
    let exe = harness().expect("run_c called without a harness");

    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot spawn {}: {e}", exe.display()));

    let mut stdin = child.stdin.take().expect("piped stdin");
    let requests: Vec<String> = cases.iter().map(Case::request).collect();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        for request in &requests {
            stdin.write_all(request.as_bytes())?;
        }
        stdin.flush()
        // `stdin` drops here, closing the pipe so the harness sees EOF.
    });

    let stdout = child.stdout.take().expect("piped stdout");
    let mut results = Vec::with_capacity(cases.len());
    for (index, line) in BufReader::new(stdout).lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("reading harness stdout: {e}"));
        let case = cases.get(index);
        results.push(parse_response(&line, case));
    }

    writer
        .join()
        .unwrap_or_else(|_| panic!("the harness stdin writer thread panicked"))
        .unwrap_or_else(|e| panic!("writing to the harness: {e}"));

    let status = child.wait().expect("waiting for the harness");
    assert!(status.success(), "the C harness exited with {status}");
    assert_eq!(
        results.len(),
        cases.len(),
        "the C harness answered {} of {} cases",
        results.len(),
        cases.len()
    );

    results
}

fn parse_response(line: &str, case: Option<&Case>) -> CResult {
    if let Some(hash) = line.strip_prefix("OK ") {
        assert_eq!(
            hash.len(),
            60,
            "the harness returned a malformed hash {hash:?} for {case:?}"
        );
        CResult::Hashed(hash.to_string())
    } else if line == "ERR null" {
        CResult::Rejected
    } else {
        panic!("unexpected harness output {line:?} for {case:?}");
    }
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Run a batch through both implementations and assert full agreement.
///
/// Returns `(hashed, rejected)`: how many cases produced an identical hash
/// string on both sides and how many produced a matching rejection. Callers
/// assert floors on those counts so a batch that silently degenerated into
/// all-rejections (or all-accepts) fails instead of proving less than its
/// case count suggests.
fn assert_batch_agrees(label: &str, cases: &[Case]) -> (usize, usize) {
    let c_results = run_c(cases);
    compare(label, cases, &c_results)
}

/// The assertion half of [`assert_batch_agrees`], split out so
/// [`the_comparison_catches_a_wrong_answer`] can feed it deliberately
/// corrupted C answers and prove it fails.
///
/// Replays the batch once per [`Driver`]. Every driver must agree with the
/// C *and* produce the same `(hashed, rejected)` split — if two backends
/// disagreed about which cases hash, one of them is wrong even if both
/// matched some string.
fn compare(label: &str, cases: &[Case], c_results: &[CResult]) -> (usize, usize) {
    let mut agreed: Option<(Driver, (usize, usize))> = None;

    for driver in drivers() {
        let counts = compare_with(label, driver, cases, c_results);
        match agreed {
            None => agreed = Some((driver, counts)),
            Some((first, expected)) => assert_eq!(
                counts, expected,
                "{label}: {driver:?} split the batch {counts:?} but {first:?} split it {expected:?}"
            ),
        }
    }

    agreed.expect("drivers() always contains the public API").1
}

/// `Driver::PublicApi`'s per-case answer: the single-hash public path,
/// formatting through [`HashParts`] exactly as a real caller would.
fn single_answer(case: &Case) -> String {
    let CaseKind::Hash { version, cost, salt } = &case.kind else {
        unreachable!("single_answer is only called on hashing cases")
    };
    hash_with_salt(&case.pwd, *cost, *salt)
        .unwrap_or_else(|e| panic!("hash_with_salt failed for {case:?}: {e:?}"))
        .format_for_version(*version)
}

/// Whole-batch answers for the batch drivers: `Hash` cases grouped by cost
/// (`bcrypt_many*` takes one cost per call), one kernel call per group,
/// each 24-byte lane output re-encoded to its expected 60-char string.
/// `None` at `Reject` positions, and everywhere for `Driver::PublicApi`.
fn batch_answers(driver: Driver, cases: &[Case]) -> Vec<Option<String>> {
    let mut answers = vec![None; cases.len()];
    if driver == Driver::PublicApi {
        return answers;
    }
    let mut by_cost: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        if let CaseKind::Hash { cost, .. } = &case.kind {
            by_cost.entry(*cost).or_default().push(index);
        }
    }
    for (cost, idxs) in by_cost {
        let passwords: Vec<&[u8]> = idxs.iter().map(|&i| cases[i].pwd.as_slice()).collect();
        let salts: Vec<[u8; 16]> = idxs
            .iter()
            .map(|&i| match cases[i].kind {
                CaseKind::Hash { salt, .. } => salt,
                CaseKind::Reject { .. } => unreachable!("grouped on the Hash kind"),
            })
            .collect();
        let outs = match driver {
            Driver::PublicBatch => bcrypt_many(cost, &passwords, &salts),
            Driver::Forced(backend) => {
                // SAFETY: `drivers()` builds its list from
                // `Backend::is_available`, so every backend reaching here
                // is runnable on this CPU — the function's whole contract.
                unsafe { bcrypt_many_with_backend(backend, cost, &passwords, &salts) }
            }
            Driver::PublicApi => unreachable!("returned early above"),
        }
        .unwrap_or_else(|e| panic!("batch driver {driver:?} failed at cost {cost}: {e:?}"));
        assert_eq!(outs.len(), idxs.len(), "batch outputs are position-preserving");
        for (j, &i) in idxs.iter().enumerate() {
            answers[i] = Some(cases[i].expected(&outs[j]));
        }
    }
    answers
}

/// One pass of [`compare`] on a single [`Driver`].
fn compare_with(
    label: &str,
    driver: Driver,
    cases: &[Case],
    c_results: &[CResult],
) -> (usize, usize) {
    assert!(!cases.is_empty(), "{label}: empty batch");
    assert_eq!(cases.len(), c_results.len());

    let unique: BTreeSet<String> = cases.iter().map(Case::key).collect();
    let answers = batch_answers(driver, cases);
    let mut hashed = 0usize;
    let mut rejected = 0usize;

    for (index, (case, c)) in cases.iter().zip(c_results).enumerate() {
        match (&case.kind, c) {
            (CaseKind::Hash { .. }, CResult::Hashed(c_hash)) => {
                hashed += 1;
                let rust = match driver {
                    Driver::PublicApi => single_answer(case),
                    _ => answers[index]
                        .clone()
                        .expect("batch answers cover every hashing case"),
                };
                if *c_hash != rust {
                    let first = c_hash
                        .bytes()
                        .zip(rust.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    panic!(
                        "\n{label}[{index}] {driver:?}: HASH MISMATCH (seed {SEED:#018x})\n\
                         \x20 case: {case:?}\n\
                         \x20 C:    {c_hash}\n\
                         \x20 Rust: {rust}\n\
                         \x20 first differing char: index {first} \
                         (C {:?} vs Rust {:?})\n\n\
                         This is a real bug in the port. Fix it; do not weaken this test.\n",
                        c_hash.as_bytes()[first] as char,
                        rust.as_bytes()[first] as char,
                    );
                }
                // (b) and (c): driver-independent (verify is always scalar),
                // so they run under the public API only — see the header.
                if driver == Driver::PublicApi {
                    assert_eq!(
                        verify(&case.pwd, c_hash),
                        Ok(true),
                        "{label}[{index}]: the C's own hash must verify (seed {SEED:#018x})\n\
                         \x20 case: {case:?}\n\x20 C: {c_hash}"
                    );
                    let bad = tampered(c_hash);
                    assert_eq!(
                        verify(&case.pwd, &bad),
                        Ok(false),
                        "{label}[{index}]: a tampered hash must not verify (seed {SEED:#018x})\n\
                         \x20 case: {case:?}\n\x20 C: {c_hash}\n\x20 tampered: {bad}"
                    );
                }
            }
            (CaseKind::Reject { rust_input }, CResult::Rejected) => {
                rejected += 1;
                let accepted = match driver {
                    Driver::PublicApi => verify(&case.pwd, rust_input).is_ok(),
                    _ => rust_input.parse::<HashParts>().is_ok(),
                };
                assert!(
                    !accepted,
                    "\n{label}[{index}] {driver:?}: Rust accepted what the C rejected \
                     (seed {SEED:#018x})\n\x20 case: {case:?}\n\x20 rust input: {rust_input:?}"
                );
            }
            (CaseKind::Hash { .. }, CResult::Rejected) => panic!(
                "\n{label}[{index}] {driver:?}: the C rejected a well-formed case \
                 (seed {SEED:#018x})\n\x20 case: {case:?}"
            ),
            (CaseKind::Reject { rust_input }, CResult::Hashed(c_hash)) => panic!(
                "\n{label}[{index}] {driver:?}: the C hashed what Rust must reject \
                 (seed {SEED:#018x})\n\x20 case: {case:?}\n\x20 rust input: {rust_input:?}\n\
                 \x20 C: {c_hash}"
            ),
        }
    }

    println!(
        "{label} {driver:?}: {} cases ({} distinct) -> {hashed} hashed identically, \
         {rejected} rejected in agreement",
        cases.len(),
        unique.len(),
    );
    (hashed, rejected)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

/// SplitMix64. Small, deterministic, dependency-free — the sweep needs
/// reproducible bytes, not statistical quality.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
        choices[self.below(choices.len())]
    }

    /// Random password bytes with `0x00` mapped to `0x01`: `crypt_rn` takes
    /// a NUL-terminated C string, so an interior NUL is a documented
    /// divergence (the C truncates, the crate embeds), excluded here on
    /// purpose. High bytes `0x80..=0xFF` stay — `$2b$`/`$2y$` settings must
    /// match the C byte for byte exactly there.
    fn password(&mut self, len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| match self.next_u64() as u8 {
                0x00 => 0x01,
                b => b,
            })
            .collect()
    }

    /// Salt bytes are unconstrained: the crate's own encoder turns any 16
    /// bytes into a canonical setting.
    fn salt(&mut self) -> [u8; 16] {
        let mut salt = [0u8; 16];
        let (a, b) = salt.split_at_mut(8);
        a.copy_from_slice(&self.next_u64().to_le_bytes());
        b.copy_from_slice(&self.next_u64().to_le_bytes());
        salt
    }
}

// ---------------------------------------------------------------------------
// The domains the sweep draws from
// ---------------------------------------------------------------------------

/// Password lengths straddling the 72-byte truncation boundary (71 = NUL
/// terminator in the last key byte, 72 = buffer full, 73 = first truncated
/// length), plus small, mid and well-past-truncation sizes.
const PWD_LENS: [usize; 10] = [1, 5, 16, 55, 56, 71, 72, 73, 100, 255];

/// Sweep prefixes. `$2y$` is byte-identical to `$2b$` in crypt_blowfish 1.3;
/// `$2a$` is excluded (its high-bit safety-XOR divergence is pinned in
/// `tests/vectors.rs`), `$2x$` likewise.
const PREFIXES: [Version; 2] = [Version::TwoB, Version::TwoY];

// ---------------------------------------------------------------------------
// Test 0: the harness must prove it is bcrypt before anything trusts it
// ---------------------------------------------------------------------------

/// Sanity: the harness really runs bcrypt and really reports rejections.
///
/// If `run_c` silently echoed whatever Rust computed, every test below would
/// pass vacuously. So the first two cases are the OpenBSD `U*U` vector from
/// `tests/vectors.rs` (#21) under `$2a$` and `$2b$`, with the expected
/// strings hard-coded here from the vector file — not derived from anything
/// the crate computes. `$2a$` is safe for exactly this password: the safety
/// XOR only triggers on high-bit bytes, and `U*U` is ASCII. The third case
/// must come back `ERR null`.
#[test]
fn the_c_harness_is_really_running_bcrypt() {
    let Some(_) = harness() else { return };
    let salt = decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("vector salt decodes");
    let cases = [
        Case::hash(Version::TwoA, 5, salt, b"U*U".to_vec()),
        Case::hash(Version::TwoB, 5, salt, b"U*U".to_vec()),
        Case::reject(
            "$2c$05$CCCCCCCCCCCCCCCCCCCCC.",
            "$2c$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
            b"U*U".to_vec(),
        ),
    ];

    let results = run_c(&cases);
    assert_eq!(
        results[0],
        CResult::Hashed("$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW".to_string()),
        "the harness did not reproduce OpenBSD vector #21 under $2a$"
    );
    assert_eq!(
        results[1],
        CResult::Hashed("$2b$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW".to_string()),
        "the harness did not reproduce OpenBSD vector #21 under $2b$"
    );
    assert_eq!(results[2], CResult::Rejected, "$2c$ must be rejected");

    // And the comparison plumbing agrees with the crate on all three.
    assert_batch_agrees("harness-sanity", &cases);
}

// ---------------------------------------------------------------------------
// Test 1: the password-length boundary, crossed with prefix and cost
// ---------------------------------------------------------------------------

/// Every length in [`PWD_LENS`] crossed with both prefixes and costs 4..=6.
/// The lengths are the point: 71/72/73 decide whether the appended NUL joins
/// the key stream, and the C's `BF_set_key` cycling is the reference for the
/// crate's `padded_key` cycling.
#[test]
fn length_boundary_grid() {
    let Some(_) = harness() else { return };
    let mut rng = Rng::new(SEED ^ 0x01);
    let mut cases = Vec::new();
    for &len in &PWD_LENS {
        for version in PREFIXES {
            for cost in 4..=6u32 {
                cases.push(Case::hash(version, cost, rng.salt(), rng.password(len)));
            }
        }
    }

    let (hashed, rejected) = assert_batch_agrees("length-boundary", &cases);
    assert_eq!(rejected, 0, "every case here is deliberately well-formed");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Test 2: randomised sweep
// ---------------------------------------------------------------------------

/// 256 configurations drawn from every domain at once: prefix, cost
/// (weighted to 4/5, some 6), length (half from the boundary list, half
/// uniform in 1..=255), and full-range salt and password bytes (NUL excepted,
/// high bytes included).
#[test]
fn randomised_sweep() {
    let Some(_) = harness() else { return };
    let mut rng = Rng::new(SEED ^ 0x02);
    let mut cases = Vec::with_capacity(256);
    for _ in 0..256 {
        let version = rng.pick(&PREFIXES);
        let cost = rng.pick(&[4, 4, 4, 5, 5, 6]);
        let len = if rng.below(2) == 0 {
            rng.pick(&PWD_LENS)
        } else {
            1 + rng.below(255)
        };
        cases.push(Case::hash(version, cost, rng.salt(), rng.password(len)));
    }

    let (hashed, rejected) = assert_batch_agrees("randomised", &cases);
    assert_eq!(rejected, 0, "every case here is deliberately well-formed");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Test 3: expensive costs
// ---------------------------------------------------------------------------

/// A small batch at costs 7 and 8: high enough that a subtle key-schedule
/// bug cannot hide behind "the cheap cases all passed", cheap enough that
/// the whole file stays fast. (Costs 9+ would buy the same signal at 4× the
/// price each step.)
#[test]
fn expensive_costs() {
    let Some(_) = harness() else { return };
    let mut rng = Rng::new(SEED ^ 0x03);
    let lens = [1usize, 72, 73, 255];
    let mut cases = Vec::new();
    for i in 0..8 {
        let version = PREFIXES[i % 2];
        cases.push(Case::hash(version, 7, rng.salt(), rng.password(lens[i % 4])));
    }
    for i in 0..4 {
        let version = PREFIXES[i % 2];
        cases.push(Case::hash(version, 8, rng.salt(), rng.password(lens[(i + 1) % 4])));
    }

    let (hashed, rejected) = assert_batch_agrees("expensive", &cases);
    assert_eq!(rejected, 0, "every case here is deliberately well-formed");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Test 4: error parity on malformed settings
// ---------------------------------------------------------------------------

/// Malformed settings: the C must return `NULL` and the crate must return
/// `Err` — accept/reject parity only, never messages. The C side validates
/// prefix, cost and salt before hashing (and this crate's parser validates
/// before any cost is used), so even the "cost 33" cases cost nothing.
///
/// Each `reject` pairs the exact 29-or-fewer-char setting the C sees with a
/// full 60-char candidate for the crate (59 for the truncated salt, where
/// length itself is the defect), so the crate is always judging the same
/// component the C judged.
#[test]
fn error_parity_malformed_settings() {
    let Some(_) = harness() else { return };
    let mut rng = Rng::new(SEED ^ 0x04);
    // A valid 31-char hash field (OpenBSD vector #21's), so only the part
    // under test is malformed.
    const TAIL: &str = "E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW";
    let pwd = rng.password(17);

    let settings = [
        // Costs outside 4..=31: too small, too big, syntactically impossible.
        "$2b$03$CCCCCCCCCCCCCCCCCCCCC.",
        "$2b$32$CCCCCCCCCCCCCCCCCCCCC.",
        "$2b$33$CCCCCCCCCCCCCCCCCCCCC.",
        "$2b$0a$CCCCCCCCCCCCCCCCCCCCC.", // non-digit cost
        // Salt: one char short, then a char outside the bcrypt alphabet in
        // the last and first positions.
        "$2b$05$CCCCCCCCCCCCCCCCCCCCC", // truncated: 21 chars
        "$2b$05$CCCCCCCCCCCCCCCCCCCC!",
        "$2b$05$!CCCCCCCCCCCCCCCCCCCCC",
        // Prefixes the C does not know: $2c$, $2z$ (flag-less), $2A$
        // (uppercase), $1$ (md5-crypt's), a missing cost separator, and the
        // failure-magic string fed back as a setting.
        "$2c$05$CCCCCCCCCCCCCCCCCCCCC.",
        "$2z$05$CCCCCCCCCCCCCCCCCCCCC.",
        "$2A$05$CCCCCCCCCCCCCCCCCCCCC.",
        "$1$05$CCCCCCCCCCCCCCCCCCCCC.",
        "$2b$05XCCCCCCCCCCCCCCCCCCCCC.",
        "*0",
    ];
    let mut cases: Vec<Case> = settings
        .iter()
        .map(|&setting| {
            // The truncated salt stays 59 chars — its shortness is the defect.
            Case::reject(setting, &format!("{setting}{TAIL}"), pwd.clone())
        })
        .collect();

    // The valid side of the boundary must still hash: cost 4 is the minimum
    // on both sides, under both prefixes.
    cases.push(Case::hash(Version::TwoB, 4, rng.salt(), rng.password(9)));
    cases.push(Case::hash(Version::TwoY, 4, rng.salt(), rng.password(72)));

    let (hashed, rejected) = assert_batch_agrees("error-parity", &cases);
    assert!(
        rejected >= 12,
        "only {rejected} of {} cases were rejected in agreement",
        cases.len()
    );
    assert!(
        hashed >= 2,
        "only {hashed} boundary cases hashed; the valid side is untested"
    );
}

// ---------------------------------------------------------------------------
// Test 5: non-canonical salt spellings must canonicalise identically
// ---------------------------------------------------------------------------

/// The bcrypt alphabet, copied from `src/base64.rs` — NOT a second encoder.
/// All encoding above goes through `__internal::encode_16`; this copy exists
/// only to *construct adversarial settings*: pick a char with the same top-2
/// bits as the canonical 22nd salt char but different spare low bits. A
/// wrong copy would be caught immediately, because the C validates the
/// spelling against its own `itoa64`.
const ALPHABET: &[u8; 64] =
    b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// The 22nd salt char carries only the top 2 of its 6 bits (16 bytes = 21⅓
/// groups). The C decodes it, then re-emits the *canonical* char on output
/// (`BF_crypt` re-encodes with `& 0x30`); this crate decodes the spare bits
/// away and re-encodes canonically too. So a non-canonical spelling must
/// produce the identical 60-char string on both sides — the one case where
/// the C's output is not a pure echo of its input, and therefore the
/// highest-risk byte in the whole comparison.
#[test]
fn salt_spelling_canonicalises_identically() {
    let Some(_) = harness() else { return };
    let mut rng = Rng::new(SEED ^ 0x05);
    let mut cases = Vec::new();
    for i in 0..12 {
        let version = PREFIXES[i % 2];
        let cost = [4u32, 5][i % 2];
        let salt = rng.salt();
        let len = rng.pick(&[1usize, 16, 72]);
        let pwd = rng.password(len);
        let mut case = Case::hash(version, cost, salt, pwd);

        let mut setting = case.setting.clone().into_bytes();
        let canonical = setting[28];
        let index = ALPHABET
            .iter()
            .position(|&c| c == canonical)
            .expect("the crate's encoder emits alphabet chars");
        // Same top-2-bits group (index & 0x30), a different low nibble.
        setting[28] = ALPHABET[(index & 0x30) + ((index + 1 + rng.below(15)) & 0x0f)];
        case.setting = String::from_utf8(setting).expect("alphabet chars are ASCII");

        // Self-check: the adversarial spelling really does decode to the
        // same 16 salt bytes, or the case would be testing something else.
        assert_eq!(decode_16(&case.setting.as_bytes()[7..29]), Ok(salt));
        cases.push(case);
    }

    let (hashed, rejected) = assert_batch_agrees("salt-spelling", &cases);
    assert_eq!(rejected, 0, "every case here is well-formed once decoded");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Meta-test: the comparison itself must be able to fail
// ---------------------------------------------------------------------------

/// Negative control for [`compare`]. Every "the strings matched" claim above
/// is only worth as much as the comparison's ability to notice when they do
/// not. Corrupt the C's answers — one flipped hash char, one accept/reject
/// swap in each direction — and check each is caught with a message that
/// names what went wrong.
#[test]
fn the_comparison_catches_a_wrong_answer() {
    let Some(_) = harness() else { return };
    let valid = Case::hash(Version::TwoB, 4, [0x42; 16], b"differential".to_vec());
    let invalid = Case::reject(
        "$2b$03$CCCCCCCCCCCCCCCCCCCCC.",
        "$2b$03$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW",
        b"differential".to_vec(),
    );

    let truth = run_c(&[valid.clone(), invalid.clone()]);
    let CResult::Hashed(good) = &truth[0] else {
        panic!("the valid case must hash on the C side, got {:?}", truth[0]);
    };
    assert_eq!(truth[1], CResult::Rejected);

    // Baseline: uncorrupted, this must pass.
    assert_eq!(
        compare("control-baseline", &[valid.clone(), invalid.clone()], &truth),
        (1, 1)
    );

    let expect_failure = |what: &str, cases: Vec<Case>, results: Vec<CResult>, needle: &str| {
        let payload = std::panic::catch_unwind(move || {
            compare("negative-control", &cases, &results);
        })
        .err()
        .unwrap_or_else(|| panic!("{what}: a corrupted C answer was accepted"));

        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("")
            .to_string();
        assert!(
            message.contains(needle),
            "{what}: the failure message should mention {needle:?}, got:\n{message}"
        );
    };

    // 1. One flipped char in the hash payload.
    expect_failure(
        "flipped hash char",
        vec![valid.clone()],
        vec![CResult::Hashed(tampered(good))],
        "HASH MISMATCH",
    );

    // 2. The C rejects what Rust hashes.
    expect_failure(
        "C rejected, Rust hashed",
        vec![valid.clone()],
        vec![CResult::Rejected],
        "the C rejected a well-formed case",
    );

    // 3. The C hashes what Rust rejects.
    expect_failure(
        "C hashed, Rust rejected",
        vec![invalid.clone()],
        vec![CResult::Hashed(good.clone())],
        "the C hashed what Rust must reject",
    );
}
