//! Stable CodSpeed benchmarks for the crate's major public APIs.
//!
//! ```text
//! cargo codspeed build --bench codspeed
//! cargo codspeed run --bench codspeed
//! ```
//!
//! This suite is a **regression net, not a throughput claim**. CodSpeed runs
//! it on every pull request under a CPU simulation instrument, and each
//! benchmark id becomes part of the long-term performance history; the
//! absolute numbers it reports are instruction-count-derived and say nothing
//! about wall-clock speed on real hardware. Publishable numbers come from
//! `benches/bcrypt.rs`, run on a named machine.
//!
//! Keep the suite deliberately small, hermetic and single-threaded:
//!
//! * **Small** — cost 4, the cheapest legal cost. The cost loop is `2^cost`
//!   identical iterations, so a regression at any higher cost is exactly
//!   `2^(cost-4)` times the regression measured here; nothing is gained by
//!   making a simulated CI run expensive.
//! * **Hermetic** — fixed passwords and salts, no OS entropy. `hash` (random
//!   salt) is therefore covered through its deterministic sibling
//!   `hash_with_salt`, which shares the whole pipeline past the salt draw.
//! * **Single-threaded** — CodSpeed's CPU simulation does not model
//!   wall-clock parallel speedup. The `parallel` feature is default-off, and
//!   a batch of 8 is below its `2 x cores` threshold anyway, so every
//!   measurement below is one thread on the detected backend.
//! * **One pinned backend** — CI runs this suite with
//!   `BCRYPT_FORCE_BACKEND=avx2`: the GitHub runner lottery mixes Intel
//!   (AVX-512) and AMD (AVX2) boxes, the simulation instrument implements
//!   no AVX-512, and one pinned backend keeps the performance history
//!   comparable across runners.

use std::time::Duration;

use bcrypt_rust::{Version, bcrypt, bcrypt_many, hash_with_salt, verify};
use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};

const PASSWORD: &[u8] = b"correct horse battery staple";
/// 16 bytes, the one salt length bcrypt has.
const SALT: [u8; 16] = *b"codspeed-salt-16";

/// The single-hash pipeline end to end: key preparation, the scalar kernel,
/// and the PHC-style `$2b$` formatting. One warm-up call runs before the
/// measured closure so nothing one-time is charged to it.
fn bench_single_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("public/single_hash");
    hash_with_salt(PASSWORD, 4, SALT).expect("warm-up hash");

    group.bench_function("hash_with_salt_cost4", |b| {
        b.iter(|| {
            black_box(hash_with_salt(black_box(PASSWORD), 4, black_box(SALT)).expect("hash"));
        });
    });

    group.finish();
}

/// The batch entry point a real caller takes: runtime-detected backend,
/// caller-supplied salts, 8 passwords. Eight keeps one simulated run cheap
/// while still exercising the lane-group loop (two NEON/SSE4.1 groups, one
/// AVX2 group, a short AVX-512 tail). The warm-up resolves runtime dispatch
/// before CodSpeed enters the measured closure.
fn bench_batch(c: &mut Criterion) {
    let store: [&[u8]; 8] = [
        b"alpha",
        b"bravo",
        b"charlie",
        b"delta",
        b"echo foxtrot",
        b"golf hotel india",
        b"juliett kilo lima mike",
        b"november oscar papa quebec",
    ];
    let salts: [[u8; 16]; 8] = core::array::from_fn(|i| {
        let mut salt = SALT;
        salt[0] = i as u8;
        salt
    });

    let mut group = c.benchmark_group("public/batch");
    bcrypt_many(4, &store, &salts).expect("warm-up batch");

    group.bench_function("bcrypt_many_n8_cost4", |b| {
        b.iter(|| {
            black_box(bcrypt_many(4, black_box(&store), black_box(&salts)).expect("batch"));
        });
    });

    group.finish();
}

/// The verifier end to end — the call consumers make most. Parses the
/// `$2b$` string into `HashParts`, decodes the embedded salt, runs the same
/// kernel as `hash_with_salt`, and constant-time-compares the result, so a
/// regression in any of those is attributed to this arm. The hash under
/// test is generated from the fixed salt at setup, keeping the suite
/// hermetic and the fixture in sync with the encoder.
fn bench_verify(c: &mut Criterion) {
    let stored = hash_with_salt(PASSWORD, 4, SALT)
        .expect("fixture hash")
        .format_for_version(Version::TwoB);
    verify(PASSWORD, &stored).expect("warm-up verify");

    let mut group = c.benchmark_group("public/verify");
    group.bench_function("verify_cost4", |b| {
        b.iter(|| {
            black_box(verify(black_box(PASSWORD), black_box(&stored)).expect("verify"));
        });
    });
    group.finish();
}

/// The raw core with no formatting and no batch machinery: the pure kernel
/// latency the two public arms above are built on. A divergence between
/// this arm and `hash_with_salt_cost4` isolates a regression to the codec.
fn bench_raw_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("core");

    group.bench_function("bcrypt_cost4", |b| {
        b.iter(|| {
            black_box(bcrypt(4, black_box(SALT), black_box(PASSWORD)));
        });
    });

    group.finish();
}

/// The dispatched base64 codec (`bench_raw_core` isolates a regression to
/// "the codec" — these arms name it). Uses `__internal` because the codec
/// is crate-private; what is timed is exactly what `hash`/`verify` run:
/// NEON on aarch64, the AVX2/SSSE3 pick on x86-64, v128 under wasm simd128.
fn bench_base64(c: &mut Criterion) {
    use bcrypt_rust::__internal::{decode_16, decode_23, encode_16, encode_23};

    let hash_bytes = [0xA5u8; 23];
    let enc16 = encode_16(&SALT);
    let enc23 = encode_23(&hash_bytes);
    let mut group = c.benchmark_group("base64");

    group.bench_function("encode_16", |b| {
        b.iter(|| black_box(encode_16(black_box(&SALT))));
    });
    group.bench_function("decode_16", |b| {
        b.iter(|| black_box(decode_16(black_box(&enc16)).expect("decodes")));
    });
    group.bench_function("encode_23", |b| {
        b.iter(|| black_box(encode_23(black_box(&hash_bytes))));
    });
    group.bench_function("decode_23", |b| {
        b.iter(|| black_box(decode_23(black_box(&enc23)).expect("decodes")));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_single_hash, bench_verify, bench_batch, bench_raw_core, bench_base64
}
criterion_main!(benches);
