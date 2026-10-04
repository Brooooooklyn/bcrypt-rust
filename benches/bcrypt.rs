//! Criterion benchmarks for `bcrypt-rust`.
//!
//! Run everything:
//!
//! ```text
//! cargo bench --bench bcrypt
//! ```
//!
//! Run one group (criterion filters on the whole benchmark id, so the group
//! name is a prefix; the cost comes first inside `single_hash`):
//!
//! ```text
//! cargo bench --bench bcrypt -- single_hash
//! cargo bench --bench bcrypt -- single_hash/cost4
//! cargo bench --bench bcrypt -- batch
//! cargo bench --bench bcrypt -- verify
//! ```
//!
//! # Groups
//!
//! | Group | What it answers |
//! |---|---|
//! | `single_hash` | One-shot latency at costs 4 (CI-fast), 10 and 12 (production-typical): `raw_core` is `bcrypt`, the bare kernel; `public_hash` is `hash`, the full path with an OS-drawn salt and `$2b$` formatting. Single-hash latency always runs the scalar kernel by design, so no backend parameter exists here. |
//! | `batch` | `bcrypt_many` throughput: 16 deterministic (password, salt) pairs at costs 4 and 12, for every backend this CPU can execute (forced through `__internal::bcrypt_many_with_backend`) plus the detected-default public path. Key/salt word preparation is included — it is part of what a caller pays. This is the only group where SIMD, and the `parallel` feature when enabled, show up. |
//! | `verify` | `verify` at cost 4: the parsed-settings path, with cost and salt decoded out of the hash string on every call. |
//!
//! A number from this file enters the README only after being measured on
//! the machine the README names, in the configuration the table claims. The
//! CI-tracked regression numbers live in `benches/codspeed.rs` and mean
//! something different: simulated instruction counts, not wall clock.

use std::hint::black_box;
use std::time::Duration;

use bcrypt_rust::__internal::{Backend, bcrypt_many_with_backend};
use bcrypt_rust::{Version, bcrypt, bcrypt_many, hash, hash_with_salt, verify};
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group,
    criterion_main,
};

const PWD: &[u8] = b"correct horse battery staple";
/// 16 bytes, the one salt length bcrypt has.
const SALT: [u8; 16] = *b"bcrypt-bench-sal";
/// Batch size for the `batch` group.
const BATCH: usize = 16;

/// SplitMix64: 64-bit state, one mixing round per output. Deterministic
/// bench inputs, no deps (not a CSPRNG, nor does it need to be) — the same
/// generator `benches/micro.rs` and `tests/backends.rs` carry.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

/// [`BATCH`] deterministic (password, salt) pairs. Fixed seed, chosen
/// arbitrarily: "bcrypt bench 2". Reproduces on every rerun and host.
fn batch_corpus() -> (Vec<Vec<u8>>, Vec<[u8; 16]>) {
    let mut rng = SplitMix64::new(0xBC79_7BE0_0000_0002);
    let store: Vec<Vec<u8>> = (0..BATCH)
        .map(|_| {
            let mut password = vec![0u8; 8 + rng.next_below(41)]; // 8..=48 bytes
            rng.fill_bytes(&mut password);
            password
        })
        .collect();
    let salts: Vec<[u8; 16]> = (0..BATCH)
        .map(|_| {
            let mut salt = [0u8; 16];
            rng.fill_bytes(&mut salt);
            salt
        })
        .collect();
    (store, salts)
}

/// Backends this CPU can actually execute — the same filter the
/// differential tests apply. This is what discharges the unsafe contract of
/// `bcrypt_many_with_backend` below.
fn runnable_backends() -> Vec<Backend> {
    Backend::ALL
        .iter()
        .copied()
        .filter(|b| b.is_available())
        .collect()
}

/// Slow arms (cost 12: a 16-item batch is one to four seconds of work per
/// iteration) get Flat sampling at criterion's minimum sample count, so the
/// suite finishes in minutes rather than an hour; fast arms keep a deeper
/// sample and a bounded measurement window.
fn tune(group: &mut BenchmarkGroup<'_, WallTime>, cost: u32) {
    if cost >= 12 {
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(5));
    } else {
        group
            .sample_size(50)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));
    }
}

/// One-shot latency, cost parameter first so `single_hash/cost4` filters.
fn bench_single_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_hash");
    for cost in [4u32, 10, 12] {
        tune(&mut group, cost);
        let label = format!("cost{cost}");
        // The bare kernel: fixed salt, no entropy draw, no formatting.
        group.bench_function(BenchmarkId::new(label.as_str(), "raw_core"), |b| {
            b.iter(|| black_box(bcrypt(black_box(cost), black_box(SALT), black_box(PWD))));
        });
        // The full public path: OS salt + `$2b$` string included.
        group.bench_function(BenchmarkId::new(label.as_str(), "public_hash"), |b| {
            b.iter(|| black_box(hash(black_box(PWD), black_box(cost)).expect("hash")));
        });
    }
    group.finish();
}

/// Batch throughput per backend. `default` is the detected-backend public
/// path — the number a caller actually gets; the named arms exist to
/// compare instruction sets on one host and to price `scalar` as the
/// baseline the SIMD claims are ratios against.
fn bench_batch(c: &mut Criterion) {
    let backends = runnable_backends();
    let (store, salts) = batch_corpus();
    let passwords: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements(BATCH as u64));
    for cost in [4u32, 12] {
        tune(&mut group, cost);
        let label = format!("cost{cost}_n{BATCH}");
        for &backend in &backends {
            // The availability check the unsafe contract asks for; the arm
            // setup asserts it once, outside the timed loop.
            assert!(backend.is_available());
            group.bench_function(BenchmarkId::new(backend.name(), &label), |b| {
                b.iter(|| {
                    black_box(
                        // SAFETY: `backend` came from `runnable_backends()`
                        // and was asserted available just above.
                        unsafe {
                            bcrypt_many_with_backend(
                                black_box(backend),
                                cost,
                                black_box(&passwords),
                                black_box(&salts),
                            )
                        }
                        .expect("batch"),
                    );
                });
            });
        }
        group.bench_function(BenchmarkId::new("default", &label), |b| {
            b.iter(|| {
                black_box(
                    bcrypt_many(cost, black_box(&passwords), black_box(&salts)).expect("batch"),
                );
            });
        });
    }
    group.finish();
}

/// `verify` at cost 4: the string is parsed for cost and salt on every
/// call, so this is the full parsed-settings path a login endpoint runs.
fn bench_verify(c: &mut Criterion) {
    let parts = hash_with_salt(PWD, 4, SALT).expect("valid cost");
    let hash_string = parts.format_for_version(Version::TwoB);
    // The arm must verify `true`; a wrong-password arm would run the same
    // kernel and differ only in the final compare, adding nothing.
    assert_eq!(verify(PWD, &hash_string), Ok(true));

    let mut group = c.benchmark_group("verify");
    group.bench_function("cost4", |b| {
        b.iter(|| {
            black_box(verify(black_box(PWD), black_box(hash_string.as_str())).expect("verify"));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_single_hash, bench_batch, bench_verify);
criterion_main!(benches);
