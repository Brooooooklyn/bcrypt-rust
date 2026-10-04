//! Head-to-head criterion benchmarks: `bcrypt-rust` vs the incumbent
//! `bcrypt` crate (a target-gated dev-dependency, pinned in Cargo.toml).
//!
//! Run everything:
//!
//! ```text
//! cargo bench --bench vs_bcrypt
//! ```
//!
//! Run one group (criterion filters on the whole benchmark id, so the group
//! name is a prefix; the cost comes first inside `single_hash`):
//!
//! ```text
//! cargo bench --bench vs_bcrypt -- single_hash/cost4
//! cargo bench --bench vs_bcrypt -- batch_16
//! cargo bench --bench vs_bcrypt -- verify
//! ```
//!
//! # Groups
//!
//! | Group | What it answers |
//! |---|---|
//! | `single_hash` | One-shot `hash` latency at costs 4 and 12, incumbent vs this crate, on one fixed 24-byte password. Single-hash latency is scalar by design on both sides, so this is the apples-to-apples arm. |
//! | `batch_16` | 16 distinct deterministic passwords at costs 4 and 12. `bcrypt_loop` is the only pattern the incumbent offers — a sequential `hash` loop, per-call OS salt draws included; `bcrypt_many` is this crate's SIMD batch on the detected backend, fed the fixed corpus salts. Throughput is reported as elements/second. |
//! | `verify` | `verify` at cost 4 on one precomputed hash, incumbent vs this crate — the parsed-settings path a login endpoint runs on every attempt. |
//!
//! # The `parallel` leg
//!
//! Features unify per build, so a `bcrypt_many` arm compiled with `parallel`
//! cannot sit next to one compiled without it in a single binary — there is
//! no distinct in-binary arm to write. The orchestrator runs the batch leg
//! twice instead, and this file keeps the two runs from sharing a criterion
//! history by relabeling the arm when the feature is compiled in:
//!
//! ```text
//! cargo bench --bench vs_bcrypt --features parallel -- batch_16
//! ```
//!
//! # Sanity gate
//!
//! Every group's setup first asserts that `hash_with_salt` from both crates
//! formats byte-identical `$2b$` strings for one fixed (password, salt,
//! cost); the bench panics on divergence — a fast-but-wrong number is worse
//! than none. The gate runs outside every timed loop.
//!
//! A number from this file is only meaningful with the machine recorded
//! alongside: paste `uname -m` and the CPU brand string next to any table
//! that leaves this repository.

use std::hint::black_box;
use std::time::Duration;

use bcrypt_rust::{Version, bcrypt_many, hash, hash_with_salt, verify};
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group,
    criterion_main,
};

/// One fixed 24-byte password for the `single_hash` and `verify` arms.
const PWD: &[u8] = b"correct horse battery!!!";
/// 16 bytes, the one salt length bcrypt has.
const SALT: [u8; 16] = *b"bcrypt-bench-sal";
/// Batch size for the `batch_16` group.
const BATCH: usize = 16;

/// Benchmark id of the `bcrypt_many` batch arm. `parallel` cannot be a
/// separate arm in one binary (see the module docs), so the arm is relabeled
/// when compiled with the feature on.
#[cfg(feature = "parallel")]
const MANY_ID: &str = "bcrypt_many+parallel";
#[cfg(not(feature = "parallel"))]
const MANY_ID: &str = "bcrypt_many";

/// SplitMix64: 64-bit state, one mixing round per output. Deterministic
/// bench inputs, no deps (not a CSPRNG, nor does it need to be) — the same
/// generator `benches/bcrypt.rs` and `tests/backends.rs` carry.
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

/// [`BATCH`] deterministic passwords (8..=48 bytes) plus one salt each for
/// the `bcrypt_many` arm. Fixed seed, chosen arbitrarily: "bcrypt bench 3".
/// Reproduces on every rerun and host.
fn batch_corpus() -> (Vec<Vec<u8>>, Vec<[u8; 16]>) {
    let mut rng = SplitMix64::new(0xBC79_7BE0_0000_0003);
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

/// The byte-identical output gate: both crates must format the same `$2b$`
/// string for the same (password, salt, cost). Called from every group's
/// setup — two cost-4 hashes, never in a timed loop — and panics on
/// divergence, because a fast-but-wrong number is worse than none.
fn sanity_gate() {
    let ours = hash_with_salt(PWD, 4, SALT)
        .expect("bcrypt-rust hash_with_salt")
        .format_for_version(Version::TwoB);
    let theirs = bcrypt::hash_with_salt(PWD, 4, SALT)
        .expect("bcrypt hash_with_salt")
        .format_for_version(bcrypt::Version::TwoB);
    assert_eq!(
        ours, theirs,
        "bcrypt-rust and bcrypt diverged on the same (password, salt, cost)"
    );
}

/// Same policy as `benches/bcrypt.rs`: slow arms (cost 12 — the incumbent's
/// 16-item sequential loop is several seconds of work per iteration) get
/// Flat sampling at criterion's minimum sample count; fast arms keep a
/// deeper sample and a bounded measurement window.
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
/// Both arms run the full public `hash` path: OS-drawn salt, `$2b$`
/// formatting. Single-hash latency is scalar by design on both sides.
fn bench_single_hash(c: &mut Criterion) {
    sanity_gate();
    let mut group = c.benchmark_group("single_hash");
    for cost in [4u32, 12] {
        tune(&mut group, cost);
        let label = format!("cost{cost}");
        group.bench_function(BenchmarkId::new(label.as_str(), "bcrypt"), |b| {
            b.iter(|| black_box(bcrypt::hash(black_box(PWD), black_box(cost)).expect("hash")));
        });
        group.bench_function(BenchmarkId::new(label.as_str(), "bcrypt_rust"), |b| {
            b.iter(|| black_box(hash(black_box(PWD), black_box(cost)).expect("hash")));
        });
    }
    group.finish();
}

/// 16-password batch. `bcrypt_loop` is the only pattern an incumbent caller
/// can write — a sequential `hash` loop, per-call OS salt draws included.
/// The `bcrypt_many` arm gets the fixed corpus salts, so it prices the
/// kernel alone; with the `parallel` feature compiled in this same call is
/// the parallel build and carries the `bcrypt_many+parallel` id.
fn bench_batch(c: &mut Criterion) {
    sanity_gate();
    let (store, salts) = batch_corpus();
    let passwords: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();

    let mut group = c.benchmark_group("batch_16");
    group.throughput(Throughput::Elements(BATCH as u64));
    for cost in [4u32, 12] {
        tune(&mut group, cost);
        let label = format!("cost{cost}_n{BATCH}");
        group.bench_function(BenchmarkId::new("bcrypt_loop", &label), |b| {
            b.iter(|| {
                for pwd in &passwords {
                    black_box(bcrypt::hash(black_box(*pwd), black_box(cost)).expect("hash"));
                }
            });
        });
        group.bench_function(BenchmarkId::new(MANY_ID, &label), |b| {
            b.iter(|| {
                black_box(
                    bcrypt_many(cost, black_box(&passwords), black_box(&salts)).expect("batch"),
                );
            });
        });
    }
    group.finish();
}

/// `verify` at cost 4 on one precomputed hash — the parsed-settings path a
/// login endpoint runs. The hash comes from this crate; the sanity gate
/// already proved the incumbent formats the identical string.
fn bench_verify(c: &mut Criterion) {
    sanity_gate();
    let hash_string = hash_with_salt(PWD, 4, SALT)
        .expect("valid cost")
        .format_for_version(Version::TwoB);
    // Both arms must verify `true`; a wrong-password arm would run the same
    // kernel and differ only in the final compare, adding nothing.
    assert_eq!(verify(PWD, &hash_string), Ok(true));
    assert!(bcrypt::verify(PWD, &hash_string).expect("verify"));

    let mut group = c.benchmark_group("verify");
    group.bench_function(BenchmarkId::new("cost4", "bcrypt"), |b| {
        b.iter(|| {
            black_box(
                bcrypt::verify(black_box(PWD), black_box(hash_string.as_str())).expect("verify"),
            );
        });
    });
    group.bench_function(BenchmarkId::new("cost4", "bcrypt_rust"), |b| {
        b.iter(|| {
            black_box(verify(black_box(PWD), black_box(hash_string.as_str())).expect("verify"));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_single_hash, bench_batch, bench_verify);
criterion_main!(benches);
