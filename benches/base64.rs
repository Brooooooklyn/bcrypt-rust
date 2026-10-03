//! Scalar vs dispatched (SIMD) base64 codec timings.
//!
//! The codec costs ~20 ns of a 163 µs cost-4 hash (0.012%) — its SIMD
//! kernels exist for completeness of the SIMD story, not throughput. This
//! bench is the regression net that keeps the kernels honest: `scalar` is
//! the reference implementation, `dispatch` is what the crate actually
//! runs (NEON on aarch64, AVX2/SSSE3 on x86-64, scalar elsewhere).

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use bcrypt_rust::__internal::{base64_scalar, decode_16, decode_23, encode_16, encode_23};

/// Salt half of a JtR test vector — a real-world input, not a pattern.
const SALT: [u8; 16] = [
    38, 113, 212, 141, 108, 213, 195, 166, 201, 38, 20, 13, 47, 40, 104, 18,
];
const HASH: [u8; 23] = *b"\x11\x22\x33\x44\x55\x66\x77\x88\x99\xAA\xBB\xCC\xDD\xEE\xFF\x00\x42\x13\x37\xC0\xDE\x00\x01";

fn enc_16(c: &mut Criterion) {
    let mut g = c.benchmark_group("base64_encode_16");
    g.bench_function("scalar", |b| {
        b.iter(|| base64_scalar::encode_16(black_box(&SALT)))
    });
    g.bench_function("dispatch", |b| b.iter(|| encode_16(black_box(&SALT))));
}

fn enc_23(c: &mut Criterion) {
    let mut g = c.benchmark_group("base64_encode_23");
    g.bench_function("scalar", |b| {
        b.iter(|| base64_scalar::encode_23(black_box(&HASH)))
    });
    g.bench_function("dispatch", |b| b.iter(|| encode_23(black_box(&HASH))));
}

fn dec_16(c: &mut Criterion) {
    let s = encode_16(&SALT);
    let mut g = c.benchmark_group("base64_decode_16");
    g.bench_function("scalar", |b| {
        b.iter(|| base64_scalar::decode_16(black_box(&s)))
    });
    g.bench_function("dispatch", |b| b.iter(|| decode_16(black_box(&s))));
}

fn dec_23(c: &mut Criterion) {
    let s = encode_23(&HASH);
    let mut g = c.benchmark_group("base64_decode_23");
    g.bench_function("scalar", |b| {
        b.iter(|| base64_scalar::decode_23(black_box(&s)))
    });
    g.bench_function("dispatch", |b| b.iter(|| decode_23(black_box(&s))));
}

criterion_group!(benches, enc_16, enc_23, dec_16, dec_23);
criterion_main!(benches);
