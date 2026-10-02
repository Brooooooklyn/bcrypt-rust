//! Dependency-free micro-benchmark for the batch kernels: one `fn main`,
//! no criterion. Times `bcrypt_many_with_backend` per backend and prints
//! one line per arm: `name lanes batch cost iters total_ms hashes_per_sec
//! [ratio]`.
//!
//! Usage: `cargo bench --bench micro -- [--backend NAME|all] [--batch N]
//! [--cost C] [--iters K] [--vs-scalar]`
//!
//! Correctness is gated inside the timing loop: the scalar reference batch
//! is computed once up front and every timed iteration's output is asserted
//! equal to it, so a divergent kernel aborts instead of posting a
//! fast-but-wrong number. Inputs come from a fixed-seed SplitMix64, so a
//! number reproduces on rerun.

use std::hint::black_box;
use std::time::Instant;

use bcrypt_rust::__internal::{Backend, bcrypt_many_with_backend};

/// SplitMix64: 64-bit state, one mixing round per output. Deterministic
/// bench inputs, no deps (not a CSPRNG, nor does it need to be).
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

/// Parsed command line.
struct Args {
    /// `None` means every backend of `Backend::ALL` (available ones run,
    /// the rest print a skip line).
    backend: Option<String>,
    batch: usize,
    cost: u32,
    /// `None` means calibrate each arm to at least one second.
    iters: Option<usize>,
    vs_scalar: bool,
}

fn usage_exit(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("micro: {msg}");
    }
    eprintln!(
        "usage: micro [--backend NAME|all] [--batch N] [--cost C] [--iters K] [--vs-scalar]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut out = Args { backend: None, batch: 64, cost: 5, iters: None, vs_scalar: false };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        // Both `--flag value` and `--flag=value`.
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (argv[i].clone(), None),
        };
        let value = |i: &mut usize| -> String {
            match inline {
                Some(v) => v,
                None => {
                    *i += 1;
                    argv.get(*i)
                        .unwrap_or_else(|| usage_exit(&format!("{flag} needs a value")))
                        .clone()
                }
            }
        };
        match flag.as_str() {
            "--backend" => out.backend = Some(value(&mut i)),
            "--batch" => {
                out.batch = value(&mut i).parse().unwrap_or_else(|_| usage_exit("bad --batch"));
            }
            "--cost" => {
                out.cost = value(&mut i).parse().unwrap_or_else(|_| usage_exit("bad --cost"));
            }
            "--iters" => {
                out.iters =
                    Some(value(&mut i).parse().unwrap_or_else(|_| usage_exit("bad --iters")));
            }
            "--vs-scalar" => out.vs_scalar = true,
            // Cargo itself appends `--bench` when it runs a harness=false
            // target; it carries no information for us.
            "--bench" => {}
            other => usage_exit(&format!("unknown argument {other}")),
        }
        i += 1;
    }
    if out.batch == 0 {
        usage_exit("--batch must be at least 1");
    }
    if matches!(out.iters, Some(0)) {
        usage_exit("--iters must be at least 1");
    }
    out
}

fn find_backend(name: &str) -> Backend {
    Backend::ALL
        .iter()
        .copied()
        .find(|b| b.name() == name)
        .unwrap_or_else(|| {
            let valid = Backend::ALL.iter().map(|b| b.name()).collect::<Vec<_>>().join(", ");
            usage_exit(&format!("unknown backend {name} (valid: {valid}, or all)"))
        })
}

/// One batch through the named backend. The caller must have checked
/// `backend.is_available()` before calling — there is no fallback here.
fn run(backend: Backend, cost: u32, passwords: &[&[u8]], salts: &[[u8; 16]]) -> Vec<[u8; 24]> {
    // SAFETY: every call site checks `backend.is_available()` first.
    unsafe { bcrypt_many_with_backend(backend, cost, black_box(passwords), black_box(salts)) }
        .unwrap_or_else(|e| {
            eprintln!("micro: batch failed on {}: {e}", backend.name());
            std::process::exit(1)
        })
}

/// Gate one iteration against the scalar reference; aborts on divergence.
fn gate(backend: Backend, outs: Vec<[u8; 24]>, reference: &[[u8; 24]]) {
    assert_eq!(
        black_box(outs),
        reference,
        "correctness gate: {} diverged from scalar",
        backend.name()
    );
}

/// Derive the iteration count for a >=1 s arm from one timed iteration
/// (which doubles as warm-up and as a gated correctness pass).
fn calibrate(
    backend: Backend,
    cost: u32,
    passwords: &[&[u8]],
    salts: &[[u8; 16]],
    reference: &[[u8; 24]],
) -> usize {
    let start = Instant::now();
    gate(backend, run(backend, cost, passwords, salts), reference);
    let secs = start.elapsed().as_secs_f64();
    // `secs` is never truly zero for a real bcrypt batch; the clamp only
    // guards degenerate --cost/--batch combinations.
    ((1.0 / secs).ceil() as usize).clamp(1, 100_000)
}

fn main() {
    let args = parse_args();

    // Surface the runtime detection pick before any arm runs: on a host with
    // both x86-64 SIMD backends this is the width shootout's winner (and pays
    // its one-time cost up front, outside every timed loop). The arms below
    // still drive each backend explicitly, so both sides can be compared
    // against the pick in one window.
    println!("detected backend: {}", bcrypt_rust::detected_backend());

    // Fixed seed, chosen arbitrarily: "bcrypt bench 1". Reproducible.
    let mut rng = SplitMix64::new(0xBC79_7BE0_0000_0001);
    let store: Vec<Vec<u8>> = (0..args.batch)
        .map(|_| {
            let mut password = vec![0u8; 8 + rng.next_below(41)]; // 8..=48 bytes
            rng.fill_bytes(&mut password);
            password
        })
        .collect();
    let passwords: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();
    let salts: Vec<[u8; 16]> = (0..args.batch)
        .map(|_| {
            let mut salt = [0u8; 16];
            rng.fill_bytes(&mut salt);
            salt
        })
        .collect();

    // The correctness reference: scalar, untimed, once.
    let reference = black_box(run(Backend::Scalar, args.cost, &passwords, &salts));

    let mut selected: Vec<Backend> = match args.backend.as_deref() {
        None | Some("all") => Backend::ALL.to_vec(),
        Some(name) => vec![find_backend(name)],
    };
    // The ratio column needs a *measured* scalar, not just the untimed
    // reference batch, so scalar joins the run list (first) when absent.
    if args.vs_scalar && !selected.contains(&Backend::Scalar) {
        selected.insert(0, Backend::Scalar);
    }

    let mut scalar_hps = None;
    for &backend in &selected {
        if !backend.is_available() {
            println!("backend {} not available on this host, skipping", backend.name());
            continue;
        }
        let iters = match args.iters {
            Some(k) => k,
            None => calibrate(backend, args.cost, &passwords, &salts, &reference),
        };
        let start = Instant::now();
        for _ in 0..iters {
            gate(backend, run(backend, args.cost, &passwords, &salts), &reference);
        }
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1e3;
        let hashes_per_sec = (iters * args.batch) as f64 / elapsed.as_secs_f64();
        if backend == Backend::Scalar {
            scalar_hps = Some(hashes_per_sec);
        }
        if args.vs_scalar {
            // `vs_scalar` forces scalar into the first arm, so this is set.
            let ratio = hashes_per_sec / scalar_hps.unwrap_or(hashes_per_sec);
            println!(
                "{} {} {} {} {} {total_ms:.1} {hashes_per_sec:.1} {ratio:.2}",
                backend.name(),
                backend.lanes(),
                args.batch,
                args.cost,
                iters
            );
        } else {
            println!(
                "{} {} {} {} {} {total_ms:.1} {hashes_per_sec:.1}",
                backend.name(),
                backend.lanes(),
                args.batch,
                args.cost,
                iters
            );
        }
    }
}
