//! Cross-backend differential tests: every backend this host can run must
//! produce byte-identical output to [`Backend::Scalar`] on the same batch —
//! random batches across lane-boundary sizes, the 71/72/73-byte password
//! boundary, and the two pinned OpenBSD vectors.
//!
//! Set `BCRYPT_REQUIRE_BACKEND=<name>` (e.g. `neon`) to make the run FAIL
//! when that backend is unavailable — skipping whatever the host lacks is
//! right for a CI matrix but wrong when the point is to exercise one
//! specific kernel.
//!
//! Randomness: an inline SplitMix64 with fixed seeds, so a failure
//! reproduces on rerun. No OS entropy anywhere below.

#![cfg(all(feature = "alloc", feature = "internal-api"))]

use bcrypt_rust::__internal::{Backend, bcrypt_many_with_backend};
use bcrypt_rust::HashParts;

/// SplitMix64: 64-bit state, one mixing round per output. Small, portable,
/// and good enough for test corpora (not a CSPRNG, nor does it need to be).
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

    /// Roughly uniform in `0..n` (modulo bias is irrelevant for test data).
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

/// One batch: `store` owns the password bytes, `salts` the random salts.
struct Batch {
    store: Vec<Vec<u8>>,
    salts: Vec<[u8; 16]>,
}

impl Batch {
    /// `n` passwords of random length in `0..=100` with full-range bytes
    /// (NULs and high bytes included), one random salt each.
    fn random(rng: &mut SplitMix64, n: usize) -> Batch {
        let mut store = Vec::with_capacity(n);
        let mut salts = Vec::with_capacity(n);
        for _ in 0..n {
            let mut password = vec![0u8; rng.next_below(101)];
            rng.fill_bytes(&mut password);
            store.push(password);
            let mut salt = [0u8; 16];
            rng.fill_bytes(&mut salt);
            salts.push(salt);
        }
        Batch { store, salts }
    }

    fn passwords(&self) -> Vec<&[u8]> {
        self.store.iter().map(Vec::as_slice).collect()
    }
}

/// The `BCRYPT_REQUIRE_BACKEND` contract: `Some(backend)` when the env var
/// names a known backend. Fails the test on an unknown name, or on a known
/// backend this CPU cannot run — requiring a backend must never degrade
/// into silently skipping it.
fn required_backend() -> Option<Backend> {
    let name = std::env::var("BCRYPT_REQUIRE_BACKEND").ok()?;
    let valid = Backend::ALL
        .iter()
        .map(|b| b.name())
        .collect::<Vec<_>>()
        .join(", ");
    let backend = Backend::ALL
        .iter()
        .copied()
        .find(|b| b.name() == name.as_str())
        .unwrap_or_else(|| panic!("BCRYPT_REQUIRE_BACKEND={name}: unknown backend (valid: {valid})"));
    assert!(
        backend.is_available(),
        "BCRYPT_REQUIRE_BACKEND={name} but {name} is not available on this host"
    );
    Some(backend)
}

/// Hash one batch at `cost` with Scalar as the reference, then with every
/// available backend, asserting byte equality. Returns the backends that
/// actually ran (always includes Scalar).
fn run_cross_backend(cost: u32, passwords: &[&[u8]], salts: &[[u8; 16]]) -> Vec<Backend> {
    // SAFETY: Scalar is available on every host.
    let reference = unsafe { bcrypt_many_with_backend(Backend::Scalar, cost, passwords, salts) }
        .expect("scalar batch failed");
    let mut ran = vec![Backend::Scalar];
    for &backend in Backend::ALL {
        if backend == Backend::Scalar || !backend.is_available() {
            continue;
        }
        // SAFETY: `backend.is_available()` was checked just above.
        let got = unsafe { bcrypt_many_with_backend(backend, cost, passwords, salts) }
            .expect("batch failed");
        assert_eq!(
            got, reference,
            "backend {backend} diverged from scalar at cost {cost} ({} passwords)",
            passwords.len()
        );
        ran.push(backend);
    }
    ran
}

/// Every backend runs the batch-size sweep; sizes straddle the lane widths
/// (1, 2, 4, 8, 16) so full groups, short tails and empty input all appear.
/// Costs rotate through 4..=6 per size to cover the range without the full
/// cross product's runtime.
#[test]
fn sweep_batch_sizes_across_backends() {
    let required = required_backend();
    // Fixed seed, chosen arbitrarily: "bcrypt seed 1". Reproducible.
    let mut rng = SplitMix64::new(0xBC79_7A5A_5EED_0001);
    let sizes = [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33];
    let mut ran_total: Vec<Backend> = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let cost = 4 + (i % 3) as u32;
        let batch = Batch::random(&mut rng, n);
        for backend in run_cross_backend(cost, &batch.passwords(), &batch.salts) {
            if !ran_total.contains(&backend) {
                ran_total.push(backend);
            }
        }
    }
    for backend in &ran_total {
        println!("backend {backend}: ran the batch-size sweep");
    }
    if let Some(req) = required {
        assert!(
            ran_total.contains(&req),
            "BCRYPT_REQUIRE_BACKEND={req} never ran in the sweep"
        );
    }
}

/// The password-length boundary, pinned explicitly rather than left to the
/// sweep's random lengths: 0 (empty key stream), 1, 71 (NUL terminator in
/// the last key byte), 72 (buffer full), 73 (first truncated length) and
/// 100 (well past truncation). One batch of six, at costs 4, 5 and 6.
#[test]
fn password_length_boundaries_across_backends() {
    let required = required_backend();
    // Fixed seed, chosen arbitrarily: "bcrypt seed 2". Reproducible.
    let mut rng = SplitMix64::new(0xBC79_7A5A_5EED_0002);
    let lengths = [0usize, 1, 71, 72, 73, 100];
    let mut store: Vec<Vec<u8>> = Vec::with_capacity(lengths.len());
    let mut salts: Vec<[u8; 16]> = Vec::with_capacity(lengths.len());
    for &len in &lengths {
        let mut password = vec![0u8; len];
        rng.fill_bytes(&mut password);
        store.push(password);
        let mut salt = [0u8; 16];
        rng.fill_bytes(&mut salt);
        salts.push(salt);
    }
    let passwords: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();
    let mut ran_total: Vec<Backend> = Vec::new();
    for cost in [4u32, 5, 6] {
        for backend in run_cross_backend(cost, &passwords, &salts) {
            if !ran_total.contains(&backend) {
                ran_total.push(backend);
            }
        }
    }
    for backend in &ran_total {
        println!("backend {backend}: ran the length-boundary batch");
    }
    if let Some(req) = required {
        assert!(
            ran_total.contains(&req),
            "BCRYPT_REQUIRE_BACKEND={req} never ran on the boundary batch"
        );
    }
}

/// The two pinned OpenBSD vectors (`""` and `"U*U"` at `$2a$05$CCCC…`),
/// hashed as one batch of two per backend. The salt and cost are decoded
/// out of the known hash strings via `HashParts`; the 24-byte raw outputs
/// must match scalar's, which the crate's unit tests pin to OpenBSD.
#[test]
fn openbsd_vectors_across_backends() {
    let required = required_backend();
    const EMPTY: &str = "$2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy";
    const U_U: &str = "$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW";
    let parts_empty: HashParts = EMPTY.parse().expect("vector parses");
    let parts_u_u: HashParts = U_U.parse().expect("vector parses");
    assert_eq!(parts_empty.get_cost(), 5);
    assert_eq!(parts_u_u.get_cost(), 5);
    assert_eq!(parts_empty.get_salt_raw(), parts_u_u.get_salt_raw());

    let store: Vec<Vec<u8>> = vec![Vec::new(), b"U*U".to_vec()];
    let passwords: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();
    let salts = [parts_empty.get_salt_raw(), parts_u_u.get_salt_raw()];

    // The reference is pinned to the vectors' own 23-byte payloads, so this
    // test fails if scalar itself ever regresses, not just on divergence.
    // SAFETY: Scalar is available on every host.
    let reference =
        unsafe { bcrypt_many_with_backend(Backend::Scalar, 5, &passwords, &salts) }
            .expect("scalar batch failed");
    assert_eq!(&reference[0][..23], &parts_empty.get_hash()[..]);
    assert_eq!(&reference[1][..23], &parts_u_u.get_hash()[..]);

    let ran = run_cross_backend(5, &passwords, &salts);
    for backend in &ran {
        println!("backend {backend}: ran the OpenBSD vectors");
    }
    if let Some(req) = required {
        assert!(
            ran.contains(&req),
            "BCRYPT_REQUIRE_BACKEND={req} never ran on the OpenBSD vectors"
        );
    }
}
