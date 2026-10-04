//! The public hashing API: the raw core, the single-hash convenience
//! functions, and the multi-lane batch entry points.
//!
//! One bcrypt hash is a strictly sequential chain of Blowfish key expansions,
//! so the single-hash functions here always run the scalar kernel — SIMD
//! never enters them. SIMD backends speed up **batch throughput only** and
//! are reached exclusively through [`bcrypt_many`], [`hash_many`],
//! [`hash_many_with_salts`] and [`verify_many`], which hash one password per
//! vector lane.
//!
//! Key preparation matches the `bcrypt` crate and OpenBSD: the password is
//! NUL-terminated and truncated at 72 bytes (see [`padded_key`]); the
//! `non_truncating_*` family instead rejects passwords of 72 bytes or more
//! with [`BcryptError::Truncation`].

// `String` appears only in the random-salt API, which is `std`-gated (std
// implies alloc). `Vec` appears in the batch API, which is `alloc`-gated.
#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::eks::Backend;
use crate::encoding::{HashParts, Version};
use crate::error::{BcryptError, BcryptResult};

/// Smallest allowed cost: `2^4` expansion-loop iterations.
pub const MIN_COST: u32 = 4;
/// Largest allowed cost: `2^31` expansion-loop iterations.
pub const MAX_COST: u32 = 31;
/// The cost used when none is specified.
///
/// This is a convention, not a recommendation: measure and pick the largest
/// cost your latency budget tolerates.
pub const DEFAULT_COST: u32 = 12;

/// bcrypt key preparation: the password bytes copied into a zeroed 72-byte
/// buffer, with the key stream cycling at `min(len + 1, 72)` — i.e.
/// NUL-terminated, truncated at 72. Only 18 words (72 bytes) are ever read
/// from the stream, so the NUL appended to a ≥72-byte password never
/// participates, which is exactly why a 72-byte and a 73-byte password with
/// the same prefix hash identically.
fn padded_key(password: &[u8]) -> ([u8; 72], usize) {
    let mut key = [0u8; 72];
    let copied = password.len().min(72);
    key[..copied].copy_from_slice(&password[..copied]);
    (key, (copied + 1).min(72))
}

/// Validate a cost *argument*. (A cost inside a hash *string* is a parse
/// error instead — see `HashParts`' `FromStr`.)
#[inline]
fn check_cost(cost: u32) -> BcryptResult<()> {
    if (MIN_COST..=MAX_COST).contains(&cost) {
        Ok(())
    } else {
        Err(BcryptError::CostNotAllowed(cost))
    }
}

/// The `non_truncating_*` gate: reject inputs bcrypt would silently cut.
/// The payload is the key-stream length including the terminator bcrypt
/// appends, matching the `bcrypt` crate.
#[inline]
fn check_truncation(password_len: usize) -> BcryptResult<()> {
    if password_len >= 72 {
        Err(BcryptError::Truncation(password_len + 1))
    } else {
        Ok(())
    }
}

/// A fresh 16-byte salt from the OS CSPRNG.
#[cfg(feature = "std")]
fn random_salt() -> BcryptResult<[u8; 16]> {
    let mut salt = [0u8; 16];
    crate::random::os_random(&mut salt).map_err(BcryptError::Rand)?;
    Ok(salt)
}

/// Wipe helpers: real wipes under `zeroize`, no-ops otherwise. Taking `&mut`
/// (rather than cfg-ing out the call sites) keeps every caller's `mut`
/// binding used in every feature configuration; the empty bodies inline away.
#[inline(always)]
fn wipe_key(key: &mut [u8; 72]) {
    #[cfg(feature = "zeroize")]
    crate::wipe::secure_wipe_bytes(key);
    #[cfg(not(feature = "zeroize"))]
    let _ = key;
}

/// See [`wipe_key`].
#[inline(always)]
fn wipe_words(words: &mut [u32]) {
    #[cfg(feature = "zeroize")]
    crate::wipe::secure_wipe_u32(words);
    #[cfg(not(feature = "zeroize"))]
    let _ = words;
}

/// See [`wipe_key`].
#[cfg(feature = "alloc")]
#[inline(always)]
fn wipe_bytes(bytes: &mut [u8]) {
    #[cfg(feature = "zeroize")]
    crate::wipe::secure_wipe_bytes(bytes);
    #[cfg(not(feature = "zeroize"))]
    let _ = bytes;
}

/// Compare two 23-byte hash payloads in data-independent time.
///
/// This guards [`verify`]'s verdict: an early-exit `memcmp` would leak the
/// position of the first differing byte to a timing side channel. The fold
/// always reads all 23 bytes, and the [`black_box`](core::hint::black_box)
/// forces the fold to actually execute — without it the compiler is entitled
/// to prove the loop equivalent to `memcmp` and reintroduce the early exit.
/// `#[inline(never)]` keeps the fold opaque so no caller-side specialization
/// can do the same.
#[inline(never)]
#[must_use]
pub fn constant_time_eq(a: &[u8; 23], b: &[u8; 23]) -> bool {
    let diff = a.iter().zip(b.iter()).fold(0u8, |d, (x, y)| d | (x ^ y));
    core::hint::black_box(diff) == 0
}

/// The raw bcrypt core: one hash of `password` with a caller-supplied salt,
/// returning the full 24-byte ciphertext.
///
/// `cost` is **not validated** here — matching the `bcrypt` crate's raw
/// function (debug builds assert it is in `4..=31`; release builds would
/// just run). Every other entry point in this crate validates the cost up
/// front and ends up here. The password is NUL-terminated and truncated at
/// 72 bytes; use the `non_truncating_*` family to reject instead of
/// truncate.
///
/// The string format drops the last of the 24 bytes; the hash-string APIs
/// encode only `out[..23]`.
#[must_use]
pub fn bcrypt(cost: u32, salt: [u8; 16], password: &[u8]) -> [u8; 24] {
    let (mut key, key_len) = padded_key(password);
    // One-lane arrays, so the scalar kernel runs exactly one hash.
    let mut key_words = [crate::eks::expand_key_words(&key, key_len)];
    wipe_key(&mut key);
    let salt_words = [crate::eks::salt_words(&salt)];
    let mut out = [0u8; 24];
    crate::eks::scalar::bcrypt_lanes(
        cost,
        &key_words,
        &salt_words,
        core::slice::from_mut(&mut out),
    );
    wipe_words(key_words.as_flattened_mut());
    out
}

// ---------------------------------------------------------------------------
// Single-hash API
// ---------------------------------------------------------------------------

/// Hash a password with a fresh random salt, returning the 60-byte `$2b$`
/// hash string.
///
/// The salt comes from the OS CSPRNG (hence `std`). Like every single-hash
/// function here, this never touches the SIMD backends: one bcrypt hash is
/// strictly sequential, so it always runs the scalar kernel.
///
/// # Errors
///
/// [`CostNotAllowed`](BcryptError::CostNotAllowed) if `cost` is outside
/// `4..=31`; [`Rand`](BcryptError::Rand) if the OS entropy source fails.
#[cfg(feature = "std")]
pub fn hash<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<String> {
    Ok(hash_with_result(password, cost)?.format_for_version(Version::TwoB))
}

/// Like [`hash`], but returns the hash string as stack bytes.
///
/// # Errors
///
/// Same as [`hash`].
#[cfg(feature = "std")]
pub fn hash_bytes<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<[u8; 60]> {
    Ok(hash_with_result(password, cost)?.format_bytes(Version::TwoB))
}

/// Like [`hash`], but returns the parsed parts rather than the string.
///
/// # Errors
///
/// Same as [`hash`].
#[cfg(feature = "std")]
pub fn hash_with_result<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<HashParts> {
    check_cost(cost)?;
    let salt = random_salt()?;
    hash_with_salt(password, cost, salt)
}

/// Hash a password with a caller-supplied salt — no OS entropy, no `std`,
/// no `alloc`.
///
/// # Errors
///
/// [`CostNotAllowed`](BcryptError::CostNotAllowed) if `cost` is outside
/// `4..=31`.
pub fn hash_with_salt<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
    salt: [u8; 16],
) -> BcryptResult<HashParts> {
    check_cost(cost)?;
    let out = bcrypt(cost, salt, password.as_ref());
    let mut hash = [0u8; 23];
    hash.copy_from_slice(&out[..23]);
    Ok(HashParts::new(cost, salt, hash))
}

/// Like [`hash_with_salt`], but returns the 60-byte hash string as stack
/// bytes.
///
/// # Errors
///
/// Same as [`hash_with_salt`].
pub fn hash_with_salt_bytes<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
    salt: [u8; 16],
) -> BcryptResult<[u8; 60]> {
    Ok(hash_with_salt(password, cost, salt)?.format_bytes(Version::TwoB))
}

/// Verify `password` against a hash string, in constant time per candidate.
///
/// The cost and salt come from the parsed string; any of the four `$2*$`
/// prefixes is accepted, and all compute the same algorithm (see the
/// crate-level "Variant semantics" section). Needs neither `std` nor `alloc`.
///
/// # Errors
///
/// [`InvalidHash`](BcryptError::InvalidHash) if the string does not parse.
pub fn verify<P: AsRef<[u8]>>(password: P, hash: &str) -> BcryptResult<bool> {
    let parts: HashParts = hash.parse()?;
    let out = bcrypt(parts.get_cost(), parts.get_salt_raw(), password.as_ref());
    let mut computed = [0u8; 23];
    computed.copy_from_slice(&out[..23]);
    Ok(constant_time_eq(&computed, &parts.get_hash()))
}

/// [`hash`], but rejects passwords of 72 bytes or more instead of silently
/// truncating them.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`hash`].
#[cfg(feature = "std")]
pub fn non_truncating_hash<P: AsRef<[u8]>>(password: P, cost: u32) -> BcryptResult<String> {
    check_truncation(password.as_ref().len())?;
    hash(password, cost)
}

/// [`hash_bytes`], but rejects passwords of 72 bytes or more.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`hash_bytes`].
#[cfg(feature = "std")]
pub fn non_truncating_hash_bytes<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
) -> BcryptResult<[u8; 60]> {
    check_truncation(password.as_ref().len())?;
    hash_bytes(password, cost)
}

/// [`hash_with_result`], but rejects passwords of 72 bytes or more.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`hash_with_result`].
#[cfg(feature = "std")]
pub fn non_truncating_hash_with_result<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
) -> BcryptResult<HashParts> {
    check_truncation(password.as_ref().len())?;
    hash_with_result(password, cost)
}

/// [`hash_with_salt`], but rejects passwords of 72 bytes or more.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`hash_with_salt`].
pub fn non_truncating_hash_with_salt<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
    salt: [u8; 16],
) -> BcryptResult<HashParts> {
    check_truncation(password.as_ref().len())?;
    hash_with_salt(password, cost, salt)
}

/// [`hash_with_salt_bytes`], but rejects passwords of 72 bytes or more.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`hash_with_salt_bytes`].
pub fn non_truncating_hash_with_salt_bytes<P: AsRef<[u8]>>(
    password: P,
    cost: u32,
    salt: [u8; 16],
) -> BcryptResult<[u8; 60]> {
    check_truncation(password.as_ref().len())?;
    hash_with_salt_bytes(password, cost, salt)
}

/// [`verify`], but rejects passwords of 72 bytes or more.
///
/// # Errors
///
/// [`Truncation`](BcryptError::Truncation) for `password.len() >= 72`;
/// otherwise same as [`verify`].
pub fn non_truncating_verify<P: AsRef<[u8]>>(password: P, hash: &str) -> BcryptResult<bool> {
    check_truncation(password.as_ref().len())?;
    verify(password, hash)
}

// ---------------------------------------------------------------------------
// Batch API — the SIMD entry points
// ---------------------------------------------------------------------------

/// The dispatch point behind every batch function: routes to
/// [`bcrypt_many_chunked`], the sequential lane-group loop — or, with the
/// `parallel` feature and a batch of at least the `parallel_workers`
/// threshold, to the scoped-worker split in `bcrypt_many_scoped`. Either
/// way `outs[i]` receives item `i`'s 24-byte ciphertext: input order is
/// preserved, and the outputs are byte-identical across both paths.
///
/// # Safety
///
/// `backend.is_available()` must hold on this CPU: the resolved kernel is a
/// `#[target_feature]` function and there is no fallback at this level.
/// Safe callers pass `crate::eks::backend()`, which by construction only
/// ever names a backend this CPU advertises.
#[cfg(feature = "alloc")]
unsafe fn bcrypt_many_inner(
    backend: Backend,
    cost: u32,
    kws: &[[u32; 18]],
    sws: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert_eq!(kws.len(), sws.len());
    debug_assert_eq!(kws.len(), outs.len());
    if kws.is_empty() {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        let workers = parallel_workers(kws.len());
        if workers > 1 {
            // SAFETY: this function's contract, forwarded; the worker split
            // writes disjoint index ranges of `outs`.
            unsafe { bcrypt_many_scoped(backend, cost, kws, sws, outs, workers) };
            return;
        }
    }
    // SAFETY: this function's contract, forwarded unchanged.
    unsafe { bcrypt_many_chunked(backend, cost, kws, sws, outs) };
}

/// The shared chunk loop behind every batch function: `kws[i]`/`sws[i]` are
/// the pre-digested key and salt words of item `i`; `outs[i]` receives its
/// 24-byte ciphertext. Preserves input order. Also the per-worker body of
/// the `parallel` path, which hands each worker a disjoint sub-slice.
///
/// # Safety
///
/// Same contract as [`bcrypt_many_inner`].
#[cfg(feature = "alloc")]
unsafe fn bcrypt_many_chunked(
    backend: Backend,
    cost: u32,
    kws: &[[u32; 18]],
    sws: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert_eq!(kws.len(), sws.len());
    debug_assert_eq!(kws.len(), outs.len());
    if kws.is_empty() {
        return;
    }
    // One dispatch for the whole batch: the caller picked `backend`; resolve
    // its kernel pointer before entering the loop — nothing dispatches per
    // chunk.
    let kernel = crate::eks::bcrypt_lanes_fn(backend);
    let scalar = crate::eks::bcrypt_lanes_fn(Backend::Scalar);
    let lanes = backend.lanes();
    // The stack arrays are sized for the widest backend that exists
    // (AVX-512, 16 lanes), so the layout is backend-independent; only
    // `0..lanes` is ever read by a kernel call.
    debug_assert!(lanes <= 16);
    let mut kw = [[0u32; 18]; 16];
    let mut sw = [[0u32; 4]; 16];
    let mut out = [[0u8; 24]; 16];
    let mut base = 0;
    while base < kws.len() {
        let remaining = kws.len() - base;
        // A remainder of one or two hashes runs faster on the scalar
        // kernel than padded to a full-width group: a padded call costs
        // `lanes x t_scalar / S`, and a backend's per-hash speedup S never
        // exceeds its lane count, so the wide call cannot beat two
        // sequential scalar hashes. Measured: r=1 saves 88% (Zen 4,
        // AVX-512) and 50% (M5 Max, NEON); r=2 wins or ties everywhere.
        // Larger remainders break even at host-dependent points (r=3 on
        // M5 Max, r=12-13 on Zen 4 under AVX-512), so they keep lockstep.
        if remaining <= 2 && remaining < lanes {
            for (slot, (&kw_in, &sw_in)) in kws[base..].iter().zip(&sws[base..]).enumerate() {
                kw[0] = kw_in;
                sw[0] = sw_in;
                // SAFETY: the scalar kernel runs on every CPU, the slices
                // are exactly its one lane, and `out` is a stack array
                // owned by this frame — nothing else aliases it here.
                unsafe { scalar(cost, &kw[..1], &sw[..1], &mut out[..1]) };
                outs[base + slot] = out[0];
            }
            wipe_words(kw.as_flattened_mut());
            wipe_words(sw.as_flattened_mut());
            wipe_bytes(out.as_flattened_mut());
            break;
        }
        let group = lanes.min(remaining);
        debug_assert!(group >= 1);
        kw[..group].copy_from_slice(&kws[base..base + group]);
        sw[..group].copy_from_slice(&sws[base..base + group]);
        // A short tail chunk is padded by repeating this chunk's first lane:
        // duplicate work whose outputs are discarded, so every kernel call
        // is exactly `lanes` wide. (A tail of one or two never reaches
        // here — it runs on the scalar kernel, above.)
        let fill_kw = kw[0];
        let fill_sw = sw[0];
        for slot in &mut kw[group..lanes] {
            *slot = fill_kw;
        }
        for slot in &mut sw[group..lanes] {
            *slot = fill_sw;
        }
        // SAFETY: `backend` is available on this CPU (this function's
        // contract), so this CPU supports the instruction set `kernel`
        // targets. All three slices are exactly `lanes` long, as the kernel
        // contract requires, and `out` is a stack array exclusively owned
        // by this frame — nothing else aliases it for the duration of the
        // call.
        unsafe { kernel(cost, &kw[..lanes], &sw[..lanes], &mut out[..lanes]) };
        outs[base..base + group].copy_from_slice(&out[..group]);
        base += group;
        wipe_words(kw.as_flattened_mut());
        wipe_words(sw.as_flattened_mut());
        wipe_bytes(out.as_flattened_mut());
    }
}

/// Worker count for the `parallel` path: one per available core, and only
/// when the batch gives every worker at least two items.
///
/// Why two items per worker: a scoped spawn+join costs tens of microseconds
/// on every supported OS, while the cheapest legal bcrypt (cost 4) is a
/// sub-millisecond unit of work. At one item per worker that fixed cost is a
/// visible fraction of the work it parallelises; at two it is amortised.
/// The threshold doubles as the guarantee that small batches run the exact
/// same sequential loop as a build without this feature. The crossover on a
/// given machine is what the `batch` group of `benches/bcrypt.rs` measures.
#[cfg(feature = "parallel")]
fn parallel_workers(items: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, core::num::NonZero::get);
    if cores > 1 && items >= 2 * cores { cores } else { 1 }
}

/// `outs`' base pointer, shared with the scoped workers of
/// `bcrypt_many_scoped`. Workers claim whole chunks through an atomic
/// counter, so no two workers ever write the same index; this wrapper is
/// what lets the pointer cross the `std::thread::scope`.
#[cfg(feature = "parallel")]
#[derive(Copy, Clone)]
struct OutBase(*mut [u8; 24]);

// SAFETY: sound exactly as used by `bcrypt_many_scoped`: chunk claims are
// unique (`fetch_add`), the claimed ranges are disjoint, and the scope join
// orders every worker's writes before the caller reads `outs`.
#[cfg(feature = "parallel")]
unsafe impl Send for OutBase {}
// SAFETY: same as the `Send` impl just above.
#[cfg(feature = "parallel")]
unsafe impl Sync for OutBase {}

#[cfg(feature = "parallel")]
impl OutBase {
    /// The `outs[start]` pointer, for a `start` the caller claimed uniquely.
    /// A method (rather than field access at the call site) so the scoped
    /// closure captures this whole `Send` wrapper, not the `!Send` field.
    ///
    /// # Safety
    ///
    /// `start` must be in bounds, and no other thread may write through the
    /// returned pointer — in `bcrypt_many_scoped` both come from the
    /// `fetch_add` chunk claim.
    unsafe fn at(self, start: usize) -> *mut [u8; 24] {
        // SAFETY: in-bounds is the caller's documented promise; this only
        // re-bases the pointer.
        unsafe { self.0.add(start) }
    }
}

/// The `parallel` path: split the batch into lane-aligned chunks and run
/// [`bcrypt_many_chunked`] on each inside a [`std::thread::scope`]. Claims
/// are dynamic (an atomic counter), so a worker the OS refuses to spawn
/// simply means the surviving workers claim its chunks. Output is
/// byte-identical to the sequential loop: chunking only changes *which*
/// duplicate lane-padding work is discarded, never an item's output. Wipe
/// behaviour is unchanged — each worker wipes its own per-group stack
/// buffers inside the chunk loop, and the caller wipes `key_words` after
/// the scope has joined.
///
/// # Safety
///
/// Same contract as [`bcrypt_many_inner`]; every worker forwards `backend`
/// to the same kernel.
#[cfg(feature = "parallel")]
unsafe fn bcrypt_many_scoped(
    backend: Backend,
    cost: u32,
    kws: &[[u32; 18]],
    sws: &[[u32; 4]],
    outs: &mut [[u8; 24]],
    workers: usize,
) {
    use core::sync::atomic::{AtomicUsize, Ordering};

    debug_assert!(workers > 1);
    // Whole lane-groups per chunk: a chunk ending mid-group pads its tail
    // with duplicate work (see `bcrypt_many_chunked`), so ragged boundaries
    // would pay that padding once per worker instead of once per batch.
    let per = kws.len().div_ceil(workers).next_multiple_of(backend.lanes());
    let chunks = kws.len().div_ceil(per);

    let base = OutBase(outs.as_mut_ptr());
    let next = AtomicUsize::new(0);
    let next = &next;

    let run = move || loop {
        // Relaxed: the counter only partitions work. The happens-before the
        // *data* needs comes from `thread::scope` joining every worker.
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= chunks {
            return;
        }
        let start = i * per;
        let end = (start + per).min(kws.len());
        // SAFETY: `i` was claimed exactly once (`fetch_add`), so only this
        // worker writes `start..end`; `start < kws.len()` because
        // `i < chunks`.
        let out = unsafe { core::slice::from_raw_parts_mut(base.at(start), end - start) };
        // SAFETY: `backend` availability is this function's contract,
        // forwarded to the chunk loop unchanged.
        unsafe { bcrypt_many_chunked(backend, cost, &kws[start..end], &sws[start..end], out) };
    };

    std::thread::scope(|scope| {
        for _ in 1..workers {
            // `Builder::spawn_scoped` returns an error where `scope.spawn`
            // panics, and this crate must not panic. A refused thread just
            // means the remaining workers claim its chunks.
            let _ = std::thread::Builder::new().spawn_scoped(scope, run);
        }
        run();
    });
}

/// The cost and length validation shared by [`bcrypt_many`] and
/// [`bcrypt_many_with_backend`].
#[cfg(feature = "alloc")]
fn check_batch(cost: u32, passwords: &[&[u8]], salts: &[[u8; 16]]) -> BcryptResult<()> {
    check_cost(cost)?;
    if passwords.len() != salts.len() {
        return Err(BcryptError::BatchLengthMismatch {
            passwords: passwords.len(),
            salts: salts.len(),
        });
    }
    Ok(())
}

/// The key/salt word digestion shared by [`bcrypt_many`] and
/// [`bcrypt_many_with_backend`].
#[cfg(feature = "alloc")]
fn prep_words(passwords: &[&[u8]], salts: &[[u8; 16]]) -> (Vec<[u32; 18]>, Vec<[u32; 4]>) {
    let mut key_words: Vec<[u32; 18]> = Vec::with_capacity(passwords.len());
    let mut salt_words: Vec<[u32; 4]> = Vec::with_capacity(salts.len());
    for (password, salt) in passwords.iter().zip(salts) {
        let (mut key, key_len) = padded_key(password);
        key_words.push(crate::eks::expand_key_words(&key, key_len));
        wipe_key(&mut key);
        salt_words.push(crate::eks::salt_words(salt));
    }
    (key_words, salt_words)
}

/// Batch-hash passwords at one cost with caller-supplied salts, returning the
/// raw 24-byte ciphertexts in input order.
///
/// This is the low-level SIMD entry point: the runtime-detected backend
/// kernel is resolved once, then the inputs are hashed in lane-width chunks.
/// An empty input returns an empty `Vec` without touching any kernel.
///
/// # Errors
///
/// [`CostNotAllowed`](BcryptError::CostNotAllowed) if `cost` is outside
/// `4..=31` (checked once, up front);
/// [`BatchLengthMismatch`](BcryptError::BatchLengthMismatch) if the two
/// slices differ in length.
#[cfg(feature = "alloc")]
pub fn bcrypt_many(
    cost: u32,
    passwords: &[&[u8]],
    salts: &[[u8; 16]],
) -> BcryptResult<Vec<[u8; 24]>> {
    check_batch(cost, passwords, salts)?;
    if passwords.is_empty() {
        // Return before `backend()` is evaluated: on a cold process an
        // empty batch would otherwise pay the one-time width shootout
        // (~0.4 s on AVX-512 hosts) to hash nothing.
        return Ok(Vec::new());
    }
    let (mut key_words, salt_words) = prep_words(passwords, salts);
    let mut outs = alloc::vec![[0u8; 24]; passwords.len()];
    // SAFETY: `crate::eks::backend()` only ever names a backend this CPU
    // advertises, which is the whole contract `bcrypt_many_inner` asks for.
    unsafe { bcrypt_many_inner(crate::eks::backend(), cost, &key_words, &salt_words, &mut outs) };
    wipe_words(key_words.as_flattened_mut());
    Ok(outs)
}

/// [`bcrypt_many`] with the backend chosen by the caller instead of the
/// runtime cascade — the differential-testing hook behind `internal-api`.
///
/// Everything else (cost and length validation, key preparation, output
/// order, empty-batch behaviour) is identical to [`bcrypt_many`].
///
/// # Safety
///
/// The caller must ensure `backend.is_available()` holds on this CPU: the
/// resolved kernel is a `#[target_feature]` function and calling it without
/// its instruction set is undefined behaviour (`SIGILL` in practice). Note
/// that an off-architecture backend resolves to the scalar kernel, so on an
/// off-arch build any backend is *sound* to pass — but that is not the
/// contract, and on-arch it will fault.
///
/// # Errors
///
/// Same as [`bcrypt_many`].
///
/// # Examples
///
/// Check availability, then make the safety promise:
///
/// ```
/// use bcrypt_rust::{Backend, BcryptResult};
/// use bcrypt_rust::__internal::bcrypt_many_with_backend;
///
/// # fn run() -> BcryptResult<()> {
/// let backend = Backend::Scalar; // always available
/// assert!(backend.is_available());
/// let passwords: [&[u8]; 2] = [b"alpha", b"bravo"];
/// let salts = [[7u8; 16], [9u8; 16]];
/// // SAFETY: `backend.is_available()` was asserted just above.
/// let outs = unsafe { bcrypt_many_with_backend(backend, 4, &passwords, &salts)? };
/// assert_eq!(outs.len(), 2);
/// #     Ok(())
/// # }
/// # run().unwrap();
/// ```
///
/// The promise is required — this does not compile:
///
/// ```compile_fail
/// use bcrypt_rust::Backend;
/// use bcrypt_rust::__internal::bcrypt_many_with_backend;
///
/// let passwords: [&[u8]; 1] = [b"alpha"];
/// let salts = [[7u8; 16]];
/// // ERROR: call to an unsafe function requires an unsafe block.
/// let _ = bcrypt_many_with_backend(Backend::Scalar, 4, &passwords, &salts);
/// ```
#[cfg(all(feature = "internal-api", feature = "alloc"))]
pub unsafe fn bcrypt_many_with_backend(
    backend: Backend,
    cost: u32,
    passwords: &[&[u8]],
    salts: &[[u8; 16]],
) -> BcryptResult<Vec<[u8; 24]>> {
    check_batch(cost, passwords, salts)?;
    let (mut key_words, salt_words) = prep_words(passwords, salts);
    let mut outs = alloc::vec![[0u8; 24]; passwords.len()];
    // SAFETY: the caller guarantees `backend.is_available()` — this
    // function's documented safety contract, forwarded to
    // `bcrypt_many_inner` unchanged.
    unsafe { bcrypt_many_inner(backend, cost, &key_words, &salt_words, &mut outs) };
    wipe_words(key_words.as_flattened_mut());
    Ok(outs)
}

/// [`bcrypt_many`], but returns parsed [`HashParts`] instead of raw
/// ciphertexts.
///
/// # Errors
///
/// Same as [`bcrypt_many`].
#[cfg(feature = "alloc")]
pub fn hash_many_with_salts(
    passwords: &[&[u8]],
    salts: &[[u8; 16]],
    cost: u32,
) -> BcryptResult<Vec<HashParts>> {
    let outs = bcrypt_many(cost, passwords, salts)?;
    Ok(outs
        .iter()
        .zip(salts.iter())
        .map(|(out, salt)| {
            let mut hash = [0u8; 23];
            hash.copy_from_slice(&out[..23]);
            HashParts::new(cost, *salt, hash)
        })
        .collect())
}

/// Batch-hash passwords at one cost with fresh random salts, returning the
/// 60-byte `$2b$` hash strings in input order.
///
/// One salt per password is drawn from the OS CSPRNG up front (hence `std`),
/// then the batch runs through the SIMD path.
///
/// # Errors
///
/// [`CostNotAllowed`](BcryptError::CostNotAllowed) if `cost` is outside
/// `4..=31`; [`Rand`](BcryptError::Rand) if the OS entropy source fails.
#[cfg(feature = "std")]
pub fn hash_many(passwords: &[&[u8]], cost: u32) -> BcryptResult<Vec<String>> {
    check_cost(cost)?;
    let mut salts: Vec<[u8; 16]> = Vec::with_capacity(passwords.len());
    for _ in passwords {
        salts.push(random_salt()?);
    }
    let parts = hash_many_with_salts(passwords, &salts, cost)?;
    Ok(parts
        .iter()
        .map(|p| p.format_for_version(Version::TwoB))
        .collect())
}

/// One [`verify_many`] work item: input position, salt, expected payload.
#[cfg(feature = "alloc")]
type VerifyItem = (usize, [u8; 16], [u8; 23]);

/// [`verify_many`]'s grouping table: one entry per distinct cost.
#[cfg(feature = "alloc")]
type CostGroups = Vec<(u32, Vec<VerifyItem>)>;

/// Verify a batch of password/hash-string pairs, one [`BcryptResult`] per
/// input position.
///
/// Per-item failures stay per-item: a malformed hash string yields
/// [`InvalidHash`](BcryptError::InvalidHash) at its position while the rest
/// of the batch verifies normally. A `passwords` slice shorter than `hashes`
/// yields [`BatchLengthMismatch`](BcryptError::BatchLengthMismatch) at the
/// missing positions (its `salts` field carries `hashes.len()` here).
///
/// Surviving items are grouped by cost — a linear scan over a
/// `Vec<(u32, …)>`: costs are `4..=31`, so at most 28 groups exist and a map
/// would buy nothing — and each group runs through the SIMD chunk loop, so
/// mixed-cost batches work in one call. Every verdict uses the same
/// constant-time compare as [`verify`].
#[cfg(feature = "alloc")]
pub fn verify_many(passwords: &[&[u8]], hashes: &[&str]) -> Vec<BcryptResult<bool>> {
    let mut results: Vec<BcryptResult<bool>> = Vec::with_capacity(hashes.len());
    let mut groups: CostGroups = Vec::new();
    for (i, hash) in hashes.iter().enumerate() {
        if passwords.get(i).is_none() {
            results.push(Err(BcryptError::BatchLengthMismatch {
                passwords: passwords.len(),
                salts: hashes.len(),
            }));
            continue;
        }
        match hash.parse::<HashParts>() {
            Ok(parts) => {
                let cost = parts.get_cost();
                let pos = match groups.iter().position(|&(c, _)| c == cost) {
                    Some(p) => p,
                    None => {
                        groups.push((cost, Vec::new()));
                        groups.len() - 1
                    }
                };
                groups[pos]
                    .1
                    .push((i, parts.get_salt_raw(), parts.get_hash()));
                // Placeholder; every grouped index is overwritten below.
                results.push(Ok(false));
            }
            Err(e) => results.push(Err(e)),
        }
    }
    for (cost, items) in &groups {
        let mut key_words: Vec<[u32; 18]> = Vec::with_capacity(items.len());
        let mut salt_words: Vec<[u32; 4]> = Vec::with_capacity(items.len());
        for &(i, salt, _) in items {
            // `i` entered the group only after `passwords.get(i)` succeeded.
            let (mut key, key_len) = padded_key(passwords[i]);
            key_words.push(crate::eks::expand_key_words(&key, key_len));
            wipe_key(&mut key);
            salt_words.push(crate::eks::salt_words(&salt));
        }
        let mut outs = alloc::vec![[0u8; 24]; items.len()];
        // SAFETY: same as in `bcrypt_many` — `crate::eks::backend()` only
        // names a backend this CPU advertises.
        unsafe { bcrypt_many_inner(crate::eks::backend(), *cost, &key_words, &salt_words, &mut outs) };
        for (j, &(i, _, expected)) in items.iter().enumerate() {
            let mut computed = [0u8; 23];
            computed.copy_from_slice(&outs[j][..23]);
            results[i] = Ok(constant_time_eq(&computed, &expected));
        }
        wipe_words(key_words.as_flattened_mut());
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;

    /// OpenBSD's vector hash of "U*U" at cost 5.
    const VECTOR_2A: &str = "$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW";

    #[test]
    fn padded_key_edge_cases() {
        // Empty password: nothing copied, the stream is just the NUL.
        let (key, len) = padded_key(b"");
        assert_eq!(len, 1);
        assert!(key.iter().all(|&b| b == 0));

        // 71 bytes: the NUL terminator is the last byte, stream covers all.
        let pw71 = [b'a'; 71];
        let (key, len) = padded_key(&pw71);
        assert_eq!(len, 72);
        assert_eq!(&key[..71], &pw71[..]);
        assert_eq!(key[71], 0);

        // 72 bytes: the buffer is full; the NUL never participates.
        let pw72 = [b'a'; 72];
        let (key, len) = padded_key(&pw72);
        assert_eq!(len, 72);
        assert_eq!(key, pw72);

        // 73 and 100 bytes: identical buffers and stream lengths to the
        // 72-byte case.
        for too_long in [&[b'a'; 73][..], &[b'a'; 100][..]] {
            let (key_long, len_long) = padded_key(too_long);
            assert_eq!(len_long, 72);
            assert_eq!(key_long, key);
        }
    }

    // Five cost-4 hashes; Miri interprets one in ~20 s, so this functional
    // coverage is native-owned (padded_key itself is unit-tested above).
    #[cfg(not(miri))]
    #[test]
    fn passwords_longer_than_72_bytes_hash_like_their_prefix() {
        let salt = [0x42; 16];
        let pw72 = [b'x'; 72];
        let pw73 = [b'x'; 73];
        let pw100 = [b'x'; 100];
        assert_eq!(bcrypt(4, salt, &pw72), bcrypt(4, salt, &pw73));
        assert_eq!(bcrypt(4, salt, &pw72), bcrypt(4, salt, &pw100));
        // …but a difference inside the first 72 bytes changes everything.
        let mut other = pw72;
        other[71] = b'y';
        assert_ne!(bcrypt(4, salt, &pw72), bcrypt(4, salt, &other));
    }

    #[test]
    fn openbsd_vector_end_to_end() {
        assert_eq!(verify(b"U*U", VECTOR_2A), Ok(true));
        // Every verify hashes at cost 5 (~40 s under the interpreter):
        // Miri runs the happy path, natives own the negatives.
        #[cfg(not(miri))]
        {
            assert_eq!(verify(b"U*U*", VECTOR_2A), Ok(false));
            assert_eq!(verify(b"", VECTOR_2A), Ok(false));
        }

        // The same vector through the hashing side, with the salt decoded
        // out of the string.
        let salt = crate::base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("salt decodes");
        let bytes = hash_with_salt_bytes(b"U*U", 5, salt).expect("valid cost");
        assert_eq!(
            core::str::from_utf8(&bytes).expect("ASCII"),
            "$2b$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW"
        );
    }

    // Four cost-4 hashes; native-owned (see openbsd_vector_end_to_end).
    #[cfg(not(miri))]
    #[test]
    fn round_trip_and_wrong_password() {
        let salt = [7u8; 16];
        let bytes = hash_with_salt_bytes(b"hunter2", 4, salt).expect("valid cost");
        let s = core::str::from_utf8(&bytes).expect("ASCII");
        assert!(s.starts_with("$2b$04$"));
        assert_eq!(s.len(), 60);
        assert_eq!(verify(b"hunter2", s), Ok(true));
        assert_eq!(verify(b"hunter3", s), Ok(false));
        assert_eq!(verify(b"", s), Ok(false));
    }

    // Six cost-4 hashes; native-owned (see openbsd_vector_end_to_end).
    #[cfg(not(miri))]
    #[test]
    fn all_version_markers_verify_the_same() {
        let salt = [3u8; 16];
        let parts = hash_with_salt(b"correct horse", 4, salt).expect("valid cost");
        for version in [Version::TwoA, Version::TwoB, Version::TwoX, Version::TwoY] {
            let bytes = parts.format_bytes(version);
            let s = core::str::from_utf8(&bytes).expect("ASCII");
            assert_eq!(verify(b"correct horse", s), Ok(true), "{version}");
        }
        assert_eq!(parts.get_cost(), 4);
        assert_eq!(parts.get_salt_raw(), salt);
        let raw = bcrypt(4, salt, b"correct horse");
        assert_eq!(&parts.get_hash()[..], &raw[..23]);
    }

    #[test]
    fn cost_arguments_are_validated() {
        let salt = [0u8; 16];
        assert!(check_cost(MIN_COST).is_ok());
        assert!(check_cost(MAX_COST).is_ok());
        for bad in [0, 3, 32, 100] {
            assert_eq!(
                hash_with_salt(b"pw", bad, salt),
                Err(BcryptError::CostNotAllowed(bad))
            );
            assert_eq!(
                hash_with_salt_bytes(b"pw", bad, salt),
                Err(BcryptError::CostNotAllowed(bad))
            );
            assert_eq!(
                non_truncating_hash_with_salt(b"pw", bad, salt),
                Err(BcryptError::CostNotAllowed(bad))
            );
        }
        // A bad cost inside a string is a parse error, not CostNotAllowed,
        // and verify never computes with it.
        assert_eq!(
            verify(b"pw", "$2b$03$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW"),
            Err(BcryptError::InvalidHash("cost"))
        );
        assert_eq!(
            verify(b"pw", "$2b$32$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW"),
            Err(BcryptError::InvalidHash("cost"))
        );
    }

    #[test]
    fn non_truncating_variants_reject_72_bytes_and_up() {
        let salt = [5u8; 16];
        #[cfg(not(miri))]
        let pw71 = [b'a'; 71];
        let pw72 = [b'a'; 72];
        let pw100 = [b'a'; 100];

        // The hashing asserts are native-owned (~20 s each under Miri).
        #[cfg(not(miri))]
        assert!(non_truncating_hash_with_salt(pw71, 4, salt).is_ok());
        assert_eq!(
            non_truncating_hash_with_salt(pw72, 4, salt),
            Err(BcryptError::Truncation(73))
        );
        assert_eq!(
            non_truncating_hash_with_salt(pw100, 4, salt),
            Err(BcryptError::Truncation(101))
        );
        assert_eq!(
            non_truncating_hash_with_salt_bytes(pw72, 4, salt),
            Err(BcryptError::Truncation(73))
        );
        assert_eq!(
            non_truncating_verify(pw72, VECTOR_2A),
            Err(BcryptError::Truncation(73))
        );
        // Truncation is reported before a malformed hash is looked at.
        assert_eq!(
            non_truncating_verify(pw72, "not a hash"),
            Err(BcryptError::Truncation(73))
        );

        // At the boundary the two families agree.
        #[cfg(not(miri))]
        {
            let a = hash_with_salt(pw71, 4, salt).expect("valid cost");
            let b = non_truncating_hash_with_salt(pw71, 4, salt).expect("71 bytes pass");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn verify_rejects_malformed_hashes() {
        assert_eq!(
            verify(b"pw", "not a hash"),
            Err(BcryptError::InvalidHash("length"))
        );
    }

    #[test]
    fn constant_time_eq_basics() {
        let a = [0xABu8; 23];
        let mut b = a;
        assert!(constant_time_eq(&a, &b));
        b[22] ^= 1;
        assert!(!constant_time_eq(&a, &b));
        b[22] ^= 1;
        b[0] ^= 0x80;
        assert!(!constant_time_eq(&a, &b));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bcrypt_many_matches_bcrypt_for_sizes_0_to_17() {
        // 306 cost-4 hashes is instant natively but many minutes under an
        // interpreter; Miri covers the same dispatch shapes (empty, single,
        // multi-lane) with sizes 0..=2.
        let max = if cfg!(miri) { 2 } else { 17 };
        for n in 0..=max {
            let store: Vec<[u8; 8]> = (0..n)
                .map(|i| {
                    let mut a = [0xA5u8; 8];
                    a[0] = i as u8;
                    a
                })
                .collect();
            let pws: Vec<&[u8]> = store.iter().map(|a| &a[..]).collect();
            let salts: Vec<[u8; 16]> = (0..n)
                .map(|i| {
                    let mut s = [0u8; 16];
                    s[0] = i as u8;
                    s[15] = 0xFF - i as u8;
                    s
                })
                .collect();
            let batch = bcrypt_many(4, &pws, &salts).expect("valid cost");
            assert_eq!(batch.len(), n);
            for (i, ((got, &salt), &pw)) in
                batch.iter().zip(&salts).zip(&pws).enumerate()
            {
                assert_eq!(*got, bcrypt(4, salt, pw), "size {n} lane {i}");
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bcrypt_many_empty_and_mismatched_inputs() {
        let empty_p: [&[u8]; 0] = [];
        let empty_s: [[u8; 16]; 0] = [];
        assert_eq!(bcrypt_many(4, &empty_p, &empty_s), Ok(Vec::new()));
        // Cost is validated before anything else, even for empty input.
        assert_eq!(
            bcrypt_many(3, &empty_p, &empty_s),
            Err(BcryptError::CostNotAllowed(3))
        );
        let pws: [&[u8]; 2] = [b"a", b"b"];
        let salts = [[0u8; 16]];
        assert_eq!(
            bcrypt_many(4, &pws, &salts),
            Err(BcryptError::BatchLengthMismatch {
                passwords: 2,
                salts: 1
            })
        );
        assert_eq!(
            hash_many_with_salts(&pws, &salts, 4),
            Err(BcryptError::BatchLengthMismatch {
                passwords: 2,
                salts: 1
            })
        );
    }

    // Six cost-4 hashes; native-owned (see openbsd_vector_end_to_end).
    #[cfg(all(feature = "alloc", not(miri)))]
    #[test]
    fn hash_many_with_salts_round_trip() {
        let pws: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
        let salts = [[1u8; 16], [2u8; 16], [3u8; 16]];
        let parts = hash_many_with_salts(&pws, &salts, 4).expect("valid cost");
        assert_eq!(parts.len(), 3);
        for ((part, &salt), pw) in parts.iter().zip(&salts).zip(&pws) {
            assert_eq!(part.get_cost(), 4);
            assert_eq!(part.get_salt_raw(), salt);
            let bytes = part.format_bytes(Version::TwoB);
            let s = core::str::from_utf8(&bytes).expect("ASCII");
            assert_eq!(verify(pw, s), Ok(true));
        }
    }

    // Nine hashes across cost 4 and 5; native-owned (see above).
    #[cfg(all(feature = "alloc", not(miri)))]
    #[test]
    fn verify_many_mixed_costs_with_failures() {
        let parts4 = hash_with_salt(b"correct horse", 4, [1u8; 16]).expect("valid cost");
        let parts5 = hash_with_salt(b"battery staple", 5, [2u8; 16]).expect("valid cost");
        let h4 = parts4.format_bytes(Version::TwoB);
        let h5 = parts5.format_bytes(Version::TwoA);
        let h4 = core::str::from_utf8(&h4).expect("ASCII");
        let h5 = core::str::from_utf8(&h5).expect("ASCII");

        let pws: [&[u8]; 4] = [
            b"correct horse",
            b"battery staple",
            b"wrong",
            b"correct horse",
        ];
        let hashes: [&str; 4] = [h4, h5, h5, "not a real hash"];
        let results = verify_many(&pws, &hashes);
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], Ok(true));
        assert_eq!(results[1], Ok(true), "cost-5 group in the same call");
        assert_eq!(results[2], Ok(false), "wrong password, right hash");
        assert_eq!(
            results[3],
            Err(BcryptError::InvalidHash("length")),
            "malformed hash stays a per-item error"
        );

        // A shorter passwords slice is a per-item BatchLengthMismatch.
        let results = verify_many(&pws[..1], &hashes[..2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Ok(true));
        assert_eq!(
            results[1],
            Err(BcryptError::BatchLengthMismatch {
                passwords: 1,
                salts: 2
            })
        );

        // Empty batch: no results, no kernel.
        assert!(verify_many(&[], &[]).is_empty());
    }

    // Seven cost-4 hashes; native-owned (see openbsd_vector_end_to_end).
    #[cfg(all(feature = "std", not(miri)))]
    #[test]
    fn hash_family_with_random_salts() {
        let h = hash(b"hunter2", 4).expect("hashing works");
        assert_eq!(h.len(), 60);
        assert!(h.starts_with("$2b$04$"));
        assert_eq!(verify(b"hunter2", &h), Ok(true));
        assert_eq!(verify(b"hunter3", &h), Ok(false));

        let hb = hash_bytes(b"hunter2", 4).expect("hashing works");
        assert_eq!(verify(b"hunter2", core::str::from_utf8(&hb).expect("ASCII")), Ok(true));

        let parts = hash_with_result(b"hunter2", 4).expect("hashing works");
        assert_eq!(parts.get_cost(), 4);

        // Random salts: two hashes of one password differ.
        assert_ne!(hash(b"hunter2", 4).expect("hashing works"), h);

        // Cost is still validated, and truncation still reported.
        assert_eq!(hash(b"pw", 3), Err(BcryptError::CostNotAllowed(3)));
        assert_eq!(hash_bytes(b"pw", 32), Err(BcryptError::CostNotAllowed(32)));
        assert_eq!(
            non_truncating_hash([b'a'; 72], 4),
            Err(BcryptError::Truncation(73))
        );
        assert_eq!(
            non_truncating_hash_bytes([b'a'; 72], 4),
            Err(BcryptError::Truncation(73))
        );
        assert_eq!(
            non_truncating_hash_with_result([b'a'; 72], 4),
            Err(BcryptError::Truncation(73))
        );
    }

    // Six cost-4 hashes; native-owned (see openbsd_vector_end_to_end).
    #[cfg(all(feature = "std", not(miri)))]
    #[test]
    fn hash_many_round_trip() {
        let pws: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
        let hashes = hash_many(&pws, 4).expect("hashing works");
        assert_eq!(hashes.len(), 3);
        for (pw, h) in pws.iter().zip(&hashes) {
            assert!(h.starts_with("$2b$04$"));
            assert_eq!(verify(pw, h), Ok(true));
        }
        // Random salts: distinct strings for distinct passwords.
        assert_ne!(hashes[0], hashes[1]);

        assert_eq!(hash_many(&pws, 3), Err(BcryptError::CostNotAllowed(3)));
        let empty: [&[u8]; 0] = [];
        assert_eq!(hash_many(&empty, 4), Ok(Vec::new()));
    }

    /// The `2 x cores` threshold: below it the parallel path never engages,
    /// at it the worker count equals the core count.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_workers_threshold() {
        let cores = std::thread::available_parallelism().map_or(1, core::num::NonZero::get);
        assert_eq!(parallel_workers(0), 1);
        assert_eq!(parallel_workers(2 * cores - 1), 1);
        // `available_parallelism` never yields 0, so `cores` is also the
        // expected count when the single-core host stays sequential.
        assert_eq!(parallel_workers(2 * cores), cores);
        assert_eq!(parallel_workers(1_000_000), cores);
    }

    /// The parallel path must be byte-identical to hashing each item on its
    /// own: chunking only changes which duplicate lane-padding work is
    /// discarded, never an item's output. The last two sizes sit at and just
    /// above the `2 x cores` threshold, so the threaded path itself runs on
    /// any multi-core host.
    #[cfg(all(feature = "alloc", feature = "parallel"))]
    #[test]
    fn parallel_batch_matches_single_hashes() {
        let cores = std::thread::available_parallelism().map_or(1, core::num::NonZero::get);
        let sizes = [0usize, 1, 2, 3, 7, 8, 16, 17, 33, 64];
        for &n in sizes.iter().chain([2 * cores, 2 * cores + 1].iter()) {
            let store: Vec<Vec<u8>> = (0..n)
                .map(|i| alloc::vec![(i as u8).wrapping_mul(37); 4 + i % 29])
                .collect();
            let pws: Vec<&[u8]> = store.iter().map(Vec::as_slice).collect();
            let salts: Vec<[u8; 16]> = (0..n)
                .map(|i| {
                    let mut s = [0x5Au8; 16];
                    s[0] = i as u8;
                    s
                })
                .collect();
            let batch = bcrypt_many(4, &pws, &salts).expect("valid cost");
            assert_eq!(batch.len(), n);
            for (i, ((got, &salt), &pw)) in batch.iter().zip(&salts).zip(&pws).enumerate() {
                assert_eq!(*got, bcrypt(4, salt, pw), "size {n} lane {i}");
            }
        }
    }
}
