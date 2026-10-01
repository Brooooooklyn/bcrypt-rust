//! Fuzzes cost validation and the whole hash pipeline with tiny,
//! fuzzer-derived parameters — the raw `bcrypt`, the string-format wrappers,
//! and the batch entry points (which drive every SIMD backend the CPU has,
//! via the normal dispatch).
//!
//! The property: any input shape must hash without a panic, hashing must be
//! deterministic (same input, same output), and batch must agree with single.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 18 {
        return;
    }
    // Cost 4..=6: valid floor, fast enough for the fuzzer (cost 6 ≈ ms).
    let cost = 4 + u32::from(data[0] % 3);
    // Passwords straddle the 72-byte truncation boundary deliberately.
    let pw_len = usize::from(data[1]) % 100; // 0..=99
    let pw = &data[2..2 + pw_len.min(data.len() - 2 - 16)];
    let salt_off = data.len() - 16;
    let salt: [u8; 16] = data[salt_off..].try_into().expect("16-byte tail");

    // Single-hash raw core: deterministic on repeat.
    let a = bcrypt_rust::bcrypt(cost, salt, pw);
    let b = bcrypt_rust::bcrypt(cost, salt, pw);
    assert_eq!(a, b, "same input must give the same hash");

    // Batch of 3 with the second lane equal to the single-hash input: batch
    // and single must agree per lane, whatever backend the CPU picked.
    let pw2: &[u8] = b"fuzz";
    let pw3: &[u8] = b"";
    let outs = bcrypt_rust::bcrypt_many(cost, &[pw, pw2, pw3], &[salt, salt, salt])
        .expect("cost 4..=6 is valid");
    assert_eq!(outs[0], a, "batch lane diverged from single hash");
    assert_eq!(outs[1], bcrypt_rust::bcrypt(cost, salt, pw2));
    assert_eq!(outs[2], bcrypt_rust::bcrypt(cost, salt, pw3));

    // String layer must not panic either (alloc formatting path).
    let parts = bcrypt_rust::hash_with_salt(pw, cost, salt).expect("valid cost");
    let s = parts.to_string();
    assert_eq!(bcrypt_rust::verify(pw, &s).ok(), Some(true));
});
