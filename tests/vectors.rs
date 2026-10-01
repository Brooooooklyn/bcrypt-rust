//! The authoritative bcrypt test-vector suite.
//!
//! Vectors are transcribed from the consolidated table in the ecosystem
//! survey (`docs/research/rust-bcrypt-crates.md` §6). Sources:
//!
//! * **JB** — jBCrypt `TestBCrypt.java` (djmdjm/jBCrypt), 20 vectors;
//! * **OW** — Openwall `crypt_blowfish-1.3` `wrapper.c` self-test array
//!   (vendored under `crypt_blowfish/`, public domain);
//! * **RB** — the `bcrypt` crate's own test suite (pyca/node/Go interop);
//! * **X** — generated with pyca/bcrypt for the survey and cross-verified
//!   against the `bcrypt` crate.
//!
//! Three survey rows were reclassified against the vendored crypt_blowfish
//! source (the authority on what its own `$2x$` / 8-bit-`$2a$` emulations
//! compute — see `BF_set_key` in `crypt_blowfish/crypt_blowfish.c`), and the
//! reclassification was confirmed empirically against this crate:
//!
//! * Survey positive #33 (`$2a$05$/OK…ZC1JEJ…`) is crypt_blowfish's `$2a$`
//!   anti-collision "safety XOR" output, not the correct algorithm. This
//!   crate deliberately does not reproduce that deviation, so the row moves
//!   to `DIVERGENT`; its place in `VECTORS` is taken by the password's
//!   correct-algorithm `$2y$` sibling — an authentic `wrapper.c` row the
//!   survey omitted.
//! * Survey divergent #44 (`$2x$…CE5el…` for `\xff\xff\xa3`): the sign
//!   extension bug is benign for this password (the appended NUL flushes the
//!   sign-extended bits out of every key word), so crypt_blowfish's `$2x$`
//!   output *is* the correct hash — identical to positives #29/#30. It must
//!   verify here; it stays in `DIVERGENT`, pinned with `must_verify: true`.
//! * Survey divergent #47 (`$2y$…nRht2l…` for `\xff\xa3345`): `$2y$` is the
//!   correct algorithm (the survey's own note says "matches Rust"), so it
//!   must verify; its `$2x$` sibling is the buggy one. Also pinned with
//!   `must_verify: true`.

// The whole suite exercises the string layer (`HashParts`, `bcrypt_many`,
// `verify_many`), which is `alloc`-gated — with alloc off this file is empty.
#![cfg(feature = "alloc")]

use std::collections::BTreeMap;

use bcrypt_rust::{
    BcryptError, HashParts, Version, bcrypt, bcrypt_many, hash, hash_with_salt, verify,
    verify_many,
};

/// One (password, hash) pair that the correct bcrypt algorithm reproduces.
struct Vector {
    password: &'static [u8],
    hash: &'static str,
}

/// `reps` copies of `block` followed by `tail`, assembled at compile time.
/// The const assertion pins the exact length, so a miscounted repetition is
/// a compile error, not a wrong password.
const fn build<const N: usize>(block: &[u8], reps: usize, tail: &[u8]) -> [u8; N] {
    assert!(block.len() * reps + tail.len() == N);
    let mut out = [0u8; N];
    let mut i = 0;
    while i < reps {
        let mut j = 0;
        while j < block.len() {
            out[i * block.len() + j] = block[j];
            j += 1;
        }
        i += 1;
    }
    let mut k = 0;
    while k < tail.len() {
        out[block.len() * reps + k] = tail[k];
        k += 1;
    }
    out
}

/// #26: 72 × 0xAA, then a suffix past the 72-byte truncation point.
static PW26: [u8; 107] = build(b"\xaa", 72, b"chars after 72 are ignored as usual");
/// #27: `\xaa\x55` × 36 — exactly 72 bytes, high bytes.
static PW27: [u8; 72] = build(b"\xaa\x55", 36, b"");
/// #28: `\x55\xaa\xff` × 24 — exactly 72 bytes, contains 0xFF.
static PW28: [u8; 72] = build(b"\x55\xaa\xff", 24, b"");
/// #36: 100 × "x" — the pyca truncation vector.
static X100: [u8; 100] = build(b"x", 100, b"");
/// #42: "hunter2" × 10 + "ab" — exactly 72 bytes.
static PW42: [u8; 72] = build(b"hunter2", 10, b"ab");

/// The 42 positive vectors: every one must verify and be byte-exactly
/// reproducible from its parsed cost + salt. Numbered as in the survey.
static VECTORS: &[Vector] = &[
    Vector { password: b"", hash: "$2a$06$DCq7YPn5Rq63x1Lad4cll.TV4S6ytwfsfvkgY8jIucDrjc8deX1s." }, // #1  JB — empty password
    Vector { password: b"", hash: "$2a$08$HqWuK6/Ng6sg9gQzbLrgb.Tl.ZHfXLhvt/SgVyWhQqgqcZ7ZuUtye" }, // #2  JB
    Vector { password: b"", hash: "$2a$10$k1wbIrmNyFAPwPVPSVa/zecw2BCEnBwVS2GbrmgzxFUOqW9dk4TCW" }, // #3  JB
    Vector { password: b"", hash: "$2a$12$k42ZFHFWqBp3vWli.nIn8uYyIkbvYRvodzbfbK18SSsY.CsIQPlxO" }, // #4  JB — cost 12
    Vector { password: b"a", hash: "$2a$06$m0CrhHm10qJ3lXRY.5zDGO3rS2KdeeWLuGmsfGlMfOxih58VYVfxe" }, // #5  JB
    Vector { password: b"a", hash: "$2a$08$cfcvVd2aQ8CMvoMpP2EBfeodLEkkFJ9umNEfPD18.hUF62qqlC/V." }, // #6  JB
    Vector { password: b"a", hash: "$2a$10$k87L/MF28Q673VKh8/cPi.SUl7MU/rWuSiIDDFayrKk/1tBsSQu4u" }, // #7  JB
    Vector { password: b"a", hash: "$2a$12$8NJH3LsPrANStV6XtBakCez0cKHXVxmvxIlcz785vxAIZrihHZpeS" }, // #8  JB — cost 12
    Vector { password: b"abc", hash: "$2a$06$If6bvum7DFjUnE9p2uDeDu0YHzrHM6tf.iqN8.yx.jNN1ILEf7h0i" }, // #9  JB
    Vector { password: b"abc", hash: "$2a$08$Ro0CUfOqk6cXEKf3dyaM7OhSCvnwM9s4wIX9JeLapehKK5YdLxKcm" }, // #10 JB
    Vector { password: b"abc", hash: "$2a$10$WvvTPHKwdBJ3uk0Z37EMR.hLA2W6N9AEBhEgrAOljy2Ae5MtaSIUi" }, // #11 JB
    Vector { password: b"abc", hash: "$2a$12$EXRkfkdmXn2gzds2SSitu.MW9.gAVqa9eLS1//RYtYCmB1eLHg.9q" }, // #12 JB — cost 12
    Vector { password: b"abcdefghijklmnopqrstuvwxyz", hash: "$2a$06$.rCVZVOThsIa97pEDOxvGuRRgzG64bvtJ0938xuqzv18d3ZpQhstC" }, // #13 JB
    Vector { password: b"abcdefghijklmnopqrstuvwxyz", hash: "$2a$08$aTsUwsyowQuzRrDqFflhgekJ8d9/7Z3GV3UcgvzQW3J5zMyrTvlz." }, // #14 JB
    Vector { password: b"abcdefghijklmnopqrstuvwxyz", hash: "$2a$10$fVH8e28OQRj9tqiDXs1e1uxpsjN0c7II7YPKXua2NAKYvM6iQk7dq" }, // #15 JB
    Vector { password: b"abcdefghijklmnopqrstuvwxyz", hash: "$2a$12$D4G5f18o7aMMfwasBL7GpuQWuP3pkrZrOAnqP.bmezbMng.QwJ/pG" }, // #16 JB — cost 12
    Vector { password: b"~!@#$%^&*()      ~!@#$%^&*()PNBFRD", hash: "$2a$06$fPIsBO8qRqkjj273rfaOI.HtSV9jLDpTbZn782DC6/t7qT67P6FfO" }, // #17 JB — symbols/spaces
    Vector { password: b"~!@#$%^&*()      ~!@#$%^&*()PNBFRD", hash: "$2a$08$Eq2r4G/76Wv39MzSX262huzPz612MZiYHVUJe/OcOql2jo4.9UxTW" }, // #18 JB
    Vector { password: b"~!@#$%^&*()      ~!@#$%^&*()PNBFRD", hash: "$2a$10$LgfYWkbzEvQ4JakH7rOvHe0y8pHKF9OaFgwUZ2q7W2FFZmZzJYlfS" }, // #19 JB
    Vector { password: b"~!@#$%^&*()      ~!@#$%^&*()PNBFRD", hash: "$2a$12$WApznUOJfkEGSmYRfnkrPOr466oFDCaj4b6HY3EXGvfxm43seyhgC" }, // #20 JB — cost 12
    Vector { password: b"U*U", hash: "$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW" }, // #21 OW
    Vector { password: b"U*U*", hash: "$2a$05$CCCCCCCCCCCCCCCCCCCCC.VGOzA784oUp/Z0DY336zx7pLYAy0lwK" }, // #22 OW
    Vector { password: b"U*U*U", hash: "$2a$05$XXXXXXXXXXXXXXXXXXXXXOAcXxm9kjPGEMsLznoKqmqw7tc8WCx4a" }, // #23 OW
    Vector { password: b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789chars after 72 are ignored", hash: "$2a$05$abcdefghijklmnopqrstuu5s2v8.iXieOjg/.AySBTTZIIVFJeBui" }, // #24 OW — >72 bytes, truncation
    Vector { password: b"", hash: "$2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy" }, // #25 OW — empty password
    Vector { password: &PW26, hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.swQOIzjOiJ9GHEPuhEkvqrUyvWhEMx6" }, // #26 OW — 72×0xAA + truncation
    Vector { password: &PW27, hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.R9xrDjiycxMbQE2bp.vgqlYpW5wx2yy" }, // #27 OW — 72-byte edge, high bytes
    Vector { password: &PW28, hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.9tQZzcJfm3uj2NvJ/n5xkhpqLrMpWCe" }, // #28 OW — 72-byte edge, 0xFF bytes
    Vector { password: b"\xff\xff\xa3", hash: "$2b$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e" }, // #29 OW — $2b$, 0xFF bytes
    Vector { password: b"\xff\xff\xa3", hash: "$2y$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e" }, // #30 OW — $2y$ == $2b$
    Vector { password: b"\xa3", hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq" }, // #31 OW — 8-bit, correct alg
    Vector { password: b"\xa3", hash: "$2y$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq" }, // #32 OW — $2y$ == $2a$ here
    Vector { password: b"\xff\xa334\xff\xff\xff\xa3345", hash: "$2y$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi" }, // #33 OW — CORRECTED (see header): the survey's $2a$ sibling is in DIVERGENT
    Vector { password: b"hunter2", hash: "$2b$12$......................21jzCB1r6pN6rp5O2Ev0ejjTAboskKm" }, // #34 RB — cost 12, zero salt
    Vector { password: b"My S3cre7 P@55w0rd!", hash: "$2b$05$HlFShUxTu4ZHHfOLJwfmCeDj/kuKFKboanXtDJXxCC7aIPTUgxNDe" }, // #35 RB — hash_with_salt
    Vector { password: &X100, hash: "$2a$05$......................YgIDy4hFBdVlc/6LHnD9mX488r9cLd2" }, // #36 RB — truncation (pyca)
    Vector { password: b"correctbatteryhorsestapler", hash: "$2b$04$EGdrhbKUv8Oc9vGiXX0HQOxSg445d458Muh7DAHskb6QbtCvdxcie" }, // #37 RB — cost 4, pyca
    Vector { password: b"correctbatteryhorsestapler", hash: "$2a$04$n4Uy0eSnMfvnESYL.bLwuuj0U/ETSsoTpRT9GVk5bektyVVa5xnIi" }, // #38 RB — node
    Vector { password: b"\x1d\xe1\xc3\xa7\xdf\xec\x55\xc3\x72\xe3\x07\x00\xd1\xef\xbd\x18\x33\x69\x7c\xa8\x97\x4b\x90\x40\xc6\xc5\xc4\x04\xf1\x61\x6e\x87", hash: "$2a$04$tjARW6ZON3PhrAIRW2LG/u9aDw5eFdstYLR8nFCNaOQmsH9XD23w." }, // #39 RB — high bytes, Go x/crypto
    Vector { password: "★★★★★★★★".as_bytes(), hash: "$2a$05$......................CVh3qAKQwo3AyWm2sH24x.4W0jOiobK" }, // #40 X — UTF-8 (U+2605 ×8 = 24 bytes)
    Vector { password: "★★★★★★★★".as_bytes(), hash: "$2b$05$CCCCCCCCCCCCCCCCCCCCC.GKRlRE2yXEq.CPNmj6AdW2OEOSzU/GW" }, // #41 X — UTF-8, $2b$
    Vector { password: &PW42, hash: "$2a$05$......................VbcD.3tcy/UErGUKxMqc6T88xuK4rWq" }, // #42 X — 72-char ASCII edge
];

/// A crypt_blowfish bug-emulation row: `$2x$` sign extension or `$2a$` 8-bit
/// safety XOR. `must_verify` is `false` unless the emulation coincides with
/// the correct algorithm for this password (see the file header).
struct Divergent {
    password: &'static [u8],
    hash: &'static str,
    must_verify: bool,
}

/// The survey's 6 expected-divergent rows (#43–#48, transcribed exactly),
/// plus the row the survey misclassified as positive #33 (its `$2a$` hash is
/// itself a bug-emulation output). Correct-algorithm siblings of the buggy
/// rows are pinned in `SIBLINGS` below.
static DIVERGENT: &[Divergent] = &[
    // #43 OW — `$2x$` buggy sign extension; the buggy hash of "\xa3"
    // collides with the CORRECT hash of "\xff\xff\xa3" (#29/#30).
    Divergent { password: b"\xa3", hash: "$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", must_verify: false },
    // #44 OW — for "\xff\xff\xa3" the bug is benign: buggy and correct key
    // schedules coincide, so this `$2x$` output IS the correct hash
    // (== #29/#30) and must verify. The genuinely divergent `$2a$` sibling
    // is the next row.
    Divergent { password: b"\xff\xff\xa3", hash: "$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", must_verify: true },
    // #45 OW — `$2a$` 8-bit anti-collision safety XOR; a deliberate deviation
    // from the correct algorithm that must NOT reproduce.
    Divergent { password: b"\xff\xff\xa3", hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.nqd1wy.pTMdcvrRWxyiGL2eMz.2a85.", must_verify: false },
    // #46 OW — `$2x$` buggy; the correct hash of "1\xa3345" is unrelated
    // ($2y$05$/OK.fbVrR/bpIqNJ5ianF.ykhStlibHmUn6GomsWXnJvyvZx4vmvy).
    Divergent { password: b"1\xa3345", hash: "$2x$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi", must_verify: false },
    // #47 OW — `$2y$` IS the correct algorithm, so this must verify.
    // crypt_blowfish emits the identical hash under `$2a$` too (the safety
    // XOR does not trigger for this password); the buggy sibling is the
    // `$2x$` row with the "o./n25…" payload.
    Divergent { password: b"\xff\xa3345", hash: "$2y$05$/OK.fbVrR/bpIqNJ5ianF.nRht2l/HRhr6zmCp9vYUvvsqynflf9e", must_verify: true },
    // #48 OW — `$2x$` buggy (correct hash: $2y$05$/OK…E737eUK7jOqGXQUPcu5iAm8pR815Cru).
    Divergent { password: b"\xd1\x91", hash: "$2x$05$6bNw2HLQYeqHYyBfLMsv/OiwqTymGIGzFsA4hOTWebfehXHNprcAS", must_verify: false },
    // Survey positive #33, moved here: crypt_blowfish's `$2a$` safety-XOR
    // output for this password. Its correct-algorithm sibling is positive
    // vector #33 ("$2y$…o./n25…") above.
    Divergent { password: b"\xff\xa334\xff\xff\xff\xa3345", hash: "$2a$05$/OK.fbVrR/bpIqNJ5ianF.ZC1JEJ8Z4gPfpe1JOr/oyPXTWl9EFd.", must_verify: false },
];

/// (password, buggy hash, correct sibling from `VECTORS`): the buggy rows
/// whose password+salt has a correct sibling in the positive table. The
/// buggy hash supplies cost + salt; recomputing must yield the sibling.
static SIBLINGS: &[(&[u8], &str, &str)] = &[
    // #43's correct sibling is #31/#32.
    (b"\xa3", "$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", "$2a$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq"),
    // #45's correct sibling is #29/#30.
    (b"\xff\xff\xa3", "$2a$05$/OK.fbVrR/bpIqNJ5ianF.nqd1wy.pTMdcvrRWxyiGL2eMz.2a85.", "$2b$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e"),
    // The moved #33's correct sibling is positive #33.
    (b"\xff\xa334\xff\xff\xff\xa3345", "$2a$05$/OK.fbVrR/bpIqNJ5ianF.ZC1JEJ8Z4gPfpe1JOr/oyPXTWl9EFd.", "$2y$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi"),
];

/// The version marker of a hash string. `HashParts` deliberately does not
/// retain it (versions are labels), so re-formatting with the original
/// prefix needs it recovered from the string.
fn version_of(hash: &str) -> Version {
    match hash.as_bytes()[2] {
        b'a' => Version::TwoA,
        b'x' => Version::TwoX,
        b'y' => Version::TwoY,
        _ => Version::TwoB,
    }
}

/// A password that must NOT verify against `pw`'s hash: below 72 bytes an
/// appended byte changes the key stream (this also covers the empty
/// password); at 72+ bytes appending is truncated away, so mutate byte 0.
fn wrong_password(pw: &[u8]) -> Vec<u8> {
    let mut wrong = pw.to_vec();
    if wrong.len() < 72 {
        wrong.push(0x21);
    } else {
        wrong[0] ^= 0x01;
    }
    wrong
}

/// Re-encode a raw 24-byte core output into the 60-char hash string, using
/// the crate's own bcrypt base64 (reachable from integration tests via the
/// `internal-api` dev feature).
fn reencode(version: Version, cost: u32, salt: &[u8; 16], out: &[u8; 24]) -> String {
    let mut payload = [0u8; 23];
    payload.copy_from_slice(&out[..23]);
    let salt_b64 = bcrypt_rust::__internal::encode_16(salt);
    let hash_b64 = bcrypt_rust::__internal::encode_23(&payload);
    format!(
        "${}${cost:02}${}{}",
        version,
        std::str::from_utf8(&salt_b64).expect("bcrypt base64 is ASCII"),
        std::str::from_utf8(&hash_b64).expect("bcrypt base64 is ASCII"),
    )
}

#[test]
fn positive_vectors_verify() {
    assert_eq!(VECTORS.len(), 42);
    for (i, v) in VECTORS.iter().enumerate() {
        assert_eq!(
            verify(v.password, v.hash),
            Ok(true),
            "vector #{} must verify",
            i + 1
        );
        assert_eq!(
            verify(wrong_password(v.password), v.hash),
            Ok(false),
            "vector #{}: a wrong password must not verify",
            i + 1
        );
    }
}

#[test]
fn positive_vectors_recompute() {
    for (i, v) in VECTORS.iter().enumerate() {
        let parts: HashParts = v.hash.parse().expect("vector hash parses");
        let recomputed = hash_with_salt(v.password, parts.get_cost(), parts.get_salt_raw())
            .expect("cost from a valid hash");
        assert_eq!(recomputed, parts, "vector #{}", i + 1);
        assert_eq!(
            recomputed.format_for_version(version_of(v.hash)),
            v.hash,
            "vector #{} must reproduce its exact string, prefix included",
            i + 1
        );
    }
}

#[test]
fn batch_matches_single() {
    let parsed: Vec<HashParts> = VECTORS
        .iter()
        .map(|v| v.hash.parse().expect("vector hash parses"))
        .collect();

    // `bcrypt_many` takes one cost per call, so the 42 vectors run as one
    // call per distinct cost — passwords, raw salts and costs all parsed
    // from the hash strings. `verify_many` below takes all 42 in a single
    // mixed-cost call.
    let mut by_cost: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, parts) in parsed.iter().enumerate() {
        by_cost.entry(parts.get_cost()).or_default().push(i);
    }

    for (cost, idxs) in &by_cost {
        let passwords: Vec<&[u8]> = idxs.iter().map(|&i| VECTORS[i].password).collect();
        let salts: Vec<[u8; 16]> = idxs.iter().map(|&i| parsed[i].get_salt_raw()).collect();
        let outs = bcrypt_many(*cost, &passwords, &salts).expect("cost came from a valid hash");
        assert_eq!(outs.len(), idxs.len());
        for (j, &i) in idxs.iter().enumerate() {
            // The batch lane must equal the single-hash raw core exactly...
            assert_eq!(
                outs[j],
                bcrypt(*cost, salts[j], VECTORS[i].password),
                "vector #{} (cost {cost})",
                i + 1
            );
            // ...and re-encoding the lane output reproduces the original
            // 60-char string, prefix included.
            assert_eq!(
                reencode(version_of(VECTORS[i].hash), *cost, &salts[j], &outs[j]),
                VECTORS[i].hash,
                "vector #{} (cost {cost})",
                i + 1
            );
        }
    }

    let passwords: Vec<&[u8]> = VECTORS.iter().map(|v| v.password).collect();
    let hashes: Vec<&str> = VECTORS.iter().map(|v| v.hash).collect();
    let results = verify_many(&passwords, &hashes);
    assert_eq!(results.len(), VECTORS.len());
    for (i, result) in results.iter().enumerate() {
        assert_eq!(*result, Ok(true), "vector #{}", i + 1);
    }
}

#[test]
fn divergent_vectors_do_not_match() {
    for (i, d) in DIVERGENT.iter().enumerate() {
        assert_eq!(
            verify(d.password, d.hash),
            Ok(d.must_verify),
            "divergent row #{} ({})",
            i + 1,
            d.hash
        );
    }

    // The buggy rows whose password+salt has a correct sibling in VECTORS:
    // recomputing from the parsed cost+salt must yield the sibling's correct
    // hash, proving the divergence comes from the prefix's bug emulation,
    // not from the inputs.
    for &(password, buggy_hash, correct_hash) in SIBLINGS {
        let parts: HashParts = buggy_hash.parse().expect("divergent hash parses");
        let computed = hash_with_salt(password, parts.get_cost(), parts.get_salt_raw())
            .expect("cost from a valid hash");
        assert_eq!(
            computed.format_for_version(version_of(correct_hash)),
            correct_hash
        );
    }
}

#[test]
fn invalid_settings_are_rejected() {
    // Openwall's 7 invalid-setting cases, as full strings: costs outside
    // 4..=31, four unknown version markers, and crypt_blowfish's "*0"
    // failure magic fed back as a hash.
    const INVALID: [&str; 7] = [
        "$2a$03$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // cost below 4
        "$2a$32$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // cost above 31
        "$2c$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // unknown marker
        "$2z$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // unknown marker
        "$2`$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // 0x60 marker
        "$2{$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", // 0x7B marker
        "*0",
    ];
    for s in INVALID {
        assert!(
            s.parse::<HashParts>().is_err(),
            "{s} must be rejected by HashParts::from_str"
        );
        assert!(
            verify(b"pw", s).is_err(),
            "{s} must be rejected by verify"
        );
    }

    // Cost arguments outside 4..=31 are rejected before any hashing happens
    // (cost 31+ would take hours; only the rejection is tested).
    assert_eq!(hash("pw", 3), Err(BcryptError::CostNotAllowed(3)));
    assert_eq!(hash("pw", 32), Err(BcryptError::CostNotAllowed(32)));
}

#[test]
fn edge_case_passwords() {
    // Fixed salts taken from vectors: #21's "CCCC…" salt and #36's zero salt.
    let salt = VECTORS[20]
        .hash
        .parse::<HashParts>()
        .expect("vector #21 parses")
        .get_salt_raw();
    let zero_salt = VECTORS[35]
        .hash
        .parse::<HashParts>()
        .expect("vector #36 parses")
        .get_salt_raw();
    assert_eq!(zero_salt, [0u8; 16]);

    // Empty password — the exact string is pinned by OW vector #25.
    let empty = hash_with_salt(b"", 5, salt).expect("cost 5 is valid");
    assert_eq!(
        empty.format_for_version(Version::TwoA),
        "$2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy"
    );

    // 71/72/73 bytes: at 71 the NUL terminator still participates (key
    // stream = 71 bytes + NUL); at 72 it is truncated away, so a 73-byte
    // password hashes exactly like its 72-byte prefix.
    let h71 = hash_with_salt([b'a'; 71], 5, salt).expect("cost 5 is valid");
    let h72 = hash_with_salt([b'a'; 72], 5, salt).expect("cost 5 is valid");
    let h73 = hash_with_salt([b'a'; 73], 5, salt).expect("cost 5 is valid");
    assert_ne!(h71, h72);
    assert_eq!(h72, h73);

    // An interior high byte participates in the key stream: changing it to
    // NUL or dropping it changes the hash, and the computed hash verifies.
    let high = b"ab\xffcd";
    let h_high = hash_with_salt(high, 5, salt).expect("cost 5 is valid");
    assert!(verify(high, &h_high.format_for_version(Version::TwoB)).expect("hash parses"));
    assert_ne!(h_high, hash_with_salt(b"ab\x00cd", 5, salt).expect("cost 5 is valid"));
    assert_ne!(h_high, hash_with_salt(b"abcd", 5, salt).expect("cost 5 is valid"));

    // 100-byte password: silently truncated to 72 bytes — the pyca
    // truncation vector (#36) pins the exact string, and 100 == 72 but
    // != 71.
    let hx100 = hash_with_salt(X100, 5, zero_salt).expect("cost 5 is valid");
    assert_eq!(
        hx100.format_for_version(Version::TwoA),
        "$2a$05$......................YgIDy4hFBdVlc/6LHnD9mX488r9cLd2"
    );
    assert_eq!(
        hx100,
        hash_with_salt([b'x'; 72], 5, zero_salt).expect("cost 5 is valid")
    );
    assert_ne!(
        hx100,
        hash_with_salt([b'x'; 71], 5, zero_salt).expect("cost 5 is valid")
    );
}
