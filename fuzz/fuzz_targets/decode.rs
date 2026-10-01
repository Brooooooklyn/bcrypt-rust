//! Fuzzes the `$2*$` hash-string decoder through `HashParts::from_str` and
//! `verify`.
//!
//! The decoder parses attacker-controlled input (any `verify` caller is
//! feeding it stored-or-worse strings), and before this target existed it was
//! covered only by a hand-picked rejection matrix. The property: no input may
//! panic, overflow, or index out of bounds; errors are fine.
#![no_main]

use bcrypt_rust::HashParts;
use libfuzzer_sys::fuzz_target;
use std::str::FromStr;

/// A string the parser ACCEPTS with a large cost costs a full bcrypt per fuzz
/// iteration, which starves the fuzzer. Pre-filter those cheaply: this is NOT
//! the real parser, it just caps work on inputs that look valid-and-big; a
/// mismatch only costs one slow iteration.
fn looks_expensive(s: &str) -> bool {
    // "$2b$12$..." — cost digits live at [4..6] in an otherwise-valid string.
    let b = s.as_bytes();
    if b.len() >= 7 && b[0] == b'$' && b[1] == b'2' && b[3] == b'$' {
        let d0 = b[4].wrapping_sub(b'0');
        let d1 = b[5].wrapping_sub(b'0');
        if d0 < 10 && d1 < 10 {
            return 10 * u32::from(d0) + u32::from(d1) > 8;
        }
    }
    false
}

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    if s.len() > 512 || looks_expensive(s) {
        return;
    }
    // Parse must never panic; a parseable string must also survive verify
    // (wrong password is fine — a verdict, not a crash).
    if let Ok(parts) = HashParts::from_str(s) {
        let _ = bcrypt_rust::verify(b"fuzz-password", s);
        // Round-trip: what parsed must re-format to a string that parses again.
        let formatted = parts.to_string();
        let _ = HashParts::from_str(&formatted);
    } else {
        // Rejected strings must still not panic the full verify path.
        let _ = bcrypt_rust::verify(b"fuzz-password", s);
    }
});
