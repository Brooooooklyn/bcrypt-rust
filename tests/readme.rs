//! The README's Rust examples, compiled and run.
//!
//! Nothing else in the tree compiles `README.md`. Adding
//! `#[doc = include_str!("../README.md")]` under `cfg(doctest)` would also
//! work, but every block would then need `# `-prefixed setup lines to be
//! self-contained — and rustdoc's hidden-line marker is not hidden on GitHub
//! or crates.io, which is where the README is actually read. So the blocks
//! stay clean prose and are transcribed here instead.
//!
//! A transcription drifts. [`transcription_still_matches_the_readme`] is what
//! stops it: it re-extracts every ```rust block from `README.md` and asserts
//! each line is present verbatim below, so editing the README without editing
//! this file fails the suite.

// `hash` / `hash_many` need OS entropy, so the whole file is std-only.
#![cfg(feature = "std")]

/// Block 1 (quick start) and block 2 (batch hashing) in order.
#[test]
fn readme_examples_compile_and_run() -> Result<(), bcrypt_rust::BcryptError> {
    // --- block 1 ---
    // Cost 4 keeps the example fast; use DEFAULT_COST (12) or higher in real code.
    let hash = bcrypt_rust::hash(b"hunter2", 4)?;
    assert_eq!(hash.len(), 60);
    assert!(bcrypt_rust::verify(b"hunter2", &hash)?);
    assert!(!bcrypt_rust::verify(b"hunter3", &hash)?);

    // --- block 2 ---
    let passwords: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
    let hashes = bcrypt_rust::hash_many(&passwords, 4)?;
    for (password, hash) in passwords.iter().zip(&hashes) {
        assert!(bcrypt_rust::verify(password, hash)?);
    }

    Ok(())
}

/// Guards the transcription above against the README moving out from under it.
///
/// Line-by-line rather than block-by-block on purpose: the README blocks are
/// fragments, so they are not textually contiguous here, but every *line* of
/// code in them must still appear.
#[test]
fn transcription_still_matches_the_readme() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("README.md must be readable from the manifest dir");
    let this_file =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/readme.rs"))
            .expect("this test file must be readable");

    // Compare code, not layout: drop the trailing comment and collapse runs of
    // whitespace, so `cargo fmt` reflowing this file is not a failure.
    fn normalize(line: &str) -> String {
        let code = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        code.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    let transcribed: Vec<String> = this_file.lines().map(normalize).collect();

    let mut in_rust_block = false;
    let mut blocks = 0usize;
    let mut checked = 0usize;

    for line in readme.lines() {
        if line.trim_start().starts_with("```") {
            if in_rust_block {
                in_rust_block = false;
            } else if line.trim() == "```rust" {
                in_rust_block = true;
                blocks += 1;
            }
            continue;
        }
        if !in_rust_block {
            continue;
        }
        let want = normalize(line);
        if want.is_empty() {
            continue; // blank line, or a line that is only a comment
        }
        assert!(
            transcribed.contains(&want),
            "README.md has a line of Rust that this file does not transcribe, so it is \
             compiled by nothing:\n    {}\nAdd it to `readme_examples_compile_and_run`.",
            line.trim()
        );
        checked += 1;
    }

    assert_eq!(
        blocks, 2,
        "README gained or lost a ```rust block; transcribe it here too"
    );
    assert!(
        checked >= 8,
        "extraction found suspiciously little code: {checked} lines"
    );
}
