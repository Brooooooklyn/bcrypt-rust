//! Portable scalar EksBlowfish and the bcrypt core built on it.
//!
//! This is the reference implementation: correct on every target, one hash
//! at a time. Every SIMD backend landed later reproduces these exact
//! semantics in lockstep lanes and is validated against the vectors pinned
//! by this file's tests, so a change here changes what "correct" means for
//! the whole crate.
//!
//! The structure mirrors OpenBSD `libcrypt`'s `Blowfish_expandstate`,
//! `Blowfish_expand0state` and `bcrypt_hashpass`, including the two details
//! that are easy to get wrong from the original paper alone:
//!
//! * the cost loop expands the **key first, then the salt** (OpenBSD order;
//!   the paper lists them the other way round);
//! * [`expand_state`] consumes salt words as **one continuous stream** across
//!   the P-array and S-box loops — the counter does not reset between them,
//!   so the S-box loop resumes at stream offset 18 (salt word index 2).

use crate::consts::{P_INIT, S_INIT};

/// Full EksBlowfish working state: 18 P-words plus four 256-word S-boxes —
/// 4.1 KiB, stack-resident per hash.
pub struct State {
    p: [u32; 18],
    s: [[u32; 256]; 4],
}

/// The digits-of-pi initial state every key expansion starts from.
fn init_state() -> State {
    State { p: P_INIT, s: S_INIT }
}

/// The Blowfish round function: `((S0[a] + S1[b]) ^ S2[c]) + S3[d]`, bytes
/// taken MSB-first, every addition wrapping (mod 2^32).
#[inline]
fn f(state: &State, x: u32) -> u32 {
    let a = (x >> 24) as usize;
    let b = ((x >> 16) & 0xff) as usize;
    let c = ((x >> 8) & 0xff) as usize;
    let d = (x & 0xff) as usize;
    (state.s[0][a].wrapping_add(state.s[1][b]) ^ state.s[2][c]).wrapping_add(state.s[3][d])
}

/// `a ^ b` behind an opaque boundary. On x86-64 this is one `xor`
/// instruction LLVM cannot fold into a reassociated xor tree: callers pass
/// `f(...)` as `a` and `other_half ^ p[i]` (computed while the S-box loads
/// are in flight) as `b`, so the post-load dependency chain is a single xor
/// instead of `(f ^ other) ^ p[i]`'s two.
#[inline(always)]
fn xor_late(a: u32, b: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    // SAFETY: a register-register xor is always valid; no memory or flags
    // are read or written.
    unsafe {
        let mut out = a;
        core::arch::asm!("xor {o:e}, {b:e}", o = inout(reg) out, b = in(reg) b,
                         options(pure, nomem, nostack, preserves_flags));
        out
    }
    // miri cannot execute inline assembly; fall back to the scalar xor.
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    {
        a ^ b
    }
}

/// One Blowfish block encryption under the state's current P-array.
///
/// The crypt_blowfish `BF_ENCRYPT` round shape: `l` takes P[0] alone, then
/// eight pairs of fused half-rounds (`r ^= P[i] ^ F(l)` alternating with
/// `l ^= P[i+1] ^ F(r)`) consume P[1]..P[16] — no swaps inside the loop, the
/// L/R roles alternate on the registers themselves. The final swap and the
/// P[17] whitening then fold into the return order: the left output word is
/// the whitened old right half.
///
/// Each fused round is written as `xor_late(f(x), other ^ p[i])`: the P-array
/// xor is combined with the other half while the four S-box loads are in
/// flight, so the post-load dependency chain is a single xor. LLVM
/// reassociates `x ^ p ^ f` the other way (two serial xors after the loads),
/// which measured ~4% slower than gcc's crypt_blowfish on Zen 4.
#[inline(always)]
fn encipher(state: &State, mut l: u32, mut r: u32) -> (u32, u32) {
    l ^= state.p[0];
    for i in (1..16).step_by(2) {
        r = xor_late(f(state, l), r ^ state.p[i]);
        l = xor_late(f(state, r), l ^ state.p[i + 1]);
    }
    // Output halves are (old r ^ P[17], old l): the post-round swap and the
    // P[17] whitening are both implicit in the tuple order.
    r ^= state.p[17];
    (r, l)
}

/// OpenBSD `Blowfish_expandstate`: XOR the key into P, then encrypt a zero
/// block 521 times, mixing salt words into the running block and writing
/// each ciphertext pair back over P (9 pairs) and then S (512 pairs).
///
/// The salt words are consumed as **one continuous stream** across both
/// loops — the counter does not reset between them. The P loop reads the
/// (0,1) and (2,3) word pairs alternately, which is `i & 2`/`(i & 2) + 1`
/// as crypt_blowfish writes it. Having consumed 18 words there (18 ≡ 2 mod
/// 4), the stream resumes inside the S loop at index 2, so every S-box
/// pair alternates salt (2,3) then (0,1) and the two-encipher iteration
/// consumes exactly 4 words — the phase survives between iterations. Each
/// write is visible to the encryptions that follow it — that is what makes
/// the chain strictly sequential.
fn expand_state(state: &mut State, salt_words: &[u32; 4], key_words: &[u32; 18]) {
    for (p, &k) in state.p.iter_mut().zip(key_words.iter()) {
        *p ^= k;
    }
    let (mut l, mut r) = (0u32, 0u32);
    for i in (0..18).step_by(2) {
        l ^= salt_words[i & 2];
        r ^= salt_words[(i & 2) + 1];
        (l, r) = encipher(state, l, r);
        state.p[i] = l;
        state.p[i + 1] = r;
    }
    for s_box in 0..4 {
        for i in (0..256).step_by(4) {
            l ^= salt_words[2];
            r ^= salt_words[3];
            (l, r) = encipher(state, l, r);
            state.s[s_box][i] = l;
            state.s[s_box][i + 1] = r;
            l ^= salt_words[0];
            r ^= salt_words[1];
            (l, r) = encipher(state, l, r);
            state.s[s_box][i + 2] = l;
            state.s[s_box][i + 3] = r;
        }
    }
}

/// The 521-encryption zero chain shared by both `expand0state` variants:
/// overwrite P (9 pairs) then S (512 pairs) exactly as [`expand_state`] does,
/// minus the salt mixing — the running block is only ever enciphered.
fn encrypt_zero_chain(state: &mut State) {
    let (mut l, mut r) = (0u32, 0u32);
    for i in (0..18).step_by(2) {
        (l, r) = encipher(state, l, r);
        state.p[i] = l;
        state.p[i + 1] = r;
    }
    for s_box in 0..4 {
        for i in (0..256).step_by(2) {
            (l, r) = encipher(state, l, r);
            state.s[s_box][i] = l;
            state.s[s_box][i + 1] = r;
        }
    }
}

/// OpenBSD `Blowfish_expand0state(key)`: XOR the password words into P, then
/// run the zero chain.
fn expand0state(state: &mut State, words: &[u32; 18]) {
    for (p, &w) in state.p.iter_mut().zip(words.iter()) {
        *p ^= w;
    }
    encrypt_zero_chain(state);
}

/// OpenBSD `Blowfish_expand0state(salt)`: the salt is exactly 4 words, so
/// the P XOR cycles it (`i & 3`), then runs the same zero chain.
fn expand0state_salt(state: &mut State, salt_words: &[u32; 4]) {
    for (i, p) in state.p.iter_mut().enumerate() {
        *p ^= salt_words[i & 3];
    }
    encrypt_zero_chain(state);
}

/// One full bcrypt/EksBlowfish hash: `2^cost` rounds of alternating key/salt
/// expansion, then 64 encryptions of the `"OrpheanBeholderScryDoubt"`
/// constant, returned as 24 big-endian bytes (callers encode the first 23).
///
/// `cost` must already be validated to `4..=31` by the caller
/// (`crate::core`); the loop counter is a `u64` because cost 31 means 2^31
/// iterations.
pub(crate) fn eks_blowfish(cost: u32, key_words: &[u32; 18], salt_words: &[u32; 4]) -> [u8; 24] {
    debug_assert!((4..=31).contains(&cost));
    let mut state = init_state();
    expand_state(&mut state, salt_words, key_words);
    for _ in 0..(1u64 << cost) {
        // OpenBSD order: the password expansion first, the salt second.
        expand0state(&mut state, key_words);
        expand0state_salt(&mut state, salt_words);
    }
    // "OrpheanBeholderScryDoubt" as six words, each read big-endian.
    let mut cdata = [
        0x4f72_7068, 0x6561_6e42, 0x6568_6f6c, 0x6465_7253, 0x6372_7944, 0x6f75_6274,
    ];
    for _ in 0..64 {
        for pair in cdata.chunks_exact_mut(2) {
            let (l, r) = encipher(&state, pair[0], pair[1]);
            pair[0] = l;
            pair[1] = r;
        }
    }
    let mut out = [0u8; 24];
    for (chunk, word) in out.chunks_exact_mut(4).zip(cdata.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    #[cfg(feature = "zeroize")]
    {
        crate::wipe::secure_wipe_u32(&mut state.p);
        for s_box in state.s.iter_mut() {
            crate::wipe::secure_wipe_u32(s_box);
        }
        crate::wipe::secure_wipe_u32(&mut cdata);
    }
    out
}

/// The scalar batch kernel: `outs.len()` independent bcrypt hashes, run one
/// after another on the calling thread.
///
/// Implements the [`super::BcryptLanesFn`] contract for
/// [`super::Backend::Scalar`]: `cost` is `4..=31` (validated upstream in
/// `crate::core`) and the three slices are non-empty and of equal length.
pub fn bcrypt_lanes(
    cost: u32,
    key_words: &[[u32; 18]],
    salt_words: &[[u32; 4]],
    outs: &mut [[u8; 24]],
) {
    debug_assert!((4..=31).contains(&cost));
    debug_assert!(!outs.is_empty());
    debug_assert_eq!(key_words.len(), outs.len());
    debug_assert_eq!(salt_words.len(), outs.len());
    for (lane, out) in outs.iter_mut().enumerate() {
        *out = eks_blowfish(cost, &key_words[lane], &salt_words[lane]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base64;
    use crate::eks::{expand_key_words, salt_words};

    /// Reduce a password to kernel key words exactly as `crate::core` will:
    /// password bytes plus the terminating NUL in a zero-padded 72-byte
    /// buffer, cycling at `password.len() + 1`.
    fn key_words_for(password: &[u8]) -> [u32; 18] {
        let mut key = [0u8; 72];
        key[..password.len()].copy_from_slice(password);
        expand_key_words(&key, password.len() + 1)
    }

    /// Hash `password` at `cost` with the base64-encoded salt and return the
    /// 31-char encoding of the first 23 ciphertext bytes.
    fn hash_23(password: &[u8], salt_b64: &[u8], cost: u32) -> [u8; 31] {
        let kw = key_words_for(password);
        let salt = base64::decode_16(salt_b64).expect("test salt must decode");
        let sw = salt_words(&salt);
        let out = eks_blowfish(cost, &kw, &sw);
        base64::encode_23(out[..23].try_into().expect("24-byte output"))
    }

    // The two vectors below are OpenBSD's own (`$2a$05$CCCCCCCCCCCCCCCCCCCCC.…`).
    // Together they pin the entire chain: key cycling (empty key vs "U*U"),
    // salt cycling, the continuous salt stream in expand_state, encipher,
    // the key-then-salt loop order, the 64 final encryptions, big-endian
    // stores, and the 23-of-24-byte base64 encoding.

    #[test]
    fn openbsd_vector_empty_password() {
        // $2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy
        let h = hash_23(b"", b"CCCCCCCCCCCCCCCCCCCCC.", 5);
        assert_eq!(&h, b"7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy");
    }

    #[test]
    fn openbsd_vector_password_u_u() {
        // $2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW
        let h = hash_23(b"U*U", b"CCCCCCCCCCCCCCCCCCCCC.", 5);
        assert_eq!(&h, b"E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW");
    }

    #[test]
    fn bcrypt_lanes_matches_single_hashes() {
        let kw = key_words_for(b"U*U");
        let salt = base64::decode_16(b"CCCCCCCCCCCCCCCCCCCCC.").expect("test salt must decode");
        let sw = salt_words(&salt);
        let single = eks_blowfish(5, &kw, &sw);
        let kws = [kw, kw];
        let sws = [sw, sw];
        let mut outs = [[0u8; 24]; 2];
        bcrypt_lanes(5, &kws, &sws, &mut outs);
        assert_eq!(outs[0], single);
        assert_eq!(outs[1], single);
    }
}
