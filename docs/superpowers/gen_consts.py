#!/usr/bin/env python3
"""Extract the Blowfish P/S pi-digit tables from vendored crypt_blowfish.c
and emit src/consts.rs. Layout in BF_init_state: S[4][256] first, then P[18]."""
import re, sys

src = open("/Users/brooklyn/workspace/github/bcrypt-rust/crypt_blowfish/crypt_blowfish.c").read()
m = re.search(r"static BF_ctx BF_init_state = \{(.*?)\n\};", src, re.S)
if not m:
    sys.exit("BF_init_state not found")
words = [int(w, 16) for w in re.findall(r"0x[0-9a-fA-F]{8}", m.group(1))]
if len(words) != 1024 + 18:
    sys.exit(f"expected 1042 words, got {len(words)}")
s, p = words[:1024], words[1024:]

# Sanity pins (from research; cross-checked against OpenBSD blowfish.c)
assert p[0] == 0x243F6A88 and p[17] == 0x8979FB1B
assert s[0] == 0xD1310BA6 and s[1023] == 0x3AC372E6

def rol32(x, r): return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF
def fold(xs):
    a = 0
    for b in xs:
        a = rol32((a + b) & 0xFFFFFFFF, 7) ^ b
    return a

def fmt(vals, indent="\t"):
    lines = []
    for i in range(0, len(vals), 4):
        lines.append(indent + ", ".join(f"0x{v:08X}" for v in vals[i:i+4]) + ",")
    return "\n".join(lines)

header = """//! The Blowfish initial state: the hexadecimal digits of the fractional part
//! of pi, filling `P[0..18]` then `S[0][0..256]`, `S[1][...]`, `S[2][...]`,
//! `S[3][...]` (first digit to the MSB of `P[0]`).
//!
//! Transcribed from the public-domain Openwall crypt_blowfish 1.3
//! (`crypt_blowfish/crypt_blowfish.c`, `BF_init_state`) — regenerate with
//! `python3 docs/superpowers/gen_consts.py` rather than editing by hand.

/// Initial P-array values (18 words).
#[rustfmt::skip]
pub const P_INIT: [u32; 18] = [
""" + fmt(p) + """
];

/// Initial S-box values (4 boxes of 256 words).
#[rustfmt::skip]
pub const S_INIT: [[u32; 256]; 4] = [
"""
body = ""
for bx in range(4):
    body += "\t[\n" + fmt(s[bx*256:(bx+1)*256], "\t\t") + "\n\t],\n"

tests = f"""];

#[cfg(test)]
mod tests {{
    use super::*;

    fn fold(xs: &[u32]) -> u32 {{
        xs.iter().fold(0u32, |a, &b| a.wrapping_add(b).rotate_left(7) ^ b)
    }}

    #[test]
    fn pins() {{
        // Spot values confirmed against both crypt_blowfish.c and OpenBSD blowfish.c.
        assert_eq!(P_INIT[0], 0x243F6A88);
        assert_eq!(P_INIT[1], 0x85A308D3);
        assert_eq!(P_INIT[2], 0x13198A2E);
        assert_eq!(P_INIT[3], 0x03707344);
        assert_eq!(P_INIT[14], 0x3F84D5B5);
        assert_eq!(P_INIT[17], 0x8979FB1B);
        assert_eq!(S_INIT[0][0], 0xD1310BA6);
        assert_eq!(S_INIT[0][1], 0x98DFB5AC);
        assert_eq!(S_INIT[0][2], 0x2FFD72DB);
        assert_eq!(S_INIT[0][3], 0xD01ADFB7);
        assert_eq!(S_INIT[0][255], 0x{s[255]:08X});
        assert_eq!(S_INIT[1][0], 0x{s[256]:08X});
        assert_eq!(S_INIT[2][255], 0x{s[767]:08X});
        assert_eq!(S_INIT[3][252], 0x{s[1020]:08X});
        assert_eq!(S_INIT[3][255], 0x3AC372E6);
    }}

    #[test]
    fn transcription_checksums() {{
        // Order-independent folds computed from the C source at generation time;
        // they catch any accidental edit of the tables above.
        assert_eq!(fold(&P_INIT), 0x{fold(p):08X});
        for bx in 0..4 {{
            assert_eq!(fold(&S_INIT[bx]), 0x{0}:08X, "S box {{}}", bx);
        }}
    }}
}}
"""
# The per-box folds differ; build them individually.
folds = "".join(
    f"""
    #[test]
    fn transcription_checksum_s{bx}() {{
        let f = S_INIT[{bx}].iter().fold(0u32, |a, &b| a.wrapping_add(b).rotate_left(7) ^ b);
        assert_eq!(f, 0x{fold(s[bx*256:(bx+1)*256]):08X});
    }}""" for bx in range(4))

tests = tests.replace('for bx in 0..4 {\n            assert_eq!(fold(&S_INIT[bx]), 0x00000000:08X, "S box {}", bx);\n        }', "")
# simpler: rewrite the checksum test body directly
tests = f"""];

#[cfg(test)]
mod tests {{
    use super::*;

    fn fold(xs: &[u32]) -> u32 {{
        xs.iter().fold(0u32, |a, &b| a.wrapping_add(b).rotate_left(7) ^ b)
    }}

    #[test]
    fn pins() {{
        // Spot values confirmed against both crypt_blowfish.c and OpenBSD blowfish.c.
        assert_eq!(P_INIT[0], 0x243F6A88);
        assert_eq!(P_INIT[1], 0x85A308D3);
        assert_eq!(P_INIT[2], 0x13198A2E);
        assert_eq!(P_INIT[3], 0x03707344);
        assert_eq!(P_INIT[14], 0x3F84D5B5);
        assert_eq!(P_INIT[17], 0x8979FB1B);
        assert_eq!(S_INIT[0][0], 0xD1310BA6);
        assert_eq!(S_INIT[0][1], 0x98DFB5AC);
        assert_eq!(S_INIT[0][2], 0x2FFD72DB);
        assert_eq!(S_INIT[0][3], 0xD01ADFB7);
        assert_eq!(S_INIT[0][255], 0x{s[255]:08X});
        assert_eq!(S_INIT[1][0], 0x{s[256]:08X});
        assert_eq!(S_INIT[2][255], 0x{s[767]:08X});
        assert_eq!(S_INIT[3][252], 0x{s[1020]:08X});
        assert_eq!(S_INIT[3][255], 0x3AC372E6);
    }}

    #[test]
    fn transcription_checksums() {{
        // Folds computed from the C source at generation time; they catch any
        // accidental edit of the tables above.
        assert_eq!(fold(&P_INIT), 0x{fold(p):08X});
        assert_eq!(fold(&S_INIT[0]), 0x{fold(s[0:256]):08X});
        assert_eq!(fold(&S_INIT[1]), 0x{fold(s[256:512]):08X});
        assert_eq!(fold(&S_INIT[2]), 0x{fold(s[512:768]):08X});
        assert_eq!(fold(&S_INIT[3]), 0x{fold(s[768:1024]):08X});
    }}
}}
"""

out = header + body + tests
open("/Users/brooklyn/workspace/github/bcrypt-rust/src/consts.rs", "w").write(out)
print("wrote src/consts.rs:", len(p), "P words,", len(s), "S words")
