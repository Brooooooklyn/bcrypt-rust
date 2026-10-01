# bcrypt + SIMD: deep technical research for a pure-Rust, runtime-dispatched bcrypt crate

Date: 2026-10-01. Scope: research only, no implementation code. All claims cite primary sources (OpenBSD CVS source, Openwall, JtR source, published slide decks/papers, measured microbenchmark databases).

---

# Part 1 — Exact algorithm spec (implementable from this section alone)

The normative references are:

- OpenBSD `bcrypt.c` (libc crypt): https://raw.githubusercontent.com/openbsd/src/master/lib/libc/crypt/bcrypt.c
- OpenBSD `blowfish.c`: https://raw.githubusercontent.com/openbsd/src/master/lib/libc/crypt/blowfish.c
- Openwall `crypt_blowfish` 1.3 (public domain): https://www.openwall.com/crypt/ , https://github.com/openwall/crypt_blowfish
- Provos & Mazières, "A Future-Adaptable Password Scheme" (USENIX 1999): https://www.usenix.org/legacy/events/usenix99/provos/provos_html/node4.html

Where the paper and OpenBSD disagree, **OpenBSD's behavior is the de-facto standard** and is what every test vector encodes.

## 1.1 Blowfish primitives

State:

- `P[18]` — 18 × u32 (the "P-array").
- `S[4][256]` — 4 S-boxes of 256 × u32 each (4 KiB total).

Initial values are the hexadecimal digits of the fractional part of π, filling P[0..17] then S[0][0..255], S[1][...], S[2][...], S[3][...] sequentially (first π hex digit goes to the MSB of P[0]). Sanity-check values:

```
P[0..3]   = 243F6A88 85A308D3 13198A2E 03707344
P[14..17] = 3F84D5B5 B5470917 9216D5D9 8979FB1B
S[0][0..3]= D1310BA6 98DFB5AC 2FFD72DB D01ADFB7
S[3][252..255] = 3AC372E6 ... (transcribe in full from vendored source; see §1.9)
```

The tables originate with Schneier's Blowfish; the file to transcribe from is Openwall `crypt_blowfish.c` (public domain) — see §1.9.

Byte→word conversion (`Blowfish_stream2word`, OpenBSD `blowfish.c`), **big-endian, cyclical** over the input buffer:

```c
u_int32_t
Blowfish_stream2word(const u_int8_t *data, u_int16_t databytes, u_int16_t *current)
{
	u_int32_t temp = 0;
	u_int16_t j = *current;
	for (int i = 0; i < 4; i++, j++) {
		if (j >= databytes)
			j = 0;
		temp = (temp << 8) | data[j];
	}
	*current = j;
	return temp;
}
```

So word = `b[j]<<24 | b[j+1]<<16 | b[j+2]<<8 | b[j+3]`, with the byte index wrapping modulo `databytes`. **Confirmed: password and salt are loaded big-endian.** There is no other endianness anywhere: the cipher operates on u32 words, and the only byte↔word boundaries are `stream2word` (input) and the final big-endian store of the ciphertext (§1.5).

Round function (all arithmetic mod 2^32):

```
F(x) = ((S[0][a] + S[1][b]) ^ S[2][c]) + S[3][d]
a = x >> 24            (most significant byte)
b = (x >> 16) & 0xff
c = (x >> 8)  & 0xff
d = x         & 0xff   (least significant byte)
```

**Confirmed: byte extraction is MSB-first** (OpenBSD `blf_enc`/`Blowfish_encipher`: `(S0[(xl >> 24)] + S1[(xl >> 16) & 0xff]) ^ S2[(xl >> 8) & 0xff]) + S3[xl & 0xff]`).

Encipher (`Blowfish_encipher`, 16 Feistel rounds + output whitening):

```c
Xl = L; Xr = R;
for (i = 0; i < 16; i++) {          // i = 0..15
	Xl ^= P[i];
	Xr ^= F(Xl);
	temp = Xl; Xl = Xr; Xr = temp;  // swap
}
temp = Xl; Xl = Xr; Xr = temp;      // undo last swap
Xr ^= P[16];
Xl ^= P[17];
L = Xl; R = Xr;
```

`blf_enc(state, data, blocks)` encrypts `blocks` consecutive 64-bit blocks **independently (ECB-style, no chaining)** by calling `Blowfish_encipher` on each word pair.

## 1.2 EksBlowfish key schedule

Two expansion functions (OpenBSD `blowfish.c`). `BLF_N + 2` = 18 (P entries).

```c
void
Blowfish_expandstate(blf_ctx *c, const u_int8_t *data, u_int16_t databytes,
    const u_int8_t *key, u_int16_t keybytes)
{
	u_int32_t temp, datal, datar;
	u_int16_t j;

	Blowfish_initstate(c);            /* P,S ← pi digits */

	/* XOR P-array with the key, cycling the key via stream2word */
	j = 0;
	for (i = 0; i < BLF_N + 2; i++) {
		temp = Blowfish_stream2word(key, keybytes, &j);
		c->P[i] ^= temp;
	}

	/* Chain-encrypt zeros, XORing the salt into the running block,
	   overwriting P then all of S */
	j = 0;
	datal = 0; datar = 0;
	for (i = 0; i < BLF_N + 2; i += 2) {
		datal ^= Blowfish_stream2word(data, databytes, &j);
		datar ^= Blowfish_stream2word(data, databytes, &j);
		Blowfish_encipher(c, &datal, &datar);
		c->P[i]     = datal;
		c->P[i + 1] = datar;
	}
	for (i = 0; i < 4; i++) {
		for (k = 0; k < 256; k += 2) {
			datal ^= Blowfish_stream2word(data, databytes, &j);
			datar ^= Blowfish_stream2word(data, databytes, &j);
			Blowfish_encipher(c, &datal, &datar);
			c->S[i][k]     = datal;
			c->S[i][k + 1] = datar;
		}
	}
}
```

```c
void
Blowfish_expand0state(blf_ctx *c, const u_int8_t *data, u_int16_t databytes)
{
	/* NOTE: does NOT re-init P/S from pi */
	j = 0;
	for (i = 0; i < BLF_N + 2; i++) {
		temp = Blowfish_stream2word(data, databytes, &j);
		c->P[i] ^= temp;
	}

	/* Same chain, but WITHOUT XORing data into the running block */
	datal = 0; datar = 0;
	for (i = 0; i < BLF_N + 2; i += 2) {
		Blowfish_encipher(c, &datal, &datar);
		c->P[i]     = datal;
		c->P[i + 1] = datar;
	}
	for (i = 0; i < 4; i++) {
		for (k = 0; k < 256; k += 2) {
			Blowfish_encipher(c, &datal, &datar);
			c->S[i][k]     = datal;
			c->S[i][k + 1] = datar;
		}
	}
}
```

Key facts for implementers:

- Each expansion touches 18 P-words + 1024 S-words = **1042 words**, performing **521 sequential Blowfish encryptions** (9 P-pairs + 512 S-pairs), each 16 rounds → 8,336 F-function evaluations per expansion. The chain `datal/datar` is strictly sequential (each encipher feeds the next), which is the fundamental latency of bcrypt.
- `Blowfish_expand0state` is applied to an **already-expanded** state; it only XORs the new data into P and re-runs the zero chain. It does not reset S.

## 1.3 The bcrypt core (OpenBSD `bcrypt_hashpass`)

```
bcrypt(cost_log2, key, key_handling, salt16):
  # key_handling per minor version, see §1.7
  state  = fresh Blowfish state
  csalt  = salt16 (16 bytes, decoded from 22 bcrypt-base64 chars)
  rounds = 1 << cost_log2                      # cost_log2 in [4, 31]

  Blowfish_expandstate(state, csalt, 16, key, key_len)
  for i in 0 .. rounds:                        # 2^cost iterations
      Blowfish_expand0state(state, key, key_len)    # FIRST
      Blowfish_expand0state(state, csalt, 16)       # SECOND

  cdata[0..5] = stream2word over "OrpheanBeholderScryDoubt" (24 bytes)
              = { 0x4f727068, 0x65616e42, 0x65686f6c, 0x64657253, 0x63727944, 0x6f756274 }
  for i in 0 .. 64:
      blf_enc(state, cdata, 3)                 # 3 independent ECB blocks, in place

  encrypted[24]: for i in 0..6: store cdata[i] big-endian
  return encrypted
```

Loop-order note: the USENIX paper's `EksBlowfishSetup` pseudocode lists the two `ExpandKey(0, …)` lines in the opposite order from OpenBSD's implementation (the Openwall Passwords'13 deck flags this; see https://www.openwall.com/presentations/Passwords13-Energy-Efficient-Cracking/). **Implement OpenBSD's order — key expansion first, salt expansion second — it is what all vectors encode.**

Final output: the 6 words are stored big-endian into 24 bytes (`encrypted[4i+0] = cdata[i]>>24`, …, `encrypted[4i+3] = cdata[i]&0xff`). **Only the first 23 bytes are base64-encoded into 31 characters** (OpenBSD: `encode_base64(encrypted + 7 + 22, ciphertext, 4*BCRYPT_WORDS - 1)` with `BCRYPT_WORDS=6`, i.e. `24-1=23` bytes; jBCrypt does `encode_base64(hashed, 24-1)` identically — **confirmed**). The dropped byte is `cdata[5] & 0xff` (LSB of the last word). John the Ripper masks that byte to zero before comparing (`BF_out[index][5] &= ~0xFF`) for bug-compat: https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.c

## 1.4 The ciphertext constant

`"OrpheanBeholderScryDoubt"` is exactly 24 ASCII bytes = 6 big-endian words:

| bytes | word |
|---|---|
| `Orph` | 0x4f727068 |
| `eanB` | 0x65616e42 |
| `ehol` | 0x65686f6c |
| `derS` | 0x64657253 |
| `cryD` | 0x63727944 |
| `oubt` | 0x6f756274 |

Cross-checked against jBCrypt `bf_crypt_ciphertext`: https://raw.githubusercontent.com/djmdjm/jBCrypt/master/src/main/java/org/mindrot/jbcrypt/BCrypt.java

The "× 64 encryptions" loop applies `blf_enc(state, cdata, 3)` 64 times; each call re-encrypts each of the 3 blocks in place with the current (now fully expanded) state. **Confirmed.**

## 1.5 bcrypt base64 — alphabet and bit order (correcting a common misconception)

Alphabet (index 0..63):

```
./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
```

Bit packing (OpenBSD `encode_base64`/`decode_base64`): for each 3-byte group → 4 chars,

```
char0 = b0 >> 2
char1 = ((b0 & 0x03) << 4) | (b1 >> 4)
char2 = ((b1 & 0x0f) << 2) | (b2 >> 6)
char3 = b2 & 0x3f
```

i.e. **the bit order is the same MSB-first streaming as RFC 4648** — the oft-repeated "bcrypt base64 is LSB-first" claim is wrong (that describes *traditional* DES/md5-crypt radix-64, which uses a different alphabet `./0-9A-Za-z` and LSB-first packing; bcrypt uses neither that alphabet nor that bit order). Verified empirically against known vectors: salt bytes `{38,113,212,…}` → `"HlFShUxTu4ZHHfOLJwfmCe"` (rust-bcrypt test), salt `0x01×16` → `".OC/.OC/.OC/.OC/.OC/.O"` (patrickfav wiki) — both reproduce exactly under RFC-4648 bit order with the bcrypt alphabet.

Differences from RFC 4648 in practice: (a) different alphabet, (b) **no padding**, (c) partial final groups are emitted (so 16 bytes → 22 chars, 23 bytes → 31 chars). Consequences:

- Salt (16 bytes = 128 bits, 21 full chars = 126 bits): the 22nd char carries only the top 2 bits in its high positions, so its index is a multiple of 16 — the last salt char is always one of `.` (0), `O` (16), `e` (32), `u` (48). Confirmed by every published salt (`…u`, `…O`, `…e`, `….`).
- Hash (23 bytes = 184 bits, 30 full chars = 180 bits): the 31st char carries 4 bits → index multiple of 4 (one of `.CGKOSWaeimquy26`). Confirmed by published hashes.

Decode (OpenBSD): `b0 = (c1<<2)|(c2>>4)`, `b1 = ((c2&0xf)<<4)|(c3>>2)`, `b2 = ((c3&3)<<6)|c4`, stopping at 16/23 bytes; invalid chars (`char64() == 255`) cause decode failure.

Note: RustCrypto's `base64` crate ships this exact alphabet as `base64::alphabet::BCRYPT` (used with `NO_PAD`), which bit-matches the above; rust-bcrypt relies on it: https://raw.githubusercontent.com/Keats/rust-bcrypt/master/src/lib.rs

## 1.6 Hash string format

```
"$2" <minor> "$" <cost: 2 decimal digits, 04..31> "$" <22 salt chars> <31 hash chars>
```

Total length **exactly 60 ASCII chars** (4 + 3 + 22 + 31). Examples:

```
$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW
$2b$12$FPWWO2RJ3CK4FINTw0Hi8OiPKJcX653gzSS.jqltHFMxyDmmQ0Hqq   (OpenBSD man page example)
```

Cost field: exactly 2 digits, zero-padded; valid range **04–31** (OpenBSD rejects `< 4` and `> 31`; crypt_blowfish's test suite asserts `$2a$03$…` and `$2a$32$…` both return failure `"*0"`). `rounds = 1U << cost` (for cost 31 this is 2^31 iterations — the OpenBSD loop counter is u32). jBCrypt caps at 30 (deviation, see §1.8).

Salt: 16 raw bytes ⇔ 22 chars. Hash: 23 bytes ⇔ 31 chars (**confirmed**: 23 of the 24 ctext bytes, see §1.3).

## 1.7 Variants `$2a$` / `$2b$` / `$2x$` / `$2y$`, password truncation, NUL handling

Password key length as computed by current OpenBSD `bcrypt.c`:

```c
switch (minor) {
case 'a':
	/* 'ab' should not yield the same as 'abab' */
	key_len = (u_int8_t)(strlen(key) + 1);   /* include NUL; wraps mod 256! */
	break;
case 'b':
	key_len = strlen(key);
	if (key_len > 72)
		key_len = 72;
	key_len++;                               /* include the NUL → max 73 */
	break;
}
```

Confirmed behaviors:

- **Trailing NUL**: for minor ≥ 'a', the terminating NUL byte is **included** in the key stream (`strlen+1`). For the pre-'a' original (minor 0, `$2$`, obsolete) it was not. Since `stream2word` cycles modulo `key_len`, `"a"` and `"a\0"` differ, but an all-NUL key of any length collapses to the same stream (rust-bcrypt has a test asserting `"\0"*8` ≡ `"\0"`).
- **72-byte truncation**: `$2b$` caps the password at 72 bytes and then appends the NUL (so a ≥72-byte password hashes as bytes[0..71] + one NUL = 73-byte key stream; "chars after 72 are ignored" — crypt_blowfish vector, §1.10). Because the effective key for an over-long `$2b$` password is `pw[0..71]‖0x00`, the widely repeated statement "bcrypt truncates at 72 bytes" is exact for the data bytes; the NUL is still appended after truncation.
- **`$2a$` wraparound bug (OpenBSD)**: `key_len = (u_int8_t)(strlen+1)` — a 255-byte password gives key_len 0, 256-byte gives 1, etc. No 72-byte cap in the 'a' path. This is why `$2b$` was introduced (OpenBSD 5.5).
- **crypt_blowfish 8-bit sign-extension bug (CVE-2011-2483)**: pre-1.1 crypt_blowfish sign-extended password bytes ≥ 0x80 when building key words (`(int)(signed char)` instead of `unsigned char` in the stream2word-equivalent). Consequences and prefix semantics per https://www.openwall.com/crypt/ :
  - `$2x$` (crypt_blowfish 1.2+): **reproduces the buggy** algorithm — for verifying legacy hashes only.
  - `$2y$` (1.2+): the **correct** algorithm (crypt_blowfish's marker).
  - `$2b$` (1.3+ / OpenBSD 5.5+): correct algorithm; identical results to `$2y$`.
  - `$2a$` on crypt_blowfish ≥1.1: correct algorithm **plus a "safety" tweak** — in `BF_set_key`, for `$2a$` a constant (`0x10000`, parameter `safety`) is XORed into one expanded-key word in cases that would otherwise collide between buggy and correct hashes (relevant only for passwords containing bytes like 0xff that rarely occur in valid UTF-8). crypt_blowfish internally computes both the buggy and correct expansions side-channel-safely and selects per prefix: `$2a$`: bug=0, safety=0x10000; `$2x$`: bug=1, safety=0; `$2y$`/`$2b$`: bug=0, safety=0. The `$2a$` vectors in §1.10 encode this behavior (note `$2a$` ≠ `$2y$`/`$2b$` for `"\xff\xff\xa3"`).
  - OpenBSD's own code never had the sign-extension bug (unsigned types), so `$2a$` hashes produced by OpenBSD/jBCrypt/bcrypt-node equal `$2y$`/`$2b$` results; only crypt_blowfish-family `$2a$` hashes of 8-bit passwords can differ.
- rust-bcrypt mimics `$2b$` semantics for **all** versions it emits/parses: it copies `min(len,72)` bytes into a zeroed 72-byte buffer and uses `(copy_len+1).min(72)` bytes (i.e. always NUL-terminated, truncation drops the NUL to stay ≤72). Its `$2a$`/`$2x$` are thus "label-only" and do not reproduce crypt_blowfish's `$2a$`-safety or `$2x$`-bug behaviors (its own test shows all four versions of `"hunter2"` produce the identical hash). Decide explicitly which semantics your crate implements per version; matching crypt_blowfish bit-for-bit for `$2x$`/`$2a$`+8-bit requires the dual key-expansion select logic.

## 1.8 Reference implementations and known deviations

| impl | license | notes |
|---|---|---|
| OpenBSD `bcrypt.c`+`blowfish.c` | BSD (Niels Provos copyright, attribution required) | cleanest canonical reading of the algorithm |
| **Openwall crypt_blowfish 1.3** | **public domain** (confirmed in source header and on https://www.openwall.com/crypt/) | best differential-testing reference; contains the only `$2x$`/`$2y$`/`$2a$`-safety logic and the full test suite (`wrapper.c`, `CRYPT_OUTPUT_SIZE=61`) |
| jBCrypt (Damien Miller) | ISC-style | exact OpenBSD port **with deviations**: `log_rounds` limited to **4–30** (not 31); `hashpw` accepts **only minor 'a'** (throws otherwise), appends `"\000"` for minor ≥ 'a', UTF-8-encodes the password before hashing, emits `$2a$`; encodes 23 of 24 bytes. Source: https://github.com/djmdjm/jBCrypt |
| RustCrypto `blowfish` 0.10 | MIT/Apache-2.0 | `Blowfish<BE>` with `bcrypt` feature exposing exactly the four bcrypt primitives: `bc_init_state()`, `salted_expand_key(salt, key)`, `bc_expand_key(key)`, `bc_encrypt([u32;2])` — the natural scalar base for a new crate: https://docs.rs/blowfish/latest/blowfish/ |
| Keats/rust-bcrypt (`bcrypt` on crates.io, v0.19.x) | MIT | full string-format/API layer over RustCrypto `blowfish`; see Part 3 |

**Vendoring recommendation (deliverable):** vendor **`crypt_blowfish.c` + `wrapper.c` from Openwall crypt_blowfish 1.3** — public domain (no attribution required), builds standalone on Linux/macOS (`cc -O2 -fPIC -shared crypt_blowfish.c wrapper.c -o cb.so`; skip `x86.S` on macOS since it's ELF asm; the C fallback is selected automatically) and exposes `crypt_r`/`crypt_gensalt_rn` etc. for `dlopen`/FFI differential testing. Transcribe the π-digit P/S tables from `crypt_blowfish.c` (`BF_init_state`). `wrapper.c`'s `#ifdef TEST` block doubles as your vector file. jBCrypt's `BCrypt.java` is a second independent transcription of the same tables for cross-checking.

## 1.9 Which C file to vendor for the π tables

`crypt_blowfish.c` (Openwall, public domain). It contains `BF_init_state` with the full P (18) + S (4×256) π-digit arrays. Alternative cross-checks: OpenBSD `blowfish.c` (`initstate`, BSD-licensed), jBCrypt `P_orig`/`S_orig` (ISC). All three agree byte-for-byte.

## 1.10 Test vectors (authoritative sets + edge cases)

### A. Openwall crypt_blowfish `wrapper.c` test suite (public domain)

Source: https://raw.githubusercontent.com/openwall/crypt_blowfish/master/wrapper.c — full table transcribed:

```
{ "$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW", "U*U" },
{ "$2a$05$CCCCCCCCCCCCCCCCCCCCC.VGOzA784oUp/Z0DY336zx7pLYAy0lwK", "U*U*" },
{ "$2a$05$XXXXXXXXXXXXXXXXXXXXXOAcXxm9kjPGEMsLznoKqmqw7tc8WCx4a", "U*U*U" },
{ "$2a$05$abcdefghijklmnopqrstuu5s2v8.iXieOjg/.AySBTTZIIVFJeBui",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "chars after 72 are ignored" },                              # 72-byte truncation
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", "\xa3" },
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", "\xff\xff\xa3" },
{ "$2y$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", "\xff\xff\xa3" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.nqd1wy.pTMdcvrRWxyiGL2eMz.2a85.", "\xff\xff\xa3" },
{ "$2b$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e", "\xff\xff\xa3" },
{ "$2y$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq", "\xa3" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq", "\xa3" },
{ "$2b$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq", "\xa3" },
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi", "1\xa3" "345" },
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi", "\xff\xa3" "345" },
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi", "\xff\xa3" "34\xff\xff\xff\xa3" "345" },
{ "$2y$05$/OK.fbVrR/bpIqNJ5ianF.o./n25XVfn6oAPaUvHe.Csk4zRfsYPi", "\xff\xa3" "34\xff\xff\xff\xa3" "345" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.ZC1JEJ8Z4gPfpe1JOr/oyPXTWl9EFd.", "\xff\xa3" "34\xff\xff\xff\xa3" "345" },
{ "$2y$05$/OK.fbVrR/bpIqNJ5ianF.nRht2l/HRhr6zmCp9vYUvvsqynflf9e", "\xff\xa3" "345" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.nRht2l/HRhr6zmCp9vYUvvsqynflf9e", "\xff\xa3" "345" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.6IflQkJytoRVc1yuaNtHfiuq.FRlSIS", "\xa3" "ab" },
{ "$2x$05$/OK.fbVrR/bpIqNJ5ianF.6IflQkJytoRVc1yuaNtHfiuq.FRlSIS", "\xa3" "ab" },
{ "$2y$05$/OK.fbVrR/bpIqNJ5ianF.6IflQkJytoRVc1yuaNtHfiuq.FRlSIS", "\xa3" "ab" },
{ "$2x$05$6bNw2HLQYeqHYyBfLMsv/OiwqTymGIGzFsA4hOTWebfehXHNprcAS", "\xd1\x91" },
{ "$2x$05$6bNw2HLQYeqHYyBfLMsv/O9LIGgn8OMzuDoHfof8AQimSGfcSWxnS", "\xd0\xc1\xd2\xcf\xcc\xd8" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.swQOIzjOiJ9GHEPuhEkvqrUyvWhEMx6", "\xaa"*72 ++ "chars after 72 are ignored as usual" },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.R9xrDjiycxMbQE2bp.vgqlYpW5wx2yy", "\xaa\x55"*36 },
{ "$2a$05$/OK.fbVrR/bpIqNJ5ianF.9tQZzcJfm3uj2NvJ/n5xkhpqLrMpWCe", "\x55\xaa\xff"*24 },
{ "$2a$05$CCCCCCCCCCCCCCCCCCCCC.7uG0VCzI2bS7j6ymqJi9CdcdxiRTWNy", "" },   # empty password
# invalid settings must fail:
{ "*0", "", "$2a$03$CCCCCCCCCCCCCCCCCCCCC." },   # cost < 4
{ "*0", "", "$2a$32$CCCCCCCCCCCCCCCCCCCCC." },   # cost > 31
{ "*0", "", "$2c$05$CCCCCCCCCCCCCCCCCCCCC." },
{ "*0", "", "$2z$05$CCCCCCCCCCCCCCCCCCCCC." },
{ "*0", "", "$2`$05$CCCCCCCCCCCCCCCCCCCCC." },
{ "*0", "", "$2{$05$CCCCCCCCCCCCCCCCCCCCC." },
{ "*1", "", "*0" },
```

This set alone covers: empty password, 72-byte truncation (ASCII and binary `\xaa`/`\xaa\x55`/`\x55\xaa\xff` patterns), high-bit bytes, `$2x$` bug emulation vs `$2y$`/`$2b$` correctness vs `$2a$`-safety divergence, and cost boundaries 03/04/31/32. (Cost 31 itself has no published vector — it is 2^31 iterations ≈ hours; acceptance is tested implicitly by rejecting 32.)

### B. jBCrypt `TestBCrypt.java` (20 vectors; passwords × costs 6/8/10/12)

Source: https://raw.githubusercontent.com/djmdjm/jBCrypt/master/test/org/mindrot/jbcrypt/TestBCrypt.java

```
""        06  $2a$06$DCq7YPn5Rq63x1Lad4cll.TV4S6ytwfsfvkgY8jIucDrjc8deX1s.
""        08  $2a$08$HqWuK6/Ng6sg9gQzbLrgb.Tl.ZHfXLhvt/SgVyWhQqgqcZ7ZuUtye
""        10  $2a$10$k1wbIrmNyFAPwPVPSVa/zecw2BCEnBwVS2GbrmgzxFUOqW9dk4TCW
""        12  $2a$12$k42ZFHFWqBp3vWli.nIn8uYyIkbvYRvodzbfbK18SSsY.CsIQPlxO
"a"       06  $2a$06$m0CrhHm10qJ3lXRY.5zDGO3rS2KdeeWLuGmsfGlMfOxih58VYVfxe
"a"       08  $2a$08$cfcvVd2aQ8CMvoMpP2EBfeodLEkkFJ9umNEfPD18.hUF62qqlC/V.
"a"       10  $2a$10$k87L/MF28Q673VKh8/cPi.SUl7MU/rWuSiIDDFayrKk/1tBsSQu4u
"a"       12  $2a$12$8NJH3LsPrANStV6XtBakCez0cKHXVxmvxIlcz785vxAIZrihHZpeS
"abc"     06  $2a$06$If6bvum7DFjUnE9p2uDeDu0YHzrHM6tf.iqN8.yx.jNN1ILEf7h0i
"abc"     08  $2a$08$Ro0CUfOqk6cXEKf3dyaM7OhSCvnwM9s4wIX9JeLapehKK5YdLxKcm
"abc"     10  $2a$10$WvvTPHKwdBJ3uk0Z37EMR.hLA2W6N9AEBhEgrAOljy2Ae5MtaSIUi
"abc"     12  $2a$12$EXRkfkdmXn2gzds2SSitu.MW9.gAVqa9eLS1//RYtYCmB1eLHg.9q
"abcdefghijklmnopqrstuvwxyz"  06  $2a$06$.rCVZVOThsIa97pEDOxvGuRRgzG64bvtJ0938xuqzv18d3ZpQhstC
"abcdefghijklmnopqrstuvwxyz"  08  $2a$08$aTsUwsyowQuzRrDqFflhgekJ8d9/7Z3GV3UcgvzQW3J5zMyrTvlz.
"abcdefghijklmnopqrstuvwxyz"  10  $2a$10$fVH8e28OQRj9tqiDXs1e1uxpsjN0c7II7YPKXua2NAKYvM6iQk7dq
"abcdefghijklmnopqrstuvwxyz"  12  $2a$12$D4G5f18o7aMMfwasBL7GpuQWuP3pkrZrOAnqP.bmezbMng.QwJ/pG
"~!@#$%^&*()      ~!@#$%^&*()PNBFRD"  06  $2a$06$fPIsBO8qRqkjj273rfaOI.HtSV9jLDpTbZn782DC6/t7qT67P6FfO
"~!@#$%^&*()      ~!@#$%^&*()PNBFRD"  08  $2a$08$Eq2r4G/76Wv39MzSX262huzPz612MZiYHVUJe/OcOql2jo4.9UxTW
"~!@#$%^&*()      ~!@#$%^&*()PNBFRD"  10  $2a$10$LgfYWkbzEvQ4JakH7rOvHe0y8pHKF9OaFgwUZ2q7W2FFZmZzJYlfS
"~!@#$%^&*()      ~!@#$%^&*()PNBFRD"  12  $2a$12$WApznUOJfkEGSmYRfnkrPOr466oFDCaj4b6HY3EXGvfxm43seyhgC
```

### C. patrickfav/bcrypt "Published Test Vectors" wiki

https://github.com/patrickfav/bcrypt/wiki/Published-Test-Vectors (raw: https://raw.githubusercontent.com/wiki/patrickfav/bcrypt/Published-Test-Vectors.md). Highlights (many more in the wiki):

```
# empty password, increasing cost
("", 4,  "zVHmKQtGGQob.b/Nc7l9NO") → $2a$04$zVHmKQtGGQob.b/Nc7l9NO8UlrYcW05FiuCj/SxsFO/ZtiN9.mNzy
("", 8,  "zVHmKQtGGQob.b/Nc7l9NO") → $2a$08$zVHmKQtGGQob.b/Nc7l9NOiLTUh/9MDpX86/DLyEzyiFjqjBFePgO
# increasing password length (NUL-inclusion sensitivity), same salt, cost 4
"a"    → $2a$04$5DCebwootqWMCp59ISrMJ.l4WvgHIVg17ZawDIrDM2IjlE64GDNQS
"aa"   → $2a$04$5DCebwootqWMCp59ISrMJ.AyUxBk.ThHlsLvRTH7IqcG7yVHJ3SXq
"aaa"  → $2a$04$5DCebwootqWMCp59ISrMJ.BxOVac5xPB6XFdRc/ZrzM9FgZkqmvbW   … (a×1..a×16)
# special salts: all-0x00, all-0x01, all-0x80, all-0xff
salt 0x00×16 = "......................" : ("-O_=*N!2JP",4) → $2a$04$......................JjuKLOX9OOwo5PceZZXSkaLDvdmgb82
salt 0x01×16 = ".OC/.OC/.OC/.OC/.OC/.O" : (")V`/UM/]1t",4) → $2a$04$.OC/.OC/.OC/.OC/.OC/.OQIvKRDAam.Hm5/IaV/.hc7P8gwwIbmi
salt 0x80×16 = "eGA.eGA.eGA.eGA.eGA.e." : ("@3YaJ^Xs]*",4) → $2a$04$eGA.eGA.eGA.eGA.eGA.e.stcmvh.R70m.0jbfSFVxlONdj1iws0C
salt 0xff×16 = "999999999999999999999u" : ("N7dHmg\\PI^",4) → $2a$04$999999999999999999999uCZfA/pLrlyngNDMq89r1uUk.bQ9icOu
# non-ASCII (UTF-8) passwords: Latin, Greek, Cyrillic, CJK at cost 4-6 — see wiki
```

### D. rust-bcrypt cross-implementation vectors

https://raw.githubusercontent.com/Keats/rust-bcrypt/master/src/lib.rs (tests):

```
"password"                    ✓ $2a$04$UuTkLRZZ6QofpDOlMz32MuuxEHA43WOemOYHPz6.SjsVsyO1tDU96   (online tool)
"correctbatteryhorsestapler"  ✓ $2b$04$EGdrhbKUv8Oc9vGiXX0HQOxSg445d458Muh7DAHskb6QbtCvdxcie   (python bcrypt)
"correctbatteryhorsestapler"  ✓ $2a$04$n4Uy0eSnMfvnESYL.bLwuuj0U/ETSsoTpRT9GVk5bektyVVa5xnIi   (node)
binary [29,225,195,…]         ✓ $2a$04$tjARW6ZON3PhrAIRW2LG/u9aDw5eFdstYLR8nFCNaOQmsH9XD23w.   (golang)
"x"×100 (truncation)          ✓ $2a$05$......................YgIDy4hFBdVlc/6LHnD9mX488r9cLd2   (python)
"My S3cre7 P@55w0rd!", cost 5, salt {38,113,212,141,108,213,195,166,201,38,20,13,47,40,104,18}
                            → $2b$05$HlFShUxTu4ZHHfOLJwfmCeDj/kuKFKboanXtDJXxCC7aIPTUgxNDe
"hunter2", cost 12, salt 0×16 → $2b$12$......................21jzCB1r6pN6rp5O2Ev0ejjTAboskKm
```

Recommended vector stack for the new crate: all of A (natively, via the vendored `wrapper.c`), plus B, the special-salt and length-series from C, and D's cross-impl checks.

---

# Part 2 — SIMD optimization of bcrypt

## 2.1 Why one bcrypt hash cannot be SIMD-parallelized

- The cost loop is `2^cost` iterations of two full key expansions, each = 521 **strictly sequential** Blowfish encryptions (the `datal/datar` chain: each encryption's output feeds the next). No ILP across iterations.
- Within one encryption, the 16 Feistel rounds are sequential; within one F evaluation the four S-box lookups are independent, but the combine `((A+B)^C)+D` is three scalar ALU ops — see §2.8 for why vectorizing just that loses.
- S-box access is data-dependent indexing — not bitsliceable at any reasonable cost.

Therefore the only viable SIMD strategy is **throughput-parallel multi-lane interleaving of independent hashes** (batch hashing / cracking / batch verify), exactly what JtR and hashcat do. This has an API consequence for the crate: SIMD backends only pay off behind a `hash_many`/`verify_many`-style batch entry point; single-hash latency is unchanged. Sources: Openwall Passwords'13 deck (https://www.openwall.com/presentations/Passwords13-Energy-Efficient-Cracking/), hashcat bcrypt notes (https://openwall.info/wiki/john/GPU/bcrypt).

## 2.2 What John the Ripper actually does (correcting the premise)

JtR's bcrypt (`john/src/BF_std.c`, `BF_std.h`, plus asm `x86-64.S`) contains **no AVX2/AVX-512 code**. Its optimizations are scalar:

1. **Scalar ILP interleaving**: `BF_X2` processes 2 independent hashes per pass (optionally 3, `BF_X2==3`), giving the out-of-order core two independent dependency chains to overlap load latencies — https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.c
2. **Precomputed schedules**: `BF_exp_key[18]` = the 18 stream2word-cycled key words (computed once; includes a `sign_extension_bug` mode using `(int)(signed char)*ptr` for `$2x$`); `BF_init_key[i] = BF_init_state.P[i] ^ BF_exp_key[i]` precomputed so each `expand0state(key)` is 18 XORs + the chain; the salt expansion reduces to XORing `P[i] ^= salt_words[i & 3]` because the 16-byte salt is exactly 4 words.
3. Final compare masks off the unencoded 24th byte (`BF_out[index][5] &= ~0xFF`).

Openwall *did* experiment with AVX2/AVX-512 gathers (Solar Designer, Steve Thomas, Sayantan Datta; john-dev threads 2013–2015, e.g. https://openwall.com/lists/john-dev/2013/11/01/1, https://www.openwall.com/lists/john-dev/2015/06/24/10) and **the gather versions lost to the ×2/×3 scalar-interleaved code on Haswell**, so they were never merged. Measured at cost 5 (Passwords'13 deck):

- i7-4770K (Haswell), 8 threads: AVX2 8-lane gather (Steve Thomas): **4,186 c/s** vs scalar JtR ×2-interleaved **6,595 c/s** → "Haswell's gather loads are just slow". 4 threads: 3,888 c/s. Re-laying out to 7 instances (<32 KiB working set) gave 3,519 c/s (still losing).
- Xeon Phi 5110P (Knights Corner): scalar OpenMP 6,246 c/s beat native VPU-intrinsics masked-gather 4,147 c/s.

The often-cited "bcrypt-simd by Sayantan Datta" is **not a standalone repo** — Datta's bcrypt work was the 2012 Xeon Phi 16-lane JtR experiments (john-users threads https://www.openwall.com/lists/john-users/2012/10/08/18 and …/23) credited in the Passwords'13 deck. The real standalone artifact of that research line is **cat-j/pbcrypt** below.

## 2.3 cat-j/pbcrypt (UBA 2019) — the best-documented AVX2 bcrypt

Repo: https://github.com/cat-j/pbcrypt (GPL-2.0, NASM x86-64, `$2b$` only); full thesis (Spanish): https://www-2.dc.uba.ar/trabajosFinalesOrga2/2020_JUARROS/informe.pdf

Design (directly transferable to Rust intrinsics):

- **SoA transposed S-boxes**: every P and S entry is replicated N× contiguously — for 4 lanes, `Sv[box][idx][lane]` at address `((box*256 + idx)*4 + lane)*4` (built with `vpbroadcastd`); for 8 lanes the context is 2,048 dwords per S-box and 144 P dwords.
- **F in SIMD**: extract bytes with `vpslld`/`vpsrld`+`vpand`; index vector = `idx*LANES + lane_offsets` (`lane_offsets = {0,1,…,N-1}` constant); **4 × `vpgatherdd` per F**; combine with `vpaddd`/`vpxord`.
- All lanes run lockstep (no divergence — Blowfish has no data-dependent branches); passwords must be padded to equal length per batch.
- Measured (i5-7600 Kaby Lake, cost 8, single thread): 4-lane AVX2 = **86.4 h/s (+33%)** vs its own scalar asm 66 h/s; 8-lane ("pbcrypt doble") = **182.1 h/s (+175% vs scalar, +110% vs 4-lane)** ≈ **2.76× scalar**. Notes: `vpermq` was the most expensive instruction in the index path; keeping P in registers was 75% *slower* (register pressure); unrolling ≈ no effect; their scalar asm ≈ OpenBSD C `-O2`.

## 2.4 Gather economics per microarchitecture (the decision data)

S-box lookups are the hot operation: 4 lookups per F, 16 F per encryption, 521 encryptions per expansion → 33,344 lookups per expansion; per bcrypt ≈ `2^cost × 2 × 33,344` vectorized across lanes.

- **Intel AVX2 `vpgatherdd` ymm (8×u32)**: ~5 uops, ~5-cycle reciprocal throughput on Skylake *and* Ice Lake (uops.info: https://uops.info/html-instr/VPGATHERDD_YMM_YMM_YMM.html). Eight scalar loads need ≥4 cycles on 2 load ports, plus ~12–16 insert/shuffle uops to build a vector — so gather wins on **front-end/uop count**, roughly ties on **load-port throughput**. On Haswell the same gather was measurably slower (hence JtR's negative result, §2.2); it improved from Broadwell/Skylake onward.
- **Intel AVX-512 `vpgatherdd` zmm (16×u32)**: ~2 elements/cycle load throughput → roughly 8–10 cycles recip; ~9 uops (Agner Fog's tables list Skylake-X ≈ 5c, Ice Lake ≈ 4–5c — treat exact zmm figures as approximate and measure on-target: https://www.agner.org/optimize/instruction_tables.pdf). With EVEX, masked gathers and `ymm` gathers via AVX512VL are also available and cheap.
- **AMD Zen 2/3/4**: gathers are **microcoded** — Zen 3 `vpgatherdd` ymm ≈ 39 uops, ~8c recip; Zen 4 ymm ≈ 42 uops (~8c), zmm ≈ 81 uops (~16c+) (Agner, ibid.). **Avoid gathers on AMD**: use per-lane scalar loads + `vmovd`/`vpinsrd`/`vinserti128` vector construction (≈ 8 loads + ~12 uops, load-port bound ~4c), or JtR-style scalar ×2/×3 interleaving.
- Rule of thumb: on Intel (SKL+), gather ≈ break-even on ports, clear win on uop cache/front-end; on AMD, gather is a loss. A dispatch layer should select gather vs load+insert by vendor, not just by ISA.

## 2.5 NEON and wasm SIMD128 (no gather)

- **NEON**: no gather in base NEON (SVE2 adds `LD1W` gather, but Apple M-series has no SVE; Neoverse V1/N2 do). No published NEON-specific bcrypt exists — JtR on aarch64 uses the same `BF_std.c` ×2 C code; hashcat has no CPU-NEON path (verified by source search). Realistic design: 4 lanes u32x4, SoA S-boxes, per-lane `ldr w,[base,idx,lsl#2]` + `ins v.s[lane]` (Apple M1: 3 loads/cycle makes 4-lane vector construction ≈ 2–3 cycles of ports, competitive), or scalar ×2/×3 interleave on wide cores. Expected ≈ **1.5–2.5×** vs single-instance scalar per core (estimate; must be measured — no published numbers exist).
- **wasm SIMD128**: 4 lanes u32x4, no gather; `v128.load32_lane` / `i32x4.replace_lane` for vector construction. Same SoA layout. Expected ≈ **1.5–2.5×** (estimate).
- For both, the scalar-interleave baseline (×2/×3 instances in plain Rust, letting the compiler/OoO core overlap chains) is the bar to beat.

## 2.6 Lockstep key-schedule vectorization (loop structure)

Per lane-independent state: `Pv[18]` vectors (u32xN), `Sv[4][256]` SoA blocks (each entry N dwords). Password/salt words become **vectors** (different per lane when hashing a batch). Structure:

```
expand0state_v(key_words_v[18], salt_words_v[4]):     # salt: 16 B = exactly 4 words
    for i in 0..18: Pv[i] ^= key_words_v[i]            # broadcast if same pw, else per-lane vector
    (L, R) = (0, 0)                                     # u32xN
    for i in (0..18).step(2):  enc_v(); Pv[i]=L; Pv[i+1]=R
    for b in 0..4: for k in (0..256).step(2): enc_v(); Sv[b][k..k+2]=…
enc_v():
    for i in 0..16:
        L ^= Pv[i]
        R ^= Fv(L)     # 4 gathers (or load+insert), 3 vpaddd/vpxord
        swap(L, R)
    swap back; R ^= Pv[16]; L ^= Pv[17]
```

The whole `2^cost` loop runs **lockstep across lanes** — identical control flow, no masks needed (except optionally tail lanes). The final 64× `blf_enc` also vectorizes trivially (3 independent block chains per lane, still lockstep).

Extra vectorizable tricks from JtR/pbcrypt worth copying: precompute `key_words_v[18]` and `P_init_key_v[18] = P_pi ^ key_words` once per batch; salt XOR is `Pv[i] ^= saltw_v[i & 3]`.

## 2.7 Cache / memory pressure — lane-count selection

Per-lane working set = 4 KiB S + 72 B P ≈ 4.1 KiB:

| lanes | set | vs caches |
|---|---|---|
| 4 | ~16 KiB | fits any 32 KiB L1D with headroom |
| 8 | ~33 KiB | marginally exceeds 32 KiB L1D (Haswell–Skylake-X); fits 48 KiB L1D (Ice Lake/Tiger Lake client, Golden Cove) |
| 16 | ~66 KiB | exceeds all current L1Ds → gathers hit L2 (~14c latency, ~64 B/c bandwidth) |

Empirically (Passwords'13): 8 lanes × 4 KiB = 32 KiB+ overflowed Haswell's L1D and hurt; re-layout to 7 instances (<32 KiB) recovered part of the loss. Guidance: **choose lane count so the S-box set stays L1D-resident** — 8 lanes on 48 KiB-L1D Intel P-cores, 6–7 lanes on 32 KiB-L1D parts; for AVX-512, 16 lanes means accepting L2-resident S-boxes (throughput still gains because L2 bandwidth is high and gathers are pipelined, but expect sub-linear scaling; alternatively run AVX-512 with 8–12 active lanes via EVEX/ymm). Also note 4 KiB/box × 4 boxes alignment: SoA blocks should be 64-byte-aligned to avoid cache-line-splits on the gathers.

## 2.8 Single-hash SIMD alternatives — why they fail

- **Vectorize the 4 S-box loads of one F with one gather**: indices `S0[a],S1[b],S2[c],S3[d]` from a flat 4 KiB table → one masked gather (4 active lanes) replacing 4 scalar loads. Cost: gather ~5c vs 4 independent scalar loads ≈ 2c (2 ports), plus the combine stays scalar. **Loss.** The F chain is latency-bound on the *serial* add/xor/lookup dependency, not load-port throughput.
- **SIMD across the 16 rounds / 521 chain steps**: impossible — strictly sequential.
- **Bitslicing**: S-box = 256×32-bit data-dependent table; a bitsliced select costs O(256) ops per lookup — orders of magnitude slower.
- **GPU** (for context): bcrypt's 4 KiB/instance local-memory need caps occupancy (~8–16 work-items/group on 32–64 KiB local memory); a top-end RTX 4090 achieves only ~180 kH/s at cost 5 in hashcat (-m 3200) — bcrypt resists GPUs by design (https://openwall.info/wiki/john/GPU/bcrypt, hashcat docs/forums).

## 2.9 Published speedup summary (measured, not extrapolated)

| platform | implementation | result | source |
|---|---|---|---|
| Haswell i7-4770K, cost 5, 8T | AVX2 8-lane gather vs JtR scalar ×2 | 4,186 vs 6,595 c/s (**gather loses**) | Passwords'13 deck |
| Xeon Phi 5110P (KNC) | 16-lane VPU masked-gather vs scalar OpenMP | 4,147 vs 6,246 c/s (**loses**) | Passwords'13 deck |
| Kaby Lake i5-7600, cost 8, 1T | pbcrypt 8-lane AVX2 gather vs own scalar asm | 182 vs 66 h/s (**+175% ≈ 2.76×**) | pbcrypt thesis |
| Kaby Lake, 1T | pbcrypt 4-lane vs scalar | +33% | pbcrypt thesis |

Honest expectation for a good Rust implementation, per-core vs a plain-C-quality scalar (and vs a JtR-quality ×2-interleaved scalar):

- **AVX2 + gather, Intel SKL+**: ~2.5–3× plain scalar (~1.3–1.6× vs ×2-interleave).
- **AVX-512, 16 lanes**: ~3–6× plain scalar, L1D-pressure-limited (64 KiB set); more if lane count is tuned to L1D size.
- **AVX2, AMD Zen (load+insert, no gather)**: ~2–2.5× plain scalar.
- **NEON 4-lane / wasm128**: ~1.5–2.5× (estimates; no published bcrypt numbers).
- **Scalar fallback (×2/×3 interleave)**: ~1.3–1.8× over single-instance — the mandatory baseline, and it *beat* gathers on Haswell.

None of this helps single-hash latency; it multiplies batch throughput. (No 4–5×/8-lane AVX2 or 8–10×/16-lane AVX-512 result exists in the literature — the ideal is unreachable because lookups don't scale with vector width the way ALU does; expect ~35% of ideal on gather-friendly Intel per pbcrypt's 2.76×/8 lanes.)

---

# Part 3 — Rust ecosystem

## 3.1 The `bcrypt` crate (crates.io) — **not** RustCrypto

- Owner/repo: **Vincent "Keats" Prouillet**, https://github.com/Keats/rust-bcrypt , https://crates.io/crates/bcrypt (current 0.19.x, MSRV 1.85, `#![forbid(unsafe_code)]`, no_std+alloc capable).
- Deps: `base64` (using the crate's built-in `alphabet::BCRYPT` + `NO_PAD` — **not** an internal b64 module), `blowfish` (RustCrypto, for the core), `getrandom`, `subtle`, optional `zeroize`, `quickcheck` (dev).
- Constants: `MIN_COST=4`, `MAX_COST=31`, `DEFAULT_COST=12`; `pub const BASE_64: GeneralPurpose`.
- Types: `Version { TwoA, TwoX, TwoY, TwoB }` (Display → "2a"…); `HashParts { cost: u32, salt: [u8;16], hash: [u8;23] }` with `get_cost`, `get_salt`, `get_salt_raw`, `format() -> [u8;60]` (2b), `format_for_version(Version) -> String`, `write_for_version<W: fmt::Write>`, `FromStr`/`Display`; `BcryptError { CostNotAllowed(u32), InvalidHash(&'static str), Rand(getrandom::Error), Truncation(usize) }`, `BcryptResult<T>`.
- Public fns: `hash`, `hash_bytes`, `hash_with_result`, `hash_with_salt`, `hash_with_salt_bytes`, plus `non_truncating_*` twins of each (error `Truncation(len+1)` when input ≥ 72 B), `verify`, `non_truncating_verify` (uses `subtle::ConstantTimeEq` on the 23-byte hash), and the re-exported low-level `pub use crate::bcrypt::bcrypt` → `fn bcrypt(cost: u32, salt: [u8;16], password: &[u8]) -> [u8;24]`.
- Behavior: always `$2b$` key semantics (min(72) bytes + NUL, see §1.7); versions are label-only on output; parser enforces exactly-60-ASCII and `$` positions, accepts 2y/2b/2a/2x.
- A drop-in-compatible superset must mirror: this exact API surface + error enum + constants, the 23-byte `HashParts.hash`, the 60-byte formatting, and the truncating-by-default vs `non_truncating_*` split.

## 3.2 RustCrypto building blocks

- **`blowfish`** (https://github.com/RustCrypto/block-ciphers/tree/master/blowfish , MIT/Apache-2.0): `Blowfish<BE>` + `bcrypt` feature → `bc_init_state()`, `salted_expand_key(salt, key)`, `bc_expand_key(key)`, `bc_encrypt([u32;2])` — verified to match OpenBSD semantics (this is what Keats/rust-bcrypt's core calls). Use it as the scalar backend/reference for the new crate.
- **`bcrypt-pbkdf`** (https://github.com/RustCrypto/PBKDFs/tree/master/bcrypt-pbkdf): a *different* construction (bcrypt_pbkdf from OpenBSD, magic string `"OxychromaticBlowfishSwatDynamite"`) — not needed for bcrypt itself but shares the core; worth matching internally.
- Runtime dispatch pattern to copy (RustCrypto-style, e.g. sha2/blake2): `cpufeatures`/`is_x86_feature_detected!` + `#[target_feature(enable = "…")]` unsafe kernels behind a cached fn-pointer or enum dispatch; on aarch64 NEON is baseline (no detection needed); on wasm32, `simd128` is a compile-time `target_feature` (no runtime detection in std) — dispatch via cfg/feature.

## 3.3 Does any Rust bcrypt already do SIMD?

No. Searches across crates.io and GitHub ("rust bcrypt simd", "bcrypt avx2 rust", bcrypt+gather, etc.) find only: Keats/rust-bcrypt (scalar, `#![forbid(unsafe_code)]` so it cannot even use intrinsics), `blowfish` (scalar), `bcrypt-pbkdf` (scalar), and bindings to C (`bcrypt` wrappers around system crypt). The only SIMD bcrypt implementations anywhere are the asm projects in Part 2 (JtR experiments, pbcrypt) and hashcat/JtR-GPU OpenCL kernels. A pure-Rust runtime-dispatched SIMD bcrypt is unclaimed territory — with the caveat from §2.9 that expected gains are ~2–3× (AVX2) / ~3–6× (AVX-512) **per core on batch workloads**, not the naive 8×/16×.

---

# Appendix — build/recipe notes for the future crate (research conclusions, not code)

1. **Scalar core**: RustCrypto `blowfish` `bcrypt` feature, or a from-scratch core vendoring `crypt_blowfish.c` π tables (public domain).
2. **Batch API first**: SIMD only helps `hash_many`/`verify_many`; keep single-hash on the scalar path.
3. **Dispatch matrix**:
   - AVX-512 (SKX+/ICL+/GLC+): 16 lanes (or 8–12 tuned to L1D), SoA S-boxes, `vpgatherdd` zmm.
   - AVX2 Intel: 8 lanes, SoA, `vpgatherdd` ymm.
   - AVX2 AMD Zen: 8 lanes, SoA, scalar loads + `vpinsrd`/`vinserti128` (no gather).
   - SSE2/SSSE3(+SSE4.1): 4 lanes, SoA, scalar loads + `pinsrd` — or just scalar ×2 interleave.
   - NEON: 4 lanes, SoA, `ldr`+`ins`; wasm128: 4 lanes, SoA, `load32_lane`/`replace_lane`.
   - Scalar: JtR-style ×2/×3 instance interleaving; precomputed key/salt XOR schedules (§2.2, §2.6).
4. **Correctness harness**: vendored crypt_blowfish 1.3 (public domain) compiled as a shared lib; differential-test every backend against it across the §1.10 vector stack + random (password, salt, cost 4–8) tuples; include `$2x$`/`$2a$`-safety cases only if you choose to implement those semantics (Keats/rust-bcrypt does not).
5. **Zeroization**: each lane's 4.1 KiB P+S state and the 72-byte padded password buffer contain key material — `zeroize` on drop, matching rust-bcrypt's behavior.
