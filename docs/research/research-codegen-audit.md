# Codegen audit: bcrypt-rust kernels vs their generated assembly

Date: 2026-10-02. Host: Apple M5 Max, rustc 1.98.0 (aarch64-apple-darwin) /
x86_64-apple-darwin cross. Repo: bcrypt-rust @ 484781a. Profile: `release`
(opt-level 3, lto = "thin", codegen-units = 1).

## Method and artifacts

The rlib `.o` members are embedded LLVM **bitcode** (thin LTO), so Apple's
`nm`/`otool` cannot read them (`Unknown attribute kind (105)`). Assembly was
obtained two ways and cross-checked:

1. `cargo rustc --release --lib -- --emit asm` (per-crate, post-opt, pre-LTO):
   `/tmp/arm.s`, `/tmp/x86.s`.
2. Final thin-LTO-linked bench binaries `target/release/deps/micro-*`
   (aarch64) and `target/x86_64-apple-darwin/release/deps/micro-db28881850567f10`
   (freshly linked), disassembled with homebrew `llvm-objdump` 23.1.1.

Every hot-loop claim below was confirmed in the **linked binary**, not just
the per-crate asm. Builds green on both targets (exit 0).

Symbols cited (legacy mangling, aarch64 build hash `CsgLAM6A6Xhnp`,
x86 build hash `Cse5KfVF5KZCZ`):
- `__RNvNtNtCs…_11bcrypt_rust3eks6scalar18encrypt_zero_chain`
- `__RNvNtNtCs…_11bcrypt_rust3eks4neon20encrypt_zero_chain_v`
- `__RINvNtNtCs…_11bcrypt_rust3eks4avx221encrypt_zero_chain_v8Kb0_EB6_` (insert)
- `__RINvNtNtCs…_11bcrypt_rust3eks4avx221encrypt_zero_chain_v8Kb1_EB6_` (gather)
- `__RINvNtNtCs…_11bcrypt_rust3eks6avx51222encrypt_zero_chain_v16Kb0_EB6_` / `Kb1`
- `__RNvNtNtCs…_11bcrypt_rust3eks6avx51210f16_insert` (outlined!)

`encrypt_zero_chain*` holds the S-box loop: 512 of 521 encryptions per chain,
2 chains × 2^cost iterations — >99.9% of all F() calls. That is the loop every
count below refers to.

---

## Ranked findings (headroom the compiler is leaving behind)

### 1. NEON: `black_box` is a stack round-trip on the serial chain — not free

**File:** src/eks/neon.rs:338 — `$y ^= core::hint::black_box(unsafe { f_word($s, $x, $lane) });`

The comment at neon.rs:330-332 claims "`black_box` compiles to zero
instructions; the value is already in a GPR." False under rustc 1.98.0:
each `black_box(u32)` lowers to an empty inline-asm block that forces the
value through a stack slot — store, slot-address materialization, reload:

```
add   w1, w1, w30          ; F result complete
stur  w1, [x29, #-92]      ; black_box: spill
sub   x5, x29, #92         ; black_box: slot address for the asm operand
; InlineAsm Start / End     ; (empty)
ldur  w8, [x29, #-92]      ; black_box: reload
eor   w8, w8, w10          ; only now: y ^= F
```

Per pair-iteration of `LBB25_5` (8 half-rounds): **8 `stur` + 8 `ldur` +
5 address-`sub`**; per encipher ≈ 64 store→load round trips. Confirmed in the
linked binary (`micro-0700571d4381d3d2`, slot `[x29, #-0x5c]`). Each round
trip sits **on the per-lane Feistel dependency chain**: store-to-load
forwarding (~3-4 cycles on M5) replaces a 0-cycle register rename.

Estimated cost: ~16 round trips per lane-chain per encipher ≈ +48-64 cycles
of the measured ~235 cycles per 4-lane encipher, plus ~170 instructions of
issue bandwidth (~9% of the loop). Largest single headroom found:
roughly **+20-25%** NEON throughput (2253 → ~2700-2900 h/s) if removed.

**Prescription:** replace `core::hint::black_box` with a register-preserving
opaque identity that LLVM's SLP still cannot see through, e.g. a tiny
`#[inline(always)] fn opaque(v: u32) -> u32` wrapping
`core::arch::asm!("", inout(reg) v, options(preserves_flags, nostack, nomem))`.
Empty `asm` with an `inout(reg)` operand is opaque to the vectorizer (it is an
unknown operation on the value) but compiles to literally zero instructions
and zero memory traffic. Verify with `--emit asm` that SLP does not return
and that no stur/ldur pairs remain in `LBB25_5`.

### 2. NEON: lanes 2-3 re-load gather base pointers from the stack every half-round

**File:** src/eks/neon.rs:265-281 (`f_word`), prologue pattern from the
"16 lane-adjusted box bases" design (neon.rs:369-375).

`rounds16_scalar!` needs 16 live base pointers (4 boxes × 4 lanes, lane*4
folded into the base). 16 pointers + 8 lane registers + P-walk pointer +
temps > 31 GPRs, so LLVM keeps lanes 0/1's bases in registers and **reloads
lanes 2/3's bases from the stack on every half-round**, inside `LBB25_5`:

```
lsr   w5, w1, #22
and   w5, w5, #0x3fc
ldr   x9, [sp, #48]        ; reload box0|lane2 base (loop-invariant!)
ldr   w5, [x9, w5, uxtw #2]; dependent chain: reload → gather
...
ldp   x16, x26, [sp, #88]  ; two more bases, same half-round
ldr   x12, [sp, #168]      ; lane 3: four reloads per half-round
```

Counted: **21 `ldr`/`ldp` of x-registers from `[sp]` per pair-iteration**
(plus 6 reg-reg `mov`s). The reloads are loop-invariant values; worse, each
feeds a gather address, adding a dependent L1 load (~4 cycles) ahead of the
S-box load on lanes 2/3's chains — the lockstep group finishes at the slowest
lane. Confirmed in the linked binary (`ldr x8, [sp, #0x50]` etc.).

**Prescription:** cut live bases from 16 to 4 by folding the lane into the
extracted index instead of the pointer: index register already has low two
bits clear (`(x>>22)&0x3fc` = a*4), so `orr wIdx, wIdx, #lane` (one ALU op,
off the critical path relative to a 4-cycle dependent load) and a single
shared `box_base[b]` per box. Frees ~10-12 GPRs — likely also relieves the
spill pressure behind finding 1 — and removes ~14 loads per pair-iteration.
Alternative: keep lane-adjusted bases only for lanes 0/1 and let lanes 2/3
use `idx*4+lane` arithmetic; measure both. This was never A/B-measured for
`rounds16_scalar!` (the tuning notes only cover the vector lookups).

### 3. AVX-512 insert flavor: `f16_insert` is an outlined call — 16 calls per encipher

**File:** src/eks/avx512.rs:223-244 (`f16_insert`, `#[target_feature(enable =
"avx512f")] #[inline]`), dispatched via `f16::<GATHER>` at avx512.rs:252.

LLVM's cost model refused to inline the 16-lane insert F (the AVX2 8-lane
sibling *was* inlined — zero `callq` in `encrypt_zero_chain_v8Kb0`). The
insert zero chain therefore makes **2 `callq` per round, 16 per encipher**
(80 static call sites crate-wide; confirmed in the freshly linked x86 binary
`micro-db28881850567f10`), each wrapped in ABI overhead:

```
vmovdqa64 %zmm1, (%rsp)          ; spill r (all zmm caller-saved)
vpxord    65536(%rbx), %zmm0, %zmm0
vmovdqa64 %zmm0, 128(%rsp)       ; spill l'
movq      %r14, %rdi             ; sret pointer (result returned BY MEMORY)
movq      %rbx, %rsi
callq     …avx51210f16_insert
vmovdqa64 64(%rsp), %zmm0        ; reload result from the sret slot
vmovdqa64 (%rsp), %zmm1          ; reload r
vpternlogd $150, 65600(%rbx), %zmm1, %zmm0
```

x goes in zmm0, but the `__m512i` **return** uses a hidden sret pointer
(Rust ABI won't return 512-bit vectors in registers), and every call forces
2×64-byte spills + 2×64-byte reloads plus call/ret — roughly 190 extra
instructions and 64 64-byte memory ops per encipher, ~10-15% of the AVX-512
insert kernel's budget. And it is what the Zen 4 shootout had to
beat the (inlined, but 81-uop-microcoded) gather flavor with.

**Prescription:** force inlining. `#[inline(always)]` on `#[target_feature]`
functions has been accepted since Rust 1.69 (the neon.rs:369-371 comment
claiming it is "rejected" is stale — it predates the stabilization); put it
on `f16_insert`/`f16_gather`/`f16`, or macro-ize the F body the way
`rounds16!` macro-izes the rounds. The sret slot, the zmm spills, and all 16
calls per encipher dissolve; the 16 scalar loads per lookup then schedule
against each other across round boundaries as they do in the AVX2 kernel.

### 4. AVX2 insert flavor: 32 `shll $5` per F — indices never prescaled in vector form

**File:** src/eks/avx2.rs:130-154 (`lookup8_insert`): the source stores the
*word* index vector and computes `ix[l] as usize * 8` per lane in GPR land.

LLVM (credit where due) did not round-trip the stack arrays: it extracts
lanes with `vpextrd`/`vmovd`/`vpshufb`+`vpextrb` and folds every gather load
as a **memory operand** of the rebuild instruction, with box base and lane*4
folded into the displacement (`vpinsrd $1, 24580(%rdi,%r13), %xmm5, %xmm5`).
But the ×32-byte scaling is paid per lane, in GPRs, every lookup:

```
vpextrd $1, %xmm5, %r13d
shll    $5, %r13d            ; ×32: once per lane per box
…
vpinsrd $1, 24580(%rdi,%r13), %xmm5, %xmm5
```

**1024 `shll` in `encrypt_zero_chain_v8Kb0`** (32 per F × 16 rounds × 2 loop
bodies), ~28% of the ~115 instructions per F. The gather flavor already
prescales in vector form (`vpsrld $13` + `vpand` with 0xff<<3 = idx*8), and
NEON spelling (c) prescales byte offsets for the same reason (neon.rs:151-158
measured it winning). The insert flavor never got the equivalent.

**Prescription:** prescale before extraction — compute the byte-offset vector
once per index (`off = (idx << 5) | lane_byte_offsets`, 2 vector ops per box,
or fold the shift into the mask as `(x>>19) & 0x1FE0` for a, `(x<<5)&0x1FE0`
for d, etc. — 8 vector ops per F total) and store/extract *finished* byte
offsets so each gather is a displacement-only `vpinsrd $k, off(%rdi,%rN)`.
Removes 32 GPR shifts per F (~28% of F instruction count) for 8 cheap vector
ops; net ≈ −24 instructions/F, expect ~+10-20% on Zen 4's insert path where
the kernel is throughput-bound at IPC ≈ 1.8.

### 5. AVX-512 insert: same unscaling, inside the outlined body

**File:** src/eks/avx512.rs:150-190 (`lookup16_insert`).

`f16_insert`'s 236-instruction body contains **64 `shll $6`** (16 lanes × 4
boxes) — same defect as finding 4, doubled. Same prescription (prescale to
byte offsets in vector form: stride is ×64 here), and it composes with
finding 3: once inlined, the prescaled offsets also shorten the address
dependency for the 16 parallel loads.

### 6. NEON: SoA ×16 stride forces 2-ALU index extraction (structural)

**File:** src/eks/neon.rs:66-68 (`SBoxes4` layout), 265-281 (`f_word`).

AArch64 32-bit `ldr [base, wm, uxtw #2]` scales by exactly 4. The SoA entry
stride is 16 bytes, so the compiler must synthesize idx*4 in a register
(`lsr w1, w2, #22 ; and w1, w1, #0x3fc`) and scale by 4 in the addressing
mode — 2 ALU ops for a/b/c vs scalar's 1 (`ubfx`/`lsr` + `uxtw #2`, stride 4).
That is +3 ALU per half-round ≈ +192 instructions per encipher (~10% of the
loop body). Per-lane stride-4 boxes would restore the scalar addressing shape
at the cost of 4 scattered expansion stores instead of one 16-byte `stp` per
entry (the reason SoA was chosen). Medium confidence — worth one measured
experiment only after findings 1-2 land, since those free the registers that
a 4-base-per-lane layout would need.

### 7. Minor / architectural (listed for completeness)

- **AVX2/AVX-512 gather flavors** (avx2.rs:115-119, avx512.rs:133-138):
  `vpcmpeqd`+`vpxor` (AVX2) / `kxnorw`+`vpxor` (AVX-512) re-materialize the
  all-ones mask and zero the destination before **every** gather —
  128 instructions/encipher. Architectural: `vpgatherdd` zeroes its mask
  operand, so the mask cannot be hoisted as-is. `vmovdqa` from a hoisted
  mask register would be rename-eliminated (0 ports); LLVM chooses
  `vpcmpeqd`. Not source-actionable; front-end bandwidth only.
- **93 `panic_bounds_check` call sites (x86)**: all cold — 32 in avx512
  `transpose_keys`/`transpose_salts` (once per group), the rest in bcrypt8
  output writeback, base64, scalar per-hash lane indexing. **Zero** in any
  `encrypt_zero_chain*` on either architecture. Cosmetic.

---

## What the compilers got right (expected vs actual, per F)

Expected per-F shape (scalar reference): 4 index extracts, 4 independent
S-box loads, `((va+vb)^vc)+vd`, plus the round `l ^= P[i]` / `r ^= F`.

**aarch64 scalar** (`encrypt_zero_chain`, 483 lines, 0 stack refs, 0 calls,
3 loop-control branches, no bounds checks — all S-box addressing is
`ldr wN, [xBase, wM, uxtw #2]`): per round 14 instructions — the ISA floor.
Loads for a/b/c issue back-to-back; the S3[d] load is deliberately scheduled
*after* the first `add`/`eor` so its latency lands exactly when the combine
needs it (balanced, not serialized). The P-array chain is scheduled **one
round ahead**: `ldr w2, [x0, #4100] ; eor w15, w2, w15` (P[i+1] load + XOR)
issues while F(i) is still in flight. Nothing to fix.

**x86 scalar**: ~13.5 instructions/round, also at the floor — fused load-op
forms (`addl (%rdi,%rax,4), %ecx`, `xorl 2048(%rdi,%rax,4), %ecx`), P XOR as a
memory operand, prescaling (`shrl $14 ; andl $1020` = b*4), and `movzbl %bh`
to get byte c with no shift. No bounds checks, no spills in the loop.

**NEON P-loops**: `f4_split` **is** fully inlined into `rounds16!` (umov.h
extracts + inline gathers; the only `bl` in the whole zero chain is one
`memcpy` per chain for the `pw` snapshot). P-loops are 9/521 of the chain —
fine as-is.

**AVX2 gather**: 4 gathers per F issue back-to-back; index math strength-
reduced (prescaled masks, `vpor` for lane_off); P XOR folded as memory
operand. Round-pair lookups are **not** interleaved — correctly so: the
second F of a pair consumes `r ^ F(l)`, a data dependency; latency hiding
comes from the 8 lanes inside each gather, not across rounds.

**AVX-512 gather**: cleanest of all — index math is `vpsrld` + one
`vpternlogd $236` (and+or fused) per lookup, constants hoisted in zmm
registers, zero calls. It loses on Zen 4 for µarch reasons (81-uop microcoded
gather, 64 KiB SoA vs 32 KiB L1d), not codegen reasons.

---

## Cycle budgets (measured h/s → cycles per F)

Work per hash: 521 + 2^(cost+1)·521 + 192 encryptions
(cost 5: 34,057; cost 4: 17,193). Assumptions: M5 Max ≈ 4.5 GHz (implied
4.34 from the 646 µs cost-4 single-hash point: 17193 encs, 163 cyc/enc);
Zen 4 EPYC ≈ 3.7-4.15 GHz; batch groups run sequentially on one thread.

| backend | measured | ns / lockstep-enc | cycles/enc | per round | floor/round | utilization |
|---|---|---|---|---|---|---|
| M5 scalar | 811.6 h/s | 36.2 (1-lane) | ~163 | ~10.2 | ~9-10 (extract 1 + L1 ~4 + 4 ALU) | ~100% — done |
| M5 NEON | 2252.8 h/s | 52.1 (4-lane) | ~235 | ~14.7/lane | ~9-10 | ~65% — findings 1,2,6 |
| Zen4 scalar | 564.2 h/s | 52.0 (1-lane) | ~192-216 | ~12-13.5 | ~10-11 | ~90% |
| Zen4 AVX2 ins | 869.7 h/s | 270.1 (8-lane) | ~1000-1120 | ~63-70/F | chain ~15, but IPC-bound 1.8 | throughput-bound; finding 4 |
| Zen4 AVX-512 ins | 798.5 h/s | 588.4 (16-lane) | ~2180-2440 | ~137-153/F | same shape + call overhead | findings 3,5 |

Scalar is latency-bound at the Feistel+L1 floor on both ISAs — no codegen
headroom remains there. NEON at IPC ≈ 8.3 is *simultaneously* near the issue
ceiling and 1.5× off its latency floor, which is why the pure-overhead
removals (findings 1+2 ≈ 340 instructions + ~50 chain cycles per encipher)
project +20-30%. AVX2/AVX-512 insert are throughput-bound; the headroom is
instruction count (findings 4, 5) and call overhead (finding 3).

## Could not disassemble / caveats

- **rlib bitcode**: `lto = "thin"` embeds LLVM bitcode; Apple `nm` errors
  (`Unknown attribute kind (105)`), and `llvm-objdump` only decodes the
  archive's rmeta members. Worked around with `--emit asm` + linked-binary
  disassembly; all hot-loop claims verified in the linked `micro-*` binaries
  of both architectures.
- **Zen 4 numbers** are from the README (Cloudflare sandbox, rustc 1.99.0);
  this audit read Rosetta-cross-built x86 objects (per the task, fine to
  READ). Cycle/enc for x86 assumes 3.7-4.15 GHz; the sandbox clock was not
  re-measured here.
- **Findings 1, 2, 4, 5 interactions**: removing register pressure (2) may
  change how (1) spills; estimates are per-defect, not strictly additive.
  All projections are static-count estimates, not re-benchmarks.
