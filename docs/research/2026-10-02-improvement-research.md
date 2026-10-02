# bcrypt-rust: improvement research (2026-10-02)

Synthesis of five research streams (supporting documents, all cited):
[JtR/cracker techniques](research-jtr-bcrypt.md) ·
[x86 microarchitecture](research-x86-microarch.md) ·
[Apple silicon ceiling](research-apple-silicon.md) ·
[wasm headroom](research-wasm-simd.md) ·
[our codegen audit](research-codegen-audit.md) —
plus two empirical experiments run for this report (PGO on M5 Max; perf in the
Cloudflare sandbox) and two probes verifying contested claims.

Baselines (correctness-gated, cost 5, batch 16 unless noted):
M5 Max: scalar 812 h/s, NEON 2.78×. Zen 4 (4 vCPU): scalar 564 h/s,
SSE4.1 1.12×, AVX2 1.54×, AVX-512 1.42×. wasm128 under wasmtime: 1.21×.

## Headline context

- **John the Ripper ships no SIMD bcrypt.** Every hardware-gather attempt lost
  to its scalar GPR interleave (Haswell AVX2×8 gather: 4,186 c/s vs scalar-X2
  6,595 c/s). Our insert-chain SoA beating scalar on Zen 4 (1.54×) already
  exceeds JtR's x86 results; our NEON GPR-resident round loop is their design.
- **Our scalar is at the ISA floor on both ISAs** (measured ~10.2 cycles/round
  vs a 9–10 cycle dependency-chain floor). Single-hash work is done; all
  remaining headroom is in the batch kernels.
- **The 4 KiB/hash S-box wall is irreducible**: on 32 KiB-L1d cores
  (Zen 3/4) any interleave saturates near 8 hashes; 48 KiB cores
  (Zen 5, Ice Lake+, SPR) raise the knee to ~12. AVX-512's Zen 4 loss is
  microarchitecture (81-µop microcoded gather + 64 KiB working set), not a
  code bug.

## Tier 1 — implement (headroom verified in our own disassembly)

| # | change | evidence | expected |
|---|---|---|---|
| 1 | NEON: replace `black_box` with register-identity `asm!` | `black_box(u32)` lowers to `str`+`ldr` on the critical chain — 64 store→load forwards per encipher; `asm!("/* {0:w} */", inout(reg) v)` adds zero instructions (probe-verified) | +20–25% NEON (~2700–2900 h/s) |
| 2 | NEON: fold lane into the S-box index (`orr`, low bits clear) with 4 shared bases | lanes 2/3 currently reload 21 base pointers from stack per half-round (dependent loads on the chain) | frees ~12 GPRs; enables #3 |
| 3 | NEON: 6-lane (then 8-lane) GPR interleave | ceiling model calibrated to our measurements: knee at N≈6 → 3,600–4,200 h/s; N=8 → 4,200–4,700 h/s | **+60–110%** — the biggest win available |
| 4 | AVX2/AVX-512 insert flavors: prescale byte-offset vectors | 28% of AVX2 F() instructions are per-lane `shll $5` (AVX-512: 27% `shll $6`); the gather flavor and NEON already prescale | up to ~+15–25% on insert flavor |
| 5 | AVX-512: macro-ize `f16_insert` (textual inline, like `rounds16!`) | LLVM outlined it: 16 `callq`/encipher with zmm sret spills. NOTE: `#[inline(always)]`+`#[target_feature]` is **still a hard error** (probe-verified, rust#145574) — macro route only | +10–15% on the AVX-512 path |
| 6 | x86: third shootout arm — extract flavor (`vpextrd`→scalar load→`vpinsrd`) | removes 32 loads + four 6–8c store-forward links per F vs insert; shootout already asserts flavor equality | +10–20% on Zen |

## Tier 2 — dispatch/policy

| # | change | evidence |
|---|---|---|
| 7 | Extend the runtime shootout to width selection (AVX2 vs AVX-512), not just flavor | Zen 4: zmm gather = 2×ymm in cycles AND µops; Zen 5 improves gathers (12.6c zmm) but 64 KiB > 48 KiB L1d persists; Sapphire Rapids is the one uarch where zmm gather clearly wins (6.0c). GCC and LLVM both mis-select gathers on znver4/5 — runtime measurement is the only reliable selector |
| 8 | wasm: X2 kernel (8 lanes, two interleaved v128 states) | the only wasm shape with native evidence of beating the 4-lane ceiling (JtR X2 > AVX2 gather); current 1.21× is ~90–95% of the single-state ceiling |

## Dead ends — verified, do not pursue

| idea | verdict |
|---|---|
| PGO | measured: 813.8 vs 817.3 h/s scalar, 2229.9 vs 2238.0 NEON — noise (±0.5%) |
| `vpternlogd` fusion | op-by-op audit: F() is add→xor→add; zero saveable instructions |
| F() algebraic reassociation | impossible — (A^B)+C ≠ (A+C)^B (counterexample in JtR report §6) |
| cross-round pipelining / cost-loop cache-blocking / prefetch | Feistel chain + mutating S-boxes forbid it; PRFM costs a load slot on an L1-resident set |
| hardware gathers on any AMD | microcoded through Zen 5 (12.6c zmm); GCC/LLVM disable for znver4/5 |
| relaxed-simd / memory64 / shuffled tables / bitslicing | nondeterminism or quantified loss (max runtime table in wasm = 16 bytes) |
| SME/AMX on Apple | outer-product engines, no gather addressing |
| 16-lane groups on 32 KiB L1d | the cache wall itself |
| perf counters in the CF sandbox | microVM blocks the PMU — noted for future bench work |

## SMT note

`parallel` already covers multi-core; JtR's data says SMT siblings add
+20–40% for this latency-bound loop and can invert the X2-vs-X3 choice — worth
a sentence in the `parallel` docs rather than code.

## Proposed implementation order

1 → 2 → 3 (NEON sequence, each measured independently) → 4+5+6 (x86 sequence,
shootout asserts equality per flavor) → 7 → 8. Every step gated on the full
suite + micro numbers on M5 Max and the Zen 4 sandbox.
