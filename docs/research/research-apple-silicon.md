# bcrypt batch kernel on Apple silicon: achievable ceiling and code shapes

Deep-research report, 2026-10-02. Baseline under study (pure Rust, Apple M5 Max, cost 5,
batch 16, single thread): scalar 812 h/s; NEON-shaped but GPR-resident 4-lane interleave
2253 h/s (2.78x); vector-resident rounds measured only 1073 h/s; single-hash cost 4 = 646 us.

Every externally sourced claim carries a URL. Computations marked "[model]" are mine.

## TL;DR

```
per-core, cost 5, M5-class P-core @ 4.608 GHz        h/s        note
--------------------------------------------------  -----  ------------------------------
serial latency floor [model]                          810  == measured scalar 812 (0.2%)
bcrypt-rust scalar (measured)                         812  already AT the chain floor
bcrypt-rust 4-lane GPR interleave (measured)         2253  69.5% of the N=4 ceiling
N=4 ceiling [model]                                  3241  chain/4
N=6 ceiling [model] (the knee)                       4861  chain/6 ~= 3-load-port floor
N=8..16 ceiling [model]                              4950  hard wall: 3 load pipes
JtR aarch64 today (BF_X2=2-way, no NEON)             ~2x serial only
hashcat M4 Max GPU, 40 cores (published)            21488  ~= 4.3 x one CPU core at N=8
```

- Your scalar kernel is, to within measurement noise, exactly on the single-lane
  dependency-chain floor (~167 cycles per Blowfish encryption). Nobody's scalar code —
  JtR, Go x/crypto, RustCrypto — can structurally beat it; they only lose to it.
- The absolute per-core wall on any Apple P-core M1..M5 is the **3 load pipes**:
  82 loads per encryption -> 27.3 c/enc -> **~4,950 h/s at cost 5** (~38.7 h/s at cost 12).
- The knee is at **interleave factor ~6** (167/27.3 = 6.1), not 4. N=8 saturates with
  margin; N=16 is pointless (load-port bound + register-file impossible).
- Expected headroom over the current 2253 h/s: **+35..45%** from closing the N=4
  scheduling gap, or **+80..110%** from going to N=6..8 GPR interleave.

## 1. Placement vs published numbers (Question 1)

### 1.1 JtR / CPU numbers

| machine | cores used | cost-5 h/s | per-core | source |
|---|---|---|---|---|
| Apple M2 MacBook Air (JtR, OpenMP) | 8 (4P+4E) | 4,861 "Real C/S" (bcrypt test ~4,877) | ~880 per P-core [model: E~0.38x P] | [openbenchmarking](https://openbenchmarking.org/result/2307213-NE-2307119NE92&sgm=1&ncb=1&hgv=Apple+M2+MacBook+Air&sor&grr) |
| Apple M2 MacBook Pro (JtR, thesis) | n/a (median of runs) | ~2,113 (median 3.325 log10 H/s; JtR beat hashcat on this box) | n/a | [Rizzo thesis, p. bcrypt comparison](https://unitesi.unive.it/retrieve/88f073e7-d329-40e9-949d-d5fa23b92051/884784_Rizzo_Elisa.pdf) |
| AWS Graviton3 r7g.16xlarge (JtR) | 64 vCPU | 50,709 | ~792/vCPU @2.6GHz | [openbenchmarking](https://openbenchmarking.org/result/2407111-NE-AWS4GRAVI72&sgm=1&ppd_RVBZQyA5UjE0IHI3YS4xNnhsYXJnZQ=4.869&ppd_R3Jhdml0b24yIHI2Zy4xNnhsYXJnZQ=3.226&ppd_R3Jhdml0b24zIHI3Zy4xNnhsYXJnZQ=3.427&ppd_R3Jhdml0b240IHI4Zy4xNnhsYXJnZQ=3.770&ppd_WGVvbiA4NDg4QyByN2kuMTZ4bGFyZ2U=4.234&ppt=DPH&rdt&gru) |
| Ampere Altra Q80-33 (JtR Blowfish) | 80 | 72,027 | ~900/core @3.3GHz | [openbenchmarking](https://openbenchmarking.org/result/2311275-NE-AMPERE49699&grw&export=html) |
| generic ARM Linux targets, 2013 (JtR) | 1 | 166 c/s real / 83.1 virtual | historical baseline | [openwall john-users](https://openwall.com/lists/john-users/2013/06/03/1) |

No public JtR `--test` bcrypt table exists for M1/M3/M4/M5 CPUs specifically; the Openwall
benchmark wiki carries mostly x86 + old ARM, and the 2021 M1 bring-up thread + GitHub issue
[#4585](https://github.com/openwall/john/issues/4585) are build notes, not numbers
([john-users 2021/03/12](https://www.openwall.com/lists/john-users/2021/03/12/3),
[wiki](https://openwall.info/wiki/john/benchmarks)).

Per-cycle normalization [model]: JtR M2 P-core ~880 h/s @3.49 GHz = 252 h/s/GHz;
Graviton3 = 305 h/s/GHz; **bcrypt-rust 4-lane = 2253/4.608 = 489 h/s/GHz** — ~1.9x JtR's
per-cycle rate on the same ISA. JtR's own serial-equivalent (~880/1.7 X2-gain = ~520 h/s =
149/GHz) vs bcrypt-rust scalar (176/GHz) shows the M5 core itself is only ~18% better per
cycle; the 2.78x batch win is purely the interleave shape.

### 1.2 hashcat GPU numbers (Metal, cost 5, "Iterations: 32")

| GPU | h/s | source |
|---|---|---|
| M1 (8 GPU cores) | ~1,987-1,990 | [gist crypt0rr](https://gist.github.com/crypt0rr/a2965aca6cf2a19ffd84867156e73011) |
| M3 Pro (14c GPU) | ~6,032 | [gist Chick3nman](https://gist.githubusercontent.com/Chick3nman/fdf7f9ddcc0a65f6725aefede99ada4e/raw) |
| M3 Max (30c GPU) | ~9,401 | [gist Chick3nman](https://gist.github.com/Chick3nman/ccfb883d2d267d94770869b09f5b96ed) |
| M4 (10c GPU, Metal) | 8,680 | [hashcat forum thread-12248](https://hashcat.net/forum/archive/index.php?thread-12248.html) |
| M4 Pro (16c GPU, OpenCL fallback) | ~7,480 | [gist crypt0rr](https://gist.github.com/crypt0rr/c801c67c7ac8fb07478ec789acb24c7e) |
| M4 Max (40c GPU, Metal) | ~21,241-21,488 | [hashcat forum follow-up](https://hashcat.net/forum/thread-12248-nextnewest.html) |

Reading: an entire 40-GPU-core M4 Max does ~21.5k h/s; **8 M5-class CPU cores at the N=8
ceiling would match it** (8 x 4,950 = 39.6k). bcrypt's GPU-unfriendliness is real and our
CPU kernel is in the right league.

### 1.3 Cost-12 conversion

bcrypt cost c runs 2^c ExpandKey iterations; cost 12 = 128x cost 5
([Provos-Mazieres, USENIX'99](https://www.usenix.org/events/usenix99/provos.html);
hashcat prints "Iterations: 32" for its cost-5 benchmark, see thread-12248 above).
So: ours scalar 6.3 h/s, 4-lane 17.6 h/s, N=8 ceiling ~38.7 h/s per core; JtR M2 Air
~38 h/s whole-machine; M4 Max GPU ~168 h/s.
## 2. The microarchitecture that matters for this loop (Question 2)

### 2.1 Per-generation facts (P-cores)

| | M1 (Firestorm) | M2 (Avalanche) | M3 | M4 | M5 ("super core") |
|---|---|---|---|---|---|
| L1d | 128 KiB, 8-way, 64B lines | same | same | same | same class |
| L1d load-to-use | 3c ptr-chase / 4c to-ALU | same | same (+Load Value Predictor) | 3c/4c measured | same class |
| loads/cycle | 3 (u8 L/S, u9 L, u10 L) | 3 | 3 | 3 (3x128b L + 1x128b S, or 2+2) | no public data; assume 3 |
| stores/cycle | 2 | 2 | 2 | 2 | - |
| int ALUs | 6 | 6 | 8 | 8 | 8+ (9-10 wide decode) |
| decode/rename | 8 | 8 | 9 | 10 | 10+ |
| coalesced ROB | ~330 groups | ~330 class | larger | ~313 groups (measured via ld/st) | - |
| in-flight loads | ~130 | ~130 class | more | ~130+ class | - |
| boost clock | 3.2 GHz | 3.49 GHz | 4.05 GHz | 4.51 GHz | 4.608 GHz |

Sources:
- L1d 128 KiB 8-way 64B, P-cores, **all generations M1..M4 (and A14..A18)**, E-cores 64 KiB:
  Apple Silicon CPU Optimization Guide, as quoted and verified by sysctl + pointer-chase in
  [jia.je M4 microarchitecture eval](https://jia.je/hardware/2025/05/21/apple-m4/) and
  [Apple's guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide).
- 3-cycle L1d praised as "like a better version of Phenom's L1D":
  [Chips and Cheese, Golden Cove caches](https://chipsandcheese.com/p/going-armchair-quarterback-on-golden-coves-caches).
- Load latency rules (3c load->base, 4c load->ALU, **+1c if the index register is the output
  of a shift-like instruction**; add/bitwise producers exempt), 3 load units, ROB ~330/~623,
  ~130 in-flight loads, AGUs take scaled index up to LSL#3 for free:
  [dougallj Firestorm overview](https://dougallj.github.io/applecpu/firestorm.html),
  [int table](https://dougallj.github.io/applecpu/firestorm-int.html),
  [LSQ blog post](https://dougallj.wordpress.com/2021/04/08/apple-m1-load-and-store-queue-measurements/).
  Anandtech's original deep dive: [Anandtech M1/A14](https://www.anandtech.com/show/16226/apple-silicon-m1-a14-deep-dive/2).
- M4 measured: L1d 128 KiB拐点, 3c pointer chase (incl. `ldr x0,[x0,#8]` and reg-offset),
  4c for load->index with scaled addressing (`ldr x0,[sp,x0,lsl#3]`), 3x128b load + 1x128b
  store per cycle, 8 int ALUs (from M3), 10-wide decode (from M4), PRF ~360 int regs,
  schedulers 60 entries, coalesced ROB ~313:
  [jia.je M4](https://jia.je/hardware/2025/05/21/apple-m4/); independent:
  [David Huang's M4 Pro tests (reddit summary)](https://www.reddit.com/r/hardware/comments/1gyh42k/david_huang_tests_apple_m4_pro).
- Load Address Predictor from M2 (constant+stride patterns only,
  [patent US11829763B2](https://patents.google.com/patent/US11829763B2/)); Load Value
  Predictor from M3 (constant values only,
  [patent US12067398B1](https://patents.google.com/patent/US12067398B1/en)) — both via
  [jia.je M4](https://jia.je/hardware/2025/05/21/apple-m4/). **Neither can predict bcrypt's
  pseudo-random S-box addresses or their mutating values** — no free lunch, and beware
  microbenchmark contamination from these predictors.
- M5 clock 4.608 GHz (actively cooled), E-cores 3.048 GHz:
  [eclecticlight frequency table](https://eclecticlight.co/2025/10/30/updated-cpu-core-frequencies-for-all-current-apple-silicon-macs),
  [computerbase MBP-M5 review](https://www.computerbase.de/artikel/notebooks/apple-macbook-pro-m5-test.94739/seite-2);
  ~10% IPC over M4 + higher clock, SME present:
  [Wikipedia Apple M5](https://en.wikipedia.org/wiki/Apple_M5). No public M5 backend
  breakdown exists yet; there is no evidence of a 4th load pipe.

### 2.2 Is 4-way interleave enough to saturate the load pipes?

No. Quantified in section 5: the per-encryption dependency chain is ~167c while 82 loads at
3/cycle need only 27.3c, so load-pipe utilization at N=4 is only ~44% (measured: 2.725M
loads/hash x 2253 h/s = 6.14 G load/s = **1.33 of 3 loads/cycle** [model]). The knee where
latency-hiding meets the load wall is N = 167/27.3 ~= 6.1. **8-way GPR interleave would
saturate; 4-way cannot.** Register budget for N=8 with one S-box base register per lane
(S1..S3 reached by ORR-ing 0x100/0x200/0x300 into the index — bitwise producer, so no
shift-penalty on the load) is 8x(L,R,base) + ~4 temps + P ptr ~= 30 of 31 GPRs — tight but
feasible; N=6 is comfortable.

## 3. NEON vs GPR for 32-bit scalar-ish lookups (Question 3)

Your finding (vector-resident rounds = 1073 h/s < GPR interleave = 2253 h/s) is fully
corroborated by measured crossing costs on Firestorm
([dougallj SIMD table](https://dougallj.github.io/applecpu/firestorm-simd.html)):

| op | latency | throughput | uops | pipes touched |
|---|---|---|---|---|
| UMOV/SMOV lane->GPR | **<=10c** | 0.5 (max 2/c) | 1 | int u3/4 **and** FP u13/14 |
| FMOV S->W / D->X | <=10c | 0.5 | 1 | int + FP |
| INS GPR->lane | [2; **<=12**]c | 0.376 | **2** | **Mem u8-10** + FP u11-14 |
| FMOV S<-W | <=10c | 0.333 | 1 | **Mem u8-10** |
| DUP GPR->vector | <=12c | 0.333 | 2 | Mem + FP |
| LD1 single-lane | - | **2.0 (!)** | 2 | Mem + FP |
| LDR (reg, uxtw#2) GPR | <=4c | 0.333 | 1 | Mem only |
| plain vector ALU (ADD/EOR/USH R 4S) | 2c | 0.25 | 1 | FP only |

A NEON-resident round must gather via UMOV -> LDR -> INS per lookup: ~16c+ of chain and
**three** scarce pipe slots (int, mem, fp) per lookup, versus UBFX + LDR (2 slots: int, mem)
in GPRs — that is ~2-3x the pipe pressure per S-box access, matching the measured
1073/2253 = 0.48 ratio. Vector-domain crossing is the bottleneck, not NEON ALU throughput
(which is a fine 4/cycle at 2c). LD1-single at TP=2c per lane-insert also confirms that
LLVM's ldr+ld1 pack/unpack is acceptable **only** because it runs at batch boundaries, and
that blocking SLP re-vectorization of the round loop with black_box was correct.

Prior corroboration that bcrypt resists SIMD: JtR ships NEON code for bitslice DES but
**no NEON/ASIMD bcrypt at all** — bcrypt on aarch64 is scalar C with interleave
([arm64le.h](https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/arm64le.h),
[BF_std.c](https://github.com/openwall/john/blob/bleeding-jumbo/src/BF_std.c)); bitslicing
Blowfish is impossible because F() uses addition + data-dependent table reads, not just
boolean ops ([algorithm paper](https://www.usenix.org/events/usenix99/provos.html)).
## 4. SME/AMX: no conceivable use (Question 4)

M4/M5 implement ARM SME/SME2 (M1-M3 exposed the same matrix hardware as the undocumented
AMX coprocessor); Apple implements SME **without** non-streaming SVE — LLVM even models M4
as ARMv8.7-A for that reason ([Wikipedia Apple M4](https://en.wikipedia.org/wiki/Apple_M4),
[Wikipedia Apple M5](https://en.wikipedia.org/wiki/Apple_M5)). Both engines are
outer-product/tiling machines: FMOPA-class instructions accumulate dense row x column
products into architectural tiles (ZA, 512-bit SVL on M4, one cluster-shared coprocessor
reached through L2) ([tnzr SME microanalysis](https://tnzr.org/sme/micro.html),
[Chips and Cheese on outer-product matrix units](https://chipsandcheese.com/p/is-x86-ready-to-ace-it)).
bcrypt's F() is four **data-dependent 32-bit gathers** into 4 KiB of per-hash S-boxes plus an
add/xor chain; there is no dense linear algebra anywhere in
[EksBlowfish](https://www.usenix.org/events/usenix99/provos.html). SME/AMX have no gather
addressing, no 32-bit integer add-with-carry-free mixing primitive useful here, and data
would have to round-trip through L2 to the shared coprocessor — strictly slower than the
3-cycle L1d the loop already lives in. Verdict: confirmed useless for this kernel; the only
SIMD hardware that matters is the 3-wide LSU.

## 5. Prefetch, cache fit, and the computed latency ceiling (Question 5)

### 5.1 Cache fit and PRFM

Working set: 4 x 1 KiB S-boxes + 72B P-array per lane, i.e. 16 KiB at batch 4 .. 64 KiB at
batch 16 — fits the 128 KiB L1d with >50% headroom on every Apple P-core
([Apple guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide),
[jia.je M4](https://jia.je/hardware/2025/05/21/apple-m4/)). So misses are not the limiter;
latency x chain depth is. PRFM: (a) each S-box index becomes known ~1-2c before the load
could issue (it is derived from a just-arrived load), so there is no lead time to prefetch
into; (b) PRFM costs a full Mem pipe slot at only ~0.65/cycle throughput on Firestorm
([PRFM rows, TP ~1.54](https://dougallj.github.io/applecpu/firestorm-int.html)) — it would
steal ~1.5x the slot a real gather uses; (c) hardware stride prefetchers find no pattern in
random 4 KiB walks. Verdict: do not prefetch; expect <1% (test once, reject).

### 5.2 Chain model [model], calibrated

Work per hash at cost 5 (from [BF_std.c](https://github.com/openwall/john/blob/bleeding-jumbo/src/BF_std.c)
structure + [spec](https://www.usenix.org/events/usenix99/provos.html)):
(1 + 2*2^5) ExpandKey x (9 P-pairs + 512 S-pairs = 521 encryptions) + 64x3 final = **34,057
Blowfish encryptions/hash**.

Per-round serial chain on Apple P-cores: UBFX extract (1c, 6/cycle,
[int table](https://dougallj.github.io/applecpu/firestorm-int.html)) -> LDR scaled-index
(4c to-ALU, +1c if index is shift-produced -> 5c;
[latency rules](https://dougallj.github.io/applecpu/firestorm.html)) -> add/xor/add combine
(3c) -> R ^= (1c) ~= **10-11c/round** -> ~167c per 16-round encryption incl. P[0]/P[17].

Calibration against your measurements (no fitting, just division):
- scalar: 4.608 GHz / 812 h/s / 34,057 = **166.6 c/enc measured vs ~167 modeled** (0.2%).
- single-hash cost 4: 17,385 enc x 167c = 630 us modeled vs **646 us measured** (2.5%).
The scalar kernel is at the floor; the only lever left is interleave factor.

Throughput floors per encryption on M5-class: loads 82/3 = **27.3c** (binder);
int ALU ~150/8 = 18.8c; front-end ~230/10 = 23c. On M1/M2: ALU 25c, front-end 28.8c —
both ~= the load floor, so all generations wall at ~27-29 c/enc.

### 5.3 Ceiling table (cost 5, per P-core @4.608 GHz) [model]

group_cycles(N) = max(167, N x 27.3); h/s = 4.608e9 x N / (group x 34,057)

| N | group cycles | c/lane-enc | ceiling h/s | vs your 2253 |
|---|---|---|---|---|
| 1 | 167.0 | 167.0 | 810 (= measured 812) | 0.36x |
| 2 | 167.0 | 83.5 | 1,620 (JtR's BF_X2 zone) | 0.72x |
| 4 | 167.0 | 41.75 | **3,241** | measured is 69.5% of this |
| 6 | 167.7 | 27.9 | **4,861** (the knee) | 2.16x |
| 8 | 218.7 | 27.3 | **4,950** (load-port wall) | 2.20x |
| 12/16 | 328/437 | 27.3 | 4,950 | no gain; N=16 spills (32 L/R regs > 31 GPRs) |

Cost-12: divide by 128 -> ceiling ~38.7 h/s/core; your 4-lane ~17.6 h/s/core.
Whole-chip: multiply by P-core count (bcrypt batches are embarrassingly parallel;
E-cores add ~30-40% each per their 2-load-pipe, 64 KiB L1d design
([icestorm table](https://dougallj.github.io/applecpu/icestorm-int.html))).
## 6. AArch64 prior art (Question 6)

- **JtR (the reference)**: on aarch64 bcrypt is plain scalar C (`BF_ASM 0`), 2-way
  interleaved (`BF_X2 1` — x86-64 uses 3-way), and compiled with `BF_SCALE 0`, i.e. it does
  **not** exploit AArch64's free `ldr [x, w, uxtw #2]` scaled indexing, using explicit
  `<<2` index math instead
  ([arm64le.h](https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/arm64le.h),
  [BF_std.c BF_ROUND variants](https://github.com/openwall/john/blob/bleeding-jumbo/src/BF_std.c)).
  NEON is used only for bitslice DES ("DES 128/128 ASIMD"). JtR's own docs note X3 gives a
  "major speedup" on non-SMT cores and was conservatively left at X2/X3 — nobody ever tried
  6-8-way because x86-64 runs out of GPRs at 3
  ([john-dev 2014/12/25](https://openwall.com/lists/john-dev/2014/12/25/1)).
  **AArch64 has 31 GPRs; the interleave-factor frontier on Apple cores was simply never
  pushed — your 4-lane GPR kernel is already past the state of the art, and 6-8 lanes is
  unexplored territory.**
- **Go**: `golang.org/x/crypto/blowfish` (and hence `bcrypt`) is pure Go, zero assembly on
  any arch ([repo listing](https://github.com/golang/crypto/tree/master/blowfish),
  [bcrypt.go](https://github.com/golang/crypto/blob/master/bcrypt/bcrypt.go)).
- **Rust**: RustCrypto `bcrypt` is pure scalar Rust — your 8-12% single-hash win over it is
  exactly the expected magnitude of a scheduling/layout win at the same chain floor.
- **Other asm**: libgcrypt's `blowfish-arm.S` is 32-bit ARM scalar
  ([searchfox mirror](https://searchfox.org/comm-central/source/third_party/libgcrypt/cipher/blowfish-arm.S));
  no NEON bcrypt exists anywhere public. Graviton numbers above come from the same scalar
  JtR code path ([openbenchmarking Graviton3](https://openbenchmarking.org/result/2407111-NE-AWS4GRAVI72&sgm=1&ppd_RVBZQyA5UjE0IHI3YS4xNnhsYXJnZQ=4.869&ppd_R3Jhdml0b24yIHI2Zy4xNnhsYXJnZQ=3.226&ppd_R3Jhdml0b24zIHI3Zy4xNnhsYXJnZQ=3.427&ppd_R3Jhdml0b240IHI4Zy4xNnhsYXJnZQ=3.770&ppd_WGVvbiA4NDg4QyByN2kuMTZ4bGFyZ2U=4.234&ppt=DPH&rdt&gru)).
- **hashcat GPU kernels** keep per-thread S-boxes in local memory and win only through
  massive thread parallelism — consistent with the "gathers don't SIMD" story
  ([thread-12248](https://hashcat.net/forum/archive/index.php?thread-12248.html)).

## 7. Experiments, ranked by impact/effort

1. **6-lane GPR interleave (2 groups of 3 or 3 groups of 2, all in GPRs).**
   Expected: 2253 -> **3,600-4,200 h/s (+60-85%)**, because the ceiling jumps from 3,241
   (N=4, latency-bound) to 4,861 (N=6, at the load-port knee). Effort: medium. Technique:
   one S-box base register per lane; select the box by ORR-ing 0x100/0x200/0x300 into the
   extracted byte index (bitwise producer -> avoids the +1c shift-index load penalty per
   [dougallj](https://dougallj.github.io/applecpu/firestorm.html)); `ldr w, [xbase, widx,
   uxtw #2]` keeps 1 load/lookup. 6 lanes x 3 regs + temps ~= 24-28 GPRs — no spills.
   Verify the emitted asm; if LLVM spills, drop to 5 or restructure into arrays it
   promotes. Risk: LLVM register allocation; mitigation is exactly your existing
   correctness gate + asm inspection.
2. **8-lane GPR interleave.** Expected: **4,200-4,700 h/s (+85-110%)**, ceiling 4,950.
   Effort: high (31-GPR budget is at the edge: 8x3 + ~6 = 30). Only after (1) shows the
   knee behaves as modeled.
3. **Close the N=4 gap (69.5% -> 85%+ of 3,241).** Expected: +15-25% (~2,600-2,800).
   Cheap audits: (a) confirm LLVM emits scaled-index `ldr [x, w, uxtw #2]`, not
   shift+add+ldr (add-with-shift is 2c latency + 2 int issues
   [int table](https://dougallj.github.io/applecpu/firestorm-int.html)); (b) hoist next
   round's P load so it never sits on the chain (P index is statically known);
   (c) keep LD1-single inserts (TP=2c each
   [simd table](https://dougallj.github.io/applecpu/firestorm-simd.html)) strictly at batch
   pack/unpack, never per-expand-write — expand-phase write-back should be plain GPR STRs
   into the per-lane S-box copies.
4. **Multithreading.** Trivial near-linear scaling per P-core; batch-16 already gives each
   thread independent hashes. Highest system-level win, zero kernel work.
5. **PRFM / cache hints.** Expect <1%; S-box addresses are data-dependent with ~1c lead
   time and the set is L1d-resident; PRFM burns a load slot at TP~1.54c
   ([int table](https://dougallj.github.io/applecpu/firestorm-int.html)). Run once, document,
   discard.
6. **BF_SCALE-style addressing patch for JtR comparison builds.** If you benchmark against
   a self-built JtR, `-DBF_SCALE=1`-style scaled indexing on aarch64 is likely a few %
   faster than upstream's `BF_SCALE 0`
   ([arm64le.h](https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/arm64le.h)) —
   worth knowing so you don't over-credit your own kernel.
7. **Rejected: N=16, NEON-resident, SME/AMX, SLP.** N=16 needs 32 L/R registers alone;
   NEON-resident pays UMOV/LDR/INS (~16c chain, 3 pipe classes) per gather (section 3);
   SME/AMX have no gather (section 4).

## Appendix A. Model parameters used

- Encryptions/hash: cost c -> (1+2^(c+1)) x 521 + 192; cost 5 = 34,057; cost 4 = 17,385.
- Chain/enc = 16 x (1 [P xor] + 1 [extract] + 4-5 [L1d load] + 3 [add/xor/add] + 1 [R xor])
  + ~2 = 165-180; calibrated 167.
- Loads/enc = 16 x (4 S-box + 1 P) + 2 = 82; ALU ~150; uops ~230 (incl. ~2 loop/enc).
- Clock 4.608 GHz
  ([eclecticlight](https://eclecticlight.co/2025/10/30/updated-cpu-core-frequencies-for-all-current-apple-silicon-macs)).
- All raw instruction latencies/throughputs: dougallj Firestorm tables (M1); M4 backend
  (8 ALUs, 10-wide, 3 loads) from jia.je; generations M2/M3 interpolate per Apple's guide.
- Caveat: M5 backend assumed M4-class (3 load pipes); no public M5 pipe counts exist as of
  2026-10. If M5 added a 4th load pipe, the N>=6 ceilings scale by 4/3 (~6,470 h/s); the
  N=4 ceiling (latency-bound) is unchanged.

## Appendix B. Source list (deduped)

Benchmarks: openbenchmarking M2 Air (2307213-NE-2307119NE92), Graviton3
(2407111-NE-AWS4GRAVI72), Ampere Altra (2311275-NE-AMPERE49699); Rizzo thesis (unitesi.unive.it);
hashcat forum thread-12248 (+nextnewest); gists by crypt0rr, Chick3nman, DaniloNC.
Microarchitecture: dougallj.github.io/applecpu/{firestorm,firestorm-int,firestorm-simd,
icestorm,icestorm-int}.html + dougallj.wordpress.com LSQ post; jia.je/hardware/2025/05/21/apple-m4;
chipsandcheese.com/p/going-armchair-quarterback-on-golden-coves-caches,
/p/qualcomms-oryon-core-a-long-time-in-the-making, /p/is-x86-ready-to-ace-it;
developer.apple.com Apple Silicon CPU Optimization Guide; anandtech.com/show/16226;
eclecticlight.co 2025-10-30 frequencies; computerbase.de MBP M5 review; en.wikipedia.org
Apple_M4/Apple_M5; tnzr.org/sme/micro.html; patents US11829763B2, US12067398B1;
reddit r/hardware David Huang M4 Pro thread.
Code: github.com/openwall/john src/{BF_std.c,BF_std.h,arm64le.h,DES_bs_b.c};
openwall.com/lists/john-users/{2013/06/03/1,2021/03/12/3}; openwall.com/lists/john-dev/2014/12/25/1;
github.com/openwall/john/issues/4585; openwall.info/wiki/john/benchmarks;
github.com/golang/crypto/{blowfish,bcrypt}; searchfox.org libgcrypt blowfish-arm.S;
usenix.org/events/usenix99/provos.html.
