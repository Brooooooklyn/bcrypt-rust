# Gather-heavy bcrypt SIMD on x86: microarchitecture ceiling analysis

**Question:** for a bcrypt SIMD kernel whose hot loop is 4 dependent 32-bit S-box gathers per F() call (16-round Feistel, strictly sequential per hash, N independent hashes interleaved in lanes), what do the microarchitecture facts say about the achievable ceiling on AMD Zen 3/4/5 and Intel Skylake-X -> Sapphire Rapids, and what code shapes reach it?

**Baseline under test** (pure Rust `bcrypt-rust`, Zen 4 EPYC 4-vCPU Cloudflare VM, cost 5, batch 16, correctness-gated): scalar 564 h/s; SSE4.1 4-lane 1.12x; AVX2 8-lane 1.54x; AVX-512 16-lane 1.42x. AVX-512 loses to AVX2. Two lookup flavors chosen by runtime shootout: hardware gather (`vpgatherdd`) vs insert (store vector to stack, N scalar loads, rebuild). SoA S-box layout: lane `l` of entry `e` at u32 offset `e*N+l`, so AVX2 working set 32 KiB, AVX-512 64 KiB, vs 32 KiB L1d on Zen 4.

Date: 2026-10-02. All quantitative claims cited; sources in section 14. Measurement caveats in section 13 — read them before quoting numbers.

---

## 0. TL;DR

* `vpgatherdd` is microcoded and slow on every AMD Zen (8-lane ymm: ~8 cycles / 39-42 uops on Zen 3/4, ~5.7-7 cycles / 32 uops on Zen 5) and hardware-fast on Intel (ymm 5c Skylake-X/Ice Lake, **3c on Sapphire/Emerald Rapids**; zmm 9c -> 9c -> **6c**). The gather-vs-insert crossover is a per-uarch, per-width decision — the runtime shootout is the correct design; GCC and LLVM both get this choice wrong by default on Zen.
* On Zen 4 the 16-lane AVX-512 kernel cannot win: zmm gathers have *zero* per-element throughput gain over 2xymm (16.5c vs 2x8c, 81 vs 84 uops), the 512-bit store path is half-rate, and — decisive for bcrypt — 16 lanes x 4 KiB of per-hash S-boxes = 64 KiB > 32 KiB L1d, so ~half of all gather lane-loads miss to L2 (+~10 cycles each). 4 KiB of mutable S-box state per concurrent hash is an irreducible floor: L1d caps useful lane-interleave at ~8 hashes (32 KiB) on Zen 3/4 and ~12 on 48-KiB cores (Zen 5, Ice Lake, Sapphire Rapids). John the Ripper hit exactly this wall in 2013 and abandoned AVX2 bcrypt.
* Verdict on the baseline: AVX2 1.54x is **not at the hardware ceiling but is within the same ballpark** — realistic headroom inside the current 8-lane shape is tens of percent (est. +15-35% via a third lookup flavor + scheduling), not a multiple. The remaining big structural lever on AMD is John-the-Ripper-style **scalar X3/X4 interleave** (3-4 independent scalar hashes in GPRs), which JtR measured *beating* 8-lane AVX2 gather by ~65% on Haswell; on Zen 4 with a 42-uop gather it plausibly beats AVX2-insert too. On Sapphire Rapids the answer flips: zmm hardware gather (6c measured) is the best shape and AVX-512 should pull clearly ahead of AVX2.
* `vpternlogd` saves **zero** instructions in this round — verified op-by-op against the real kernel source (section 8). The F() chain is add -> xor -> add and every XOR/AND is adjacent to a shift or an add, never to another bitwise op. Do not spend effort here.
* Zen 5: gathers ~30% faster, 4 scalar loads/cycle, 2-cycle GPR store-forward, 48 KiB L1d — the insert flavor improves most; AVX2-8-lane (32 KiB, fits) likely still beats AVX-512-16-lane (64 KiB, still overflows). Expect ~1.3-1.6x per-clock over Zen 4 for the insert path; keep the shootout (mobile Zen 5 / Zen 5c are double-pumped 256-bit and differ from desktop/Turin).

---

## 1. The kernel under analysis (facts from the codebase)

Verified in `/Users/brooklyn/workspace/github/bcrypt-rust/src/eks/scalar.rs`:

* F(x) = `((S0[x>>24] + S1[(x>>16)&0xff]) ^ S2[(x>>8)&0xff]) + S3[x&0xff]`, all adds wrapping (`f()`, lines 33-42). The four lookups depend only on `x` — they are **independent of each other**; only the add/xor/add combine is serial.
* `encipher()` (lines 50-60): 16 rounds of `l ^= P[i]; r ^= F(l); swap`, then final swap-undo and `r ^= P[16]; l ^= P[17]`. Round i+1 depends on round i: strictly serial per hash.
* Key expansion (`expand_state`/`encrypt_zero_chain`) runs 521 sequential encryptions that **rewrite P and all four S-boxes** — the S-boxes are *mutable state being written while later encryptions read them*. At cost 5 there are ~34.4k encryptions/hash (~550k F-calls).

SIMD backends (`src/eks/avx2.rs`, `src/eks/avx512.rs`): N hashes in lockstep lanes, SoA S-boxes (entry = one 32B/64B block holding all lanes), gather flavor = one `vpgatherdd` per lookup (`lookup8_gather`/`lookup16_gather`), insert flavor = store idx vector to stack + N scalar loads + rebuild (`lookup8_insert`, lines 130-155). One-time runtime shootout picks the flavor per CPU, with outputs asserted identical. AVX2 file header already documents: gather "on AMD Zen 2-4 is microcoded and loses ~2-3x to scalar loads"; insert "on Intel runs ~10-20% behind the hardware gather".

Working-set arithmetic (irreducible): 4 S-boxes x 256 x u32 = **4 KiB per concurrent hash** (+72 B P). 8 lanes = 32 KiB, 16 lanes = 64 KiB.

---

## 2. VPGATHERDD per uarch: uops, throughput, latency

All throughput figures are best-case (L1-resident, all lanes same cache line for uops.info; Agner assumes L1 hits). Real bcrypt gathers touch 8-16 distinct lines per instruction and will be *slower*. Zen 5 numbers are desktop Granite Ridge (mobile Zen 5 is a different, 256-bit, implementation).

### Table A — throughput (reciprocal, cycles/instr) and uop counts

| uarch | ymm (VEX) | ymm {k} (EVEX) | zmm {k} | per-element, zmm | source |
|---|---|---|---|---|---|
| AMD Zen 3 | **8.0c / 39 uops** (Agner: 9c/39) | n/a (no AVX-512) | n/a | n/a | uops.info [S1], Agner [S10] |
| AMD Zen 4 | **8.0c / 42 uops** | 8.6-9.0c / 41 uops | **16.5-16.6c / 81 uops** | 1.03c/el | uops.info [S2][S3][S4], Agner [S10] |
| AMD Zen 5 | **5.7c / 32 uops** (Agner: 7c/32) | 5.7c / 32 uops | **12.6c / 64 uops** (Agner: 13c/64) | 0.79c/el | uops.info [S5][S6][S7], Agner [S10] |
| Intel Skylake-X | **5.0c / 5 uops** | 5.0c / 4 uops | **9.25c / 4 uops** | 0.58c/el | uops.info [S8], Agner [S10] |
| Intel Ice Lake | 5.0c / 5 uops | 5.0c / 4 uops | **9.0c / 4 uops** | 0.56c/el | uops.info [S9], Agner [S10] |
| Intel Sapphire/Emerald Rapids (Golden/Raptor Cove) | **3.04c / 7 uops** | 3.02c / 6 uops | **6.0c / 6 uops** | **0.375c/el** | uops.info EMR [S11][S12][S13] (measured on Emerald Rapids; SPR core is the same Golden Cove family) |

Cross-check on SPR: an SC'24 workshop paper cites Intel's official SPR throughput/latency document giving zmm gather ~3c reciprocal throughput / ~20c latency [S27] — more optimistic than the 6c uops.info measures on EMR; treat 3-6c as the SPR zmm range, latency ~20-26c.

Latency (uops.info html-lat pages, L1-hit best case): Zen 4 ymm dest-to-dest 8c, index->dest <=21c, mask->dest 19c [S3-lat]; Zen 4 zmm dest 17c, mask 24c [S4-lat]; Zen 5 zmm ~19-27c [S7-lat]; EMR zmm ~6-26c depending on operand path [S13-lat]. Gathers complete when the *slowest lane* completes, so L2-resident lanes dominate real latency.

Agner's rows used (from instruction_tables.pdf, 2025-09-20): Zen 3 x=23ops/5c, y=39ops/9c; Zen 4 x=24/5, y=42/8, y{k}=41/9, z{k}=81/17; Zen 5 x=20/4, y=32/7, y{k}=32/6, z{k}=64/13; Skylake x=4uops/4c, y=4/5; Skylake-X z{k}=4/9; Cannon/Ice Lake x=4/3, y=4/5, z=4/9. Agner's tables stop at Ice Lake/Tiger Lake for big Intel cores — no Golden Cove/Sapphire Rapids chapter — hence EMR measurements above.

### Key reading of Table A

* AMD gathers cost ~**1 cycle per element** (Zen 3/4) or ~0.8 (Zen 5); Intel hardware gathers cost 0.56-0.63c/el on SKX/ICL and 0.375c/el on SPR/EMR. Same instruction, ~2.7x per-element gap between Zen 4 and SPR.
* On Zen 4, 1 zmm gather == 2 ymm gathers in both time (16.5 vs 16c) and uops (81 vs 84): zmm buys nothing but frontend slots.
* Zen 5 improves gathers ~30-40% over Zen 4 but they remain microcoded (32/64 uops) — AMD's own Zen 5 Software Optimization Guide still recommends avoiding GATHER when indices are known, and GCC keeps gathers disabled by default for znver5 [S20].

---

## 3. Load/store unit and cache facts per uarch

| uarch | scalar (GPR) loads/c | vector loads/c | stores/c | L1d | L2 | store->load forwarding | source |
|---|---|---|---|---|---|---|---|
| Zen 3 | 3 mem ops/c total; 2x256b loads + 1 store (essentially unchanged from Zen 2) | 2x256b | 1x256b | 32 KiB 8-way, 4c | 512 KiB, 12c | exact-match ~free (2 IPC chain) | C&C [S13b], Agner [S11b] |
| Zen 4 | **3 reads/c** (<=64-bit, incl. complex addressing); mixed max 2 reads + 1 write | 2x256b **or 1x512b** | 2 GPR or 1x256b; **512b store = 1 per 2c** (and 2 store-queue entries) | 32 KiB 8-way, 4c | 1 MiB, **14c**; L3 ~47c | exact-match GPR 8c combined / 9c vector; contained-in-larger-store 6-7c; partial-overlap fail ~19c | Agner microarch 24.9/24.16/24.17 [S11b], C&C Zen 4 part 2 [S13b] |
| Zen 5 (desktop/server) | **4 reads/c**; total <=4 mem ops/c | **2x512b** | 2x256b (512b store = 2 uops, 2 SQ entries) | **48 KiB 12-way, 4c** | 1 MiB 16-way, 14c; L3 ~55c | GPR exact-match **2c**; vector 9c; 512b 11c | Agner microarch 25.8/25.16/25.17 [S11b], C&C Zen 5 [S14] |
| Skylake-X | 2 loads/c any width | 2x512b | 1x512b | 32 KiB 8-way, 4c | 1 MiB, ~14c | exact match ~free; contained ~5-7c | Agner [S10][S11b] |
| Ice Lake | 2 loads/c | 2x512b | 2x256b (1x512b) | 48 KiB 12-way, 5c | 1.25 MiB, ~14c | similar | Agner [S10][S11b] |
| Sapphire/Emerald Rapids (Golden Cove) | **3 loads/c** (max 2x512b) | **2x512b** | 2/c | 48 KiB 12-way, **5c** | 2 MiB, ~15-16c; L3 ~65c-class | exact match ~free | Agner ADL table 13.1 [S11b], Anandtech [S17], C&C SPR [S16], C&C Zen 4 part 2 (Golden Cove LSU) [S13b] |

False dependence worth knowing on both Zen 4 and Zen 5: "there is a false dependence when the address of a memory read is spaced a multiple of 1024 bytes from a preceding write" (Agner microarch 24.16, 25.16 [S11b]). During key expansion the kernel continuously writes S-box entries while F()-lookups read random entries in the same 32-64 KiB region, so some fraction of lookups eat this re-check delay. No layout change eliminates it (writes sweep the whole region); listed for awareness when profiling, not as an actionable fix.

Zen 4 frontend fact that bounds the gather flavor: the uop cache delivers at most **6 uops/cycle** sustained (Agner 24.5 [S11b]). One 8-lane F() with hardware gathers = 4 x 42 = **168 uops ~= 28 cycles of pure frontend/uop-cache bandwidth** — before a single ALU op. The gather flavor on Zen 4 is frontend/uop-bound as much as LSU-bound.

---

## 4. Q1 — when does a gather beat N scalar loads + rebuild? The crossover

Cost model per 8-lane S-box lookup (L1-resident):

* gather: `TP_gather` cycles, `U_gather` uops (Table A).
* insert (current kernel): 1 vector store + 8 idx loads + 8 box loads = 16 GPR loads + ~7-9 rebuild uops (vpinsrd chains or 8 stores + 1 vector load). On Zen 4: 16 loads @ 3/c ~= 5.3c of LSU time, plus a ~6-8c store-forward link in the dependency chain (Agner 24.17; C&C measured 6-7c for contained-in-larger-store [S11b][S13b]).
* extract (NOT in the kernel — see recommendations): 8 vpextrd (no memory round-trip, ~3-4c cross-domain latency) + 8 box loads = 8 loads @ 3/c ~= 2.7c LSU + FP-port uops.

Rule of thumb: gather wins when `TP_gather < N / loads_per_cycle + rebuild_overhead` *and* the uop count does not make the frontend the bottleneck.

Per uarch verdict for bcrypt's 4-lookups-per-F (per lane-set):

| uarch | gather 8-lane (4 gathers) | insert 8-lane (64 loads + rebuild) | winner |
|---|---|---|---|
| Zen 3 | 32c, 156 uops (~26c frontend alone) | ~21-27c LSU-bound | **insert** (2-3x, matches kernel comment) |
| Zen 4 | 32c, 168 uops (~28c frontend) | ~21-27c LSU-bound | **insert** |
| Zen 5 | ~23c, 128 uops | 64 loads @ 4/c = 16c + 2c store-fwd | **insert** (narrower) |
| Skylake-X | 20c, 20 uops | 64 loads @ 2/c = 32c + rebuild | **gather** (~1.5x) |
| Ice Lake | 20c, 20 uops | 64 loads @ 2/c = 32c | **gather** |
| SPR/EMR | **12c**, 28 uops | 64 loads @ 3/c = 21c | **gather** (~1.7x) |

For 16 lanes (zmm): SKX 37c vs insert 64c -> gather; ICL 36c vs 64c -> gather; **SPR/EMR 24c vs insert ~43c -> gather decisively**; Zen 4 66c vs insert 43c (and 64 KiB footprint) -> **insert, and 2xymm beats 1xzmm** (next section); Zen 5 ~50c vs insert ~32c @ 4/c -> insert/ymm still.

This exactly reproduces the kernel's measured behavior (shootout picks insert on Zen, gather on Intel with insert ~10-20% behind) and the industry default: GCC disabled gather/scatter auto-vectorization for znver4 (commit `7790d4b2`, tuning `X86_TUNE_USE_GATHER` off; bug PR108346, PR116582) and later znver5 [S19][S20]; LLVM's cost model instead treats all AVX-512 CPUs as "fast gather" (`getGatherOverhead() == 2`), which miscompiles on Zen 4 — e.g. SPEC 481.wrf regression, llvm-project issue #137213 [S21]. A runtime shootout is empirically the only reliable selector.

---

## 5. Q2 — Zen 4 AVX-512 double-pumping, and does 2xymm ever beat 1xzmm?

Facts (Agner microarch 24.9/24.10 [S11b]; C&C Zen 4 part 1 [S12]; AMD Family 19h SOG as quoted by C&C):

* 512-bit instruction = **one uop** through the frontend/ROB/scheduler (saves rename/ROB bandwidth vs 2 uops), then occupies **two 256-bit pipes for two consecutive cycles** ("Because the data paths are 256 bits wide, the scheduler uses two consecutive cycles to issue a 512-bit operation" — AMD SOG). Net ALU throughput for 512-bit = half the 256-bit rate; latency ~same (+~1c for cross-half ops).
* Loads: per cycle the L1d sustains either 2x256-bit reads **or one 512-bit read** — equal bytes/cycle (64 B/c); a 64B-crossing access costs extra.
* Stores: 512-bit store = half the 256-bit store rate (one per 2 cycles) **and two entries in the small 64-entry store queue** (C&C part 2 [S13b]).
* Exception: a dedicated full-width 512-bit shuffle/permute unit — permutes are the one zmm op that is *not* double-pumped (C&C part 1 [S12]).
* Gathers/scatters are microcoded scalar-load sequences on Zen; zmm gather = 81 uops / 16.5c == 2x (ymm 42 uops / 8c). "Gather/Scatter slow on AMD's Zen4 implementation — probably owing to its weaker load/store unit" (Mysticial's Zen 4 AVX-512 teardown [S18]).

So does 2xymm beat 1xzmm for load-bound kernels on Zen 4?

* Pure load streams: tie on bytes (64 B/c either way); zmm wins on uop count. -> zmm >= 2ymm.
* Store streams: **2xymm wins 2x per byte** (2x256b stores/c vs one 512b store per 2c). The bcrypt expansion phase does 2 vector S-box stores per encipher, so this directly taxes the 16-lane kernel.
* Gathers: exact tie per element (Table A). -> no zmm gain, ever, on Zen 4.
* Cache footprint: 16 lanes SoA = 64 KiB > 32 KiB L1d -> ~50% miss (section 9). **This is the decisive term for bcrypt**, on top of the store-rate penalty.

Conclusion: on Zen 4, for this kernel class, 2xymm >= 1xzmm in every component and strictly better in stores and cache residency. The measured 1.42x (AVX-512) vs 1.54x (AVX2) is fully explained; it is not an implementation bug.

---

## 6. Q3 — Zen 5 (Granite Ridge / Turin)

* Full 512-bit datapath on desktop/server (all vector pipes 512b; stores still 256b-wide, 512b store = 2 uops, 2 SQ entries); mobile Strix Point and Zen 5c stay double-pumped 256-bit (Agner 25.1/25.9 [S11b]; C&C Zen 5 desktop [S14], Strix Point [S15]).
* LSU roughly doubles: 4 scalar loads/c, 2x512b vector loads/c, 4 mem ops total (Agner 25.8 [S11b]); L1d grows to 48 KiB 12-way, still 4c; L2 still 14c; GPR store-forward drops to 2c (Agner 25.16/25.17 [S11b]).
* Gathers: still microcoded, but cheaper — ymm 32 uops, measured TP ~5.7c (uops.info [S5]; Agner says 7c), zmm 64 uops, ~12.6c [S7]. Per-element: 0.71c (ymm) / 0.79c (zmm) vs Zen 4's 1.0/1.03c.
* No published bcrypt-on-Zen 5 SIMD numbers exist as of writing; the most relevant measured gather-heavy datapoints are the uops.info/Agner figures above, and the fact that AMD's Zen 5 SOG still says to avoid GATHER for known indices and GCC keeps `use_gather` off for znver5 [S20].

Expected bcrypt behavior on Zen 5 (model, not measurement):

* Insert flavor benefits most: 64 loads/F @ 4/c = 16c (vs 21.3c) + store-forward 2c (vs 6-8c) -> AVX2-insert should gain ~1.3-1.6x per clock over Zen 4.
* 8-lane SoA (32 KiB) now fits with 16 KiB of headroom in the 48 KiB L1d; 16-lane (64 KiB) still overflows (~25% miss). Combined with zmm gathers still being per-element *worse* than ymm (0.79 vs 0.71c/el), **AVX2-8-lane likely remains the peak shape on Zen 5**, with AVX-512 closer than on Zen 4 but not ahead. Scalar-X4 interleave (section 10) also gets stronger (4 loads/c, 6 ALUs).
* On mobile Zen 5 / Zen 5c / Turin Dense, zmm is double-pumped again — the runtime shootout (and the `BCRYPT_REQUIRE_BACKEND` override) must decide per part.

---

## 7. Q4 — Sapphire Rapids / Emerald Rapids

* Golden Cove family: L1d 48 KiB (5c), L2 2 MiB (~15-16c), 3 load AGUs, 3 loads/c up to 2x512b, AVX-512 fully enabled (Agner ADL ch.13 [S11b]; Anandtech Golden Cove deep-dive [S17]; C&C "A Peek at Sapphire Rapids" [S16]).
* Gathers are the best in the industry here: measured on EMR — ymm 3.04c/7 uops, zmm 6.0c/6 uops (uops.info [S11][S12][S13]); Intel's official SPR document claims ~3c zmm TP / ~20c latency (via [S27]); latency measurements cluster ~20-26c. Emerald Rapids is the same core with more L3; Skylake-X/Cascade Lake are the previous generation (ymm 5c, zmm 9-9.25c).
* Crossover math (section 4): zmm gather (24c per 16-lane F) beats zmm-insert (~43c) by ~1.8x; ymm gather (12c per 8-lane F) matches zmm per-element. **On SPR, AVX-512 with the hardware gather is the right shape and should pull clearly ahead of AVX2** — the only x86 uarch class where that is true for a 64 KiB-footprint lookup kernel. The 64 KiB SoA still exceeds the 48 KiB L1d (~25% miss to a fast 2 MiB L2), which is why ymm-gather is a credible fallback; the shootout should arbitrate.
* Not verified in this research: current license-based frequency behavior for 512-bit *integer* code on SPR (much reduced vs the Skylake-X era, but confirm on the target machine).

---

## 8. Q5 — vpternlogd: nothing to fuse (verified against the real kernel)

`vpternlogd` computes an arbitrary 3-input *bitwise* function in one instruction — it can collapse a tree of two 2-input boolean ops. The bcrypt round's op sequence (scalar.rs `f()` + `encipher()`, and the lane-wise copies in avx2.rs/avx512.rs):

1. `l ^= p` (XOR) — next op on `l` is a **shift** (extraction); no boolean-tree to fuse.
2. extraction: `srli 24`; `srli 16 + and 0xff`; `srli 8 + and 0xff`; `and 0xff` — every AND follows a **shift**; ANDs of *different* shifts cannot share a ternlog (no shift inside ternlog).
3. F combine: `(s0 + s1) ^ s2` then `+ s3` — the XOR's left input is an **add** (carries; not bitwise), and its result feeds an **add**. Not fusable.
4. `r ^= F(l)` — F ends in the `+ s3` add, so this XOR is adjacent to an add on one side and the next round's `l ^= p`... on the *other* variable. Confirmed: the round-final XOR is fusable with nothing.

The only boolean identity available anywhere is in the gather-flavor index math: `(idx << 3) + lane_off` == `(idx << 3) | lane_off` (no carry overlap, since `lane_off < 8`) — but OR costs the same 1 op as ADD, and a ternlog with a don't-care input saves nothing. **Expected vpternlogd win in this kernel: 0 instructions. Do not spend effort.** (Instruction semantics: Intel SDM / felixcloutier reference; uarch cost 1c lat, 0.5c TP on Zen 4/Intel — irrelevant here.)

Adjacent AVX-512 features that *would* matter (masking, 32 registers) are already largely captured by the existing kernel; the extraction chain (3 shifts + 3 ANDs per F) is irreducible without gathers-with-byte-indices, which do not exist.

---

## 9. Q6 — L1d miss cost and the 64 KiB vs 32 KiB working set

* Zen 4: L1d hit 4c; L1 miss / L2 hit = **14c total** (+~10c over hit); L3 ~47c (Agner table 24.1 [S11b]; C&C Zen 4 part 2 measured the same 4/14 [S13b]). Golden Cove: L1d 5c, L2 ~15-16c, L3 ~65c (Agner ADL table 13.1 [S11b], C&C [S16]). Zen 5: L1d 4c, L2 14c, L3 ~55c (Agner table 25.1 [S11b]).
* The L1 fill path competes with demand loads: AMD's L1d "doesn't have enough ports to handle a fill request from L2 and deliver full bandwidth to the core at the same time" (C&C part 2, citing travisdowns' L2-bandwidth study [S13b][S29]); Golden Cove has a wider 64 B/c L1<->L2 path vs Zen 4's ~32 B/c [S13b].
* Effect for the 16-lane SoA (64 KiB) on a 32 KiB 8-way L1d with uniform-random 64B-entry accesses: ~50% miss. Gathers complete at the *slowest lane*, so nearly every 16-lane gather pays L2 latency on some lane (~14c+) instead of L1 (4c), and every miss also spends fill bandwidth. On the serial Feistel chain this directly inflates the per-F latency floor; on the throughput side it halves effective LSU service rate for the lookup stream.
* Published measurements of exactly this cache-resident-vs-L2 effect for lookup-heavy kernels: (a) John the Ripper's 2013 experiment — packing 8 AVX2 bcrypt instances (32+ KiB of S-boxes) into L1d "easily exceeds 32 KB L1D... causes L2 hits" and lost to fewer instances [S23]; (b) Polychroniou/Raghavan/Ross (SIGMOD 2015): hardware gather throughput is bounded by the cache's 1-2 accesses/cycle and "depends on the number of distinct cache lines accessed" — i.e., a gather is internally a sequence of scalar loads with the same cache-residency behavior [S22]; (c) chipsandcheese pointer-chase latency curves showing the 4c->14c step at 32 KiB on Zen 4 [S13b]. No one has published a bcrypt-specific L2-resident gather curve; the ~50%-miss estimate above is a model from capacity, not a measurement.
* Note the L1d-residency wall is *per-core concurrency*, not per-SIMD-width: N concurrent hashes need N x 4 KiB. Any shape — zmm, 2xymm groups, or scalar-X8 — that runs 16 hashes per core on a 32 KiB L1d pays the same L2 tax. This is why JtR's answer (section 10) caps at X3 (~12 KiB) and why "just add another ymm group" is an anti-pattern on Zen 4.

---

## 10. Q7 — best-known code shapes for 32-bit-lookup-heavy SIMD kernels

1. **John the Ripper's scalar interleave (the reference shape for exactly this kernel).** JtR tried AVX2 bcrypt with `vpgatherdd` on Haswell and *reverted to scalar*: 8-way AVX2 gather ran ~4k c/s vs ~6.6k c/s for 2-3-way interleaved scalar on a quad-core Haswell [S24]. Their findings map 1:1 onto this project: (a) the real bottleneck is generating 4 indices + effective addresses fast enough to keep the L1 read ports busy (~60% utilization even in good scalar code) [S24]; (b) 8 instances x ~4 KiB S-boxes blow the 32 KiB L1d [S23]; (c) SSE4.1 "manual gather" (extracts + scalar loads) performed about the same as `vpgatherdd` [S25]. JtR's shipping shape is **2-3 independent scalar hashes interleaved in GPRs** (`BF_std` X2/X3): enough dependency chains to hide the 4-5c L1 latency, small enough footprint (12 KiB) to stay L1-resident, zero vector-rebuild cost. Our scalar backend currently runs hashes strictly one at a time — it leaves this entire trick on the table.
2. **The database community's rule:** gathers are scalar loads in a trench coat. Polychroniou/Raghavan/Ross ("Rethinking SIMD Vectorization for In-Memory Databases", SIGMOD 2015) model gather throughput as bounded by the cache's 1-2 accesses/cycle, proportional to distinct lines touched, and show permute-based emulation comes within ~13% of hardware gathers for hash-table probing on Haswell [S22]. Vectorized hash tables in production (ClickHouse, abseil/hashbrown-style SwissTables) therefore probe via contiguous SIMD *metadata* loads + scalar element loads, never gathers; gathers are used only when indices are irreducibly random — which bcrypt's are.
3. **Compiler community consensus:** both GCC and LLVM have been burned by auto-selecting gathers on Zen — GCC disables them by default on znver4/znver5 (commit `7790d4b2`; PR108346, PR116582; Zen 5 tuning also disabled, with AMD's SOG recommending against GATHER for known indices) [S19][S20]; LLVM has the opposite bug (treats AVX-512 as fast-gather; issue #137213, SPEC wrf regression on Zen 4) [S21]. **A runtime shootout — what this kernel already does — is the validated best practice.**
4. **Prior bcrypt-specific SIMD:** `pbcrypt` (2019) got ~175% vs naive OpenBSD C with 8-way AVX2 `vpgatherdd` [S26] — against an *uninterleaved* scalar baseline, consistent with our 1.54x against an uninterleaved scalar baseline, and consistent with JtR's conclusion that interleaved scalar is the stronger baseline.
5. **The missing third flavor (extract):** `vpextrd` lane -> GPR -> scalar box load, then `vpinsrd` rebuild. Versus the current insert flavor it removes 8 idx loads + the 6-8c store-forward link per lookup at the cost of cross-domain extract/insert uops; versus gather on Zen it removes the 42-uop microcode. This is precisely JtR's "manual gather" [S25], updated: on Zen 4 it should beat both current flavors because the box loads start ~4c earlier (no memory round-trip for indices) and LSU pressure halves (8 loads vs 16 per lookup).

## 11. Verdict — are the Zen 4 numbers (1.54x / 1.42x) at the ceiling?

Cycle accounting (ratios; absolute clock of the VM is unknown and irrelevant):

* Scalar: 564 h/s ~= 8-12 cycles per F-call depending on actual clock (2.5-3.7 GHz). Matches a latency-bound chain: 4 independent L1 loads (4c, overlapped) + add/xor/add (3c) + extraction + round overhead. Load ports run at ~15-20% utilization — scalar is **latency-bound, not LSU-bound**.
* AVX2 1.54x -> one 8-lane vector-F costs ~5.2 scalar-F equivalents (~42-62c). Floors for the insert shape on Zen 4: LSU-only floor = 64 loads @ 3/c ~= 21c; realistic floor with the store-forward link, rebuild uops and round-to-round serialization ~= 26-35c. **The kernel is at roughly 50-75% of its own shape's realistic floor.**
* AVX-512 1.42x < AVX2 1.54x is *explained, not buggy*: zmm gather = 2x ymm gather cost with zero per-element gain (Table A), half-rate 512-bit stores in the expansion phase, and the 64 KiB > 32 KiB L1d residency wall (sections 5, 9).

Verdict: **1.54x is near the ceiling of the *8-lane store+reload shape* but not of the hardware.** Realistic remaining headroom on Zen 4, in decreasing confidence: extract flavor +10-20%; scheduling/uop-hygiene +5-15%; scalar-X3/X4 interleave = unknown-but-plausibly-larger (JtR's evidence says this shape beat 8-lane AVX2 outright on Haswell [S24]; on Zen 4's 3-load/cycle LSU it models at parity-to-better vs AVX2-insert — must be measured). There is no path to >2x on Zen 4 without changing the algorithm's memory behavior, because the 32 dependent-latency box loads per 8-hash F-step and the 4 KiB/hash footprint are irreducible.

Expect on other uarchs: Zen 5 ~1.3-1.6x per-clock over Zen 4 for the insert path (4 loads/c, 2c store-forward, 48 KiB L1d) with AVX2-8 still the likely peak; SPR/EMR flips to hardware gather, zmm likely best (6c/16 lanes), AVX-512 clearly ahead of AVX2 (section 7); SKX/ICL favor ymm gather (5c/8 lanes) with zmm throughput-equal but cache-disadvantaged.

## 12. Recommendations (impact / effort / correctness risk)

| # | Change | Expected impact | Effort | Correctness risk |
|---|---|---|---|---|
| R1 | Add **extract flavor** (`vpextrd`+scalar loads+`vpinsrd`) as a third shootout arm on x86 (sections 4, 10.5) | +10-20% on Zen 3/4/5 (removes 32 loads + 4 store-forward links per F); neutral elsewhere | Small: one `lookupN_extract` per backend + shootout arm | Low — shootout already asserts flavor outputs identical; unrolled const lane indices |
| R2 | Add **scalar X3/X4 interleaved backend** (JtR `BF_std` shape; 1 base register + immediate per-instance offsets) | Unknown, plausibly parity to +50% over AVX2-insert on Zen 4/5; JtR measured it beating AVX2 gather ~1.65x on Haswell [S24]; also the best portable fallback | Medium: mechanical interleave of the existing scalar kernel; watch register pressure/spills | Low-medium — pure scalar code gated by the same differential harness; spill regressions are perf-only, caught by bench |
| R3 | **Do not pursue 16-lane interleave on AMD** (and no 2xymm double-group on Zen 4): 4 KiB/hash x N > L1d (sections 9, 11). On 48-KiB-L1d cores (Zen 5, ICL, SPR) a 12-lane ymm variant (48 KiB) is the most that could fit — speculative, low priority | Avoids certain regression | Zero (policy) | None |
| R4 | Keep gather flavor on Intel; on SPR-class expect **zmm gather** to win — verify shootout picks it; benchmark on real SPR/EMR hardware | Up to ~1.8x over insert on SPR (model, section 7) | Zero (shootout) + one benchmark run | None |
| R5 | **Skip vpternlogd** entirely (section 8 — verified zero fusion sites in the round) | 0 | 0 | 0 |
| R6 | For batch/parallel mode, prefer **SMT 2 threads/core** where available: the kernel is latency-bound with idle load ports; classical +20-40% for this workload class [S24 used HT similarly] | +20-40% throughput on SMT hosts | Zero (thread-count config) | None |
| R7 | When profiling on Zen 4/5, account for the 1024-byte read/write **false dependence** (Agner 24.16/25.16 [S11b]) and store->gather ordering during expansion (gathers interact conservatively with in-flight stores [S28][S29]) before attributing stalls | Diagnostic only | Zero | None |
| R8 | Keep the runtime shootout as the *only* selector; GCC/LLVM demonstrably mis-select on Zen (section 4). Extend it to choose per-width (ymm vs zmm) on Zen 5/SPR/mobile-Zen 5 rather than per-backend | Protects against uarch drift (mobile Zen 5 is double-pumped; Turin vs Turin Dense differ) | Small | None |

---

## 13. Measurement caveats

* uops.info gather measurements use `vindex = 0` (all lanes hit the **same cache line** — best case) and a zeroed VEX mask; real bcrypt gathers touch 8-16 distinct lines per instruction and pay per-line costs. Agner's numbers likewise assume L1 hits. Treat Table A as *lower bounds on cost*.
* Zen 5 uops.info timings are APERF-based (actual core cycles under turbo); Agner's Zen 5 recip. throughputs (7c/13c) are ~20% worse than uops.info's (5.7c/12.6c). Both are cited; the spread is the honest range.
* Sapphire Rapids gather pages do not exist on uops.info; **Emerald Rapids** (same Golden Cove-family core, bigger L3 only) is used as the measurement proxy, cross-checked against Intel's official SPR document figures via [S27] (which are more optimistic: ~3c zmm TP).
* Agner's instruction tables end at Ice Lake/Tiger Lake for big Intel cores (no Golden Cove), and his Alder Lake cache table (48 KiB L1d, 15c L2) is used as the Golden Cove proxy for SPR, consistent with Anandtech and C&C.
* Latency values quoted from uops.info html-lat pages are the operand-path measurements (dest/index/mask -> dest), not load-to-use of a single lane; gathers retire when the slowest masked-in lane completes.
* The per-F cycle floors in section 11 are models from documented LSU rates and measured instruction costs, not end-to-end measurements; they exist to size the headroom, not to predict to the cycle.
* The Cloudflare VM's actual clock is unknown; all baseline-derived cycle figures are stated as ranges over 2.5-3.7 GHz and only ratios are used in conclusions.

## 14. Sources

Gather instruction measurements (throughput/uops):
* [S1] https://uops.info/html-tp/ZEN3/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
* [S2] https://uops.info/html-tp/ZEN4/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
* [S3] https://uops.info/html-tp/ZEN4/VPGATHERDD_YMM_K_VSIB_YMM-Measurements.html (latency: https://uops.info/html-lat/ZEN4/VPGATHERDD_YMM_K_VSIB_YMM-Measurements.html ; ymm VEX latency: https://uops.info/html-lat/ZEN4/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html)
* [S4] https://uops.info/html-tp/ZEN4/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html (latency: https://uops.info/html-lat/ZEN4/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html)
* [S5] https://uops.info/html-tp/ZEN5/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
* [S6] https://uops.info/html-tp/ZEN5/VPGATHERDD_YMM_K_VSIB_YMM-Measurements.html
* [S7] https://uops.info/html-tp/ZEN5/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html (latency: https://uops.info/html-lat/ZEN5/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html)
* [S8] https://uops.info/html-tp/SKX/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html and https://uops.info/html-tp/SKX/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html
* [S9] https://uops.info/html-tp/ICL/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html and https://uops.info/html-tp/ICL/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
* [S11] https://uops.info/html-tp/EMR/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
* [S12] https://uops.info/html-tp/EMR/VPGATHERDD_YMM_K_VSIB_YMM-Measurements.html
* [S13] https://uops.info/html-tp/EMR/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html (latency: https://uops.info/html-lat/EMR/VPGATHERDD_ZMM_K_VSIB_ZMM-Measurements.html)

Microarchitecture manuals and deep dives:
* [S10] Agner Fog, Instruction Tables, 2025-09-20: https://www.agner.org/optimize/instruction_tables.pdf (Zen 3/4/5, Skylake/Skylake-X, Cannon/Ice Lake gather rows quoted in section 2)
* [S11b] Agner Fog, The microarchitecture of Intel, AMD and VIA CPUs: https://www.agner.org/optimize/microarchitecture.pdf (Zen 4 ch.24: 24.5 uop cache 6/c, 24.9 LSU rates, 24.10 double-pumping, 24.16 cache table + 1024-byte false dependence, 24.17 store forwarding; Zen 5 ch.25: 25.8 LSU, 25.16 cache table, 25.17 2c GPR store-forward; Alder Lake P ch.13 cache table)
* [S12] Chips and Cheese, "AMD's Zen 4, Part 1: Frontend and Execution Engine": https://chipsandcheese.com/2022/11/05/amds-zen-4-part-1-frontend-and-execution-engine/ (double-pumped 512b, single-uop-until-execution, dedicated 512b shuffle unit, AMD SOG quote)
* [S13b] Chips and Cheese, "AMD's Zen 4, Part 2: Memory Subsystem and Conclusion": https://chipsandcheese.com/2022/11/08/amds-zen-4-part-2-memory-subsystem-and-conclusion/ (L1d 4c/L2 14c, store queue 64 + 2 entries per 512b store, 6-7c contained store-forward, 19c forwarding-fail, Golden Cove 3x256b/2x512b loads + 64B/c L1<->L2, AMD L1 fill vs demand-load port conflict, L2 latency +2c over Zen 3)
* [S14] Chips and Cheese, "AMD's Ryzen 9950X: Zen 5 on Desktop": https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/ (2x512b loads/c desktop vs 1 on Zen 4/mobile, full 512b FP, 512b store = 2 uops/2 SQ entries, SQ 104)
* [S15] Chips and Cheese, "AMD's Strix Point: Zen 5 Hits Mobile": https://chipsandcheese.com/2024/08/10/amds-strix-point-zen-5-hits-mobile/ (mobile 256b datapath; L1d 48 KiB 12-way 4c)
* [S16] Chips and Cheese, "A Peek at Sapphire Rapids": https://chipsandcheese.com/p/a-peek-at-sapphire-rapids (SPR core = Golden Cove, L2 ~16c)
* [S17] Anandtech, Golden Cove deep dive (Intel Architecture Day 2021), archived: https://web.archive.org/web/20210828102737/https://www.anandtech.com/show/16881/a-deep-dive-into-intels-alder-lake-microarchitectures/3 (3rd load AGU, L1d 48 KiB, server L2 2 MiB, AVX-512 enabled on SPR)
* [S18] Mysticial (Alexander Yee), "AVX-512 on AMD Zen 4" teardown (archived mersenneforum thread): https://web.archive.org/web/20230706110515/https://www.mersenneforum.org/showthread.php?t=28102 ("Gather/Scatter slow on AMD's Zen4 implementation — probably owing to its weaker load/store unit"; GCC znver4 gather-disable discussion)

Compilers:
* [S19] GCC commit 7790d4b2 / x86-tune.def, "Disable gather/scatter for zen4" (Jan Hubicka, 2023): https://mirrors.git.embecosm.com/mirrors/gcc/-/commit/7790d4b2e5c6ed0d4957e3b7948e24023447fbfd ; bugs https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108346 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=116582
* [S20] Phoronix, "AMD Zen 5 Tuning Part 2 (GCC)": https://www.phoronix.com/news/AMD-Zen-5-Tuning-Part-2-GCC (znver5 also disables gather/scatter; AMD Zen 5 SOG recommends avoiding GATHER when indices are known in advance)
* [S21] LLVM: issue https://github.com/llvm/llvm-project/issues/137213 (SPEC 481.wrf slower with masked gather on Zen 4); X86TTI gather cost model discussion: https://stackoverflow.com/questions/75845054/intel-vs-amd-gather-avx-performance (`getGatherOverhead()==2` for any AVX-512 CPU; `hasFastGather` Intel-only)

Workload-class evidence (bcrypt and gather-heavy kernels):
* [S22] Polychroniou, Raghavan, Ross, "Rethinking SIMD Vectorization for In-Memory Databases", SIGMOD 2015: https://dl.acm.org/doi/pdf/10.1145/2723372.2747645?download=true (gather = 1-2 cache accesses/cycle, cost proportional to distinct lines; permute-emulation within ~13%)
* [S23] john-dev, 2013-11-01 (Solar Designer): https://openwall.com/lists/john-dev/2013/11/01/2 (8 AVX2 bcrypt instances exceed 32 KiB L1d -> L2 hits; layout experiments lost)
* [S24] john-dev, 2015-06 (Solar Designer): https://marc.info/?l=john-dev&m=143511957528608&w=2 (AVX2-gather bcrypt ~4k c/s vs interleaved scalar ~6.6k c/s on quad Haswell; index/address generation is the real bottleneck; JtR ships Blowfish 32/64 X3)
* [S25] john-dev, 2013: http://marc.info/?l=john-dev&m=138332222012798&w=2 (SSE4.1 manual gather ~= vpgatherdd on Haswell; both lose to scalar interleave)
* [S26] pbcrypt (8-way AVX2 gather bcrypt, 2019): https://github.com/cat-j/pbcrypt (~175% vs unoptimized OpenBSD C)
* [S27] SC'24 workshop paper citing Intel's official SPR instruction throughput/latency document (ID 765484): https://dl.acm.org/doi/pdf/10.1109/SCW63240.2024.00181 (zmm gather ~3c recip TP / ~20c latency, official figures)
* [S28] Intel Optimization Reference Manual (doc 248966): https://cdrdv2.intel.com/v1/dl/getContent/787036 (gather/scatter guidance: prefer unit-stride loads + shuffles when indices are known; conservative store-forwarding/disambiguation for multi-element accesses)
* [S29] Travis Downs, uarch-bench wiki: https://github.com/travisdowns/uarch-bench/wiki/How-much-bandwidth-does-the-L2-have-to-give,-anyway%3F and https://github.com/travisdowns/uarch-bench/wiki/Memory-Disambiguation-on-Skylake (L1 fill vs demand loads; disambiguation machinery)
