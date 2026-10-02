# CPU-side bcrypt optimization techniques beyond lockstep SIMD — deep research report

Date: 2026-10-02. Scope: John the Ripper (core + jumbo), Openwall crypt_blowfish, hashcat (CPU-relevant),
academia. Question: what exists beyond "N independent hashes in lockstep SIMD lanes"?

Our baselines (bcrypt-rust, pure Rust): scalar; SIMD batch AVX-512 x16 / AVX2 x8 / SSE4.1 x4 / NEON x4 /
wasm128 x4 with SoA S-boxes (lane l of entry e at u32 offset e*N+l; working set N*4 KiB per lane group);
runtime gather-vs-insert shootout on x86; aarch64 JtR-style 4-lane GPR-resident interleave (2233 vs 1073 h/s
on M5 Max vs vector-resident rounds). Measured: M5 Max NEON 2.78x scalar; Zen 4 AVX2 1.54x / AVX-512 1.42x
scalar (AVX-512 loses to AVX2: 64 KiB working set vs 32 KiB L1d + double-pumped zmm).

All throughput figures below are bcrypt cost 5 ("$2a$05", 32 cost-loop iterations) unless noted — the
traditional JtR benchmark setting. "c/s" = candidates/second = hashes/second for one salt.

---

## 1. What JtR/Openwall actually do (ground truth from source)

### 1a. JtR's bcrypt is scalar-interleaved, not SIMD — to this day

JtR's CPU bcrypt format reports as `bcrypt [Blowfish 32/64 X3]` (or X2): pure 32-bit scalar code with
2- or 3-way *instruction interleaving* of independent hashes inside one thread (BF_ENCRYPT2/BF_ENCRYPT3
macros in BF_std.c). There is no SSE/AVX2/AVX-512 bcrypt in JtR.

Evidence:
- BF_std.c source (X2/X3 interleave macros, per-instance BF_ctx arrays):
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.c
- BF_std.h (BF_X2==3 -> "Blowfish 32/64 X3"; BF_Nmin=3; OpenMP multiplies BF_N by threads):
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.h
- Solar Designer, john-users 2020-10-15 (about AWS c5.24xlarge AVX-512 vs c5a AVX2 benchmarks):
  "The Intel benchmark uses AVX-512, the AMD one uses AVX2, except where the corresponding JtR format
  doesn't support SIMD (e.g., bcrypt)".
  https://openwall.com/lists/john-users/2020/10/15/1
- Same thread, 2020-11-11 follow-up: https://openwall.com/lists/john-users/2020/11/11/1

Interleave factor is a compile-time choice (arch.h / BF_X2): X3 on most 64-bit builds, X2 historically for
AVX-enabled builds to avoid regressions on older CPUs (Core 2 era); discussion of X2 vs X3 per CPU:
https://openwall.com/lists/john-dev/2014/12/25/1

### 1b. Why: gathers lost to scalar interleave on every CPU they tried (2013-2014, Haswell/Xeon Phi)

The definitive experiments (Steve Thomas's AVX2 gather code, Solar Designer's hacks), i7-4770K Haswell,
single-thread h/s unless noted:

- AVX2 gather, 8 lanes ("8*256"): 1025 h/s. Scalar single instance ("Normal"): 661 h/s.
  So gather-based AVX2 x8 = 1.55x scalar single — but JtR's X2 scalar interleaved with 8 OpenMP threads:
  **6595 c/s**, i.e. ~824 c/s per thread vs 1025/8... per-thread comparison: JtR X2 interleave alone gives
  ~2x over single scalar (608-661 h/s -> the 6595/8=824 is with HT contention; Solar states JtR achieves
  higher than any AVX2 number "because JtR's scalar code runs two instances with interleaved instructions").
- Staying in L1 didn't fix it: 7 lanes in 28 KiB: 926 h/s (worse per-hash). "I tested for the hypothesis
  that the slowness was primarily due to us slightly exceeding L1 data cache size - and no, this does not
  appear to be the case."
- 8 concurrent processes of the AVX2 build: ~522-524 h/s each (HT/port contention kills it).
- Quote: "Apparently, Haswell's gather loads are just slow; maybe a future CPU will do better."

Sources:
- https://openwall.com/lists/john-dev/2013/11/01/1 (full numbers above)
- Passwords^14 deck, slide 42 (Haswell: 8 lanes/8 threads 4186 c/s; 7-lane 28 KiB 3519 c/s; JtR X2 6595 c/s)
  and slide 40 (Xeon Phi 5110P: scalar OpenMP 6246 c/s vs 512-bit VPU masked gathers 4147 c/s — wider SIMD
  *lost* on MIC too):
  https://www.openwall.com/presentations/Passwords14-Energy-Efficient-Cracking/Passwords14-Energy-Efficient-Cracking.pdf

### 1c. Instruction scheduling and index-extraction tricks (the scalar core)

JtR's actual speed comes from minimizing uops per Blowfish round and hiding S-box load latency with
interleaving, not from SIMD:

- **Round latency structure**: BF_ROUND (BF_std.c) issues all 4 S-box loads (independent, indices from the
  same L) then combines: `tmp3 = S1[b]; tmp3 += S0[a]; tmp3 ^= S2[c]; R ^= P[N+1]; tmp3 += S3[d]; R ^= tmp3`.
  The independent `R ^= P[N+1]` is scheduled in the middle to fill load-latency slots. With BF_SCALE
  (x86/ARM scaled addressing) indices stay unshifted; without it, indices are pre-shifted (`>>6 & 0x3FC`).
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.c
- **x86.S (32-bit asm, 1998-2010)**: Pentium pairing + Pentium Pro partial-register-stall avoidance;
  high-byte register trick (`movb %ch,%al`) to extract byte 2 without a full shift; note: "It is possible
  to do 15% faster on the Pentium Pro family ... but, unfortunately, that would make things twice slower
  for the original Pentium. An additional 2% speedup may be achieved with non-reentrant code."
  https://raw.githubusercontent.com/openwall/crypt_blowfish/master/x86.S
- **BMI (2015 experiment)**: BEXTR with preloaded descriptors (0x0818/0x0810 = 8 bits at offset 24/16)
  extracts two of the four indices in one instruction each; loads folded into ALU ops
  (`addl Sa(1)(,tmp2,4),tmp1`, `xorl Sa(2)(,tmp3,4),tmp1`). Result: 20 instructions per round for two
  interleaved instances = **10 instructions/round/instance** — the best known x86 figure. Not committed:
  "we actually need to optimize uops, and latency hiding", and 64-bit OpenMP builds didn't improve.
- **MMX2 "2x2" (2015 experiment)**: two instances' L values packed in one MMX register; PEXTRW extracts
  indices; S-box words loaded scalar (`movd`) and paired with `punpckldq`; the F() adds/xors then run as
  `paddd`/`pxor` on pairs. 50 instr/round for 4 instances = 12.5/instance. Results: +15% over previous best
  on Pentium 3 (211 -> 244 c/s); single-instance on Haswell 1049 -> 1347 c/s (+28%); but only 6080 -> 6176
  c/s (+1.6%) with 8 concurrent processes on i7-4770K. Not committed (data layout changes, non-universal).
- **Port-bound analysis** (key quote): "if L1 data cache reads were the bottleneck, i7-4770K could
  theoretically do up to ~11.1k c/s, but it only achieves ~6.6k c/s with our current code. That's about
  60%. ... the bottleneck does indeed appear to be different. It appears to be taking too many instructions
  and uops to extract all of the S-box indices and compute the effective addresses to keep the L1 data
  cache read ports 100% busy." P reads are sequential and could be combined into 64-bit loads
  (5 -> 4.5 reads/round -> theoretical 12.3k). https://marc.info/?l=john-dev&m=143511957528608&w=2
  Theoretical-peak model (slide 51): c/s = Nports * f / ((2^cost * 1024 + 585) * Nreads * 16), with
  Nreads = 4-5 reads/Blowfish round. Slides 51-52 of
  https://www.openwall.com/presentations/Passwords14-Energy-Efficient-Cracking/Passwords14-Energy-Efficient-Cracking.pdf
- **Interleave factor vs SMT**: X3 beats X2 significantly on non-SMT Intel (Core 2), but on 2-thread/core
  CPUs it is flat or negative: i7-5820K (12 threads): BF_X2=1 (X2) 8625-8928 c/s vs BF_X2=3 (X3)
  8208-8424 c/s; on 2x E5-2670 X3 gave 16.0k -> 16.8k+ with one gcc but not another. "I don't know
  how/whether we can reasonably detect which BF_X2 setting is best." (runtime benchmarks too unstable)
  https://openwall.com/lists/john-dev/2014/12/25/1
- **In-order/low-IPC cores need interleave even more** (Epiphany, slide 13): single instance can't hide
  4-cycle FPU-as-integer latency; two instances per core: 947 -> 1194 c/s after asm (+26%); "Preload
  P-boxes: 996 c/s" (+5%); "Transfer keys only when changed: 1207 c/s". Same PDF as above.

### 1d. Format-layer (cracker-only) shortcuts in JtR

From BF_fmt.c / BF_std.c / BF_common.c:

- **Partial final encryption**: JtR computes only the first 8 bytes (one Blowfish block) of the
  64x-encrypted "OrpheanBeholderScryDoubt" for candidate rejection; BF_std_crypt_exact() computes the
  remaining 4 words only when a candidate matches (BINARY_SIZE=4, compares only BF_out[i][0]).
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_fmt.c
- **Sign-extension bug as per-salt mode**: BF_std_set_key(key, index, sign_extension_bug); subtype 'x'
  hashes -> signed-char key schedule, everything else ($2a$ treated as $2y$/$2b$, documented in
  BF_common_get_salt) -> unsigned. Exactly one variant is computed per candidate.
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_common.c
- **Bug-compat output mask**: `BF_out[index][5] &= ~0xFF` — bcrypt's 23-byte output quirk preserved.
- **Per-candidate precompute**: BF_exp_key (key words XORed into P every cost-loop iteration) and
  BF_init_key (initP ^ key, for the initial ExpandKey) are built once per candidate; the cost loop just
  XORs them in (18 unrolled XORs) and re-encrypts. Salt words are held in registers (u1-u4) across both
  BF_body calls per iteration and XORed into P[0..17] in an unrolled rotating pattern.
- **valid() rejects non-canonical base64 padding** (last salt char low 2 bits must be 0; char 28 low 4
  bits must be 0) — avoids wasting compute on hashes whose unused bits would never match canonical output.

---

## 2. Sign-extension / 8-bit-safety tricks — what exists, what transfers

What JtR actually does (full extent; there is no BF_body_512 in JtR — that name does not exist in the
source; the 512-bit-era material is the MIC masked-gather experiment of 2014):

1. One key-schedule variant per salt, chosen by subtype ('x' = buggy sign extension; everything else =
   correct zero extension). Cost is identical either way; the "trick" is only that a cracker never
   computes both. Evidence: BF_std_set_key signature and BF_fmt.c crypt_all switching sign_extension_bug
   on salt subtype (URLs in 1d).
2. Deferred completion of the output (compute 1 of 3 final blocks; finish on hit) — this is the real
   "skip cleanup work" trick. Evidence: BF_std_crypt vs BF_std_crypt_exact in BF_std.c (1d URLs).
3. Password-byte wrap without length precompute: `if (!*ptr) ptr = key; else ptr++;` inside the 4x
   unrolled byte loop — branchless-ish cycling through the key with no strlen on the hot path
   (set_key is off the hot path anyway; called once per candidate).

Applicability to a correctness-first library:
- (1) is what a correct library already does (implement $2b$ semantics; optionally $2x$ for compat).
  No speedup available here.
- (2) does not apply to hashing (must emit all 23 bytes). For a verify() API an early-reject after the
  first 8 bytes is possible, but: (a) savings are ~0.4% of total work at cost 5 (final stage = 192 block
  encryptions of ~34k total; skipping 128 of them) and ~0.05% at cost 10; (b) early exit on mismatch is a
  timing side channel on the candidate, unacceptable for a general library. Rejected.
- (3) is irrelevant off the hot path.
Verdict: the sign-extension family of tricks is cracker bookkeeping, not throughput. Nothing transfers.

---

## 3. The S-box lookup bottleneck — everything tried beyond lockstep lanes

### 3a. Hardware gathers: lost everywhere they were tried (2013-2014), mixed since
- Haswell AVX2 vpgatherdd x8: 4186 c/s vs JtR scalar X2 6595 c/s (slide 42); single-thread 1025 h/s vs
  661 scalar. 7 lanes to fit 28 KiB L1: 926 h/s (worse). "Apparently, Haswell's gather loads are just
  slow; maybe a future CPU will do better." https://openwall.com/lists/john-dev/2013/11/01/1
- Xeon Phi 5110P 512-bit masked gathers x16: 4147 c/s vs scalar OpenMP 6246 c/s (slide 40).
- Manual gathers (SSE4.1 insert chains, "Alain", 2015): "~4k c/s on quad-core Haswell", i.e. comparable
  to hardware gather: "It is not surprising that there's little difference between Haswell's microcoded
  AVX2 gather loads and manually performed ones at x86 instruction stream level."
  https://marc.info/?l=john-dev&m=143511957528608&w=2
- AMD Zen 4/Zen 5: gathers remain microcoded; GCC znver5 tuning disables gather auto-vectorization by
  default (5-30% losses in some benchmarks). https://www.phoronix.com/news/AMD-Zen-5-Tuning-Part-2-GCC
  uops.info Zen 5 VPGATHERDD YMM: ~32 uops, ~12 cycles/instr; ZMM ~64 uops, ~20 cycles:
  https://uops.info/html-tp/ZEN5/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
- Intel Downfall/GDS (CVE-2022-40982) microcode adds latency to gather results on Skylake-Ice Lake/Tiger
  Lake/Rocket Lake: gather-heavy workloads 6-39% slower (Phoronix), up to ~50% worst case (Intel);
  GCC 14+ avoids emitting gathers on affected parts. https://www.phoronix.com/review/intel-downfall-benchmarks
  https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/resources/gds-mitigation-performance-analysis.html
  Sapphire Rapids and later are NOT affected. https://www.phoronix.com/news/GCC-Workaround-Intel-Downfall
- pbcrypt (2020, UBA student project): AVX2 x8 with vpgatherdd claims +175% vs its own non-interleaved
  scalar baseline — but never compared against JtR-class interleaved scalar; consistent with the 2013
  result that gather-x8 beats *single* scalar (~1.55x) but loses to interleaved scalar.
  https://github.com/cat-j/pbcrypt , https://www-2.dc.uba.ar/trabajosFinalesOrga2/2020_JUARROS/informe.pdf

### 3b. Hybrid scalar-load + SIMD-combine (the only "vector" bcrypt that ever won on x86)
The 2015 MMX2 2x2 code (1c): scalar loads, SIMD adds/xors across instance pairs. Won big on narrow cores
(P3 +15%, single Haswell instance +28%), ~nothing on a wide OoO core under full load (+1.6%). Lesson for
our SoA design: when load ports are the ceiling, moving F() ALU work into vector units buys little; when
ALU/issue-bound (in-order, low-IPC, or few threads), it buys 15-28%.
https://marc.info/?l=john-dev&m=143511957528608&w=2

### 3c. L1-residency engineering
- Instance layouts: adjacent-lane-words (SoA-like) vs 4-KiB-per-instance offsets; Solar's 4-KiB-offset
  hack let 7 instances fit 28 KiB — tested, still lost to interleave (3a). Working set math that matters:
  lanes x 4 KiB x threads-per-core. The Haswell AVX2 failure was with 8 lanes x 4 KiB x 2 HT threads =
  "Over 64 KB per core, but we only have 32 KB L1 data cache" (slide 42).
- Bank/set conflicts: hashcat's bcrypt OpenCL kernel gained ~11% on NVIDIA from
  BCRYPT_AVOID_BANK_CONFLICTS (Sc00bz) — GPU shared-memory banks, but the analog on CPU is L1 set
  conflicts: any layout where S-box regions are a multiple of 4 KiB apart maps S0[e],S1[e],S2[e],S3[e]
  to the same 64-set L1 congruence class (all 4 lookups of a round land in one set; 8-12 ways absorb it,
  but it stacks with P + stack + lanes). https://fossies.org/linux/privat/hashcat-7.1.2.tar.gz/hashcat-7.1.2/docs/releases_notes_v7.0.0.md?M=913
- Hashcat also gained ~25% on NVIDIA via OPTS_TYPE_DYNAMIC_SHARED (use >48 KiB shared mem -> more
  resident 4-KiB instances). https://sources.debian.org/data/main/h/hashcat/6.2.6+ds1-1/docs/hashcat-plugin-development-guide.md
  CPU analog: match lane count to actual L1d size (48 KiB Zen 5 / SPR vs 32 KiB Zen 4 / older Intel).

### 3d. Things that do NOT exist in the literature (verified absent; with reasons)
- Two-round software pipelining across Blowfish rounds: impossible — round i+1's indices depend on round
  i's output (Feistel chain). Within a round the 4 lookups are independent; that parallelism is already
  exploited by every implementation above. No JtR/hashcat/academic source does cross-round pipelining.
- Data-dependent prefetch: useless — S-box addresses are known only after the previous round completes
  (1-2 cycles before the load issues anyway). No published use.
- Cache-blocking the 2^cost loop: impossible — iterations are sequentially dependent on mutating state;
  the S-boxes themselves are rewritten every ExpandKey (521 block encryptions). No published use.
- L2-resident S-boxes for wider lane counts: implicitly tested and rejected — Haswell 8-lane (L2-bound
  past 32 KiB/thread-pair) lost badly to L1-resident interleave (3a); Epiphany lost 12% when code+data
  left local memory (932 vs 822 c/s, slide 12-14). bcrypt punishes L2 latency on the F() critical path.

---

## 4. Measured bcrypt throughput corpus (cost 5, JtR --test unless noted)

### CPU (JtR, scalar interleaved — the only thing JtR ships for bcrypt)
| CPU | Result | Config | Source |
|---|---|---|---|
| i7-4770K (Haswell 4c/8t) | 6,595 c/s | X2, 8 threads | Passwords14 slide 42/44 (PDF above) |
| i7-4770K theoretical L1-port peak | ~11,093 c/s (60% achieved) | 4c, 2 load ports, 3.7 GHz | slides 51-52; marc.info 2015 email |
| i7-5820K (Haswell-E 6c/12t) | X2 8,625-8,928 vs X3 8,208-8,424 c/s | 12 threads | john-dev 2014/12/25/1 |
| 2x E5-2670 (SNB 16c/32t) | 16,900 c/s (X3) | 32 threads | Passwords14 slide 44 |
| Xeon Phi 5110P (MIC 60c) | 6,246 c/s scalar vs 4,147 c/s 512-bit gather | 240 threads | Passwords14 slides 40 |
| Ryzen 7 8700F (Zen 4 8c) | 22,065 c/s | 16 threads | https://openwall.info/wiki/john/benchmarks |
| Ryzen 9 7950X (Zen 4 16c) | 44,611-47,718 c/s (~2.8-3.0k/core) | 32 threads | https://openbenchmarking.org/result/2311203-NE-DUPLICATE64&sro&grw |
| Ryzen 9 9950X (Zen 5 16c) | 57,200-60,361 c/s (~3.6-3.8k/core; +27% vs 7950X) | 32 threads | https://openbenchmarking.org/result/2507290-PTS-THREADRI83 |
| Threadripper 7980X (Zen 4 64c) | 139,601-143,996 c/s | 128 threads | https://openbenchmarking.org/result/2505255-NE-JTRTR960581&export=html |
| 2x Xeon Platinum 8380 (Ice Lake 80c) | 113,219 c/s | 160 threads | https://openbenchmarking.org/result/2502248-NE-TURINLAUN20 |
| 2x Xeon Platinum 8490H (SPR 120c) | 183,603 c/s; 2x 8592+: 206,778 c/s | 240t | same |
| Apple M2 (8c) | ~2,113 c/s (median, JtR) | 8 threads | https://unitesi.unive.it/retrieve/88f073e7-d329-40e9-949d-d5fa23b92051/884784_Rizzo_Elisa.pdf |

### GPU / other (context only; techniques largely don't transfer)
- Hashcat bcrypt (-m 3200) keeps per-work-item 4 KiB S-boxes in local/shared memory; occupancy ~12
  threads/SM at 48 KiB. https://raw.githubusercontent.com/hashcat/hashcat/master/OpenCL/m03200-pure.cl
- RTX 2080 Ti: ~42k -> ~55k H/s via dynamic shared memory (2020). (debian docs URL in 3c)
- Apple GPUs barely beat Apple CPUs at bcrypt: M1 ~1,990 H/s (Metal) vs M2 JtR ~2,113 c/s (CPU);
  M1 Max 9,342; M3 Pro 6,032; M3 Max ~9,401 H/s. https://gist.github.com/matrix/c40d59a6254b1dc33c626bd79c244bf1
  https://gist.github.com/Chick3nman/ccfb883d2d267d94770869b09f5b96ed
- FPGA (lesson: many small L1-local instances beat wide SIMD): Zynq-7045, 196 instances @ 71.4 MHz =
  20,538 c/s at ~5 W (3x a 4770K). Wiemer & Zimmermann: 40 instances, ZedBoard 80 MHz, 5,208 c/s.
  Passwords14 slides 46-53; paper: https://www.usenix.org/system/files/conference/woot14/woot14-malvoni.pdf

Cross-checks against our baselines: Zen 4 JtR X3 = ~2.8-3.0k c/s/core (all-core SMT-on), i.e. ~1.4-1.5k
per thread — the number to beat per core with our AVX2-x8 batch (currently 1.54x our scalar). M2 JtR
~2.1k c/s for the whole chip vs our M5 Max GPR-interleave 2,233 h/s per core — ARM cores are far stronger
per-core here, consistent with our NEON 2.78x / GPR-interleave wins on Apple silicon.

---

## 5. Zen 5 and Intel AVX10/AVX10.2 — does the AVX-512-vs-AVX2 crossover move?

Zen 5 facts (Granite Ridge desktop / Turin server; NOT Strix Point mobile, which stays double-pumped):
- Native full-width 512-bit datapath (vs Zen 4 double-pumped zmm); many 512-bit ops 2x Zen 4 throughput.
  https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/
- L1d grows 32 KiB/8-way -> 48 KiB/12-way, 4 load pipes (up to 2x 512-bit loads/cycle), 4 AGUs.
  https://techpowerup.com/review/amd-zen-5-technical-deep-dive/3.html
- Gathers still microcoded (see 3a: ~32 uops YMM / ~64 uops ZMM, ~12/~20 cycles) — "only a slight
  improvement over Zen 4 and remains slower than Intel's hardware gathers"; GCC znver5 default: no gathers.
- JtR scalar X3 on 9950X: +27% over 7950X at equal thread counts (table above) — plausibly explained by
  the 4th load pipe, since interleaved scalar bcrypt is load-port-bound (Solar's 60%-of-peak analysis).
  This is an inference from the cited numbers, not a measured attribution.

Implication for our SoA backends on Zen 5: 16 lanes x 4 KiB = 64 KiB still exceeds the 48 KiB L1d;
8 lanes x 4 KiB = 32 KiB fits with headroom. Prediction: AVX2-x8 remains the winner on Zen 5, but the
AVX-512-x16 penalty shrinks (native 512-bit datapath, 12-way L1d softens evictions; gathers don't matter
if we use insert chains). The crossover to AVX-512-x16 needs L1d >= 64 KiB (+P/stack headroom), i.e.
not Zen 5. A 12-lane x 4 KiB = 48 KiB variant would exactly fill L1d with zero headroom — not viable.

Intel AVX10/AVX10.2:
- AVX10.2 confirmed for Nova Lake (~2026, P-cores "Coyote Cove" + E-cores "Arctic Wolf", native 512-bit
  on both) and Diamond Rapids; NOT in Panther Lake (256-bit max). AVX10.2 adds no new gather opcodes —
  it re-enables the AVX-512 gather family at 512-bit across all cores.
  https://www.techpowerup.com/342881/intel-officially-confirms-avx10-2-and-apx-support-in-nova-lake
  https://www.phoronix.com/news/Nova-Lake-Does-AVX10.2-APX
  https://en.wikipedia.org/wiki/Panther_Lake_(microprocessor)
- Intel's hardware gathers are the good kind (dedicated hardware since Skylake-X), post-Downfall parts
  (SPR+, future AVX10.2) are unaffected by the gather penalty. If Nova Lake keeps a 48 KiB L1d
  (as Redwood Cove/Cougar Cove do), the same 8-lane-fits/16-lane-spills math applies.
- Net: nothing in Zen 5 or AVX10.2 changes the fundamental constraint — bcrypt's crossover is governed
  by L1d bytes vs N x 4 KiB and by load ports, not by vector width. Wide vectors win only where per-core
  local memory per lane stays ~4 KiB in L1 (or with FPGA-style many-instance designs).

---

## 6. F() algebra — what is legal, what is not

F(x) = ((S0[a] + S1[b]) XOR S2[c]) + S3[d], all adds mod 2^32 (wrapping).

- The two inner adds are fine to reorder/parallelize: S0[a]+S1[b] is a single addition of two
  independent loads; issue both loads, one add.
- The final + S3[d] CANNOT be moved before the XOR. Counterexample (bit-exactness): A=1, B=1, C=1:
  (A^B)+C = 0+1 = 1, but (A+C)^B = 2^1 = 3. Wrapping-add carries interact with XOR, so no reassociation
  across the XOR preserves bit-exactness. There is no algebraic shortening of F().
- What IS legally exploitable (and JtR exploits all of it):
  1. All four index extractions depend only on L — fully parallel (BEXTR/movzbl/ubfx).
  2. All four S-box loads are independent — issue in parallel; OoO does this given free AGUs.
  3. The S3[d] load and the (S0+S1) add proceed in parallel; critical path after loads land is
     add(1) -> xor(1) -> add(1) -> xor R(1) = 4 cycles. Per-round latency ~= L1 load latency + 4.
  4. R ^= P[N+1] is independent of F() — scheduled between loads and combines (BF_ROUND source, 1c).
  5. On x86, fold loads into ALU ops: `addl S1(,idx,4), %reg`, `xorl S2(,idx,4), %reg` — one micro-fused
     load-op each; this is how the BMI version reaches 10 instructions/round/instance (1c).
- Consequence: single-instance bcrypt is latency-bound (~9-10 cycles/round, IPC << width); the ONLY
  general fixes are interleaving independent instances (ILP) or SIMD lanes (same thing, wider). This is
  the theoretical root of everything in sections 1-3.

---

## 7. Prioritized technique list (mapped to bcrypt-rust)

Baselines: M5 Max NEON 2.78x scalar, GPR-interleave 2,233 vs NEON 1,073 h/s; Zen 4 AVX2-x8 1.54x /
AVX-512-x16 1.42x scalar; runtime gather-vs-insert shootout already in place.

### P1. GPR-resident scalar interleave on x86-64 (X2/X3), SMT-aware factor choice — MEDIUM/LARGE
What: JtR's only shipping technique: 2-3 independent hashes, instructions interleaved in scalar regs,
per-instance S-boxes in L1. No vector registers involved.
Evidence + quotes:
- "For comparison, JtR achieves higher speeds on this same CPU than any of the numbers seen above ...
  because JtR's scalar code runs two instances with interleaved instructions" (6,595 vs AVX2's best
  4,186 on 4770K). https://openwall.com/lists/john-dev/2013/11/01/1
- X2 vs X3 flips with SMT: i7-5820K 12t: X2 8,625-8,928 vs X3 8,208-8,424; non-SMT CPUs prefer X3.
  https://openwall.com/lists/john-dev/2014/12/25/1
- +15-28% from interleave-adjacent techniques on narrow/in-order cores (MMX2, Epiphany 2-instance).
  https://marc.info/?l=john-dev&m=143511957528608&w=2
Applicability: our aarch64 GPR-interleave already proves the pattern (2.08x over vector-resident on
M5 Max). On x86 we currently offer scalar(single) + SIMD batch only; an X2/X3 GPR backend would (a) beat
single scalar by ~1.6-2x per JtR history, (b) possibly beat AVX2-x8-insert under SMT (working set per
core halves vs lanes x threads), (c) give a gather-free path for Downfall-affected Intel. JtR's Zen 4
number (~3k c/s/core X3) is the bar for our AVX2 backend on the CF sandbox.
Impact: MEDIUM (5-20%) on x86 overall; potentially LARGE (>20%) on SMT-saturated or Downfall-affected
parts where our current SIMD backends degrade. Effort: moderate (port the existing ARM interleave
structure; 18 P-words + 4 S-base pointers + 2x3 instance regs fit x86-64's 15 GPRs about as tightly as
ARM's — expect some P spills; JtR reads P from memory for the same reason).

### P2. Working-set discipline: lanes x 4 KiB x threads-per-core <= L1d — MEDIUM
What: size lane groups to per-core L1d including SMT siblings; 1 lane-group per physical core under SMT,
or halve lanes per thread.
Evidence: "8 bcrypt instances per thread, 8 threads on 4 cores: 4186 c/s — Over 64 KB per core, but we
only have 32 KB L1 data cache" (slide 42, PDF); 8 concurrent AVX2 processes dropped to ~522 h/s each vs
1025 single-process (port+L1 contention). https://openwall.com/lists/john-dev/2013/11/01/1
Applicability: our Zen 4 sandbox result (AVX2 1.54x vs AVX-512 1.42x) is exactly this effect at
32 KiB L1d. Extend the rule into the backend: expose lane-group-size selection (16/8/4) keyed to
detected L1d size and threads-per-core, instead of fixing it by ISA width. On 48 KiB L1d (Zen 5, SPR,
Nova Lake) an 8-lane group still fits; on 32 KiB parts with 2-way SMT, 4-lane groups may win.
Impact: MEDIUM on SMT servers (up to ~1.5x in the pathological 2-threads-x-8-lanes case per the Haswell
data); SMALL on 1-thread/core setups. Effort: low (bench + dispatch policy).

### P3. Instruction-stream diet in the scalar/interleave path (BMI/ubfx index extraction, load-op folding,
64-bit P reads, salt in registers) — SMALL/MEDIUM
What: minimize non-load uops per round so load ports saturate; fold S-box loads into add/xor memory
operands; extract byte-indices with BEXTR (x86) / UBFX (ARM); read the sequential P-array as u64 pairs;
hold salt words and exp_key in registers across the cost loop (JtR does all four).
Evidence: "the bottleneck ... appears to be taking too many instructions and uops to extract all of the
S-box indices and compute the effective addresses to keep the L1 data cache read ports 100% busy. (We
only keep them about 60% busy.)" + "10 instructions per round per instance" BMI sketch + "4.5 reads/round,
it'd be 12.3k". https://marc.info/?l=john-dev&m=143511957528608&w=2
Applicability: audit LLVM's codegen for our scalar/interleave kernels: check for redundant AND/SHR
chains, non-folded loads, P re-loads per round, salt spills. Rust-side fixes: u64 P-array type, explicit
`& 0xFF` patterns LLVM maps to movzbl/ubfx, `get_unchecked` to drop bounds-check uops.
Impact: SMALL to MEDIUM — JtR's own ceiling estimate says up to +40% is theoretically available
(60% -> 100% port utilization) but their best hand-asm attempt captured only ~1.6-5% on wide cores;
realistically <5-10% for us. Effort: low-medium (codegen audit first).

### P4. Gather-vs-insert policy per microarchitecture (extend our shootout's defaults) — SMALL/MEDIUM
What: hardware gathers only where they are real hardware: Intel Skylake-X..Ice Lake post-Downfall is
penalized (6-39% gather-heavy regressions), SPR/GNR fine, AMD Zen 4/5 microcoded (avoid), AVX10.2
Nova Lake unknown (test at launch).
Evidence: Phoronix Downfall review; Intel GDS perf analysis; GCC znver5 disables gathers; GCC 14
Downfall workaround emits scalar instead of gathers on affected Intel. (URLs in 3a)
Applicability: we already measure at runtime; add a static default table so first-call latency and
pathological cases (VM without PERF) pick sane defaults: insert on AMD always; insert on Downfall-affected
Intel unless mitigations=off; gather-allowed on SPR+/GNR.
Impact: SMALL/MEDIUM (fleet-dependent; up to ~30% on a Downfall-patched Ice Lake if we currently pick
gathers there). Effort: low.

### P5. L1 set-conflict padding of the SoA S-box stride — SMALL
What: our SoA S-box regions for lane-group size N are N KiB each; for N in {4,8,16} that is a multiple
of 4 KiB, so S0[e],S1[e],S2[e],S3[e] all map to the same L1 set (64 sets x 64 B = 4 KiB span). Offsetting
each S-box region by i*64 B rotates the four lookups across sets.
Evidence: hashcat's BCRYPT_AVOID_BANK_CONFLICTS "+~11%" on NVIDIA (shared-mem banks — different hardware,
same structural fix). https://fossies.org/linux/privat/hashcat-7.1.2.tar.gz/hashcat-7.1.2/docs/releases_notes_v7.0.0.md?M=913
Solar's 4-KiB-offset layout work shows they cared about the same congruence on CPU.
https://openwall.com/lists/john-dev/2013/11/01/1
Applicability: 4 same-set lines/round is absorbed by 8-12-way L1d on modern cores, so expected gain is
small — but the experiment is ~free (one stride constant).
Impact: SMALL (<5%). Effort: trivial.

### P6. Keep everything outside the cost loop interleaved/batched too — SMALL
What: Solar's unrealized plan: "making the uses of Blowfish outside of the variable cost loop use the
interleaved implementation as well ... something like 6.8k c/s" (from 6.6k, +3%).
Evidence: same 2015 email; slide 50 formula (585 of 33,353 block encryptions are outside the 2^cost loop
at cost 5 — 1.8%; at cost 10: 0.06%).
Applicability: our batch backends should already batch the initial ExpandKey and the final 64x loop;
verify no scalar fallback remains in those paths at low cost.
Impact: SMALL (<2%, only at benchmark cost 4-5; negligible at production costs). Effort: low.

### P7. Cracker-only shortcuts — REJECTED for a correctness-first library (documented for completeness)
- Partial final encryption + deferred exact compare (BF_std_crypt_exact): saves 128 of ~34k encryptions
  (~0.4% at cost 5) and leaks timing on verify. Not worth it. (BF_fmt.c/BF_std.c URLs, 1d)
- Sign-extension single-variant: we already compute exactly one variant ($2b$ semantics). No-op.
- $2a$-as-$2y$ assumption: a library must not guess; keep correct per-variant behavior.

### P8. Techniques confirmed NOT to exist / not to work (do not spend time)
- Cross-round software pipelining, data-dependent prefetch, cost-loop cache-blocking, L2-resident S-boxes
  (section 3d) — absent from JtR/hashcat/academia for structural reasons.
- Mixed SIMD+scalar co-scheduling on one core ("careful mix ... would be slightly faster. This is yet to
  be tested" — 2013; "Replacing one of the MMX2 instances with one or two scalar instances ... somehow
  didn't help" — 2015). Only try after P1-P4, on Intel P-cores with 3 load ports.

---

## 8. Recommended experiments, ranked by impact/effort

1. **x86-64 GPR interleave backend (X2 and X3), benchmarked against AVX2-x8-insert on the Zen 4 sandbox
   and on an Intel part with and without SMT siblings loaded.**
   Why first: it is the only technique JtR itself ships after 25 years of trying everything else; our ARM
   results (GPR 2,233 vs NEON 1,073 h/s) suggest the same effect may hold on x86 under SMT. Cost: a port
   of a pattern we already have. Success metric: beat 1.54x-scalar AVX2-x8 per physical core with SMT on;
   also compare vs JtR's ~3k c/s/core Zen 4 X3 figure as an external reference point.
2. **Lane-group-size dispatch matrix (16/8/4 lanes) x (1 or 2 threads/core) x (32/48 KiB L1d) on Zen 4,
   Zen 5 (via cloud), SPR-class Intel.** Pure benchmark work; directly operationalizes P2 and produces
   the dispatch table for production. Include a 4-lane AVX2 config on SMT hardware — the Haswell data
   predicts it can beat 8-lane when two siblings share 32 KiB.
3. **Codegen audit of the scalar/interleave hot loop (objdump + uops.info/llvm-mca):** verify folded
   S-box load-ops, movzbl/ubfx-only index extraction, u64 P reads, no bounds-check uops, salt/exp_key
   register-resident. Fix via Rust type/pattern changes before considering any asm. (P3; cheap, and it
   de-risks experiment 1 by ensuring the GPR backend is uop-lean.)
4. **S-box stride padding (+64 B per region) A/B test** on M5 Max (NEON x4, GPR x4) and Zen 4
   (AVX2 x8, AVX-512 x16). One constant change; keep if >1% and consistent. (P5)
5. **Static gather/insert policy table** keyed on CPUID family/model + Downfall mitigations flag
   (read /sys/devices/system/cpu/vulnerabilities/gather_data_sampling on Linux), runtime shootout kept as
   fallback. Validate on the oldest Intel we support and on SPR+ if accessible. (P4)
6. **Zen 5 AVX-512-x16 re-test** when Zen 5 cloud hardware is available (or via the existing CF sandbox
   if it can schedule Zen 5): hypothesis — still <= AVX2-x8 (64 KiB > 48 KiB L1d), but the gap narrows
   from Zen 4's 1.42/1.54. Publish whichever way it lands; it calibrates the dispatch rule. (P2/P5)
7. **Batch-coverage audit at low cost:** confirm initial ExpandKey + final 64x loop also run batched
   (matters only at cost 4-6, ~1-3%). (P6)
8. **(Last, only if 1-3 leave headroom) mixed SIMD-batch + GPR-interleave co-scheduling on one physical
   core**, Intel P-core only. JtR's own notes call this untested-and-unpromising; treat as a lottery
   ticket. (P8)

Methodology notes (from JtR's experience): benchmark interleave/lane choices with full thread counts —
relative results invert between 1-thread and all-thread runs (2014/12/25/1); pin cost (JtR uses cost 5;
production costs shift the outside-loop share to nil); watch for frequency/wobble — Solar rejected
runtime auto-tuning of BF_X2 because "running benchmarks at build- or run-time is unstable or slow".

---

## Appendix: primary sources

- JtR BF_std.c (interleave macros, cost loop, crypt_exact):
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.c
- JtR BF_std.h (X2/X3 naming): https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_std.h
- JtR BF_fmt.c (BINARY_SIZE=4, sign-extension switch):
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_fmt.c
- JtR BF_common.c ($2a$->$2y$ policy, base64 checks, 23-byte mask):
  https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/src/BF_common.c
- crypt_blowfish x86.S (Pentium-era scheduling):
  https://raw.githubusercontent.com/openwall/crypt_blowfish/master/x86.S
- Malvoni, Designer, Knezovic — "Energy-efficient bcrypt cracking" (Passwords^14 / WOOT'14):
  https://www.openwall.com/presentations/Passwords14-Energy-Efficient-Cracking/Passwords14-Energy-Efficient-Cracking.pdf
  https://www.usenix.org/system/files/conference/woot14/woot14-malvoni.pdf
- Solar Designer, john-dev 2013-11-01 (AVX2 gather experiments):
  https://openwall.com/lists/john-dev/2013/11/01/1
- Solar Designer, john-dev 2015-06-24 (BMI/MMX2/port analysis):
  https://marc.info/?l=john-dev&m=143511957528608&w=2
- Solar Designer, john-dev 2014-12-25 (X2 vs X3 vs SMT): https://openwall.com/lists/john-dev/2014/12/25/1
- Solar Designer, john-users 2020-10/11 (bcrypt has no SIMD in JtR; AVX-512 elsewhere):
  https://openwall.com/lists/john-users/2020/10/15/1 , https://openwall.com/lists/john-users/2020/11/11/1
- JtR benchmarks wiki: https://openwall.info/wiki/john/benchmarks
- OpenBenchmarking JtR results: https://openbenchmarking.org/test/pts/john-the-ripper (2311203-NE-DUPLICATE64,
  2507290-PTS-THREADRI83, 2502248-NE-TURINLAUN20, 2505255-NE-JTRTR960581)
- Hashcat bcrypt kernel: https://raw.githubusercontent.com/hashcat/hashcat/master/OpenCL/m03200-pure.cl
- Hashcat v7.0.0 notes (bank conflicts): https://fossies.org/linux/privat/hashcat-7.1.2.tar.gz/hashcat-7.1.2/docs/releases_notes_v7.0.0.md?M=913
- Hashcat plugin dev guide (dynamic shared mem):
  https://sources.debian.org/data/main/h/hashcat/6.2.6+ds1-1/docs/hashcat-plugin-development-guide.md
- pbcrypt (AVX2 gather academic project): https://github.com/cat-j/pbcrypt ,
  https://www-2.dc.uba.ar/trabajosFinalesOrga2/2020_JUARROS/informe.pdf
- Zen 5 AVX-512 teardown: https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/
- Zen 5 architecture (L1d): https://techpowerup.com/review/amd-zen-5-technical-deep-dive/3.html
- Zen 5 GCC tuning (gathers off): https://www.phoronix.com/news/AMD-Zen-5-Tuning-Part-2-GCC
- uops.info Zen 5 gather: https://uops.info/html-tp/ZEN5/VPGATHERDD_YMM_VSIB_YMM_YMM-Measurements.html
- Intel GDS/Downfall: https://www.phoronix.com/review/intel-downfall-benchmarks ,
  https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/resources/gds-mitigation-performance-analysis.html ,
  https://www.phoronix.com/news/GCC-Workaround-Intel-Downfall
- AVX10.2 / Nova Lake: https://www.techpowerup.com/342881/intel-officially-confirms-avx10-2-and-apx-support-in-nova-lake ,
  https://www.phoronix.com/news/Nova-Lake-Does-AVX10.2-APX ,
  https://en.wikipedia.org/wiki/Panther_Lake_(microprocessor)
- Apple silicon numbers: https://unitesi.unive.it/retrieve/88f073e7-d329-40e9-949d-d5fa23b92051/884784_Rizzo_Elisa.pdf ,
  https://gist.github.com/matrix/c40d59a6254b1dc33c626bd79c244bf1 ,
  https://gist.github.com/Chick3nman/ccfb883d2d267d94770869b09f5b96ed
- Context decks: https://www.openwall.com/presentations/OffensiveCon2024-Password-Cracking/ ,
  https://www.openwall.com/presentations/Passwords12-The-Future-Of-Hashing/ (the "25 GH/s"-era material:
  fast-hash GPU cracking context that motivated memory-hard KDFs; contains no bcrypt-SIMD techniques)
