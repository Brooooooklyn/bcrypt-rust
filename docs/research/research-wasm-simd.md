# bcrypt under WebAssembly SIMD128: achievable ceiling and closing techniques

Date: 2026-10-02. Baseline under test: bcrypt-rust `wasm128` backend, wasm32-wasip1
`+simd128`, wasmtime 48, Apple M5 Max, cost 4, batch 8: scalar-in-wasm 1257 h/s,
wasm128 4-lane lockstep 1522 h/s (**1.21x**). Backend shape: SoA interleaved S-boxes
(`SBoxes4x`, 16 KiB), one lookup = `v128.store` of indices to stack + 4 scalar
`i32.load` + `v128.load` of rebuilt results (`src/eks/wasm128.rs`, `lookup4`).

**Verdict up front:** yes, ~1.2x is essentially at the ceiling for the 4-lane
lockstep shape. The strongest evidence is native: a 4-lane AVX2 bcrypt with a
*real hardware gather instruction* (`vpgatherdd`) only reached **1.33x** over its
own scalar baseline (pbcrypt), and John the Ripper's 8-lane AVX2-gather bcrypt
(~1.55x over 1-instance scalar) *lost* to 2-3x instruction-interleaved scalar
(~1100 vs ~1030 h/s on Haswell). Our emulated-gather wasm kernel at 1.21x is
already ~91% of the native 4-lane-with-gather result. The remaining upside is
not lookup mechanics — it is interleaving more independent instances per kernel
(the JtR X2/X3 trick, transplanted into lanes), plus threads for batch work.

## 1. S-box lookups in wasm SIMD: what the crypto ports do, and why their tricks don't transfer

WASM SIMD128 has **no gather/scatter** and no runtime-indexed table op wider
than 16 bytes. The complete inventory of data-movement ops is: contiguous
`v128.load/store`, lane loads/stores, `i8x16.shuffle` (two-source but *immediate*
indices), `i8x16.swizzle` (runtime indices into a **single** 16-byte register
table), and extract/replace lane. Gather was deliberately omitted from the
fixed-width proposal (portability/determinism; 128-bit gather rarely pays), and
the successor "Flexible Vectors" proposal (Phase 1, not shipping) does not
include gather in its first tier either. So no gather is coming on any
forecastable timeline.
- https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md
- https://github.com/WebAssembly/flexible-vectors/blob/main/proposals/flexible-vectors/Overview.md
- https://v8.dev/features/simd

How existing wasm crypto kernels cope with S-boxes:

- **AES:** the modern wasm answer is Mike Hamburg's register-table trick —
  decompose GF(2^8) as GF(2^4)[t], compute the S-box with ~15 `i8x16.swizzle`s
  against 16-byte tables held in registers (constant-time, no memory lookups).
  This works *because the AES S-box has algebraic structure* (inversion +
  affine map). Summary and out-of-range-selector semantics table:
  https://00f.net/2026/07/16/aes-with-simd-swizzles/ and
  https://www.shiftleft.org/papers/vector_aes/vector_aes.pdf
- **ChaCha/Blake:** no tables at all — pure add/rotate/xor, vectorizes cleanly.
- **Blowfish/bcrypt:** S-boxes are (a) arbitrary data (pi hex digits), (b) 32-bit
  wide, (c) 256 entries = 4 KiB each, and (d) **key-dependent, mutated during
  the EksBlowfish expansion**. There is no GF structure to exploit, so the
  Hamburg construction does not exist for it. Bitslicing an arbitrary,
  key-dependent 8->32-bit LUT is a boolean circuit of tens of thousands of gates
  that would have to be *re-derived per key* — a dead end, not an optimization.
- **"Transposed lookup via byte shuffles":** dead end, quantifiably. The largest
  runtime-indexed table wasm can express is 16 bytes (`i8x16.swizzle`);
  `i8x16.shuffle` reaches 32 bytes but only with compile-time-constant indices,
  which table lookups are not. A 256-entry u32 table equals 64 v128 registers
  of candidate data; reducing 64 candidates to one per lane with a bitselect
  tree is ~63 selects + 6 broadcast-compares per lookup — versus 4 scalar loads.
  No engine lowering changes that arithmetic.

That leaves the two spellings that actually exist, and both are in the wild:

1. **Store-to-stack + scalar reload** (what `lookup4` ships): 1 `v128.store`,
   4 independent scalar `i32.load`, 1 `v128.load`. On wasmtime this relies on
   the CPU's own store-to-load forwarding: Cranelift's alias analysis tracks
   stack slots as alias regions but only forwards/redundancy-eliminates on an
   *exact* (address, offset, **type**) match — a `v128.store` (I8X16) followed
   by `i32.load` at the same slot is *not* forwarded at the IR level (soundly;
   cross-width partial forwarding is unimplemented). The generated machine code
   is the naive one, and modern x86/ARM cores forward the contained scalar
   loads microarchitecturally.
   https://docs.wasmtime.dev/api/src/cranelift_codegen/alias_analysis.rs.html
2. **`i32x4.extract_lane` x4** (no memory round-trip): each extract is one
   instruction but a SIMD->GPR domain crossing. V8 x64: `pextrd` (lane 0
   sometimes `movd`); aarch64: `umov`; Liftoff emits the same helper per op.
   Cost ~2-3 cycle latency each, plus register-pressure effects that can spill
   other live v128s in Liftoff's naive allocation.
   https://github.com/zeux/wasm-simd/blob/master/Instructions.md

Neither spelling is fundamentally better; both cost ~4 serial-ish crossings per
gather. The stack spelling additionally keeps the GPR->SIMD return crossing as
one `v128.load` instead of 3 `replace_lane` inserts, which is why it is the
right default.

## 2. relaxed-simd: support, determinism, applicability to the bcrypt mix

Engine support (as of late 2025 / 2026): V8 shipped relaxed-simd by default in
Chrome 114 (2023-05); SpiderMonkey only unflagged it at the very end of 2025
(Firefox ~145-146); JavaScriptCore had nothing in stable Safari through 2025
(WebKit landed it in 2026, first visible in Safari Technology Preview 250).
Wasmtime/Wasmer/WasmEdge/WAVM have supported it by default for a long time.
Rust intrinsics (`i8x16_relaxed_swizzle`, the `*_relaxed_laneselect` family,
etc.) are stable since Rust 1.82 behind `+relaxed-simd`.
- https://caniuse.com/wf-wasm-simd-relaxed
- https://bugzilla.mozilla.org/show_bug.cgi?id=1930192
- https://bugs.webkit.org/show_bug.cgi?id=306939
- https://github.com/rust-lang/rust/releases/tag/1.82.0

Determinism (we need bit-exactness). Relaxed ops are *locally, boundedly*
non-deterministic; per the proposal's Entropy analysis, the classes are:
- `i8x16.relaxed_swizzle`: deterministic whenever every index is in 0..=15 or
  has the high bit set (>=0x80 -> 0). Only indices 0x10..0x7F differ across
  engines (zero on ARM semantics, low-nibble select on x86 PSHUFB semantics).
- `relaxed_laneselect`: deterministic when each mask *lane* is all-0s or
  all-1s; mixed-bit lanes are implementation-defined (some hosts look only at
  the top bit). With well-formed masks it equals `v128.bitselect`.
- `relaxed_dot_i8x16_i7x16_s`: deterministic while the second operand's bytes
  are <= 0x7F.
- `relaxed_min/max` (float): differ only on NaN / signed zero.
- `relaxed_q15mulr_s` and `relaxed_trunc`: rounding/saturation edge cases.
- `relaxed_madd/nmadd`: **non-deterministic for ordinary inputs** (fused vs
  unfused FMA rounding) — the one family to avoid for bit-exactness; the
  spec's deterministic profile forces the unfused path.
- Wasmtime offers `relaxed_simd_deterministic(true)` to pin the deterministic
  profile globally (at some perf cost).
- https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md
- https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Entropy.md
- https://docs.wasmtime.dev/examples-deterministic-wasm-execution.html

**Applicability verdict for bcrypt's F: none.** The kernel's entire vector mix
is `u32x4_shr`, `v128_and`, `v128_xor`, `i32x4_add` — every one already lowers
1:1 to a single NEON/SSE instruction (SHR/AND/EOR/ADD, PSRLD/PAND/PXOR/PADDD).
There is no bitselect to relax into `relaxed_laneselect`, no float anything, and
the 16-byte swizzle ceiling makes `relaxed_swizzle` useless for 256-entry
S-boxes even ignoring determinism. relaxed-simd's integer wins exist for
q15/dot-product DSP kernels, not add/xor/shift Feistel mixes. Shipping a
second `+relaxed-simd` build flavor would buy ~0% and add a validation-time
deployment hazard (Safari); skip it.

## 3. Engine codegen: V8 vs wasmtime vs SpiderMonkey, and the linear-memory question

**v128 shuffle/extract lowering quality is a wash across the three engines**;
differences are scheduling/regalloc noise, not missing patterns:
- V8 (TurboFan): `i32x4.extract_lane` -> 1x `pextrd` (x64) / `umov` (arm64);
  shuffles matched to PSHUFB/TBL; some ops have multi-instruction fixups
  (i8x16 shifts, i64x2 mul) that bcrypt never uses.
- SpiderMonkey (Ion): same class — lane-0 `movd`, else `vpextrd`; comparable
  shuffle specializations; long history of SIMD-specific peepholes.
- Wasmtime (Cranelift, aarch64): extract -> `UMOV`/`SMOV`/`MOV`; common
  shuffle masks pattern-matched to single DUP/EXT/UZP/ZIP/TRN/REV
  instructions with a `TBL2` fallback; `ADDV`-fused pairwise reductions. The
  known codegen gap vs V8 is *instruction scheduling and move elimination*:
  on XNNPACK SIMD kernels Wasmtime trailed V8 by ~10-20% on aarch64/x86_64.
- https://github.com/zeux/wasm-simd/blob/master/Instructions.md
- https://searchfox.org/firefox-main/source/js/src/jit/x86-shared/MacroAssembler-x86-shared-SIMD.cpp
- https://github.com/bytecodealliance/wasmtime/pull/5977
- https://github.com/bytecodealliance/wasmtime/issues/6159

**"S-boxes in linear memory instead of a Rust static array" changes nothing.**
On wasm32, *all* data is linear memory: a Rust `static` is a data segment in
linear memory, and Rust's stack (where `lookup4` stores the index vector) is
the wasm shadow stack in that same linear memory. The store-to-stack spelling
already *is* the "linear-memory aliasing" design — there is no second memory
to move the S-boxes into. The only things that actually affect codegen here:
(a) **bounds checks**: wasmtime/V8/SpiderMonkey all use guard-page virtual
memory reservations for 32-bit memories (4 GiB reservation + 32 MiB guard in
Wasmtime's default config), so the 16 scalar loads per F carry **zero**
explicit bounds-check instructions; keep the module memory32.
(b) **alias regions**: Cranelift puts unique stack slots and heap in disjoint
alias regions, so the index-vector store can never block reordering of the
S-box loads — the current spelling is already optimal for this compiler.
(c) address folding: `idx*4 + lane` is one shift + the load's immediate/address
add on both ISAs. Alignment (64 B, entry stride 16 B) is already ideal.
- https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html
- https://docs.wasmtime.dev/api/src/wasmtime_internal_cranelift/alias_region.rs.html

## 4. Measured ratios: bcrypt/Blowfish specifically

Native reference points for the exact same kernel shape (lockstep lanes,
per-lane key-dependent S-boxes):
- **pbcrypt** (cat-j, AVX2 asm cracker, UBA 2020): 4-key XMM = **+33%** vs its
  scalar baseline; 8-key YMM = **+175%** (i.e. 2.75x). Notably the 4-lane
  version *had `vpgatherdd` available* — real hardware gather — and still only
  got 1.33x. Our emulated-gather wasm 4-lane at 1.21x is ~91% of that.
  https://github.com/cat-j/pbcrypt ,
  https://www-2.dc.uba.ar/trabajosFinalesOrga2/2020_JUARROS/informe.pdf
- **John the Ripper / Solar Designer (2013, Haswell i7-4770K):** AVX2 8-instance
  gather bcrypt ~1000-1030 h/s vs 1-instance scalar ~600-660 h/s (~1.55x) —
  but **2x-interleaved scalar hit ~1100-1110 h/s and beat the vector code**.
  Gather latency + 8 instances x 4 KiB S-boxes blowing past 32 KiB L1d killed
  it. JtR production bcrypt is *scalar X3 instruction interleaving* to this
  day; SIMD gathers never shipped for this format.
  https://openwall.com/lists/john-dev/2013/11/01/1 ,
  https://www.openwall.com/lists/john-users/2019/04/12/2 ,
  https://openwall.info/wiki/_export/xhtml/john/benchmarks
- Lesson both ways: bcrypt throughput is bound by *dependent, L1-resident,
  scalar* S-box loads; lane count and gather quality both matter less than
  hiding that load latency with independent instances, and L1 capacity caps
  how many instances fit (16 KiB of S-boxes per 4 lanes in SoA layout).

wasm bcrypt numbers in the wild (all scalar C/Rust compiled to wasm; no
SIMD128 bcrypt port besides ours was found — hashcat has no wasm port at all):
- `bcrypt-wasm` crate: ~835 h/s at cost 4 (older laptop, native-compiled
  numbers; wasm typically lands within ~10-30% of that for compute code).
  https://crates.io/crates/bcrypt-wasm/
- `hash-wasm` (hand-tuned C -> wasm): ~50-60 h/s at cost 8 in browsers
  (benchmark page: https://daninet.github.io/hash-wasm-benchmark/);
  `@blackberry/bcrypt` wasm ~59 h/s at cost 8, ~15 h/s at cost 10.
- Cloudflare Workers: wasm bcrypt runs (V8, SIMD enabled) but published
  numbers are sparse — ~50-100 ms at cost 10 expected, i.e. same class as
  other wasm scalar ports; Workers-specific forks (cf-hash-wasm) exist
  because `WebAssembly.compile()` of streaming source is disallowed, not
  because of SIMD.
  https://github.com/7Hazard/cf-hash-wasm ,
  https://github.com/Daninet/hash-wasm/discussions/56
- General wasm-vs-native SIMD gap for context: SIMD128 typically lands within
  5-50% of equivalent 128-bit native code (vs ~1.45-1.55x average scalar-wasm
  gap in "Not So Fast", arXiv:1901.09056); libsodium-on-Wasmtime ~1.46x
  native; V8 post-warmup often ~88-94% of native, Wasmtime ~79-82%.
  https://arxiv.org/html/1901.09056v3 ,
  https://www.hostmycode.com/blog/webassembly-runtime-performance-analysis-v8-wasmtime-wamr-benchmarks-production-deployments-2026

## 5. memory64, threads, arm64 lowering: what would actually move the number

- **memory64: irrelevant-to-negative.** bcrypt's working set is ~16.5 KiB per
  instance. memory32 rides free on guard-page reservations (no check
  instructions at all); memory64 forces explicit bounds checks in
  Wasmtime/Cranelift and hybrid checks in V8 (V8 saw ~10% slowdowns in early
  wasm64-vs-wasm32 measurements). Zero address-space need, nonzero cost. Skip.
  https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html ,
  https://github.com/WebAssembly/memory64/issues/31
- **Threads: the real batch multiplier, orthogonal to SIMD.** Hashes are
  independent, so N workers ~= Nx throughput. Wasmtime: host threads, one
  instance each — no shared memory needed (note `shared_memory` is off by
  default and wasi-threads was removed in Wasmtime 47; spawn from the host).
  Browsers: Web Workers + `SharedArrayBuffer` require cross-origin isolation —
  COOP/COEP (`require-corp`/`credentialless`) everywhere, or Chrome 137+'s
  simpler Document Isolation Policy; `Atomics.wait` still banned on the main
  thread. Deploy cost is real but the payoff is linear, unlike anything else
  on this list.
  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer ,
  https://developer.chrome.com/blog/document-isolation-policy ,
  https://docs.rs/wasmtime/latest/wasmtime/struct.SharedMemory.html ,
  https://bytecodealliance.org/articles/wasi-threads
- **arm64 lowering quality: not the bottleneck.** Extract -> UMOV, shuffles ->
  single NEON ops or TBL2, add/xor/shr 1:1. Wasmtime may leave ~10-20% on the
  table vs V8 TurboFan on SIMD-heavy aarch64 code through scheduling/moves
  (issue 6159 above), but the kernel is load-latency-bound, so the realistic
  engine-to-engine delta on *this* kernel is smaller than that.

## 6. Ceiling analysis: is 1.21x near it?

Cost accounting per F() per 4-lane group (all engines, either ISA):
- SIMD domain: ~10 cheap v128 ops (2 shr, 3 and, 2 add, 1 xor, splats hoisted)
  — these collapse ~24 scalar ALU ops into ~10 and are effectively free.
- The gathers: 4 x (v128.store + 4 scalar loads + address math + v128.load)
  = 16 table loads + 8 domain-crossing memory ops + ~16-24 scalar address ops.

The 16 scalar table loads are irreducible — scalar bcrypt does exactly those
same 16 loads for 4 hashes. So SIMD's gross win is only the ALU collapse
(~24 -> ~10 ops), and its gross cost is the ~30+ extra crossing/address ops.
On paper that is near break-even; the measured 1.21x comes from (a) the 16
independent loads per F exposing excellent memory-level parallelism on wide
cores (M5 Max sustains 3-4 loads/cycle), and (b) fast contained store-to-load
forwarding on Apple silicon. The serial round-to-round dependency (64 rounds
x F) means load+forward latency is the critical path, not throughput.

Cross-checking against native ceilings for the identical shape:
- 4-lane with *hardware* gather: 1.33x (pbcrypt). We get 1.21x with an
  *emulated* gather. Gap to the native 4-lane shape: ~9%.
- 8-lane with hardware gather: ~1.55x (JtR) to 2.75x vs a weaker scalar base
  (pbcrypt-8) — but JtR's gather code lost to scalar X2 interleave outright.
Conclusion: **for the 4-lane lockstep shape, ~1.2-1.35x IS the ceiling,
independent of engine or lookup spelling.** 1.21x is ~90-95% of it. No
byte-shuffle transposition, bitslice, relaxed-simd op, memory-layout change,
or memory64/threads tweak changes the shape's ceiling — the first three are
proven dead ends (sections 1-2), the layout question is moot (section 3).

The one shape that demonstrably beats this ceiling is **more independent work
per kernel invocation**: JtR X2/X3 interleaved scalar beat AVX2 gather by
hiding load latency behind a second instance's independent dependency chain.
Transplanted to wasm: one kernel running 8 lanes as two interleaved v128
states (X2-in-vector), optionally 16 lanes (X4) where L1d allows
(2 x 16 KiB SoA S-boxes fits M-series' 128 KiB L1d easily; x86 32-48 KiB L1d
caps you at ~8 lanes — the 2013 JtR cache lesson). Expected: 1.5-1.9x over
scalar-in-wasm, i.e. recovering most of what JtR's X2 gets natively. This is
a kernel-shape change, not a lookup trick — which is exactly the point.

## 7. What V8 would do differently (estimate, no local harness)

TurboFan will emit the same shape as Cranelift for `lookup4`: `str q0` to the
stack slot, 4x `ldr` (arm64) with folded address adds, `ldr q0` back — its
load elimination, like Cranelift's, does not forward across store/load width
mismatches, so the microarchitecture does the forwarding on both engines.
Extract-lane spelling would be 4x `umov` + inserts; TurboFan's better
scheduler/regalloc may win a few percent on the surrounding vector chain
(V8 leads Cranelift ~10-20% on SIMD-heavy aarch64 suites, issue 6159), and
its SIMD->GPR crossing handling is mature. Expected V8 result for this
kernel: **same 1.2x ratio, plausibly stretching to ~1.3x**, no qualitative
difference. Two V8-specific caveats: (1) tiering — Liftoff compiles first and
its naive per-op codegen with v128 spills is meaningfully slower; a cost-4
batch-8 run is ~5 ms total, so short browser sessions may measure mostly
Liftoff. Warm up or benchmark at cost >= 10. (2) Liftoff vs TurboFan delta is
much larger than any spelling delta — don't A/B lookup spellings on cold code.

## 8. Ranked experiments

1. **X2-in-vector kernel (8 lanes, two interleaved v128 states per kernel).**
   The only change with native evidence of beating the 4-lane ceiling (JtR X2
   > AVX2-gather). Hides store-forward + table-load latency behind a second
   independent chain. Cheap: replicate state, interleave the round stream.
   Watch L1d on x86 targets (cap at X2 there; X4 plausible on M-series).
2. **Threads for batch** (orthogonal, near-linear): host threads under
   wasmtime; Web Workers + DIP/COOP-COEP in browsers. Biggest real-world
   multiplier available; no kernel work.
3. **extract_lane spelling A/B** (`i32x4_extract_lane` x4 + `replace_lane`
   rebuild vs stack round-trip): on arm64 UMOV chains may edge out the
   store-forward latency; on V8/x64 PEXTRD is similar. Expected +-5-10% —
   measure on both engines, warm, before adopting.
4. **Scalar-F hybrid:** cross to scalar domain once per round (store `x`,
   4 loads), do byte extraction with scalar shifts (free), 16 table loads,
   scalar add/xor combine, one store + `v128.load` back; keep SIMD only for
   the P-xor and expansion streaming writes. Trades 16 crossings for ~8;
   plausible small win on narrow engines, likely a wash on M5 Max.
5. **Verify load scheduling:** dump wasmtime codegen (`--emit-clif` /
   objdump the compiled cwasm) and confirm all 16 table loads issue before
   the combine tree; if Cranelift interleaves them serially, hand-order via
   gathering into an array before combining (current code already does).
6. **Skip list (with reasons):** relaxed-simd flavor (nothing applicable;
    Safari validation hazard); memory64 (adds bounds checks, no need);
    byte-shuffle transposed lookup (16-byte table ceiling, proven dead end);
    bitslicing (key-dependent arbitrary LUT, circuit re-derived per key);
    S-box "in linear memory" relayout (already linear memory; alias regions
    already disjoint).

## Summary

- The 1.21x is ~90-95% of the ceiling for the 4-lane lockstep shape; native
  4-lane bcrypt *with a hardware gather* only reaches 1.33x.
- wasm has no gather, no >16-byte runtime table op, and no algebraic structure
  in Blowfish S-boxes to exploit — the stack round-trip is the right spelling,
  and relaxed-simd/memory64/layout changes have nothing to offer this kernel.
- The proven lever past the ceiling is interleaving independent instances
  (JtR X2/X3 beat AVX2 gather natively); as lanes in one kernel, that is the
  X2-in-vector experiment. Threads multiply batch throughput linearly.
- V8 will land in the same place (est. 1.2-1.3x); watch Liftoff tiering on
  short runs.
