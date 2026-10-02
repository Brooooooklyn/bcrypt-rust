#!/usr/bin/env bash
# codegen-check.sh — static codegen regression net for the SIMD bcrypt
# kernels. The crate's hot loops stay CORRECT when LLVM codegen drifts;
# they just get slower. This campaign was burned three times, and this
# script pins each fix by disassembling release objects (no CPU features
# needed, so any CI host runs it):
#
#   1. Gather re-formation (x86): LLVM re-writes the prescaled scalar-load
#      pattern inside the avx2/avx512 INSERT/EXTRACT flavor kernels back
#      into microcoded `vpgatherdd` (avx512 fell 786 -> 383 h/s when it
#      re-formed). Source fix: the `lane_load!` opaque-asm macro. Guard:
#      v0 mangling keeps the const-generic flavor tag in the symbol
#      (`bcrypt8Kh<N>_E`, N = the FLAVOR_* u8 read back out of the
#      source), so the guard counts `vpgather*` per flavor body: the
#      GATHER flavor must HAVE gathers (sanity), INSERT/EXTRACT flavors
#      must have NONE. The `lane_load!` marker itself is grep-pinned too.
#   2. Outlined F() helpers: LLVM outlined the F()-lookup helpers into
#      real calls — 16 `callq` per encipher in avx512. `#[inline(always)]`
#      + `#[target_feature]` is a hard error (rust#145574), so the fix was
#      textual macro-ization. Guard: zero `callq` inside every
#      encipher*/expand_state_*/encrypt_zero_chain_* body (the rounds
#      regions), plus a cap on call sites inside the bcryptN outer bodies
#      (transpose/memcpy/expand_state/encipher-loop calls are legit there;
#      HEAD max is 18, cap 24 — outlining adds >=16 at once).
#   3. NEON black_box round-trips: `black_box` lowers to a store->load
#      through the stack ON the critical chain (~64 str/ldr pairs per
#      encipher; fixed with register-identity asm). Legit spills make a
#      str/ldr pattern grep useless (305 sp-relative pairs in bcrypt_lanes
#      at HEAD), so the guard is an instruction-count BUDGET on the
#      rounds-bearing bodies, plus `bl` discipline.
#
# Calibration (HEAD @ 1511d2c, rustc 1.98.0, aarch64-apple-darwin objects;
# budgets carry 15% slack for toolchain drift):
#   neon::bcrypt_lanes          2887 insns, 19 bl -> budget 3320, bl cap 24
#     (the 19 bl: 15 panic_bounds_check + 2 bzero + 2 encrypt_zero_chain)
#   neon::encrypt_zero_chain     879 insns,  0 bl -> budget 1010, bl must stay 0
#   x86 bcryptN outer bodies: max 18 callq -> cap 24
#
# Why `-C linker-plugin-lto=no`: profile.release sets lto="thin", which
# makes cargo defer codegen and store LLVM BITCODE in the rlib — nothing
# to disassemble. Counteracting just that flag keeps opt-level=3 and
# codegen-units=1 and runs the same LLVM pipeline in-crate; the crate has
# zero dependencies, so thin LTO has nothing cross-crate to merge. The
# failure modes reproduce in this configuration: deleting lane_load!'s
# asm re-forms vpgatherdd in the Kh2/Kh3 bodies and this script FAILs.
#
# Usage: ci/codegen-check.sh [aarch64|x86_64|all]  (default: all buildable)
# Env:   CODEGEN_CHECK_ASM_AARCH64 / CODEGEN_CHECK_ASM_X86_64 — inspect a
#        prebuilt `llvm-objdump -d` dump instead of building (debugging,
#        negative tests).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

want=${1:-all}
case "$want" in aarch64|x86_64|all) ;;
  *) echo "usage: $0 [aarch64|x86_64|all]" >&2; exit 2 ;;
esac

host=$(rustc -vV | sed -n 's/host: //p')

# llvm-objdump from the llvm-tools rustup component, PATH as fallback.
sysroot=$(rustc --print sysroot)
OBJDUMP="$sysroot/lib/rustlib/$host/bin/llvm-objdump"
if [ ! -x "$OBJDUMP" ]; then
  rustup component list --installed | grep -q llvm-tools \
    || rustup component add llvm-tools
fi
if [ ! -x "$OBJDUMP" ]; then
  if command -v llvm-objdump >/dev/null 2>&1; then
    OBJDUMP=llvm-objdump
  else
    echo "FATAL: llvm-objdump not found (llvm-tools component missing)" >&2
    exit 1
  fi
fi

# Own target dir: `cargo rustc` flags use a separate fingerprint, so this
# never invalidates the main target/ cache of concurrent work.
TD=${CARGO_TARGET_DIR:-$ROOT/target/codegen-check}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

failures=0
pass() { echo "PASS $1"; }
fail() { failures=$((failures + 1)); echo "FAIL $1 -- $2"; }

# sym_stats ASM SYMRE INSRE: one "<name>\t<insns>\t<hits>" line per symbol
# whose label matches SYMRE; insns = body instruction lines, hits = those
# matching INSRE (mnemonic-anchored regex). Portable awk (no \b, no gawk).
sym_stats() {
  awk -v sym="$2" -v ins="$3" '
    $0 ~ "^[0-9a-f]+ <[^>]*" sym "[^>]*>:$" {
      if (inbody) printf "%s\t%d\t%d\n", name, n, h
      inbody = 1; name = $2; gsub(/[<>]/, "", name); sub(/:$/, "", name)
      n = 0; h = 0; next
    }
    inbody && /^[0-9a-f]+ </ { printf "%s\t%d\t%d\n", name, n, h; inbody = 0 }
    inbody && /^ *[0-9a-f]+:/ { n++; if ($0 ~ ins) h++ }
    END { if (inbody) printf "%s\t%d\t%d\n", name, n, h }
  ' "$1"
}

# check: vpgather confined to the FLAVOR_GATHER bodies (guard 1).
check_x86_gathers() {
  local asm=$1 bad=0 mod pair kern g name n h flav seen
  for mod in avx2 avx512; do
    if ! grep -q 'macro_rules! lane_load' "src/eks/$mod.rs"; then
      fail "x86-gathers" "src/eks/$mod.rs lost the lane_load! opaque-asm blocker"
      bad=1
    elif ! sed -n '/macro_rules! lane_load/,/^}/p' "src/eks/$mod.rs" | grep -q 'asm!'; then
      fail "x86-gathers" "lane_load! in src/eks/$mod.rs lost its asm! identity"
      bad=1
    fi
  done
  for pair in bcrypt8:avx2 bcrypt16:avx512; do
    kern=${pair%%:*}; mod=${pair##*:}
    g=$(sed -n 's/^const FLAVOR_GATHER: u8 = \([0-9][0-9]*\);.*/\1/p' "src/eks/$mod.rs" | head -1)
    if [ -z "$g" ]; then
      fail "x86-gathers" "FLAVOR_GATHER const gone from src/eks/$mod.rs (flavor scheme changed?)"
      bad=1; continue
    fi
    seen=0
    while IFS=$'\t' read -r name n h; do
      [ -n "$name" ] || continue
      seen=$((seen + 1))
      flav=$(printf '%s' "$name" | sed -n "s/.*${kern}Kh\([0-9][0-9]*\)_.*/\1/p")
      if [ "$flav" = "$g" ]; then
        [ "$h" -gt 0 ] || { fail "x86-gathers" "$kern gather flavor Kh$g has no vpgather (symbol mix-up?)"; bad=1; }
      elif [ "$h" -ne 0 ]; then
        fail "x86-gathers" "$kern flavor Kh$flav: $h vpgather re-formed (lane_load! defeated)"
        bad=1
      fi
    done < <(sym_stats "$asm" "${kern}Kh" ':[[:space:]]+vpgather')
    [ "$seen" -ge 2 ] || { fail "x86-gathers" "$seen $kern flavor symbols found (need >= 2)"; bad=1; }
  done
  [ "$bad" -eq 0 ] && pass "x86-gathers: vpgather only in FLAVOR_GATHER bodies; lane_load! intact"
  return 0
}

# check: no calls inside the rounds regions (guard 2).
check_x86_calls() {
  local asm=$1 bad=0 matched=0 cap=24 name n h
  while IFS=$'\t' read -r name n h; do
    [ -n "$name" ] || continue
    matched=$((matched + 1))
    if [ "$h" -ne 0 ]; then
      fail "x86-calls" "$name: $h callq in a rounds body (F()-helper outlined; rust#145574 regression)"
      bad=1
    fi
  done < <(sym_stats "$asm" 'encipher(8|16)Kh|expand_state_v(8|16)Kh|encrypt_zero_chain_v(8|16)Kh' ':[[:space:]]+callq?[[:space:]]')
  [ "$matched" -ge 4 ] || { fail "x86-calls" "only $matched rounds-body symbols found (need >= 4)"; bad=1; }
  while IFS=$'\t' read -r name n h; do
    [ -n "$name" ] || continue
    if [ "$h" -gt "$cap" ]; then
      fail "x86-calls" "$name: $h callq > cap $cap (helpers outlined into the outer kernel)"
      bad=1
    fi
  done < <(sym_stats "$asm" 'bcrypt(8|16)Kh' ':[[:space:]]+callq?[[:space:]]')
  [ "$bad" -eq 0 ] && pass "x86-calls: no callq in encipher/expand/zero-chain bodies; bcryptN within cap $cap"
  return 0
}

# checks 3: NEON bl discipline + instruction budgets (see calibration above).
NEON_LANES_BUDGET=3320 NEON_CHAIN_BUDGET=1010 NEON_BL_CAP=24
check_neon_calls() {
  local asm=$1 bad=0 name n h
  while IFS=$'\t' read -r name n h; do
    [ -n "$name" ] || continue
    case "$name" in
      *neon18encrypt_zero_chain*)
        [ "$h" -eq 0 ] || { fail "neon-calls" "encrypt_zero_chain: $h bl (call on the rounds path)"; bad=1; } ;;
      *neon12bcrypt_lanes*)
        [ "$h" -le "$NEON_BL_CAP" ] || { fail "neon-calls" "bcrypt_lanes: $h bl > cap $NEON_BL_CAP (calibrated 19)"; bad=1; } ;;
    esac
  done < <(sym_stats "$asm" 'neon1[28](bcrypt_lanes|encrypt_zero_chain)' ':[[:space:]]+bl[[:space:]]')
  [ "$bad" -eq 0 ] && pass "neon-calls: 0 bl in encrypt_zero_chain; bcrypt_lanes within bl cap $NEON_BL_CAP"
  return 0
}
check_neon_budget() {
  local asm=$1 bad=0 matched=0 name n h
  while IFS=$'\t' read -r name n h; do
    [ -n "$name" ] || continue
    matched=$((matched + 1))
    case "$name" in
      *neon18encrypt_zero_chain*)
        [ "$n" -le "$NEON_CHAIN_BUDGET" ] || { fail "neon-budget" "encrypt_zero_chain: $n insns > budget $NEON_CHAIN_BUDGET (calibrated 879)"; bad=1; } ;;
      *neon12bcrypt_lanes*)
        [ "$n" -le "$NEON_LANES_BUDGET" ] || { fail "neon-budget" "bcrypt_lanes: $n insns > budget $NEON_LANES_BUDGET (calibrated 2887; stack round-trips back?)"; bad=1; } ;;
    esac
  done < <(sym_stats "$asm" 'neon1[28](bcrypt_lanes|encrypt_zero_chain)' ':nomatch:')
  [ "$matched" -eq 2 ] || { fail "neon-budget" "$matched neon hot symbols found (need 2)"; bad=1; }
  [ "$bad" -eq 0 ] && pass "neon-budget: bcrypt_lanes <= $NEON_LANES_BUDGET, encrypt_zero_chain <= $NEON_CHAIN_BUDGET insns"
  return 0
}

buildable() {
  case "$1:$host" in
    aarch64:aarch64-* | x86_64:x86_64-* | x86_64:aarch64-apple-darwin) return 0 ;;
    *) return 1 ;;
  esac
}

# Build the release lib and print its disassembly. cargo output goes to
# stderr so stdout is pure objdump text.
build_leg() {
  local leg=$1 triple= rlib
  if [ "$leg:$host" = x86_64:aarch64-apple-darwin ]; then
    triple=x86_64-apple-darwin
    rustup target list --installed | grep -qx "$triple" || rustup target add "$triple" >&2
    CARGO_TARGET_DIR="$TD" cargo rustc --release --lib --target "$triple" -- -C linker-plugin-lto=no >&2
    rlib=$TD/$triple/release/libbcrypt_rust.rlib
  else
    CARGO_TARGET_DIR="$TD" cargo rustc --release --lib -- -C linker-plugin-lto=no >&2
    rlib=$TD/release/libbcrypt_rust.rlib
  fi
  [ -f "$rlib" ] || { echo "FATAL: $rlib not produced" >&2; exit 1; }
  "$OBJDUMP" -d --no-show-raw-insn "$rlib" 2>/dev/null
}

run_leg() {
  local leg=$1 asm var
  var=CODEGEN_CHECK_ASM_$(printf '%s' "$leg" | tr '[:lower:]' '[:upper:]')
  if [ -n "${!var:-}" ]; then
    asm=${!var}
    [ -f "$asm" ] || { echo "FATAL: \$$var=$asm missing" >&2; exit 1; }
    echo "== $leg: inspecting prebuilt dump $asm"
  else
    asm=$WORK/$leg.asm
    echo "== $leg: building release objects (host $host)"
    build_leg "$leg" > "$asm"
  fi
  case "$leg" in
    aarch64) check_neon_calls "$asm"; check_neon_budget "$asm" ;;
    x86_64) check_x86_gathers "$asm"; check_x86_calls "$asm" ;;
  esac
}

legs=aarch64; [ "$want" = all ] && legs="aarch64 x86_64"; [ "$want" = x86_64 ] && legs=x86_64
ran=0
for leg in $legs; do
  var=CODEGEN_CHECK_ASM_$(printf '%s' "$leg" | tr '[:lower:]' '[:upper:]')
  if [ -z "${!var:-}" ] && ! buildable "$leg"; then
    echo "SKIP $leg: host $host cannot produce $leg objects"
    continue
  fi
  ran=$((ran + 1))
  run_leg "$leg"
done
[ "$ran" -ge 1 ] || { echo "FATAL: no leg runnable on host $host" >&2; exit 1; }
if [ "$failures" -gt 0 ]; then
  echo "codegen-check: $failures check(s) FAILED" >&2
  exit 1
fi
echo "codegen-check: all checks passed"
