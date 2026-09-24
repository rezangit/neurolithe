#!/usr/bin/env bash
# Local quality gate. There is no CI, so run this before every commit.
#
#   scripts/check.sh            # full gate: scrub guard, fmt, clippy + tests (default and kafka)
#   scripts/check.sh --quick    # scrub + fmt + clippy + tests for the default feature set only
#
# Every step runs through scripts/cargo.sh, which uses a native cargo if one is
# installed and falls back to the pinned rust:1.94 Docker image otherwise.
# All steps run even if one fails; the exit code is non-zero if any failed.
# Env: NL_TARGET (target subdir for the Docker build, default "check").
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CARGO="$ROOT/scripts/cargo.sh"
export NL_TARGET="${NL_TARGET:-check}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# A step whose command starts with "@" runs that script directly (not cargo).
STEPS=(
  "scrub|@scripts/scrub-check.sh"
  "fmt|fmt --check"
  "clippy (default)|clippy --all-targets -- -D warnings"
  "test (default)|test --no-fail-fast"
)
if [ "$QUICK" -eq 0 ]; then
  STEPS+=(
    "clippy (kafka)|clippy --all-targets --features kafka -- -D warnings"
    "test (kafka)|test --no-fail-fast --features kafka"
  )
fi

if [ -t 1 ]; then GREEN=$'\e[32m'; RED=$'\e[31m'; BOLD=$'\e[1m'; RESET=$'\e[0m'
else GREEN=""; RED=""; BOLD=""; RESET=""; fi

# Per-target log dir: concurrent gates (different NL_TARGET) never clobber each other.
LOG_DIR="$ROOT/target/check-logs/$NL_TARGET"
mkdir -p "$LOG_DIR"

results=()
failed=0
total_start=$SECONDS
cd "$ROOT"
for step in "${STEPS[@]}"; do
  name="${step%%|*}"
  args="${step#*|}"
  log="$LOG_DIR/$(printf '%s' "$name" | tr -cs 'a-z0-9' '-' | sed 's/-$//').log"
  start=$SECONDS
  if [[ "$args" == @* ]]; then
    echo "${BOLD}==> $name${RESET}  (${args#@})"
    "$ROOT/${args#@}" 2>&1 | tee "$log"
  else
    echo "${BOLD}==> $name${RESET}  (cargo $args)"
    # shellcheck disable=SC2086
    "$CARGO" $args 2>&1 | tee "$log"
  fi
  status=${PIPESTATUS[0]}
  dur=$((SECONDS - start))
  summary=""
  if [[ "$name" == test* ]]; then
    # Sum the "test result:" lines across all test binaries.
    summary=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | awk '
      /^test result:/ { for (i=1;i<=NF;i++) { if ($(i+1) ~ /^passed/) p+=$i; if ($(i+1) ~ /^failed/) f+=$i; if ($(i+1) ~ /^ignored/) g+=$i } }
      END { if (p+f+g>0) printf "%d passed, %d failed, %d ignored", p, f, g }')
  fi
  if [ "$status" -eq 0 ]; then
    results+=("${GREEN}PASS${RESET}  $(printf '%-17s' "$name") ${dur}s  $summary")
  else
    failed=$((failed + 1))
    results+=("${RED}FAIL${RESET}  $(printf '%-17s' "$name") ${dur}s  $summary  (log: ${log#$ROOT/})")
  fi
done

echo
echo "${BOLD}Quality gate summary${RESET} ($((SECONDS - total_start))s total)"
for r in "${results[@]}"; do echo "  $r"; done
if [ "$failed" -eq 0 ]; then
  echo "${GREEN}${BOLD}ALL CHECKS PASSED${RESET}"
else
  echo "${RED}${BOLD}$failed CHECK(S) FAILED${RESET}"
fi
exit $((failed > 0))
