#!/usr/bin/env bash
# De-JARVIS scrub guard (Phase 2 §6): the repository must not mention the
# owner's private deployment (JARVIS, Pithos, Cadmus, LAN IPs, host names).
#
#   scripts/scrub-check.sh      # exit 0 = clean, 1 = forbidden mentions found
#
# Scans every tracked + untracked-but-not-ignored file (text files only),
# case-insensitively, for the patterns below.
#
# Documented exceptions (each deliberate; see the team-lead ruling of Phase 2):
#   CHANGELOG.md                       history: records what was removed
#   tests/fixtures/                    the legacy v0.2 store fixture mirrors the
#                                      owner's real store, incl. its "jarvis" tenant
#   tests/legacy_fixture.rs            generates/guards that fixture
#   tests/store_migrations.rs          asserts the legacy "jarvis" tenant is merged (P2R-11)
#   tests/cli_workspace.rs             asserts no "jarvis" tenant survives import
#   tests/example_config.rs            the example-config guard's own needle list
#   scripts/scrub-check.sh             this file (its pattern list)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PATTERN='jarvis|pithos|192\.168\.|cadmus|mymini'
EXCEPTIONS=(
  'CHANGELOG.md'
  'tests/fixtures/'
  'tests/legacy_fixture.rs'
  'tests/store_migrations.rs'
  'tests/cli_workspace.rs'
  'tests/example_config.rs'
  'scripts/scrub-check.sh'
)

is_exception() {
  local f="$1" e
  for e in "${EXCEPTIONS[@]}"; do
    [[ "$f" == "$e" || "$f" == "$e"* && "$e" == */ ]] && return 0
  done
  return 1
}

hits=0
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  is_exception "$f" && continue
  if matches=$(grep -I -n -i -E "$PATTERN" -- "$f" 2>/dev/null); then
    while IFS= read -r m; do
      echo "$f:$m"
      hits=$((hits + 1))
    done <<<"$matches"
  fi
done < <(git ls-files -z --cached --others --exclude-standard)

if [ "$hits" -gt 0 ]; then
  echo "scrub-check: $hits forbidden mention(s) (pattern: $PATTERN)" >&2
  exit 1
fi
echo "scrub-check: clean"
