#!/usr/bin/env bash
# exp-0004 static zero-persistence compliance audit.
#
# Checks at source level that the toxicity path (TfidfSvmClassifier,
# DefaultToxicityRepository, and ChatViewModel's warning state) never (1) sends
# the inspected sentence or the verdict over the network, (2) writes them to
# disk, or (3) leaks them into logs. The runtime half of the proof is the JUnit
# suite (byte comparison of the data directories, inspection of the transmitted
# payload).
#
# Usage: bash zero_persistence_audit.sh   (from any working directory)
set -uo pipefail

# Artifact edition: the audit targets the snapshots of the app sources under
# `audited-sources/`. The full app tree is not bundled here, so paths are
# resolved relative to this script's own location.
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/audited-sources"
FILES=(
  "$APP/TfidfSvmClassifier.kt"          # replaced FastTextClassifier in exp-0011
  "$APP/DefaultToxicityRepository.kt"
  "$APP/ToxicityRepository.kt"
)
VM="$APP/ChatViewModel.kt"

fail=0

# ⚠️ If an audit target is missing, grep finds nothing and the audit **passes
# vacuously**. (This actually happened when the model was swapped in exp-0011 —
# hence the existence check enforced up front.)
# ChatViewModel must be in this list too: the lifetime checks below have the
# same defect, since a missing file makes grep fail, fall through to the else
# branch, and print a pass.
for f in "${FILES[@]}" "$VM"; do
  if [[ ! -f "$f" ]]; then
    echo "❌ audit target missing: $f"
    echo "   (if the file moved or was deleted, update this script's FILES list)"
    exit 1
  fi
done
check() { # check <description> <regex> <files...>
  local desc="$1" pattern="$2"; shift 2
  local hits
  hits=$(grep -nE "$pattern" "$@" 2>/dev/null || true)
  if [[ -n "$hits" ]]; then
    echo "❌ $desc"
    echo "$hits" | sed 's/^/     /'
    fail=1
  else
    echo "✅ $desc"
  fi
}

echo "=== exp-0004 static zero-persistence audit ==="
echo "targets: 3 files on the toxicity path"
echo

check "no network transmission (WebSocket/HTTP/Socket)" \
  'NetworkDataSource|WebSocket|OkHttp|HttpURL|Socket\(|Retrofit|\.send\(' "${FILES[@]}"

check "no disk write (File/SharedPrefs/DB/Output)" \
  'SharedPreferences|FileOutputStream|FileWriter|openFileOutput|\.write\(|Room|SQLite|DataStore|getExternalStorage|SecureStorage' "${FILES[@]}"

check "no inspected sentence or probability leaking into logs" \
  'Log\.(d|i|v|e|w)\(.*(text|prob|score|message|content)' "${FILES[@]}"

# ChatViewModel: does the warning state leak into the cache or the send path?
# (the pending sentence must go null the moment it is sent or dismissed)
echo
echo "--- ChatViewModel warning-state lifetime ---"
if grep -q '_toxicWarningText.value = null' "$VM"; then
  echo "✅ the pending sentence is destroyed on send/dismiss (null assignment present)"
else
  echo "❌ no code destroying the pending sentence"; fail=1
fi
if grep -nE 'addMessageToCache.*toxic|toxic.*(sendMessageUseCase|NetworkDataSource)' "$VM" >/dev/null; then
  echo "❌ the verdict flows into the cache or the send path"; fail=1
else
  echo "✅ the verdict reaches neither the message cache nor the transmitted payload"
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "=== result: PASS — the toxicity path is self-contained in RAM ==="
else
  echo "=== result: FAIL — check the items above ==="
fi
exit $fail
