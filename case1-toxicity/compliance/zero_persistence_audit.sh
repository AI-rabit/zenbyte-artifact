#!/usr/bin/env bash
# exp-0004 무기록 준수 정적 감사.
#
# 독성 탐지 경로(TfidfSvmClassifier, DefaultToxicityRepository, ChatViewModel 경고 상태)가
# 검사 문장이나 판정 결과를 (1) 네트워크로 내보내거나 (2) 디스크에 쓰거나 (3) 로그로 흘리지
# 않는지 소스 수준에서 확인한다. 런타임 증명은 JUnit(디스크 바이트 비교·전송 페이로드 검사)이 담당.
#
# 사용: bash zero_persistence_audit.sh   (작업 디렉터리 무관)
set -uo pipefail

# 아티팩트 판: 감사 대상은 `audited-sources/`에 담긴 앱 소스 스냅샷이다
# (앱 트리 전체를 동봉하지 않으므로, 경로는 이 스크립트 위치를 기준으로 잡는다).
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/audited-sources"
FILES=(
  "$APP/TfidfSvmClassifier.kt"          # exp-0011에서 FastTextClassifier를 대체
  "$APP/DefaultToxicityRepository.kt"
  "$APP/ToxicityRepository.kt"
)
VM="$APP/ChatViewModel.kt"

fail=0

# ⚠️ 감사 대상 파일이 없으면 grep이 아무것도 찾지 못해 **공허하게 통과**한다.
# (exp-0011의 모델 교체 때 실제로 발생했던 결함 — 존재 검사를 먼저 강제한다.)
# ChatViewModel도 반드시 이 목록에 포함한다: 아래 수명 검사는 파일이 없으면
# grep이 실패해 else 분기로 빠지면서 ✅를 출력하는 같은 결함을 갖는다.
for f in "${FILES[@]}" "$VM"; do
  if [[ ! -f "$f" ]]; then
    echo "❌ 감사 대상 파일 없음: $f"
    echo "   (파일이 이동·삭제되었다면 이 스크립트의 FILES 목록을 갱신할 것)"
    exit 1
  fi
done
check() { # check <설명> <정규식> <파일들...>
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

echo "=== exp-0004 무기록 정적 감사 ==="
echo "대상: 독성 탐지 경로 3개 파일"
echo

check "네트워크 전송 없음 (WebSocket/HTTP/Socket)" \
  'NetworkDataSource|WebSocket|OkHttp|HttpURL|Socket\(|Retrofit|\.send\(' "${FILES[@]}"

check "디스크 쓰기 없음 (File/SharedPrefs/DB/Output)" \
  'SharedPreferences|FileOutputStream|FileWriter|openFileOutput|\.write\(|Room|SQLite|DataStore|getExternalStorage|SecureStorage' "${FILES[@]}"

check "검사 문장·확률의 로그 유출 없음" \
  'Log\.(d|i|v|e|w)\(.*(text|prob|score|message|content)' "${FILES[@]}"

# ChatViewModel: 경고 상태가 캐시/전송 경로로 새지 않는지 (대기 문장은 전송 또는 취소 시 즉시 null)
echo
echo "--- ChatViewModel 경고 상태 수명 ---"
if grep -q '_toxicWarningText.value = null' "$VM"; then
  echo "✅ 대기 문장이 전송/취소 시 즉시 파기됨 (null 대입 존재)"
else
  echo "❌ 대기 문장 파기 코드 없음"; fail=1
fi
if grep -nE 'addMessageToCache.*toxic|toxic.*(sendMessageUseCase|NetworkDataSource)' "$VM" >/dev/null; then
  echo "❌ 판정 결과가 캐시/전송 경로로 유입됨"; fail=1
else
  echo "✅ 판정 결과가 메시지 캐시·전송 페이로드에 포함되지 않음"
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "=== 결과: 통과 — 독성 탐지 경로는 RAM 안에서 완결됨 ==="
else
  echo "=== 결과: 실패 — 위 항목 확인 필요 ==="
fi
exit $fail
