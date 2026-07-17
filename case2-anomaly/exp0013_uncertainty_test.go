package anomaly

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// exp-0013 Phase C: 시뮬레이션 seed 분산 측정 (doc/research/exp-0013-teacher-server-variance/).
//
// exp-0005의 채택 파라미터 수치(오탐 0.2%, burst 즉시, ramp 20초, low-and-slow 미탐)는
// 고정 seed(정상 42, 공격 7) 단일 실행이었다. 본 하니스는 동일 프로토콜(TestSelectedConfig:
// 정상 500키 × 3600틱 × base 2.0/s, 공격 warmup 120틱 + 300틱)을 seed 30쌍
// (정상 1000+i, 공격 2000+i)으로 반복해 분포를 산출한다.
//
// 평상시 `go test ./...`에서는 skip된다 — ZENBYTE_EXP0013_OUT에 CSV 출력 경로를
// 지정했을 때만 실행 (기존 테스트 시간·CI 불변). 테스트 파일이므로 배포 바이너리와 무관.
func TestExp0013SeedSweep(t *testing.T) {
	out := os.Getenv("ZENBYTE_EXP0013_OUT")
	if out == "" {
		t.Skip("exp-0013 하니스: ZENBYTE_EXP0013_OUT 미설정 — skip")
	}

	const (
		seeds      = 30
		normalKeys = 500
		duration   = 3600
		base       = 2.0
	)

	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("출력 파일 생성 실패: %v", err)
	}
	defer f.Close()
	fmt.Fprintln(f, "i,normal_seed,attack_seed,fp_rate,burst_delay,ramp_delay,lowslow_delay")

	cfg := DefaultConfig()
	for i := 0; i < seeds; i++ {
		d := New(cfg)

		normalSeed := int64(1000 + i)
		r := rand.New(rand.NewSource(normalSeed))
		keysWithAlarm := 0
		for k := 0; k < normalKeys; k++ {
			if simulateNormal(d, r, base, duration) > 0 {
				keysWithAlarm++
			}
		}
		fp := float64(keysWithAlarm) / float64(normalKeys)

		attackSeed := int64(2000 + i)
		ra := rand.New(rand.NewSource(attackSeed))
		burst := simulateAttack(d, ra, base, "burst")
		ramp := simulateAttack(d, ra, base, "ramp")
		lowSlow := simulateAttack(d, ra, base, "low-and-slow")

		fmt.Fprintf(f, "%d,%d,%d,%.6f,%d,%d,%d\n", i, normalSeed, attackSeed, fp, burst, ramp, lowSlow)
		t.Logf("seed %2d: FP %.2f%% burst %ds ramp %ds low&slow %d", i, fp*100, burst, ramp, lowSlow)
	}
	t.Logf("→ %s 저장 (판정은 exp-0013 analyze_c.py — 사전 등록 기준)", out)
}
