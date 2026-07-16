package anomaly

import (
	"math"
	"math/rand"
	"testing"
	"time"
	"unsafe"
)

// exp-0005 오프라인 시뮬레이션.
//
// 합성 트래픽으로 (Alpha, K, H)를 튜닝하고 탐지 성능·오탐율을 측정한다.
// 모든 시뮬레이션은 고정 시드로 재현 가능하다.
//
// 트래픽 모델: 1초 = 1틱. 각 틱마다 키별 이벤트 수를 관측한다.
//   - 정상: 일주기(낮/밤) 기저율 위의 포아송 잡음
//   - burst: 특정 시점부터 기저율의 20배로 급증
//   - ramp: 60초에 걸쳐 서서히 10배까지 증가
//   - low-and-slow: 기저율의 2배로 지속 (탐지 난이도 최상)

const tickSec = 1

// diurnalRate는 하루 주기의 기저 트래픽 수준을 반환한다 (낮 피크, 새벽 저점).
func diurnalRate(tick int, base float64) float64 {
	hour := float64(tick%86400) / 3600.0
	// 0.4 ~ 1.6배로 완만히 변동 (새벽 4시 최저, 오후 4시 최고)
	factor := 1.0 + 0.6*math.Sin(2*math.Pi*(hour-10)/24)
	return base * factor
}

func poisson(r *rand.Rand, lambda float64) float64 {
	if lambda <= 0 {
		return 0
	}
	// Knuth
	l := math.Exp(-lambda)
	k, p := 0, 1.0
	for {
		p *= r.Float64()
		if p <= l {
			return float64(k)
		}
		k++
		if k > 10000 {
			return float64(k)
		}
	}
}

// simulateNormal은 정상 키 하나를 duration 틱 동안 관측하고 경보 횟수를 센다.
func simulateNormal(d *Detector, r *rand.Rand, base float64, duration int) (alarms int) {
	s := &KeyState{}
	t0 := time.Now()
	for tick := 0; tick < duration; tick++ {
		x := poisson(r, diurnalRate(tick, base))
		if d.Observe(s, x, t0.Add(time.Duration(tick)*time.Second)) {
			alarms++
		}
	}
	return
}

// simulateAttack은 워밍업 후 공격을 주입하고, 주입 시점부터 첫 경보까지의 지연(초)을 반환한다.
// 미탐이면 -1.
func simulateAttack(d *Detector, r *rand.Rand, base float64, attack string) int {
	s := &KeyState{}
	t0 := time.Now()
	const warmup = 120 // 2분간 정상 트래픽으로 기준선 형성
	const attackDur = 300

	for tick := 0; tick < warmup; tick++ {
		d.Observe(s, poisson(r, diurnalRate(tick, base)), t0.Add(time.Duration(tick)*time.Second))
	}
	for i := 0; i < attackDur; i++ {
		tick := warmup + i
		rate := diurnalRate(tick, base)
		switch attack {
		case "burst":
			rate *= 20
		case "ramp":
			factor := 1.0 + 9.0*math.Min(1.0, float64(i)/60.0)
			rate *= factor
		case "low-and-slow":
			rate *= 2
		}
		if d.Observe(s, poisson(r, rate), t0.Add(time.Duration(tick)*time.Second)) {
			return i * tickSec
		}
	}
	return -1
}

// 베이스라인: 고정 임계 rate-limit (기저율의 3배 초과 시 경보)
func simulateFixedThreshold(r *rand.Rand, base float64, multiplier float64, duration int) (alarms int) {
	threshold := base * multiplier
	for tick := 0; tick < duration; tick++ {
		if poisson(r, diurnalRate(tick, base)) > threshold {
			alarms++
		}
	}
	return
}

// TestParameterSweep은 (Alpha, K, H) 격자에서 오탐율과 burst 탐지 지연을 측정한다.
func TestParameterSweep(t *testing.T) {
	const (
		normalKeys = 200  // 정상 키 수 (오탐율 모수)
		duration   = 3600 // 1시간 관측
		base       = 2.0  // 초당 평균 2 이벤트
	)
	type row struct {
		cfg           Config
		falsePositive float64
		burstDelay    int
		rampDelay     int
		lowSlowDelay  int
	}
	var rows []row

	for _, alpha := range []float64{0.1, 0.2, 0.3} {
		for _, k := range []float64{0.5, 1.0} {
			for _, h := range []float64{3.0, 5.0, 8.0} {
				cfg := Config{Alpha: alpha, K: k, H: h, MinSigma: 1.0, WarmupMin: 5}
				d := New(cfg)

				// 오탐: 정상 키 중 1회 이상 경보한 키의 비율
				r := rand.New(rand.NewSource(42))
				keysWithAlarm := 0
				for i := 0; i < normalKeys; i++ {
					if simulateNormal(d, r, base, duration) > 0 {
						keysWithAlarm++
					}
				}
				fp := float64(keysWithAlarm) / float64(normalKeys)

				ra := rand.New(rand.NewSource(7))
				rows = append(rows, row{
					cfg:           cfg,
					falsePositive: fp,
					burstDelay:    simulateAttack(d, ra, base, "burst"),
					rampDelay:     simulateAttack(d, ra, base, "ramp"),
					lowSlowDelay:  simulateAttack(d, ra, base, "low-and-slow"),
				})
			}
		}
	}

	t.Log("=== exp-0005 파라미터 스윕 (정상 200키 × 1h, base=2/s) ===")
	t.Log("alpha    K    H | 오탐율 | burst | ramp | low&slow  (지연 초, -1=미탐)")
	for _, r := range rows {
		t.Logf("%5.1f %4.1f %4.1f | %5.1f%% | %5d | %4d | %8d",
			r.cfg.Alpha, r.cfg.K, r.cfg.H, r.falsePositive*100,
			r.burstDelay, r.rampDelay, r.lowSlowDelay)
	}

	// 베이스라인 비교 (고정 임계)
	t.Log("--- 베이스라인: 고정 임계 rate-limit ---")
	for _, mult := range []float64{2.0, 3.0, 5.0} {
		rb := rand.New(rand.NewSource(42))
		keysWithAlarm := 0
		for i := 0; i < normalKeys; i++ {
			if simulateFixedThreshold(rb, base, mult, duration) > 0 {
				keysWithAlarm++
			}
		}
		fp := float64(keysWithAlarm) / float64(normalKeys)
		// 고정 임계의 탐지 지연: burst(20배)는 1틱, low-and-slow(2배)는 임계 미만이면 영구 미탐
		lowSlowDetect := "미탐"
		if base*2 > base*mult {
			lowSlowDetect = "즉시"
		}
		t.Logf("임계=%.0f× base | 오탐율 %5.1f%% | burst 즉시 | low&slow %s", mult, fp*100, lowSlowDetect)
	}
}

// TestSelectedConfig는 채택 파라미터가 성공 기준을 만족하는지 검증한다 (회귀 방지).
func TestSelectedConfig(t *testing.T) {
	cfg := DefaultConfig()
	d := New(cfg)

	const (
		normalKeys = 500
		duration   = 3600
		base       = 2.0
	)
	r := rand.New(rand.NewSource(42))
	keysWithAlarm := 0
	for i := 0; i < normalKeys; i++ {
		if simulateNormal(d, r, base, duration) > 0 {
			keysWithAlarm++
		}
	}
	fp := float64(keysWithAlarm) / float64(normalKeys)
	if fp > 0.05 {
		t.Errorf("오탐율 %.1f%% > 5%% (성공 기준 위반)", fp*100)
	}

	ra := rand.New(rand.NewSource(7))
	burst := simulateAttack(d, ra, base, "burst")
	if burst < 0 || burst > 3 {
		t.Errorf("burst 탐지 지연 %ds (기준: ≤3초)", burst)
	}

	ramp := simulateAttack(d, ra, base, "ramp")
	if ramp < 0 || ramp > 30 {
		t.Errorf("ramp 탐지 지연 %ds (기대: ≤30초)", ramp)
	}

	// low-and-slow는 구조적 미탐이 예상된다 — 이것이 바뀌면(탐지되면) 오히려 알림을 남긴다.
	lowSlow := simulateAttack(d, ra, base, "low-and-slow")
	if lowSlow >= 0 {
		t.Logf("참고: low-and-slow가 %ds에 탐지됨 (기존 기록은 미탐 — 재확인 필요)", lowSlow)
	}

	t.Logf("채택 파라미터 %+v", cfg)
	t.Logf("오탐율=%.1f%% burst=%ds ramp=%ds low&slow=%ds(-1=미탐, 알려진 한계)",
		fp*100, burst, ramp, lowSlow)
}

// TestAbsoluteCeiling은 low-and-slow의 구조적 미탐을 절대 상한이 보완함을 보인다.
// EWMA/CUSUM은 "평소 대비 이상"을 보고, 절대 상한은 "어떤 경우에도 넘으면 안 되는 선"을 지킨다.
// 두 장치는 상보적이며, exp-0006의 허브 통합에서 함께 적용한다.
func TestAbsoluteCeiling(t *testing.T) {
	d := New(DefaultConfig())
	const ceiling = 10.0 // 초당 10 이벤트: 정상 사용자가 결코 넘지 않는 선 (base=2)
	r := rand.New(rand.NewSource(11))
	t0 := time.Now()

	// 정상 트래픽은 절대 상한을 건드리지 않는다
	sNormal := &KeyState{}
	normalCeilingHits := 0
	for tick := 0; tick < 3600; tick++ {
		x := poisson(r, diurnalRate(tick, 2.0))
		d.Observe(sNormal, x, t0.Add(time.Duration(tick)*time.Second))
		if x > ceiling {
			normalCeilingHits++
		}
	}
	if normalCeilingHits > 0 {
		t.Errorf("정상 트래픽이 절대 상한을 %d회 초과 (상한이 너무 낮음)", normalCeilingHits)
	}

	// low-and-slow(2×)는 CUSUM이 놓치지만, 공격이 상한을 넘는 순간 절대 상한이 잡는다.
	// 2× = 초당 4 이벤트는 상한(10) 아래이므로 여전히 통과한다 — 즉 절대 상한도 만능이 아니다.
	// 이 실험이 증명하는 것: 두 장치의 사각지대가 서로 다르며, 겹치는 영역이 남는다는 사실 자체다.
	sSlow := &KeyState{}
	cusumCaught, ceilingCaught := false, false
	for i := 0; i < 300; i++ {
		x := poisson(r, 4.0) // 2× base
		if d.Observe(sSlow, x, t0.Add(time.Duration(i)*time.Second)) {
			cusumCaught = true
		}
		if x > ceiling {
			ceilingCaught = true
		}
	}
	t.Logf("low-and-slow(2×): CUSUM 탐지=%v, 절대상한(%.0f/s) 탐지=%v", cusumCaught, ceiling, ceilingCaught)
	t.Log("→ 결론: 2× 수준의 저속 남용은 두 장치 모두 통과한다. 이는 설계상 수용된 잔여 위험이며,")
	t.Log("  무기록 서버에서는 과거 트래픽 이력을 보관할 수 없어 장기 프로파일링으로 잡을 수도 없다.")
}

// TestBaselinePoisoning은 지속 공격이 기준선을 끌어올려 탐지를 무력화하지 못함을 확인한다.
func TestBaselinePoisoning(t *testing.T) {
	d := New(DefaultConfig())
	s := &KeyState{}
	r := rand.New(rand.NewSource(1))
	t0 := time.Now()

	for tick := 0; tick < 120; tick++ {
		d.Observe(s, poisson(r, 2.0), t0.Add(time.Duration(tick)*time.Second))
	}
	baselineBefore := s.Mean

	// 10분간 지속 공격 (20배)
	alarmTicks := 0
	for i := 0; i < 600; i++ {
		if d.Observe(s, poisson(r, 40.0), t0.Add(time.Duration(120+i)*time.Second)) {
			alarmTicks++
		}
	}

	if alarmTicks < 500 {
		t.Errorf("지속 공격 중 경보가 %d/600틱만 발생 — 기준선 오염 의심", alarmTicks)
	}
	if s.Mean > baselineBefore*2 {
		t.Errorf("기준선이 %.2f → %.2f로 오염됨 (공격 트래픽이 정상으로 학습됨)", baselineBefore, s.Mean)
	}
	t.Logf("지속 공격 600틱 중 %d틱 경보, 기준선 %.2f → %.2f (오염 없음)", alarmTicks, baselineBefore, s.Mean)
}

// TestRecoveryAfterRestart는 상태 전소(재시작) 후 재수렴 시간을 측정한다.
func TestRecoveryAfterRestart(t *testing.T) {
	d := New(DefaultConfig())
	r := rand.New(rand.NewSource(3))
	t0 := time.Now()

	// 재시작 직후: 빈 상태에서 정상 트래픽 관측 시작
	s := &KeyState{}
	falseAlarms := 0
	for tick := 0; tick < 30; tick++ {
		if d.Observe(s, poisson(r, 2.0), t0.Add(time.Duration(tick)*time.Second)) {
			falseAlarms++
		}
	}
	if falseAlarms > 0 {
		t.Errorf("재시작 직후 오경보 %d회 (워밍업이 보호하지 못함)", falseAlarms)
	}

	// 재수렴 확인: 30초 후 burst를 넣으면 즉시 잡히는가
	detected := -1
	for i := 0; i < 10; i++ {
		if d.Observe(s, poisson(r, 40.0), t0.Add(time.Duration(30+i)*time.Second)) {
			detected = i
			break
		}
	}
	if detected < 0 || detected > 3 {
		t.Errorf("재시작 30초 후 burst 탐지 실패/지연: %ds", detected)
	}
	t.Logf("재시작 후 30초 내 재수렴 완료 (오경보 0, burst 탐지 %ds)", detected)
}

// TestStateSizeIsConstant는 키당 상태가 스칼라 상수 크기임을 문서화한다 (exp-0006 전제).
func TestStateSizeIsConstant(t *testing.T) {
	var s KeyState
	size := int(unsafeSizeof(s))
	if size > 64 {
		t.Errorf("KeyState가 %d바이트 — 스칼라 상태 설계 위반 의심", size)
	}
	t.Logf("KeyState 크기: %d바이트 (키당 상수) — 메모리 상한 = 최대키수 × %d바이트", size, size)
}

func unsafeSizeof(s KeyState) uintptr {
	return unsafe.Sizeof(s)
}

// TestRefinedSweep은 1차 스윕에서 드러난 경계(H≈8)를 정밀 탐색한다.
// 목표: 오탐 ≤5%를 지키면서 ramp(완만한 상승)를 놓치지 않는 지점이 있는가?
func TestRefinedSweep(t *testing.T) {
	const (
		normalKeys = 200
		duration   = 3600
		base       = 2.0
	)
	t.Log("=== 2차 정밀 스윕 ===")
	t.Log("alpha    K    H | 오탐율 | burst | ramp | low&slow")
	for _, alpha := range []float64{0.05, 0.1, 0.15, 0.2} {
		for _, k := range []float64{0.5, 0.75} {
			for _, h := range []float64{8.0, 10.0, 12.0, 15.0} {
				cfg := Config{Alpha: alpha, K: k, H: h, MinSigma: 1.0, WarmupMin: 5}
				d := New(cfg)
				r := rand.New(rand.NewSource(42))
				alarmed := 0
				for i := 0; i < normalKeys; i++ {
					if simulateNormal(d, r, base, duration) > 0 {
						alarmed++
					}
				}
				fp := float64(alarmed) / float64(normalKeys)
				ra := rand.New(rand.NewSource(7))
				t.Logf("%5.2f %4.2f %5.1f | %5.1f%% | %5d | %4d | %8d",
					alpha, k, h, fp*100,
					simulateAttack(d, ra, base, "burst"),
					simulateAttack(d, ra, base, "ramp"),
					simulateAttack(d, ra, base, "low-and-slow"))
			}
		}
	}
}
