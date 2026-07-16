// Package anomaly는 EWMA + CUSUM 기반 트래픽 이상탐지를 제공한다 (exp-0005).
//
// 설계 원칙 (Zero-Persistence):
//   - 키당 상태는 스칼라 4개뿐 (평균·분산·누적합·마지막 관측 시각). 과거 관측치를 보존하지 않는다.
//     → 트래픽 기록의 축적 자체가 메타데이터 영속화이므로 금지된다.
//   - 이벤트당 O(1) 갱신. 릴레이 hot path에 들어갈 수 있다.
//   - 상태 전소(서버 재시작)를 전제로 설계 — 워밍업 후 수 초 내 재수렴한다.
package anomaly

import (
	"math"
	"time"
)

// KeyState는 키(IP·공개키) 하나의 탐지 상태다. 스칼라만 보유한다.
type KeyState struct {
	Mean     float64   // EWMA 기준선 (정상 수준)
	Variance float64   // EWMA 분산 추정 (정규화용)
	CUSUM    float64   // 누적합 (작지만 지속적인 상승을 축적)
	Samples  int       // 워밍업 카운터
	Alarming bool      // 현재 경보 중인가
	LastSeen time.Time // TTL 만료 판정용 (exp-0006)
}

// Config는 탐지기 파라미터다.
type Config struct {
	Alpha     float64 // EWMA 평활 계수 (0.2~0.3 권장): 클수록 최근값에 민감
	K         float64 // CUSUM 허용 편차 (표준편차 배수): 잡음을 흡수하는 여유
	H         float64 // CUSUM 경보 임계값 (누적합이 이를 넘으면 이상)
	MinSigma  float64 // 표준편차 하한 (0 분산 시 0으로 나누기 방지 + 과민 억제)
	WarmupMin int     // 최소 관측 수 — 그전에는 경보하지 않는다 (재시작 직후 오탐 방지)
}

// DefaultConfig는 exp-0005 스윕에서 선정된 파라미터다.
//
// 선정 근거 (정상 200키 × 1시간, base 2 events/s):
//
//	오탐율 0.0% · burst(20×) 즉시 탐지 · ramp(60초에 걸쳐 10×) 20초 탐지.
//
// Alpha를 0.1로 낮춘 것이 핵심 — 기준선이 느리게 움직여야 완만한 상승(ramp)을 흡수하지 않는다.
//
// 알려진 한계: low-and-slow(기저율의 2배로 지속)는 **탐지하지 못한다**.
// 적응형 기준선이 이를 "새로운 정상"으로 학습하기 때문이며, 고정 임계 rate-limit도 동일하게 놓친다
// (2× < 임계 3×). 이 계열 공격은 AbsoluteCeiling(절대 상한)으로 방어해야 한다.
func DefaultConfig() Config {
	return Config{Alpha: 0.1, K: 0.5, H: 10.0, MinSigma: 1.0, WarmupMin: 5}
}

// Detector는 상태를 갖지 않는다 (파라미터만). 상태는 호출자가 키별로 보관한다.
type Detector struct {
	cfg Config
}

func New(cfg Config) *Detector { return &Detector{cfg: cfg} }

// Observe는 키의 새 관측치 x(예: 이번 틱의 이벤트 수)를 반영하고 경보 여부를 반환한다.
//
// 핵심 설계: 경보 중에는 기준선(Mean/Variance)을 갱신하지 않는다.
// 공격 트래픽으로 기준선이 끌려 올라가면 "새로운 정상"이 되어 탐지가 무력화되기 때문이다
// (baseline poisoning 방지).
func (d *Detector) Observe(s *KeyState, x float64, now time.Time) bool {
	s.LastSeen = now

	// 워밍업: 기준선이 설 때까지는 경보하지 않고 관측만 축적한다.
	if s.Samples < d.cfg.WarmupMin {
		if s.Samples == 0 {
			s.Mean = x
			s.Variance = 0
		} else {
			delta := x - s.Mean
			s.Mean += d.cfg.Alpha * delta
			s.Variance = (1 - d.cfg.Alpha) * (s.Variance + d.cfg.Alpha*delta*delta)
		}
		s.Samples++
		return false
	}

	sigma := math.Sqrt(s.Variance)
	if sigma < d.cfg.MinSigma {
		sigma = d.cfg.MinSigma
	}

	// 표준화 잔차: 기준선 대비 몇 시그마 위인가 (상방 단측 — 트래픽 급증만 관심)
	z := (x - s.Mean) / sigma

	// CUSUM: 허용 편차 K를 넘는 부분만 누적. 음수는 0으로 잘라 과거 여유가 쌓이지 않게 한다.
	s.CUSUM = math.Max(0, s.CUSUM+z-d.cfg.K)

	if s.CUSUM > d.cfg.H {
		s.Alarming = true
		return true // 경보 중: 기준선 갱신 안 함 (poisoning 방지)
	}

	// 정상 복귀
	s.Alarming = false
	delta := x - s.Mean
	s.Mean += d.cfg.Alpha * delta
	s.Variance = (1 - d.cfg.Alpha) * (s.Variance + d.cfg.Alpha*delta*delta)
	return false
}

// Reset은 경보 해제 시 누적합을 비운다 (rate-limit 해제 후 재출발용, exp-0006).
func (d *Detector) Reset(s *KeyState) {
	s.CUSUM = 0
	s.Alarming = false
}
