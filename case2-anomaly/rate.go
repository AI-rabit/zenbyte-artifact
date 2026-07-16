package anomaly

import (
	"sync"
	"time"
)

// RateTracker는 "이벤트 발생" 단위의 호출을 초당 관측치로 변환해 Tracker에 넘긴다.
//
// 사용법 (릴레이 hot path):
//
//	if rt.IsBlocked(key) { drop }   // O(1), 잠금 최소
//	rt.Record(key)                  // 카운터 증가만
//
// 그리고 1초마다 rt.Tick(now)가 카운터를 관측치로 flush한다.
// 카운터는 flush 즉시 비워지므로 과거 트래픽이 축적되지 않는다 (Zero-Persistence).
type RateTracker struct {
	tracker *Tracker

	mu      sync.Mutex
	counts  map[string]float64 // 현재 틱의 키별 이벤트 수 (flush 시 비워짐)
	ceiling float64            // 틱 내 즉시 차단 상한 (0 = 비활성)
}

func NewRateTracker(cfg TrackerConfig) *RateTracker {
	return &RateTracker{
		tracker: NewTracker(cfg),
		counts:  make(map[string]float64),
		ceiling: cfg.Ceiling,
	}
}

// Record는 키의 이벤트 1건을 현재 틱에 기록한다. 절대 상한을 즉시 넘으면 true(차단)를 반환한다.
// 틱 경계를 기다리지 않고 폭주를 끊기 위한 인라인 백스톱이다.
func (rt *RateTracker) Record(key string) (overCeiling bool) {
	rt.mu.Lock()
	rt.counts[key]++
	c := rt.counts[key]
	rt.mu.Unlock()
	return rt.ceiling > 0 && c > rt.ceiling
}

// IsBlocked는 이전 틱들의 판정으로 현재 차단 중인지 확인한다 (상태 갱신 없음).
func (rt *RateTracker) IsBlocked(key string, now time.Time) bool {
	return rt.tracker.IsBlocked(key, now)
}

// Tick은 현재 틱의 카운터를 탐지기로 흘려보내고 카운터를 비운다. 1초마다 호출한다.
// 반환값: 이번 틱에 새로 차단된 키 수.
func (rt *RateTracker) Tick(now time.Time) int {
	rt.mu.Lock()
	counts := rt.counts
	rt.counts = make(map[string]float64, len(counts))
	rt.mu.Unlock()

	newlyBlocked := 0
	for key, x := range counts {
		if rt.tracker.Observe(key, x, now) {
			newlyBlocked++
		}
	}
	// 트래픽이 끊긴 키의 상태도 TTL로 소멸시킨다
	rt.tracker.Sweep(now)
	return newlyBlocked
}

func (rt *RateTracker) Stats(now time.Time) Stats { return rt.tracker.Stats(now) }
