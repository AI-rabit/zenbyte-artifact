package anomaly

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// TestMemoryBoundUnderKeyFlood는 exp-0006의 핵심 증명이다.
//
// 10만 개의 상이한 키를 주입해도 추적 상태가 LRU 상한에서 고정되며,
// 힙 사용량이 상한 × 엔트리크기 규모를 넘지 않음을 시계열로 보인다.
// (이 테스트가 실패하면 "DB는 없지만 메모리가 무한히 쌓이는" 새로운 영속화가 발생한 것이다.)
func TestMemoryBoundUnderKeyFlood(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.MaxKeys = 10_000
	tr := NewTracker(cfg)

	now := time.Now()
	const totalKeys = 100_000

	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	heapBefore := m.HeapAlloc

	t.Log("=== 키 홍수 부하: 10만 개 상이한 키 주입 (LRU 상한 1만) ===")
	t.Log("주입키수 | 추적키수 | 축출누적 | 힙(MB)")

	for i := 0; i < totalKeys; i++ {
		key := fmt.Sprintf("key-%d", i)
		tr.Observe(key, 1, now.Add(time.Duration(i)*time.Millisecond))

		if (i+1)%20_000 == 0 {
			s := tr.Stats(now)
			runtime.ReadMemStats(&m)
			t.Logf("%8d | %8d | %8d | %6.1f",
				i+1, s.TrackedKeys, s.TotalEvicted, float64(m.HeapAlloc)/1024/1024)

			if s.TrackedKeys > cfg.MaxKeys {
				t.Fatalf("추적 키 수 %d > 상한 %d — 메모리 상한 붕괴", s.TrackedKeys, cfg.MaxKeys)
			}
		}
	}

	runtime.GC()
	runtime.ReadMemStats(&m)
	heapAfter := m.HeapAlloc
	growth := int64(heapAfter) - int64(heapBefore)

	final := tr.Stats(now)
	t.Logf("최종: 추적 %d키 (상한 %d), 축출 %d회, 힙 증가 %.1fMB",
		final.TrackedKeys, final.MaxKeys, final.TotalEvicted, float64(growth)/1024/1024)

	// 상한 검증: 이론 상한(1만 × 200B ≈ 2MB)의 여유 배수 안에 들어와야 한다.
	// (Go 런타임 오버헤드·GC 타이밍을 감안해 10MB를 실패선으로 둔다 —
	//  무한 성장이면 10만 키 × 200B = 20MB+ 로 이 선을 넘는다.)
	const failLimit = 10 << 20
	if growth > failLimit {
		t.Errorf("힙 증가 %.1fMB > %dMB — 상태가 상한을 넘어 성장함",
			float64(growth)/1024/1024, failLimit>>20)
	}
	if final.TrackedKeys != cfg.MaxKeys {
		t.Errorf("추적 키 수 %d ≠ 상한 %d", final.TrackedKeys, cfg.MaxKeys)
	}
}

// TestTTLEviction은 비활동 키의 상태가 자동 소멸함을 확인한다.
func TestTTLEviction(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.IdleTTL = time.Hour
	tr := NewTracker(cfg)

	now := time.Now()
	for i := 0; i < 100; i++ {
		tr.Observe(fmt.Sprintf("old-%d", i), 1, now)
	}
	if got := tr.Stats(now).TrackedKeys; got != 100 {
		t.Fatalf("추적 키 %d ≠ 100", got)
	}

	// 2시간 후 청소
	later := now.Add(2 * time.Hour)
	tr.Sweep(later)

	s := tr.Stats(later)
	if s.TrackedKeys != 0 {
		t.Errorf("TTL 경과 후에도 %d키 잔존 — 상태가 영속화됨", s.TrackedKeys)
	}
	if s.TotalExpired != 100 {
		t.Errorf("만료 카운트 %d ≠ 100", s.TotalExpired)
	}
	t.Logf("비활동 100키가 TTL(1h) 경과 후 전부 소멸 (만료 %d)", s.TotalExpired)
}

// TestRateLimitAndAutoRelease는 경보 키 차단과 TTL 기반 자동 해제를 확인한다.
func TestRateLimitAndAutoRelease(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.LimitDuration = time.Minute
	tr := NewTracker(cfg)

	r := rand.New(rand.NewSource(5))
	now := time.Now()
	key := "attacker"

	// 워밍업: 정상 트래픽
	for i := 0; i < 30; i++ {
		if tr.Observe(key, poisson(r, 2.0), now.Add(time.Duration(i)*time.Second)) {
			t.Fatalf("정상 트래픽이 %d초에 차단됨", i)
		}
	}

	// 공격: burst
	blockedAt := -1
	for i := 30; i < 40; i++ {
		if tr.Observe(key, poisson(r, 40.0), now.Add(time.Duration(i)*time.Second)) {
			blockedAt = i
			break
		}
	}
	if blockedAt < 0 {
		t.Fatal("burst 공격이 차단되지 않음")
	}

	// 차단 유지 확인
	during := now.Add(time.Duration(blockedAt+10) * time.Second)
	if !tr.IsBlocked(key, during) {
		t.Error("차단 기간 중인데 IsBlocked=false")
	}

	// TTL(1분) 경과 후 자동 해제
	after := now.Add(time.Duration(blockedAt)*time.Second + 2*time.Minute)
	if tr.IsBlocked(key, after) {
		t.Error("차단 시간 경과 후에도 해제되지 않음")
	}
	if tr.Observe(key, 2, after) {
		t.Error("해제 후 정상 트래픽이 다시 차단됨 (CUSUM 리셋 실패)")
	}
	t.Logf("burst를 %d초에 차단, 1분 후 자동 해제 확인", blockedAt-30)
}

// TestAbsoluteCeilingBackstop은 절대 상한이 적응형 탐지의 백스톱으로 동작함을 확인한다.
func TestAbsoluteCeilingBackstop(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.Ceiling = 50
	tr := NewTracker(cfg)
	now := time.Now()

	// 워밍업 없이 곧바로 상한 초과 → 워밍업 중이라 CUSUM은 침묵하지만 절대 상한이 잡는다
	if !tr.Observe("flooder", 100, now) {
		t.Error("절대 상한(50)을 넘는 첫 관측이 차단되지 않음 (워밍업 사각지대)")
	}
	t.Log("워밍업 중에도 절대 상한이 즉시 차단 — CUSUM 워밍업 사각지대를 보완")
}

// BenchmarkObserve는 hot path 비용을 측정한다 (릴레이 처리에 얹을 수 있는가).
func BenchmarkObserve(b *testing.B) {
	tr := NewTracker(DefaultTrackerConfig())
	now := time.Now()
	keys := make([]string, 1000)
	for i := range keys {
		keys[i] = fmt.Sprintf("key-%d", i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.Observe(keys[i%len(keys)], 1, now)
	}
}

// TestTrackerRestartRecovery는 상태 전소(서버 재시작) 후 재수렴을 확인한다.
// 무기록 서버는 상태를 백업할 수 없으므로, 빈 상태에서 빠르게 복귀해야 한다.
func TestTrackerRestartRecovery(t *testing.T) {
	tr := NewTracker(DefaultTrackerConfig()) // 재시작 = 빈 Tracker
	r := rand.New(rand.NewSource(9))
	now := time.Now()

	// 재시작 직후 정상 트래픽 10초 — 오차단이 없어야 한다
	for i := 0; i < 10; i++ {
		if tr.Observe("user", poisson(r, 2.0), now.Add(time.Duration(i)*time.Second)) {
			t.Fatalf("재시작 %d초 후 정상 트래픽이 차단됨 (워밍업 실패)", i)
		}
	}

	// 10초 시점에 공격 → 즉시 탐지되어야 한다 (성공 기준: 10초 내 기능 복귀)
	if !tr.Observe("user", 60, now.Add(10*time.Second)) {
		t.Error("재시작 10초 후 공격을 탐지하지 못함")
	}
	t.Log("재시작 후 10초 내 탐지 기능 복귀 확인 (오차단 0)")
}
