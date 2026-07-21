package anomaly

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// TestMemoryBoundUnderKeyFlood is the central proof of exp-0006.
//
// It injects 100,000 distinct keys and shows, as a time series, that tracked
// state stays pinned at the LRU cap and that heap usage never exceeds the scale
// of cap × entry size.
// (If this test fails, a new form of persistence has appeared — no database,
// but memory growing without bound.)
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

	t.Log("=== key flood: injecting 100k distinct keys (LRU cap 10k) ===")
	t.Log("injected | tracked | evicted | heap(MB)")

	for i := 0; i < totalKeys; i++ {
		key := fmt.Sprintf("key-%d", i)
		tr.Observe(key, 1, now.Add(time.Duration(i)*time.Millisecond))

		if (i+1)%20_000 == 0 {
			s := tr.Stats(now)
			runtime.ReadMemStats(&m)
			t.Logf("%8d | %8d | %8d | %6.1f",
				i+1, s.TrackedKeys, s.TotalEvicted, float64(m.HeapAlloc)/1024/1024)

			if s.TrackedKeys > cfg.MaxKeys {
				t.Fatalf("tracked keys %d > cap %d — the memory bound has collapsed", s.TrackedKeys, cfg.MaxKeys)
			}
		}
	}

	runtime.GC()
	runtime.ReadMemStats(&m)
	heapAfter := m.HeapAlloc
	growth := int64(heapAfter) - int64(heapBefore)

	final := tr.Stats(now)
	t.Logf("final: %d keys tracked (cap %d), %d evictions, heap growth %.1fMB",
		final.TrackedKeys, final.MaxKeys, final.TotalEvicted, float64(growth)/1024/1024)

	// Bound check: growth must stay within a slack multiple of the theoretical
	// bound (10k × 200B ≈ 2MB). The failure line is set at 10MB to allow for Go
	// runtime overhead and GC timing — unbounded growth would put 100k keys ×
	// 200B = 20MB+ well past it.
	const failLimit = 10 << 20
	if growth > failLimit {
		t.Errorf("heap growth %.1fMB > %dMB — state grew past its bound",
			float64(growth)/1024/1024, failLimit>>20)
	}
	if final.TrackedKeys != cfg.MaxKeys {
		t.Errorf("tracked keys %d ≠ cap %d", final.TrackedKeys, cfg.MaxKeys)
	}
}

// TestTTLEviction confirms that state for idle keys expires on its own.
func TestTTLEviction(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.IdleTTL = time.Hour
	tr := NewTracker(cfg)

	now := time.Now()
	for i := 0; i < 100; i++ {
		tr.Observe(fmt.Sprintf("old-%d", i), 1, now)
	}
	if got := tr.Stats(now).TrackedKeys; got != 100 {
		t.Fatalf("tracked keys %d ≠ 100", got)
	}

	// Sweep two hours later
	later := now.Add(2 * time.Hour)
	tr.Sweep(later)

	s := tr.Stats(later)
	if s.TrackedKeys != 0 {
		t.Errorf("%d keys survived past the TTL — state persisted", s.TrackedKeys)
	}
	if s.TotalExpired != 100 {
		t.Errorf("expiry count %d ≠ 100", s.TotalExpired)
	}
	t.Logf("all 100 idle keys vanished after the TTL (1h) elapsed (%d expired)", s.TotalExpired)
}

// TestRateLimitAndAutoRelease confirms that an alarming key is blocked and then
// released automatically once its deadline passes.
func TestRateLimitAndAutoRelease(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.LimitDuration = time.Minute
	tr := NewTracker(cfg)

	r := rand.New(rand.NewSource(5))
	now := time.Now()
	key := "attacker"

	// Warm-up: normal traffic
	for i := 0; i < 30; i++ {
		if tr.Observe(key, poisson(r, 2.0), now.Add(time.Duration(i)*time.Second)) {
			t.Fatalf("normal traffic blocked at %ds", i)
		}
	}

	// Attack: burst
	blockedAt := -1
	for i := 30; i < 40; i++ {
		if tr.Observe(key, poisson(r, 40.0), now.Add(time.Duration(i)*time.Second)) {
			blockedAt = i
			break
		}
	}
	if blockedAt < 0 {
		t.Fatal("the burst attack was not blocked")
	}

	// The block must hold
	during := now.Add(time.Duration(blockedAt+10) * time.Second)
	if !tr.IsBlocked(key, during) {
		t.Error("IsBlocked=false while still inside the block window")
	}

	// Automatic release once the limit (1 minute) has passed
	after := now.Add(time.Duration(blockedAt)*time.Second + 2*time.Minute)
	if tr.IsBlocked(key, after) {
		t.Error("still blocked after the block window elapsed")
	}
	if tr.Observe(key, 2, after) {
		t.Error("normal traffic blocked again after release (CUSUM reset failed)")
	}
	t.Logf("burst blocked at %ds, automatic release after 1 minute confirmed", blockedAt-30)
}

// TestAbsoluteCeilingBackstop confirms that the absolute ceiling acts as a
// backstop to the adaptive detector.
func TestAbsoluteCeilingBackstop(t *testing.T) {
	cfg := DefaultTrackerConfig()
	cfg.Ceiling = 50
	tr := NewTracker(cfg)
	now := time.Now()

	// Exceed the ceiling immediately, with no warm-up → CUSUM stays silent
	// because it is still warming up, but the absolute ceiling catches it.
	if !tr.Observe("flooder", 100, now) {
		t.Error("the first observation above the absolute ceiling (50) was not blocked (warm-up blind spot)")
	}
	t.Log("the absolute ceiling blocks immediately even during warm-up — covering the CUSUM warm-up blind spot")
}

// BenchmarkObserve measures the hot-path cost (can this ride along with relay
// processing?).
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

// TestTrackerRestartRecovery confirms reconvergence after total state loss (a
// server restart). A no-log server cannot back its state up, so it has to
// recover quickly from empty.
func TestTrackerRestartRecovery(t *testing.T) {
	tr := NewTracker(DefaultTrackerConfig()) // restart = an empty Tracker
	r := rand.New(rand.NewSource(9))
	now := time.Now()

	// 10s of normal traffic right after the restart — nothing may be blocked
	for i := 0; i < 10; i++ {
		if tr.Observe("user", poisson(r, 2.0), now.Add(time.Duration(i)*time.Second)) {
			t.Fatalf("normal traffic blocked %ds after restart (warm-up failed)", i)
		}
	}

	// Attack at the 10s mark → must be detected at once (success criterion:
	// functional recovery within 10 seconds)
	if !tr.Observe("user", 60, now.Add(10*time.Second)) {
		t.Error("failed to detect the attack 10s after restart")
	}
	t.Log("detection recovered within 10s of restart, with zero false blocks")
}
