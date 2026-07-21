package anomaly

import (
	"math"
	"math/rand"
	"testing"
	"time"
	"unsafe"
)

// exp-0005 offline simulation.
//
// Tunes (Alpha, K, H) on synthetic traffic and measures detection performance
// and the false positive rate. Every simulation is reproducible from a fixed
// seed.
//
// Traffic model: 1 second = 1 tick. Each tick observes a per-key event count.
//   - normal:       Poisson noise on top of a diurnal (day/night) base rate
//   - burst:        a jump to 20× the base rate from a given instant
//   - ramp:         a gradual rise to 10× over 60 seconds
//   - low-and-slow: sustained at 2× the base rate (the hardest case)

const tickSec = 1

// diurnalRate returns the base traffic level over a daily cycle (peak in the
// afternoon, trough before dawn).
func diurnalRate(tick int, base float64) float64 {
	hour := float64(tick%86400) / 3600.0
	// Varies gently between 0.4× and 1.6× (lowest at 4am, highest at 4pm)
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

// simulateNormal observes one normal key for duration ticks and counts alarms.
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

// simulateAttack injects an attack after a warm-up and returns the delay in
// seconds from injection to the first alarm, or -1 if it is missed.
func simulateAttack(d *Detector, r *rand.Rand, base float64, attack string) int {
	s := &KeyState{}
	t0 := time.Now()
	const warmup = 120 // 2 minutes of normal traffic to establish a baseline
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

// Baseline: a fixed-threshold rate limit (alarms above 3× the base rate)
func simulateFixedThreshold(r *rand.Rand, base float64, multiplier float64, duration int) (alarms int) {
	threshold := base * multiplier
	for tick := 0; tick < duration; tick++ {
		if poisson(r, diurnalRate(tick, base)) > threshold {
			alarms++
		}
	}
	return
}

// TestParameterSweep measures the false positive rate and burst detection delay
// over an (Alpha, K, H) grid.
func TestParameterSweep(t *testing.T) {
	const (
		normalKeys = 200  // number of normal keys (the false-positive denominator)
		duration   = 3600 // one hour of observation
		base       = 2.0  // 2 events per second on average
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

				// False positives: the fraction of normal keys that alarmed at least once
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

	t.Log("=== exp-0005 parameter sweep (200 normal keys × 1h, base=2/s) ===")
	t.Log("alpha    K    H |    FP | burst | ramp | low&slow  (delay in seconds, -1 = missed)")
	for _, r := range rows {
		t.Logf("%5.1f %4.1f %4.1f | %5.1f%% | %5d | %4d | %8d",
			r.cfg.Alpha, r.cfg.K, r.cfg.H, r.falsePositive*100,
			r.burstDelay, r.rampDelay, r.lowSlowDelay)
	}

	// Baseline comparison (fixed threshold)
	t.Log("--- baseline: fixed-threshold rate limit ---")
	for _, mult := range []float64{2.0, 3.0, 5.0} {
		rb := rand.New(rand.NewSource(42))
		keysWithAlarm := 0
		for i := 0; i < normalKeys; i++ {
			if simulateFixedThreshold(rb, base, mult, duration) > 0 {
				keysWithAlarm++
			}
		}
		fp := float64(keysWithAlarm) / float64(normalKeys)
		// Detection delay of a fixed threshold: burst (20×) takes one tick, while
		// low-and-slow (2×) is missed forever whenever it stays below the threshold
		lowSlowDetect := "missed"
		if base*2 > base*mult {
			lowSlowDetect = "immediate"
		}
		t.Logf("threshold=%.0f× base | FP %5.1f%% | burst immediate | low&slow %s", mult, fp*100, lowSlowDetect)
	}
}

// TestSelectedConfig verifies that the adopted parameters still meet the
// success criteria (regression guard).
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
		t.Errorf("false positive rate %.1f%% > 5%% (success criterion violated)", fp*100)
	}

	ra := rand.New(rand.NewSource(7))
	burst := simulateAttack(d, ra, base, "burst")
	if burst < 0 || burst > 3 {
		t.Errorf("burst detection delay %ds (criterion: ≤3s)", burst)
	}

	ramp := simulateAttack(d, ra, base, "ramp")
	if ramp < 0 || ramp > 30 {
		t.Errorf("ramp detection delay %ds (expected: ≤30s)", ramp)
	}

	// Missing low-and-slow is a documented, structural limitation — if it were
	// ever detected, that documentation would be stale, so the test fails to
	// force an update (pinning the negative claim as well).
	lowSlow := simulateAttack(d, ra, base, "low-and-slow")
	if lowSlow >= 0 {
		t.Errorf("low-and-slow detected at %ds — the documented limitation of 'structurally undetectable' no longer holds", lowSlow)
	}

	t.Logf("adopted parameters %+v", cfg)
	t.Logf("FP=%.1f%% burst=%ds ramp=%ds low&slow=%ds (-1 = missed, a known limitation)",
		fp*100, burst, ramp, lowSlow)
}

// TestAbsoluteCeiling shows how an absolute ceiling complements the structural
// blindness to low-and-slow abuse. EWMA/CUSUM watches for "unusual relative to
// this key's own normal"; the absolute ceiling holds a line that must not be
// crossed under any circumstance. The two are complementary and are applied
// together in the exp-0006 hub integration.
func TestAbsoluteCeiling(t *testing.T) {
	d := New(DefaultConfig())
	const ceiling = 10.0 // 10 events/s: a line a normal user never crosses (base=2)
	r := rand.New(rand.NewSource(11))
	t0 := time.Now()

	// Normal traffic never touches the absolute ceiling
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
		t.Errorf("normal traffic exceeded the absolute ceiling %d times (ceiling set too low)", normalCeilingHits)
	}

	// CUSUM misses low-and-slow (2×), but the absolute ceiling catches the attack
	// the moment it crosses the line. At 2× = 4 events/s the attack stays under
	// the ceiling (10) and still gets through — so the absolute ceiling is not a
	// cure-all either. What this experiment establishes is precisely that: the
	// two mechanisms have different blind spots, and an overlapping region
	// remains.
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
	t.Logf("low-and-slow (2×): caught by CUSUM=%v, caught by absolute ceiling (%.0f/s)=%v", cusumCaught, ceiling, ceilingCaught)
	t.Log("→ conclusion: low-rate abuse around 2× passes both mechanisms. This is residual risk accepted by design,")
	t.Log("  and a no-log server cannot retain traffic history, so long-horizon profiling cannot catch it either.")
}

// TestBaselinePoisoning confirms that a sustained attack cannot drag the
// baseline up and neutralize detection.
func TestBaselinePoisoning(t *testing.T) {
	d := New(DefaultConfig())
	s := &KeyState{}
	r := rand.New(rand.NewSource(1))
	t0 := time.Now()

	for tick := 0; tick < 120; tick++ {
		d.Observe(s, poisson(r, 2.0), t0.Add(time.Duration(tick)*time.Second))
	}
	baselineBefore := s.Mean

	// 10 minutes of sustained attack (20×)
	alarmTicks := 0
	for i := 0; i < 600; i++ {
		if d.Observe(s, poisson(r, 40.0), t0.Add(time.Duration(120+i)*time.Second)) {
			alarmTicks++
		}
	}

	if alarmTicks < 500 {
		t.Errorf("only %d/600 ticks alarmed during a sustained attack — baseline poisoning suspected", alarmTicks)
	}
	if s.Mean > baselineBefore*2 {
		t.Errorf("baseline poisoned from %.2f to %.2f (attack traffic learned as normal)", baselineBefore, s.Mean)
	}
	t.Logf("%d of 600 attack ticks alarmed, baseline %.2f → %.2f (no poisoning)", alarmTicks, baselineBefore, s.Mean)
}

// TestRecoveryAfterRestart measures the reconvergence time after total state
// loss (a restart).
func TestRecoveryAfterRestart(t *testing.T) {
	d := New(DefaultConfig())
	r := rand.New(rand.NewSource(3))
	t0 := time.Now()

	// Right after a restart: start observing normal traffic from empty state
	s := &KeyState{}
	falseAlarms := 0
	for tick := 0; tick < 30; tick++ {
		if d.Observe(s, poisson(r, 2.0), t0.Add(time.Duration(tick)*time.Second)) {
			falseAlarms++
		}
	}
	if falseAlarms > 0 {
		t.Errorf("%d false alarms immediately after restart (warm-up failed to protect)", falseAlarms)
	}

	// Reconvergence check: is a burst caught at once 30 seconds later?
	detected := -1
	for i := 0; i < 10; i++ {
		if d.Observe(s, poisson(r, 40.0), t0.Add(time.Duration(30+i)*time.Second)) {
			detected = i
			break
		}
	}
	if detected < 0 || detected > 3 {
		t.Errorf("burst missed or detected late 30s after restart: %ds", detected)
	}
	t.Logf("reconverged within 30s of restart (0 false alarms, burst detected in %ds)", detected)
}

// TestStateSizeIsConstant documents that per-key state is a constant-size set of
// scalars (the premise of exp-0006).
func TestStateSizeIsConstant(t *testing.T) {
	var s KeyState
	size := int(unsafeSizeof(s))
	if size > 64 {
		t.Errorf("KeyState is %d bytes — the scalar-state design looks violated", size)
	}
	t.Logf("KeyState size: %d bytes (constant per key) — memory bound = max keys × %d bytes", size, size)
}

func unsafeSizeof(s KeyState) uintptr {
	return unsafe.Sizeof(s)
}

// TestRefinedSweep searches finely around the boundary the first sweep exposed
// (H≈8). Question: is there a point that keeps false positives ≤5% while still
// catching a ramp (a gradual rise)?
func TestRefinedSweep(t *testing.T) {
	const (
		normalKeys = 200
		duration   = 3600
		base       = 2.0
	)
	t.Log("=== second, refined sweep ===")
	t.Log("alpha    K    H |    FP | burst | ramp | low&slow")
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
