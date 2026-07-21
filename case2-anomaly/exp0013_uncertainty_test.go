package anomaly

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// exp-0013 Phase C: simulation seed variance
// (see case1-toxicity/pipeline/exp-0013-teacher-server-variance/).
//
// The figures reported for the parameters adopted in exp-0005 (0.2% false
// positives, burst detected immediately, ramp in 20s, low-and-slow missed) came
// from a single run on fixed seeds (normal 42, attack 7). This harness repeats
// the identical protocol (TestSelectedConfig: 500 normal keys × 3600 ticks at
// base 2.0/s; attack warm-up 120 ticks + 300 ticks) over 30 seed pairs
// (normal 1000+i, attack 2000+i) to obtain a distribution.
//
// It is skipped in an ordinary `go test ./...` run — it executes only when
// ZENBYTE_EXP0013_OUT names a CSV output path, so existing test time and CI
// remain unchanged. Being a test file, it has no bearing on the shipped binary.
func TestExp0013SeedSweep(t *testing.T) {
	out := os.Getenv("ZENBYTE_EXP0013_OUT")
	if out == "" {
		t.Skip("exp-0013 harness: ZENBYTE_EXP0013_OUT not set — skipping")
	}

	const (
		seeds      = 30
		normalKeys = 500
		duration   = 3600
		base       = 2.0
	)

	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("could not create output file: %v", err)
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
	t.Logf("→ written to %s (the verdict is decided by exp-0013 analyze_c.py against its pre-registered criteria)", out)
}
