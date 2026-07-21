// Package anomaly provides EWMA + CUSUM traffic anomaly detection (exp-0005).
//
// Design principles (zero-persistence):
//   - Per-key state is four scalars only (mean, variance, cumulative sum, last
//     seen). No past observation is retained.
//     → Accumulating a traffic history is itself metadata persistence, so it
//     is forbidden.
//   - O(1) update per event, cheap enough to sit on the relay hot path.
//   - Designed on the assumption that all state burns on restart — the
//     detector reconverges within seconds after warm-up.
package anomaly

import (
	"math"
	"time"
)

// KeyState is the detection state of a single key (IP or public key).
// It holds scalars only.
type KeyState struct {
	Mean     float64   // EWMA baseline (the normal level)
	Variance float64   // EWMA variance estimate (used for normalization)
	CUSUM    float64   // cumulative sum (accumulates small but sustained rises)
	Samples  int       // warm-up counter
	Alarming bool      // currently alarming?
	LastSeen time.Time // used to decide TTL expiry (exp-0006)
}

// Config holds the detector parameters.
type Config struct {
	Alpha     float64 // EWMA smoothing factor (0.2–0.3 typical): higher reacts faster to recent values
	K         float64 // CUSUM slack in standard deviations: the headroom that absorbs noise
	H         float64 // CUSUM alarm threshold (the cumulative sum crossing it means anomaly)
	MinSigma  float64 // lower bound on sigma (avoids division by zero at zero variance, and damps over-sensitivity)
	WarmupMin int     // minimum observations before any alarm (suppresses false positives right after a restart)
}

// DefaultConfig returns the parameters selected by the exp-0005 sweep.
//
// Selection evidence (200 normal keys × 1 hour, base 2 events/s):
//
//	0.0% false positive rate · burst (20×) detected immediately · ramp (10× over 60s) detected in 20s.
//
// Lowering Alpha to 0.1 is the crucial part — the baseline has to move slowly,
// otherwise it absorbs a gradual ramp.
//
// Known limitation: low-and-slow abuse (sustained at 2× the base rate) is **not
// detected**. An adaptive baseline learns it as the "new normal", and a fixed
// rate limit misses it just the same (2× < the 3× threshold). This family of
// attacks has to be handled by AbsoluteCeiling instead.
func DefaultConfig() Config {
	return Config{Alpha: 0.1, K: 0.5, H: 10.0, MinSigma: 1.0, WarmupMin: 5}
}

// Detector is stateless (it holds parameters only). The caller keeps the
// per-key state.
type Detector struct {
	cfg Config
}

func New(cfg Config) *Detector { return &Detector{cfg: cfg} }

// Observe folds a new observation x for a key (e.g. the event count in this
// tick) into its state and reports whether it alarms.
//
// Key design point: while alarming, the baseline (Mean/Variance) is not
// updated. If attack traffic were allowed to drag the baseline up it would
// become the "new normal" and neutralize detection (baseline-poisoning
// defense).
func (d *Detector) Observe(s *KeyState, x float64, now time.Time) bool {
	s.LastSeen = now

	// Warm-up: accumulate observations without alarming until a baseline exists.
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

	// Standardized residual: how many sigma above the baseline (one-sided —
	// only traffic surges are of interest).
	z := (x - s.Mean) / sigma

	// CUSUM: accumulate only the part exceeding the slack K. Clamp at 0 so that
	// past headroom does not bank up.
	s.CUSUM = math.Max(0, s.CUSUM+z-d.cfg.K)

	if s.CUSUM > d.cfg.H {
		s.Alarming = true
		return true // alarming: baseline left untouched (poisoning defense)
	}

	// Back to normal
	s.Alarming = false
	delta := x - s.Mean
	s.Mean += d.cfg.Alpha * delta
	s.Variance = (1 - d.cfg.Alpha) * (s.Variance + d.cfg.Alpha*delta*delta)
	return false
}

// Reset clears the cumulative sum when an alarm is lifted (a fresh start after
// a rate limit is released, exp-0006).
func (d *Detector) Reset(s *KeyState) {
	s.CUSUM = 0
	s.Alarming = false
}
