package anomaly

import (
	"sync"
	"time"
)

// RateTracker converts per-event calls into per-second observations and hands
// those to the Tracker.
//
// Usage (relay hot path):
//
//	if rt.IsBlocked(key) { drop }   // O(1), minimal locking
//	rt.Record(key)                  // counter increment only
//
// Then once a second rt.Tick(now) flushes the counters as observations.
// The counters are cleared as they are flushed, so past traffic never
// accumulates (zero-persistence).
type RateTracker struct {
	tracker *Tracker

	mu      sync.Mutex
	counts  map[string]float64 // per-key event count for the current tick (cleared on flush)
	ceiling float64            // in-tick immediate block ceiling (0 = disabled)
}

func NewRateTracker(cfg TrackerConfig) *RateTracker {
	return &RateTracker{
		tracker: NewTracker(cfg),
		counts:  make(map[string]float64),
		ceiling: cfg.Ceiling,
	}
}

// Record logs one event for a key in the current tick. It returns true (block)
// as soon as the absolute ceiling is crossed — an inline backstop that cuts off
// a flood without waiting for the tick boundary.
func (rt *RateTracker) Record(key string) (overCeiling bool) {
	rt.mu.Lock()
	rt.counts[key]++
	c := rt.counts[key]
	rt.mu.Unlock()
	return rt.ceiling > 0 && c > rt.ceiling
}

// IsBlocked reports whether a key is currently blocked by earlier ticks'
// verdicts, without updating any state.
func (rt *RateTracker) IsBlocked(key string, now time.Time) bool {
	return rt.tracker.IsBlocked(key, now)
}

// Tick feeds the current tick's counters into the detector and clears them.
// Call it once a second. It returns the number of keys newly blocked this tick.
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
	// Let the TTL expire state for keys whose traffic has stopped as well.
	rt.tracker.Sweep(now)
	return newlyBlocked
}

func (rt *RateTracker) Stats(now time.Time) Stats { return rt.tracker.Stats(now) }
