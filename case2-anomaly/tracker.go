package anomaly

import (
	"container/list"
	"sync"
	"time"
)

// Tracker manages per-key detection state under an LRU bound plus a TTL
// (exp-0006).
//
// ⚠️ Why this file exists:
// Not using a database does not by itself make a system zero-persistence. If
// per-key state accumulates without bound, that accumulation *is* persistence
// by memory — a violation of the principle and a new failure mode besides.
// Two mechanisms prevent it:
//  1. an LRU bound (MaxKeys) — past the cap, the least recently used keys are evicted
//  2. a TTL (IdleTTL)        — state for keys idle beyond it expires on its own
//
// Memory use is therefore bounded by MaxKeys × sizeof(entry), which is
// **provable by arithmetic** rather than by inspection.
type Tracker struct {
	mu       sync.Mutex
	detector *Detector
	entries  map[string]*list.Element // key → LRU node
	lru      *list.List               // front = most recently used
	maxKeys  int
	idleTTL  time.Duration

	// Absolute ceiling: the line that must not be crossed under any
	// circumstance, independent of the adaptive baseline. It partly covers the
	// EWMA/CUSUM blind spot (abuse that never spikes). 0 disables it.
	ceiling float64

	// Rate limiting: an alarming key is blocked until this instant (released
	// automatically once the deadline passes).
	limitDuration time.Duration

	// Aggregate counters (for the admin API — cumulative totals only, never
	// per-key history).
	totalAlarms  uint64
	totalBlocked uint64
	totalEvicted uint64
	totalExpired uint64
}

type entry struct {
	key          string
	state        KeyState
	blockedUntil time.Time
}

// TrackerConfig holds the Tracker's bounding parameters.
type TrackerConfig struct {
	Detector      Config
	MaxKeys       int           // LRU cap (the value that fixes the memory bound)
	IdleTTL       time.Duration // idle keys expire after this
	Ceiling       float64       // absolute ceiling (events per tick). 0 = disabled
	LimitDuration time.Duration // how long a block is held after an alarm
}

func DefaultTrackerConfig() TrackerConfig {
	return TrackerConfig{
		Detector:      DefaultConfig(),
		MaxKeys:       10_000,
		IdleTTL:       time.Hour,
		Ceiling:       50, // 50 events/s: a line a legitimate client never crosses
		LimitDuration: 5 * time.Minute,
	}
}

func NewTracker(cfg TrackerConfig) *Tracker {
	return &Tracker{
		detector:      New(cfg.Detector),
		entries:       make(map[string]*list.Element),
		lru:           list.New(),
		maxKeys:       cfg.MaxKeys,
		idleTTL:       cfg.IdleTTL,
		ceiling:       cfg.Ceiling,
		limitDuration: cfg.LimitDuration,
	}
}

// Observe folds this tick's observation for a key into its state and reports
// whether the key should be blocked. O(1) per event, so it can be called from
// the relay hot path.
func (t *Tracker) Observe(key string, x float64, now time.Time) (blocked bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	e := t.touch(key, now)

	// Already blocked: keep it (released automatically once the deadline passes).
	if now.Before(e.blockedUntil) {
		t.totalBlocked++
		return true
	}
	if !e.blockedUntil.IsZero() && !now.Before(e.blockedUntil) {
		// Moment of release: clear the cumulative sum and start over, so the key
		// is not immediately re-blocked.
		e.blockedUntil = time.Time{}
		t.detector.Reset(&e.state)
	}

	alarm := t.detector.Observe(&e.state, x, now)
	overCeiling := t.ceiling > 0 && x > t.ceiling

	if alarm || overCeiling {
		t.totalAlarms++
		e.blockedUntil = now.Add(t.limitDuration)
		t.totalBlocked++
		return true
	}
	return false
}

// IsBlocked reports whether a key is blocked without updating any state (for
// checking before a connection is accepted).
func (t *Tracker) IsBlocked(key string, now time.Time) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	el, ok := t.entries[key]
	if !ok {
		return false
	}
	return now.Before(el.Value.(*entry).blockedUntil)
}

// touch fetches or creates a key's entry and moves it to the front of the LRU,
// evicting when the cap is exceeded. The caller must hold mu.
func (t *Tracker) touch(key string, now time.Time) *entry {
	if el, ok := t.entries[key]; ok {
		t.lru.MoveToFront(el)
		return el.Value.(*entry)
	}

	// New key: reclaim space by clearing TTL-expired entries first.
	t.evictExpired(now)

	for len(t.entries) >= t.maxKeys {
		t.evictOldest()
	}

	e := &entry{key: key}
	t.entries[key] = t.lru.PushFront(e)
	return e
}

// evictOldest evicts the key at the back of the LRU (the least recently used).
func (t *Tracker) evictOldest() {
	el := t.lru.Back()
	if el == nil {
		return
	}
	t.lru.Remove(el)
	delete(t.entries, el.Value.(*entry).key)
	t.totalEvicted++
}

// evictExpired clears keys past IdleTTL, working from the back (LRU order is
// last-used order).
func (t *Tracker) evictExpired(now time.Time) {
	for {
		el := t.lru.Back()
		if el == nil {
			return
		}
		e := el.Value.(*entry)
		if e.state.LastSeen.IsZero() || now.Sub(e.state.LastSeen) < t.idleTTL {
			return // if the back is still alive, everything ahead of it is too
		}
		t.lru.Remove(el)
		delete(t.entries, e.key)
		t.totalExpired++
	}
}

// Sweep is the periodic TTL cleanup, so that state disappears even when no
// traffic arrives.
func (t *Tracker) Sweep(now time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.evictExpired(now)
}

// Stats are the aggregate figures exposed by the admin API: cumulative and
// current counts only, never per-key history.
type Stats struct {
	TrackedKeys  int    `json:"trackedKeys"`
	MaxKeys      int    `json:"maxKeys"`
	BlockedKeys  int    `json:"blockedKeys"`
	TotalAlarms  uint64 `json:"totalAlarms"`
	TotalBlocked uint64 `json:"totalBlocked"`
	TotalEvicted uint64 `json:"totalEvicted"`
	TotalExpired uint64 `json:"totalExpired"`
	StateBytes   int    `json:"stateBytesApprox"` // for computing the memory bound
}

func (t *Tracker) Stats(now time.Time) Stats {
	t.mu.Lock()
	defer t.mu.Unlock()

	blocked := 0
	for _, el := range t.entries {
		if now.Before(el.Value.(*entry).blockedUntil) {
			blocked++
		}
	}
	return Stats{
		TrackedKeys:  len(t.entries),
		MaxKeys:      t.maxKeys,
		BlockedKeys:  blocked,
		TotalAlarms:  t.totalAlarms,
		TotalBlocked: t.totalBlocked,
		TotalEvicted: t.totalEvicted,
		TotalExpired: t.totalExpired,
		StateBytes:   len(t.entries) * entryBytes,
	}
}

// entryBytes is the approximate size of one entry (KeyState 64B + the key
// string + LRU node overhead).
const entryBytes = 200
