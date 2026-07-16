package anomaly

import (
	"container/list"
	"sync"
	"time"
)

// Tracker는 키별 탐지 상태를 LRU + TTL로 관리한다 (exp-0006).
//
// ⚠️ 이 파일의 존재 이유:
// DB를 쓰지 않는다고 Zero-Persistence가 되는 게 아니다. 키별 상태가 무한히 쌓이면
// 그것이 곧 "메모리에 의한 영속화"이며, 원칙 위반이자 새로운 장애 원인이다.
// 두 장치가 이를 막는다:
//  1. LRU 상한 (MaxKeys) — 초과 시 가장 오래 안 쓰인 키부터 축출
//  2. TTL (IdleTTL)     — 일정 시간 활동 없는 키의 상태는 자동 소멸
//
// 따라서 메모리 사용량 상한 = MaxKeys × sizeof(entry) 로 **산술적으로 증명 가능**하다.
type Tracker struct {
	mu       sync.Mutex
	detector *Detector
	entries  map[string]*list.Element // key → LRU 노드
	lru      *list.List               // front = 최근 사용
	maxKeys  int
	idleTTL  time.Duration

	// 절대 상한: 적응형 기준선과 무관하게 "어떤 경우에도 넘으면 안 되는 선".
	// EWMA/CUSUM의 사각지대(급격하지 않은 남용)를 일부 보완한다. 0이면 비활성.
	ceiling float64

	// rate-limit: 경보 키는 이 시각까지 차단된다 (TTL 기반 자동 해제).
	limitDuration time.Duration

	// 집계 카운터 (관리자 API용, 개별 키 이력 아님 — 누적 수치만)
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

// TrackerConfig는 Tracker의 상한 파라미터다.
type TrackerConfig struct {
	Detector      Config
	MaxKeys       int           // LRU 상한 (메모리 상한을 결정하는 값)
	IdleTTL       time.Duration // 비활동 키 자동 소멸
	Ceiling       float64       // 절대 상한 (틱당 이벤트 수). 0 = 비활성
	LimitDuration time.Duration // 경보 시 차단 유지 시간
}

func DefaultTrackerConfig() TrackerConfig {
	return TrackerConfig{
		Detector:      DefaultConfig(),
		MaxKeys:       10_000,
		IdleTTL:       time.Hour,
		Ceiling:       50, // 초당 50 이벤트: 정상 클라이언트가 결코 넘지 않는 선
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

// Observe는 키의 이번 틱 관측치를 반영하고, 그 키를 차단해야 하는지 반환한다.
// 이벤트당 O(1) — 릴레이 hot path에서 호출 가능하다.
func (t *Tracker) Observe(key string, x float64, now time.Time) (blocked bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	e := t.touch(key, now)

	// 이미 차단 중이면 유지 (TTL 경과 시 자동 해제)
	if now.Before(e.blockedUntil) {
		t.totalBlocked++
		return true
	}
	if !e.blockedUntil.IsZero() && !now.Before(e.blockedUntil) {
		// 차단 해제 시점: 누적합을 비우고 재출발 (해제 직후 즉시 재차단 방지)
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

// IsBlocked는 상태를 갱신하지 않고 차단 여부만 확인한다 (연결 수락 전 검사용).
func (t *Tracker) IsBlocked(key string, now time.Time) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	el, ok := t.entries[key]
	if !ok {
		return false
	}
	return now.Before(el.Value.(*entry).blockedUntil)
}

// touch는 키의 엔트리를 가져오거나 만들고, LRU 앞으로 옮긴다. 상한 초과 시 축출한다.
// 호출자가 mu를 잡고 있어야 한다.
func (t *Tracker) touch(key string, now time.Time) *entry {
	if el, ok := t.entries[key]; ok {
		t.lru.MoveToFront(el)
		return el.Value.(*entry)
	}

	// 신규 키: 먼저 TTL 만료분을 정리해 자리를 확보한다
	t.evictExpired(now)

	for len(t.entries) >= t.maxKeys {
		t.evictOldest()
	}

	e := &entry{key: key}
	t.entries[key] = t.lru.PushFront(e)
	return e
}

// evictOldest는 LRU 뒤쪽(가장 오래 안 쓰인) 키를 축출한다.
func (t *Tracker) evictOldest() {
	el := t.lru.Back()
	if el == nil {
		return
	}
	t.lru.Remove(el)
	delete(t.entries, el.Value.(*entry).key)
	t.totalEvicted++
}

// evictExpired는 IdleTTL을 넘긴 키를 뒤쪽부터 정리한다 (LRU 순서 = 마지막 사용 순서).
func (t *Tracker) evictExpired(now time.Time) {
	for {
		el := t.lru.Back()
		if el == nil {
			return
		}
		e := el.Value.(*entry)
		if e.state.LastSeen.IsZero() || now.Sub(e.state.LastSeen) < t.idleTTL {
			return // 뒤쪽이 아직 살아있으면 앞쪽도 모두 살아있다
		}
		t.lru.Remove(el)
		delete(t.entries, e.key)
		t.totalExpired++
	}
}

// Sweep은 주기적 TTL 청소용이다 (트래픽이 없어도 상태가 사라지도록).
func (t *Tracker) Sweep(now time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.evictExpired(now)
}

// Stats는 관리자 API용 집계 수치다. 개별 키 이력이 아니라 누적/현재 카운트만 노출한다.
type Stats struct {
	TrackedKeys  int    `json:"trackedKeys"`
	MaxKeys      int    `json:"maxKeys"`
	BlockedKeys  int    `json:"blockedKeys"`
	TotalAlarms  uint64 `json:"totalAlarms"`
	TotalBlocked uint64 `json:"totalBlocked"`
	TotalEvicted uint64 `json:"totalEvicted"`
	TotalExpired uint64 `json:"totalExpired"`
	StateBytes   int    `json:"stateBytesApprox"` // 상태 메모리 상한 계산용
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

// entryBytes는 엔트리 하나의 대략적 크기다 (KeyState 64B + 키 문자열 + LRU 노드 오버헤드).
const entryBytes = 200
