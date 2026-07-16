# Case study 2: server traffic anomaly detection

Paper §6. A per-key EWMA baseline + CUSUM accumulator whose global memory is
bounded by arithmetic (LRU cap 10,000 keys × ~200 B ≈ 2 MB) — because
unbounded in-memory state would be *persistence by memory*, the violation the
detector exists to avoid.

Self-contained Go module (standard library only):

```
go test ./... -v        # everything below, deterministic seeds
go test -bench=Observe  # hot-path cost (paper: 41 ns/op)
```

| Test | What it proves | Paper |
|---|---|---|
| `TestMemoryBoundUnderKeyFlood` | 100,000 distinct keys injected → tracked state flat at the 10,000 cap, heap +2.1 MB | §6.2, Fig. 3 |
| `TestParameterSweep`, `TestRefinedSweep` | the (α, K, H) grid; adopted (0.1, 0.5, 10): FP 0.2%, burst first tick, ramp 20 s | §6.2, docs/C.4 |
| `TestBaselinePoisoning` | baseline updates freeze while an alarm is active — 600/600 alarm ticks under a 10-min attack, baseline unchanged | §6.2 |
| `TestRecoveryAfterRestart`, `TestTrackerRestartRecovery` | reconvergence after total state loss (the perpetual cold start) | §6.3 |
| `TestAbsoluteCeiling`, `TestAbsoluteCeilingBackstop` | the history-free ceiling covers the warm-up blind window | §6.3 |
| `TestSelectedConfig` | low-and-slow (2× sustained) **non-detection asserted** — the limitation is pinned so it cannot silently go stale | §6.5 |
| `TestTTLEviction`, `TestRateLimitAndAutoRelease`, `TestStateSizeIsConstant` | idle TTL, block + auto-release, per-key state is O(1) scalars (64 B) | §6.2 |

Files: `detector.go` (EWMA/CUSUM), `tracker.go` (LRU + TTL container — the
memory bound), `rate.go` (event counters → per-second observations, flushed
each tick, no accumulation).

In production this package is wired to the relay hub (per-sender message rate)
and connection handler (per-IP connection rate); the admin surface is
token-gated and aggregate-only — which key was blocked is never exposed,
because that is itself the metadata the system protects (§6.4).
