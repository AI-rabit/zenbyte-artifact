# Zero-persistence compliance proof (three layers)

The paper's claim (§5.6): zero-persistence is proven by executable tests, not
by review. The three layers, all included here verbatim:

| Layer | File | What it asserts |
|---|---|---|
| **Static** | `zero_persistence_audit.sh` | The detection path (`audited-sources/`) contains no network, disk-write, or content-logging calls. **Fails hard if any audit target file is missing** — added after the audit once passed vacuously by grepping a file that no longer existed. |
| **Network** | `ChatViewModelToxicityTest.kt` | Nothing is transmitted before the user confirms the warning; after confirming, only the original message bytes leave — the wire format has no field where a verdict or score could ride. |
| **Disk** | `DefaultToxicityRepositoryTest.kt` | The app's entire data directories are byte-compared before and after repeated inference: zero files added, removed, or changed. |

`audited-sources/` contains the exact files the static audit targets:
`TfidfSvmClassifier.kt` (inference), `DefaultToxicityRepository.kt`
(lazy asset load, fail-open), `ToxicityRepository.kt` (domain interface).

## Running

The script and tests execute inside the app's Gradle project (the messenger is
in closed beta and its full source is not published); they are included here
so that reviewers can inspect exactly what is asserted. The script's checks
are plain `grep -E` patterns over the audited sources — auditable by eye — and
can be pointed at `audited-sources/` by editing its `FILES` array.
