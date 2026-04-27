# Implementation Log (Pre-NVIDIA Phase)

## 2026-04-25

### Completed
- Added a Q3 regression proving that if a newer-leader repair is already in flight around a refreshed compatible suffix and an even newer advanced-frontier snapshot then lands on the repaired boundary, that advanced snapshot becomes the new exact durable `snapshot_id` everywhere while any still-compatible fresh suffix survives only as live speculative tail until later commit/apply retirement.
- Added a Q3 regression proving that a compatible advanced-frontier snapshot which preserves a speculative suffix still keeps its exact durable `snapshot_id` pinned if a later role/term handoff discards that tail; status/recovery/restart projections all collapse cleanly to the same advanced frontier.
- Added a Q3 regression proving that if a second same-frontier same-term snapshot refresh lands during newer-leader repair and the node then changes role/term, the speculative tail is still discarded cleanly while the refreshed durable `snapshot_id` remains pinned across status/recovery/restart truth surfaces.
- Added a follow-on Q3 regression proving that even after a second same-frontier same-term snapshot refresh lands during newer-leader repair, later stale snapshot installs remain complete no-ops throughout repair, commit, and final apply completion; the newer refreshed durable `snapshot_id`, recovery-gap accounting, and restart/resume projection all stay pinned.
- Added `Engine::status_snapshot()` as the engine-level truth surface for Q1, unifying served snapshot identity/frontier, replication lag, readiness flags, fallback rollups, and active runtime fallback reasons in one validated status object.
- Added `EngineStatusSnapshot`, `SnapshotStatus`, `ReadinessStatus`, and `FallbackStatus` plus invariant validation covering commit/apply/visible ordering, snapshot/frontier consistency, blocker-mask/count consistency, and mutation-admission saturation semantics.
- Extended engine regression coverage to prove the truth surface answers the key operator questions in both healthy and degraded/fallback states.
- Updated README, replication interface docs, and operations runbooks so `status_snapshot()` is the documented source of truth for "what snapshot served this?", "why did this route to fallback?", and "how far behind is replication?"
- Started Q2 with a real engine-facing MVCC read slice: committed writes now mirror into an in-memory MVCC tuple store, and `Engine::execute_mvcc_query()` runs snapshot-bound full scans / key lookups with filtering + projection through `VecOperator`.
- Added explicit GPU-first device declaration for the MVCC slice (`planned_target = GPU`, `executed_target = CPU`) plus tracked fallback reason `GpuMvccReadParityGap` → `GPU-123` so bootstrap reads stay parity-auditable instead of silently becoming CPU-only product direction.
- Added deterministic regression workload fixture `tests/fixtures/mvcc-read-workload.txt` to seed future measurement/extension loops.
- Replaced the temporary MVCC read `VecOperator` shim with explicit `ScanOperator` → `FilterOperator` → `ProjectOperator` execution stages, so the bootstrap vertical slice now flows through dedicated execution operators instead of one row-materialization helper.
- Widened the MVCC read shape to support composite `All([...])` / `Any([...])` filters at the engine-facing query boundary, keeping the same explicit GPU-parity fallback contract while making the thin slice meaningfully more expressive.
- Added `LimitOperator` plus `MvccReadQuery.limit`, so the MVCC vertical slice now supports deterministic post-filter row caps through the execution layer instead of open-coded truncation.
- Added `SortOperator` plus `MvccReadQuery.order`, so the MVCC vertical slice now supports deterministic key ordering (`KeyAsc` / `KeyDesc`) before limit/projection while keeping the same explicit GPU-parity fallback contract.
- Added lexicographic `MvccReadFilter::KeyRange { start_inclusive, end_exclusive }`, so the MVCC vertical slice now covers a real range-style read predicate instead of prefix/point filters only.
- Started Q3 replication hardening with explicit follower-resume semantics: added `RecoveryState`, `RaftReplicator::recovery_state()`, and `RaftReplicator::resume_as_follower(...)` so interruption/restart behavior is defined in code instead of implied by tests.
- Added replication regressions covering lagging follower apply delay, restart/resume with committed-but-unapplied backlog preserved, and rejection of non-contiguous recovery tails.
- Tightened the recovery helper contract itself with `RecoveryState::validate()` and derived boundary helpers (`commit_index`, `next_index`, pending-apply count/boolean) so restart automation can reason about durable state without re-deriving commit/apply semantics out-of-band.
- Added `ReplicationProgress` as a validated per-replicator progress snapshot so ordering/apply/resume tests now line up with a first-class helper surface instead of raw field peeking.
- Extended `ReplicationProgress` with `apply_gap()` / `is_caught_up()` so operator logic can distinguish commit/apply lag from truly quiescent replication without re-deriving semantics from multiple fields.
- Added `RecoveryState::progress_as_follower()` so restart bundles and live follower state now share the same validated progress vocabulary, tightening Q3 status/recovery alignment.
- Extended `RecoveryState` with `apply_gap()` / `is_caught_up()` so durable restart bundles can answer the same catch-up question directly even before a live replicator is resumed.
- Added replication regressions proving stale-term and non-contiguous append rejections leave `ReplicationProgress` unchanged, tightening Q3 guarantees around rejected-path observability stability.
- Added replication regressions proving snapshot install advances `ReplicationProgress` monotonically at the durable boundary while preserving surviving uncommitted tail accounting, tightening Q3 snapshot/catch-up semantics.
- Added replication regressions proving snapshot-boundary append rejections (term mismatch / behind-boundary prev index) leave `ReplicationProgress` unchanged, tightening Q3 compacted-boundary observability guarantees.
- Added replication regressions proving accepted appends at the snapshot boundary advance `ReplicationProgress` monotonically and idempotently across append/apply/replay, tightening Q3 compacted-boundary happy-path semantics.
- Added replication regressions proving empty heartbeats can advance durable commit progress while `ReplicationProgress` correctly converts uncommitted tail into commit/apply lag, tightening Q3 catch-up semantics for no-op append rounds.
- Added recovery regressions proving snapshot-only restart bundles project to the same caught-up follower progress surface as a live resumed node, tightening Q3 resume semantics at pure snapshot boundaries.
- Added recovery regressions proving compacted snapshot + committed-tail restart bundles project to the same lagging follower progress surface as a live resumed node, tightening Q3 resume semantics when catch-up continues past the snapshot boundary.
- Added a resumed-follower stress regression proving rejected non-contiguous catch-up leaves `ReplicationProgress` unchanged, accepted catch-up advances `next_index` without fake commits, heartbeat commit promotion converts tail into apply lag, and a later snapshot install catches the follower up without regressing the compacted boundary.
- Added local/raft regressions proving overshoot `mark_applied(...)` requests clamp at the committed boundary while leaving `ReplicationProgress` valid and explicit about any remaining uncommitted tail, tightening Q3 apply-frontier semantics.
- Added replication regressions proving role changes discard prior-epoch uncommitted tail in `ReplicationProgress` immediately while preserving committed frontier and `next_index`, tightening Q3 term-transition semantics.
- Added replication regressions proving contiguous quorum-ack promotion moves entries from uncommitted tail into committed-but-unapplied backlog in `ReplicationProgress` without allowing out-of-order ack arrival to skip commit boundaries, tightening Q3 commit-promotion semantics.
- Added replication regressions proving conflict-repair truncation updates `ReplicationProgress` by dropping only uncommitted tail and rewinding `next_index` to the durable frontier, while committed-boundary truncation remains a no-op, tightening Q3 catch-up repair semantics.
- Added local replication regressions proving rollback of unapplied tail rewinds `ReplicationProgress` back to the applied frontier and clears pending-apply backlog, tightening Q3 rollback semantics.
- Added replication regressions proving single-node leaders surface immediate commit as pending-apply backlog with no uncommitted tail in `ReplicationProgress`, tightening Q3 quorum=1 semantics.
- Added replication regressions proving ignored ack traffic (off-leader or unknown-index acks) leaves `ReplicationProgress` unchanged, tightening Q3 ack-path observability stability.
- Added replication regressions proving reserved self-acks and duplicate follower acks leave `ReplicationProgress` unchanged until quorum coverage really changes, tightening Q3 ack dedup semantics.
- Added replication regressions proving ack-tracking pruning for committed entries does not create extra `ReplicationProgress` transitions beyond the actual commit promotion, tightening Q3 bookkeeping-vs-observability boundaries.
- Added replication regressions proving `wait_committed(...)` polling leaves `ReplicationProgress` unchanged both before quorum and after resolution, tightening Q3 commit-observation semantics.
- Added local/raft/engine regressions proving stale snapshot installs are status/progress no-ops unless they advance (or exactly match) the current frontier term, preventing newer-but-stale `snapshot_id` values from drifting away from the actually served durable frontier.
- Added `RaftReplicator::recovery_progress()` as the live durable-state projection of `recovery_state()`, plus stress regressions proving rejected follower transitions leave it unchanged, accepted catch-up/heartbeat/snapshot advancement keep it aligned with the restart surface, and speculative leader-only tail remains excluded until committed.
- Added `RaftReplicator::recovery_progress_gap()` plus Q3 regressions proving the live-vs-durable delta stays zero for restart-equivalent followers, surfaces only speculative `next_index`/uncommitted-tail drift while catch-up is still in flight, snaps back to zero after quorum commit, snapshot install, or role/epoch tail discard, and remains unchanged when a same-frontier wrong-term snapshot is correctly ignored.
- Added `ReplicationStatusSnapshot` plus `LocalReplicator::status_snapshot()` / `RaftReplicator::status_snapshot()` so Q3 status checks publish live progress, durable progress, and validated live-vs-durable gap together instead of forcing callers to stitch helper surfaces back together manually.
- Tightened `ReplicationStatusSnapshot::validate()` so impossible surfaces are rejected when durable restart state would outrun the live node's snapshot frontier, commit/apply/next-index, or uncommitted-tail counters.
- Added a Q3 regression proving same-frontier same-term snapshot refreshes may advance `snapshot_id` while preserving speculative-tail `status_snapshot()` gap semantics and progress counters exactly.
- Added a follow-on Q3 regression proving that refreshed same-frontier snapshot identity survives later role/term transitions while speculative tail is discarded, keeping the durable truth surface aligned across epoch changes.
- Added a richer Q3 stress regression proving that refreshed same-frontier snapshot identity also survives a newer-leader rejection followed by accepted catch-up, heartbeat commit advancement, and final apply completion; stale-tail discard and fresh-tail repair now keep `status_snapshot()` aligned to the refreshed durable snapshot identity through the whole epoch handoff.
- Added a follow-on Q3 regression proving that the same handoff keeps `recovery_state()`, `recovery_progress()`, and `resume_as_follower(...)` aligned to that refreshed durable snapshot identity, so restart/export surfaces stay trustworthy even while live state temporarily carries fresh speculative tail.
- Added another Q3 regression locking `recovery_progress_gap()` across the refreshed-snapshot newer-leader handoff: the old speculative-tail gap shape survives the refresh, collapses to zero when the newer-leader rejection discards stale tail, reappears only for the fresh accepted tail, and returns to zero again after heartbeat/apply completion.
- Hardened `ReplicationStatusSnapshot::validate()` with a same-frontier snapshot-id alignment invariant, so status surfaces now reject live/durable identity drift even when commit/apply/frontier counters still match numerically.
- Added a focused regression locking that invalid same-frontier snapshot-id drift is rejected explicitly, tightening Q3 truth-surface semantics around snapshot identity.
- Hardened `ReplicationStatusSnapshot::validate()` again so the durable restart projection must share the live node's current term; older-term durable truth surfaces are now rejected explicitly instead of relying on implicit constructor behavior.
- Hardened local/raft snapshot installs so impossible higher-frontier snapshots with a regressed term are ignored as status/progress no-ops, and added local/raft/engine regressions locking that incoherent frontier metadata cannot outrun the served durable truth surface or perturb speculative-tail gap semantics.
- Hardened `RaftReplicator::append_entries_from_leader(...)` so newer-leader term discovery reuses the normal follower step-down path before later prev-log rejection checks, guaranteeing prior-epoch speculative tail is discarded and `status_snapshot()` returns to a restart-equivalent durable boundary even when the append itself is rejected.
- Added focused candidate/follower regressions locking that newer-leader append rejections cannot leak stale speculative tail across epochs.
- Added a follow-on Q3 stress regression proving successful newer-leader catch-up after stale-tail discard surfaces only the fresh epoch's committed/apply gap and surviving speculative tail in `status_snapshot()`, instead of leaking prior-epoch follower tail into the live-vs-durable delta.
- Extended that path with heartbeat/apply progression coverage so once the fresh epoch commits and applies, `status_snapshot()` returns to restart-equivalent state without reintroducing stale-gap artifacts.
- Hardened local/raft snapshot install identity semantics so accepted advanced-frontier snapshots now replace `snapshot_id` exactly instead of pinning to older local maxima; added local/raft/engine regressions proving truth surfaces stay aligned even when the new frontier arrives with a numerically lower snapshot id.
- Added a Q3 stress regression proving that same exact advanced-frontier snapshot identity also survives later newer-leader rejection, accepted repair, restart/resume projection, and final commit/apply completion while speculative tail is discarded and rebuilt around it.
- Hardened `RaftReplicator::install_snapshot(...)` so an accepted advanced-frontier snapshot now drops any speculative suffix that no longer attaches to the installed frontier term immediately, instead of waiting for a later epoch change to flush it; the exact installed `snapshot_id` still remains stable through subsequent handoffs.
- Added a follow-on Q3 regression proving the same advanced-frontier install still preserves a compatible surviving suffix and its recovery-gap accounting, so snapshot hardening only discards incompatible speculative tail.
- Added another Q3 regression proving that even a compatible surviving suffix is still discarded cleanly if a later newer-leader append rejects on prev-log validation after term discovery; the advanced snapshot identity remains the restart-equivalent truth surface across that handoff.
- Added a follow-on Q3 regression proving that when an advanced-frontier snapshot preserves a compatible speculative suffix, restart/export truth surfaces still publish the exact advanced snapshot identity while excluding that suffix, and a later newer-leader repair reintroduces only fresh-tail delta without changing the durable snapshot identity.
- Updated replication interface docs and runbooks to document current guarantees and explicit non-guarantees around ordering, apply progression, restart behavior, stale-snapshot no-op semantics, and the new durable-progress alignment helper.

- Added a Q3 stress regression proving that when an advanced-frontier snapshot preserves a compatible speculative suffix, a later same-frontier same-term snapshot-id refresh updates the durable identity everywhere (`status_snapshot()`, `recovery_state()`, `progress_as_follower()`, and `resume_as_follower(...)`) without perturbing the speculative-gap shape, and that a subsequent newer-leader repair still rebuilds only fresh-tail delta around the refreshed durable identity.
- Added a follow-on Q3 regression proving that the same refreshed compatible-suffix path also survives a newer-leader rejection by collapsing straight back to restart-equivalent truth on the refreshed durable snapshot identity.
- Added another Q3 regression proving stale snapshot installs remain complete no-ops even after the richer advanced-frontier + compatible speculative suffix + same-frontier snapshot-id refresh stack is in place; refreshed durable identity and speculative-gap accounting stay unchanged.
- Extended the refreshed compatible-suffix newer-leader repair stress path through heartbeat commit and final apply completion, proving the fresh-tail delta retires cleanly back to restart-equivalent truth while preserving the refreshed durable snapshot identity across status, recovery export, and resume surfaces.
- Added a follow-on Q3 regression proving those stale snapshot installs stay complete no-ops even later in that same combined stack: during newer-leader repair while fresh tail is present, after heartbeat commit retires speculative tail, and after final apply completion returns to restart-equivalent truth.
- Extended that same combined-stack stale-snapshot regression so `resume_as_follower(...)` and `RecoveryState::progress_as_follower()` stay aligned to the durable truth surface during repair, after commit advancement, and after final apply completion; stale metadata cannot perturb restart/resume projection either.
- Added a follow-on Q3 regression proving that once a newer leader has already repaired around a refreshed compatible suffix, a second same-frontier same-term snapshot refresh during that repair phase may update the durable `snapshot_id` everywhere without changing the fresh-tail live-vs-durable gap, and that later commit/apply completion preserves the newer refreshed identity through restart/resume surfaces.
- Added a follow-on Q3 regression proving that the same repair-phase second refresh still collapses cleanly if an even newer leader later rejects before repair can complete: speculative fresh tail is discarded, restart-equivalent truth returns immediately, and the twice-refreshed durable `snapshot_id` remains stable across `status_snapshot()`, `recovery_state()`, and `resume_as_follower(...)`.

### Current blockers
- None in-repo.

### Next loops
1. Extend Q2 with richer value-aware ordering or join-adjacent slices without weakening the explicit fallback contract.
2. Continue Q3 with stale/resume semantics under richer follower catch-up and snapshot-install stress, now that the status surface also rejects older-term or durable-ahead-of-live impossible states, ignores incoherent higher-frontier/lower-term snapshots, keeps accepted advanced-frontier snapshot identity exact even when ids are non-monotonic and later epoch handoffs/restart projections occur, discards incompatible speculative suffix immediately on advanced-frontier snapshot install, discards stale speculative tail on newer-leader rejection paths, preserves gap semantics across same-frontier snapshot identity refresh (including refreshes layered atop compatible advanced-frontier suffix preservation, in-flight newer-leader repair, repair-phase second refresh, and rejection back to restart-equivalent truth), keeps stale snapshot installs inert throughout the later repair/commit/apply phases of that combined stack, and cleanly resets the live-vs-durable delta when a newer leader replaces stale follower tail with fresh catch-up entries that later commit/apply back to restart-equivalent state; the next thin slice is any remaining combined-stack stress path not yet locked by an explicit regression.

## 2026-04-24

### Completed
- Extended startup-packet compatibility to recognize PostgreSQL `GSSENCRequest` probes alongside existing SSL/cancel control packets, reducing pre-auth handshake friction for clients that negotiate GSSAPI encryption before normal startup.
- Hardened startup cancel-request parsing to accept PostgreSQL's variable-length secret-key payloads instead of assuming the legacy fixed 4-byte key, improving compatibility with newer servers, poolers, and middleware that carry wrapped cancel tokens.
- Relaxed startup-version parsing to accept PostgreSQL protocol major-version 3 packets with newer minor versions, preserving the exact requested protocol version for later negotiation instead of rejecting structurally valid 3.x startup probes too early.
- Hardened cancel-request parsing to enforce PostgreSQL's 256-byte secret-key ceiling, so wrapped cancel tokens remain compatibility-safe without silently accepting out-of-spec startup control frames.
- Added regression coverage proving valid `GSSENCRequest` packets parse successfully, extended cancel requests preserve the full secret-key payload, zero process IDs still parse, exact-maximum 256-byte cancel keys remain accepted, startup packets with a standard trailing parameter terminator, multiple parameter pairs including multiple UTF-8 pairs, duplicate keys, mixed empty/non-empty parameter values, UTF-8 parameter key/value pairs, UTF-8 duplicate keys, UTF-8 duplicate keys with an empty trailing value, UTF-8 empty values, UTF-8 empty values followed by additional UTF-8 params, multiple UTF-8 params with an empty middle value, multiple UTF-8 params with a trailing empty value, multiple UTF-8 empty values, duplicate UTF-8 keys with multiple empty values, three duplicate UTF-8 keys with mixed empty/non-empty values, interleaved UTF-8 duplicate keys, interleaved UTF-8 duplicate keys with an empty value, interleaved UTF-8 empty duplicates before a later non-empty value, and interleaved UTF-8 duplicate keys with both duplicate values empty still parse structurally, 3.x minor-version startup packets parse structurally, oversized cancel keys are rejected deterministically, malformed double-NUL empty startup parameter payloads and extra trailing startup terminators are rejected as invalid pairing, invalid UTF-8 is rejected in both startup parameter keys and values, UTF-8 simple queries/password messages/parse queries and statement names including multiline UTF-8 queries and parameter OID lists, bind names plus text/empty-text/binary/empty-binary parameters including multiline UTF-8 text payloads, describe portal+statement names including multiline UTF-8 portal and statement names, close portal+statement names including multiline UTF-8 portal and statement names, and execute portal names including multiline UTF-8 names, plus raw binary, UTF-8, mixed UTF-8/binary, and multiline UTF-8 `SaslResponse` payloads without NUL bytes, binary, UTF-8, mixed UTF-8/binary, and multiline UTF-8 SASL-initial response bytes, UTF-8 SASL mechanism names including empty, UTF-8, mixed binary, and mixed UTF-8/binary initial responses, binary, UTF-8, and mixed UTF-8/binary `CopyData` payloads, binary `FunctionCall` arguments with embedded NULs, zero length, empty text, or multiline UTF-8 text payloads are preserved byte-for-byte/character-for-character, and UTF-8 `CopyFail` reasons including multiline UTF-8 payloads are preserved byte-for-byte/character-for-character, with UTF-8 parse queries still preserved when parameter OID lists are present; `p`-tag payloads with embedded NULs but no valid SASL-initial framing are rejected as `InvalidSaslInitialResponsePayload`, and malformed-length variants are rejected with the same deterministic minimum-length behavior as existing control probes.
- Added frontend-parser regression coverage proving PostgreSQL simple-query (`Q`) frames accept an empty query string (`"\0"`), keeping wire compatibility explicit for clients that rely on backend `EmptyQueryResponse` handling.
- Added frontend-parser regression coverage proving PostgreSQL password-message (`p`) frames accept an empty password payload (`"\0"`), keeping cleartext/MD5 auth framing compatible with clients that may submit a blank credential.
- Added frontend-parser regression coverage proving malformed password-message-style `p` frames with embedded NULs (`"secret\0extra\0"`) are rejected deterministically before decode side effects, matching the parser's ambiguous-`p` auth payload boundary handling.
- Added frontend-parser regression coverage proving PostgreSQL `CopyData` (`d`) frames accept an empty payload, keeping COPY streaming compatibility explicit when clients flush a zero-byte segment.
- Added frontend-parser regression coverage proving PostgreSQL SASL-initial (`p`) frames accept a declared zero-length initial response (`len=0`) as distinct from the null sentinel (`len=-1`), keeping auth framing compatible with clients that send an explicit empty first payload.
- Added frontend-parser regression coverage proving PostgreSQL `CopyFail` (`f`) frames accept an empty error string payload (`"\0"`), keeping COPY-abort framing compatible with clients that send a blank reason.
- Added malformed PostgreSQL `FunctionCall` (`F`) regression coverage proving truncated function OID fields are rejected deterministically as `InvalidFunctionCallPayload` when the 4-byte object identifier is incomplete.
- Added malformed PostgreSQL `Bind` (`B`) regression coverage proving truncated parameter-format and result-format code vectors are rejected deterministically as `InvalidBindPayload` when multi-entry format sections end mid-`i16`.
- Added malformed PostgreSQL `Bind` (`B`) regression coverage proving truncated parameter-format-count (`C`) and result-format-count (`R`) fields are rejected deterministically as `InvalidBindPayload` when only a partial `i16` payload is present.
- Added malformed PostgreSQL `Parse` (`P`) regression coverage proving truncated parameter-type-count fields (partial `i16` payload) are rejected deterministically as `InvalidParseParameterPayload`.
- Added malformed PostgreSQL `Parse` (`P`) regression coverage proving truncated per-parameter type OID fields (partial `u32` payload) are rejected deterministically as `InvalidParseParameterPayload`.
- Added startup-packet regression coverage proving unsupported protocol codes are rejected deterministically as `UnsupportedProtocolCode`, keeping pre-auth handshake boundary behavior explicit for non-V3/non-SSL/non-cancel startup probes.
- Added frontend-parser regression coverage proving unsupported message tags are rejected deterministically as `UnsupportedMessageType`, keeping wire-surface behavior explicit for unknown frontend frame types.
- Hardened the `psql` golden harness so scenarios can assert expected process exit codes via optional `tests/compat/psql-golden/expected/<scenario>.rc` artifacts (default remains `0`), enabling explicit expected-failure compatibility cases without brittle ad hoc checks.
- Added baseline exit-code artifact for the existing bootstrap scenario (`01_bootstrap_and_simple_query.rc`) and updated suite docs to describe `.rc` usage for unsupported-yet-expected flows.
- Continued PostgreSQL extended-query malformed-frame hardening with two new deterministic parser regressions:
  - `FunctionCall` (`F`) now explicitly rejects truncated argument-count fields (partial `i16` payload) as `InvalidFunctionCallPayload`.
  - `Bind` (`B`) now explicitly rejects truncated parameter-count fields (partial `i16` payload) as `InvalidBindPayload`.
- Re-ran full safety gates after each atomic delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`) with all checks green.
- Added malformed PostgreSQL `FunctionCall` (`F`) regression coverage proving truncated result-format code fields (partial `i16` payload) are rejected deterministically as `InvalidFunctionCallPayload`.
- Scaffolded a real-client `psql` golden compatibility harness (`scripts/run_psql_golden.sh`) plus first scenario/expected artifact set under `tests/compat/psql-golden/`, including deterministic output normalization and local run/update docs.
- Commits pushed:
  - `67ac011` — `test(protocol): reject truncated function-call oid field`
  - `a5b0f7e` — `test(protocol): reject truncated bind format-code vectors`
  - `dc4debc` — `test(protocol): reject truncated bind format-count fields`
  - `65bca82` — `test(protocol): reject truncated function call arg-count field`
  - `fbbba5e` — `test(protocol): reject truncated bind parameter-count field`
  - `b444b34` — `test(protocol): reject truncated function call result-format code field`
  - `175dabe` — `test(protocol): reject truncated parse parameter fields`
  - `2923868` — `test(protocol): cover unsupported startup protocol codes`
  - `a74783d` — `test(protocol): cover unterminated execute portal name`
  - `cf8a2be` — `feat(protocol): parse startup gssenc request`
  - `fffd5a4` — `feat(protocol): accept extended cancel request keys`
  - `f6b8fc5` — `feat(protocol): accept startup protocol minor versions`
  - `339d8c8` — `feat(protocol): bound cancel request secret keys`
  - `d19c1be` — `test(protocol): cover max-length cancel request keys`
  - `7b2f734` — `test(protocol): cover empty simple query frame`
  - `d3e4875` — `test(protocol): cover empty copy fail frame`
  - `aea498b` — `test(protocol): cover empty password message frame`
  - `55f5c85` — `test(protocol): cover empty copy data frame`
  - `4b75a06` — `test(protocol): cover empty sasl initial response`
  - `d77a190` — `docs(roadmap): record pushed gssenc protocol commit`
  - `350c0a3` — `test(protocol): cover embedded-null password payload`

### Current blockers
- None in-repo.

### Next loops
1. Continue malformed-frame boundary hardening in extended-query lifecycle paths, prioritizing partial-field truncation cases not yet covered by deterministic error assertions.
2. Keep loops atomic: protocol delta + full fmt/clippy/test validation + focused commit.

## 2026-04-23

### Completed
- Landed CI compatibility scorecard automation (`fcc287c`): CI now captures real `cargo test --workspace` output, generates machine-readable and Markdown scorecards, and uploads them as a `compatibility-scorecard` artifact.
- Added `scripts/generate_compat_scorecard.py` to bucket real test outcomes into protocol/client flow, SQL/parser, transaction, durability, replication, and execution compatibility categories with explicit pass/fail totals, top failing categories, and a baseline trend hook.
- Added scorecard docs and baseline (`docs/compatibility/scorecard.md`, `docs/compatibility/scorecard.baseline.json`) plus generated fixtures (`scorecard.latest.json`, `scorecard.latest.md`) so local and CI workflows stay aligned.
- Extended compatibility scorecard trend reporting to emit per-bucket failed-count deltas against baseline (`trend_hook.bucket_failed_delta`) in addition to overall failed-count drift, so CI can distinguish broad regressions from category-specific movement.
- Regenerated scorecard fixtures from a fresh `cargo test --workspace -- --color never` log so checked-in JSON/Markdown outputs include the new bucket-delta trend section.
- Added two explicit high-priority autonomous-loop queue items to `docs/roadmap/no-nvidia-bootstrap-plan.md`: (1) golden-wire `psql` compatibility suite and (2) CI compatibility scorecard, each with concrete goals and acceptance criteria so the standard autoloop can pick them up as normal roadmap work.
- Re-ran lock-guard bootstrap checks for the autonomous loop (`cargo 1.94.0`, `rustc 1.94.0`) with stale-lock cleanup semantics before any repository work.
- Re-validated full repository safety gates in a no-delta maintenance slice (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.
- Added malformed PostgreSQL `FunctionCall` (`F`) regression coverage proving truncated argument-length fields are rejected deterministically as `InvalidFunctionCallPayload`.
- Added malformed PostgreSQL `FunctionCall` (`F`) regression coverage proving truncated argument-format vectors (`C > 1` with missing format entries) are rejected deterministically as `InvalidFunctionCallPayload`.
- Re-ran full repository safety gates after each protocol hardening delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`) with all checks green.
- Added malformed SASL-initial-response regression coverage proving protocol parser rejects negative non-null response lengths (`-2`) and empty mechanism names as deterministic `InvalidSaslInitialResponsePayload` errors.
- Added malformed PostgreSQL `FunctionCall` (`F`) regression coverage proving negative result-format vector cardinalities (`R=-1`) are rejected deterministically as `InvalidFunctionCallPayload`, plus explicit boundary checks for a truncated result-format-count field and `R=1` negative format-code (`-1`) payload rejection.
- Re-ran full repository safety gates after the protocol hardening delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`) with all checks green.
- Commits pushed:
  - `f164e88` — `test(protocol): reject truncated function call arg length field`
  - `ffe6353` — `test(protocol): reject truncated function call format vectors`
  - `2da083f` — `test(protocol): cover malformed sasl initial payload boundaries`
- Logged this checkpoint so clean/no-delta and protocol-hardening autonomous runs remain auditable while preserving GPU-first and WAL-before-visibility invariants.

### Current blockers
- None in-repo.

### Next loops
1. Continue PostgreSQL extended-query lifecycle hardening with additional malformed-frame regressions around boundary and payload invariants.
2. Keep loops atomic: protocol delta + full fmt/clippy/test validation + focused commit.

## 2026-04-22

### Completed
- Extended PostgreSQL session-reset compatibility so `SET LOCAL TRANSACTION ...` mode lists are accepted as `ResetAll` no-op aliases (matching existing `SET TRANSACTION ...` handling), reducing parser friction for clients that scope transaction characteristics locally.
- Added regression coverage for accepted `SET LOCAL TRANSACTION` forms (`READ ONLY`, `READ WRITE, DEFERRABLE`) plus malformed-form rejection (`SET LOCAL TRANSACTION`, `SET LOCAL TRANSACTION NOW`) to keep parser behavior deterministic.
- Added malformed `Bind` regression coverage proving negative result-format codes (`R=1`, code `-1`) are rejected deterministically as `InvalidBindPayload`, preserving strict result-format validation for signed underflow wire values.
- Added malformed `FunctionCall` regression coverage proving negative shared argument-format codes (`C=1`, code `-1`) are rejected deterministically as `InvalidFunctionCallPayload`, preserving strict extended-query format-code validation for signed underflow inputs.
- Added malformed `Bind` regression coverage proving negative shared parameter-format codes (`C=1`, code `-1`) are rejected deterministically as `InvalidBindPayload`, preserving strict frontend frame/type validation even for signed underflow format-code inputs.
- Re-ran the autonomous safety gate from a lock-guarded cron loop with Rust bootstrap verification (`cargo 1.94.0`, `rustc 1.94.0`).
- Re-validated repository health in a no-delta maintenance pass (`cargo fmt --all --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`) to confirm parser/engine invariants remain stable before the next protocol-hardening delta.
- Added this maintenance checkpoint so cron runs that intentionally ship no code delta still leave an auditable trail in the roadmap log.
- Verified repository health remains green with no pending source deltas (`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`).
- Added this implementation-log checkpoint so autonomous loops that land in a clean/no-delta state are explicitly auditable.
- Added extended-query `Bind` regression coverage proving shared parameter-format vectors (`C=1`) decode correctly when multiple parameters are present (`N=2`), including mixed `NULL`/non-`NULL` parameter payloads and multi-entry result-format vectors (`R=2`).
- Added malformed `Bind` regression coverage proving negative result-format vector cardinalities (`R=-1`) are rejected deterministically as `InvalidBindPayload`, preserving strict frontend frame-boundary/type invariants.
- Re-validated the full safety gate after the parser-test delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.

### Current blockers
- None in-repo.

### Next loops
1. Continue PostgreSQL extended-query lifecycle hardening with additional malformed-frame regressions around boundary and target/payload invariants.
2. Keep loops atomic: protocol delta + full fmt/clippy/test validation + focused commit.

## 2026-04-21

### Completed
- Added extended-query `Bind` regression coverage proving zero-parameter frames (`N=0`) accept a single shared binary parameter-format code (`C=1`, code `1`) and deterministic binary result-format vectors (`R=1`, code `1`).
- Added extended-query regression coverage proving PostgreSQL `Bind` (`B`) frames with zero parameters (`N=0`) accept a single shared text parameter-format code (`C=1`, code `0`) and deterministic text result-format vectors (`R=1`, code `0`).
- Added extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) frames with zero arguments (`N=0`) accept a single shared text format code (`C=1`, code `0`) and decode deterministically with text result format (`R=0`).
- Added malformed `Bind` (`B`) regression coverage proving zero-parameter frames (`N=0`) still validate shared parameter-format codes (`C=1`) and deterministically reject out-of-range values (`2`) as `InvalidBindPayload`.
- Added malformed extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) frames reject out-of-range positive result-format codes (`2`) deterministically as `InvalidFunctionCallPayload`.
- Extended reset-command compatibility to accept `RESET SESSION AUTHORIZATION TO DEFAULT` and `RESET SESSION AUTH TO DEFAULT` as `ResetAll` no-op aliases, reducing parser friction for PostgreSQL-style reset probes that include an explicit `TO` keyword.
- Added regression coverage for the new `RESET SESSION AUTH* TO DEFAULT` forms across baseline acceptance, statement-terminator handling, and malformed extra-token rejection to keep parser behavior deterministic.
- Updated reset-command error guidance so documented accepted forms now include optional `TO DEFAULT` variants for session auth reset aliases.
- Added malformed extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) frames reject mismatched argument-format cardinality vectors (`C=2`, `N=1`) deterministically as `InvalidFunctionCallPayload`.
- Added malformed `FunctionCall` boundary regression coverage proving declared argument lengths that overrun payload bytes (`N=1`, length `3`, only two bytes present) are rejected deterministically as `InvalidFunctionCallPayload`.
- Added malformed `Bind` boundary regression coverage proving truncated per-parameter length fields (only three of four bytes present) are rejected deterministically as `InvalidBindPayload`.
- Re-validated the full safety gate after the parser delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.

### Current blockers
- None in-repo.

### Next loops
1. Continue PostgreSQL extended-query lifecycle hardening with malformed-frame regressions around boundary and target/payload invariants.
2. Keep loops atomic: protocol delta + full fmt/clippy/test validation + focused commit.

## 2026-04-20

### Completed
- Added malformed `FunctionCall` regression coverage proving argument-value boundaries remain strict when declared argument lengths overrun payload bytes (`N=1`, arg length `3`, only two bytes present), with deterministic `InvalidFunctionCallPayload` rejection.
- Added malformed `Bind` regression coverage proving parameter-value boundaries remain strict when wire lengths overrun payload bytes (`N=1`, declared length `3`, only two bytes present), with deterministic `InvalidBindPayload` rejection.
- Added extended-query `Bind` regression coverage proving default parameter-format semantics (`C=0`) decode deterministically when parameters are present (`N>0`, including `NULL`) and still preserve explicit result-format decoding (`R=1`).
- Added extended-query parser regression coverage proving PostgreSQL `Parse` (`P`) frames accept unnamed-statement flows with zero parameter OIDs (`"\0SELECT 1\0" + nparams=0`), keeping startup/prepare compatibility explicit for clients that use the unnamed prepared statement.
- Added extended-query `Bind` regression coverage proving unnamed portal + unnamed statement payloads with zero parameter formats/values (`C=0`, `N=0`) and zero result formats (`R=0`) decode deterministically, keeping default-format unnamed bind flows explicit.
- Added malformed `Bind` regression coverage proving result-format cardinality sections still enforce full payload boundaries (`R=2` with only one provided format code now remains a deterministic `InvalidBindPayload` rejection).
- Added malformed extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) frames still enforce cardinality when no arguments are supplied (`N=0`): multi-entry argument-format vectors (`C=2`) are now explicitly covered as deterministic `InvalidFunctionCallPayload` rejections.
- Added malformed extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) frames with multi-entry argument-format sections still reject out-of-range per-argument format codes (for example `C=2` with a `2` code) deterministically as `InvalidFunctionCallPayload`.
- Added extended-query regression coverage proving PostgreSQL `FunctionCall` (`F`) parsing accepts per-argument format-code vectors (`C` format codes matching `N` arguments) and decodes mixed text/binary argument payloads without relaxing malformed-frame rejection behavior.
- Aligned PostgreSQL `Bind` (`B`) frontend parsing with protocol cardinality semantics by accepting multiple result-format codes (`R > 1`) instead of rejecting them as malformed; parser still enforces per-code validity (`0` or `1`) and full frame-boundary correctness.
- Added regression coverage proving multi-result-format bind frames parse deterministically into `FrontendMessage::Bind { result_format_codes: vec![...] }` without weakening malformed-payload rejection paths.
- Re-validated the full safety gate after the bind-cardinality fix (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.
- Added malformed extended-query lifecycle regressions proving unnamed lowercase and uppercase `Describe`/`Close` targets (`D p\0`, `D S\0`, `C s\0`, `C P\0`) still enforce strict frame boundaries and reject trailing bytes deterministically as `UnterminatedDescribeName` / `UnterminatedCloseName`.
- Added malformed `Bind` regression coverage proving multi-result-format sections still enforce per-code validity: payloads with `R=2` and an out-of-range format code (`2`) now remain deterministically rejected as `InvalidBindPayload`.
- Hardened PostgreSQL extended-query compatibility by accepting lowercase `Describe`/`Close` targets (`s`/`p`) in addition to canonical uppercase (`S`/`P`), reducing parser friction for clients that emit lowercase target tags while preserving strict invalid-target rejection.
- Added regression coverage proving lowercase target decoding parity for both `Describe` and `Close` lifecycle messages.
- Added extended-query lifecycle regressions covering unnamed `Describe`/`Close` targets (`D S\0`, `D p\0`, `C S\0`, `C p\0`) so PostgreSQL unnamed statement/portal flows stay explicitly validated across uppercase/lowercase target tags.
- Clarified frontend parser diagnostics for `Describe`/`Close` target validation to match the implemented case-insensitive target contract (`S/s`, `P/p`).
- Re-validated the full safety gate after the parser delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.
- Added malformed extended-query lifecycle regressions proving lowercase `Describe`/`Close` targets (`p`/`s`) still enforce C-string termination boundaries (`D pportal`, `C sstmt`) and reject truncated payloads deterministically as `UnterminatedDescribeName` / `UnterminatedCloseName`.
- Added bind-cardinality regressions for zero-parameter frames: parser now has explicit coverage that `B` payloads reject multi-entry parameter format vectors when `N=0` (`C=2` -> `InvalidBindPayload`) while continuing to accept single shared format vectors (`C=1`, `N=0`) with deterministic decode.
- Added function-call parity regression for zero-argument frames: `F` payloads now explicitly cover acceptance of a single shared argument format code when `N=0` (while existing coverage still rejects invalid multi-code vectors for zero args).
- Added execute-frame regression for unnamed portal/unlimited-row semantics: parser now explicitly covers `E` with empty portal name and `max_rows = 0` decoding deterministically as unlimited execution on the unnamed portal.

### Current blockers
- None in-repo.

### Next loops
1. Continue PostgreSQL extended-query lifecycle hardening with malformed-frame regressions around boundary and target/payload invariants.
2. Keep loops atomic: protocol delta + full fmt/clippy/test validation + focused commit.

## 2026-04-19

### Completed
- Added malformed extended-query lifecycle regressions for `Describe`/`Close` (`D`/`C`) payload boundaries, proving trailing bytes after a complete C-string name (`...\0\xFF`) are rejected deterministically as `UnterminatedDescribeName` / `UnterminatedCloseName` before lifecycle state can advance.
- Added malformed extended-query lifecycle regressions for `Parse` (`P`) payload boundaries, proving frames with declared zero parameter-type count still reject trailing bytes (`...\0\0\0\xFF`) deterministically as `InvalidParseParameterPayload`.
- Added malformed extended-query lifecycle regressions for `Execute` (`E`) payload boundaries, proving short `max_rows` sections and embedded-NUL portal-name payloads (`portal\0extra\0...`) are rejected deterministically as `InvalidExecutePayload` before execution state can advance.
- Added malformed extended-query lifecycle regressions for target-only `Describe`/`Close` frontend frames (`D`/`C` with missing trailing C-string terminators), asserting deterministic `UnterminatedDescribeName` / `UnterminatedCloseName` rejection for truncated payload boundaries.
- Added malformed auth-frame regression coverage proving SASL initial-response mechanism names with invalid UTF-8 bytes are rejected deterministically as `InvalidUtf8` before payload-length handling.
- Added PostgreSQL extended-query malformed-frame regression coverage to assert deterministic `InvalidUtf8` rejection when non-UTF8 bytes appear in decoded text fields across lifecycle messages (`Parse` statement/query, `Bind` portal/statement, `Describe` name, `Close` name, `Execute` portal).
- Added malformed SASL initial-response boundary regressions proving `p` auth frames reject trailing bytes after null initial-response lengths (`-1`) and reject declared initial-response length/payload mismatches deterministically as `InvalidSaslInitialResponsePayload`.
- Added startup/auth malformed-frame regressions for dangling startup parameter key segments and zero-length SASL initial-response frames with trailing payload bytes, keeping pairing and SASL length invariants deterministic.
- Re-validated the full safety gate after the parser regression delta (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with all checks green.

### Current blockers
- None in-repo.

### Next loops
1. Continue extended-query wire-protocol hardening with additional malformed-frame regressions around lifecycle boundaries and decode invariants.
2. Keep loops atomic: parser delta + full fmt/clippy/test validation + focused commit.

## 2026-04-18

### Completed
- Re-validated the current mainline with the full autonomous safety gate (`cargo fmt --all --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) after lock-guarded cron bootstrap, confirming the GPU-first + WAL-before-visibility invariants remain green with no pending code deltas in this loop.
- Added malformed extended-query boundary regressions for truncated tail sections in `Bind` and `FunctionCall` frontend frames, proving both paths deterministically reject missing result-format payload bytes (`InvalidBindPayload` / `InvalidFunctionCallPayload`) before decode side effects.
- Extended PostgreSQL session-control compatibility so `NOTIFY channel, payload` accepts prefixed string literal forms (`E'...'`, `B'...'`, `X'...'`, `U&'...'`) as reset no-op payload fragments, reducing parser friction for clients that emit typed literal payloads.
- Hardened `NOTIFY` payload validation for prefixed string literals by adding malformed unterminated regression cases (`E'`, `B'`, `X'`, `U&'`) to keep deterministic malformed-input rejection behavior explicit.
- Added fixed-length frontend control frame regressions for `Terminate`, `Sync`, and `Flush` so one-byte payload drift is explicitly rejected with deterministic `LengthMismatch` behavior (matching the existing `CopyDone` contract checks).
- Expanded `NOTIFY` prefixed-literal compatibility regressions to cover lowercase prefixes (`e'...'`, `b'...'`, `x'...'`, `u&'...'`) plus lowercase malformed unterminated variants for strict parity with case-insensitive SQL keyword handling.
- Hardened PostgreSQL frontend frame boundary validation to reject invalid message length fields below the protocol minimum (`len < 4`) with a dedicated `InvalidLengthField` error before payload decode.
- Added malformed-frame regression coverage for underflowed frontend length declarations (`Q` frame length `3`) so extended-query/simple-query boundary rejection stays deterministic.
- Added malformed extended-query boundary regressions covering empty `Describe`/`Close` payloads and negative cardinality fields in `Bind`/`FunctionCall` frames, proving deterministic rejection for truncated lifecycle/control frames before decode side effects.
- Added malformed extended-query boundary regressions for negative cardinality fields in `Parse` parameter-type counts, `Bind` parameter-format counts, and `FunctionCall` argument-format counts, keeping deterministic rejection coverage explicit for signed underflow payloads across the full parse/bind/call lifecycle.
- Added malformed extended-query boundary regressions proving `Parse`/`Bind`/`Execute` reject trailing payload bytes after structurally complete fields, keeping frame-boundary validation deterministic for lifecycle/control messages.
- Added malformed `FunctionCall` frame boundary regression proving trailing bytes after a structurally complete payload are rejected deterministically (`InvalidFunctionCallPayload`) before decode side effects.
- Added malformed `FunctionCall` regression coverage proving negative result-format codes are rejected deterministically (`InvalidFunctionCallPayload`).
- Added malformed extended-query boundary regressions proving `Bind` and `FunctionCall` reject wire argument/value lengths below PostgreSQL's null sentinel (`-1`) so signed underflow lengths (`-2`) are deterministically rejected as invalid payloads.
- Hardened startup packet framing to reject declared length fields below the PostgreSQL minimum startup frame size (`len < 8`) with a dedicated `InvalidLengthField` error, plus regression coverage for underflowed startup length declarations.

### Current blockers
- None in-repo.

### Next loops
1. Continue the PostgreSQL wire-protocol hardening track with additional malformed-frame regressions around extended-query lifecycle message boundaries.
2. Keep each loop atomic: parser/engine delta + full fmt/clippy/test validation + focused commit.

## 2026-04-17

### Completed
- Verified bootstrap quality gates remain green on current mainline (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) with no pending code deltas, confirming the GPU-first/WAL-before-visibility baseline is stable before the next feature loop.
- Added this status checkpoint so roadmap history explicitly records a clean validation pass (useful for auditing autonomous loop runs where no code changes are required).
- Hardened PostgreSQL frontend `FunctionCall` (`F`) frame parsing to enforce format-code cardinality rules (`0`, `1`, or exact argument count), preventing malformed mixed-cardinality payloads from being accepted.
- Added regression coverage for malformed `FunctionCall` format-code cardinality so extended-query parser rejection behavior stays deterministic under malformed inputs.
- Hardened PostgreSQL frontend `Bind` (`B`) frame parsing to reject result-format sections with more than one format code, aligning with wire-level cardinality requirements and preventing malformed multi-result-format payloads from being accepted.
- Added malformed-frame regression coverage for oversized `Bind` result-format code sections so extended-query parser rejection behavior remains deterministic for invalid result-format cardinality inputs.
- Hardened frontend C-string parsing (`SimpleQuery`, `PasswordMessage`, `Describe`, `Close`, `CopyFail`) to reject embedded NUL bytes instead of silently accepting malformed multi-segment payloads.
- Added malformed-frame regression coverage for embedded-NUL C-string payloads so parser rejection stays deterministic for these frontend message paths.
- Hardened PostgreSQL frontend auth-message compatibility by accepting zero-length `SaslResponse` (`p`) frames as valid SASL continuation payloads instead of misclassifying them as unterminated password messages.
- Added regression coverage for empty `SaslResponse` frames so `p`-tag auth parsing remains deterministic across password, SASL-initial, and SASL-response payload classes.
- Hardened PostgreSQL frontend `Execute` (`E`) frame parsing to reject negative `max_rows` values, preventing signed wire values from being reinterpreted as oversized unsigned limits.
- Added malformed-frame regression coverage for negative `Execute.max_rows` payloads so extended-query parser rejection behavior stays deterministic under invalid row-limit inputs.
- Added parser regression coverage for `COMMIT|END|ROLLBACK|ABORT WORK AND CHAIN` aliases so PostgreSQL transaction-control `WORK` scope variants remain explicitly validated for chain semantics.
- Added engine transaction-state regression coverage (immediate + enqueue paths) proving `END WORK AND CHAIN` and `ABORT WORK AND CHAIN` reopen transaction context exactly like existing `... AND CHAIN` aliases.
- Added startup-packet regression coverage proving protocol-v3 startup frames with an empty parameter map (`...\0`) parse successfully instead of being treated as malformed payloads.
- Added startup control-frame length regressions proving malformed SSL/cancel request frame sizes are rejected with deterministic `LengthMismatch` errors.

### Current blockers
- None in-repo. Core Rust/toolchain and test gates are available in this environment.

### Next loops
1. Extend PostgreSQL wire protocol coverage beyond current frontend parsing skeleton (next likely targets: additional extended-query lifecycle frames and stricter edge-case framing regressions).
2. Continue compatibility-surface hardening with deterministic malformed-input rejection tests, while preserving no-op alias behavior expected by PostgreSQL clients.
3. Keep each loop atomic: parser/engine delta + full fmt/clippy/test validation + focused commit.

## 2026-04-16

### Completed
- Extended reset-command parser compatibility to accept scoped role reset aliases `RESET SESSION ROLE` and `RESET LOCAL ROLE` as `ResetAll` no-op forms (matching existing `RESET ROLE` handling), reducing parser friction for PostgreSQL-compatible clients/proxies that scope role-reset probes explicitly during bootstrap/reset workflows.
- Added regression coverage for scoped-role reset alias acceptance (including statement terminators), keeping reset-command parsing deterministic across unscoped and scoped role-reset forms.
- Extended session-reset parser compatibility to accept `SET SESSION ROLE ...` and `SET LOCAL ROLE ...` as `ResetAll` no-op aliases (matching existing `SET ROLE ...` handling), reducing parser friction for PostgreSQL clients that scope role changes explicitly during bootstrap/reset flows.
- Added regression coverage for scoped-role alias acceptance (`SESSION`/`LOCAL`) plus malformed missing-target rejection to keep parser behavior deterministic.
- Extended reset-command compatibility to accept `RESET SESSION AUTHORIZATION DEFAULT` and `RESET SESSION AUTH DEFAULT` as `ResetAll` no-op aliases, reducing parser friction for PostgreSQL-compatible clients/proxies that emit explicit default-reset variants during session bootstrap/cleanup flows.
- Added regression coverage for the new `RESET SESSION AUTH* DEFAULT` aliases (including statement terminators) and malformed extra-token rejection, keeping reset-command parsing deterministic.
- Added regression coverage for escaped-double-quote quoted identifiers across session-control no-op aliases (`CLOSE`, `UNLISTEN`, `LISTEN`, `NOTIFY`), proving parser acceptance for PostgreSQL-style quoted names like `"updates""channel"` without relaxing malformed syntax checks.
- Added `NOTIFY channel, "payload"` regression coverage for double-quoted payload fragments (including embedded escaped quotes and commas) plus malformed unterminated double-quote rejection, hardening parser stability without relaxing strict malformed payload checks.
- Hardened quoted-identifier parsing for reset/session-control aliases to reject zero-length delimited identifiers (`""`) across `DEALLOCATE`/`CLOSE`/`LISTEN`/`UNLISTEN`/`NOTIFY` and `SET ROLE`, with regression coverage proving malformed empty-quoted forms are rejected deterministically.
- Added parser regression coverage proving escaped-double-quote quoted identifiers remain accepted for `SET ROLE` and `CLOSE` aliases after the empty-quoted-identifier hardening.
- Extended `SET` session-reset alias parsing to accept identifier and quoted-identifier role/auth targets (`SET ROLE app_role`, `SET ROLE "app role"`, `SET SESSION AUTHORIZATION "app user"`, `SET SESSION AUTH "app user"`) as `ResetAll` no-op aliases, reducing parser friction for PostgreSQL clients that emit quoted principals in reset flows.
- Added regression coverage proving quoted role/auth targets are accepted and unterminated quoted targets are still rejected with `InvalidSet`, keeping reset alias parsing deterministic.

## 2026-04-15

### Completed
- Extended session-control parser compatibility to accept quoted identifier aliases in `DEALLOCATE`/`CLOSE`/`LISTEN`/`UNLISTEN`/`NOTIFY` reset no-op forms (including quoted names with spaces and embedded commas), reducing parser friction for PostgreSQL clients that emit quoted prepared-statement/cursor/channel names during bootstrap/reset flows.
- Added regression coverage for quoted identifier acceptance plus unterminated-quoted identifier rejection across deallocate/close/listen/notify/unlisten paths, while retaining strict malformed comma validation for `NOTIFY` payload forms.
- Hardened `UNLISTEN` reset-alias parsing to reject comma-delimited channel fragments (`UNLISTEN a,b`) so malformed multi-target listener cleanup probes are no longer accepted as valid no-ops.
- Added regression coverage for malformed comma-delimited `UNLISTEN` input to keep listener reset alias parsing deterministic.
- Hardened `DEALLOCATE` reset-alias parsing to reject comma-delimited target fragments (`DEALLOCATE a,b`, `DEALLOCATE PREPARE a,b`) so malformed multi-target probes no longer slip through as valid `ResetAll` no-ops.
- Added regression coverage for malformed comma-delimited `DEALLOCATE` forms to keep prepared-statement cleanup alias parsing deterministic.
- Extended session-reset parser compatibility to accept `DEALLOCATE PREPARED name` as a `ResetAll` no-op alias alongside existing `DEALLOCATE PREPARE name` handling, reducing parser friction for PostgreSQL-style clients that emit the alternate prepared-statement cleanup keyword.
- Added regression coverage for `DEALLOCATE PREPARED name` acceptance plus malformed-form rejection (`DEALLOCATE PREPARED`, extra-token variants) and updated reset-command error text to reflect the expanded deallocate alias contract.
- Extended session-control parser compatibility to accept `CLOSE name` cursor cleanup probes as `ResetAll` no-op aliases (in addition to `CLOSE ALL`), reducing parser friction for PostgreSQL clients that explicitly close named cursors during reset flows.
- Added regression coverage for `CLOSE name` acceptance (including terminator handling) and malformed multi-token rejection (`CLOSE name NOW`), keeping no-op alias parsing deterministic.
- Extended session-reset parser compatibility to accept `SET SESSION CHARACTERISTICS AS TRANSACTION ...` forms as `ResetAll` no-op aliases, reducing parser friction for PostgreSQL clients that emit transaction-default probes during session setup.
- Added regression coverage for accepted `SET SESSION CHARACTERISTICS AS TRANSACTION ...` aliases plus malformed-form rejection when the transaction characteristics suffix is missing.
- Hardened PostgreSQL session-control `NOTIFY` alias parsing to reject comma-only payload fragments (for example `NOTIFY channel, ,`), preventing malformed reset probes from being accepted as valid no-op aliases.
- Tightened compact `NOTIFY channel,payload` parsing to reject malformed double-comma forms (`NOTIFY channel,,payload` / `NOTIFY channel,, payload`) while preserving accepted single-comma payload aliases.
- Added regression coverage for malformed comma-only and double-comma `NOTIFY` payload variants to keep parser behavior deterministic.
- Hardened `NOTIFY channel[, payload]` alias parsing to reject unquoted multi-fragment payload streams (`NOTIFY channel, payload, extra`) so malformed comma-delimited payload probes no longer pass as valid no-op resets.
- Added regression coverage proving quoted payloads with embedded commas (for example JSON string payloads) remain accepted while multi-fragment payload forms are rejected deterministically.
- Hardened `NOTIFY` alias payload validation to reject unterminated quoted payload fragments (for example `NOTIFY channel, 'unterminated`), preventing malformed quote state from being accepted as reset no-ops.

## 2026-04-14

### Completed
- Hardened PostgreSQL session-control `NOTIFY` alias parsing to accept compact payload forms without whitespace after the channel comma (for example `NOTIFY channel,'payload'`) as `ResetAll` no-op aliases, reducing parser friction for clients that emit tightly packed notify probes.
- Extended `NOTIFY` alias parsing to also accept payload tokens prefixed directly by a comma after whitespace-separated channel names (for example `NOTIFY channel ,'{"ok":true}'`), preserving deterministic no-op handling for additional compact client payload formats.
- Added regression coverage for compact `NOTIFY channel,'payload'` acceptance plus malformed trailing-comma rejection (`NOTIFY channel,`) and updated reset-command error text to include `NOTIFY` in the documented alias set.
- Updated the compatibility matrix session-control alias row to include `NOTIFY`, keeping phase-gated parser-contract docs aligned with the implemented no-op alias surface.
- Extended session-control parser compatibility to accept `UNLISTEN ALL` as a `ResetAll` no-op alias (alongside existing `UNLISTEN`/`UNLISTEN *`/`UNLISTEN channel` forms), reducing parser friction for PostgreSQL clients that emit `ALL`-style listener reset probes.
- Added regression coverage for `UNLISTEN ALL` acceptance (including statement terminators) and updated command-reference docs/error text to reflect the expanded `UNLISTEN` alias contract.
- Extended session-control parser compatibility to accept PostgreSQL `NOTIFY channel[, payload]` forms as `ResetAll` no-op aliases, reducing parser friction for clients that probe pub/sub notification lifecycles during bootstrap/reset flows.
- Added regression coverage for `NOTIFY` alias acceptance (with and without payload), malformed-form rejection (`NOTIFY`, malformed payload form), and statement-terminator handling.
- Updated command-reference docs to reflect the expanded session-control alias contract.
- Extended session-reset parser compatibility to accept PostgreSQL-style `SET ROLE {NONE|DEFAULT}` and `SET SESSION AUTHORIZATION value` / `SET SESSION AUTH value` forms as `ResetAll` no-op aliases, reducing parser friction for clients that reinitialize role/auth context during connection reset flows.
- Added regression coverage for accepted `SET ROLE`/`SET SESSION AUTH*` aliases, malformed-form rejection (`SET ROLE`, missing auth target), and statement-terminator handling to keep parser behavior deterministic.
- Updated command-reference docs to reflect the expanded reset alias contract.
- Extended admin flush compatibility to accept PostgreSQL `CHECKPOINT` as a `Flush` alias, so bootstrap control paths can ingest checkpoint probes without client-side command rewriting.
- Added regression coverage for `CHECKPOINT` acceptance plus malformed-form rejection (`CHECKPOINT NOW`) and statement-terminator handling.
- Extended session-reset parser compatibility to accept PostgreSQL-style prepared-statement cleanup forms `DEALLOCATE name` and `DEALLOCATE PREPARE name` as `ResetAll` no-op aliases (in addition to `DEALLOCATE ALL`), reducing bootstrap parser friction for clients that emit explicit prepared-statement teardown probes.
- Added regression coverage for `DEALLOCATE name` and `DEALLOCATE PREPARE name` acceptance plus malformed-form rejection (`DEALLOCATE PREPARE`, extra-token variants) and statement-terminator handling, keeping parser behavior deterministic.
- Extended session-control parser compatibility to accept `LISTEN channel` as a `ResetAll` no-op alias, reducing bootstrap parser friction for PostgreSQL clients that probe pub/sub lifecycle commands during connection setup/reset flows.
- Added regression coverage for `LISTEN channel` acceptance plus malformed-form rejection (`LISTEN`, extra-token variants) and statement-terminator handling.
- Updated command-reference docs/error text to reflect the expanded `DEALLOCATE`/`LISTEN` compatibility contract.

## 2026-04-13

### Completed
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to percent-decode `%HH` telemetry payloads before delimiter splitting, allowing URL-encoded streams like `wal%2Cpending-batch%7CACTIVE%20TXN` to decode without pre-normalization; added regression coverage for mixed encoded delimiters and labels.
- Extended frontend auth-message parsing to distinguish cleartext `PasswordMessage` (`p`) from SASL authentication payloads, adding explicit `SaslInitialResponse` / `SaslResponse` decoding with strict payload-length validation and malformed-frame regression coverage.
- Hardened frontend `Bind` (`B`) and `FunctionCall` (`F`) decoding to reject non-PostgreSQL format codes (values other than text=0 or binary=1), with regression coverage for invalid parameter/result format payloads.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to also split on backslash (`\\`) delimiters, allowing Windows-style streams like `wal\\active_txn\\apply_visible_gap` to decode without pre-normalization.
- Extended frontend-message parsing scaffolding to decode PostgreSQL `FunctionCall` (`F`) frames (function OID, argument/result format codes, nullable argument payloads) with strict payload-length validation and regression coverage for malformed argument bodies.
- Extended frontend-message parsing scaffolding to decode PostgreSQL copy-protocol frontend frames: `CopyData` (`d`), `CopyDone` (`c`), and `CopyFail` (`f`), with deterministic framing/null-termination validation and regression coverage for malformed payloads.
- Added a PostgreSQL startup/auth packet parser skeleton in `gpu_db_protocol` (`parse_startup_packet`) covering protocol v3 startup parameters plus SSL and cancel request decoding, with deterministic regression coverage for framing/length and malformed parameter payloads.
- Added a baseline `SessionLifecycle` state machine (`Startup -> Authenticating -> Ready -> InTransaction -> Terminating/Closed`) with explicit transition errors so session bootstrap/auth/transaction teardown sequencing is testable before full wire-protocol integration.
- Added frontend-message parsing scaffolding (`parse_frontend_message`) for `SimpleQuery`, `Sync`, and `Terminate` frames with strict length/null-termination validation, establishing a deterministic wire-level simple-query path before full protocol coverage.
- Extended frontend-message parsing scaffolding to decode PostgreSQL `Flush` (`H`) frames with strict fixed-length validation, so simple query and extended-query flow-control messages now share deterministic wire-level framing checks.
- Extended frontend-message parsing scaffolding to decode PostgreSQL `Parse` (`P`) frames (statement name, SQL text, parameter type OIDs) with strict null-termination and parameter-payload validation, providing a deterministic baseline for extended-query statement preparation.
- Extended frontend-message parsing scaffolding to decode PostgreSQL extended-query control/data frames: `Describe` (`D`), `Close` (`C`), `Execute` (`E`), and `Bind` (`B`), with strict target/null-termination/length validation and deterministic payload decoding for portal/statement lifecycle and bound-parameter execution paths.
- Added a new `gpu_db_storage` crate and wired it into the workspace member list so the Phase 0 repository skeleton now includes an explicit storage boundary alongside protocol/planner/execution/wal/replication modules.
- Introduced bootstrap storage contracts (`TupleStore`, `seq_scan_open`, `index_scan_open`, and tuple insert/update/delete/fetch operations) plus shared MVCC-oriented value types (`TupleVersion`, `Visibility`, `NewTuple`) to keep storage API shape explicit before backend implementation.
- Added `StorageError` taxonomy and a baseline contract test to lock in visibility-scoped read behavior for no-GPU bootstrap iterations.
- Added `RuntimeMetrics::snapshot()` plus `RuntimeMetricsSnapshot` so observability/export paths can capture an immutable metrics view (including per-reason counters and latest observation fields) without reading mutable internals directly.
- Added a new `gpu_db_observability` crate with bootstrap telemetry contracts (`EngineTelemetrySnapshot`, `ReplicationLagSnapshot`, `TelemetrySink`) and an in-memory sink implementation for deterministic test capture.

## 2026-04-06

### Completed
- Extended reset-command compatibility to accept `RESET AUTHORIZATION` and shorthand `RESET AUTH` as `ResetAll` aliases (including statement-terminator regression coverage), reducing parser friction for PostgreSQL-style session cleanup probes that omit the explicit `SESSION` keyword.
- Extended session-cleanup compatibility to accept `CLOSE ALL` and `UNLISTEN`/`UNLISTEN *`/`UNLISTEN channel` as `ResetAll` no-op aliases (with invalid-form and statement-terminator regression coverage), reducing parser friction for PostgreSQL-style connection reset probes that clear cursor/listener state.
- Extended flush-command compatibility to accept underscore-separated `FLUSH WRITE_AHEAD` forms (with `{LOG|WAL}` targets and regression coverage), reducing parser friction for telemetry/control clients that emit tokenized command names with underscore separators.
- Extended flush-command compatibility to accept compact single-token forms `FLUSH WRITE_AHEAD_LOG` and `FLUSH WRITE_AHEAD_WAL` (with statement-terminator regression coverage), reducing parser friction for control clients that emit fully tokenized command targets in one identifier.

## 2026-04-05

### Completed
- Extended reset-command compatibility to accept `RESET SESSION AUTH` as a `ResetAll` alias (with statement-terminator coverage), reducing parser friction for PostgreSQL-style session cleanup probes that use the shorthand form.
- Extended flush-command compatibility to also accept target-less `FLUSH WRITE AHEAD`, `FLUSH WRITE-AHEAD`, and `FLUSH WRITEAHEAD` forms as `Flush` aliases, preserving the same admin flush semantics while reducing parser friction for clients that omit explicit `LOG`/`WAL` suffixes.
- Extended flush-command compatibility to accept both `FLUSH WRITE AHEAD LOG` and `FLUSH WRITE-AHEAD LOG` as `Flush` aliases (with statement-terminator regression coverage) so PostgreSQL-style wording variants map to the same bootstrap admin flush path without client-side rewrites.
- Extended session-reset compatibility to accept `DEALLOCATE ALL` as a `ResetAll` no-op alias (with invalid-form and terminator regression coverage) so PostgreSQL cleanup probes pass parser validation without client-side rewrites.
- Added `SET key TO value` parsing support alongside `SET key=value` so PostgreSQL-style assignment probes map to the same key/value command path without requiring pre-rewrites.
- Added regression coverage for `SET ... TO ...` acceptance (including multi-word values) and invalid `TO` forms to keep parser behavior deterministic.
- Extended text-protocol reset compatibility so `DISCARD TEMP TABLES` and `DISCARD TEMPORARY TABLES` are accepted as `ResetAll` no-op session-control commands, reducing bootstrap parser friction with PostgreSQL-style reset probes.
- Added regression coverage for the new `DISCARD ... TABLES` aliases (including statement terminators) and updated command-reference docs to keep parser contract text aligned.

## 2026-04-04

### Completed
- Added regression coverage proving `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` decodes braced streams (`{'wal';'active_txn';'apply visible gap'}`) into the canonical backlog mask, hardening parser stability for JSON-ish telemetry payload wrappers.
- Hardened `BacklogBlocker::from_label` normalization to ignore leading/trailing underscore wrappers after separator folding, so noisy labels like `__wal__` and `___active.txn___` decode to canonical blocker kinds without pre-cleaning.
- Extended backlog blocker label normalization in `BacklogBlocker::from_label` to treat dot separators (`.`) like hyphen/space (`_`), so telemetry labels like `pending.batch` decode to canonical blocker kinds without pre-normalization.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to strip backtick wrappers in addition to single/double quotes, so streams like `` `wal`,`active_txn` `` decode without preprocessing.
- Hardened `BacklogBlocker::from_label` normalization to collapse repeated separator runs into a single underscore, so labels like `commit--apply  gap` decode to canonical blocker kinds without pre-cleaning.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to accept colon (`:`) delimiters in addition to existing comma/semicolon/pipe/slash/newline/tab separators, so telemetry streams like `wal:active_txn` decode without pre-normalization.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to also split parenthesized/angle-bracket wrappers (`()`, `<>`), so streams like `<(wal|active_txn|apply visible gap)>` decode without pre-stripping envelope characters.
- Expanded regression coverage for CSV-like delimited decoding to assert colon-separated blocker labels are folded into the canonical backlog mask deterministically.
- Added colon-delimiter round-trip regression coverage (`mask -> labels -> mask`) so canonical blocker-order encoding remains stable for colon-separated exports.
- Updated replication interface docs to document colon-delimited blocker stream support.

## 2026-04-03

### Completed
- Added `WalBuffer::checkpoint_meta()` and `WalCheckpointMeta` so recovery/bootstrap paths can query durable-prefix checkpoint state (`durable_record_count`, `last_durable_txn_id`) without recomputing from raw WAL slices.
- Extended `ReplicationWatermarks` with `wal_last_durable_txn_id` sourced from WAL checkpoint metadata so replication/admission telemetry exposes the durable transaction frontier directly alongside WAL depth counters.
- Added WAL regression coverage proving checkpoint metadata only advances after successful flushes and remains pinned on flush failure paths, preserving WAL-before-visibility durability semantics.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to also split on newline/carriage-return/tab delimiters, allowing multiline telemetry streams to decode blocker labels without pre-flattening.
- Added regression coverage for multiline delimited-label decoding and updated replication interface docs to reflect newline/tab delimiter support.
- Added backlog-mask aggregate helpers (`ReplicationWatermarks::backlog_blocker_count_from_mask`, `ReplicationWatermarks::has_backlog_blockers_in_mask`) and wired `Engine::replication_watermarks` to derive aggregate blocker count/boolean through those helpers, preventing drift when mixed known/unknown blocker bits appear in automation inputs.
- Extended regression coverage for mixed-mask helper behavior to assert sanitized count/boolean semantics while unknown-only masks remain non-blocking.
- Updated replication interface docs to include the new backlog-mask aggregate helper APIs.
- Extended `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` to also split on slash (`/`) delimiters so telemetry streams like `wal/active_txn` decode without pre-normalization.
- Added backlog-blocker mask hygiene helpers (`known_backlog_blocker_mask`, `unknown_backlog_blocker_mask`, `sanitize_backlog_blocker_mask`) so automation can separate forward-compatible unknown bits from canonical blocker classes without open-coded bit arithmetic.
- Updated `ReplicationWatermarks::backlog_blockers_from_mask` to sanitize unknown bits up front and added regression coverage for mixed known/unknown masks.
- Added `ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, delimiter)` so canonical blocker label sets can be emitted directly as delimited strings for metrics/export surfaces without reimplementing ordering logic.
- Added regression coverage for delimited label encoding and encode/decode round-trip stability to keep blocker mask <-> string transforms deterministic.
- Added `ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels` so telemetry consumers can decode comma/semicolon/pipe-delimited blocker label streams directly into canonical blocker masks while ignoring unknown/empty segments safely.
- Added regression coverage for delimited-label decoding (`wal, pending-batch; ACTIVE TXN | ...`) to keep blocker-mask derivation deterministic for CSV-like automation inputs.
- Updated replication interface docs to include the new delimited-label helper in the published watermark API contract.
- Normalized backlog blocker label decoding in `BacklogBlocker::from_label`: surrounding whitespace is trimmed, labels are case-insensitive, and hyphen/space separators map to underscores so telemetry/admission automation can ingest mixed-format labels safely.
- Added regression coverage for mixed-format label decoding via `ReplicationWatermarks::backlog_blocker_mask_from_labels` (`" WAL "`, `"pending-batch"`, `"ACTIVE TXN"`, etc.) to keep round-trip blocker semantics deterministic.
- Added reverse label decode support for backlog blockers via `BacklogBlocker::from_label`, enabling typed parsing of string-form blocker classes (`wal`, `pending_batch`, etc.) emitted in telemetry.
- Added `ReplicationWatermarks::backlog_blocker_mask_from_labels` to fold blocker label streams back into the canonical bitmask, ignoring unknown labels safely while preserving deterministic blocker semantics.
- Added `ReplicationWatermarks::backlog_blocker_labels_from_mask` to project blocker masks back to stable label sequences in canonical blocker order, closing the loop for label/mask round-trip automation.
- Extended engine regression coverage for label round-trips and label-to-mask decoding (including duplicate + unknown labels), and updated replication interface docs with the new helper APIs.

## 2026-04-02

### Completed
- Added `ReplicationWatermarks::backlog_blocker_bits()` so automation can consume active blocker bit flags directly (without enum mapping) while preserving deterministic blocker order from `BacklogBlocker::ALL`.
- Extended engine regression coverage to assert blocker-bit output for single- and multi-blocker watermark snapshots.
- Added typed backlog blocker decode helpers (`BacklogBlocker::from_bit`, `ReplicationWatermarks::backlog_blockers_from_mask`) plus regression coverage for unknown-bit masking, so automation can safely decode watermark bitsets without re-implementing mapping logic.
- Added typed backlog blocker APIs via `BacklogBlocker` (`bit()`, `as_str()`, `ReplicationWatermarks::has_blocker_kind`, `ReplicationWatermarks::backlog_blockers`) plus regression coverage, so downstream automation can enumerate blocker classes without open-coded bitmask logic.
- Refactored replication backlog aggregate derivation so `backlog_blocker_count` is computed directly from `backlog_blocker_mask.count_ones()`, preventing drift between per-flag booleans and aggregate telemetry fields.
- Added `ReplicationWatermarks::has_backlog_blocker(bit)` helper plus regression assertions, so downstream automation can query blocker classes by bit without manual mask arithmetic.
- Added `ReplicationWatermarks::backlog_blocker_labels()` so automation consumers can read stable string labels (`wal`, `pending_batch`, etc.) directly from watermark snapshots instead of re-mapping enum variants externally.
- Extended engine regression coverage to assert blocker-label output for both single- and multi-blocker watermark states.

## 2026-04-01

### Completed
- Added `ReplicationWatermarks::has_backlog_blockers` as a boolean aggregate over backlog/gap blocker signals so automation can short-circuit readiness checks without recomputing from individual flags or counts.
- Added engine regression assertions proving `has_backlog_blockers` stays false in clean snapshots and flips true for single- and multi-blocker states.
- Updated replication interface docs so the aggregate blocker boolean is part of the published telemetry contract.
- Extended `ReplicationWatermarks` with `pending_batch_remaining_capacity_permyriad` (inverse of queue utilization) so operators can read normalized enqueue headroom directly without recomputing from depth/cap values.
- Added engine regression assertions covering empty, partial, and saturated queue states for the new headroom metric to keep admission telemetry deterministic.
- Updated replication and admission-control interface docs so the normalized pending-queue headroom field is part of the documented runtime contract.
- Extended `ReplicationWatermarks` with `backlog_blocker_mask` (bitset for wal/pending-batch/active-txn/commit-apply/apply-visible blockers) so automation can branch on active blocker classes without recomputing booleans.
- Added engine regression coverage proving blocker-mask values stay zero in clean snapshots and encode pending-batch and active-transaction backlog combinations deterministically.

## 2026-03-31

### Completed
- Added `ReplicationWatermarks::pending_batch_remaining_capacity` so queue telemetry now exposes free enqueue headroom directly (`cap - len`) alongside saturation/utilization fields, with regression coverage for empty, partially filled, and saturated queue states.
- Added `ReplicationWatermarks::backlog_blocker_count` so telemetry now includes an aggregate count of active backlog/gap blockers (`wal`, pending batch, active txn, commit/apply lag, apply/visibility lag) for simpler failover-readiness diagnostics without recomputing booleans downstream.
- Added engine regression coverage proving `backlog_blocker_count` increments across combined blocker states (e.g., pending-batch backlog + active transaction backlog).
- Optimized `TxnManager::active_count()` to O(1) by caching active transaction depth instead of rescanning all transaction states on every query, while preserving WAL-before-visibility and role-gating behavior through explicit regression coverage across `NotFound`, `NotActive`, and duplicate-id error paths.
- Extended `ReplicationWatermarks` with `pending_batch_utilization_permyriad` (0..10_000) so telemetry exposes queue pressure as a normalized saturation gauge in addition to raw depth/cap counters; added regression assertions for empty, partial (1/3 and 1/2), and saturated (2/2) queue states.
- Updated design traceability testing coverage to point at the concrete parity/fault-validation plan (`docs/testing/parity-and-jepsen-plan.md`) and marked the testing-strategy row as covered under phased execution.
- Expanded `docs/architecture/09-session-management-and-admission.md` with an explicit runtime admission-state contract (`active_sessions`, queue saturation, role, active txn depth) and deterministic signal-to-action mapping so session admission and failover-readiness decisions stay aligned.
- Added `docs/interfaces/error-interfaces.md` to document crate-level error taxonomy (`ParseError`, `TxnError`, `EngineError`, `ExecuteError`), side-effect expectations, and operator-response mapping so WAL-before-visibility and role/admission rejection semantics are explicit and auditable.
- Updated docs index and design traceability mappings to include the new error-interface contract and mark error-taxonomy hardening as complete.
- Extended `ReplicationWatermarks` with explicit backlog/gap blocker booleans (`has_wal_backlog`, `has_pending_batch_backlog`, `has_active_txn_backlog`, `has_commit_apply_gap`, `has_apply_visible_gap`) so automation can explain *why* readiness gates are false without recomputing conditions externally.
- Refactored failover readiness calculations (`quiescent_for_failover`, `follower_promotion_ready`) to derive from the new blocker fields, keeping gate logic centralized and auditable.
- Added/expanded engine regression assertions so baseline, follower-rejection, pending-queue backlog, and active-transaction backlog paths validate blocker flag behavior.
- Updated replication interface docs to document the new blocker fields in telemetry snapshots.

## 2026-04-25

### Completed
- Extended `EngineTelemetrySnapshot` so published observability snapshots now carry write-path readiness/backlog state (`wal_unflushed_count`, pending-batch depth/cap, active transaction depth, blocker aggregates, and failover/admission booleans) in addition to replication lag and runtime metrics.
- Added telemetry helper coverage for backlog detection, pending-capacity math, and write-path quiescence so downstream automation can reason about failover/admission state without re-fetching raw engine watermarks.
- Updated README + replication interface docs so the published telemetry contract now explicitly documents the richer readiness snapshot surface.
- Followed up by exposing the durable WAL/snapshot frontier (`snapshot_id`, flushed-record count, last durable txn id, buffered WAL depth) through `EngineTelemetrySnapshot`, with helper coverage for buffered-WAL detection.
- Hardened `EngineTelemetrySnapshot` backlog helpers so unknown blocker bits no longer masquerade as real readiness blockers, and added canonical blocker-label iteration/count helpers for downstream observability sinks.
- Added `EngineTelemetrySnapshot::backlog_blocker_delimited_labels(delimiter)` plus interface docs, so text-oriented telemetry sinks can emit canonical blocker streams without rejoining labels themselves.
- Added `EngineTelemetrySnapshot::total_backlog_items()` and `is_fully_caught_up()` so downstream sinks can distinguish mere WAL buffering from true backlog-free replication readiness without re-deriving the aggregate from raw counters.
- Added `EngineTelemetrySnapshot::known_backlog_blocker_mask()` / `sanitize_backlog_blocker_mask(mask)` so snapshot consumers can forward-compatible-sanitize blocker masks without reaching back into engine internals.
- Added `EngineTelemetrySnapshot::has_unknown_backlog_blockers()` / `unknown_backlog_blocker_count()` so sinks can distinguish forward-compatible unknown blocker bits from canonical readiness blockers without open-coded bit counting.
- Added `EngineTelemetrySnapshot::pending_batch_utilization_permyriad()` / `pending_batch_remaining_capacity_permyriad()` so snapshot consumers can read normalized enqueue pressure directly from the published telemetry contract without re-deriving percentages from raw queue counters.
- Added regression coverage proving telemetry snapshots ignore unknown backlog bits while still surfacing canonical blocker labels in deterministic order.

## 2026-03-30

### Completed
- Added regression coverage proving `follower_promotion_ready` stays false while any active transaction context remains on follower nodes, hardening the promotion gate against in-flight session state.
- Added `follower_promotion_ready` to `ReplicationWatermarks` so follower telemetry now exposes a promotion gate (no commit/apply or apply/visibility lag, no WAL unflushed backlog, no pending batch backlog, no active transactions), with regression coverage for both clean follower state and backlog-blocked state.
- Extended `ReplicationWatermarks` with commit/apply/visibility lag gauges (`commit_apply_gap`, `apply_visible_gap`) so telemetry snapshots expose index drift directly alongside role and durability counters, with regression assertions covering baseline, follower-rejection, and snapshot-install paths.
- Extended `ReplicationWatermarks` with two operational readiness flags: `mutation_admission_saturated` (pending queue at cap) and `quiescent_for_failover` (leader with zero WAL backlog, zero pending batch depth, and zero active transactions), plus regression coverage for follower state, pending-queue pressure, and active-transaction non-quiescent windows.
- Added deterministic replay parity regression coverage for GPU-eligible mutation traces, proving immediate and batched mutation paths produce identical applied-entry ordering, visible index progression, WAL flush counts, and final key/value state.
- Added `Engine::visible_state_fingerprint()` (deterministic FNV-1a over visible KV state) plus regression coverage so parity/fault harnesses can assert replay convergence via stable state digests.
- Added engine regression coverage proving `GET` rejects candidate role consistently across both immediate (`execute_text`) and queued (`enqueue_set_text`) paths, with no queue/fallback/D2H side effects when leadership gates fail.
- Extended `ReplicationWatermarks` with pending-queue timing telemetry (`pending_batch_oldest_age_ms`, `pending_batch_time_until_deadline_ms`) so replication snapshots expose queue staleness/deadline pressure in addition to depth.
- Added `pending_batch_cap` to `ReplicationWatermarks` so queue depth is reported with explicit admission capacity context for overload diagnostics.
- Added engine regression coverage proving pending-batch timing watermarks appear while queue items are buffered and clear immediately after admin flush drains the queue.
- Updated replication interface docs so `ReplicationWatermarks` telemetry fields (durability, pending depth/capacity, pending timing, txn depth) are explicit and traceable.
- Extended `ReplicationWatermarks` with `active_txn_count` so replication/durability telemetry now exposes live transaction depth alongside role, index, WAL, and pending-batch signals.
- Added engine regression coverage proving watermark snapshots report active transaction depth after `BEGIN`, and remain zero across follower rejection, buffered WAL, and pending-batch-only scenarios.
- Updated README scope notes to reflect that replication watermark reporting now includes active transaction depth.

## 2026-03-29

### Completed
- Added replication watermark coverage for queued-but-not-yet-flushed mutations: `ReplicationWatermarks` now reports `pending_batch_len` so role/commit/apply visibility telemetry includes current batch backlog depth (alongside WAL counters), with regression coverage for both empty and non-empty queue states.
- Added a new `gpu_db_planner` crate with a minimal device-aware planning surface (`Planner`, `ExecutionPlan`, `PlanNode`) so every planned command now has an explicit `DeviceTarget` annotation (`Cpu` or `Gpu(id)`) instead of relying on implicit routing assumptions.
- Added `Engine::with_planner_config` so engine instances can override the planner's default GPU id at construction time (instead of always assuming GPU 0), with regression coverage proving `plan_text` emits `DeviceTarget::Gpu(custom_id)` for mutation commands.
- Added `Engine::with_batching_and_planner_config` so custom batching thresholds and non-default planner GPU targets can be configured together in one constructor, with regression coverage proving both settings are honored simultaneously.
- Wired bootstrap planning policy to preserve GPU-first intent: mutation commands (`SET`/`DEL`/`DELETE`) are emitted as GPU-targeted plan nodes, while control/read/admin commands currently declare explicit CPU targets as the safe fallback path.
- Added planner regression tests proving write commands are GPU-targeted, read commands are explicitly CPU-targeted fallback, and the planner never emits device-agnostic nodes.
- Integrated planner scaffolding into `gpu_db_engine` via a new `Engine::plan_text` entrypoint so protocol commands can be translated into explicit device-annotated plans before execution, with engine-level regression coverage for mutation/read routing.
- Added `docs/operations/runbooks.md` with deterministic pre-deploy, WAL durability incident, role-transition, snapshot safety, and fallback-monitoring procedures to operationalize DR/security controls without weakening WAL-before-visibility invariants.
- Updated documentation index/traceability docs so operations runbooks are first-class references and prior traceability hardening tasks are explicitly recorded as complete.
- Added `docs/architecture/09-session-management-and-admission.md` with bootstrap session model, admission limits, overload rejection semantics, and forward v1 adaptive-control path; linked it into docs navigation and traceability mapping.
- Implemented pending-mutation enqueue saturation handling during retry backlogs: when the batched queue is already at cap, new mutation enqueues now fail fast with `EngineError::MutationQueueOverloaded { pending, cap }` and emit `GpuQueueSaturated` fallback telemetry instead of allowing unbounded queue growth after flush failures.
- Added engine regression coverage proving failed WAL-triggered retry backlogs keep queued items intact while rejecting additional enqueues with explicit overload errors.

## 2026-03-28

### Completed
- Hardened `RaftReplicator::append_entries_from_leader` to reject append RPCs whose `prev_log_index` is behind the local snapshot-compaction boundary, preventing invalid reintroduction of pre-snapshot log entries.
- Added regression coverage proving behind-boundary append attempts fail without mutating follower commit/apply/next-index watermarks or in-memory entry state.
- Fixed snapshot install next-index tracking in both `LocalReplicator` and `RaftReplicator` so retained uncompacted tail entries keep log indexing monotonic after snapshot ingestion.
- Added regression coverage proving snapshot installs preserve `next_index` continuity when uncommitted post-snapshot tail entries remain in memory.

## 2026-03-21

### Completed
- Hardened follower append RPC validation to reject entry batches that claim terms ahead of the sender's leader term, with regression coverage proving the invalid batch is dropped without mutating local log/commit/index state.
- Hardened follower append conflict checks to reject payload divergence when index+term already exist locally; identical index+term entries must now also carry identical payload bytes, with regression coverage proving mismatch rejection preserves local log/commit/next-index state.
- Added raft regression coverage proving follower append processing never commits past the local log tail even when leader commit is far ahead, and that missing `prev_log_index` append attempts fail without mutating follower role/term/index state.
- Added raft regression coverage proving `RaftReplicator::append_entries_from_leader` rejects leader-role callers without mutating term/role/commit/next-index state, preventing accidental use of follower append ingestion on leader paths.
- Added raft regression coverage proving candidate nodes still step down to follower and adopt a newer leader term even when append RPC validation later rejects the payload (`missing prev_log_index`), preserving Raft term/role monotonicity under rejection paths.

## 2026-03-20

### Completed
- Added `RaftReplicator::append_entries_from_leader(leader_term, prev_log_index, prev_log_term, entries, leader_commit)` term handling so follower append processing now rejects stale leaders and always updates local term/role to follower on accepted append RPCs, with regression tests for term bump + stale-term rejection.
- Added `RaftReplicator::append_entries_from_leader(prev_log_index, prev_log_term, entries, leader_commit)` to model follower-side append handling with prev-log validation, conflict truncation of uncommitted tails, and safe commit-index advancement bounded by local log availability.
- Hardened follower append ingestion to reject non-contiguous entry batches (including first-entry index skips), preserving deterministic log continuity instead of silently accepting sparse append payloads.
- Added raft regression coverage proving empty AppendEntries heartbeats (no new entries) can still advance follower commit index when a previously replicated entry becomes committed by leader progress.
- Added raft regression coverage for follower append conflict repair, prev-log term mismatch rejection, and committed-entry overwrite protection to preserve monotonic durability/visibility boundaries during catch-up flows.
- Fixed snapshot metadata term reporting in both `LocalReplicator` and `RaftReplicator` so `snapshot_meta().last_included_term` now stays tied to the last applied index (instead of drifting to the node's current term after leadership/term changes).
- Added regression coverage proving snapshot metadata preserves the applied-entry term across later term bumps for both local and raft replicators.
- Added `RaftReplicator::truncate_uncommitted_from(index_inclusive)` to model follower catch-up conflict repair by dropping only uncommitted tail entries at/after a conflicting index, pruning matching ack-tracking state, and resetting `next_index` to the surviving log tail.
- Added regression coverage proving truncation drops uncommitted tails safely, preserves committed boundaries, and allows replacement proposals to reuse the truncated index without violating monotonic commit progression.

## 2026-03-19

### Completed
- Added engine regression coverage for transaction-control aliases `END`/`ABORT` with `AND CHAIN` in both immediate and enqueue paths, proving alias parity reopens transaction context identically to `COMMIT`/`ROLLBACK`.
- Added read-transfer telemetry parity for immediate text execution: `Engine::execute_text` now records D2H bytes for successful `GET` hits (matching `execute_read_text` semantics) while leaving misses unchanged, with regression coverage proving hit-only accounting.
- Expanded `START WORK` transaction-mode regression coverage to include isolation/deferrable mode lists and duplicate-isolation rejection, guarding parser parity for PostgreSQL-style aliases.
- Tightened transaction-begin mode parsing so conflicting or duplicate mode classes are rejected (e.g. `READ ONLY` + `READ WRITE`, repeated isolation clauses, mixed `DEFERRABLE`/`NOT DEFERRABLE`), with regression coverage proving unsupported combinations fail fast instead of being silently accepted.
- Wired placeholder kernel-occupancy telemetry into batched mutation flushes: `Engine::apply_batch` now records simulated occupancy per payload (capped at 100% permyriad), with regression coverage proving occupancy sample/total/latest metrics advance alongside existing H2D + kernel-exec counters.
- Added saturation-path occupancy coverage so large batched payloads pin simulated occupancy at exactly 10_000 permyriad (100%), preventing telemetry overflow and making the placeholder signal bounded until real CUDA counters land.
- Hardened role-aware read command handling across mixed execution entry points: `execute_text` and `enqueue_set_text` now reject `GET` when role is follower/candidate (matching `execute_read_text` leadership gates), with regression coverage confirming no fallback/queue side effects on rejected reads.
- Extended text-protocol transaction compatibility so `BEGIN READ ONLY` and `BEGIN READ WRITE` are accepted directly (without requiring `WORK`/`TRANSACTION`), with negative coverage for unsupported partial/isolation-style suffixes.
- Added raft regression coverage proving `install_snapshot` prunes ack-tracking and compacted entries at/under the installed snapshot boundary while preserving newer pending entries.
- Added role-aware read gating in `Engine::execute_read_text`: `GET` now returns `EngineError::NotLeader` when node role is follower/candidate, with regression coverage ensuring read fallback/transfer metrics are not emitted on rejected reads.
- Tightened engine read-path contracts: `execute_read_text` now returns an explicit `NonReadCommand` error for non-`GET` commands instead of silently returning `None`, with regression coverage to prevent accidental mutation/control usage through read-only entry points.
- Clarified text-protocol delete diagnostics so malformed `DEL`/`DELETE` commands now report `expected: DEL|DELETE key`, matching accepted aliases.
- Extended SQL transaction-control compatibility by accepting `START WORK` as a `BEGIN` alias in the text protocol parser.
- Hardened protocol regression coverage for `START WORK` with optional statement terminator handling and explicit rejection of unsupported extra-token forms.
- Exposed snapshot progression in engine replication watermarks by adding `snapshot_id` to `ReplicationWatermarks`, with regression assertions for pre-snapshot, post-commit, and installed-snapshot paths to improve observability around compaction/snapshot boundaries.
- Added read-path transfer telemetry for `GET`: `Engine::execute_read_text` now records `d2h_bytes_total` from returned values, with regression coverage for hit/miss cases to keep simulated GPU transfer accounting stable before CUDA integration.
- Optimized `Engine::apply_batch` flush draining to avoid `Vec::remove(0)` quadratic behavior by streaming items via iterator + replay-safe tail requeue on commit failures.
- Hardened `RaftReplicator` leadership transitions by dropping uncommitted log tail + clearing in-flight follower ack maps on `become_follower`/`become_leader`, preventing stale quorum evidence from leaking across term/role changes.
- Switched Raft ack tracking to per-index voter-id sets so duplicate follower acks cannot satisfy quorum counts incorrectly.
- Added regression coverage proving post-transition leadership starts from the last committed index and re-proposes new work in a clean epoch, and that duplicate ack events from one follower are ignored.
- Extended text protocol command coverage with `GET key` parsing/validation (`InvalidGet` on malformed forms) plus regression tests.
- Added `Engine::execute_read_text` for deterministic read command execution without mutating WAL/visibility state.
- Added engine regression coverage proving `GET` returns current values and that non-mutation command handling (`BEGIN`/`COMMIT`/`ROLLBACK`/`GET`) increments explicit `NotGpuEligible` fallback metrics consistently for immediate and batched paths.
- Extended `Engine::execute_read_text` to emit `NotGpuEligible` fallback telemetry for read commands too, keeping observability parity between read-only execution and immediate/batched non-mutation paths.
- Hardened batched mutation leadership gates so follower mode rejects mutation enqueue/flush before draining batch buffers, preserving queued work and preventing silent drop-on-flush during role changes.
- Refined batching tick semantics so follower background ticks are a no-op when the queue is empty, while still surfacing `NotLeader` if queued mutations would have flushed.
- Added Raft ack-map pruning after commit advancement so committed indices are dropped from in-memory quorum tracking, plus regression tests proving committed entries are evicted while pending entries remain.
- Added engine-level snapshot hooks (`export_snapshot_meta`, `install_snapshot`, `snapshot_meta`) and regression coverage proving snapshot install advances commit/apply/visibility watermarks without breaking WAL-before-visibility behavior for subsequent commits.
- Added engine candidate-role transition surface plus regression tests confirming candidate mode rejects direct and batched mutations with no WAL/visibility/queue side effects.

## 2026-03-18

### Completed
- Added engine tests proving transaction control commands (`BEGIN`/`COMMIT`/`ROLLBACK`) are counted as explicit `NotGpuEligible` fallback events in both immediate and batched command paths.
- Added deterministic execution routing primitives (`DeviceRouter`, `GpuRuntime`, `MockGpuRuntime`) with explicit GPU fallback reasons (`Unavailable`, `QueueSaturated`, `MemoryPressure`) and route-decision tests.
- Extended runtime metrics with `last_fallback_reason` and `last_batch_flush_reason` observability fields plus regression tests, and validated those latest-reason signals from the engine integration tests.
- Added engine regression coverage proving failed batch flush paths (count-triggered and admin-triggered) do not advance flush counters or latest-flush-reason telemetry.
- Added phase-gated compatibility matrix (`docs/compatibility/matrix.md`) to make protocol/SQL, durability, replication, and CPU/GPU support boundaries explicit by v0/v0.5/v1.
- Added deterministic parity + Jepsen-style fault-validation plan (`docs/testing/parity-and-jepsen-plan.md`) with concrete streams, exit criteria, and artifact requirements.
- Updated docs navigation (`docs/README.md`) to include the new compatibility and validation-gate docs.
- Added runtime batch-wait telemetry (`batch_wait_samples`, `batch_wait_total_ms`, `last_batch_wait_ms`) and wired engine flush paths to record per-item queue wait at count/time/admin flush boundaries.
- Added `RaftReplicator` state skeleton in `gpu_db_replication` with quorum-aware ack tracking, in-order commit advancement, snapshot hooks, and regression tests to keep v0.5 replication interfaces executable while preserving WAL-before-visibility boundaries in engine paths.

## 2026-03-15

### Completed
- Workspace scaffold with core crates (`types`, `replication`, `wal`, `txn`, `execution`, `engine`).
- Replication-shaped local commit path with WAL-before-visibility invariant tests.
- Deterministic dual-trigger batcher crate (`batching`) with count/time flush tests.
- Runtime metrics scaffold (`metrics`) including fallback-reason counters.
- Minimal text command parsing crate (`protocol`) with command tests.
- Engine integration for `SET key=value` command path and commit accounting.
- CI workflow for fmt/clippy/test and local `Justfile` tasks.

### Current blockers
- System packages not installed yet: `clang`, `protoc`, `bison`, `flex`, `m4`, `zlib1g-dev`.
- These block parser-native and protobuf/native toolchain work, but not core Rust implementation loops.

### Next loops
1. Add an in-engine queue using `DualTriggerBatcher` and batch flush telemetry.
2. Add `LocalReplicator` role transition simulation tests (leader/follower reject path).
3. Introduce durability error-path tests for commit pipeline.
