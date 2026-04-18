# Implementation Log (Pre-NVIDIA Phase)

## 2026-04-18

### Completed
- Re-validated the current mainline with the full autonomous safety gate (`cargo fmt --all --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`) after lock-guarded cron bootstrap, confirming the GPU-first + WAL-before-visibility invariants remain green with no pending code deltas in this loop.
- Hardened PostgreSQL frontend frame boundary validation to reject invalid message length fields below the protocol minimum (`len < 4`) with a dedicated `InvalidLengthField` error before payload decode.
- Added malformed-frame regression coverage for underflowed frontend length declarations (`Q` frame length `3`) so extended-query/simple-query boundary rejection stays deterministic.
- Added malformed extended-query boundary regressions covering empty `Describe`/`Close` payloads and negative cardinality fields in `Bind`/`FunctionCall` frames, proving deterministic rejection for truncated lifecycle/control frames before decode side effects.
- Added malformed extended-query boundary regressions for negative cardinality fields in `Parse` parameter-type counts, `Bind` parameter-format counts, and `FunctionCall` argument-format counts, keeping deterministic rejection coverage explicit for signed underflow payloads across the full parse/bind/call lifecycle.
- Added malformed extended-query boundary regressions proving `Parse`/`Bind`/`Execute` reject trailing payload bytes after structurally complete fields, keeping frame-boundary validation deterministic for lifecycle/control messages.
- Added malformed `FunctionCall` frame boundary regression proving trailing bytes after a structurally complete payload are rejected deterministically (`InvalidFunctionCallPayload`) before decode side effects.
- Added malformed `FunctionCall` regression coverage proving negative result-format codes are rejected deterministically (`InvalidFunctionCallPayload`).

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
