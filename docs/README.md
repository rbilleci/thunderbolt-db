# Documentation governance

Each kind of information has one owner:

- [`CHARTER.md`](CHARTER.md) — mandate and invariants: the **musts**.
- [`PLAN.md`](PLAN.md) — the only open/deferred/blocked work ledger: the **next**.
- [`STATUS.md`](STATUS.md) — current built and verified facts: the **now**.
- [`HANDOVER.md`](HANDOVER.md) — short resume baton pointing to PLAN IDs: the **resume point**.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design: the **how**.
- [`DECISIONS.md`](DECISIONS.md) — accepted ADRs and rationale: the **why**.
- [`CODE_SIZE.md`](CODE_SIZE.md) — source-size, decomposition, reference-update, and exception rules.
- [`CONFIG.md`](CONFIG.md) and [`SHARD_STORAGE.md`](SHARD_STORAGE.md) — focused current contracts.

`design/` contains non-authoritative mechanism references. Slice orders, open questions, and recommendations in
those documents are design history; only PLAN may activate them.

`archive/` contains concluded plans, handovers, proposals, reviews, benchmark evidence, research, and historical
runbooks. Its local `AGENTS.md` makes all `TODO`, `NEXT`, `OPEN`, and deferred language non-actionable.

## Single-plan rule

Outside `PLAN.md`, actionable language must be one of:

1. A current fact that references an existing PLAN ID.
2. A normative invariant or acceptance gate.
3. An explicitly historical quotation under `docs/archive/`.

Do not create new roadmap, remaining-work, proposal-sequence, open-board, or campaign-handover files. Add one task
row to PLAN and link supporting evidence. When work completes, remove its row and record the outcome in STATUS or
the archive.

## Documentation audit

After documentation changes:

```bash
# Action markers outside PLAN should occur only in governance text, stable design labels,
# or archived history.
rg -n -i 'TODO|NEXT|OPEN|DEFERRED|REMAINING WORK|OPEN BOARD' \
  . --glob '*.md' --glob '!docs/PLAN.md' --glob '!docs/archive/**' --glob '!target/**'

# Old working-set directories and canonical duplicates must stay absent.
test ! -e docs/HANDOVER_REMAINING_WORK.md
test ! -e docs/WRITE_CONVEYOR.md
test ! -d docs/proposals
test ! -d docs/reviews
test ! -d docs/optimizations
test ! -d docs/future

git diff --check
```

Review every reported non-archive marker; the goal is zero independent task ownership, not blindly zero words.
