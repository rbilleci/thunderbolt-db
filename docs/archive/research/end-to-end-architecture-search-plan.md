# GPU DB End-to-End Architecture Search Plan

## Purpose

Use the completed research dataset to design the best currently defensible
end-to-end GPU DB architecture. The goal is not to adopt isolated papers one at
a time. The goal is to compose compatible mechanisms into coherent candidate
architectures, score them against product objectives, and document the final
candidate set with evidence, assumptions, proof gates, and references.

This plan treats architecture design as constrained multi-objective search
under uncertainty. Correctness constraints are hard filters; performance,
latency, scale, operability, implementation cost, and risk are optimization
axes. Benchmarks falsify or refine candidates rather than randomly discovering
features.

## Source Inputs

- Mechanism catalog:
  `docs/research/architecture-compatibility/mechanisms.json`
- Compatibility edges:
  `docs/research/architecture-compatibility/compatibility-edges.json`
- Paper traceability:
  `docs/research/architecture-compatibility/paper-mechanism-links.json`
- Coverage and evidence report:
  `docs/research/architecture-compatibility/paper-mechanism-coverage.md`
- Benchmark/proof backlog:
  `docs/research/architecture-compatibility/benchmark-backlog.md`
- Architecture compatibility view:
  `docs/research/architecture-compatibility.md`
- Current architecture docs:
  `docs/architecture/01-system-invariants.md`,
  `docs/architecture/04-execution-model-cpu-gpu.md`,
  `docs/architecture/09-session-management-and-admission.md`,
  `docs/architecture/10-p8-gpu-optimized-storage-engine.md`,
  `docs/architecture/11-high-throughput-query-runtime.md`, and
  `docs/architecture/12-acid-isolation-and-gpu-memory.md`
- Current benchmark status:
  `docs/testing/benchmarks/README.md`

## End Deliverables

1. A design-space model that converts the research dataset into architecture
   variables, hard constraints, compatibility rules, risk penalties, and
   workload/objective weights.
2. A ranked candidate set of end-to-end architecture families, not isolated
   mechanism notes.
3. A detailed final candidate dossier with references back to mechanism cards,
   compatibility edges, paper evidence, benchmark gates, current architecture
   docs, and existing benchmark reports.
4. A prioritized proof plan naming which assumptions must be tested first and
   what result would change the chosen architecture.

Expected final files:

- `docs/research/end-to-end-architecture-design-space.md`
- `docs/research/end-to-end-architecture-candidates.md`
- `docs/research/end-to-end-architecture-final-dossier.md`

## Five-Point Execution Plan

### P1 - Define Objective Model And Hard Constraints

Goal: make the optimization problem explicit before scoring candidates.

Work:

- Define hard constraints that no candidate may violate:
  WAL-before-visibility, SQL-visible correctness, isolation traceability,
  crash/recovery proof, explicit fallback semantics, bounded admission, bounded
  memory/session cost, and operator-observable failure modes.
- Define objective axes:
  write throughput, retained-read p50/p99, mixed workload stability, 1M logical
  session viability, recovery time, HBM/DRAM/NVMe efficiency, implementation
  complexity, operational clarity, and benchmarkability.
- Define target workload classes:
  COPY/INSERT-heavy ingest, hot point lookups, retained aggregates, mixed HTAP
  freshness reads, over-resident partitioned reads, high-concurrency idle plus
  bursty active sessions, and crash/recovery/replay paths.
- Define scoring semantics:
  hard constraint pass/fail, weighted objective scores, confidence level,
  implementation risk, and evidence strength.

Done when:

- `docs/research/end-to-end-architecture-design-space.md` has a clear objective
  model and hard-constraint checklist.
- Every hard constraint maps to at least one source mechanism, architecture
  doc, or benchmark/proof gate.

### P2 - Build The Research-Derived Design Space

Goal: convert the research dataset into composable architecture variables.

Work:

- Group mechanisms into decision dimensions:
  durability/visibility, MVCC/snapshot frontiers, metadata publication,
  storage layout, memory reclamation, runtime admission, scheduling, execution,
  routing, query optimization, validation, and placement.
- Convert compatibility edges into design-space rules:
  requires, strengthens, compatible, tension, and alternative-to.
- Separate mechanism states:
  baseline invariant, candidate default, optional extension, benchmark-only
  experiment, deferred feature, and rejected/incompatible option.
- Capture unresolved uncertainty as named assumptions rather than prose.

Done when:

- The design-space file contains a mechanism-to-dimension matrix.
- Candidate construction rules are explicit enough for a worker to build
  architecture families without rereading the entire literature journal.

### P3 - Generate End-To-End Candidate Families

Goal: compose whole-system architectures, not single-feature proposals.

Work:

- Produce 3 to 5 coherent candidate families. At minimum consider:
  - conservative retained-read evolution from the current P8 engine,
  - partition-owner plus retained partition snapshot architecture,
  - aggressive GPU-resident hot-path architecture,
  - multi-tier placement and freshness-router architecture,
  - runtime-first 1M-session architecture.
- For each family, document:
  mechanism set, owner topology, data movement model, visibility model,
  route/fallback model, scheduling/admission model, storage/tier model,
  recovery model, validation model, expected strengths, risks, and non-claims.
- Reject combinations that violate hard constraints or compatibility rules.

Done when:

- `docs/research/end-to-end-architecture-candidates.md` contains the first
  complete candidate set.
- Each candidate references mechanism ids and names the assumptions it depends
  on.

### P4 - Score, Compare, And Select The Preferred Architecture

Goal: choose a preferred architecture family and explain why.

Work:

- Score each candidate against the objective model.
- Identify Pareto-frontier candidates rather than pretending every axis has one
  scalar optimum.
- Name dominating risks and decisive proof gates.
- Compare candidates against current engine facts from P8 and runtime docs.
- Pick:
  - preferred architecture,
  - fallback architecture if key assumptions fail,
  - deferred architecture ideas that should not shape current implementation.

Done when:

- The candidate document includes a comparison section with scores, confidence,
  and rationale.
- The preferred candidate is named with a concise thesis and explicit
  invalidation conditions.

### P5 - Produce Final Dossier And Worker-Ready Proof Plan

Goal: document the selected architecture in enough detail for future execution.

Work:

- Write `docs/research/end-to-end-architecture-final-dossier.md`.
- Include:
  - executive architecture thesis,
  - component-by-component design,
  - end-to-end request/write/read/recovery flows,
  - mechanism references,
  - paper evidence references through traceability ids,
  - compatibility-edge rationale,
  - current implementation fit/gaps,
  - benchmark/proof gates in priority order,
  - worker-ready next packets with target docs/code/tests where possible,
  - non-claims and assumptions.
- Update `docs/README.md` so the dossier is discoverable.

Done when:

- The final dossier can be read without opening the raw literature journal.
- Every major architectural claim has a reference path back to the research
  dataset or an existing architecture/benchmark artifact.
- The proof plan says what would change the architecture if a gate fails.

## Operating Rules For The Loop

- Execute one bounded slice per run.
- Do not resume broad paper ingestion.
- Do not replace the research dataset; consume it.
- Prefer generated or structured summaries when possible, but hand-written
  synthesis is allowed when it cites stable source files and ids.
- Do not claim an architecture is optimal. Claim it is the best currently
  defensible candidate under named constraints and assumptions.
- Keep candidate architectures compatible with existing correctness guardrails.
- Commit and push focused repo changes after validation.
- Report to Discord only on meaningful milestones, blockers, or completion.

## Validation

Each slice must run at least:

```sh
python3 scripts/generate_research_architecture_compatibility.py
python3 scripts/generate_research_paper_mechanism_links.py
git diff --check
```

If a slice changes JSON or generated files, also run:

```sh
python3 -m json.tool docs/research/architecture-compatibility/mechanisms.json >/dev/null
python3 -m json.tool docs/research/architecture-compatibility/compatibility-edges.json >/dev/null
python3 -m json.tool docs/research/architecture-compatibility/paper-mechanism-links.json >/dev/null
```

Before each commit, run:

```sh
git diff --cached --check
```
