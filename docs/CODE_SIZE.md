# Source File Size and Decomposition Standard

Large files are a change-locality and review problem before they are a style problem. They make it harder for a
person or model to load the relevant invariants, identify ownership boundaries, and validate a focused edit.
Line count is therefore an analysis trigger, not a license to split cohesive code into arbitrary shards.

`PLAN.md` owns all remediation work. This document owns the standing standard and the exception registry.

## Size envelopes

Count physical lines in tracked, human-maintained source files. Rust, PTX, CUDA/C/C++, Python, shell, JavaScript,
and TypeScript are in scope. Markdown, lockfiles, vendored dependencies, build output, and machine-generated source
are not. A generated file must be reproducible and clearly identified before it is excluded.

| File class | Preferred envelope | Required analysis | Critical outlier |
|---|---:|---:|---:|
| Production source | 300–1,500 lines | Over 2,000 lines | Over 5,000 lines |
| Tests, examples, benchmarks, and tools | 300–2,000 lines | Over 3,000 lines | Over 5,000 lines |
| Generated or archived source | No numeric target | Confirm provenance and non-ownership | Manual edits or mixed generated/handwritten content |

Files under 500 lines are usually easy to load in one context. Files between 500 and the preferred ceiling are
not presumptively too large when they own one coherent concept. Crossing a required-analysis threshold means the
file is outside the guideline until it is decomposed or entered in the exception registry. Every critical outlier
must have a `PLAN.md` task until that disposition is complete.

New files should remain within the preferred envelope. Do not create a production file over 2,000 lines or a
test/tool file over 3,000 lines without an accepted exception. A change that pushes an existing file across a
threshold, or grows an already-outlying file materially, must include decomposition or update its PLAN-owned
disposition.

## Analyze before splitting

For every file outside the envelope:

1. **Confirm the count and class.** Separate production, test, example/benchmark, tool, generated, vendored, and
   archived code. Do not exempt a handwritten generator merely because its output is generated.
2. **Map responsibilities.** Inventory major types, functions, traits, tests, embedded kernels, and initialization
   sections. State the invariant each cluster owns and identify sections that change together.
3. **Map coupling.** Record callers, imports, re-exports, feature gates, shared state, unsafe boundaries, generated
   symbols, and cross-crate API consumers. Use history to distinguish stable cohesion from accidental growth.
4. **Choose a disposition.** Decompose along ownership boundaries; retain a cohesive file through a documented
   exception; identify reproducible generated material; or archive/delete code that is no longer live.
5. **Design the destination first.** Name the proposed modules and their responsibilities, public surface, test
   placement, dependency direction, and validation gates. Target modules normally remain below 1,500 lines and
   must not simply move the same ambiguity into another oversized file.

Good boundaries include protocol phases, storage layers, operator families, data types, transaction phases,
kernel families, and independent test behaviors. Bad boundaries include `part1`/`part2`, line-number slices,
mutually dependent modules, catch-all `common` modules, and splits that require broadening most items to
`pub(crate)`.

## Execute a decomposition

Keep structural and behavioral changes separate:

1. Extract one coherent leaf or ownership domain at a time. Preserve behavior, feature gates, symbol names, and
   the external API; use deliberate re-exports where callers should not change yet.
2. Move the closest tests with the behavior they protect. Keep integration tests at the public seam and preserve
   test names so CI history remains legible.
3. For embedded CUDA/PTX, split by operator or data-type ownership while preserving compilation, cache, symbol,
   launch, stream, and error-handling boundaries. A text-only kernel move still needs a real GPU smoke gate.
4. Update `mod` declarations, `use` paths, re-exports, feature lists, build scripts, examples, tests, doc links,
   and file maps in the same slice. Search for the old path and moved symbol after the edit.
5. Run the narrowest compile and test gates that cover the module, followed by the affected crate's broader
   gate. Run `cargo fmt --check` for touched Rust and `git diff --check`. Runtime, kernel, residency, or result-path
   behavior changes also require the performance and GPU gates in `AGENTS.md`; a verified pure move does not
   require an unrelated benchmark campaign.
6. Delete the old implementation only after references are clean and gates pass. Record completed structural
   outcomes in `STATUS.md`; remove the PLAN row when every outlier it owns has a disposition.

A decomposition is complete when each resulting module has one explainable responsibility, dependencies point
in a clear direction, no duplicate implementation remains, public API growth is justified, references and docs
resolve, and relevant gates pass.

## Exceptions

An exception is appropriate only when separation would obscure a stronger invariant—for example a generated
artifact, a stable declarative table, or a tightly coupled kernel corpus whose compilation and symbol ownership
cannot be separated safely. “It is old,” “splitting is inconvenient,” and “the tests pass” are not sufficient.

Add accepted exceptions to the table below. An exception records a current architecture fact, not future work;
any proposed remediation belongs in `PLAN.md`. Re-review an exception when the file grows by 20%, gains a new
responsibility, changes its public boundary, or reaches the stated trigger.

| File | Class and current size | Cohesion rationale | Boundary that must remain intact | Re-review trigger |
|---|---:|---|---|---|
| `crates/engine/src/engine_expr.rs` | Production, 2,421 lines | After all independently stable contracts and execution phases moved to bounded leaves, the root contains exactly one cohesive resident SELECT/grouped GPU orchestration function | Keep group-key planning, derived device-buffer ownership, pass construction/execution, cross-pass alignment, grouped result/HAVING/final framing together; do not introduce a context bag or detach lifetime guards solely for line count | Growth by 20% (~2,905 lines), a second production responsibility or method family, a stable grouped-pass contract, public-boundary change, or 5,000 lines |

## Inventory command

Run from the repository root:

```bash
git ls-files -z -- '*.rs' '*.ptx' '*.cu' '*.cuh' '*.c' '*.h' '*.hpp' '*.cc' '*.cpp' \
  '*.py' '*.sh' '*.js' '*.ts' \
  | xargs -0 wc -l \
  | sort -nr
```

Review classifications manually. The command deliberately reports archived and generated candidates so they
must prove their provenance rather than disappearing silently from the audit.
