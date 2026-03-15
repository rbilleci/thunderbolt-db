## Summary
- What changed?
- Why now?

## GPU-First Checklist (required)
- [ ] Device strategy declared (CPU + GPU path, or explicit GPU TODO + fallback rule)
- [ ] GPU execution impact described
- [ ] Cost model impact described (if planner/runtime behavior changed)
- [ ] Fallback behavior described (when/why CPU fallback occurs)
- [ ] GPU memory/layout impact described (SoA/AoS, alignment, null bitmap, transfer volume)
- [ ] Telemetry impact described (H2D/D2H, kernel time, occupancy, batch wait, fallback rate)
- [ ] Parity tests added/updated (CPU reference vs GPU path) for affected GPU-eligible behavior
- [ ] If introducing/expanding fallback paths: linked parity issue with owner + milestone

## Correctness and Safety
- [ ] WAL-before-visibility invariant preserved
- [ ] Transaction/isolation semantics unaffected or explicitly documented
- [ ] Failure modes considered (GPU error, timeout, memory pressure)

## Validation
- [ ] Unit tests
- [ ] Integration tests
- [ ] Benchmark or perf note (if execution path changed)
