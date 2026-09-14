## Summary
- What changed?
- Why now?

## GPU-First Checklist (required)
- [ ] Production GPU route and host control-plane boundary declared
- [ ] GPU execution impact described
- [ ] Cost model impact described (if planner/runtime behavior changed)
- [ ] GPU decline/fault behavior remains fail-loud with no CPU relational fallback
- [ ] GPU memory/layout impact described (SoA/AoS, alignment, null bitmap, transfer volume)
- [ ] Telemetry impact described (H2D/D2H, kernel time, occupancy, batch wait, fallback rate)
- [ ] GPU-native oracle/parity tests added or updated for affected behavior
- [ ] Superseded product authority and selector removed with the cutover

## Correctness and Safety
- [ ] WAL-before-visibility invariant preserved
- [ ] Transaction/isolation semantics unaffected or explicitly documented
- [ ] Failure modes considered (GPU error, timeout, memory pressure)

## Validation
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance applicability declared: full card, carried comparable card with rationale, or not applicable
- [ ] `--quick` used only as a development screen, never cited as acceptance evidence
- [ ] Applicable full-card artifact and independent final audit bind to the exact frozen candidate

## Provenance
- [ ] Every commit has a DCO `Signed-off-by` trailer (`git commit -s`)
- [ ] New third-party code/data identifies its source, copyright, license, and required notices
