# ADR-014 acceptance archive

This directory preserves the non-actionable process record that led to ADR-014 acceptance on 2026-07-16. The
accepted decision is [`docs/DECISIONS.md`](../../../DECISIONS.md), stable system contracts are in
[`docs/ARCHITECTURE.md`](../../../ARCHITECTURE.md), the detailed design is
[`docs/design/write-path-adr-014.md`](../../../design/write-path-adr-014.md), and all open implementation work lives
only in [`docs/PLAN.md`](../../../PLAN.md).

The exact final reviewed snapshot is commit `c96287668864b1f776d0c982aaf4a22981d4a09e`. Use that commit—not the
archive copies with their added non-actionable banners—to verify packet v8's 24 hashes. Historical links and
commands inside these files intentionally retain their original pre-archive paths and decision-time wording.

## Preserved process material

- `write-path-design-inputs.md`: pre-decision constraints and alternatives.
- `write-path-adr-{performance,durability,acid,consistency}-audit.md`: independent REVISE reviews whose findings
  were incorporated before acceptance.
- `write-path-adr-evidence.md`: pinned source crosswalk at baseline `f701d8b6`.
- `write-path-adr-{slo-footprint,physical-selection,controller-injections}.md`: one-off selection evidence.
- `write-path-adr-review-packet*.md` and `write-path-adr-final-independent-review*.md`: frozen packet/verdict history
  from the first rejection through v8 acceptance.
- `artifacts/*.rs.txt`: disposable build-only controller and physical-candidate models, removed from Cargo example
  discovery after acceptance.

Nothing in this directory owns current status, sequencing, blockers, or acceptance gates.
