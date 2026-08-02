# HANDOVER — Resume Baton

This is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, blocked, and sequenced work;
[`STATUS.md`](STATUS.md) owns accepted facts.

## Current boundary

**WRITE-001** is the sole active NOW milestone and is incomplete. The tree has substantial accepted typed-batch,
GPU-preflight, codec/replay-witness, narrow nullable-Int4 recovery, bootstrap, and private publication foundations.
The latest accepted internal fact is placement-only resource republication, recorded in `STATUS.md`, but these
foundations do not yet form the one general production INSERT/WAL/GPU apply/publication/reopen route.

The working tree is an integrated WIP candidate spanning engine generation/bootstrap, execution completion/rebuild,
WAL terminal encoding, and documentation. Preserve it; do not discard or relabel it as a completed checkpoint.

## Resume here

Resume at [`PLAN.md`'s WRITE-001 12-hour end-to-end recovery](PLAN.md#current-focus--write-001-12-hour-end-to-end-recovery).
Start the clock only when implementation begins. Reconcile the current tree, then make the representative production
pgwire-to-fresh-reopen route pass by the 2:30 gate. Do not design, audit, accept, or hand over another private
sub-boundary. Intermediate helpers remain WIP inside the one WRITE-001 candidate.

The concluded causal analysis is
[`archive/reviews/write-001-delivery-rca-2026-08-02.md`](archive/reviews/write-001-delivery-rca-2026-08-02.md).
The unchanged product contract is
[`design/write-001-general-insert-pipeline.md`](design/write-001-general-insert-pipeline.md).

**CARD-001** remains blocked on accepted WRITE-001, and **COPY-001** remains blocked on both.
