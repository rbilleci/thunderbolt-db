# Changelog

All notable release changes are recorded here. The project uses semantic version identifiers while its public
surface is experimental.

## 0.1.0-alpha.1 — 2026-09-14

First experimental GPL-3.0-only source release candidate, with the CUDA Driver Additional Permission in
`CUDA_EXCEPTION` for the separately installed CUDA Driver API.

This release license applies to this source release and later project distributions that say so. It does not
withdraw any valid license grant made for an earlier copy under the repository's former MIT metadata.

- Provides the single `gpu-db-engine-server` PostgreSQL wire endpoint.
- Executes the documented bounded relational surface on NVIDIA Blackwell-class GPUs.
- Supports an explicit crash-durable local WAL configuration and recovery across process restart.
- Includes checked-in PTX with preferred-form CUDA sources and regeneration commands.
- Adds source licensing, dependency policy/inventory, contributor provenance, security reporting, and reproducible
  host/GPU release gates.
- Excludes the historical literature-review corpus from the versioned source archive pending a separate provenance
  review; it is not required to build, test, modify, or run the engine.

This version is single-node evaluation software. The open HA, scale, compatibility, checkpoint/PITR, and
comparative-performance work remains in `docs/PLAN.md` and is not claimed by this release.
