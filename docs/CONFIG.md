# CONFIG — the single source of truth for what this engine runs

> **Policy (user ruling, 2026-07-07 — BINDING, escalated the same day): NO FEATURE FLAGS.**
> The development unit is the BRANCH, not the flag: develop and A/B on a branch; merge ONLY when
> the path is correct AND complete, and the merge REPLACES the old arm (deletion in the same
> merge). Unfinished or unproven work stays on its branch, never on main.
>
> Two things may exist on main:
> 1. **Product configuration** — few, documented HERE, permanent. Target ≈ a handful.
> 2. **Bench-only knobs** — confined to `examples/`; never read by engine code.
>
> When an adaptive mechanism wins, the manual knobs it replaced are deleted in the same merge.

## 1. Product configuration (the complete list)

| Setting | Default | Meaning |
|---|---|---|
| `GPU_DB_WAL_DURABILITY` | `fua` | Durability backend: `fua` (FUA fence-pool write-through, the default engine) or `serial` (fdatasync group commit — compat/opt-out). Reopen is disk-authoritative regardless. |
| `GPU_DB_SYNCHRONOUS_COMMIT` | `on` | PostgreSQL-model commit mode default. `off` = async commit (ack at applied cut; bounded power-failure loss, never consistency). Per-statement override via `submit_covered_insert_intent_with_commit`. |
| `GPU_DB_INTENT_LANES` | `10` | Intent-lane count (deployment sizing). `0`/`1` = lanes off (classic write path only). Disk-authoritative on reopen. |
| `GPU_DB_INTENT_LANE_SEGMENT_BYTES` | 64 MiB | Per-lane WAL segment size (deployment sizing). M-TPS boxes: 512 MiB+ (roll cadence; see HANDOVER). Never raise the *default* — tests prewrite segments ×2/lane. |
| `GPU_DB_OPEN_SHARD_FLOOR_ROWS` | 262144 | Open-shard capacity floor (deployment sizing; prevents geometric re-admission storms). |
| Server/wire settings | — | `GPU_DB_AUTH_*`, `GPU_DB_TLS_*`, `GPU_DB_REPLICATION_*`, `GPU_DB_SECURITY_PROFILE`, `GPU_DB_WAL_SEGMENT`/`GPU_DB_WAL_CHECKPOINT_BYTES`/`GPU_DB_WAL_PREALLOC_BYTES` (serial-WAL sizing), `GPU_DB_WAL_FUA_LANES`/`GPU_DB_WAL_FUA_SEGMENT_BYTES` (non-lane FUA sizing). Owned by the server/replication campaigns; unchanged here. |

## 2. Feature flags

**None, permanently — flags are banned.** Alternate paths live on branches until they replace
the incumbent.

## 3. Folded/deleted in the 2026-07-07 flag reckoning

| Former flag | Verdict | Where it went |
|---|---|---|
| `GPU_DB_MEGA_FUSE` | A/B loser (both load ends) | Arm REVERTED to `feature/mega-fuse`; record in HANDOVER. Returns as a replacement with the cross-lane coalescer + WAL-first reorder. |
| `GPU_DB_FUSED_APPLY` | A/B winner (+14–17%) | Flag DELETED; fused apply is always-on **eligibility dispatch** (the unfused sequence remains only as the ineligible-shape fallback: i64 sections / no live index / in-pass decline). |
| `GPU_DB_INTENT_LANE_MIN_WAVE` | 512 won every sweep | Hardcoded (adaptive formation cap). |
| `GPU_DB_INTENT_LANE_GROUP_US` | 2000 won | Hardcoded (population-scaled deadline cap). |
| `GPU_DB_INTENT_LANE_WAVE_MAX` | never a lever | Hardcoded 1024. |
| `GPU_DB_INTENT_LANE_FENCES` | 16 = the drive knee; 24/32 regressed | Hardcoded 16. |
| `GPU_DB_INTENT_LANE_SUBFRAMES` | AUTO won | Hardcoded AUTO. |
| `GPU_DB_INTENT_LANE_SHIP_DIV` | 2 stands | Hardcoded 2. |

## 4. Phase-2 sweep (documented debt, next reckoning)

- `GPU_DB_INTENT_SEQUENCER_SHARDS` (default 1 = off): gates the E2.4a sharded-sequencer arm — a
  recorded NEGATIVE superseded by lanes. Deleting the arm is surgery on the classic sequencer,
  which is itself scheduled for retirement (ADR-006); delete arm + knob + its shards=4 test
  together.
- The 17 `set_*_enabled` runtime setters (sharded-flags campaign): most gate paths that are now
  unconditionally-default (`shard_residency`, `constrained_elision`, `device_write_locate`, …).
  Each needs the same verdict treatment: winner → delete setter + losing arm; the losing arms are
  largely the CPU/host paths whose deletion IS the retirement program.
- `GPU_DB_BATCHING`, `GPU_DB_COMMIT_WAVE_MAX`, `GPU_DB_WAVE_TAIL_PIPELINE_DEPTH`: classic-path
  knobs from earlier campaigns; fold or delete with the classic-path retirement.
- Bench knob prune: `examples/` accumulates `GPU_DB_BENCH_*` vars per campaign; prune dead ones
  when touching each bench.
