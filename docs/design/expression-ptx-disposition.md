# Expression PTX disposition

STRUCT-001FR dispositions the former hand-maintained
`crates/execution/src/expr_proto.ptx` ownership hub. This document is factual evidence for the completed split;
`PLAN.md` remains the only work ledger.

## Decision

The 9,745-line source was neither generated nor cohesive enough for a size exception. It contained 69 independent
`.entry` kernels spanning expression types, gather, derived columns, staged joins, aggregates, fill/finalization,
group compaction, sort, and group execution. It is decomposed into 13 operator/type leaves, all below 1,500 lines.
There are no PTX-to-PTX calls, shared globals, constants, functions, or link-time dependencies; each leaf is a
standalone `.version 8.4`, `.target sm_90`, `.address_size 64` CUDA module.

The CUDA module cache is keyed by the static entry `CStr`, not by the input PTX byte slice. Every surviving symbol
therefore has exactly one canonical PTX definition, and all consumers of a shared symbol embed the same leaf.
Symbol names, parameter declarations, entry bodies, launch geometry, streams, synchronization, and error paths are
unchanged. Smaller per-owner modules intentionally reduce the amount of unrelated PTX presented to the driver when
an entry is first cached; there is no new eager module load.

## History and co-change boundaries

The old file accumulated 9,874 additions and 129 deletions across 77 commits from 2026-06-17 through 2026-07-13.
History confirms distinct clusters rather than one co-changing unit:

- 2026-06-17/18 established the i32, i64, and i128 expression type matrix.
- 2026-06-23/24 added nullable expression finalization, derived/group keys, and four staged-join families.
- 2026-06-29/30 changed gather, grouped aggregate pruning, and group result compaction independently.
- 2026-07-09/10 changed text/UUID predicates without touching aggregate, join, sort, or group kernels.
- 2026-07-12/13 hardened filter, derived-column, and grouped host contracts around otherwise stable kernel ABIs.

Those boundaries are now the file ownership boundaries below.

## Exact symbol and consumer map

The include-site count is the number of live `include_bytes!` sites after decomposition. A symbol used by more than
one Rust owner remains defined once. Full parameter declarations live beside each entry and compare byte-for-byte
with the pre-split source at `044ebf0c`.

| PTX leaf | Lines / entries | Canonical symbols | Rust consumers / include sites |
|---|---:|---|---|
| `expression_i32.ptx` | 520 / 6 | `gpu_db_resident_i32_load_column`; `gpu_db_buffer_i32_binary`; `gpu_db_buffer_i32_binary_scalar`; `gpu_db_buffer_i32_compare_scalar_to_mask`; `gpu_db_buffer_i32_compare_buffers_to_mask`; `gpu_db_mask_binary` | `expression_vm.rs`, `resident_filter.rs` / 2 |
| `resident_gather.ptx` | 271 / 4 | `gpu_db_resident_i32_gather_rows`; `gpu_db_resident_i64_gather_rows`; `gpu_db_resident_i128_gather_rows`; `gpu_db_resident_bool_gather_rows` | `resident_gather.rs` / 1 |
| `expression_i64.ptx` | 717 / 7 | `gpu_db_resident_i64_compare_scalar_to_mask`; `gpu_db_resident_i64_compare_columns_to_mask`; `gpu_db_resident_i64_load_column`; `gpu_db_buffer_i64_binary`; `gpu_db_buffer_i64_binary_scalar`; `gpu_db_buffer_i64_compare_scalar_to_mask`; `gpu_db_buffer_i64_compare_buffers_to_mask` | `expression_vm.rs`, `resident_filter.rs` / 3 |
| `expression_i128.ptx` | 1,113 / 9 | `gpu_db_resident_i128_compare_scalar_to_mask`; `gpu_db_resident_i128_compare_columns_to_mask`; `gpu_db_resident_i128_load_column`; `gpu_db_buffer_i128_binary`; `gpu_db_buffer_i128_binary_scalar`; `gpu_db_buffer_i128_compare_scalar_to_mask`; `gpu_db_buffer_i128_compare_buffers_to_mask`; `gpu_db_buffer_i128_mul_scalar`; `gpu_db_buffer_i128_mul` | `expression_vm.rs`, `resident_filter.rs` / 3 |
| `expression_varlen.ptx` | 1,013 / 7 | `gpu_db_resident_text_eq_scalar_to_mask`; `gpu_db_resident_text_compare_scalar_to_mask`; `gpu_db_resident_text_compare_columns_to_mask`; `gpu_db_resident_text_like_scalar_to_mask`; `gpu_db_resident_uuid_compare_scalar_to_mask`; `gpu_db_resident_uuid_compare_columns_to_mask`; `gpu_db_resident_bool_to_mask` | `expression_vm.rs`, `resident_filter.rs`, `derived_column.rs` / 9 |
| `derived_column.ptx` | 844 / 7 | `gpu_db_pack_two_int4_cols`; `gpu_db_pack_two_cols_i128`; `gpu_db_widen_col_to_i64`; `gpu_db_build_wide_key`; `gpu_db_mark_new_distinct`; `gpu_db_validate_distinct_text_offsets`; `gpu_db_mark_new_distinct_text` | `derived_column.rs` / 6 |
| `staged_hash_join.ptx` | 1,398 / 8 | `gpu_db_hash_join_build_i32`; `gpu_db_hash_join_probe_i32`; `gpu_db_hash_join_build_text`; `gpu_db_hash_join_probe_text`; `gpu_db_hash_join_build_i64_nn`; `gpu_db_hash_join_emit_i64_nn`; `gpu_db_hash_join_build_text_nn`; `gpu_db_hash_join_emit_text_nn` | `staged_hash_join.rs` / 4 |
| `resident_aggregate.ptx` | 552 / 6 | `gpu_db_resident_i32_sum_at_indices`; `gpu_db_resident_i32_minmax_at_indices`; `gpu_db_resident_i64_minmax_at_indices`; `gpu_db_resident_i64_sum_at_indices_i128`; `gpu_db_resident_i128_minmax_partials_at_indices`; `gpu_db_resident_i128_sum_partials_at_indices` | `resident_aggregate.rs` / 6 |
| `device_fill.ptx` | 171 / 3 | `gpu_db_fill_i64`; `gpu_db_blend_widen_null_sentinel`; `gpu_db_fill_i128` | `resident_group.rs`, `staged_hash_join.rs`, `expression_filter.rs` / 7 |
| `resident_group_compact.ptx` | 420 / 1 | `gpu_db_group_by_slot_compact` | `resident_group.rs` / 1 |
| `resident_sort.ptx` | 831 / 4 | `gpu_db_bitonic_sort_i64_step`; `gpu_db_bitonic_sort_text_step`; `gpu_db_bitonic_sort_multikey_step`; `gpu_db_bitonic_sort_hetero_step` | `resident_sort.rs` / 4 |
| `resident_group.ptx` | 1,150 / 1 | `gpu_db_group_by_i32_count_sum` | `resident_group.rs` / 2 |
| `resident_group_extra.ptx` | 448 / 2 | `gpu_db_group_by_numeric_minmax_lo`; `gpu_db_group_by_i32_count_sum_twolevel` | `resident_group.rs` / 2 |

Dependency direction is Rust operator owner to PTX leaf. The only deliberately shared leaves are expression type
primitives and device fill/finalization. No PTX leaf imports another leaf, and no Rust visibility was widened.

## Obsolete content removed

Two entries had no `cached_function` call, launch handle, or other executable Rust reference:

- `gpu_db_buffer_i32_compare_buffers_to_indices` was the old atomic-append/host-sort comparator. The live
  col-vs-col route already uses the ordered two-input compactor.
- `gpu_db_mask_compact_to_indices` was the old atomic-append/host-sort mask compactor. The live predicate and
  filter routes already use `compact_mask_i32_to_indices`, backed by ordered compaction.

STRUCT-001GE later deleted two more entries after the product-live two-column facade moved to the checked postfix
VM plus ordered compaction:

- `gpu_db_resident_i32_binary_elementwise` accepted unchecked resident offsets/opcodes in a special-case launcher.
  The typed VM's load/buffer-binary primitives now provide the same checked arithmetic with aligned exact windows.
- `gpu_db_buffer_i32_compare_to_indices` atomically appended row indices and required a host sort. The shared ordered
  two-pass compactor now emits the same ascending indices by construction.

All four definitions were deleted rather than retained as unreferenced prototypes. Historical comments that explain
route replacement may still name former symbols as legacy behavior; there is no live executable reference.

## Normalized source and ABI proof

The pre-split file has 69 entries; the new leaves now have 65. A parser keyed by `.visible .entry` compared every
surviving entry from its declaration through its closing brace against `044ebf0c`: 65 unchanged bodies, zero added
symbols, and exactly the four obsolete symbols above removed. The declaration is part of that byte comparison, so
parameter order, width, signedness, and count are included. All 13 leaves also assemble independently with local
`ptxas -arch=sm_90`, and the permanent ASCII test now enumerates every leaf.

## GPU evidence

The family matrix runs each route three times sequentially and two copies concurrently, with no CUDA
700/716/717 faults:

| Ownership exercised | Real-GPU gate |
|---|---|
| i32 prototype and i32 mask VM | `cuda_resident_expr_two_col_filter_evaluates_arithmetic_predicate_on_gpu`; `cuda_resident_expr_predicate_filter_evaluates_boolean_predicates_on_gpu` |
| fixed/bool gather | `a4c_device_gather_matches_host_store` |
| i64 VM | `gpu_execute_resident_expr_select_sql_runs_int8_arithmetic` |
| i128 VM | `cuda_buffer_i128_arith_adds_subtracts_with_numeric_overflow` |
| text/UUID/bool predicates | `gpu_resident_expr_col_vs_col_text_uuid` |
| derived/wide/distinct support | `cuda_derived_column_inputs_fail_closed_then_reuse_context` |
| unique and N:N int/text staged joins | `cuda_staged_hash_joins_validate_and_apply_validity_bitmaps` |
| i32/i64/i128 aggregate kernels | `gpu_execute_resident_expr_select_sql_runs_int8_aggregates`; `gpu_execute_resident_expr_select_sql_runs_numeric_sum_avg` |
| fill, main group, dense compaction | `gpu_group_by_expression` |
| nullable expression finalization | `gpu_resident_expr_order_by_places_nulls_per_pg_default` |
| numeric group pass 2 | `gpu_grouped_numeric_min_max_same_high_limb_tie` |
| two-level group | `gpu_execute_resident_expr_select_sql_group_by_two_level_at_scale` |
| i64/text/multikey/heterogeneous sort | `gpu_nongrouped_order_by_mixed_int_text` |

This is a structural ownership change. Device entry bodies and result behavior are unchanged; the standard report
card is nevertheless run because module load/cache behavior changed.
