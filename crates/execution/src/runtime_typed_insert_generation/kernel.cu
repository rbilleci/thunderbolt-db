// Canonical source for kernel.ptx. Regenerate from repository root:
// nvcc --ptx --gpu-architecture=compute_60 --std=c++14 -O3 -I crates/execution/src \
//   -o crates/execution/src/runtime_typed_insert_generation/kernel.ptx \
//   crates/execution/src/runtime_typed_insert_generation/kernel.cu && \
// perl -0pi -e 's/[ \t]+(?=\n)//g; s/\n\n\z/\n/' \
//   crates/execution/src/runtime_typed_insert_generation/kernel.ptx
//
// This is a correctness-first plural typed INSERT generation program.  It accepts flat logical
// typed cells; no branch is specialized for an INT4 table or column count.  The first production
// vertical retains the existing zero-index roots and adds an explicitly bounded indexed route.
// Unsupported breadth fails in the device status word instead of taking the nullable-INT4 rebuild
// or a host hash.

#include "../sha256_device.cuh"

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

enum {
  HEADER_BYTES = 256,
  TABLE_BYTES = 6352,
  ROW_BYTES = 40,
  CELL_BYTES = 48,
  INDEX_BYTES = 80,
  INDEX_KEY_BYTES = 64,
  INDEX_EFFECT_BYTES = 40,
  INDEX_EFFECT_COMPONENT_BYTES = 8,
  MAX_INDEX_KEYS = 32,
  OUTPUT_HEADER_BYTES = 32,
  RADIX_DEPTH = 64,
  EMPTY_ROOTS = 65,
  TABLE_ACTION_CREATE_EMPTY = 1,
  TABLE_ACTION_ROW_SET_INSERT = 2,
  TABLE_ACTION_ENROLL_INDEX = 3,
  TABLE_ACTION_RESET_THEN_ROW_SET_INSERT = 4,
  TABLE_ACTION_CREATE_WITH_ROW_SET = 5,
  TABLE_ACTION_CREATE_INDEX_THEN_ROW_SET_INSERT = 6,
  TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE = 1,
  TABLE_MAP_PREDECESSOR_PINNED = 2,
  TABLE_MAP_PREDECESSOR_PINNED_RETAINED = 3,
  TABLE_MAP_SIBLINGS_OFFSET = 144,
  TABLE_MAP_RETAINED_EMPTY_ROOTS_OFFSET = 2192,
  TABLE_MAP_RETAINED_INITIAL_LEAF_OFFSET = 4272,
  TABLE_MAP_RETAINED_INITIAL_PATH_OFFSET = 4304,
  SLOT_GENERATION_INPUT = 0,
  SLOT_INITIAL_TABLE_ROOT = 1,
  SLOT_FINAL_TABLE_ROOT = 2,
  SLOT_INITIAL_TABLE_MAP_ROOT = 3,
  SLOT_FINAL_TABLE_MAP_ROOT = 4,
  SLOT_INITIAL_DATABASE_ROOT = 5,
  SLOT_FINAL_DATABASE_ROOT = 6,
  SLOT_ROW_EMPTY = 7,
  MAP_NODE_PREFIX_STATE_WORDS = 8,
  MAP_NODE_PREFIX_STATE_BYTES =
      RADIX_DEPTH * MAP_NODE_PREFIX_STATE_WORDS * sizeof(u32),
};

__device__ __constant__ u8 generation_input_domain[] =
    "gpu-db/write001/generation-input/v2";
__device__ __constant__ u8 generation_row_domain[] =
    "gpu-db/write001/generation-row-input/v2";
__device__ __constant__ u8 column_shape_domain[] =
    "gpu-db/runtime-generation/column-shape/v1";
__device__ __constant__ u8 column_manifest_domain[] =
    "gpu-db/runtime-generation/column-manifest/v1";
__device__ __constant__ u8 column_empty_domain[] =
    "gpu-db/runtime-generation/column-empty/v1";
__device__ __constant__ u8 typed_value_domain[] =
    "gpu-db/runtime-generation/typed-value/v1";
__device__ __constant__ u8 current_row_domain[] =
    "gpu-db/runtime-generation/current-row/v1";
__device__ __constant__ u8 row_empty_leaf_domain[] =
    "gpu-db/runtime-generation/row-empty-leaf/v1";
__device__ __constant__ u8 row_empty_node_domain[] =
    "gpu-db/runtime-generation/row-empty-node/v1";
__device__ __constant__ u8 row_leaf_domain[] =
    "gpu-db/runtime-generation/row-leaf/v1";
__device__ __constant__ u8 row_node_domain[] =
    "gpu-db/runtime-generation/row-node/v1";
__device__ __constant__ u8 table_root_domain[] =
    "gpu-db/runtime-generation/table-root/v1";
// A successor commits the exact prior immutable table root plus the newly generated row radix
// tree.  It is deliberately independent of SQL storage: every typed cell has already entered the
// same current-row commitment above.  This lets a later append remain GPU-authenticated without
// rebuilding or staging the predecessor's resident rows on the host.
__device__ __constant__ u8 table_successor_domain[] =
    "gpu-db/runtime-generation/table-successor/v2";
__device__ __constant__ u8 table_reset_successor_domain[] =
    "gpu-db/runtime-generation/table-reset-successor/v1";
__device__ __constant__ u8 map_empty_leaf_domain[] =
    "gpu-db/runtime-generation/map-empty-leaf/v1";
__device__ __constant__ u8 map_empty_node_domain[] =
    "gpu-db/runtime-generation/map-empty-node/v1";
__device__ __constant__ u8 map_leaf_domain[] =
    "gpu-db/runtime-generation/map-leaf/v1";
__device__ __constant__ u8 map_node_domain[] =
    "gpu-db/runtime-generation/map-node/v1";
__device__ __constant__ u8 database_root_domain[] =
    "gpu-db/runtime-generation/database-root/v1";
// The first table uses the materialized one-table map above.  Later immutable table generations
// advance the database witness by chaining the prior database root with the exact table
// predecessor/successor pair; no host map reconstruction is admissible on the append path.
__device__ __constant__ u8 database_successor_domain[] =
    "gpu-db/runtime-generation/database-successor/v2";
// The v2 launch is a data-parallel, type-neutral construction.  Its compact batch tree keeps
// the exact source order in the commitment while allowing one CUDA block to hash every cell and
// row concurrently; it is not an INT4 or fixed-layout special case.
__device__ __constant__ u8 parallel_current_row_domain[] =
    "gpu-db/runtime-generation/current-row/v2";
__device__ __constant__ u8 parallel_empty_row_domain[] =
    "gpu-db/runtime-generation/row-empty/v2";
__device__ __constant__ u8 parallel_row_leaf_domain[] =
    "gpu-db/runtime-generation/row-leaf/v2";
__device__ __constant__ u8 parallel_row_node_domain[] =
    "gpu-db/runtime-generation/batch-row-node/v2";
__device__ __constant__ u8 parallel_table_root_domain[] =
    "gpu-db/runtime-generation/table-root/v2";
__device__ __constant__ u8 parallel_database_genesis_domain[] =
    "gpu-db/runtime-generation/database-genesis/v2";
__device__ __constant__ u8 parallel_generation_input_domain[] =
    "gpu-db/write001/generation-input/v4";
// The live codec-5 writer consumes these named row digests when it emits S7 transitions. They
// bind the same type-neutral logical cells as strict recovery; this is not a fixed-width path.
__device__ __constant__ u8 s7_final_row_domain[] =
    "gpu-db/write001/s7-final-row/v2";
__device__ __constant__ u8 s7_transition_domain[] =
    "gpu-db/write001/s7-transition/v2";
__device__ __constant__ u8 generation_index_shape_domain[] =
    "gpu-db/write001/generation-index-shape/v2";
__device__ __constant__ u8 generation_index_effect_domain[] =
    "gpu-db/write001/generation-index-effect-input/v2";
__device__ __constant__ u8 s7_typed_key_value_domain[] =
    "gpu-db/write001/s7-typed-key-value/v2";
__device__ __constant__ u8 index_shape_domain[] =
    "gpu-db/runtime-generation/index-shape/v1";
__device__ __constant__ u8 index_empty_leaf_domain[] =
    "gpu-db/runtime-generation/index-empty-leaf/v1";
__device__ __constant__ u8 index_empty_node_domain[] =
    "gpu-db/runtime-generation/index-empty-node/v1";
__device__ __constant__ u8 index_entry_domain[] =
    "gpu-db/runtime-generation/index-entry/v1";
__device__ __constant__ u8 index_leaf_domain[] =
    "gpu-db/runtime-generation/index-leaf/v1";
__device__ __constant__ u8 index_node_domain[] =
    "gpu-db/runtime-generation/index-node/v1";
__device__ __constant__ u8 index_root_domain[] =
    "gpu-db/runtime-generation/index-root/v1";
// A nonempty named index advances by committing the authenticated predecessor root plus the
// device-built radix root for this INSERT batch. The predecessor remains immutable and resident;
// neither its entries nor a host index are reconstructed to derive the successor generation.
__device__ __constant__ u8 index_successor_domain[] =
    "gpu-db/runtime-generation/index-successor/v2";

__device__ u16 load_u16(const u8* p) {
  return (u16)p[0] | ((u16)p[1] << 8);
}

__device__ u32 load_u32(const u8* p) {
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ u64 load_u64(const u8* p) {
  u64 value = 0;
  #pragma unroll
  for (u32 byte = 0; byte < 8; ++byte) value |= (u64)p[byte] << (byte * 8);
  return value;
}

__device__ void store_u32(u8* p, u32 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 4; ++byte) p[byte] = (u8)(value >> (byte * 8));
}

__device__ bool table_map_predecessor_is_pinned(u8 predecessor) {
  return predecessor == TABLE_MAP_PREDECESSOR_PINNED ||
         predecessor == TABLE_MAP_PREDECESSOR_PINNED_RETAINED;
}

__device__ void append_u16(u8* p, u32* at, u16 value) {
  p[(*at)++] = (u8)value; p[(*at)++] = (u8)(value >> 8);
}

__device__ void append_i16(u8* p, u32* at, u16 value) { append_u16(p, at, value); }

__device__ void append_u32(u8* p, u32* at, u32 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 4; ++byte) p[(*at)++] = (u8)(value >> (byte * 8));
}

__device__ void append_u64(u8* p, u32* at, u64 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 8; ++byte) p[(*at)++] = (u8)(value >> (byte * 8));
}

__device__ void append_bytes(u8* p, u32* at, const u8* source, u32 count) {
  #pragma unroll 1
  for (u32 byte = 0; byte < count; ++byte) p[(*at)++] = source[byte];
}

__device__ void append_domain(u8* p, u32* at, const u8* domain, u32 count) {
  append_u64(p, at, (u64)count); append_bytes(p, at, domain, count);
}

__device__ void copy_32(u8* destination, const u8* source) {
  #pragma unroll
  for (u32 byte = 0; byte < 32; ++byte) destination[byte] = source[byte];
}

__device__ u8* slot(u8* output, u64 ordinal) {
  return output + OUTPUT_HEADER_BYTES + ordinal * 32ull;
}

__device__ bool equal_32(const u8* left, const u8* right) {
  u8 difference = 0;
  #pragma unroll
  for (u32 byte = 0; byte < 32; ++byte) difference |= left[byte] ^ right[byte];
  return difference == 0;
}

__device__ bool zero_bytes(const u8* value, u32 count) {
  u8 any = 0;
  #pragma unroll 1
  for (u32 byte = 0; byte < count; ++byte) any |= value[byte];
  return any == 0;
}

// Map-node/v1 has a 128-byte preimage. Its first 64-byte block is exactly the
// domain/version/database/depth prefix; the two child roots occupy the second block and the
// third block is fixed SHA padding. The final COW chain is still serial, but the first block of
// each depth is independent. Keep this compressor private and out-of-line so this physical
// optimization cannot inflate the generic typed-value SHA callers or the finalizer frame.
__device__ __noinline__ void v3_sha256_compress_block(const u8* input, u32* state) {
  u32 words[64];
  #pragma unroll
  for (u32 word = 0; word < 16; ++word) {
    u32 value = 0;
    #pragma unroll
    for (u32 byte = 0; byte < 4; ++byte)
      value = (value << 8) | input[word * 4 + byte];
    words[word] = value;
  }
  #pragma unroll
  for (u32 word = 16; word < 64; ++word) {
    const u32 x = words[word - 15], y = words[word - 2];
    const u32 small0 = ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3);
    const u32 small1 = ((y >> 17) | (y << 15)) ^ ((y >> 19) | (y << 13)) ^ (y >> 10);
    words[word] = words[word - 16] + small0 + words[word - 7] + small1;
  }
  u32 a = state[0], b = state[1], c = state[2], d = state[3];
  u32 e = state[4], f = state[5], g = state[6], h = state[7];
  #pragma unroll
  for (u32 word = 0; word < 64; ++word) {
    const u32 s1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^
                   ((e >> 25) | (e << 7));
    const u32 choice = (e & f) ^ (~e & g);
    const u32 temporary1 = h + s1 + choice + gpu_db_sha256_k[word] + words[word];
    const u32 s0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^
                   ((a >> 22) | (a << 10));
    const u32 majority = (a & b) ^ (a & c) ^ (b & c);
    const u32 temporary2 = s0 + majority;
    h = g; g = f; f = e; e = d + temporary1; d = c; c = b; b = a;
    a = temporary1 + temporary2;
  }
  state[0] += a; state[1] += b; state[2] += c; state[3] += d;
  state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __noinline__ void v3_map_node_prefix_state(
    const u8* database_id, u32 depth, u32* state) {
  u8 prefix[64];
  u32 at = 0;
  append_domain(prefix, &at, map_node_domain, sizeof(map_node_domain) - 1);
  append_u16(prefix, &at, 1);
  append_bytes(prefix, &at, database_id, 16);
  prefix[at++] = (u8)depth;
  state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
  state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
  state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
  state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
  v3_sha256_compress_block(prefix, state);
}

__device__ __noinline__ void v3_map_node_from_prefix_state(
    const u32* prefix_state, const u8* left, const u8* right, u8* output) {
  u32 state[MAP_NODE_PREFIX_STATE_WORDS];
  u8 children[64];
  u8 padding[64];
  #pragma unroll
  for (u32 word = 0; word < MAP_NODE_PREFIX_STATE_WORDS; ++word)
    state[word] = prefix_state[word];
  copy_32(children, left);
  copy_32(children + 32, right);
  #pragma unroll
  for (u32 byte = 0; byte < 64; ++byte) padding[byte] = 0;
  padding[0] = 0x80;
  // The full map-node/v1 preimage is 128 bytes, so its SHA bit length is 1024 = 0x0400.
  padding[62] = 0x04;
  v3_sha256_compress_block(children, state);
  v3_sha256_compress_block(padding, state);
  #pragma unroll
  for (u32 word = 0; word < MAP_NODE_PREFIX_STATE_WORDS; ++word) {
    output[word * 4] = (u8)(state[word] >> 24);
    output[word * 4 + 1] = (u8)(state[word] >> 16);
    output[word * 4 + 2] = (u8)(state[word] >> 8);
    output[word * 4 + 3] = (u8)state[word];
  }
}

__device__ u32 compact_level_capacity(u32 rows, u32 shift) {
  if (shift == 64) return rows == 0 ? 0 : 1;
  const u64 span = 1ull << shift;
  const u64 rounded = ((u64)rows + span - 1ull) / span;
  const u64 capacity = rounded + 1ull;
  return (u32)(capacity < rows ? capacity : rows);
}

struct node {
  u64 representative_id;
  u64 count;
  u8 digest[32];
};

__device__ bool logical_value_length(const u8* cell, u32* expected) {
  const bool is_null = cell[22] != 0;
  const u32 actual = load_u32(cell + 28);
  if (is_null) { *expected = 0; return actual == 0; }
  const u8 tag = cell[12];
  if (tag == 1 || tag == 2 || tag == 7) *expected = 4;
  else if (tag == 3 || tag == 8) *expected = 8;
  else if (tag == 4 || tag == 9) *expected = 16;
  else if (tag == 5) *expected = 1;
  else if (tag == 6) *expected = actual;
  else return false;
  return actual == *expected;
}

__device__ bool same_cell_shape(const u8* left, const u8* right) {
  return load_u32(left) == load_u32(right) &&
         load_u32(left + 4) == load_u32(right + 4) &&
         load_u16(left + 8) == load_u16(right + 8) &&
         load_u32(left + 12) == load_u32(right + 12) &&
         load_u32(left + 16) == load_u32(right + 16) &&
         load_u16(left + 20) == load_u16(right + 20);
}

// A later stream phase may reject a malformed descriptor after an earlier phase has begun
// independent work.  The result is never consumed unless the final status remains zero; this
// atomic therefore preserves one generic failure fence without requiring a cross-block barrier.
__device__ void reject_generation_input(u8* output, u32 status) {
  atomicCAS((u32*)output, 0u, status);
}

extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v1(
    const u8* input, u8* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  store_u32(output, 0);

  const u16 root_format = load_u16(input + 104);
  const u16 bootstrap_empty_predecessor = load_u16(input + 106);
  const u32 table_count = load_u32(input + 108);
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u32 index_count = load_u32(input + 124);
  const u8* table = input + load_u64(input + 144);
  const u8* row_base = input + load_u64(input + 136);
  const u8* cell_base = input + load_u64(input + 144);
  const u8* values = input + load_u64(input + 152);
  u8* workspace = (u8*)(u64)load_u64(input + 160);
  const u64 table_id = load_u64(table);
  const u64 base_generation = load_u64(table + 8);
  const u64 commit_sequence = load_u64(input + 64);
  const u64 initial_rows = load_u64(table + 64);
  const u64 final_rows = load_u64(table + 72);
  const bool create_empty = rows == 0;
  const bool append_to_existing = !create_empty && initial_rows != 0;
  if (root_format != 1 || bootstrap_empty_predecessor > 1 || table_count != 1 ||
      cells == 0 || index_count != 0 ||
      table_id == 0 || base_generation == 0 || commit_sequence == 0 ||
      final_rows < initial_rows || final_rows - initial_rows != rows || workspace == 0 ||
      zero_bytes(input, 16) ||
      (create_empty && (!bootstrap_empty_predecessor || initial_rows != 0 ||
                        base_generation != commit_sequence || value_bytes != 0 ||
                        zero_bytes(table + 80, 32) || !zero_bytes(table + 112, 32))) ||
      (!create_empty && (zero_bytes(table + 80, 32) || zero_bytes(table + 112, 32))) ||
      (bootstrap_empty_predecessor &&
       !zero_bytes(table + 16, 32)) ||
      (!bootstrap_empty_predecessor &&
       (zero_bytes(input + 72, 32) || zero_bytes(table + 16, 32)))) {
    store_u32(output, 1); return;
  }

  const u64 shape_slot = SLOT_ROW_EMPTY + EMPTY_ROOTS;
  const u64 typed_slot = shape_slot + cells;
  const u64 column_slot = typed_slot + cells;
  const u64 current_slot = column_slot + cells;
  const u64 row_leaf_slot = current_slot + rows;
  const u64 row_node_slot = row_leaf_slot + rows;
  u64 row_node_capacity = 0;
  for (u32 shift = 1; shift <= RADIX_DEPTH; ++shift) {
    row_node_capacity += compact_level_capacity(rows, shift);
  }
  const u64 table_empty_slot = row_node_slot + row_node_capacity;
  const u64 initial_table_leaf_slot = table_empty_slot + EMPTY_ROOTS;
  const u64 initial_table_path_slot = initial_table_leaf_slot + 1;
  const u64 final_table_leaf_slot = initial_table_path_slot + RADIX_DEPTH;
  const u64 final_table_path_slot = final_table_leaf_slot + 1;
  const u64 digest_count = final_table_path_slot + RADIX_DEPTH;
  store_u32(output + 8, rows);
  store_u32(output + 12, cells);
  store_u32(output + 16, (u32)digest_count);

  // Validate canonical flat ownership and compute generic column/value/current-row roots.
  u64 previous_row_id = 0;
  u32 expected_cell_start = 0;
  u32 expected_value_start = 0;
  u32 first_row_cell_count = 0;
  const u64 logical_input_bytes = (u64)HEADER_BYTES + TABLE_BYTES + (u64)rows * ROW_BYTES +
                                  (u64)cells * CELL_BYTES + value_bytes;
  u8* scratch = workspace;
  u8* row_scratch = workspace + logical_input_bytes + 512ull;
  const u64 unaligned_node_scratch = (u64)(row_scratch + logical_input_bytes + 512ull);
  node* node_scratch = (node*)((unaligned_node_scratch + 15ull) & ~15ull);
  if (create_empty) {
    for (u32 ordinal = 0; ordinal < cells; ++ordinal) {
      const u8* cell = cell_base + (u64)ordinal * CELL_BYTES;
      u32 expected_length = 0;
      if (load_u32(cell) != ordinal || load_u32(cell + 4) == 0 || load_u16(cell + 8) == 0 ||
          cell[22] != 1 || load_u32(cell + 24) != 0 ||
          !logical_value_length(cell, &expected_length)) {
        store_u32(output, 12); return;
      }
      u32 at = 0;
      append_domain(row_scratch, &at, column_shape_domain, sizeof(column_shape_domain) - 1);
      append_u64(row_scratch, &at, table_id);
      append_u64(row_scratch, &at, (u64)load_u32(cell + 4));
      append_i16(row_scratch, &at, load_u16(cell + 8));
      append_bytes(row_scratch, &at, cell + 12, 4);
      append_u32(row_scratch, &at, load_u32(cell + 16));
      append_i16(row_scratch, &at, load_u16(cell + 20));
      gpu_db_sha256_bytes(row_scratch, at, slot(output, shape_slot + ordinal));

      at = 0;
      append_domain(row_scratch, &at, typed_value_domain, sizeof(typed_value_domain) - 1);
      append_bytes(row_scratch, &at, slot(output, shape_slot + ordinal), 32);
      row_scratch[at++] = 1;
      append_u32(row_scratch, &at, 0);
      gpu_db_sha256_bytes(row_scratch, at, slot(output, typed_slot + ordinal));
    }
  }
  for (u32 row_index = 0; row_index < rows; ++row_index) {
    const u8* row = row_base + (u64)row_index * ROW_BYTES;
    const u64 row_id = load_u64(row + 8);
    const u32 cell_start = load_u32(row + 24);
    const u32 cell_count = load_u32(row + 28);
    if (load_u64(row) != table_id || row_id == 0 ||
        (row_index != 0 && row_id != previous_row_id + 1ull) ||
        (row_index != 0 && previous_row_id == ~0ull) || cell_start != expected_cell_start ||
        cell_count == 0 || cell_count > cells - cell_start) {
      store_u32(output, 2); return;
    }
    if (row_index == 0) first_row_cell_count = cell_count;
    if (cell_count != first_row_cell_count) { store_u32(output, 3); return; }
    u32 current_at = 0;
    append_domain(scratch, &current_at, current_row_domain, sizeof(current_row_domain) - 1);
    append_u64(scratch, &current_at, table_id);
    append_u64(scratch, &current_at, row_id);
    append_u64(scratch, &current_at, commit_sequence);
    append_u32(scratch, &current_at, cell_count);
    for (u32 ordinal = 0; ordinal < cell_count; ++ordinal) {
      const u32 cell_index = cell_start + ordinal;
      const u8* cell = cell_base + (u64)cell_index * CELL_BYTES;
      const u32 value_start = load_u32(cell + 24);
      const u32 value_count = load_u32(cell + 28);
      u32 expected_length = 0;
      if (load_u32(cell) != ordinal || load_u32(cell + 4) == 0 || load_u16(cell + 8) == 0 ||
          cell[22] > 1 || value_start != expected_value_start ||
          value_count > value_bytes - value_start || !logical_value_length(cell, &expected_length) ||
          (row_index != 0 && !same_cell_shape(cell, cell_base + (u64)ordinal * CELL_BYTES))) {
        store_u32(output, 4); return;
      }
      if (cell[12] == 5 && !cell[22] && value_count == 1 && values[value_start] > 1) {
        store_u32(output, 5); return;
      }
      u8* preimage = row_scratch;
      u32 at = 0;
      append_domain(preimage, &at, column_shape_domain, sizeof(column_shape_domain) - 1);
      append_u64(preimage, &at, table_id);
      append_u64(preimage, &at, (u64)load_u32(cell + 4));
      append_i16(preimage, &at, load_u16(cell + 8));
      append_bytes(preimage, &at, cell + 12, 4);
      append_u32(preimage, &at, load_u32(cell + 16));
      append_i16(preimage, &at, load_u16(cell + 20));
      gpu_db_sha256_bytes(preimage, at, slot(output, shape_slot + cell_index));

      at = 0;
      append_domain(preimage, &at, typed_value_domain, sizeof(typed_value_domain) - 1);
      append_bytes(preimage, &at, slot(output, shape_slot + cell_index), 32);
      preimage[at++] = cell[22];
      append_u32(preimage, &at, value_count);
      if (value_count != 0) append_bytes(preimage, &at, values + value_start, value_count);
      gpu_db_sha256_bytes(preimage, at, slot(output, typed_slot + cell_index));

      append_u64(scratch, &current_at, (u64)load_u32(cell + 4));
      append_bytes(scratch, &current_at, slot(output, shape_slot + cell_index), 32);
      append_bytes(scratch, &current_at, slot(output, typed_slot + cell_index), 32);
      expected_value_start += value_count;
    }
    gpu_db_sha256_bytes(scratch, current_at, slot(output, current_slot + row_index));
    expected_cell_start += cell_count;
    previous_row_id = row_id;
  }
  if ((!create_empty && expected_cell_start != cells) || expected_value_start != value_bytes ||
      (!create_empty && load_u64(table + 56) < previous_row_id)) {
    store_u32(output, 6); return;
  }

  const u32 column_count = create_empty ? cells : first_row_cell_count;
  for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
    u32 at = 0;
    append_domain(row_scratch, &at, column_empty_domain, sizeof(column_empty_domain) - 1);
    append_u64(row_scratch, &at, table_id);
    append_bytes(row_scratch, &at, slot(output, shape_slot + ordinal), 32);
    append_u64(row_scratch, &at, 0);
    gpu_db_sha256_bytes(row_scratch, at, slot(output, column_slot + ordinal));
  }

  // Empty row-map roots and current-row-backed logical leaves.
  u8 preimage[256];
  u32 at = 0;
  append_domain(preimage, &at, row_empty_leaf_domain, sizeof(row_empty_leaf_domain) - 1);
  append_u16(preimage, &at, root_format); append_u64(preimage, &at, table_id);
  append_u64(preimage, &at, 0);
  gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_ROW_EMPTY + 64));
  for (int depth = 63; depth >= 0; --depth) {
    at = 0; append_domain(preimage, &at, row_empty_node_domain, sizeof(row_empty_node_domain) - 1);
    append_u16(preimage, &at, root_format); append_u64(preimage, &at, table_id);
    preimage[at++] = (u8)depth; append_u64(preimage, &at, 0);
    append_bytes(preimage, &at, slot(output, SLOT_ROW_EMPTY + depth + 1), 32);
    append_bytes(preimage, &at, slot(output, SLOT_ROW_EMPTY + depth + 1), 32);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_ROW_EMPTY + depth));
  }
  for (u32 row_index = 0; row_index < rows; ++row_index) {
    const u64 row_id = load_u64(row_base + (u64)row_index * ROW_BYTES + 8);
    at = 0; append_domain(preimage, &at, row_leaf_domain, sizeof(row_leaf_domain) - 1);
    append_u16(preimage, &at, root_format); append_u64(preimage, &at, table_id);
    append_u64(preimage, &at, row_id); append_u64(preimage, &at, 1);
    append_bytes(preimage, &at, slot(output, current_slot + row_index), 32);
    gpu_db_sha256_bytes(preimage, at, slot(output, row_leaf_slot + row_index));
  }

  // Build the final batch radix tree. Each depth owns a fixed `rows` output segment; only the
  // active prefix is read by the typed proof materializer.
  node* current = node_scratch;
  node* next = current + rows;
  for (u32 row_index = 0; row_index < rows; ++row_index) {
    current[row_index].representative_id = load_u64(row_base + (u64)row_index * ROW_BYTES + 8);
    current[row_index].count = 1;
    copy_32(current[row_index].digest, slot(output, row_leaf_slot + row_index));
  }
  u32 current_count = rows;
  u32 total_nodes = 0;
  u64 node_output_cursor = 0;
  for (int depth = 63; !create_empty && depth >= 0; --depth) {
    const u32 level_capacity = compact_level_capacity(rows, 64u - (u32)depth);
    u32 read = 0, written = 0;
    while (read < current_count) {
      const u64 id = current[read].representative_id;
      const u64 parent = depth == 0 ? 0 : id >> (64u - (u32)depth);
      const bool first_right = ((id >> (63u - (u32)depth)) & 1ull) != 0;
      const node* left = first_right ? 0 : &current[read];
      const node* right = first_right ? &current[read] : 0;
      ++read;
      if (read < current_count) {
        const u64 next_id = current[read].representative_id;
        const u64 next_parent = depth == 0 ? 0 : next_id >> (64u - (u32)depth);
        if (next_parent == parent) {
          const bool next_right = ((next_id >> (63u - (u32)depth)) & 1ull) != 0;
          if (next_right == first_right) { store_u32(output, 7); return; }
          if (next_right) right = &current[read]; else left = &current[read];
          ++read;
        }
      }
      next[written].representative_id = id;
      next[written].count = (left ? left->count : 0) + (right ? right->count : 0);
      at = 0; append_domain(preimage, &at, row_node_domain, sizeof(row_node_domain) - 1);
      append_u16(preimage, &at, root_format); append_u64(preimage, &at, table_id);
      preimage[at++] = (u8)depth; append_u64(preimage, &at, next[written].count);
      const u8* empty = slot(output, SLOT_ROW_EMPTY + depth + 1);
      append_bytes(preimage, &at, left ? left->digest : empty, 32);
      append_bytes(preimage, &at, right ? right->digest : empty, 32);
      gpu_db_sha256_bytes(preimage, at, next[written].digest);
      if (written >= level_capacity) { store_u32(output, 11); return; }
      copy_32(slot(output, row_node_slot + node_output_cursor + written),
              next[written].digest);
      ++written;
    }
    total_nodes += written;
    node* swap = current; current = next; next = swap;
    current_count = written;
    node_output_cursor += level_capacity;
  }
  if (!create_empty && (current_count != 1 || current[0].count != rows)) {
    store_u32(output, 8); return;
  }
  store_u32(output + 4, total_nodes);

  // Initial/final table manifests.
  const u8* column_shapes = slot(output, shape_slot);
  u8 column_manifest_root[32];
  at = 0;
  append_domain(scratch, &at, column_manifest_domain, sizeof(column_manifest_domain) - 1);
  append_u64(scratch, &at, table_id);
  append_u32(scratch, &at, column_count);
  for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
    append_bytes(scratch, &at, column_shapes + (u64)ordinal * 32ull, 32);
    append_bytes(scratch, &at, slot(output, column_slot + ordinal), 32);
  }
  gpu_db_sha256_bytes(scratch, at, column_manifest_root);
  if (append_to_existing) {
    // A nonempty predecessor is represented only by its already-authenticated root.  Rebuilding
    // it from a host image would create a second write authority, so bind that exact root and
    // make the appended batch a new immutable successor commitment instead.
    copy_32(slot(output, SLOT_INITIAL_TABLE_ROOT), table + 16);
  } else {
    at = 0; append_domain(preimage, &at, table_root_domain, sizeof(table_root_domain) - 1);
    append_u64(preimage, &at, table_id); append_u64(preimage, &at, base_generation);
    append_u64(preimage, &at, 0); append_bytes(preimage, &at, slot(output, SLOT_ROW_EMPTY), 32);
    append_u32(preimage, &at, column_count); append_bytes(preimage, &at, column_manifest_root, 32);
    append_u32(preimage, &at, 0);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_INITIAL_TABLE_ROOT));
    if (!bootstrap_empty_predecessor &&
        !equal_32(slot(output, SLOT_INITIAL_TABLE_ROOT), table + 16)) {
      store_u32(output, 9); return;
    }
  }
  if (append_to_existing) {
    at = 0; append_domain(preimage, &at, table_successor_domain,
                          sizeof(table_successor_domain) - 1);
    append_u16(preimage, &at, root_format); append_u64(preimage, &at, table_id);
    append_u64(preimage, &at, base_generation); append_u64(preimage, &at, commit_sequence);
    append_u64(preimage, &at, initial_rows); append_u64(preimage, &at, final_rows);
    append_bytes(preimage, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
    append_bytes(preimage, &at, current[0].digest, 32);
    append_u32(preimage, &at, column_count); append_bytes(preimage, &at, column_manifest_root, 32);
    append_u32(preimage, &at, 0);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_FINAL_TABLE_ROOT));
  } else {
    at = 0; append_domain(preimage, &at, table_root_domain, sizeof(table_root_domain) - 1);
    append_u64(preimage, &at, table_id); append_u64(preimage, &at, commit_sequence);
    append_u64(preimage, &at, rows);
    append_bytes(preimage, &at,
                 create_empty ? slot(output, SLOT_ROW_EMPTY) : current[0].digest, 32);
    append_u32(preimage, &at, column_count); append_bytes(preimage, &at, column_manifest_root, 32);
    append_u32(preimage, &at, 0);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_FINAL_TABLE_ROOT));
  }

  // One-table persistent map, once for the empty-table predecessor and once for the successor.
  const u8* database_id = input;
  at = 0; append_domain(preimage, &at, map_empty_leaf_domain, sizeof(map_empty_leaf_domain) - 1);
  append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
  gpu_db_sha256_bytes(preimage, at, slot(output, table_empty_slot + 64));
  for (int depth = 63; depth >= 0; --depth) {
    at = 0; append_domain(preimage, &at, map_empty_node_domain, sizeof(map_empty_node_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    preimage[at++] = (u8)depth;
    append_bytes(preimage, &at, slot(output, table_empty_slot + depth + 1), 32);
    append_bytes(preimage, &at, slot(output, table_empty_slot + depth + 1), 32);
    gpu_db_sha256_bytes(preimage, at, slot(output, table_empty_slot + depth));
  }
  at = 0; append_domain(preimage, &at, map_leaf_domain, sizeof(map_leaf_domain) - 1);
  append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
  append_u64(preimage, &at, table_id); append_bytes(preimage, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
  gpu_db_sha256_bytes(preimage, at, slot(output, initial_table_leaf_slot));
  at = 0; append_domain(preimage, &at, map_leaf_domain, sizeof(map_leaf_domain) - 1);
  append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
  append_u64(preimage, &at, table_id); append_bytes(preimage, &at, slot(output, SLOT_FINAL_TABLE_ROOT), 32);
  gpu_db_sha256_bytes(preimage, at, slot(output, final_table_leaf_slot));
  for (int depth = 63; depth >= 0; --depth) {
    const u8* empty = slot(output, table_empty_slot + depth + 1);
    const u8* initial_child = depth == 63 ? slot(output, initial_table_leaf_slot)
                                          : slot(output, initial_table_path_slot + depth + 1);
    const u8* final_child = depth == 63 ? slot(output, final_table_leaf_slot)
                                        : slot(output, final_table_path_slot + depth + 1);
    at = 0; append_domain(preimage, &at, map_node_domain, sizeof(map_node_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    preimage[at++] = (u8)depth;
    if (((table_id >> (63u - (u32)depth)) & 1ull) == 0) {
      append_bytes(preimage, &at, initial_child, 32); append_bytes(preimage, &at, empty, 32);
    } else {
      append_bytes(preimage, &at, empty, 32); append_bytes(preimage, &at, initial_child, 32);
    }
    gpu_db_sha256_bytes(preimage, at, slot(output, initial_table_path_slot + depth));
    at = 0; append_domain(preimage, &at, map_node_domain, sizeof(map_node_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    preimage[at++] = (u8)depth;
    if (((table_id >> (63u - (u32)depth)) & 1ull) == 0) {
      append_bytes(preimage, &at, final_child, 32); append_bytes(preimage, &at, empty, 32);
    } else {
      append_bytes(preimage, &at, empty, 32); append_bytes(preimage, &at, final_child, 32);
    }
    gpu_db_sha256_bytes(preimage, at, slot(output, final_table_path_slot + depth));
  }
  copy_32(slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), slot(output, initial_table_path_slot));
  copy_32(slot(output, SLOT_FINAL_TABLE_MAP_ROOT), slot(output, final_table_path_slot));
  if (append_to_existing) {
    copy_32(slot(output, SLOT_INITIAL_DATABASE_ROOT), input + 72);
    at = 0; append_domain(preimage, &at, database_successor_domain,
                          sizeof(database_successor_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    append_bytes(preimage, &at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32);
    append_u64(preimage, &at, table_id);
    append_bytes(preimage, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
    append_bytes(preimage, &at, slot(output, SLOT_FINAL_TABLE_ROOT), 32);
    append_u64(preimage, &at, base_generation); append_u64(preimage, &at, commit_sequence);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_FINAL_DATABASE_ROOT));
  } else {
    at = 0; append_domain(preimage, &at, database_root_domain, sizeof(database_root_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    append_bytes(preimage, &at, slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), 32);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_INITIAL_DATABASE_ROOT));
    at = 0; append_domain(preimage, &at, database_root_domain, sizeof(database_root_domain) - 1);
    append_u16(preimage, &at, root_format); append_bytes(preimage, &at, database_id, 16);
    append_bytes(preimage, &at, slot(output, SLOT_FINAL_TABLE_MAP_ROOT), 32);
    gpu_db_sha256_bytes(preimage, at, slot(output, SLOT_FINAL_DATABASE_ROOT));
    if (!bootstrap_empty_predecessor &&
        !equal_32(slot(output, SLOT_INITIAL_DATABASE_ROOT), input + 72)) {
      store_u32(output, 10); return;
    }
  }

  // Independently compute the semantics-v2 neutral input digest from the flat logical facts.
  // Bootstrap sentinels never enter the commitment: the GPU-derived predecessor roots do.
  u32 top_at = 0;
  append_domain(scratch, &top_at, generation_input_domain, sizeof(generation_input_domain) - 1);
  append_bytes(scratch, &top_at, input, 16);
  append_u64(scratch, &top_at, load_u64(input + 16));
  append_bytes(scratch, &top_at, input + 24, 32);
  append_u64(scratch, &top_at, load_u64(input + 56));
  append_u64(scratch, &top_at, commit_sequence);
  append_bytes(scratch, &top_at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32);
  append_u32(scratch, &top_at, 1);
  append_u64(scratch, &top_at, table_id);
  append_u64(scratch, &top_at, base_generation);
  append_bytes(scratch, &top_at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
  append_u64(scratch, &top_at, load_u64(table + 48));
  append_u64(scratch, &top_at, load_u64(table + 56));
  append_u64(scratch, &top_at, initial_rows);
  append_u64(scratch, &top_at, final_rows);
  append_u32(scratch, &top_at, rows);
  for (u32 row_index = 0; row_index < rows; ++row_index) {
    const u8* row = row_base + (u64)row_index * ROW_BYTES;
    const u32 cell_start = load_u32(row + 24);
    const u32 cell_count = load_u32(row + 28);
    u32 row_at = 0;
    append_domain(row_scratch, &row_at, generation_row_domain, sizeof(generation_row_domain) - 1);
    append_u64(row_scratch, &row_at, table_id);
    append_u64(row_scratch, &row_at, load_u64(row + 8));
    append_u32(row_scratch, &row_at, load_u32(row + 16));
    append_u32(row_scratch, &row_at, load_u32(row + 20));
    append_u32(row_scratch, &row_at, cell_count);
    for (u32 ordinal = 0; ordinal < cell_count; ++ordinal) {
      const u8* cell = cell_base + (u64)(cell_start + ordinal) * CELL_BYTES;
      const u32 value_start = load_u32(cell + 24);
      const u32 value_count = load_u32(cell + 28);
      append_u32(row_scratch, &row_at, load_u32(cell));
      append_u32(row_scratch, &row_at, load_u32(cell + 4));
      append_i16(row_scratch, &row_at, load_u16(cell + 8));
      append_bytes(row_scratch, &row_at, cell + 12, 4);
      append_u32(row_scratch, &row_at, load_u32(cell + 16));
      append_i16(row_scratch, &row_at, load_u16(cell + 20));
      row_scratch[row_at++] = cell[22];
      append_u32(row_scratch, &row_at, value_count);
      if (value_count != 0) append_bytes(row_scratch, &row_at, values + value_start, value_count);
    }
    u8 row_digest[32];
    gpu_db_sha256_bytes(row_scratch, row_at, row_digest);
    append_bytes(scratch, &top_at, row_digest, 32);
  }
  append_bytes(scratch, &top_at, table + 80, 32);
  append_bytes(scratch, &top_at, table + 112, 32);
  append_u32(scratch, &top_at, 0);
  gpu_db_sha256_bytes(scratch, top_at, slot(output, SLOT_GENERATION_INPUT));
}

// The first implementation above established the descriptor grammar.  This successor preserves
// that grammar and output layout while making the independent typed-cell and row commitments
// parallel.  A 1,000-row SQL statement therefore remains one generic CUDA generation, rather
// than one CUDA thread serially hashing every value in an otherwise GPU-native write path.
extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v2(
    const u8* input, u8* output) {
  if (blockIdx.x != 0) return;
  const u32 tid = threadIdx.x;
  __shared__ u32 status;
  __shared__ u32 column_count;

  if (tid == 0) {
    status = 0;
    store_u32(output, 0);
    const u16 root_format = load_u16(input + 104);
    const u16 bootstrap = load_u16(input + 106);
    const u32 tables = load_u32(input + 108);
    const u32 rows = load_u32(input + 112);
    const u32 cells = load_u32(input + 116);
    const u32 value_bytes = load_u32(input + 120);
    const u32 indexes = load_u32(input + 124);
    const u8* table = input + load_u64(input + 128);
    const u64 table_id = load_u64(table);
    const u64 base_generation = load_u64(table + 8);
    const u64 commit_sequence = load_u64(input + 64);
    const u64 initial_rows = load_u64(table + 64);
    const u64 final_rows = load_u64(table + 72);
    const bool create_empty = rows == 0;
    if (root_format != 1 || bootstrap > 1 || tables != 1 || cells == 0 || indexes != 0 ||
        table_id == 0 || base_generation == 0 || commit_sequence == 0 ||
        load_u64(input + 160) == 0) {
      status = 1;
    } else if (create_empty) {
      if (!bootstrap || initial_rows != 0 || final_rows != 0 || base_generation != commit_sequence ||
          value_bytes != 0 || zero_bytes(table + 80, 32) || !zero_bytes(table + 112, 32) ||
          !zero_bytes(table + 16, 32) || !zero_bytes(input + 72, 32)) status = 2;
      column_count = cells;
    } else {
      if (bootstrap || cells % rows != 0 || initial_rows > final_rows ||
          final_rows - initial_rows != rows || zero_bytes(table + 16, 32) ||
          zero_bytes(input + 72, 32) || zero_bytes(table + 80, 32) ||
          zero_bytes(table + 112, 32)) status = 3;
      column_count = cells / rows;
    }
  }
  __syncthreads();
  if (status != 0) { if (tid == 0) store_u32(output, status); return; }

  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u8* table = input + load_u64(input + 128);
  const u8* row_base = input + load_u64(input + 136);
  const u8* cell_base = input + load_u64(input + 144);
  const u8* values = input + load_u64(input + 152);
  u8* workspace = (u8*)(u64)load_u64(input + 160);
  const u64 table_id = load_u64(table);
  const u64 base_generation = load_u64(table + 8);
  const u64 commit_sequence = load_u64(input + 64);
  const u64 initial_rows = load_u64(table + 64);
  const u64 final_rows = load_u64(table + 72);
  const bool create_empty = rows == 0;
  const bool append_to_existing = !create_empty && initial_rows != 0;
  const u64 shape_slot = SLOT_ROW_EMPTY + EMPTY_ROOTS;
  const u64 typed_slot = shape_slot + cells;
  const u64 column_slot = typed_slot + cells;
  const u64 current_slot = column_slot + cells;
  const u64 row_leaf_slot = current_slot + rows;
  const u64 row_node_slot = row_leaf_slot + rows;
  u64 row_node_capacity = 0;
  for (u32 shift = 1; shift <= RADIX_DEPTH; ++shift) {
    row_node_capacity += compact_level_capacity(rows, shift);
  }
  const u64 table_empty_slot = row_node_slot + row_node_capacity;
  const u64 initial_table_leaf_slot = table_empty_slot + EMPTY_ROOTS;
  const u64 initial_table_path_slot = initial_table_leaf_slot + 1;
  const u64 final_table_leaf_slot = initial_table_path_slot + RADIX_DEPTH;
  const u64 final_table_path_slot = final_table_leaf_slot + 1;
  const u64 digest_count = final_table_path_slot + RADIX_DEPTH;

  // Structural validation is intentionally linear but hash-free. It protects the parallel
  // writers from malformed offsets while keeping all typed value commitments on the device.
  if (tid == 0 && !create_empty) {
    u64 previous_row_id = 0;
    u32 expected_cell_start = 0, expected_value_start = 0;
    for (u32 row_index = 0; row_index < rows && status == 0; ++row_index) {
      const u8* row = row_base + (u64)row_index * ROW_BYTES;
      const u64 row_id = load_u64(row + 8);
      if (load_u64(row) != table_id || row_id == 0 ||
          (row_index != 0 && row_id != previous_row_id + 1ull) ||
          (row_index != 0 && previous_row_id == ~0ull) ||
          load_u32(row + 24) != expected_cell_start ||
          load_u32(row + 28) != column_count) { status = 4; break; }
      previous_row_id = row_id;
      expected_cell_start += column_count;
    }
    for (u32 cell_index = 0; cell_index < cells && status == 0; ++cell_index) {
      const u8* cell = cell_base + (u64)cell_index * CELL_BYTES;
      const u32 value_start = load_u32(cell + 24);
      const u32 value_count = load_u32(cell + 28);
      u32 expected_length = 0;
      const u32 ordinal = cell_index % column_count;
      if (load_u32(cell) != ordinal || load_u32(cell + 4) == 0 || load_u16(cell + 8) == 0 ||
          cell[22] > 1 || value_start != expected_value_start || value_start > value_bytes ||
          value_count > value_bytes - value_start ||
          !logical_value_length(cell, &expected_length) ||
          !same_cell_shape(cell, cell_base + (u64)ordinal * CELL_BYTES) ||
          (cell[12] == 5 && !cell[22] && value_count == 1 && values[value_start] > 1)) {
        status = 5; break;
      }
      expected_value_start += value_count;
    }
    if (status == 0 && (expected_cell_start != cells || expected_value_start != value_bytes ||
                        load_u64(table + 56) < previous_row_id)) status = 6;
  }
  if (tid == 0 && create_empty) {
    for (u32 ordinal = 0; ordinal < cells && status == 0; ++ordinal) {
      const u8* cell = cell_base + (u64)ordinal * CELL_BYTES;
      u32 expected_length = 0;
      if (load_u32(cell) != ordinal || load_u32(cell + 4) == 0 || load_u16(cell + 8) == 0 ||
          cell[22] != 1 || load_u32(cell + 24) != 0 || !logical_value_length(cell, &expected_length)) status = 7;
    }
  }
  __syncthreads();
  if (status != 0) { if (tid == 0) store_u32(output, status); return; }

  // Every cell owns an independent shape and typed-value commitment. Variable-width values use
  // a disjoint prefix in the already-reserved workspace, never a host staging vector.
  for (u32 cell_index = tid; cell_index < cells; cell_index += blockDim.x) {
    const u8* cell = cell_base + (u64)cell_index * CELL_BYTES;
    u8 local[128];
    u32 at = 0;
    append_domain(local, &at, column_shape_domain, sizeof(column_shape_domain) - 1);
    append_u64(local, &at, table_id); append_u64(local, &at, (u64)load_u32(cell + 4));
    append_i16(local, &at, load_u16(cell + 8)); append_bytes(local, &at, cell + 12, 4);
    append_u32(local, &at, load_u32(cell + 16)); append_i16(local, &at, load_u16(cell + 20));
    gpu_db_sha256_bytes(local, at, slot(output, shape_slot + cell_index));

    const u32 value_start = load_u32(cell + 24);
    const u32 value_count = load_u32(cell + 28);
    u8* typed_preimage = workspace + (u64)cell_index * 96ull + value_start;
    at = 0;
    append_domain(typed_preimage, &at, typed_value_domain, sizeof(typed_value_domain) - 1);
    append_bytes(typed_preimage, &at, slot(output, shape_slot + cell_index), 32);
    typed_preimage[at++] = cell[22]; append_u32(typed_preimage, &at, value_count);
    if (value_count != 0) append_bytes(typed_preimage, &at, values + value_start, value_count);
    gpu_db_sha256_bytes(typed_preimage, at, slot(output, typed_slot + cell_index));
  }
  __syncthreads();

  // Prefix capacity covers the length-delimited v2 domain plus table/row/generation and the
  // three u32 provenance fields.  The former 64-byte constant was smaller than that prefix,
  // causing adjacent rows to overlap once their current-row hashes ran concurrently.
  const u32 current_bytes = 96u + 72u * column_count;
  if (!create_empty) {
    for (u32 row_index = tid; row_index < rows; row_index += blockDim.x) {
      const u8* row = row_base + (u64)row_index * ROW_BYTES;
      u8* current_preimage = workspace + (u64)row_index * current_bytes;
      u32 at = 0;
      append_domain(current_preimage, &at, parallel_current_row_domain,
                    sizeof(parallel_current_row_domain) - 1);
      append_u64(current_preimage, &at, table_id); append_u64(current_preimage, &at, load_u64(row + 8));
      append_u64(current_preimage, &at, commit_sequence); append_u32(current_preimage, &at, load_u32(row + 16));
      append_u32(current_preimage, &at, load_u32(row + 20)); append_u32(current_preimage, &at, column_count);
      const u32 cell_start = load_u32(row + 24);
      for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
        const u8* cell = cell_base + (u64)(cell_start + ordinal) * CELL_BYTES;
        append_u64(current_preimage, &at, (u64)load_u32(cell + 4));
        append_bytes(current_preimage, &at, slot(output, shape_slot + cell_start + ordinal), 32);
        append_bytes(current_preimage, &at, slot(output, typed_slot + cell_start + ordinal), 32);
      }
      gpu_db_sha256_bytes(current_preimage, at, slot(output, current_slot + row_index));
    }
  }
  for (u32 ordinal = tid; ordinal < column_count; ordinal += blockDim.x) {
    u8 local[128]; u32 at = 0;
    append_domain(local, &at, column_empty_domain, sizeof(column_empty_domain) - 1);
    append_u64(local, &at, table_id); append_bytes(local, &at, slot(output, shape_slot + ordinal), 32);
    append_u64(local, &at, 0); gpu_db_sha256_bytes(local, at, slot(output, column_slot + ordinal));
  }
  __syncthreads();

  if (tid == 0) {
    u8 local[128]; u32 at = 0;
    append_domain(local, &at, parallel_empty_row_domain, sizeof(parallel_empty_row_domain) - 1);
    append_u16(local, &at, 1); append_u64(local, &at, table_id);
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_ROW_EMPTY));
    for (u32 depth = 1; depth < EMPTY_ROOTS; ++depth)
      copy_32(slot(output, SLOT_ROW_EMPTY + depth), slot(output, SLOT_ROW_EMPTY));
    store_u32(output + 8, rows); store_u32(output + 12, cells); store_u32(output + 16, (u32)digest_count);
  }
  __syncthreads();

  const u64 logical_input_bytes = (u64)HEADER_BYTES + TABLE_BYTES + (u64)rows * ROW_BYTES +
                                  (u64)cells * CELL_BYTES + value_bytes;
  const u64 node_at = (u64)(workspace + logical_input_bytes * 2ull + 512ull);
  node* node_scratch = (node*)((node_at + 15ull) & ~15ull);
  node* nodes_a = node_scratch;
  node* nodes_b = nodes_a + rows;
  if (!create_empty) {
    for (u32 row_index = tid; row_index < rows; row_index += blockDim.x) {
      nodes_a[row_index].representative_id = load_u64(row_base + (u64)row_index * ROW_BYTES + 8);
      nodes_a[row_index].count = 1;
      u8 local[128]; u32 at = 0;
      append_domain(local, &at, parallel_row_leaf_domain, sizeof(parallel_row_leaf_domain) - 1);
      append_u16(local, &at, 1); append_u64(local, &at, table_id);
      append_u64(local, &at, nodes_a[row_index].representative_id); append_u64(local, &at, 1);
      append_bytes(local, &at, slot(output, current_slot + row_index), 32);
      gpu_db_sha256_bytes(local, at, nodes_a[row_index].digest);
      copy_32(slot(output, row_leaf_slot + row_index), nodes_a[row_index].digest);
    }
  }
  __syncthreads();

  bool current_is_a = true;
  u32 active = rows;
  u32 cursor = 0;
  u32 total_nodes = 0;
  if (rows <= 1024) {
    for (u32 level = 0; active > 1; ++level) {
      const u32 next_count = (active + 1) / 2;
      node* current = current_is_a ? nodes_a : nodes_b;
      node* next = current_is_a ? nodes_b : nodes_a;
      for (u32 node_index = tid; node_index < next_count; node_index += blockDim.x) {
        const node* left = current + node_index * 2;
        const node* right = node_index * 2 + 1 < active ? left + 1 : 0;
        next[node_index].representative_id = left->representative_id;
        next[node_index].count = left->count + (right ? right->count : 0);
        u8 local[256]; u32 at = 0;
        append_domain(local, &at, parallel_row_node_domain, sizeof(parallel_row_node_domain) - 1);
        append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, commit_sequence);
        append_u32(local, &at, level); append_u64(local, &at, left->count);
        append_u64(local, &at, right ? right->count : 0);
        append_bytes(local, &at, left->digest, 32);
        append_bytes(local, &at, right ? right->digest : slot(output, SLOT_ROW_EMPTY), 32);
        gpu_db_sha256_bytes(local, at, next[node_index].digest);
        copy_32(slot(output, row_node_slot + cursor + node_index), next[node_index].digest);
      }
      __syncthreads();
      total_nodes += next_count;
      cursor += compact_level_capacity(rows, level + 1);
      active = next_count;
      current_is_a = !current_is_a;
    }
  } else if (tid == 0) {
    u32 level = 0;
    while (active > 1) {
      const u32 next_count = (active + 1) / 2;
      node* current = current_is_a ? nodes_a : nodes_b;
      node* next = current_is_a ? nodes_b : nodes_a;
      for (u32 node_index = 0; node_index < next_count; ++node_index) {
        const node* left = current + node_index * 2;
        const node* right = node_index * 2 + 1 < active ? left + 1 : 0;
        next[node_index].representative_id = left->representative_id;
        next[node_index].count = left->count + (right ? right->count : 0);
        u8 local[256]; u32 at = 0;
        append_domain(local, &at, parallel_row_node_domain, sizeof(parallel_row_node_domain) - 1);
        append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, commit_sequence);
        append_u32(local, &at, level); append_u64(local, &at, left->count);
        append_u64(local, &at, right ? right->count : 0); append_bytes(local, &at, left->digest, 32);
        append_bytes(local, &at, right ? right->digest : slot(output, SLOT_ROW_EMPTY), 32);
        gpu_db_sha256_bytes(local, at, next[node_index].digest);
        copy_32(slot(output, row_node_slot + cursor + node_index), next[node_index].digest);
      }
      total_nodes += next_count; cursor += compact_level_capacity(rows, level + 1);
      active = next_count; current_is_a = !current_is_a; ++level;
    }
  }
  __syncthreads();

  if (tid == 0) {
    node* final_nodes = current_is_a ? nodes_a : nodes_b;
    const u8* batch_root = create_empty ? slot(output, SLOT_ROW_EMPTY) : final_nodes[0].digest;
    u8* manifest = workspace;
    u32 at = 0;
    append_domain(manifest, &at, column_manifest_domain, sizeof(column_manifest_domain) - 1);
    append_u64(manifest, &at, table_id); append_u32(manifest, &at, column_count);
    for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
      append_bytes(manifest, &at, slot(output, shape_slot + ordinal), 32);
      append_bytes(manifest, &at, slot(output, column_slot + ordinal), 32);
    }
    u8 column_manifest_root[32]; gpu_db_sha256_bytes(manifest, at, column_manifest_root);

    // `generation-input/v3` commits two 32-byte image digests in addition to the complete
    // identity/table lineage and exceeds the old fixed 256-byte thread-local scratch.  This
    // final serial section runs after all workspace consumers above have completed, so reuse
    // the already-reserved device arena instead of imposing a type- or column-count ceiling.
    u8* local = workspace;
    if (append_to_existing) copy_32(slot(output, SLOT_INITIAL_TABLE_ROOT), table + 16);
    else {
      at = 0; append_domain(local, &at, parallel_table_root_domain, sizeof(parallel_table_root_domain) - 1);
      append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, base_generation);
      append_u64(local, &at, 0); append_bytes(local, &at, slot(output, SLOT_ROW_EMPTY), 32);
      append_u32(local, &at, column_count); append_bytes(local, &at, column_manifest_root, 32);
      gpu_db_sha256_bytes(local, at, slot(output, SLOT_INITIAL_TABLE_ROOT));
      if (!create_empty && !equal_32(slot(output, SLOT_INITIAL_TABLE_ROOT), table + 16)) status = 8;
    }
    if (create_empty) copy_32(slot(output, SLOT_FINAL_TABLE_ROOT), slot(output, SLOT_INITIAL_TABLE_ROOT));
    else {
      at = 0; append_domain(local, &at, table_successor_domain, sizeof(table_successor_domain) - 1);
      append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, base_generation);
      append_u64(local, &at, commit_sequence); append_u64(local, &at, initial_rows); append_u64(local, &at, final_rows);
      append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32); append_bytes(local, &at, batch_root, 32);
      append_u32(local, &at, column_count); append_bytes(local, &at, column_manifest_root, 32);
      gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_TABLE_ROOT));
    }
    if (create_empty) {
      at = 0; append_domain(local, &at, parallel_database_genesis_domain,
                            sizeof(parallel_database_genesis_domain) - 1);
      append_u16(local, &at, 1); append_bytes(local, &at, input, 16); append_u64(local, &at, table_id);
      append_bytes(local, &at, slot(output, SLOT_FINAL_TABLE_ROOT), 32);
      gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_DATABASE_ROOT));
      copy_32(slot(output, SLOT_INITIAL_DATABASE_ROOT), slot(output, SLOT_FINAL_DATABASE_ROOT));
    } else {
      copy_32(slot(output, SLOT_INITIAL_DATABASE_ROOT), input + 72);
      at = 0; append_domain(local, &at, database_successor_domain, sizeof(database_successor_domain) - 1);
      append_u16(local, &at, 1); append_bytes(local, &at, input, 16);
      append_bytes(local, &at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32); append_u64(local, &at, table_id);
      append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
      append_bytes(local, &at, slot(output, SLOT_FINAL_TABLE_ROOT), 32);
      append_u64(local, &at, base_generation); append_u64(local, &at, commit_sequence);
      gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_DATABASE_ROOT));
    }
    if (status == 0) {
      copy_32(slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), slot(output, SLOT_INITIAL_TABLE_ROOT));
      copy_32(slot(output, SLOT_FINAL_TABLE_MAP_ROOT), slot(output, SLOT_FINAL_TABLE_ROOT));
      for (u32 ordinal = 0; ordinal < EMPTY_ROOTS; ++ordinal)
        copy_32(slot(output, table_empty_slot + ordinal), slot(output, SLOT_INITIAL_DATABASE_ROOT));
      copy_32(slot(output, initial_table_leaf_slot), slot(output, SLOT_INITIAL_TABLE_ROOT));
      copy_32(slot(output, final_table_leaf_slot), slot(output, SLOT_FINAL_TABLE_ROOT));
      for (u32 depth = 0; depth < RADIX_DEPTH; ++depth) {
        copy_32(slot(output, initial_table_path_slot + depth), slot(output, SLOT_INITIAL_TABLE_ROOT));
        copy_32(slot(output, final_table_path_slot + depth), slot(output, SLOT_FINAL_TABLE_ROOT));
      }
      at = 0; append_domain(local, &at, parallel_generation_input_domain,
                            sizeof(parallel_generation_input_domain) - 1);
      append_bytes(local, &at, input, 16); append_u64(local, &at, load_u64(input + 16));
      append_bytes(local, &at, input + 24, 32); append_u64(local, &at, load_u64(input + 56));
      append_u64(local, &at, commit_sequence); append_bytes(local, &at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32);
      append_u64(local, &at, table_id); append_u64(local, &at, base_generation);
      append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
      append_u64(local, &at, load_u64(table + 48)); append_u64(local, &at, load_u64(table + 56));
      append_u64(local, &at, initial_rows); append_u64(local, &at, final_rows);
      append_u32(local, &at, rows); append_bytes(local, &at, batch_root, 32);
      append_bytes(local, &at, table + 80, 32); append_bytes(local, &at, table + 112, 32); append_u32(local, &at, 0);
      gpu_db_sha256_bytes(local, at, slot(output, SLOT_GENERATION_INPUT));
    }
    store_u32(output, status); store_u32(output + 4, total_nodes);
  }
}

// The indexed route has one device-owned logical index map. Its facts remain flat in the
// descriptor; these helpers derive both index grammars from that sealed input and never accept a
// caller-provided root or entry digest.
__device__ void v3_one_index_shapes(
    const u8* index, const u8* key_base, u64 table_id, u64 shape_slot, u8* output,
    u8* index_shape, u8* generation_shape) {
  const u64 index_id = load_u64(index);
  const u32 key_start = load_u32(index + 64);
  const u32 key_count = load_u32(index + 68);
  u8 local[4096];
  u32 at = 0;
  append_domain(local, &at, index_shape_domain, sizeof(index_shape_domain) - 1);
  append_u64(local, &at, table_id);
  append_u64(local, &at, index_id);
  append_u32(local, &at, load_u32(index + 12));
  local[at++] = index[16];
  #pragma unroll
  for (u32 byte = 0; byte < 32; ++byte) local[at++] = 0;
  append_u32(local, &at, key_count);
  for (u32 ordinal = 0; ordinal < key_count; ++ordinal) {
    const u8* key = key_base + (u64)(key_start + ordinal) * INDEX_KEY_BYTES;
    append_u64(local, &at, (u64)load_u32(key + 8));
    append_bytes(local, &at, slot(output, shape_slot + load_u32(key + 4)), 32);
  }
  gpu_db_sha256_bytes(local, at, index_shape);

  at = 0;
  append_domain(local, &at, generation_index_shape_domain,
                sizeof(generation_index_shape_domain) - 1);
  append_u64(local, &at, table_id);
  append_u64(local, &at, index_id);
  append_u32(local, &at, load_u32(index + 12));
  local[at++] = index[16];
  append_u64(local, &at, load_u64(index + 24));
  append_bytes(local, &at, index + 32, 32);
  append_u32(local, &at, key_count);
  for (u32 ordinal = 0; ordinal < key_count; ++ordinal) {
    const u8* key = key_base + (u64)(key_start + ordinal) * INDEX_KEY_BYTES;
    append_u32(local, &at, load_u32(key));
    append_u32(local, &at, load_u32(key + 4));
    append_u32(local, &at, load_u32(key + 8));
    append_i16(local, &at, load_u16(key + 12));
    append_bytes(local, &at, key + 16, 4);
    append_u32(local, &at, load_u32(key + 20));
    append_i16(local, &at, load_u16(key + 24));
    append_bytes(local, &at, key + 32, 32);
  }
  gpu_db_sha256_bytes(local, at, generation_shape);
}

__device__ void v3_index_empty_roots(
    u64 table_id, const u8* index, const u8* index_shape, u8* empty_roots) {
  const u64 index_id = load_u64(index);
  u8 local[256];
  u32 at = 0;
  append_domain(local, &at, index_empty_leaf_domain, sizeof(index_empty_leaf_domain) - 1);
  append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, index_id);
  append_bytes(local, &at, index_shape, 32); append_u64(local, &at, 0);
  gpu_db_sha256_bytes(local, at, empty_roots + 64ull * 32ull);
  for (int depth = 63; depth >= 0; --depth) {
    at = 0;
    append_domain(local, &at, index_empty_node_domain, sizeof(index_empty_node_domain) - 1);
    append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, index_id);
    append_bytes(local, &at, index_shape, 32); local[at++] = (u8)depth; append_u64(local, &at, 0);
    append_bytes(local, &at, empty_roots + (u64)(depth + 1) * 32ull, 32);
    append_bytes(local, &at, empty_roots + (u64)(depth + 1) * 32ull, 32);
    gpu_db_sha256_bytes(local, at, empty_roots + (u64)depth * 32ull);
  }
}

__device__ void v3_index_root(
    u64 table_id, const u8* index, u64 generation, const u8* index_shape, u64 entry_count,
    const u8* content_root, u8* root) {
  u8 local[192]; u32 at = 0;
  append_domain(local, &at, index_root_domain, sizeof(index_root_domain) - 1);
  append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
  append_u64(local, &at, generation); append_bytes(local, &at, index_shape, 32);
  append_u64(local, &at, entry_count); append_bytes(local, &at, content_root, 32);
  gpu_db_sha256_bytes(local, at, root);
}

__device__ void v3_index_successor_root(
    u64 table_id, const u8* index, u64 generation, const u8* index_shape,
    u64 initial_entry_count, u64 final_entry_count, u64 batch_entry_count,
    const u8* batch_content_root, u8* root) {
  u8 local[256]; u32 at = 0;
  append_domain(local, &at, index_successor_domain, sizeof(index_successor_domain) - 1);
  append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
  append_u64(local, &at, load_u64(index + 24)); append_bytes(local, &at, index + 32, 32);
  append_u64(local, &at, generation); append_bytes(local, &at, index_shape, 32);
  append_u64(local, &at, initial_entry_count); append_u64(local, &at, final_entry_count);
  append_u64(local, &at, batch_entry_count); append_bytes(local, &at, batch_content_root, 32);
  gpu_db_sha256_bytes(local, at, root);
}

// A CREATE INDEX inside a transaction over a published table has no prior index root, but it
// does have one authenticated table prefix.  Bind that prefix and its exact row horizon to the
// typed suffix rather than treating a zero index root as an empty table/index.  The host cannot
// select this grammar without the S3 stable-identity proof; the GPU still derives the root.
__device__ void v3_created_index_successor_root(
    u64 table_id, const u8* index, u64 generation, const u8* index_shape,
    const u8* initial_table_root, u64 initial_entry_count, u64 final_entry_count,
    u64 batch_entry_count, const u8* batch_content_root, u8* root) {
  u8 local[288]; u32 at = 0;
  append_domain(local, &at, "gpu-db/runtime-generation/index-created-successor/v1", 52);
  append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
  append_u64(local, &at, generation); append_bytes(local, &at, index_shape, 32);
  append_bytes(local, &at, initial_table_root, 32);
  append_u64(local, &at, initial_entry_count); append_u64(local, &at, final_entry_count);
  append_u64(local, &at, batch_entry_count); append_bytes(local, &at, batch_content_root, 32);
  gpu_db_sha256_bytes(local, at, root);
}

__device__ bool v3_index_entry(
    const u8* row_base, const u8* cell_base, const u8* values, const u8* key_base,
    const u8* effect_base, const u8* component_base, const u8* index, u64 table_id,
    u64 commit_sequence, u64 typed_slot, u8* output, const u8* index_shape, u32 row_ordinal,
    u8* entry) {
  const u8* row = row_base + (u64)row_ordinal * ROW_BYTES;
  const u8* effect = effect_base +
      (u64)(load_u32(index + 72) + row_ordinal) * INDEX_EFFECT_BYTES;
  const u32 cell_start = load_u32(row + 24);
  const u32 component_start = load_u32(effect + 28);
  const u32 key_start = load_u32(index + 64);
  const u32 key_count = load_u32(index + 68);
  u8 local[2048];
  u32 at = 0;
  append_domain(local, &at, index_entry_domain, sizeof(index_entry_domain) - 1);
  append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
  append_bytes(local, &at, index_shape, 32); append_u64(local, &at, load_u64(row + 8));
  append_u64(local, &at, commit_sequence); append_u32(local, &at, key_count);
  for (u32 ordinal = 0; ordinal < key_count; ++ordinal) {
    const u8* key = key_base + (u64)(key_start + ordinal) * INDEX_KEY_BYTES;
    const u8* component = component_base + (u64)(component_start + ordinal) * INDEX_EFFECT_COMPONENT_BYTES;
    const u32 column_ordinal = load_u32(component);
    const u8* cell = cell_base + (u64)(cell_start + column_ordinal) * CELL_BYTES;
    append_u64(local, &at, (u64)load_u32(key + 8));
    // The opaque current typed root was completed by the preceding stream phase.  It is the
    // canonical value committed by the logical index leaf, while the S7-key form below closes
    // the independent generation-input effect grammar.
    append_bytes(local, &at, slot(output, typed_slot + cell_start + column_ordinal), 32);
    if (load_u32(component) != load_u32(key + 4) || load_u32(component + 4) != load_u32(key + 8) ||
        load_u32(cell) != column_ordinal || load_u32(cell + 4) != load_u32(key + 8) ||
        load_u16(cell + 8) != load_u16(key + 12) ||
        load_u32(cell + 16) != load_u32(key + 20) || load_u16(cell + 20) != load_u16(key + 24)) {
      return false;
    }
  }
  gpu_db_sha256_bytes(local, at, entry);
  return true;
}

// V3 keeps the codec-v2 commitment grammar and output slots, but turns its internal
// block-local barriers into explicit same-stream kernel boundaries.  That permits independent
// cells and rows to occupy many SMs without treating any SQL storage type as a distinct route.
struct v3_layout {
  u64 shape_slot;
  u64 typed_slot;
  u64 column_slot;
  u64 current_slot;
  u64 s7_final_row_slot;
  u64 s7_transition_slot;
  u64 row_leaf_slot;
  u64 row_node_slot;
  u64 row_node_capacity;
  u64 table_empty_slot;
  u64 initial_table_leaf_slot;
  u64 initial_table_path_slot;
  u64 final_table_leaf_slot;
  u64 final_table_path_slot;
  u64 index_initial_roots_slot;
  u64 index_final_roots_slot;
  u64 digest_count;
};

__device__ void v3_output_layout(u32 rows, u32 cells, u32 indexes, v3_layout* layout) {
  layout->shape_slot = SLOT_ROW_EMPTY + EMPTY_ROOTS;
  layout->typed_slot = layout->shape_slot + cells;
  layout->column_slot = layout->typed_slot + cells;
  layout->current_slot = layout->column_slot + cells;
  layout->s7_final_row_slot = layout->current_slot + rows;
  layout->s7_transition_slot = layout->s7_final_row_slot + rows;
  layout->row_leaf_slot = layout->s7_transition_slot + rows;
  layout->row_node_slot = layout->row_leaf_slot + rows;
  layout->row_node_capacity = 0;
  for (u32 shift = 1; shift <= RADIX_DEPTH; ++shift)
    layout->row_node_capacity += compact_level_capacity(rows, shift);
  layout->table_empty_slot = layout->row_node_slot + layout->row_node_capacity;
  layout->initial_table_leaf_slot = layout->table_empty_slot + EMPTY_ROOTS;
  layout->initial_table_path_slot = layout->initial_table_leaf_slot + 1;
  layout->final_table_leaf_slot = layout->initial_table_path_slot + RADIX_DEPTH;
  layout->final_table_path_slot = layout->final_table_leaf_slot + 1;
  layout->index_initial_roots_slot = layout->final_table_path_slot + RADIX_DEPTH;
  layout->index_final_roots_slot = layout->index_initial_roots_slot + indexes;
  layout->digest_count = layout->index_final_roots_slot + indexes;
}

// V3 rows materialize two type-neutral, variable-width preimage arenas concurrently: the
// runtime current-row commitment, then the codec-5 S7 final-row commitment.  The host reserves
// their exact generic geometry before the alternating tree-node arena.
__device__ u64 v3_current_row_preimage_bytes(u32 rows, u32 cells) {
  return (u64)rows * 96ull + (u64)cells * 72ull;
}

__device__ u64 v3_s7_final_row_preimage_bytes(u32 rows, u32 cells, u32 value_bytes) {
  return (u64)rows * 67ull + (u64)cells * 25ull + value_bytes;
}

// This mirrors the host ABI's maximum of the established/index and v3 arenas. Prefix states sit
// after that complete reservation: the indexed finalizer reuses `input_bytes + 512` below for
// effect scratch, so placing a block-shared handoff beside v3's node arena would race it.
__device__ u64 v3_workspace_capacity(const u8* input) {
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u64 exact_input_bytes = load_u64(input + 208) +
      (u64)load_u32(input + 136) * INDEX_EFFECT_COMPONENT_BYTES;
  const u64 scratch = exact_input_bytes + 512ull;
  const u64 node_bytes = (u64)rows * 48ull * 2ull;
  const u64 established_capacity = scratch * 2ull + 16ull + node_bytes;
  const u64 v3_capacity = v3_current_row_preimage_bytes(rows, cells) +
      v3_s7_final_row_preimage_bytes(rows, cells, value_bytes) + 512ull + node_bytes;
  return established_capacity > v3_capacity ? established_capacity : v3_capacity;
}

__device__ u32* v3_map_node_prefix_states(const u8* input, u8* workspace) {
  u64 state_at = (u64)workspace + v3_workspace_capacity(input);
  state_at = (state_at + 3ull) & ~3ull;
  return (u32*)state_at;
}

__device__ node* v3_nodes_a(u8* workspace, u32 rows, u32 cells, u32 value_bytes) {
  const u64 node_at = (u64)(workspace +
      v3_current_row_preimage_bytes(rows, cells) +
      v3_s7_final_row_preimage_bytes(rows, cells, value_bytes) + 512ull);
  return (node*)((node_at + 15ull) & ~15ull);
}

// Build the fixed-height membership map bottom-up in the already-reserved alternating node
// arenas. This avoids recursive device stack growth while retaining the documented MSB-first
// sparse radix grammar for every contiguous first-seed row range.
__device__ bool v3_index_content_root(
    const u8* row_base, const u8* cell_base, const u8* values, const u8* key_base,
    const u8* effect_base, const u8* component_base, const u8* index, u64 table_id,
    u64 commit_sequence, u64 typed_slot, u8* output, const u8* index_shape,
    const u8* empty_roots, node* nodes_a, node* nodes_b, u32 rows, u8* root) {
  for (u32 row = 0; row < rows; ++row) {
    u8 entry[32];
    if (!v3_index_entry(row_base, cell_base, values, key_base, effect_base, component_base,
                        index, table_id, commit_sequence, typed_slot, output, index_shape,
                        row, entry)) return false;
    node& leaf = nodes_a[row];
    leaf.representative_id = load_u64(row_base + (u64)row * ROW_BYTES + 8);
    leaf.count = 1;
    u8 local[256]; u32 at = 0;
    append_domain(local, &at, index_leaf_domain, sizeof(index_leaf_domain) - 1);
    append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
    append_bytes(local, &at, index_shape, 32); append_u64(local, &at, leaf.representative_id);
    append_u64(local, &at, 1); append_bytes(local, &at, entry, 32);
    gpu_db_sha256_bytes(local, at, leaf.digest);
  }
  node* current = nodes_a; node* next = nodes_b; u32 active = rows;
  for (int depth = 63; depth >= 0; --depth) {
    u32 written = 0;
    for (u32 position = 0; position < active;) {
      const node* first = current + position;
      const u64 first_id = first->representative_id;
      const u64 parent = depth == 0 ? 0ull : first_id >> (64 - (u32)depth);
      const bool first_is_right = ((first_id >> (63 - (u32)depth)) & 1ull) != 0;
      const node* left = first_is_right ? 0 : first;
      const node* right = first_is_right ? first : 0;
      ++position;
      if (!first_is_right && position < active) {
        const node* candidate = current + position;
        const u64 candidate_parent = depth == 0 ? 0ull :
            candidate->representative_id >> (64 - (u32)depth);
        const bool candidate_is_right =
            ((candidate->representative_id >> (63 - (u32)depth)) & 1ull) != 0;
        if (candidate_parent == parent && candidate_is_right) { right = candidate; ++position; }
      }
      node& parent_node = next[written++];
      parent_node.representative_id = first_id;
      parent_node.count = (left ? left->count : 0) + (right ? right->count : 0);
      u8 local[256]; u32 at = 0;
      append_domain(local, &at, index_node_domain, sizeof(index_node_domain) - 1);
      append_u16(local, &at, 1); append_u64(local, &at, table_id); append_u64(local, &at, load_u64(index));
      append_bytes(local, &at, index_shape, 32); local[at++] = (u8)depth;
      append_u64(local, &at, parent_node.count);
      append_bytes(local, &at, left ? left->digest : empty_roots + (u64)(depth + 1) * 32ull, 32);
      append_bytes(local, &at, right ? right->digest : empty_roots + (u64)(depth + 1) * 32ull, 32);
      gpu_db_sha256_bytes(local, at, parent_node.digest);
    }
    if (written == 0 || written > active) return false;
    active = written;
    node* swap = current; current = next; next = swap;
  }
  if (active != 1 || current[0].count != rows) return false;
  copy_32(root, current[0].digest);
  return true;
}

// The preflight owns all structural checks and publishes only a status/header.  Later kernels
// observe this status through the ordered stream before they dereference any row/cell payload.
extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_validate(
    const u8* input, u8* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  store_u32(output, 0);

  const u16 root_format = load_u16(input + 104);
  const u8 action = input[106];
  const u8 table_map_predecessor = input[107];
  const u32 tables = load_u32(input + 108);
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u32 indexes = load_u32(input + 124);
  const u32 index_keys = load_u32(input + 128);
  const u32 index_effects = load_u32(input + 132);
  const u32 index_effect_components = load_u32(input + 136);
  const u8* table = input + load_u64(input + 144);
  const u64 table_id = load_u64(table);
  const u64 base_generation = load_u64(table + 8);
  const u64 commit_sequence = load_u64(input + 64);
  const u64 initial_rows = load_u64(table + 64);
  const u64 final_rows = load_u64(table + 72);
  const bool create_empty = action == TABLE_ACTION_CREATE_EMPTY;
  const bool row_set_insert = action == TABLE_ACTION_ROW_SET_INSERT;
  const bool enroll_index = action == TABLE_ACTION_ENROLL_INDEX;
  const bool reset_row_set = action == TABLE_ACTION_RESET_THEN_ROW_SET_INSERT;
  const bool create_with_row_set = action == TABLE_ACTION_CREATE_WITH_ROW_SET;
  const bool create_index_then_row_set = action == TABLE_ACTION_CREATE_INDEX_THEN_ROW_SET_INSERT;
  const bool zero_row_reset = reset_row_set && rows == 0;
  u32 status = 0;
  const u64 expected_row_offset = (u64)HEADER_BYTES + TABLE_BYTES;
  const u64 expected_cell_offset = expected_row_offset + (u64)rows * ROW_BYTES;
  const u64 expected_value_offset = expected_cell_offset + (u64)cells * CELL_BYTES;
  const u64 expected_index_offset = expected_value_offset + value_bytes;
  const u64 expected_key_offset = expected_index_offset + (u64)indexes * INDEX_BYTES;
  const u64 expected_effect_offset = expected_key_offset + (u64)index_keys * INDEX_KEY_BYTES;
  const u64 expected_component_offset = expected_effect_offset + (u64)index_effects * INDEX_EFFECT_BYTES;

  if (root_format != 1 ||
      (!create_empty && !row_set_insert && !enroll_index && !reset_row_set && !create_with_row_set && !create_index_then_row_set) ||
      (table_map_predecessor != TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE &&
       !table_map_predecessor_is_pinned(table_map_predecessor)) ||
      tables != 1 || cells == 0 || table_id == 0 || base_generation == 0 || commit_sequence == 0 ||
      load_u64(input + 176) == 0 ||
      load_u64(input + 144) != HEADER_BYTES || load_u64(input + 152) != expected_row_offset ||
      load_u64(input + 160) != expected_cell_offset || load_u64(input + 168) != expected_value_offset ||
      load_u64(input + 184) != expected_index_offset || load_u64(input + 192) != expected_key_offset ||
      load_u64(input + 200) != expected_effect_offset || load_u64(input + 208) != expected_component_offset) {
    status = 1;
  } else if (indexes == 0 && (index_keys != 0 || index_effects != 0 ||
                              index_effect_components != 0)) {
    status = 12;
  } else if (create_empty) {
    if (rows != 0 || initial_rows != 0 || final_rows != 0 || base_generation != commit_sequence ||
        value_bytes != 0 || zero_bytes(table + 80, 32) || !zero_bytes(table + 112, 32) ||
        !zero_bytes(table + 16, 32) ||
        (indexes == 0 && index_keys != 0) ||
        (indexes != 0 && index_keys == 0) ||
        index_effects != 0 || index_effect_components != 0 ||
        (table_map_predecessor == TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE &&
         !zero_bytes(input + 72, 32))) status = 2;
  } else if (row_set_insert) {
    if (rows == 0 || !table_map_predecessor_is_pinned(table_map_predecessor) ||
        cells % rows != 0 || initial_rows > final_rows ||
        final_rows - initial_rows != rows || zero_bytes(table + 16, 32) ||
        zero_bytes(input + 72, 32) || zero_bytes(table + 80, 32) ||
        zero_bytes(table + 112, 32) || base_generation >= commit_sequence) status = 3;
  } else if (create_index_then_row_set) {
    if (rows == 0 || !table_map_predecessor_is_pinned(table_map_predecessor) ||
        cells % rows != 0 || initial_rows == 0 || initial_rows > final_rows ||
        final_rows - initial_rows != rows || zero_bytes(table + 16, 32) ||
        zero_bytes(input + 72, 32) || zero_bytes(table + 80, 32) ||
        zero_bytes(table + 112, 32) || base_generation >= commit_sequence) status = 22;
  } else if (create_with_row_set) {
    if (rows == 0 || cells % rows != 0 || initial_rows != 0 || final_rows != rows ||
        base_generation != commit_sequence || !zero_bytes(table + 16, 32) ||
        zero_bytes(table + 80, 32) || zero_bytes(table + 112, 32) ||
        (table_map_predecessor == TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE &&
         !zero_bytes(input + 72, 32))) status = 21;
  } else if (reset_row_set) {
    if (!table_map_predecessor_is_pinned(table_map_predecessor) ||
        (!zero_row_reset && cells % rows != 0) || final_rows != rows ||
        (zero_row_reset && value_bytes != 0) || zero_bytes(table + 16, 32) ||
        zero_bytes(input + 72, 32) || zero_bytes(table + 80, 32) ||
        (zero_row_reset ? !zero_bytes(table + 112, 32) : zero_bytes(table + 112, 32)) ||
        base_generation >= commit_sequence) status = 20;
  } else if (rows != 0 || !table_map_predecessor_is_pinned(table_map_predecessor) ||
             initial_rows != final_rows || zero_bytes(table + 16, 32) ||
             zero_bytes(input + 72, 32) || zero_bytes(table + 80, 32) ||
             !zero_bytes(table + 112, 32) || base_generation >= commit_sequence || indexes != 1 ||
             index_keys == 0 || index_keys > MAX_INDEX_KEYS ||
             index_effects != 0 || index_effect_components != 0) {
    status = 13;
  }

  // Index descriptors remain flat, catalog-ordered, and range-addressed. Each index has its own
  // 1..32 ordered key slice and one maintenance effect per inserted row; the device validates
  // every range before deriving any root.
  if (status == 0 && indexes != 0) {
    const u8* index_base = input + load_u64(input + 184);
    const u8* key_base = input + load_u64(input + 192);
    const u8* effect_base = input + load_u64(input + 200);
    const u8* component_base = input + load_u64(input + 208);
    const u8* cell_base = input + load_u64(input + 160);
    const u32 column_count = (create_empty || enroll_index || zero_row_reset) ? cells : cells / rows;
    u32 expected_key_start = 0;
    u32 expected_effect_start = 0;
    u32 expected_component_start = 0;
    u64 prior_index_id = 0;
    for (u32 index_ordinal = 0; index_ordinal < indexes && status == 0; ++index_ordinal) {
      const u8* index = index_base + (u64)index_ordinal * INDEX_BYTES;
      const u64 index_id = load_u64(index);
      const u32 raw_catalog_ordinal = load_u32(index + 8);
      const u32 index_flags = load_u32(index + 12);
      const u32 key_start = load_u32(index + 64);
      const u32 key_count = load_u32(index + 68);
      const u32 effect_start = load_u32(index + 72);
      const u32 effect_count = load_u32(index + 76);
      const bool malformed_flags = (index_flags & ~15u) != 0 || (index_flags & 8u) == 0 ||
                                   ((index_flags & 6u) != 0 && (index_flags & 1u) == 0);
      if (index_id == 0 || (index_ordinal != 0 && index_id <= prior_index_id) ||
          ((!enroll_index || indexes != 1) && raw_catalog_ordinal != index_ordinal) ||
          malformed_flags || index[16] != 1 || !zero_bytes(index + 17, 7) ||
          key_start != expected_key_start || key_count == 0 || key_count > MAX_INDEX_KEYS ||
          effect_start != expected_effect_start ||
          ((create_empty || enroll_index) &&
           (!zero_bytes(index + 32, 32) || load_u64(index + 24) != 0 || effect_count != 0)) ||
          (create_with_row_set &&
           (!zero_bytes(index + 32, 32) || load_u64(index + 24) != 0 ||
            rows > 0x7fffffffu || effect_count != rows)) ||
          (row_set_insert && (zero_bytes(index + 32, 32) || load_u64(index + 24) == 0 ||
                              rows > 0x7fffffffu || effect_count != rows)) ||
          (create_index_then_row_set &&
           (!zero_bytes(index + 32, 32) != (load_u64(index + 24) != 0) ||
            rows > 0x7fffffffu || effect_count != rows)) ||
          (reset_row_set &&
           ((zero_bytes(index + 32, 32) != (load_u64(index + 24) == 0)) ||
            rows > 0x7fffffffu || effect_count != rows))) {
        status = 14; break;
      }
      prior_index_id = index_id;
      expected_key_start += key_count;
      expected_effect_start += effect_count;
      for (u32 ordinal = 0; ordinal < key_count && status == 0; ++ordinal) {
        const u8* key = key_base + (u64)(key_start + ordinal) * INDEX_KEY_BYTES;
        const u32 column_ordinal = load_u32(key + 4);
        if (load_u32(key) != ordinal || column_ordinal >= column_count ||
            load_u32(key + 8) == 0 || load_u16(key + 12) == 0 || !zero_bytes(key + 14, 2) ||
            !zero_bytes(key + 26, 6) || zero_bytes(key + 32, 32)) {
          status = 15;
        }
        if (status == 0) {
          const u8* cell = cell_base + (u64)column_ordinal * CELL_BYTES;
          if (load_u32(key + 8) != load_u32(cell + 4) ||
              load_u16(key + 12) != load_u16(cell + 8) ||
              load_u32(key + 16) != load_u32(cell + 12) ||
              load_u32(key + 20) != load_u32(cell + 16) ||
              load_u16(key + 24) != load_u16(cell + 20)) status = 15;
        }
        for (u32 prior = 0; prior < ordinal && status == 0; ++prior) {
          const u8* prior_key = key_base + (u64)(key_start + prior) * INDEX_KEY_BYTES;
          if (load_u32(prior_key + 4) == column_ordinal ||
              load_u32(prior_key + 8) == load_u32(key + 8)) status = 15;
        }
      }
      for (u32 row = 0; row < effect_count && status == 0; ++row) {
        const u8* effect = effect_base + (u64)(effect_start + row) * INDEX_EFFECT_BYTES;
        const u8* row_input = input + load_u64(input + 152) + (u64)row * ROW_BYTES;
        const u32 component_start = load_u32(effect + 28);
        if (load_u64(effect) != table_id || load_u64(effect + 8) != index_id ||
            load_u64(effect + 16) != load_u64(row_input + 8) ||
            load_u32(effect + 24) != raw_catalog_ordinal ||
            component_start != expected_component_start ||
            load_u32(effect + 32) != key_count || !zero_bytes(effect + 36, 4)) {
          status = 16; break;
        }
        for (u32 ordinal = 0; ordinal < key_count; ++ordinal) {
          const u8* key = key_base + (u64)(key_start + ordinal) * INDEX_KEY_BYTES;
          const u8* component = component_base +
              (u64)(component_start + ordinal) * INDEX_EFFECT_COMPONENT_BYTES;
          if (load_u32(component) != load_u32(key + 4) ||
              load_u32(component + 4) != load_u32(key + 8)) { status = 17; break; }
        }
        expected_component_start += key_count;
      }
    }
    if (status == 0 &&
        (expected_key_start != index_keys || expected_effect_start != index_effects ||
         expected_component_start != index_effect_components)) status = 14;
  }

  // Per-row ordering and per-cell typed/value framing are checked by the independent workers in
  // the following two stream-ordered phases. INSERT supplies consecutive ids; a full catalog
  // rewrite retains a strictly ordered sparse set. Both use the same stable-id row grammar.
  if (status == 0) {
    v3_layout layout;
    v3_output_layout(rows, cells, indexes, &layout);
    store_u32(output + 8, rows);
    store_u32(output + 12, cells);
    store_u32(output + 16, (u32)layout.digest_count);
  }
  store_u32(output, status);
}

// Every generic typed cell owns one disjoint shape/value commitment and one disjoint workspace
// range.  This phase has no cross-block dependence.
extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_cells(
    const u8* input, u8* output) {
  if (load_u32(output) != 0) return;
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const bool zero_row_shape = input[106] == TABLE_ACTION_CREATE_EMPTY ||
                              input[106] == TABLE_ACTION_ENROLL_INDEX ||
                              (input[106] == TABLE_ACTION_RESET_THEN_ROW_SET_INSERT && rows == 0);
  const u32 column_count = zero_row_shape ? cells : cells / rows;
  const u8* table = input + load_u64(input + 144);
  const u8* cell_base = input + load_u64(input + 160);
  const u8* values = input + load_u64(input + 168);
  u8* workspace = (u8*)(u64)load_u64(input + 176);
  const u64 table_id = load_u64(table);
  v3_layout layout;
  v3_output_layout(load_u32(input + 112), cells, load_u32(input + 124), &layout);
  const u64 first = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const u64 stride = (u64)gridDim.x * blockDim.x;
  for (u64 index = first; index < cells; index += stride) {
    const u32 cell_index = (u32)index;
    const u8* cell = cell_base + (u64)cell_index * CELL_BYTES;
    u32 expected_length = 0;
    if (zero_row_shape) {
      if (load_u32(cell) != cell_index || load_u32(cell + 4) == 0 ||
          load_u16(cell + 8) == 0 || cell[22] != 1 || load_u32(cell + 24) != 0 ||
          !logical_value_length(cell, &expected_length)) {
        reject_generation_input(output, 7); return;
      }
    } else {
      const u32 value_start = load_u32(cell + 24);
      const u32 value_count = load_u32(cell + 28);
      const u64 value_end = (u64)value_start + (u64)value_count;
      const u64 expected_value_start = cell_index == 0 ? 0ull :
          (u64)load_u32(cell_base + (u64)(cell_index - 1) * CELL_BYTES + 24) +
          (u64)load_u32(cell_base + (u64)(cell_index - 1) * CELL_BYTES + 28);
      const u32 ordinal = cell_index % column_count;
      if (load_u32(cell) != ordinal || load_u32(cell + 4) == 0 ||
          load_u16(cell + 8) == 0 || cell[22] > 1 ||
          (u64)value_start != expected_value_start || value_end > value_bytes ||
          !logical_value_length(cell, &expected_length) ||
          !same_cell_shape(cell, cell_base + (u64)ordinal * CELL_BYTES) ||
          (cell[12] == 5 && !cell[22] && value_count == 1 &&
           value_start < value_bytes && values[value_start] > 1)) {
        reject_generation_input(output, 5); return;
      }
      if (cell_index + 1 == cells && value_end != value_bytes) {
        reject_generation_input(output, 6); return;
      }
    }
    u8 local[128];
    u32 at = 0;
    append_domain(local, &at, column_shape_domain, sizeof(column_shape_domain) - 1);
    append_u64(local, &at, table_id);
    append_u64(local, &at, (u64)load_u32(cell + 4));
    append_i16(local, &at, load_u16(cell + 8));
    append_bytes(local, &at, cell + 12, 4);
    append_u32(local, &at, load_u32(cell + 16));
    append_i16(local, &at, load_u16(cell + 20));
    gpu_db_sha256_bytes(local, at, slot(output, layout.shape_slot + cell_index));

    const u32 value_start = load_u32(cell + 24);
    const u32 value_count = load_u32(cell + 28);
    u8* typed_preimage = workspace + (u64)cell_index * 96ull + value_start;
    at = 0;
    append_domain(typed_preimage, &at, typed_value_domain, sizeof(typed_value_domain) - 1);
    append_bytes(typed_preimage, &at, slot(output, layout.shape_slot + cell_index), 32);
    typed_preimage[at++] = cell[22];
    append_u32(typed_preimage, &at, value_count);
    if (value_count != 0) append_bytes(typed_preimage, &at, values + value_start, value_count);
    gpu_db_sha256_bytes(typed_preimage, at, slot(output, layout.typed_slot + cell_index));
  }
}

// Row/image leaves and column empties depend on the completed cell phase but are mutually
// independent.  One task index therefore safely covers both data-driven sets.
extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_rows(
    const u8* input, u8* output) {
  if (load_u32(output) != 0) return;
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const bool zero_row_shape = input[106] == TABLE_ACTION_CREATE_EMPTY ||
                              input[106] == TABLE_ACTION_ENROLL_INDEX ||
                              (input[106] == TABLE_ACTION_RESET_THEN_ROW_SET_INSERT && rows == 0);
  const u32 column_count = zero_row_shape ? cells : cells / rows;
  const u8* table = input + load_u64(input + 144);
  const u8* row_base = input + load_u64(input + 152);
  const u8* cell_base = input + load_u64(input + 160);
  const u8* values = input + load_u64(input + 168);
  u8* workspace = (u8*)(u64)load_u64(input + 176);
  const u64 table_id = load_u64(table);
  const u64 commit_sequence = load_u64(input + 64);
  v3_layout layout;
  v3_output_layout(rows, cells, load_u32(input + 124), &layout);
  const u64 first = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const u64 stride = (u64)gridDim.x * blockDim.x;

  if (first == 0) {
    u8 local[128];
    u32 at = 0;
    append_domain(local, &at, parallel_empty_row_domain, sizeof(parallel_empty_row_domain) - 1);
    append_u16(local, &at, 1);
    append_u64(local, &at, table_id);
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_ROW_EMPTY));
    for (u32 depth = 1; depth < EMPTY_ROOTS; ++depth)
      copy_32(slot(output, SLOT_ROW_EMPTY + depth), slot(output, SLOT_ROW_EMPTY));
  }

  const u64 work = rows > column_count ? rows : column_count;
  for (u64 index = first; index < work; index += stride) {
    if (index < rows) {
      const u32 row_index = (u32)index;
      const u8* row = row_base + (u64)row_index * ROW_BYTES;
      const u64 row_id = load_u64(row + 8);
      const u64 expected_cell_start = (u64)row_index * column_count;
      if (load_u64(row) != table_id || row_id == 0 ||
          (row_index != 0 &&
           row_id <= load_u64(row_base + (u64)(row_index - 1) * ROW_BYTES + 8)) ||
          (u64)load_u32(row + 24) != expected_cell_start ||
          load_u32(row + 28) != column_count) {
        reject_generation_input(output, 4); return;
      }
      if (row_index + 1 == rows && load_u64(table + 56) < row_id) {
        reject_generation_input(output, 6); return;
      }
      const u32 current_bytes = 96u + 72u * column_count;
      u8* current_preimage = workspace + (u64)row_index * current_bytes;
      u32 at = 0;
      append_domain(current_preimage, &at, parallel_current_row_domain,
                    sizeof(parallel_current_row_domain) - 1);
      append_u64(current_preimage, &at, table_id);
      append_u64(current_preimage, &at, load_u64(row + 8));
      append_u64(current_preimage, &at, commit_sequence);
      append_u32(current_preimage, &at, load_u32(row + 16));
      append_u32(current_preimage, &at, load_u32(row + 20));
      append_u32(current_preimage, &at, column_count);
      const u32 cell_start = load_u32(row + 24);
      for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
        const u8* cell = cell_base + (u64)(cell_start + ordinal) * CELL_BYTES;
        append_u64(current_preimage, &at, (u64)load_u32(cell + 4));
        append_bytes(current_preimage, &at, slot(output, layout.shape_slot + cell_start + ordinal), 32);
        append_bytes(current_preimage, &at, slot(output, layout.typed_slot + cell_start + ordinal), 32);
      }
      gpu_db_sha256_bytes(current_preimage, at, slot(output, layout.current_slot + row_index));

      // S7 binds the original typed cells and image order used by canonical codec-5 WAL closure.
      // Its arena begins after every concurrent current-row preimage, so a wide or variable-width
      // value can never overlap an in-flight row of either generic digest grammar.
      const u32 s7_prefix_bytes = 67u + 25u * column_count;
      const u32 s7_value_start = load_u32(cell_base + (u64)cell_start * CELL_BYTES + 24);
      u8* s7_preimage = workspace + v3_current_row_preimage_bytes(rows, cells) +
          (u64)row_index * s7_prefix_bytes + s7_value_start;
      at = 0;
      append_domain(s7_preimage, &at, s7_final_row_domain, sizeof(s7_final_row_domain) - 1);
      append_u64(s7_preimage, &at, table_id);
      append_u64(s7_preimage, &at, load_u64(row + 8));
      append_u32(s7_preimage, &at, load_u32(input + 140));
      append_u32(s7_preimage, &at, row_index);
      append_u32(s7_preimage, &at, column_count);
      for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
        const u8* cell = cell_base + (u64)(cell_start + ordinal) * CELL_BYTES;
        const u32 value_start = load_u32(cell + 24);
        const u32 value_count = load_u32(cell + 28);
        append_u32(s7_preimage, &at, ordinal);
        append_u32(s7_preimage, &at, load_u32(cell + 4));
        append_i16(s7_preimage, &at, load_u16(cell + 8));
        append_bytes(s7_preimage, &at, cell + 12, 4);
        append_u32(s7_preimage, &at, load_u32(cell + 16));
        append_i16(s7_preimage, &at, load_u16(cell + 20));
        s7_preimage[at++] = cell[22];
        append_u32(s7_preimage, &at, value_count);
        if (value_count != 0) append_bytes(s7_preimage, &at, values + value_start, value_count);
      }
      gpu_db_sha256_bytes(s7_preimage, at, slot(output, layout.s7_final_row_slot + row_index));

      // The exact S7 transition record is fixed-width and binds the same device-produced final
      // row digest. Its typed statement digest is a named optional ABI field: live/replay codec-5
      // callers provide it, while generic generation callers cannot consume this output.
      u8 transition_preimage[256];
      at = 0;
      append_domain(transition_preimage, &at, s7_transition_domain,
                    sizeof(s7_transition_domain) - 1);
      append_u32(transition_preimage, &at, row_index);
      append_u32(transition_preimage, &at, 0);
      append_u64(transition_preimage, &at, row_id);
      transition_preimage[at++] = 1;
      transition_preimage[at++] = 0;
      transition_preimage[at++] = 0;
      transition_preimage[at++] = 0;
      append_u32(transition_preimage, &at, row_index);
      append_u32(transition_preimage, &at, 0);
      append_u32(transition_preimage, &at, row_index);
      append_u32(transition_preimage, &at, 0);
      append_u32(transition_preimage, &at, row_index);
      // The indexed route has exactly one physical maintenance effect per row. The
      // complete S7 effect/component bytes and their WAL transition digest remain with the
      // canonical writer; this device slot is intentionally not exposed for indexed rows.
      const bool indexed = load_u32(input + 124) != 0;
      append_u32(transition_preimage, &at, indexed ? row_index : 0);
      append_u32(transition_preimage, &at, indexed ? 1 : 0);
      append_u32(transition_preimage, &at, 0);
      #pragma unroll
      for (u32 word = 0; word < 3; ++word) append_u32(transition_preimage, &at, 0);
      append_bytes(transition_preimage, &at, input + 216, 32);
      append_bytes(transition_preimage, &at,
                   slot(output, layout.s7_final_row_slot + row_index), 32);
      #pragma unroll
      for (u32 word = 0; word < 16; ++word) append_u32(transition_preimage, &at, 0);
      gpu_db_sha256_bytes(transition_preimage, at,
                          slot(output, layout.s7_transition_slot + row_index));

      node* nodes_a = v3_nodes_a(workspace, rows, cells, value_bytes);
      nodes_a[row_index].representative_id = row_id;
      nodes_a[row_index].count = 1;
      u8 local[128];
      at = 0;
      append_domain(local, &at, parallel_row_leaf_domain, sizeof(parallel_row_leaf_domain) - 1);
      append_u16(local, &at, 1);
      append_u64(local, &at, table_id);
      append_u64(local, &at, nodes_a[row_index].representative_id);
      append_u64(local, &at, 1);
      append_bytes(local, &at, slot(output, layout.current_slot + row_index), 32);
      gpu_db_sha256_bytes(local, at, nodes_a[row_index].digest);
      copy_32(slot(output, layout.row_leaf_slot + row_index), nodes_a[row_index].digest);
    }
    if (index < column_count) {
      const u32 ordinal = (u32)index;
      u8 local[128];
      u32 at = 0;
      append_domain(local, &at, column_empty_domain, sizeof(column_empty_domain) - 1);
      append_u64(local, &at, table_id);
      append_bytes(local, &at, slot(output, layout.shape_slot + ordinal), 32);
      append_u64(local, &at, 0);
      gpu_db_sha256_bytes(local, at, slot(output, layout.column_slot + ordinal));
    }
  }
}

// The row-tree phase preserves the exact ordered reduction.  Up to 1,024 rows use every thread
// in this launch; larger statements retain the existing generic serial reduction until its own
// multi-grid reduction is introduced, rather than changing the commitment algorithm.
extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_reduce(
    const u8* input, u8* output) {
  if (blockIdx.x != 0 || load_u32(output) != 0) return;
  const u32 tid = threadIdx.x;
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u8* table = input + load_u64(input + 144);
  u8* workspace = (u8*)(u64)load_u64(input + 176);
  const u64 table_id = load_u64(table);
  const u64 commit_sequence = load_u64(input + 64);
  v3_layout layout;
  v3_output_layout(rows, cells, load_u32(input + 124), &layout);

  node* nodes_a = v3_nodes_a(workspace, rows, cells, value_bytes);
  node* nodes_b = nodes_a + rows;
  bool current_is_a = true;
  u32 active = rows;
  u32 cursor = 0;
  u32 total_nodes = 0;

  if (rows <= 1024) {
    for (u32 level = 0; active > 1; ++level) {
      const u32 next_count = (active + 1) / 2;
      node* current = current_is_a ? nodes_a : nodes_b;
      node* next = current_is_a ? nodes_b : nodes_a;
      for (u32 node_index = tid; node_index < next_count; node_index += blockDim.x) {
        const node* left = current + node_index * 2;
        const node* right = node_index * 2 + 1 < active ? left + 1 : 0;
        next[node_index].representative_id = left->representative_id;
        next[node_index].count = left->count + (right ? right->count : 0);
        u8 local[256];
        u32 at = 0;
        append_domain(local, &at, parallel_row_node_domain, sizeof(parallel_row_node_domain) - 1);
        append_u16(local, &at, 1);
        append_u64(local, &at, table_id);
        append_u64(local, &at, commit_sequence);
        append_u32(local, &at, level);
        append_u64(local, &at, left->count);
        append_u64(local, &at, right ? right->count : 0);
        append_bytes(local, &at, left->digest, 32);
        append_bytes(local, &at, right ? right->digest : slot(output, SLOT_ROW_EMPTY), 32);
        gpu_db_sha256_bytes(local, at, next[node_index].digest);
        copy_32(slot(output, layout.row_node_slot + cursor + node_index), next[node_index].digest);
      }
      __syncthreads();
      total_nodes += next_count;
      cursor += compact_level_capacity(rows, level + 1);
      active = next_count;
      current_is_a = !current_is_a;
      __syncthreads();
    }
  } else if (tid == 0) {
    u32 level = 0;
    while (active > 1) {
      const u32 next_count = (active + 1) / 2;
      node* current = current_is_a ? nodes_a : nodes_b;
      node* next = current_is_a ? nodes_b : nodes_a;
      for (u32 node_index = 0; node_index < next_count; ++node_index) {
        const node* left = current + node_index * 2;
        const node* right = node_index * 2 + 1 < active ? left + 1 : 0;
        next[node_index].representative_id = left->representative_id;
        next[node_index].count = left->count + (right ? right->count : 0);
        u8 local[256];
        u32 at = 0;
        append_domain(local, &at, parallel_row_node_domain, sizeof(parallel_row_node_domain) - 1);
        append_u16(local, &at, 1);
        append_u64(local, &at, table_id);
        append_u64(local, &at, commit_sequence);
        append_u32(local, &at, level);
        append_u64(local, &at, left->count);
        append_u64(local, &at, right ? right->count : 0);
        append_bytes(local, &at, left->digest, 32);
        append_bytes(local, &at, right ? right->digest : slot(output, SLOT_ROW_EMPTY), 32);
        gpu_db_sha256_bytes(local, at, next[node_index].digest);
        copy_32(slot(output, layout.row_node_slot + cursor + node_index), next[node_index].digest);
      }
      total_nodes += next_count;
      cursor += compact_level_capacity(rows, level + 1);
      active = next_count;
      current_is_a = !current_is_a;
      ++level;
    }
  }
  if (tid == 0) store_u32(output + 4, total_nodes);
}

// Finalization consumes only roots completed by earlier stream phases. It derives and checks the
// one table-map predecessor/successor path entirely on-device before it emits database roots.
//
// The unindexed production workload must not carry the indexed-only local frame.  Both
// instantiations produce the exact same v3 commitments for their admitted input; the launch
// selection below is a physical code-generation choice beneath the one DeviceInsertPlan, not a
// terminal, format, or root authority split.
template <bool HAS_INDEXES>
__device__ void v3_finalize(const u8* input, u8* output) {
  const u32 tid = threadIdx.x;
  const u32 rows = load_u32(input + 112);
  const u32 cells = load_u32(input + 116);
  const u32 value_bytes = load_u32(input + 120);
  const u32 indexes = load_u32(input + 124);
  if ((!HAS_INDEXES && indexes != 0) || (HAS_INDEXES && indexes == 0)) {
    if (tid == 0) store_u32(output, 20);
    return;
  }
  const bool create_empty = input[106] == TABLE_ACTION_CREATE_EMPTY;
  const bool create_with_row_set = input[106] == TABLE_ACTION_CREATE_WITH_ROW_SET;
  const bool create_index_then_row_set = input[106] == TABLE_ACTION_CREATE_INDEX_THEN_ROW_SET_INSERT;
  const bool enroll_index = input[106] == TABLE_ACTION_ENROLL_INDEX;
  const bool reset_row_set = input[106] == TABLE_ACTION_RESET_THEN_ROW_SET_INSERT;
  const bool zero_row_reset = reset_row_set && rows == 0;
  const u32 column_count = (create_empty || enroll_index || zero_row_reset) ? cells : cells / rows;
  const u8* table = input + load_u64(input + 144);
  u8* workspace = (u8*)(u64)load_u64(input + 176);
  const u64 table_id = load_u64(table);
  const u64 base_generation = load_u64(table + 8);
  const u64 commit_sequence = load_u64(input + 64);
  const u64 initial_rows = load_u64(table + 64);
  const u64 final_rows = load_u64(table + 72);
  v3_layout layout;
  v3_output_layout(rows, cells, load_u32(input + 124), &layout);
  // Each map depth has a distinct but independent fixed first SHA block. The device derives
  // these transient states from the same database id the serial finalizer formerly placed in
  // every full preimage. The scratch starts after all existing arenas (including indexed effect
  // scratch), so the barrier hands tid 0 only authentic, non-aliased prefix state.
  u32* map_node_prefix_states = v3_map_node_prefix_states(input, workspace);
  if (tid < RADIX_DEPTH)
    v3_map_node_prefix_state(input, tid,
                             map_node_prefix_states + tid * MAP_NODE_PREFIX_STATE_WORDS);
  __syncthreads();
  if (tid != 0) return;
  node* nodes_a = v3_nodes_a(workspace, rows, cells, value_bytes);
  node* nodes_b = nodes_a + rows;
  bool current_is_a = true;
  u32 active = rows;
  while (active > 1) {
    active = (active + 1) / 2;
    current_is_a = !current_is_a;
  }
  node* final_nodes = current_is_a ? nodes_a : nodes_b;
  u8 batch_root[32];
  if (create_empty || enroll_index || zero_row_reset)
    copy_32(batch_root, slot(output, SLOT_ROW_EMPTY));
  else copy_32(batch_root, final_nodes[0].digest);
  u8* local = workspace;
  u32 at = 0;
  append_domain(local, &at, column_manifest_domain, sizeof(column_manifest_domain) - 1);
  append_u64(local, &at, table_id);
  append_u32(local, &at, column_count);
  for (u32 ordinal = 0; ordinal < column_count; ++ordinal) {
    append_bytes(local, &at, slot(output, layout.shape_slot + ordinal), 32);
    append_bytes(local, &at, slot(output, layout.column_slot + ordinal), 32);
  }
  u8 column_manifest_root[32];
  gpu_db_sha256_bytes(local, at, column_manifest_root);

  // Every maintained index root is derived from its descriptor/key/effect range after the typed
  // cell phase completes. The host supplies only the ordered predecessor descriptors; it never
  // supplies an index entry, map, or successor root.
  if (HAS_INDEXES) for (u32 index_ordinal = 0; index_ordinal < indexes; ++index_ordinal) {
    const u8* index = input + load_u64(input + 184) + (u64)index_ordinal * INDEX_BYTES;
    const u8* key_base = input + load_u64(input + 192);
    const u8* effect_base = input + load_u64(input + 200);
    const u8* component_base = input + load_u64(input + 208);
    u8 index_shape[32], generation_shape[32];
    v3_one_index_shapes(index, key_base, table_id, layout.shape_slot, output,
                        index_shape, generation_shape);
    u8* empty_roots = workspace;
    v3_index_empty_roots(table_id, index, index_shape, empty_roots);
    if (create_empty || enroll_index) {
      #pragma unroll
      for (u32 byte = 0; byte < 32; ++byte)
        slot(output, layout.index_initial_roots_slot + index_ordinal)[byte] = 0;
      v3_index_root(table_id, index, commit_sequence, index_shape, 0, empty_roots,
                    slot(output, layout.index_final_roots_slot + index_ordinal));
    } else if (zero_row_reset) {
      copy_32(slot(output, layout.index_initial_roots_slot + index_ordinal), index + 32);
      v3_index_root(table_id, index, commit_sequence, index_shape, 0, empty_roots,
                    slot(output, layout.index_final_roots_slot + index_ordinal));
    } else {
      const bool absent_predecessor = create_with_row_set ||
          (create_index_then_row_set && load_u64(index + 24) == 0 && zero_bytes(index + 32, 32)) ||
          (reset_row_set && load_u64(index + 24) == 0 && zero_bytes(index + 32, 32));
      copy_32(slot(output, layout.index_initial_roots_slot + index_ordinal), index + 32);
      u8 expected_base_root[32];
      bool empty_predecessor = false;
      if (!absent_predecessor) {
        v3_index_root(table_id, index, load_u64(index + 24), index_shape, 0, empty_roots,
                      expected_base_root);
        empty_predecessor = equal_32(expected_base_root, index + 32);
      }
      // Every supported nonunique INSERT contributes exactly one index entry per logical row.
      // This count/root cross-check prevents an empty enrollment root from being relabelled as a
      // nonempty predecessor (or vice versa) before the immutable successor is derived.
      if (!reset_row_set && !create_with_row_set && !create_index_then_row_set &&
          empty_predecessor != (initial_rows == 0)) {
        store_u32(output, 18); return;
      }
      u8 content_root[32]; u64 entry_count = 0;
      const u8* row_base = input + load_u64(input + 152);
      const u8* cell_base = input + load_u64(input + 160);
      const u8* values = input + load_u64(input + 168);
      if (!v3_index_content_root(row_base, cell_base, values, key_base, effect_base,
                                 component_base, index, table_id, commit_sequence,
                                 layout.typed_slot, output, index_shape, empty_roots, nodes_a,
                                 nodes_b, rows, content_root)) { store_u32(output, 19); return; }
      entry_count = rows;
      if (create_index_then_row_set && absent_predecessor) {
        v3_created_index_successor_root(table_id, index, commit_sequence, index_shape,
                                        table + 16, initial_rows, final_rows, entry_count,
                                        content_root,
                                        slot(output, layout.index_final_roots_slot + index_ordinal));
      } else if (empty_predecessor || reset_row_set || create_with_row_set) {
        v3_index_root(table_id, index, commit_sequence, index_shape, entry_count, content_root,
                      slot(output, layout.index_final_roots_slot + index_ordinal));
      } else {
        v3_index_successor_root(table_id, index, commit_sequence, index_shape, initial_rows,
                                final_rows, entry_count, content_root,
                                slot(output, layout.index_final_roots_slot + index_ordinal));
      }
    }
  }
  // `CreateEmpty` has no table predecessor, so its table-root before value remains the explicit
  // absence sentinel.  `RowSetInsert` must bind the exact pinned predecessor table root.
  if (create_empty || create_with_row_set) {
    #pragma unroll
    for (u32 byte = 0; byte < 32; ++byte) slot(output, SLOT_INITIAL_TABLE_ROOT)[byte] = 0;
    at = 0;
    if (create_empty) {
      // `table-root/v2` is the genesis grammar: base generation, zero logical rows, and the
      // canonical empty-row root.  It deliberately has no commit-generation or final-row-count
      // fields; adding those zero values changes the durable table identity and therefore the
      // table-map/database roots.
      append_domain(local, &at, parallel_table_root_domain,
                    sizeof(parallel_table_root_domain) - 1);
      append_u16(local, &at, 1);
      append_u64(local, &at, table_id);
      append_u64(local, &at, base_generation);
      append_u64(local, &at, 0);
      append_bytes(local, &at, batch_root, 32);
      append_u32(local, &at, column_count);
      append_bytes(local, &at, column_manifest_root, 32);
    } else {
      append_domain(local, &at, table_successor_domain,
                    sizeof(table_successor_domain) - 1);
      append_u16(local, &at, 1);
      append_u64(local, &at, table_id);
      append_u64(local, &at, base_generation);
      append_u64(local, &at, commit_sequence);
      append_u64(local, &at, 0);
      append_u64(local, &at, final_rows);
      append_bytes(local, &at, batch_root, 32);
      append_u32(local, &at, column_count);
      append_bytes(local, &at, column_manifest_root, 32);
    }
    if (HAS_INDEXES) {
      append_u32(local, &at, indexes);
      for (u32 index_ordinal = 0; index_ordinal < indexes; ++index_ordinal) {
        const u8* index = input + load_u64(input + 184) + (u64)index_ordinal * INDEX_BYTES;
        append_u64(local, &at, load_u64(index));
        append_u64(local, &at, commit_sequence);
        append_bytes(local, &at, slot(output, layout.index_final_roots_slot + index_ordinal), 32);
      }
    }
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_TABLE_ROOT));
  } else {
    copy_32(slot(output, SLOT_INITIAL_TABLE_ROOT), table + 16);
    at = 0;
    if (reset_row_set)
      append_domain(local, &at, table_reset_successor_domain,
                    sizeof(table_reset_successor_domain) - 1);
    else
      append_domain(local, &at, table_successor_domain, sizeof(table_successor_domain) - 1);
    append_u16(local, &at, 1);
    append_u64(local, &at, table_id);
    append_u64(local, &at, base_generation);
    append_u64(local, &at, commit_sequence);
    append_u64(local, &at, initial_rows);
    append_u64(local, &at, final_rows);
    append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
    append_bytes(local, &at, batch_root, 32);
    append_u32(local, &at, column_count);
    append_bytes(local, &at, column_manifest_root, 32);
    if (HAS_INDEXES) {
      append_u32(local, &at, indexes);
      for (u32 index_ordinal = 0; index_ordinal < indexes; ++index_ordinal) {
        const u8* index = input + load_u64(input + 184) + (u64)index_ordinal * INDEX_BYTES;
        append_u64(local, &at, load_u64(index));
        append_u64(local, &at, commit_sequence);
        append_bytes(local, &at, slot(output, layout.index_final_roots_slot + index_ordinal), 32);
      }
    }
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_TABLE_ROOT));
  }

  // A retained predecessor contains only immutable roots that the prior GPU publication already
  // authenticated. Copying that witness saves its redundant rederivation; the device still owns
  // the final leaf/path and rebinds the carried initial map root to the pinned database root.
  const u8 predecessor = input[107];
  const bool retained_predecessor = predecessor == TABLE_MAP_PREDECESSOR_PINNED_RETAINED;
  const u8* table_map_siblings = table + TABLE_MAP_SIBLINGS_OFFSET;
  const u8* retained_empty_roots = table + TABLE_MAP_RETAINED_EMPTY_ROOTS_OFFSET;
  const u8* retained_initial_leaf = table + TABLE_MAP_RETAINED_INITIAL_LEAF_OFFSET;
  const u8* retained_initial_path = table + TABLE_MAP_RETAINED_INITIAL_PATH_OFFSET;
  if (retained_predecessor) {
    for (u32 depth = 0; depth < EMPTY_ROOTS; ++depth)
      copy_32(slot(output, layout.table_empty_slot + depth),
              retained_empty_roots + (u64)depth * 32ull);
    copy_32(slot(output, layout.initial_table_leaf_slot), retained_initial_leaf);
    for (u32 depth = 0; depth < RADIX_DEPTH; ++depth)
      copy_32(slot(output, layout.initial_table_path_slot + depth),
              retained_initial_path + (u64)depth * 32ull);
  } else {
    // A first CREATE does not get to name an initial root: it derives the empty database
    // on-device. The legacy pinned schedule preserves the same derivation for compatibility.
    at = 0;
    append_domain(local, &at, map_empty_leaf_domain, sizeof(map_empty_leaf_domain) - 1);
    append_u16(local, &at, 1); append_bytes(local, &at, input, 16);
    gpu_db_sha256_bytes(local, at, slot(output, layout.table_empty_slot + 64));
    for (int depth = 63; depth >= 0; --depth) {
      at = 0;
      append_domain(local, &at, map_empty_node_domain, sizeof(map_empty_node_domain) - 1);
      append_u16(local, &at, 1); append_bytes(local, &at, input, 16); local[at++] = (u8)depth;
      append_bytes(local, &at, slot(output, layout.table_empty_slot + depth + 1), 32);
      append_bytes(local, &at, slot(output, layout.table_empty_slot + depth + 1), 32);
      gpu_db_sha256_bytes(local, at, slot(output, layout.table_empty_slot + depth));
    }
    if (create_empty || create_with_row_set) {
      copy_32(slot(output, layout.initial_table_leaf_slot),
              slot(output, layout.table_empty_slot + 64));
    } else {
      at = 0;
      append_domain(local, &at, map_leaf_domain, sizeof(map_leaf_domain) - 1);
      append_u16(local, &at, 1); append_bytes(local, &at, input, 16); append_u64(local, &at, table_id);
      append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
      gpu_db_sha256_bytes(local, at, slot(output, layout.initial_table_leaf_slot));
    }
  }
  at = 0;
  append_domain(local, &at, map_leaf_domain, sizeof(map_leaf_domain) - 1);
  append_u16(local, &at, 1); append_bytes(local, &at, input, 16); append_u64(local, &at, table_id);
  append_bytes(local, &at, slot(output, SLOT_FINAL_TABLE_ROOT), 32);
  gpu_db_sha256_bytes(local, at, slot(output, layout.final_table_leaf_slot));

  // Reconstruct the final root-to-leaf path from the same 64 supplied siblings. The ordinary
  // pinned schedule also derives the initial path; the retained schedule's initial path is
  // subsequently checked against the immutable publication owner before the new path is linked.
  for (int depth = 63; depth >= 0; --depth) {
    const u8* final_child = depth == 63 ? slot(output, layout.final_table_leaf_slot)
                                        : slot(output, layout.final_table_path_slot + depth + 1);
    const u8* sibling = table_map_siblings + (u64)depth * 32ull;
    if (predecessor == TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE) {
      if (!zero_bytes(sibling, 32)) { store_u32(output, 9); return; }
      sibling = slot(output, layout.table_empty_slot + depth + 1);
    }
    if (!retained_predecessor) {
      const u8* initial_child = depth == 63 ? slot(output, layout.initial_table_leaf_slot)
                                            : slot(output, layout.initial_table_path_slot + depth + 1);
      // An absent target may traverse an entirely empty subtree. Its predecessor root is the
      // canonical `map-empty-node`, not a synthetic `map-node` with two empty children. Once
      // either child is nonempty, the persistent-map grammar requires the exact map-node below.
      if (equal_32(initial_child, slot(output, layout.table_empty_slot + depth + 1)) &&
          equal_32(sibling, slot(output, layout.table_empty_slot + depth + 1))) {
        copy_32(slot(output, layout.initial_table_path_slot + depth),
                slot(output, layout.table_empty_slot + depth));
      } else {
        if (((table_id >> (63u - (u32)depth)) & 1ull) == 0) {
          v3_map_node_from_prefix_state(
              map_node_prefix_states + (u32)depth * MAP_NODE_PREFIX_STATE_WORDS,
              initial_child, sibling, slot(output, layout.initial_table_path_slot + depth));
        } else {
          v3_map_node_from_prefix_state(
              map_node_prefix_states + (u32)depth * MAP_NODE_PREFIX_STATE_WORDS,
              sibling, initial_child, slot(output, layout.initial_table_path_slot + depth));
        }
      }
    }
    if (((table_id >> (63u - (u32)depth)) & 1ull) == 0) {
      v3_map_node_from_prefix_state(
          map_node_prefix_states + (u32)depth * MAP_NODE_PREFIX_STATE_WORDS,
          final_child, sibling, slot(output, layout.final_table_path_slot + depth));
    } else {
      v3_map_node_from_prefix_state(
          map_node_prefix_states + (u32)depth * MAP_NODE_PREFIX_STATE_WORDS,
          sibling, final_child, slot(output, layout.final_table_path_slot + depth));
    }
  }
  copy_32(slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), slot(output, layout.initial_table_path_slot));
  copy_32(slot(output, SLOT_FINAL_TABLE_MAP_ROOT), slot(output, layout.final_table_path_slot));
  if (predecessor == TABLE_MAP_PREDECESSOR_UNINITIALIZED_EMPTY_DATABASE &&
      !equal_32(slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), slot(output, layout.table_empty_slot))) {
    store_u32(output, 10); return;
  }
  at = 0;
  append_domain(local, &at, database_root_domain, sizeof(database_root_domain) - 1);
  append_u16(local, &at, 1); append_bytes(local, &at, input, 16);
  append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_MAP_ROOT), 32);
  gpu_db_sha256_bytes(local, at, slot(output, SLOT_INITIAL_DATABASE_ROOT));
  if (table_map_predecessor_is_pinned(predecessor) &&
      (!equal_32(slot(output, SLOT_INITIAL_DATABASE_ROOT), input + 72) ||
       zero_bytes(input + 72, 32))) {
    store_u32(output, 11); return;
  }
  at = 0;
  append_domain(local, &at, database_root_domain, sizeof(database_root_domain) - 1);
  append_u16(local, &at, 1); append_bytes(local, &at, input, 16);
  append_bytes(local, &at, slot(output, SLOT_FINAL_TABLE_MAP_ROOT), 32);
  gpu_db_sha256_bytes(local, at, slot(output, SLOT_FINAL_DATABASE_ROOT));

  if (!HAS_INDEXES) {
    // Preserve the accepted zero-index generation-input/v4 bytes exactly.  Widening the input
    // descriptor must not silently change the established unindexed durable commitment.
    at = 0;
    append_domain(local, &at, parallel_generation_input_domain,
                  sizeof(parallel_generation_input_domain) - 1);
    append_bytes(local, &at, input, 16);
    append_u64(local, &at, load_u64(input + 16));
    append_bytes(local, &at, input + 24, 32);
    append_u64(local, &at, load_u64(input + 56));
    append_u64(local, &at, commit_sequence);
    append_bytes(local, &at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32);
    local[at++] = input[106];
    append_u64(local, &at, table_id);
    append_u64(local, &at, base_generation);
    append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
    append_u64(local, &at, load_u64(table + 48));
    append_u64(local, &at, load_u64(table + 56));
    append_u64(local, &at, initial_rows);
    append_u64(local, &at, final_rows);
    append_u32(local, &at, rows);
    append_bytes(local, &at, batch_root, 32);
    append_bytes(local, &at, table + 80, 32);
    append_bytes(local, &at, table + 112, 32);
    append_u32(local, &at, 0);
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_GENERATION_INPUT));
  } else {
    const u8* index = input + load_u64(input + 184);
    const u8* key_base = input + load_u64(input + 192);
    const u8* effect_base = input + load_u64(input + 200);
    const u8* component_base = input + load_u64(input + 208);
    const u8* row_base = input + load_u64(input + 152);
    const u8* cell_base = input + load_u64(input + 160);
    const u8* values = input + load_u64(input + 168);
    const u32 key_count = load_u32(index + 68);
    const u64 exact_input_bytes = load_u64(input + 208) +
        (u64)load_u32(input + 136) * INDEX_EFFECT_COMPONENT_BYTES;
    u8* effect_scratch = workspace + exact_input_bytes + 512ull;
    u8 index_shape[32], generation_shape[32];
    v3_one_index_shapes(index, key_base, table_id, layout.shape_slot, output,
                        index_shape, generation_shape);
    at = 0;
    append_domain(local, &at, generation_input_domain, sizeof(generation_input_domain) - 1);
    append_bytes(local, &at, input, 16);
    append_u64(local, &at, load_u64(input + 16));
    append_bytes(local, &at, input + 24, 32);
    append_u64(local, &at, load_u64(input + 56));
    append_u64(local, &at, commit_sequence);
    append_bytes(local, &at, slot(output, SLOT_INITIAL_DATABASE_ROOT), 32);
    append_u32(local, &at, 1);
    append_u64(local, &at, table_id);
    append_u64(local, &at, base_generation);
    append_bytes(local, &at, slot(output, SLOT_INITIAL_TABLE_ROOT), 32);
    append_u64(local, &at, load_u64(table + 48));
    append_u64(local, &at, load_u64(table + 56));
    append_u64(local, &at, initial_rows);
    append_u64(local, &at, final_rows);
    append_u32(local, &at, rows);
    // The descriptor's exact tail end is below the two input-sized scratch regions reserved
    // by the host. Keep variable-width row preimages beyond it so they cannot overlap the
    // top-level generation-input preimage accumulating at workspace base.
    u64 row_cursor = load_u64(input + 208) +
                     (u64)load_u32(input + 136) * INDEX_EFFECT_COMPONENT_BYTES;
    for (u32 row_ordinal = 0; row_ordinal < rows; ++row_ordinal) {
      const u8* row = row_base + (u64)row_ordinal * ROW_BYTES;
      const u32 cell_start = load_u32(row + 24);
      const u32 row_cells = load_u32(row + 28);
      u8* row_preimage = workspace + row_cursor;
      u32 row_at = 0;
      append_domain(row_preimage, &row_at, generation_row_domain, sizeof(generation_row_domain) - 1);
      append_u64(row_preimage, &row_at, table_id); append_u64(row_preimage, &row_at, load_u64(row + 8));
      append_u32(row_preimage, &row_at, load_u32(row + 16)); append_u32(row_preimage, &row_at, load_u32(row + 20));
      append_u32(row_preimage, &row_at, row_cells);
      for (u32 ordinal = 0; ordinal < row_cells; ++ordinal) {
        const u8* cell = cell_base + (u64)(cell_start + ordinal) * CELL_BYTES;
        const u32 value_start = load_u32(cell + 24); const u32 value_count = load_u32(cell + 28);
        append_u32(row_preimage, &row_at, load_u32(cell)); append_u32(row_preimage, &row_at, load_u32(cell + 4));
        append_i16(row_preimage, &row_at, load_u16(cell + 8)); append_bytes(row_preimage, &row_at, cell + 12, 4);
        append_u32(row_preimage, &row_at, load_u32(cell + 16)); append_i16(row_preimage, &row_at, load_u16(cell + 20));
        row_preimage[row_at++] = cell[22]; append_u32(row_preimage, &row_at, value_count);
        if (value_count != 0) append_bytes(row_preimage, &row_at, values + value_start, value_count);
      }
      gpu_db_sha256_bytes(row_preimage, row_at, slot(output, layout.current_slot + row_ordinal));
      append_bytes(local, &at, slot(output, layout.current_slot + row_ordinal), 32);
      row_cursor += row_at;
    }
    append_bytes(local, &at, table + 80, 32); append_bytes(local, &at, table + 112, 32);
    append_u32(local, &at, 1); append_bytes(local, &at, generation_shape, 32);
    append_u32(local, &at, rows);
    for (u32 row_ordinal = 0; row_ordinal < rows; ++row_ordinal) {
      const u8* row = row_base + (u64)row_ordinal * ROW_BYTES;
      const u8* effect = effect_base + (u64)row_ordinal * INDEX_EFFECT_BYTES;
      const u32 cell_start = load_u32(row + 24); const u32 component_start = load_u32(effect + 28);
      u8* effect_preimage = effect_scratch;
      u32 effect_at = 0;
      append_domain(effect_preimage, &effect_at, generation_index_effect_domain,
                    sizeof(generation_index_effect_domain) - 1);
      append_u64(effect_preimage, &effect_at, table_id); append_u64(effect_preimage, &effect_at, load_u64(row + 8));
      append_u32(effect_preimage, &effect_at, load_u32(effect + 24)); append_bytes(effect_preimage, &effect_at, generation_shape, 32);
      append_u32(effect_preimage, &effect_at, key_count);
      for (u32 ordinal = 0; ordinal < key_count; ++ordinal) {
        const u8* component = component_base + (u64)(component_start + ordinal) * INDEX_EFFECT_COMPONENT_BYTES;
        const u8* cell = cell_base + (u64)(cell_start + load_u32(component)) * CELL_BYTES;
        const u32 value_start = load_u32(cell + 24); const u32 value_count = load_u32(cell + 28);
        u8* key_preimage = effect_scratch + 128ull + (u64)key_count * 32ull;
        u32 key_at = 0;
        append_domain(key_preimage, &key_at, s7_typed_key_value_domain,
                      sizeof(s7_typed_key_value_domain) - 1);
        append_bytes(key_preimage, &key_at, cell + 12, 4); append_u32(key_preimage, &key_at, load_u32(cell + 16));
        append_i16(key_preimage, &key_at, load_u16(cell + 20)); key_preimage[key_at++] = cell[22];
        append_u32(key_preimage, &key_at, value_count);
        if (value_count != 0) append_bytes(key_preimage, &key_at, values + value_start, value_count);
        u8 key_digest[32]; gpu_db_sha256_bytes(key_preimage, key_at, key_digest);
        append_bytes(effect_preimage, &effect_at, key_digest, 32);
      }
      u8 effect_digest[32]; gpu_db_sha256_bytes(effect_preimage, effect_at, effect_digest);
      append_bytes(local, &at, effect_digest, 32);
    }
    gpu_db_sha256_bytes(local, at, slot(output, SLOT_GENERATION_INPUT));
  }
  store_u32(output, 0);
}

extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_finalize(
    const u8* input, u8* output) {
  if (blockIdx.x != 0 || load_u32(output) != 0) return;
  v3_finalize<true>(input, output);
}

extern "C" __global__ void gpu_db_runtime_typed_insert_generation_v3_finalize_unindexed(
    const u8* input, u8* output) {
  if (blockIdx.x != 0 || load_u32(output) != 0) return;
  v3_finalize<false>(input, output);
}
