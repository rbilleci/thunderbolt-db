// Canonical source for runtime_generation_rebuild/kernel.ptx. Regenerate from repository root:
// nvcc --ptx --gpu-architecture=compute_60 --std=c++14 -O3 -I crates/execution/src \
//   -o crates/execution/src/runtime_generation_rebuild/kernel.ptx \
//   crates/execution/src/runtime_generation_rebuild/kernel.cu && \
// perl -0pi -e 's/\n\n\z/\n/' crates/execution/src/runtime_generation_rebuild/kernel.ptx
//
// The operator intentionally uses one GPU thread. It is a bounded correctness-first rebuild
// program: all table bytes, ordering checks, visibility checks, typed values, and logical roots
// remain on the device. The host only stages descriptors and receives opaque digests.

#include "../sha256_device.cuh"

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

enum {
  GPU_DB_REBUILD_HEADER_BYTES = 80,
  GPU_DB_REBUILD_SHARD_BYTES = 56,
  GPU_DB_REBUILD_OUTPUT_STATUS_BYTES = 4,
  GPU_DB_REBUILD_PROOF_SLOTS = 135,
  GPU_DB_REBUILD_SLOT_COLUMN_SHAPE = 0,
  GPU_DB_REBUILD_SLOT_TYPED_VECTOR = 1,
  GPU_DB_REBUILD_SLOT_ROW_LEAVES = 2,
  GPU_DB_REBUILD_SLOT_ROW_EMPTY = 3,
  GPU_DB_REBUILD_SLOT_TABLE_ROOT = 68,
  GPU_DB_REBUILD_SLOT_DATABASE_EMPTY = 69,
  GPU_DB_REBUILD_SLOT_DATABASE_ROOT = 134,
};

__device__ __constant__ u8 gpu_db_column_shape_domain[] =
    "gpu-db/runtime-generation/column-shape/v1";
__device__ __constant__ u8 gpu_db_typed_value_domain[] =
    "gpu-db/runtime-generation/typed-value/v1";
__device__ __constant__ u8 gpu_db_current_row_domain[] =
    "gpu-db/runtime-generation/current-row/v1";
__device__ __constant__ u8 gpu_db_row_empty_leaf_domain[] =
    "gpu-db/runtime-generation/row-empty-leaf/v1";
__device__ __constant__ u8 gpu_db_row_empty_node_domain[] =
    "gpu-db/runtime-generation/row-empty-node/v1";
__device__ __constant__ u8 gpu_db_row_leaf_domain[] =
    "gpu-db/runtime-generation/row-leaf/v1";
__device__ __constant__ u8 gpu_db_row_node_domain[] =
    "gpu-db/runtime-generation/row-node/v1";
__device__ __constant__ u8 gpu_db_table_root_domain[] =
    "gpu-db/runtime-generation/table-root/v1";
__device__ __constant__ u8 gpu_db_map_empty_leaf_domain[] =
    "gpu-db/runtime-generation/map-empty-leaf/v1";
__device__ __constant__ u8 gpu_db_map_empty_node_domain[] =
    "gpu-db/runtime-generation/map-empty-node/v1";
__device__ __constant__ u8 gpu_db_map_leaf_domain[] =
    "gpu-db/runtime-generation/map-leaf/v1";
__device__ __constant__ u8 gpu_db_map_node_domain[] =
    "gpu-db/runtime-generation/map-node/v1";
__device__ __constant__ u8 gpu_db_database_root_domain[] =
    "gpu-db/runtime-generation/database-root/v1";
// Opaque proof-vector domains are transport evidence only. They are never installed as logical
// roots and do not replace the typed-value/current-row grammars above.
__device__ __constant__ u8 gpu_db_typed_vector_proof_domain[] =
    "gpu-db/runtime-generation/rebuild-proof/typed-vector/v1";
__device__ __constant__ u8 gpu_db_row_vector_proof_domain[] =
    "gpu-db/runtime-generation/rebuild-proof/current-rows/v1";

static_assert(sizeof(gpu_db_typed_vector_proof_domain) - 1 == 55,
              "host/device typed-vector frame length");
static_assert(sizeof(gpu_db_row_vector_proof_domain) - 1 == 55,
              "host/device row-vector frame length");

__device__ u16 gpu_db_load_u16(const u8* p) {
  return (u16)p[0] | ((u16)p[1] << 8);
}

__device__ u32 gpu_db_load_u32(const u8* p) {
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ u64 gpu_db_load_u64(const u8* p) {
  u64 value = 0;
  #pragma unroll
  for (u32 byte = 0; byte < 8; ++byte) value |= (u64)p[byte] << (byte * 8);
  return value;
}

__device__ void gpu_db_store_u32(u8* p, u32 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 4; ++byte) p[byte] = (u8)(value >> (byte * 8));
}

__device__ void gpu_db_append_u16(u8* p, u32* at, u16 value) {
  p[(*at)++] = (u8)value;
  p[(*at)++] = (u8)(value >> 8);
}

__device__ void gpu_db_append_u32(u8* p, u32* at, u32 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 4; ++byte) p[(*at)++] = (u8)(value >> (byte * 8));
}

__device__ void gpu_db_append_u64(u8* p, u32* at, u64 value) {
  #pragma unroll
  for (u32 byte = 0; byte < 8; ++byte) p[(*at)++] = (u8)(value >> (byte * 8));
}

__device__ void gpu_db_append_bytes(u8* p, u32* at, const u8* source, u32 count) {
  #pragma unroll 1
  for (u32 byte = 0; byte < count; ++byte) p[(*at)++] = source[byte];
}

__device__ void gpu_db_append_domain(u8* p, u32* at, const u8* domain, u32 length) {
  gpu_db_append_u64(p, at, (u64)length);
  gpu_db_append_bytes(p, at, domain, length);
}

__device__ u8* gpu_db_slot(u8* output, u32 slot) {
  return output + GPU_DB_REBUILD_OUTPUT_STATUS_BYTES + (u64)slot * 32ull;
}

__device__ const u8* gpu_db_shard(const u8* descriptor, u32 shard) {
  return descriptor + GPU_DB_REBUILD_HEADER_BYTES + (u64)shard * GPU_DB_REBUILD_SHARD_BYTES;
}

__device__ bool gpu_db_find_row(const u8* descriptor, u64 row, const u8** shard, u64* local) {
  const u32 shard_count = gpu_db_load_u32(descriptor + 68);
  for (u32 index = 0; index < shard_count; ++index) {
    const u8* candidate = gpu_db_shard(descriptor, index);
    const u64 start = gpu_db_load_u64(candidate);
    const u64 count = gpu_db_load_u64(candidate + 8);
    if (row >= start && row - start < count) {
      *shard = candidate;
      *local = row - start;
      return true;
    }
  }
  return false;
}

__device__ u64 gpu_db_row_id(const u8* descriptor, u64 row) {
  const u8* shard = 0;
  u64 local = 0;
  if (!gpu_db_find_row(descriptor, row, &shard, &local)) return 0;
  const u8* ids = (const u8*)(u64)gpu_db_load_u64(shard + 16);
  return gpu_db_load_u64(ids + local * 8ull);
}

__device__ bool gpu_db_vector_frame_bytes(
    u32 domain_bytes, u64 count, u32* value_bytes, u32* frame_bytes) {
  const u64 max_u32 = 0xffffffffull;
  const u64 prefix = 16ull + (u64)domain_bytes;
  if (prefix > max_u32 || count > (max_u32 - prefix) / 32ull) return false;
  const u64 payload = count * 32ull;
  *value_bytes = (u32)payload;
  *frame_bytes = (u32)(prefix + payload);
  return true;
}

__device__ bool gpu_db_hash_vector(
    const u8* domain, u32 domain_bytes, u64 count, const u8* values, u32 value_bytes,
    u32 frame_bytes, u8* scratch, u8* out) {
  u32 at = 0;
  gpu_db_append_domain(scratch, &at, domain, domain_bytes);
  gpu_db_append_u64(scratch, &at, count);
  gpu_db_append_bytes(scratch, &at, values, value_bytes);
  if (at != frame_bytes) return false;
  gpu_db_sha256_bytes(scratch, at, out);
  return true;
}

struct gpu_db_digest_count {
  u8 digest[32];
  u64 count;
};

struct gpu_db_row_node {
  u64 representative_id;
  u64 count;
  u8 digest[32];
};

__device__ gpu_db_digest_count gpu_db_build_row_map_iterative(
    const u8* descriptor, const u8* leaves, const u8* empties, u64 rows, u64 table_id,
    gpu_db_row_node* current, gpu_db_row_node* next) {
  for (u64 row = 0; row < rows; ++row) {
    current[row].representative_id = gpu_db_row_id(descriptor, row);
    current[row].count = 1;
    u8 preimage[160];
    u32 at = 0;
    gpu_db_append_domain(preimage, &at, gpu_db_row_leaf_domain, sizeof(gpu_db_row_leaf_domain) - 1);
    gpu_db_append_u16(preimage, &at, 1);
    gpu_db_append_u64(preimage, &at, table_id);
    gpu_db_append_u64(preimage, &at, current[row].representative_id);
    gpu_db_append_u64(preimage, &at, 1);
    gpu_db_append_bytes(preimage, &at, leaves + row * 32ull, 32);
    gpu_db_sha256_bytes(preimage, at, current[row].digest);
  }
  u64 current_count = rows;
  for (int depth = 63; depth >= 0; --depth) {
    u64 read = 0;
    u64 written = 0;
    while (read < current_count) {
      const u64 id = current[read].representative_id;
      const u64 parent = depth == 0 ? 0 : id >> (64u - (u32)depth);
      const bool first_right = ((id >> (63u - (u32)depth)) & 1ull) != 0;
      const gpu_db_row_node* left = first_right ? 0 : &current[read];
      const gpu_db_row_node* right = first_right ? &current[read] : 0;
      ++read;
      if (read < current_count) {
        const u64 next_id = current[read].representative_id;
        const u64 next_parent = depth == 0 ? 0 : next_id >> (64u - (u32)depth);
        if (next_parent == parent) {
          const bool next_right = ((next_id >> (63u - (u32)depth)) & 1ull) != 0;
          if (next_right == first_right) {
            gpu_db_digest_count corrupt;
            corrupt.count = 0;
            #pragma unroll
            for (u32 byte = 0; byte < 32; ++byte) corrupt.digest[byte] = 0;
            return corrupt;
          }
          if (next_right) right = &current[read]; else left = &current[read];
          ++read;
        }
      }
      const u8* empty = empties + ((u64)depth + 1ull) * 32ull;
      u8 preimage[192];
      u32 at = 0;
      gpu_db_append_domain(preimage, &at, gpu_db_row_node_domain, sizeof(gpu_db_row_node_domain) - 1);
      gpu_db_append_u16(preimage, &at, 1);
      gpu_db_append_u64(preimage, &at, table_id);
      preimage[at++] = (u8)depth;
      next[written].representative_id = id;
      next[written].count = (left ? left->count : 0) + (right ? right->count : 0);
      gpu_db_append_u64(preimage, &at, next[written].count);
      gpu_db_append_bytes(preimage, &at, left ? left->digest : empty, 32);
      gpu_db_append_bytes(preimage, &at, right ? right->digest : empty, 32);
      gpu_db_sha256_bytes(preimage, at, next[written].digest);
      ++written;
    }
    gpu_db_row_node* swap = current; current = next; next = swap;
    current_count = written;
  }
  gpu_db_digest_count result;
  result.count = current_count == 1 ? current[0].count : 0;
  #pragma unroll
  for (u32 byte = 0; byte < 32; ++byte) result.digest[byte] = current[0].digest[byte];
  return result;
}

extern "C" __global__ void gpu_db_runtime_generation_v1_single_table_int4_rebuild(
    const u8* descriptor, u8* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const u64 table_id = gpu_db_load_u64(descriptor + 16);
  const u64 generation = gpu_db_load_u64(descriptor + 24);
  const u64 rows = gpu_db_load_u64(descriptor + 32);
  const u64 cut = gpu_db_load_u64(descriptor + 40);
  const u64 column_id = gpu_db_load_u64(descriptor + 48);
  const u16 attnum = gpu_db_load_u16(descriptor + 56);
  const u32 declared_oid = gpu_db_load_u32(descriptor + 60);
  const u16 signed_size = gpu_db_load_u16(descriptor + 64);
  const u16 root_format = gpu_db_load_u16(descriptor + 66);
  const u32 shard_count = gpu_db_load_u32(descriptor + 68);
  u8* workspace = (u8*)(u64)gpu_db_load_u64(descriptor + 72);
  if (table_id == 0 || generation == 0 || rows == 0 || cut == 0 || column_id == 0 ||
      attnum == 0 || declared_oid != 23 || signed_size != 4 || root_format != 1 || shard_count == 0) {
    gpu_db_store_u32(output, 1); return;
  }
  u32 typed_vector_value_bytes = 0;
  u32 typed_vector_frame_bytes = 0;
  u32 row_vector_value_bytes = 0;
  u32 row_vector_frame_bytes = 0;
  if (!gpu_db_vector_frame_bytes(
          sizeof(gpu_db_typed_vector_proof_domain) - 1, rows,
          &typed_vector_value_bytes, &typed_vector_frame_bytes) ||
      !gpu_db_vector_frame_bytes(
          sizeof(gpu_db_row_vector_proof_domain) - 1, rows,
          &row_vector_value_bytes, &row_vector_frame_bytes)) {
    gpu_db_store_u32(output, 8); return;
  }
  gpu_db_store_u32(output, 0);

  // The host validates contiguous shard coverage too. The device repeats it because it is the
  // root authority for row identity and must not trust a staged descriptor.
  u64 covered = 0;
  for (u32 shard_index = 0; shard_index < shard_count; ++shard_index) {
    const u8* shard = gpu_db_shard(descriptor, shard_index);
    const u64 start = gpu_db_load_u64(shard);
    const u64 count = gpu_db_load_u64(shard + 8);
    if (count == 0 || start != covered || count > rows - covered) { gpu_db_store_u32(output, 2); return; }
    const u8* validity = (const u8*)(u64)gpu_db_load_u64(shard + 24);
    const u64 validity_bytes = (count + 31ull) / 32ull * 4ull;
    if ((count & 31ull) != 0) {
      const u32 tail = gpu_db_load_u32(validity + validity_bytes - 4ull);
      const u32 allowed = (1u << (u32)(count & 31ull)) - 1u;
      if ((tail & ~allowed) != 0) { gpu_db_store_u32(output, 3); return; }
    }
    covered += count;
  }
  if (covered != rows) { gpu_db_store_u32(output, 4); return; }

  // Shape root binds the catalog facts; no physical layout address enters any logical digest.
  u8 preimage[256];
  u32 at = 0;
  gpu_db_append_domain(preimage, &at, gpu_db_column_shape_domain, sizeof(gpu_db_column_shape_domain) - 1);
  gpu_db_append_u64(preimage, &at, table_id);
  gpu_db_append_u64(preimage, &at, column_id);
  gpu_db_append_u16(preimage, &at, attnum);
  gpu_db_append_bytes(preimage, &at, (const u8*)"int4", 4);
  gpu_db_append_u32(preimage, &at, declared_oid);
  gpu_db_append_u16(preimage, &at, signed_size);
  gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_COLUMN_SHAPE));

  u8* typed = workspace;
  u8* leaves = typed + rows * 32ull;
  u8* scratch = leaves + rows * 32ull;
  u64 previous = 0;
  for (u64 row = 0; row < rows; ++row) {
    const u8* shard = 0;
    u64 local = 0;
    if (!gpu_db_find_row(descriptor, row, &shard, &local)) { gpu_db_store_u32(output, 5); return; }
    const u8* ids = (const u8*)(u64)gpu_db_load_u64(shard + 16);
    const u8* validity = (const u8*)(u64)gpu_db_load_u64(shard + 24);
    const u8* values = (const u8*)(u64)gpu_db_load_u64(shard + 32);
    const u8* created = (const u8*)(u64)gpu_db_load_u64(shard + 40);
    const u8* deleted = (const u8*)(u64)gpu_db_load_u64(shard + 48);
    const u64 row_id = gpu_db_load_u64(ids + local * 8ull);
    const u64 created_by = gpu_db_load_u64(created + local * 8ull);
    const u64 deleted_by = gpu_db_load_u64(deleted + local * 8ull);
    if (row_id == 0 || (row != 0 && row_id <= previous) || created_by == 0 || created_by > cut || deleted_by <= cut) {
      gpu_db_store_u32(output, 6); return;
    }
    previous = row_id;
    const bool is_null = ((gpu_db_load_u32(validity + (local >> 5) * 4ull) >> (local & 31ull)) & 1u) == 0;
    at = 0;
    gpu_db_append_domain(preimage, &at, gpu_db_typed_value_domain, sizeof(gpu_db_typed_value_domain) - 1);
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_COLUMN_SHAPE), 32);
    preimage[at++] = is_null ? 1 : 0;
    gpu_db_append_u32(preimage, &at, is_null ? 0 : 4);
    if (!is_null) gpu_db_append_bytes(preimage, &at, values + local * 4ull, 4);
    gpu_db_sha256_bytes(preimage, at, typed + row * 32ull);
    at = 0;
    gpu_db_append_domain(preimage, &at, gpu_db_current_row_domain, sizeof(gpu_db_current_row_domain) - 1);
    gpu_db_append_u64(preimage, &at, table_id);
    gpu_db_append_u64(preimage, &at, row_id);
    gpu_db_append_u64(preimage, &at, created_by);
    gpu_db_append_u32(preimage, &at, 1);
    gpu_db_append_u64(preimage, &at, column_id);
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_COLUMN_SHAPE), 32);
    gpu_db_append_bytes(preimage, &at, typed + row * 32ull, 32);
    gpu_db_sha256_bytes(preimage, at, leaves + row * 32ull);
  }
  if (!gpu_db_hash_vector(
          gpu_db_typed_vector_proof_domain, sizeof(gpu_db_typed_vector_proof_domain) - 1,
          rows, typed, typed_vector_value_bytes, typed_vector_frame_bytes, scratch,
          gpu_db_slot(output, GPU_DB_REBUILD_SLOT_TYPED_VECTOR)) ||
      !gpu_db_hash_vector(
          gpu_db_row_vector_proof_domain, sizeof(gpu_db_row_vector_proof_domain) - 1,
          rows, leaves, row_vector_value_bytes, row_vector_frame_bytes, scratch,
          gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_LEAVES))) {
    gpu_db_store_u32(output, 8); return;
  }
  // Empty roots are emitted at the fixed proof coordinates depth 0..64.
  at = 0;
  gpu_db_append_domain(preimage, &at, gpu_db_row_empty_leaf_domain, sizeof(gpu_db_row_empty_leaf_domain) - 1);
  gpu_db_append_u16(preimage, &at, 1); gpu_db_append_u64(preimage, &at, table_id); gpu_db_append_u64(preimage, &at, 0);
  gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_EMPTY + 64));
  for (int depth = 63; depth >= 0; --depth) {
    at = 0; gpu_db_append_domain(preimage, &at, gpu_db_row_empty_node_domain, sizeof(gpu_db_row_empty_node_domain) - 1);
    gpu_db_append_u16(preimage, &at, 1); gpu_db_append_u64(preimage, &at, table_id); preimage[at++] = (u8)depth;
    gpu_db_append_u64(preimage, &at, 0);
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_EMPTY + depth + 1), 32);
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_EMPTY + depth + 1), 32);
    gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_EMPTY + depth));
  }
  gpu_db_row_node* current_nodes = (gpu_db_row_node*)scratch;
  gpu_db_row_node* next_nodes = current_nodes + rows;
  gpu_db_digest_count row_map = gpu_db_build_row_map_iterative(descriptor, leaves,
      gpu_db_slot(output, GPU_DB_REBUILD_SLOT_ROW_EMPTY), rows, table_id, current_nodes, next_nodes);
  if (row_map.count != rows) { gpu_db_store_u32(output, 7); return; }

  at = 0; gpu_db_append_domain(preimage, &at, gpu_db_table_root_domain, sizeof(gpu_db_table_root_domain) - 1);
  gpu_db_append_u64(preimage, &at, table_id); gpu_db_append_u64(preimage, &at, generation);
  gpu_db_append_u64(preimage, &at, rows); gpu_db_append_bytes(preimage, &at, row_map.digest, 32);
  gpu_db_append_u32(preimage, &at, 0);
  gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_TABLE_ROOT));

  const u8* database_id = descriptor;
  at = 0; gpu_db_append_domain(preimage, &at, gpu_db_map_empty_leaf_domain, sizeof(gpu_db_map_empty_leaf_domain) - 1);
  gpu_db_append_u16(preimage, &at, 1); gpu_db_append_bytes(preimage, &at, database_id, 16);
  gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_EMPTY + 64));
  for (int depth = 63; depth >= 0; --depth) {
    at = 0; gpu_db_append_domain(preimage, &at, gpu_db_map_empty_node_domain, sizeof(gpu_db_map_empty_node_domain) - 1);
    gpu_db_append_u16(preimage, &at, 1); gpu_db_append_bytes(preimage, &at, database_id, 16); preimage[at++] = (u8)depth;
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_EMPTY + depth + 1), 32);
    gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_EMPTY + depth + 1), 32);
    gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_EMPTY + depth));
  }
  u8 table_map[32];
  at = 0; gpu_db_append_domain(preimage, &at, gpu_db_map_leaf_domain, sizeof(gpu_db_map_leaf_domain) - 1);
  gpu_db_append_u16(preimage, &at, 1); gpu_db_append_bytes(preimage, &at, database_id, 16);
  gpu_db_append_u64(preimage, &at, table_id); gpu_db_append_bytes(preimage, &at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_TABLE_ROOT), 32);
  gpu_db_sha256_bytes(preimage, at, table_map);
  for (int depth = 63; depth >= 0; --depth) {
    u8 next[32]; at = 0; gpu_db_append_domain(preimage, &at, gpu_db_map_node_domain, sizeof(gpu_db_map_node_domain) - 1);
    gpu_db_append_u16(preimage, &at, 1); gpu_db_append_bytes(preimage, &at, database_id, 16); preimage[at++] = (u8)depth;
    const u8* empty = gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_EMPTY + depth + 1);
    if (((table_id >> (63u - (u32)depth)) & 1ull) == 0) {
      gpu_db_append_bytes(preimage, &at, table_map, 32); gpu_db_append_bytes(preimage, &at, empty, 32);
    } else {
      gpu_db_append_bytes(preimage, &at, empty, 32); gpu_db_append_bytes(preimage, &at, table_map, 32);
    }
    gpu_db_sha256_bytes(preimage, at, next);
    #pragma unroll
    for (u32 byte = 0; byte < 32; ++byte) table_map[byte] = next[byte];
  }
  at = 0; gpu_db_append_domain(preimage, &at, gpu_db_database_root_domain, sizeof(gpu_db_database_root_domain) - 1);
  gpu_db_append_u16(preimage, &at, 1); gpu_db_append_bytes(preimage, &at, database_id, 16);
  gpu_db_append_bytes(preimage, &at, table_map, 32);
  gpu_db_sha256_bytes(preimage, at, gpu_db_slot(output, GPU_DB_REBUILD_SLOT_DATABASE_ROOT));
}
