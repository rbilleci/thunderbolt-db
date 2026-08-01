// Canonical source for sha256_kernel.ptx. Regenerate from repository root with:
// nvcc --ptx --gpu-architecture=compute_60 --std=c++14 -O3 \
//   -o crates/execution/src/sha256_kernel.ptx crates/execution/src/sha256_kernel.cu && \
// perl -0pi -e 's/\n\n\z/\n/' crates/execution/src/sha256_kernel.ptx
// The pragma controls keep register/local-memory geometry bounded for one independent digest.

#include "sha256_device.cuh"
extern "C" __global__ void gpu_db_sha256_buffers(
    const unsigned long long* descriptors,
    unsigned int count,
    unsigned char* output) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (; index < count; index += stride) {
    const unsigned char* input =
        (const unsigned char*)(unsigned long long)descriptors[index * 2];
    gpu_db_sha256_bytes(input, descriptors[index * 2 + 1], output + index * 32ull);
  }
}

__device__ __constant__ unsigned char gpu_db_map_empty_leaf_domain[] =
    "gpu-db/runtime-generation/map-empty-leaf/v1";
__device__ __constant__ unsigned char gpu_db_map_empty_node_domain[] =
    "gpu-db/runtime-generation/map-empty-node/v1";
__device__ __constant__ unsigned char gpu_db_status_empty_leaf_domain[] =
    "gpu-db/runtime-generation/status-empty-leaf/v1";
__device__ __constant__ unsigned char gpu_db_status_empty_node_domain[] =
    "gpu-db/runtime-generation/status-empty-node/v1";
__device__ __constant__ unsigned char gpu_db_database_root_domain[] =
    "gpu-db/runtime-generation/database-root/v1";

__device__ void gpu_db_append_u16_le(unsigned char* destination, unsigned int* cursor, unsigned short value) {
  destination[(*cursor)++] = (unsigned char)value;
  destination[(*cursor)++] = (unsigned char)(value >> 8);
}

__device__ void gpu_db_append_u64_le(unsigned char* destination, unsigned int* cursor, unsigned long long value) {
  #pragma unroll
  for (unsigned int byte = 0; byte < 8; ++byte) {
    destination[(*cursor)++] = (unsigned char)(value >> (byte * 8));
  }
}

__device__ void gpu_db_append_bytes(
    unsigned char* destination,
    unsigned int* cursor,
    const unsigned char* source,
    unsigned int length) {
  #pragma unroll 1
  for (unsigned int byte = 0; byte < length; ++byte) {
    destination[(*cursor)++] = source[byte];
  }
}

__device__ void gpu_db_append_domain(
    unsigned char* destination,
    unsigned int* cursor,
    const unsigned char* domain,
    unsigned int domain_length) {
  gpu_db_append_u64_le(destination, cursor, (unsigned long long)domain_length);
  gpu_db_append_bytes(destination, cursor, domain, domain_length);
}

// The fixed runtime-generation-v1 genesis grammar has no caller-supplied domain text, type code,
// child digest, or ordering. `configuration[0]` is the device pointer to exactly 16 database-ID
// bytes and `configuration[1]` is the fixed root-format version 1. It emits the logical slot
// order map empty depth 0..64, status empty depth 0..64, database root.
extern "C" __global__ void gpu_db_runtime_generation_v1_genesis_roots(
    const unsigned long long* configuration,
    unsigned char* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const unsigned char* database_id =
      (const unsigned char*)(unsigned long long)configuration[0];
  const unsigned short root_format = (unsigned short)configuration[1];
  unsigned char preimage[160];
  unsigned int cursor = 0;

  // Table map empty leaf is depth 64. Every parent copies the immediately deeper root twice.
  gpu_db_append_domain(preimage, &cursor, gpu_db_map_empty_leaf_domain,
                       sizeof(gpu_db_map_empty_leaf_domain) - 1);
  gpu_db_append_u16_le(preimage, &cursor, root_format);
  gpu_db_append_bytes(preimage, &cursor, database_id, 16);
  gpu_db_sha256_bytes(preimage, cursor, output + 64ull * 32ull);
  for (int depth = 63; depth >= 0; --depth) {
    cursor = 0;
    gpu_db_append_domain(preimage, &cursor, gpu_db_map_empty_node_domain,
                         sizeof(gpu_db_map_empty_node_domain) - 1);
    gpu_db_append_u16_le(preimage, &cursor, root_format);
    gpu_db_append_bytes(preimage, &cursor, database_id, 16);
    preimage[cursor++] = (unsigned char)depth;
    gpu_db_append_bytes(preimage, &cursor, output + ((unsigned long long)depth + 1ull) * 32ull, 32);
    gpu_db_append_bytes(preimage, &cursor, output + ((unsigned long long)depth + 1ull) * 32ull, 32);
    gpu_db_sha256_bytes(preimage, cursor, output + (unsigned long long)depth * 32ull);
  }

  // Status-view empty roots have their independent domain and bind a zero subtree count.
  cursor = 0;
  gpu_db_append_domain(preimage, &cursor, gpu_db_status_empty_leaf_domain,
                       sizeof(gpu_db_status_empty_leaf_domain) - 1);
  gpu_db_append_u16_le(preimage, &cursor, root_format);
  gpu_db_append_bytes(preimage, &cursor, database_id, 16);
  gpu_db_append_u64_le(preimage, &cursor, 0);
  gpu_db_sha256_bytes(preimage, cursor, output + (65ull + 64ull) * 32ull);
  for (int depth = 63; depth >= 0; --depth) {
    cursor = 0;
    gpu_db_append_domain(preimage, &cursor, gpu_db_status_empty_node_domain,
                         sizeof(gpu_db_status_empty_node_domain) - 1);
    gpu_db_append_u16_le(preimage, &cursor, root_format);
    gpu_db_append_bytes(preimage, &cursor, database_id, 16);
    preimage[cursor++] = (unsigned char)depth;
    gpu_db_append_u64_le(preimage, &cursor, 0);
    gpu_db_append_bytes(preimage, &cursor,
                        output + (65ull + (unsigned long long)depth + 1ull) * 32ull, 32);
    gpu_db_append_bytes(preimage, &cursor,
                        output + (65ull + (unsigned long long)depth + 1ull) * 32ull, 32);
    gpu_db_sha256_bytes(preimage, cursor,
                        output + (65ull + (unsigned long long)depth) * 32ull);
  }

  cursor = 0;
  gpu_db_append_domain(preimage, &cursor, gpu_db_database_root_domain,
                       sizeof(gpu_db_database_root_domain) - 1);
  gpu_db_append_u16_le(preimage, &cursor, root_format);
  gpu_db_append_bytes(preimage, &cursor, database_id, 16);
  gpu_db_append_bytes(preimage, &cursor, output, 32);
  gpu_db_sha256_bytes(preimage, cursor, output + 130ull * 32ull);
}
