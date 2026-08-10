// Shared device SHA-256 primitive for closed CUDA root programs.
//
// This file is intentionally the sole implementation authority. Kernel translation units may
// compose canonical preimages differently, but they all call this exact byte-oriented SHA-256.

__device__ __constant__ unsigned int gpu_db_sha256_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

// Keep the one byte-oriented SHA authority out of callers' enormous finalizer frames.  Its
// fixed 64-word schedule and 64 rounds are fully unrolled here, so the device compiler can keep
// the short commitment schedule in registers instead of repeatedly indexing a local array.  The
// input-length loop remains dynamic for the existing variable-width typed values.
__device__ __noinline__ void gpu_db_sha256_bytes(
    const unsigned char* input,
    unsigned long long length,
    unsigned char* output) {
  unsigned long long blocks = (length + 72ull) >> 6;
  unsigned int h0 = 0x6a09e667u, h1 = 0xbb67ae85u, h2 = 0x3c6ef372u, h3 = 0xa54ff53au;
  unsigned int h4 = 0x510e527fu, h5 = 0x9b05688cu, h6 = 0x1f83d9abu, h7 = 0x5be0cd19u;
  #pragma unroll 1
  for (unsigned long long block = 0; block < blocks; ++block) {
    unsigned int w[64];
    unsigned long long base = block << 6;
    #pragma unroll
    for (unsigned int word = 0; word < 16; ++word) {
      unsigned int value = 0;
      #pragma unroll 1
      for (unsigned int byte = 0; byte < 4; ++byte) {
        unsigned long long position = base + word * 4ull + byte;
        unsigned int octet = 0;
        if (position < length) octet = input[position];
        else if (position == length) octet = 0x80u;
        else if (block + 1 == blocks && position >= base + 56ull) {
          unsigned int shift = (unsigned int)((base + 63ull - position) * 8ull);
          octet = (unsigned int)((length << 3) >> shift) & 0xffu;
        }
        value = (value << 8) | octet;
      }
      w[word] = value;
    }
    #pragma unroll
    for (unsigned int word = 16; word < 64; ++word) {
      unsigned int x = w[word - 15], y = w[word - 2];
      unsigned int small0 = ((x >> 7) | (x << 25)) ^ ((x >> 18) | (x << 14)) ^ (x >> 3);
      unsigned int small1 = ((y >> 17) | (y << 15)) ^ ((y >> 19) | (y << 13)) ^ (y >> 10);
      w[word] = w[word - 16] + small0 + w[word - 7] + small1;
    }
    unsigned int a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
    #pragma unroll
    for (unsigned int word = 0; word < 64; ++word) {
      unsigned int s1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
      unsigned int choice = (e & f) ^ (~e & g);
      unsigned int temporary1 = h + s1 + choice + gpu_db_sha256_k[word] + w[word];
      unsigned int s0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
      unsigned int majority = (a & b) ^ (a & c) ^ (b & c);
      unsigned int temporary2 = s0 + majority;
      h=g; g=f; f=e; e=d + temporary1; d=c; c=b; b=a; a=temporary1 + temporary2;
    }
    h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=h;
  }
  unsigned int values[8] = {h0,h1,h2,h3,h4,h5,h6,h7};
  for (unsigned int word = 0; word < 8; ++word) {
    output[word * 4ull] = (unsigned char)(values[word] >> 24);
    output[word * 4ull + 1] = (unsigned char)(values[word] >> 16);
    output[word * 4ull + 2] = (unsigned char)(values[word] >> 8);
    output[word * 4ull + 3] = (unsigned char)values[word];
  }
}
