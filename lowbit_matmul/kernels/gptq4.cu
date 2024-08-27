#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

#if 1
constexpr int BLOCKWIDTH = 256;
constexpr int BLOCKHEIGHT = 256; // 24

template <typename T>
__global__ void
vec_mm_s4_kernel(const T *__restrict__ vec, const int32_t *__restrict__ mat,
                 T *__restrict__ mul, const T *__restrict__ scales,
                 const T *__restrict__ zeros, int K, int N) {
  int row = (BLOCKHEIGHT >> 3) * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ T blockvec[BLOCKHEIGHT];
  T val = 0;
  // if (threadIdx.x < BLOCKHEIGHT) {
#pragma unroll
  for (int i = 0; i < BLOCKHEIGHT; i += BLOCKWIDTH) {
    val = vec[blockIdx.x * BLOCKHEIGHT + i + threadIdx.x];
    blockvec[i + threadIdx.x] = val;
  }

  __syncthreads();
  // val = vec[blockIdx.x * BLOCKHEIGHT + threadIdx.x];
  // blockvec[threadIdx.x] = val;
  // }
  // if (threadIdx.x == 0) {
  // printf("vec[%d] = %f, K: %d\n",
  //        int((row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x), val, K);
  // }

  T scale = scales[col];
  T zero = zeros[col];

  T res = 0;
  int i = N * row + col;
  unsigned int tmp;

#pragma unroll
  for (size_t k = 0; k < BLOCKHEIGHT; k += 32) {
    // while (k < BLOCKHEIGHT) {
    // 73516240
    tmp = mat[i];
    res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];
    i += N;
    tmp = mat[i];
    res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 8];
    res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 9];
    res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 10];
    res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 11];
    res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 12];
    res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 13];
    res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 14];
    res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 15];
    i += N;
    tmp = mat[i];
    res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 16];
    res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 17];
    res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 18];
    res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 19];
    res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 20];
    res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 21];
    res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 22];
    res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 23];
    i += N;
    tmp = mat[i];
    res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 24];
    res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 25];
    res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 26];
    res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 27];
    res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 28];
    res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 29];
    res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 30];
    res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 31];
    i += N;
    // tmp = mat[i];
    // res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    // res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    // res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    // res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    // res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    // res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    // res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    // res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];
    // res += (scale * T((tmp >> 32) & 0xF) - zero) * blockvec[k + 8];
    // res += (scale * T((tmp >> 36) & 0xF) - zero) * blockvec[k + 9];
    // res += (scale * T((tmp >> 40) & 0xF) - zero) * blockvec[k + 10];
    // res += (scale * T((tmp >> 44) & 0xF) - zero) * blockvec[k + 11];
    // res += (scale * T((tmp >> 48) & 0xF) - zero) * blockvec[k + 12];
    // res += (scale * T((tmp >> 52) & 0xF) - zero) * blockvec[k + 13];
    // res += (scale * T((tmp >> 56) & 0xF) - zero) * blockvec[k + 14];
    // res += (scale * T((tmp >> 60) & 0xF) - zero) * blockvec[k + 15];
    // i += N;
    // tmp = mat[i];
    // res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 16];
    // res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 17];
    // res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 18];
    // res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 19];
    // res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 20];
    // res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 21];
    // res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 22];
    // res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 23];
    // res += (scale * T((tmp >> 32) & 0xF) - zero) * blockvec[k + 24];
    // res += (scale * T((tmp >> 36) & 0xF) - zero) * blockvec[k + 25];
    // res += (scale * T((tmp >> 40) & 0xF) - zero) * blockvec[k + 26];
    // res += (scale * T((tmp >> 44) & 0xF) - zero) * blockvec[k + 27];
    // res += (scale * T((tmp >> 48) & 0xF) - zero) * blockvec[k + 28];
    // res += (scale * T((tmp >> 52) & 0xF) - zero) * blockvec[k + 29];
    // res += (scale * T((tmp >> 56) & 0xF) - zero) * blockvec[k + 30];
    // res += (scale * T((tmp >> 60) & 0xF) - zero) * blockvec[k + 31];
    // i += N;
    // tmp = mat[i];
    // res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 32];
    // res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 33];
    // res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 34];
    // res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 35];
    // res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 36];
    // res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 37];
    // res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 38];
    // res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 39];
    // res += (scale * T((tmp >> 32) & 0xF) - zero) * blockvec[k + 40];
    // res += (scale * T((tmp >> 36) & 0xF) - zero) * blockvec[k + 41];
    // res += (scale * T((tmp >> 40) & 0xF) - zero) * blockvec[k + 42];
    // res += (scale * T((tmp >> 44) & 0xF) - zero) * blockvec[k + 43];
    // res += (scale * T((tmp >> 48) & 0xF) - zero) * blockvec[k + 44];
    // res += (scale * T((tmp >> 52) & 0xF) - zero) * blockvec[k + 45];
    // res += (scale * T((tmp >> 56) & 0xF) - zero) * blockvec[k + 46];
    // res += (scale * T((tmp >> 60) & 0xF) - zero) * blockvec[k + 47];
    // i += N;
    // tmp = mat[i];
    // res += (scale * T((tmp >> 0) & 0xF) - zero) * blockvec[k + 48];
    // res += (scale * T((tmp >> 4) & 0xF) - zero) * blockvec[k + 49];
    // res += (scale * T((tmp >> 8) & 0xF) - zero) * blockvec[k + 50];
    // res += (scale * T((tmp >> 12) & 0xF) - zero) * blockvec[k + 51];
    // res += (scale * T((tmp >> 16) & 0xF) - zero) * blockvec[k + 52];
    // res += (scale * T((tmp >> 20) & 0xF) - zero) * blockvec[k + 53];
    // res += (scale * T((tmp >> 24) & 0xF) - zero) * blockvec[k + 54];
    // res += (scale * T((tmp >> 28) & 0xF) - zero) * blockvec[k + 55];
    // res += (scale * T((tmp >> 32) & 0xF) - zero) * blockvec[k + 56];
    // res += (scale * T((tmp >> 36) & 0xF) - zero) * blockvec[k + 57];
    // res += (scale * T((tmp >> 40) & 0xF) - zero) * blockvec[k + 58];
    // res += (scale * T((tmp >> 44) & 0xF) - zero) * blockvec[k + 59];
    // res += (scale * T((tmp >> 48) & 0xF) - zero) * blockvec[k + 60];
    // res += (scale * T((tmp >> 52) & 0xF) - zero) * blockvec[k + 61];
    // res += (scale * T((tmp >> 56) & 0xF) - zero) * blockvec[k + 62];
    // res += (scale * T((tmp >> 60) & 0xF) - zero) * blockvec[k + 63];
    // i += N;
    // k += 16;
  }

  atomicAdd(&mul[col], res);
}

void vec_mm_s4_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                    torch::Tensor scales, torch::Tensor zeros) {
  int K = mat.size(0) << 3; // 4 bits in uint64
  int N = mat.size(1);

  dim3 blocks((K + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
              (N + BLOCKWIDTH - 1) / BLOCKWIDTH);
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(vec.type(), "vec_mm_s4_cuda", [&] {
    vec_mm_s4_kernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int32_t>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(), K, N);
  });
}
#else

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT = 32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int *>(&i);
}

template <typename scalar_t>
__global__ void
vec_mm_s4_kernel(const scalar_t *__restrict__ vec, const int *__restrict__ mat,
                 scalar_t *__restrict__ mul,
                 const scalar_t *__restrict__ scales,
                 const scalar_t *__restrict__ zeros, int height, int width) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  // printf("row: %d, col: %d\n", row, col);

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  scalar_t val = 0;
  // printf("3\n");
  // height << 3 (height * 8 (32-bit / 4-bit))
  if ((row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x < (height << 3)) {
    val = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  }
  // printf("4\n");
  blockvec[threadIdx.x] = val;
  __syncthreads();

  // printf("5\n");
  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];
  // printf("6\n");

  scalar_t res = 0;
  int i = width * row + col;
  int k = 0;

  // unsigned int tmp1;
  // unsigned int tmp2;
  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = mat[i];
    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];
    i += width;
    tmp = mat[i];
    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 9];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 10];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 11];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 12];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 13];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 14];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 15];
    i += width;
    tmp = mat[i];
    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 16];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 17];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 18];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 19];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 20];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 21];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 22];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 23];
    i += width;
    tmp = mat[i];
    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 24];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 25];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 26];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 27];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 28];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 29];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 30];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 31];
    i += width;
    k += 32;
    // printf("i: %d, k: %d, row: %d, col: %d\n", i, k, i / width, i % width);
  }

  atomicAdd(&mul[col], res);
  // mul[col] = res;
}

void vec_mm_s4_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                    torch::Tensor scales, torch::Tensor zeros) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks((height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
              (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
      vec.type(), "vec_mm_s4_cuda", ([&] {
        vec_mm_s4_kernel<<<blocks, threads>>>(
            vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
            scales.data<scalar_t>(), zeros.data<scalar_t>(), height, width);
      }));
}
#endif