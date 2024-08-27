/***
Copy and pasted from Mao, Lei. Thx bro :D, nice name btw.

But a lot of optimization could still be done. (For future me)
1. vectorized loading
2. I think there's some problem with how the thread is tiling. Like loading from
shared memory should still affected by coalescing. So the each thread should be
interlacing, but right now each thread calculate a tile block. (Maybe this will
be better if I use the gemm_v06 from Mao, Lei?)
3. shared memory coalscing. Apparently it's a thing, guess that make sense.
(like I should transpose A shared memory probably)
4. tensorcore.

TODO: Read gemm_v05~7 from Mao, Lei. sub-byte type makes implementing v06 way
too hard. Probably need a lot more time to figure that out.
****/

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

#include <iostream>
#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const *const file, int const line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cout << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cout << cudaGetErrorString(err) << std::endl;
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_K,
          size_t NUM_THREADS>
__device__ __forceinline__ void load_data_to_shared_memory_b(
    T const *__restrict__ B, size_t ldb,
    T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t n,
    size_t k) {
#pragma unroll
  for (size_t load_idx{0U};
       load_idx <
       (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
       ++load_idx) {
    size_t const B_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
    size_t const B_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
    size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           B_thread_block_tile_row_idx};
    size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                           B_thread_block_tile_col_idx};

    T val{static_cast<T>(0)};
    if (B_row_idx < k && B_col_idx < n) {
      val = B[B_row_idx * ldb + B_col_idx];
    }
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    B_thread_block_tile[B_thread_block_tile_row_idx]
                       [B_thread_block_tile_col_idx] = val;
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K,
          size_t NUM_THREADS>
__device__ __forceinline__ void load_data_to_shared_memory_a(
    T const *__restrict__ A, size_t lda,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m,
    size_t k) {
#pragma unroll
  for (size_t load_idx{0U};
       load_idx <
       (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
       ++load_idx) {
    size_t const A_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
    size_t const A_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
    size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                           A_thread_block_tile_row_idx};
    size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           A_thread_block_tile_col_idx};

    T val{static_cast<T>(0)};
    if (A_row_idx < m && A_col_idx < k) {
      val = A[A_row_idx * lda + A_col_idx];
    }
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    A_thread_block_tile[A_thread_block_tile_row_idx]
                       [A_thread_block_tile_col_idx] = val;
  }
}

// GEMM kernel v04.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void
gemm_v04(size_t m, size_t n, size_t k, T alpha, T const *__restrict__ A,
         size_t lda, uint8_t const *__restrict__ B, size_t ldb, T beta,
         T *__restrict__ C, size_t ldc, T const *__restrict__ scales,
         T const *__restrict__ zeros) {
  // Avoid using blockDim.x * blockDim.y as the number of threads per block.
  // Because it is a runtime constant and the compiler cannot optimize the
  // loop unrolling based on that.
  // Use a compile time constant instead.
  constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                               (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
  size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

  // Cache a tile of A and B in shared memory for data reuse.
  __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
  size_t constexpr BLOCK_TILE_SIZE_K_b = BLOCK_TILE_SIZE_K / 2;
  __shared__ uint8_t
      B_thread_block_tile[BLOCK_TILE_SIZE_K_b][BLOCK_TILE_SIZE_X];

  __shared__ T scales_thread_block_tile[BLOCK_TILE_SIZE_X];
  __shared__ T zeros_thread_block_tile[BLOCK_TILE_SIZE_X];

  size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                      BLOCK_TILE_SIZE_K};

  // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
  // Specifically, these values corresponds to
  // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
  // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
  // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
  // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
  // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
  // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
  T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
      static_cast<T>(0)};
  // A_vals is cached in the register.
  T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
  // B_vals is cached in the register. (or L1 L2?)
  uint8_t B_vals[THREAD_TILE_SIZE_X] = {0};
  T scale_vals[THREAD_TILE_SIZE_X] = {0};
  T zero_vals[THREAD_TILE_SIZE_X] = {0};
#if 1
  for (size_t block_col_idx = threadIdx.x; block_col_idx < BLOCK_TILE_SIZE_X;
       block_col_idx += blockDim.x) {
    if ((blockIdx.x * BLOCK_TILE_SIZE_X + block_col_idx) < n) {
      scales_thread_block_tile[block_col_idx] =
          scales[blockIdx.x * BLOCK_TILE_SIZE_X + block_col_idx];
      zeros_thread_block_tile[block_col_idx] =
          zeros[blockIdx.x * BLOCK_TILE_SIZE_X + block_col_idx];
    }
  }
  __syncthreads();
#endif

#if 0 // DEBUG PRINT
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("==============Barreir==============\n");
    for (size_t i = 0; i < BLOCK_TILE_SIZE_X; ++i) {
      printf("scales[%d]: %f, zeros[%d]: %f\n", (int32_t)i,
             (float)scales_thread_block_tile[i], (int32_t)i,
             (float)zeros_thread_block_tile[i]);
    }
  }
  __syncthreads();
#endif

  for (size_t thread_block_tile_idx{0U};
       thread_block_tile_idx < num_thread_block_tiles;
       ++thread_block_tile_idx) {

    load_data_to_shared_memory_a<T, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                                 NUM_THREADS>(A, lda, A_thread_block_tile,
                                              thread_block_tile_idx,
                                              thread_linear_idx, m, k);
    load_data_to_shared_memory_b<uint8_t, BLOCK_TILE_SIZE_X,
                                 BLOCK_TILE_SIZE_K_b, NUM_THREADS>(
        B, ldb, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx,
        n, k / 2);

    __syncthreads();

    size_t const B_thread_block_tile_col_idx{
        thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
        THREAD_TILE_SIZE_X};

    // if (threadIdx.x == 0 && threadIdx.y == 0)
    //   printf("==============Barreir==============\n");
    __syncthreads();

#if 1
    for (size_t thread_tile_col_idx{0U};
         thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
      scale_vals[thread_tile_col_idx] =
          scales_thread_block_tile[B_thread_block_tile_col_idx +
                                   thread_tile_col_idx];
      zero_vals[thread_tile_col_idx] =
          zeros_thread_block_tile[B_thread_block_tile_col_idx +
                                  thread_tile_col_idx];
    }
#endif

#pragma unroll
    for (size_t k_i_2{0U}; k_i_2 < (BLOCK_TILE_SIZE_K >> 1); ++k_i_2) {

      size_t const B_thread_block_tile_row_idx{k_i_2};

#pragma unroll
      for (size_t thread_tile_col_idx{0U};
           thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
        B_vals[thread_tile_col_idx] =
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx +
                                thread_tile_col_idx];
      }

      for (size_t k_off = 0; k_off < 2; ++k_off) {
        size_t k_i = (k_i_2 << 1) + k_off;
        size_t const A_thread_block_tile_row_idx{
            thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
            THREAD_TILE_SIZE_Y};
        size_t const A_thread_block_tile_col_idx{k_i};

#pragma unroll
        for (size_t thread_tile_row_idx{0U};
             thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
          // There will be shared memory bank conflicts accessing the
          // values from A_thread_block_tile. We can do it better by
          // transposing the A_thread_block_tile when we load the data
          // from DRAM.
          A_vals[thread_tile_row_idx] =
              A_thread_block_tile[A_thread_block_tile_row_idx +
                                  thread_tile_row_idx]
                                 [A_thread_block_tile_col_idx];
        }

#pragma unroll
        for (size_t thread_tile_row_idx{0U};
             thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
#pragma unroll
          for (size_t thread_tile_col_idx{0U};
               thread_tile_col_idx < THREAD_TILE_SIZE_X;
               ++thread_tile_col_idx) {
            uint8_t b_val = B_vals[thread_tile_col_idx];
            if (k_off) { // 1
              b_val = b_val >> 4;
            } else { // 0
              b_val = b_val & 0x0F;
            }
            C_thread_results[thread_tile_row_idx][thread_tile_col_idx] +=
                A_vals[thread_tile_row_idx] *
                (scale_vals[thread_tile_col_idx] * static_cast<T>(b_val) -
                 zero_vals[thread_tile_col_idx]);
          }
        }
      }
    }
    __syncthreads();
  }

// Write the results to DRAM.
#pragma unroll
  for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y;
       ++thread_tile_row_idx) {
#pragma unroll
    for (size_t thread_tile_col_idx{0U};
         thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx) {
      size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                             threadIdx.x /
                                 (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                                 THREAD_TILE_SIZE_Y +
                             thread_tile_row_idx};
      size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                             threadIdx.x %
                                 (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                                 THREAD_TILE_SIZE_X +
                             thread_tile_col_idx};
      if (C_row_idx < m && C_col_idx < n) {
        C[C_row_idx * ldc + C_col_idx] =
            C_thread_results[thread_tile_row_idx][thread_tile_col_idx];
      }
    }
  }
}

template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T const alpha,
                            T const *__restrict__ A, size_t lda,
                            uint8_t const *__restrict__ B, size_t ldb,
                            T const beta, T *__restrict__ C, size_t ldc,
                            T const *__restrict__ scales,
                            T const *__restrict__ zeros, cudaStream_t stream) {
  // Feel free to play with the block tile sizes.
  // The algorithm correctness should always be guaranteed.
  constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
  constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
  constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
  // Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
  constexpr unsigned int THREAD_TILE_SIZE_X{8U};
  constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
  constexpr unsigned int NUM_THREADS_PER_BLOCK{
      BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
      (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
  static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
  static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
  static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
  static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
  static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK ==
                0U);
  static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK ==
                0U);
  dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
  dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
                          BLOCK_TILE_SIZE_X,
                      (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
                          BLOCK_TILE_SIZE_Y,
                      1U};
  gemm_v04<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
           THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
      <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, lda, B, ldb,
                                            beta, C, ldc, scales, zeros);
  CHECK_LAST_CUDA_ERROR();
}

void matmul_test_cuda(torch::Tensor &A, torch::Tensor &B, torch::Tensor &scales,
                      torch::Tensor &zeros, torch::Tensor &C) {
  long m = A.size(0);
  long k = A.size(1);
  long n = B.size(1);

  assert(k % 2 == 0); // k must be even, not considering if it's not.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul", [&] {
    scalar_t a = 1, b = 0;
    launch_gemm_kernel_v04<scalar_t>(
        m, n, k, a, A.data_ptr<scalar_t>(), k, B.data_ptr<uint8_t>(), n, b,
        // C.data_ptr<scalar_t>(), n, 0, scales, zeros);
        C.data_ptr<scalar_t>(), n, scales.data_ptr<scalar_t>(),
        zeros.data_ptr<scalar_t>(), 0);
  });
}