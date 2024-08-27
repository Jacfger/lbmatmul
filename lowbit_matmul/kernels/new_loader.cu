#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum class Mode { Crosswise, Congruous };

template <typename T, size_t BLOCK_TILE_SIZE_MN,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, Mode MODE>
__device__ void load_data_to_shared_memory(
    T const *A, size_t lda,
    T A_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_MN],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t mn,
    size_t k) {
  // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
  for (size_t load_idx{0U};
       load_idx <
       (BLOCK_TILE_SIZE_MN * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
       ++load_idx) {
    size_t const A_thread_block_tile_row_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
    size_t const A_thread_block_tile_col_idx{
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
    size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_MN +
                           A_thread_block_tile_row_idx};
    size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                           A_thread_block_tile_col_idx};

    // These boundary checks might slow down the kernel to some extent.
    // But they guarantee the correctness of the kernel for all
    // different GEMM configurations.
    T val{static_cast<T>(0)};
    if (A_row_idx < mn && A_col_idx < k) {
      val = A[A_row_idx * lda + A_col_idx];
    }
    // This if will slow down the kernel.
    // Add static asserts from the host code to guarantee this if is
    // always true.
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_MN % NUM_THREADS == 0U);
    // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_MN &&
    //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
    // {
    //     A_thread_block_tile[A_thread_block_tile_row_idx]
    //                        [A_thread_block_tile_col_idx] = val;
    // }
    // A_thread_block_tile[A_thread_block_tile_row_idx]
    //                    [A_thread_block_tile_col_idx] = val;
    A_thread_block_tile[A_thread_block_tile_col_idx]
                       [A_thread_block_tile_row_idx] = val;
  }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
}