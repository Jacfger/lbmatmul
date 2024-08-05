#include "gemm.h"
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <stdexcept>

void matmul_host(const uint8_t *A, const uint8_t *B, uint32_t M, uint32_t N,
                 uint32_t K, int32_t *C) {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,               // ElementA
      cutlass::layout::RowMajor,      // LayoutA
      cutlass::int4b_t,               // ElementB
      cutlass::layout::ColumnMajor,   // LayoutB
      int32_t,                        // ElementOutput
      cutlass::layout::RowMajor,      // LayoutOutput
      int32_t,                        // ElementAccumulator
      cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
      cutlass::arch::Sm80 // tag indicating target GPU compute architecture  //
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{{static_cast<GemmCoord::Index>(M),
                                      static_cast<GemmCoord::Index>(N),
                                      static_cast<GemmCoord::Index>(K)},
                                     {(cutlass::int4b_t *)A, K},
                                     {(cutlass::int4b_t *)B, K},
                                     {C, N},
                                     {C, N},
                                     {1, 0}};

  auto status = gemmOp(arguments);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Runtime error: CUTLASS execution failed:"
              << cutlassGetStatusString(status) << std::endl;
    throw std::runtime_error(cutlassGetStatusString(status));
  }
}

inline __device__ int8_t __lower_i4(int8_t x) {
  if (x & 0x08)
    return x | 0xF0;
  else
    return x & 0x0F;
}

inline __device__ int8_t __upper_i4(int8_t x) { return x >> 4; }

/// Naive reference GEMM computation of half precision and 4-bit interger
__global__ void hi4_mma(int M, int N, int K, __half const *A, int lda,
                        int8_t const *B, int ldb, __half *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("parameters: M: %d, N: %d, K: %d\n", M, N, K);
  // printf("ld: %d, %d, %d\n", lda, ldb, ldc);
  if (i < M && j < N) {
    __half accumulator = 0;

    for (int k = 0; k < K / 2; ++k) {
      int _k = k << 1;
      // int8_t b = B[k + j * ldb];
      // printf("A: %d, B: %d\n", i * lda + _k, _k + j * ldb);
      // printf("A[%d * %d + %d] = [%d], valid: %d\n", i, lda, _k, i * lda + _k, (i * lda + _k) < (2048 * 8192));
      int8_t b = B[j * ldb + k];
      // if (b != 0) {
      //   printf("B[%d * %d + %d] = %d\n", j, ldb, k, b);
      // }
      accumulator += A[i * lda + _k] * __int2half_rn(__lower_i4(b));
      accumulator += A[i * lda + _k + 1] * __int2half_rn(__upper_i4(b));
    }
    if (__half2float(C[i * ldc + j]) != 0) {
      printf("C[%d * %d + %d] = %f\n", i, ldc, j, __half2float(C[i * ldc + j]));
    }
    C[i * ldc + j] = accumulator; // + C[i + j * ldc];
    // printf("C[%d * %d + %d] = %f\n", i, ldc, j, __half2float(accumulator));
  }
}

void matmul_hi4_naive_cu(const __half *A, const int8_t *B, uint32_t M,
                         uint32_t N, uint32_t K, __half *C) {
  // cudaError_t matmul_hi4_naive_cu(int M, int N, int K, float alpha,
  //                                 __half const *A, int lda, int8_t const *B,
  //                                 int ldb, float beta, float *C, int ldc) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  // printf("grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
  // printf("parameters: M: %d, N: %d, K: %d\n", M, N, K);
  hi4_mma<<<grid, block>>>(M, N, K, A, K, B, K >> 1, C, N);
  // return cudaGetLastError();
}