#include "c10/core/ScalarType.h"
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
// Include all files
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>

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

  if (i < M && j < N) {
    __half accumulator = 0;

    for (int k = 0; k < K / 2; ++k) {
      int _k = k << 1;
      int8_t b = B[k + j * ldb];
      accumulator += A[i + _k * lda] * __int2half_ru(__lower_i4(b));
      accumulator += A[i + _k * lda + 1] * __int2half_ru(__upper_i4(b));
    }
    C[i + j * ldc] += accumulator; // + C[i + j * ldc];
  }
}

/// Reference GEMM computation.
void matmul_hi4_naive_cu(const __half *A, const int8_t *B, uint32_t M,
                 uint32_t N, uint32_t K, __half *C){
// cudaError_t matmul_hi4_naive_cu(int M, int N, int K, float alpha,
//                                 __half const *A, int lda, int8_t const *B,
//                                 int ldb, float beta, float *C, int ldc) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  hi4_mma<<<grid, block>>>(M, N, K, A, K, B, N, C, N);
  //return cudaGetLastError();
}

torch::Tensor matmul_hi4_naive(const torch::Tensor &A, const torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * 2; // 4 bits packed into 8-bits
  auto C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));

  matmul_hi4_naive_cu(A.data_ptr<__half>(), B.data_ptr<int8_t>(), M, N, K,
              C.data_ptr<__half>());

  return C;
}

//====== pybind ======

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_hi4", &matmul_hi4_naive,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
}
