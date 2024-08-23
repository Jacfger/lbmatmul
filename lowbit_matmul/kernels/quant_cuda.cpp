#include "ATen/Dispatch.h"
#include "ATen/ops/matmul_ops.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/python.h>

void _nonmul_packedmask_cuda(torch::Tensor &mat, torch::Tensor mask);

void _nonmul_packedmask(torch::Tensor &mat, torch::Tensor mask) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  _nonmul_packedmask_cuda(mat, mask);
}

void _nonmul_packed4mask_cuda(torch::Tensor &mat, torch::Tensor mask);

void _nonmul_packed4mask(torch::Tensor &mat, torch::Tensor mask) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  _nonmul_packed4mask_cuda(mat, mask);
}

void _nonmul_packedmask_2d_cuda(torch::Tensor &mat, torch::Tensor mask);

void _nonmul_packedmask_2d(torch::Tensor &mat, torch::Tensor mask) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  _nonmul_packedmask_2d_cuda(mat, mask);
}

template <typename T1>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T1 const alpha, T1 const *A, size_t lda,
                                       T1 const *B, size_t ldb, T1 const beta,
                                       T1 *C, size_t ldc, cudaStream_t stream);

torch::Tensor matmul_test_cuda(torch::Tensor &A, torch::Tensor &B);
torch::Tensor matmul_test(torch::Tensor &A, torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  return matmul_test_cuda(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix
  // Multiplication (CUDA)"); m.def("vecquant3matmul_faster",
  // &vecquant3matmul_faster, "Vector 3-bit Quantized Matrix Multiplication
  // (CUDA), faster version");
  m.def("_nonmul_packedmask", &_nonmul_packedmask,
        "Non-multiplicative Masking Operation (In Place)");
  m.def("_nonmul_packed4mask", &_nonmul_packed4mask,
        "Non-multiplicative Masking Operation (In Place)");
  m.def("_nonmul_packedmask_2d", &_nonmul_packedmask_2d,
        "Non-multiplicative Masking Operation (In Place)");
  m.def("matmul_test", &matmul_test, "Matrix Multiplication (CUDA)");
}
