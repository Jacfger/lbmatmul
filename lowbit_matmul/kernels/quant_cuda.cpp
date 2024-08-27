#include "ATen/Dispatch.h"
#include "ATen/ops/matmul_ops.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
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

void matmul_test_cuda(torch::Tensor &A, torch::Tensor &B, torch::Tensor &scales,
                      torch::Tensor &zeros, torch::Tensor &C);

torch::Tensor matmul_test(torch::Tensor &A, torch::Tensor &B,
                          torch::Tensor &scales, torch::Tensor &zeros) {
  // printf("non-inplace\n");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  long m = A.size(0);
  long n = B.size(1);
  auto C = torch::zeros({m, n}, A.options());
  matmul_test_cuda(A, B, scales, zeros, C);
  return C;
}

void matmul_test(torch::Tensor &A, torch::Tensor &B, torch::Tensor &scales,
                 torch::Tensor &zeros, torch::Tensor &C) {
  // printf("inplace\n");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  matmul_test_cuda(A, B, scales, zeros, C);
}

void vec_mm_s4_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                    torch::Tensor scales, torch::Tensor zeros);
void vec_mm_s4(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
               torch::Tensor scales, torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vec_mm_s4_cuda(vec, mat, mul, scales, zeros);
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
  // m.def("matmul_test", (&matmul_test), "Matrix Multiplication (CUDA)");
  m.def("mat4mul",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &>(&matmul_test),
        "Matrix Multiplication (CUDA)");
  m.def("mat4mul",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &>(&matmul_test),
        "Matrix Multiplication (CUDA)");
  m.def("vec_mat4mul", &vec_mm_s4, "Vector Matrix Multiplication (CUDA)");
}
