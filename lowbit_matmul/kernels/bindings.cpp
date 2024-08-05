#include "torch/types.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
// Include all files
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>
#include <gemm.h>

torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * 2; // 4 bits packed into 8-bits
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_host(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), M, N, K,
              C.data_ptr<int32_t>());

  return C;
}

torch::Tensor matmul_hi4_naive(const torch::Tensor &A, const torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1); // 4 bits packed into 8-bits
  auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  matmul_hi4_naive_cu(A.data_ptr<float>(),
                      B.data_ptr<int8_t>(), M, N, K,
                      C.data_ptr<float>());
  // matmul_hi4_naive_cu(reinterpret_cast<float *>(A.data_ptr()),
  //                     B.data_ptr<int8_t>(), M, N, K,
  //                     reinterpret_cast<float *>(C.data_ptr()));

  return C;
}

//====== pybind ======


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def("matmul_hi4", &matmul_hi4_naive, "");
}
