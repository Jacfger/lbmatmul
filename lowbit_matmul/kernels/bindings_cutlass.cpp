#include "cutlass/half.h"
#include "cutlass/integer_subbyte.h"
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
// Include all files
#include <gemm.h>

torch::Tensor matmul_i4i4(const torch::Tensor &A, const torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1) * 2; // 4 bits packed into 8-bits
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_i4i4_cutlass_cu(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), M, N, K,
                 C.data_ptr<int32_t>());

  return C;
}

#ifdef __CUTLASS_FP16S4__
torch::Tensor matmul_hi4(const torch::Tensor &A, const torch::Tensor &B) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1) * 2; // 4 bits packed into 8-bits
  auto C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));

  matmul_hi4_cutlass_cu(A.data_ptr<cutlass::half_t>(),
                        B.data_ptr<cutlass::int4b_t>(), M, N, K,
                        C.data_ptr<cutlass::half_t>());

  return C;
}
#endif
//====== pybind ======

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_i4i4", &matmul_i4i4,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

#ifdef __CUTLASS_FP16S4__
  m.def("matmul_hi4", &matmul_hi4,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));
#endif
}
