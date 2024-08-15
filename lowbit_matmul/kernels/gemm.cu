#include "cutlass/layout/matrix.h"
#include "gemm.h"
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <stdexcept>

void matmul_host(const uint8_t *A, const uint8_t *B, int32_t M, int32_t N,
                 int32_t K, int32_t *C) {
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

  typename Gemm::Arguments arguments{{M, N, K},
                                     {(cutlass::int4b_t *)A, K},
                                     {(cutlass::int4b_t *)B, N},
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

// inline __device__ int8_t __lower_i4(int8_t x) {
//   if (x & 0x08)
//     return x | 0xF0;
//   else
//     return x & 0x0F;
// }

// inline __device__ int8_t __upper_i4(int8_t x) { return x >> 4; }
