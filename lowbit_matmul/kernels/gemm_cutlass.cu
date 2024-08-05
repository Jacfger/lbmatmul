#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>
#include <iostream>
#include <stdexcept>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include "cutlass/gemm/gemm_enumerated_types.h"
#include <gemm.h>

void matmul_i4i4_cu(const uint8_t *A, const uint8_t *B, int M, int N, int K,
                    int32_t *C) {
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

  typename Gemm::Arguments arguments{{M, N, K},
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
#ifdef __CUTLASS_FP16S4__
void matmul_hi4_cutlass_cu(const cutlass::half_t *A, const cutlass::int4b_t *B,
                           int M, int N, int K, cutlass::half_t *C) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::int4b_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, cutlass::layout::RowMajor, ElementB,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      4,  // Stages
      8,  // AlignmentA
      32, // AlignmentB
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone>;

  Gemm gemmOp;
  cutlass::gemm::GemmCoord problem_size = {M, N, K};

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     problem_size,
                                     1,      // batch_size
                                     {1, 0}, // alpha, beta
                                     A, // <- reference to matrix A on device
                                     B, // <- reference to matrix B on device
                                     C, // <- reference to matrix C on device
                                     C, // <- reference to matrix D on device
                                     M * K, // size of A
                                     K * N, // size of B
                                     M * N, // size of C
                                     M * N, // size of C
                                     K,
                                     K,
                                     N,
                                     N};

  auto status = gemmOp(arguments);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Runtime error: CUTLASS execution failed:"
              << cutlassGetStatusString(status) << std::endl;
    throw std::runtime_error(cutlassGetStatusString(status));
  }
}
#endif