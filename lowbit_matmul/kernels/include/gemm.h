#ifndef __GEMM_H__
#define __GEMM_H__
#include <cuda_fp16.h>
#include <stdint.h>

void matmul_host(const uint8_t *A, const uint8_t *B, uint32_t M,
                 uint32_t N, uint32_t K, int32_t *C);

#endif // __GEMM_H__