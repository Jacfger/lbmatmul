#pragma once

void matmul_host(const uint8_t *A, const uint8_t *B, uint32_t M,
                 uint32_t N, uint32_t K, int32_t *C);