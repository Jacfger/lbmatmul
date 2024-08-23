#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

#include <stdexcept>

// Non-multiplicative masking
template <typename T>
__global__ void _nonmul_packedmask_cuda_kernel(T* __restrict__ mat,
                                               const uint8_t* __restrict__ mask,
                                               int length);

void _nonmul_packedmask_cuda(torch::Tensor& mat, torch::Tensor mask) {
    int length = mat.size(0);
    int threadsPerBlock =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    dim3 blocksPerGrid((length + threadsPerBlock - 1) / threadsPerBlock);
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, mat.scalar_type(), "nonmulmask_cuda_", ([&] {
            _nonmul_packedmask_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                mat.data_ptr<scalar_t>(), mask.data_ptr<uint8_t>(), length);
        }));
}

template <typename T>
__global__ void _nonmul_packedmask_cuda_kernel(T* __restrict__ mat,
                                               const uint8_t* __restrict__ mask,
                                               int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;  // Early exit if out of bounds
    unsigned int block_size = blockDim.x * gridDim.x;

    // In my very rough testing, it looks like having the for loop here doens't
    // matter very much, even though it will loop once only. I guess the extra
    // NOP from branch diverging doesn't matter very much
#pragma unroll
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length;
         idx += block_size) {
        int mask_idx = idx >> 3;  // Index into the mask array
        int bit_position = 7 - (idx & 0x7);
        uint8_t _m = mask[mask_idx];  // Safely access mask
        bool ismasked = (_m >> bit_position) & 1;
        if (!ismasked) mat[idx] = 0;
    }
}

// Non-multiplicative masking
template <typename T>
__global__ void _nonmul_packed4mask_cuda_kernel(
    T* __restrict__ mat, const uint32_t* __restrict__ mask, int length);

void _nonmul_packed4mask_cuda(torch::Tensor& mat, torch::Tensor mask) {
    int length = mat.size(0);
    int threadsPerBlock =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    dim3 blocksPerGrid((length + threadsPerBlock - 1) / threadsPerBlock);
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, mat.scalar_type(), "nonmulmask_cuda_", ([&] {
            _nonmul_packed4mask_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                mat.data_ptr<scalar_t>(), mask.data_ptr<uint32_t>(), length);
        }));
}

template <typename T>
__global__ void _nonmul_packed4mask_cuda_kernel(
    T* __restrict__ mat, const uint32_t* __restrict__ mask, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;  // Early exit if out of bounds
    unsigned int block_size = blockDim.x * gridDim.x;

    // In my very rough testing, it looks like having the for loop here doens't
    // matter very much, even though it will loop once only. I guess the extra
    // NOP from branch diverging doesn't matter very much
#pragma unroll
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length;
         idx += block_size) {
        int mask_idx = idx >> 5;  // Index into the mask array
        int bit_position = 31 - (idx & 0x1F);
        uint32_t _m = mask[mask_idx];  // Safely access mask
        bool ismasked = (_m >> bit_position) & 1;
        if (!ismasked) mat[idx] = 0;
    }
}

template <typename T>
__global__ void _nonmul_packedmask_2d_cuda_kernel(
    T* __restrict__ mat, const uint8_t* __restrict__ mask, int rows, int cols);

void _nonmul_packedmask_2d_cuda(torch::Tensor& mat, torch::Tensor mask) {
    int rows = mat.size(0);
    int cols = mat.size(1);
    int threadsPerBlock =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int x_threads = threadsPerBlock / 8;
    dim3 blocksPerGrid((cols + x_threads - 1) / x_threads, rows / 8);
    dim3 blockSize(threadsPerBlock / 8, 8);
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, mat.scalar_type(), "_nonmul_packedmask_2d", ([&] {
            _nonmul_packedmask_2d_cuda_kernel<<<blocksPerGrid, blockSize>>>(
                mat.data_ptr<scalar_t>(), mask.data_ptr<uint8_t>(), rows, cols);
        }));
}

template <typename T>
__global__ void _nonmul_packedmask_2d_cuda_kernel(
    T* __restrict__ mat, const uint8_t* __restrict__ mask, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cols || row >= rows) return;  // Early exit if out of bounds

    int idx = row * cols + col;
    int mask_idx = (row >> 3) * cols + col;
    // int bit_position = 7 - (threadIdx.y & 0x7);
    int bit_position = 7 - (row & 0x7);

    bool ismasked = (mask[mask_idx] >> bit_position) & 1;
    // printf("mat[%d, %d: %d]=%f, mask[%d]=%u, bit_position=%d, ismasked=%d\n",
    //        row, col, idx, (float)mat[idx], mask_idx, mask[mask_idx],
    //        bit_position, ismasked);

    if (!ismasked) mat[idx] = 0;
}