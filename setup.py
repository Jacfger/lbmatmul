from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib
import subprocess

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent


if __name__ == "__main__":
    setup(
        name="quant_ops",
        ext_modules=[
            CUDAExtension(
                name="quant_ops",
                sources=[
                    "lowbit_matmul/kernels/quant_cuda.cpp",
                    "lowbit_matmul/kernels/quant_cuda_kernel.cu",
                    "lowbit_matmul/kernels/h_s4_gemm.cu",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
