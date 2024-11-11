# %%
import torch

# import lowbit_matmul.kernels
# impo
import time
import quarot._CUDA

from tqdm import tqdm


def two_compl(x, bits: int):
    # This is NOT two complements, even though original author says so
    # https://github.com/spcl/QuaRot/blob/main/quarot/functional/quantization.py#L4
    return torch.where(x < 0, 1 << bits + x, x)


def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # minq, maxq = get_minq_maxq(4, True)
    # assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    # q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i8 = q.to(dtype=torch.int8) & 0xF
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4.to(torch.uint8)


size = 16
bsz, seq_len = 1, 2048
feature_in, feature_out = (8192, 28672)


def test_matmul_i4i4():

    # benchmark run for a few iteration
    benchmark_iter = 10
    benchmark_time = {
        "i4": [],
        "half": [],
        "float": [],
    }

    for _ in tqdm(range(benchmark_iter)):
        A = torch.randint(-8, 7, (seq_len, feature_in), dtype=torch.int32)
        B = torch.randint(
            -8, 7, (feature_out, feature_in), dtype=torch.int32
        )  # transposed

        A_i4 = pack_i4(A.to(torch.int8))
        B_i4 = pack_i4(B.to(torch.int8))

        perf_counter = time.perf_counter()
        y_i4_i32 = quarot._CUDA.matmul(A_i4.cuda(), B_i4.cuda())
        benchmark_time["i4"] += [time.perf_counter() - perf_counter]

        A_half, B_half = A.half(), B.half().T
        perf_counter = time.perf_counter()
        y_half = A_half.cuda() @ B_half.cuda()
        benchmark_time["half"] += [time.perf_counter() - perf_counter]

        A_float, B_float = A.float(), B.float().T
        perf_counter = time.perf_counter()
        y = A_float.cuda() @ B_float.cuda()
        benchmark_time["float"] += [time.perf_counter() - perf_counter]
        # print(y, y_i4_i32)

        assert torch.all(y_i4_i32 == y)  # half will not yield correct answer
    return benchmark_time

test_matmul_i4i4()
# %%

# %%

benchmark_iter = 1
benchmark_time = {
    "i4": [],
    "half": [],
    "float": [],
}

for _ in tqdm(range(benchmark_iter)):
    A = torch.randint(-8, 7, (seq_len, feature_in), dtype=torch.half)
    B = torch.randint(-8, 7, (feature_out, feature_in), dtype=torch.int8)  # transposed
    # B = torch.eye(feature_out, feature_in, dtype=torch.int8) # transposed

    A_i4 = pack_i4(A.to(torch.int8))
    B_i4 = pack_i4(B.to(torch.int8))

    perf_counter = time.perf_counter()
    # print(A.shape, B_i4.shape)
    y_i4_i32 = lowbit_matmul.kernels.matmul_hi4(A.cuda(), B_i4.cuda())
    benchmark_time["i4"] += [time.perf_counter() - perf_counter]

    A_half, B_half = A.half(), B.half().T
    perf_counter = time.perf_counter()
    y_half = A_half.cuda() @ B_half.cuda()
    benchmark_time["half"] += [time.perf_counter() - perf_counter]

    A_float, B_float = A.float(), B.float().T
    perf_counter = time.perf_counter()
    y = A_float.cuda() @ B_float.cuda()
    benchmark_time["float"] += [time.perf_counter() - perf_counter]

    # assert (y_i4_i32 == y) # half will not yield correct answer
    print((y_i4_i32 != y).sum(), (y_i4_i32 != y).sum() / y.numel())

# benchmark_time = test_matmul_i4i4()
for k, v in benchmark_time.items():
    print(f"{k}: {sum(v)/len(v):.6f} s")
# %%
y_i4_i32[0]
# %%
y[0]
# %%
y_half[0]
# %%
(y_i4_i32[0] != y[0]).sum()
y_i4_i32[0][y_i4_i32[0] != y_half[0]], y[0][y_i4_i32[0] != y_half[0]]

((y_i4_i32.float() - y)).abs().max(), (y_i4_i32.float() - y).abs().min()
# (y_i4_i32.float() - y)
# %%
import torch, time
import quant_ops
from tqdm import tqdm

benchmark_iter = 5
benchmark_time = {
    "i4": [],
    "float": [],
}


def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # minq, maxq = get_minq_maxq(4, True)
    # assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    # q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i8 = q.to(dtype=torch.int8) & 0xF
    q_i4 = q_i8[0::2] | (q_i8[1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4


bsz, seq_len = 1, 2048 
# feature_in, feature_out = (8192, 28672)
# feature_in, feature_out = (12288 * 4, 12288)
feature_in, feature_out = (4 * 4096, 4096)
for _ in tqdm(range(benchmark_iter)):
    type = torch.half
    A = torch.rand(seq_len, feature_in, dtype=type)
    B = torch.randint(0, 16, (feature_in, feature_out), dtype=torch.int8)  # transposed
    # B = torch.eye(feature_out, feature_in, dtype=torch.int8) # transposed
    scales = torch.ones(feature_out, dtype=torch.half).to(type)
    zeros = torch.ones(feature_out, dtype=torch.half).to(type)

    B_i4 = pack_i4(B.to(torch.int8)).to(torch.uint8)

    y_i4_i32 = torch.zeros(seq_len, feature_out, dtype=type).cuda()

    A_float, B_float = A.to(type), (scales * B.to(type) - zeros).to(type)
    perf_counter = time.perf_counter()
    y = A_float.cuda() @ B_float.cuda()
    torch.cuda.synchronize()
    benchmark_time["float"] += [time.perf_counter() - perf_counter]

    print(A.dtype, B_i4.dtype)
    perf_counter = time.perf_counter()
    # print(A.shape, B_i4.shape)
    y_i4_i32 = quant_ops.mat4mul(A.cuda(), B_i4.cuda(), scales.cuda(), zeros.cuda())
    # quant_ops.mat4mul(A.cuda(), B_i4.cuda(), scales.cuda(), zeros.cuda(), y_i4_i32)
    torch.cuda.synchronize()
    benchmark_time["i4"] += [time.perf_counter() - perf_counter]

    # assert (y_i4_i32 == y) # half will not yield correct answer
    # print((y_i4_i32 != y).sum(), (y_i4_i32 != y).sum() / y.numel())
    print(torch.isclose(y_i4_i32, y).all())

# benchmark_time = test_matmul_i4i4()
for k, v in benchmark_time.items():
    print(f"{k}: {sum(v)/len(v):.6f} s")