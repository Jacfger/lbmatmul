# %%
import torch
import lowbit_matmul.kernels
import time

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
    q_i8 = q.to(dtype=torch.int8).to(torch.uint8) & 0xF
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4


size = 16
bsz, seq_len = 1, 2048
feature_in, feature_out = (8192, 28672)

# benchmark run for a few iteration
benchmark_iter = 10
benchmark_time = {
    "i4": [],
    "half": [],
    "float": [],
}

for _ in tqdm(range(benchmark_iter)):

    A = torch.randint(-8, 7, (seq_len, feature_out), dtype=torch.int32)
    B = torch.randint(-8, 7, (feature_in, feature_out), dtype=torch.int32)

    A_i4 = pack_i4(A.to(torch.int8))
    B_i4 = pack_i4(B.to(torch.int8))

    perf_counter = time.perf_counter()
    y_i4_i32 = lowbit_matmul.kernels.matmul(A_i4.cuda(), B_i4.cuda())
    benchmark_time["i4"] += [time.perf_counter() - perf_counter]

    A_half, B_half = A.half(), B.half().T
    perf_counter = time.perf_counter()
    y_half = A_half.cuda() @ B_half.cuda()
    benchmark_time["half"] += [time.perf_counter() - perf_counter]

    A_float, B_float = A.float(), B.float().T
    perf_counter = time.perf_counter()
    y = A_float.cuda() @ B_float.cuda()
    benchmark_time["float"] += [time.perf_counter() - perf_counter]

    assert torch.all(y_i4_i32 == y) # half will not yield correct answer
# %%
for k, v in benchmark_time.items():
    print(f"{k}: {sum(v)/len(v):.6f} s")