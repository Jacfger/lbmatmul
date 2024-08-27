# %%
import torch
import quant_ops

quant_ops.__dict__

m, k, n = 2048, 4096, 8192


A = torch.rand(m, k).cuda(0)
B = torch.rand(k, n).cuda(0)
A.shape, B.shape

torch.all(quant_ops.matmul_test(A, B) == A @ B)
# %%
import torch
import quant_ops, quant_cuda
import numpy as np

m, k, n = 1, 2048, 4096
# m, k, n = 1, 32, 1
A = torch.ones(m, k, dtype=torch.float32).cuda(0)

B = torch.randint(0, 8, (k, n), dtype=torch.int32).cuda(0)
# B = torch.ones((k, n), dtype=torch.int32).cuda(0)


def pack(intweight):
    intweight = intweight.cpu().numpy().astype(np.uint32)
    qweight = np.zeros(
        (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
    )
    i = 0
    row = 0
    while row < qweight.shape[0]:
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i))
        i += 10
        qweight[row] |= intweight[i] << 30
        row += 1
        qweight[row] |= (intweight[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 1)
        i += 10
        qweight[row] |= intweight[i] << 31
        row += 1
        qweight[row] |= (intweight[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 2)
        i += 10
        row += 1

    return qweight


B_q = torch.tensor(pack(B), dtype=torch.int32).cuda(0)
scale = torch.rand(n, dtype=torch.float32).cuda(0)
zero = torch.rand(n, dtype=torch.float32).cuda(0)
# %%

C = torch.zeros(m, n, dtype=torch.double).cuda(0)
quant_cuda.vecquant3matmul(A.double(), B_q, C, scale.double(), zero.double())

C_ref = A.double() @ (scale.double() * B.double() - zero.double())

print("Error upper bound:", (C_ref - C).max(), "lower bound:", (C_ref - C).min())
# %%
# # DEBUG PACK IS WOKRING AS EXPECTED
for i in B_q[:3, 0].to(torch.uint32):
    print(f"{i:032b}")
for idx, i in enumerate(B[:32, 0].to(torch.uint32).cpu().numpy() & 0x7):
    if idx % 10 == 0:
        print()
    print(f"{i & 0x7:03b}", end=",")
# %%
import torch
import quant_ops

quant_ops.__dict__

# m, k, n = 2048, 4096, 8192
m, k, n = 2048, 4096, 8192
n = 128
A_type = torch.float
# A = torch.rand(m, k, dtype=A_type).cuda(0)
A = torch.eye(32).cuda(0)
# B = torch.randint(0, 16, (k, n)).cuda(0)
B = torch.arange(0, 16, dtype=torch.int8).cuda(0).view(16)
B = torch.stack([B] * n, dim=1)
B = torch.cat([B, B], dim=0)
# B[8:16] = 0x7F
A.shape, B.shape


def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # minq, maxq = get_minq_maxq(4, True)
    # assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    # q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i8 = q.to(dtype=torch.int8).to(torch.int8) & 0xF
    # q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    q_i4 = q_i8[0::2] | (q_i8[1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4


scales = torch.ones(n, dtype=torch.float32).cuda(0)
zeros = torch.zeros(n, dtype=torch.float32).cuda(0)
B_q = pack_i4(B).to(torch.uint8)
# %%
C = quant_ops.mat4mul(A.clone(), B_q.clone(), scales.clone(), zeros.clone())
A.shape, B_q.shape, scales.shape, zeros.shape, C.shape
# C = quant_ops.matmul_test(A.clone(), B_q.clone())
# %%
# C, A @ B.to(A_type)
C[:, 0], C[:, 1]
# %%
torch.all((C == (A @ B.to(A_type))))
# %%
B_q[0, 0]
# # DEBUG PACK IS WOKRING AS EXPECTED
for i in B_q[:1, 0].to(torch.uint32):
    print(f"{i:032b}")
for idx, i in enumerate(B[:2, 0].to(torch.uint32).cpu().numpy() & 0xF):
    if idx % 10 == 0:
        print()
    print(f"{i & 0x7:04b}", end=",")
# %%

import torch
import quant_ops

quant_ops.__dict__

# m, k, n = 2048, 4096, 8192
m, k, n = 2048, 4096, 8192

A_type = torch.half
A = torch.rand(m, k, dtype=A_type).cuda(0)
B = torch.randint(0, 16, (k, n)).cuda(0)


def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    q_i8 = q.to(dtype=torch.int8).to(torch.int8) & 0xF
    q_i4 = q_i8[0::2] | (q_i8[1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4


scales = torch.rand(n, dtype=A_type).cuda(0)
zeros = torch.rand(n, dtype=A_type).cuda(0)
B_q = pack_i4(B).to(torch.uint8)
# %%
C = quant_ops.mat4mul(A, B_q, scales, zeros)
C
# %%
C_ref = A @ (scales * B.to(A_type) - zeros)
C_ref
# %%
(C == C_ref).all(), (C - C_ref).min(), (C - C_ref).max()
# %%
import torch
import quant_ops


def pack_i4_64(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # assert q.shape[0] % 16 == 0, "The first dimension should be multiple of 16"
    q_i8 = q.to(dtype=torch.int64) & 0xF
    q_i4 = (
        q_i8[0::8]
        | (q_i8[1::8] << 4)
        | (q_i8[2::8] << 8)
        | (q_i8[3::8] << 12)
        | (q_i8[4::8] << 16)
        | (q_i8[5::8] << 20)
        | (q_i8[6::8] << 24)
        | (q_i8[7::8] << 28)
        # | (q_i8[8::16] << 32)
        # | (q_i8[9::16] << 36)
        # | (q_i8[10::16] << 40)
        # | (q_i8[11::16] << 44)
        # | (q_i8[12::16] << 48)
        # | (q_i8[13::16] << 52)
        # | (q_i8[14::16] << 56)
        # | (q_i8[15::16] << 60)
    )
    return q_i4
M, K, N = 1, 256, 256
# B = torch.randint(0, 16, (K, N), dtype=torch.int64)
B = torch.arange(0, N, dtype=torch.int32) & 0xF
B = torch.stack([B] * K, dim=0)
B_q = pack_i4_64(B).to(torch.int32).cuda()
B.shape , B_q.shape
# %%
mul = torch.zeros(1, N, dtype=torch.float32).cuda()
vec = torch.ones(K, dtype=torch.float32).cuda()
scales = torch.ones(K, dtype=torch.float32).cuda()
zeros = torch.zeros(K, dtype=torch.float32).cuda()
torch.cuda.synchronize()
quant_ops.vec_mat4mul(vec.unsqueeze(0), B_q, mul, scales, zeros)
mul, vec @ B.to(torch.float32).cuda()

# %%
m, k, n = 1, 12288 * 2, 12288
# m, k, n = 1, 32, 1
A = torch.ones(m, k, dtype=torch.float32).cuda(0)

B = torch.randint(0, 16, (k, n), dtype=torch.int32).cuda(0)
# B = torch.ones((k, n), dtype=torch.int32).cuda(0)
def pack_i4_64(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # assert q.shape[0] % 16 == 0, "The first dimension should be multiple of 16"
    q_i8 = q.to(dtype=torch.int64) & 0xF
    q_i4 = (
        q_i8[0::8]
        | (q_i8[1::8] << 4)
        | (q_i8[2::8] << 8)
        | (q_i8[3::8] << 12)
        | (q_i8[4::8] << 16)
        | (q_i8[5::8] << 20)
        | (q_i8[6::8] << 24)
        | (q_i8[7::8] << 28)
        # | (q_i8[8::16] << 32)
        # | (q_i8[9::16] << 36)
        # | (q_i8[10::16] << 40)
        # | (q_i8[11::16] << 44)
        # | (q_i8[12::16] << 48)
        # | (q_i8[13::16] << 52)
        # | (q_i8[14::16] << 56)
        # | (q_i8[15::16] << 60)
    )
    return q_i4


B_q = pack_i4_64(B).to(torch.int32).cuda(0)
scale = torch.ones(n, dtype=torch.float32).cuda(0)
zero = torch.zeros(n, dtype=torch.float32).cuda(0)

C = torch.zeros(m, n, dtype=torch.double).cuda(0)
quant_ops.vec_mat4mul(A.double(), B_q, C, scale.double(), zero.double())

C_ref = A.double() @ (scale.double() * B.double() - zero.double())

print("Error upper bound:", (C_ref - C).max().item(), "lower bound:", (C_ref - C).min().item())
C, C_ref
# %%
