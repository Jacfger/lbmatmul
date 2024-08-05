# %%
import torch

# import lowbit_matmul.naive
# import lowbit_matmul.cutlass
import lowbit_matmul.kernels

size = 32


# %%
def pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    # minq, maxq = get_minq_maxq(4, True)
    # assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    # q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i8 = q.to(dtype=torch.int8).to(torch.int8) & 0xF
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
    return q_i4


# # %% i4i4
# A = torch.arange(256, dtype=torch.int8).reshape(-1, size)
# # %%
# # fmt:off
# # A = torch.tensor(
# #         [
# #             [ 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 ],
# #             [ 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 ],
# #         ],
# #         dtype=torch.uint8,
# #     ).reshape(2, -1)[:, :size].cuda()
# # fmt:on


# B = torch.eye(size, dtype=torch.int8).cuda()
# B = pack_i4(B.to(torch.int8)).cuda()
# print(A, B)
# print(A.shape, B.shape)
# y = lowbit_matmul.kernels.matmul(pack_i4(A).cuda(), B)
# print(y, y.shape, A.shape, pack_i4(A).shape)

# %%

# A = torch.arange(size, dtype=torch.half).reshape(-1, size)
A = torch.arange(size, dtype=torch.float32).reshape(-1, size)
B = torch.eye(size, dtype=torch.int8).cuda()

y = lowbit_matmul.kernels.matmul_hi4(A.cuda(), pack_i4(B).cuda())
torch.cuda.synchronize()
y, y.shape, A.shape, B.shape
# %%
pack_i4(b)
# %%
