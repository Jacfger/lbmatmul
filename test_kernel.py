# %%
import torch
import time
import torch.nn as nn
import numpy as np
import einops

print(f"torch version {torch.version.cuda}")
import quant_ops

from packing import pack, unpack
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import quant_ops


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    return torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    # return scale * (q - zero)


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


import quant_ops


## This is the mask kernel part
def generate_input():
    ## create random binary mask, shape: 4096 x 11008
    mask = torch.randint(0, 2, (4096, 11008), dtype=torch.uint8, device="cuda")

    ## create two low rank matrices
    A = torch.rand((1536, 11008), dtype=torch.float16, device="cuda")
    B = torch.rand((4096, 1536), dtype=torch.float16, device="cuda")
    return mask, A, B


print(f"{'Only masking':=^80}")

mask, A, B = generate_input()

tick = time.perf_counter()
pack_mask = pack(mask, 8, 1, by_rows=True, device="cuda")
pack_mask = pack_mask.to(dtype=torch.uint8)

unpack_mask = unpack(pack_mask, data_size=1, by_rows=True, device="cuda")
weight = (B @ A) * unpack_mask
torch.cuda.synchronize()

tick = time.perf_counter()
pack_mask = pack(mask, 8, 1, by_rows=True, device="cuda")
pack_mask = pack_mask.to(dtype=torch.uint8)

weight_packedmask = B @ A
quant_ops._nonmul_packedmask_2d(weight_packedmask, pack_mask)
torch.cuda.synchronize()
print("Kern:", weight_packedmask)
print("Simu:", weight)
assert (weight_packedmask == weight).all(), "Panick, two masks operation are not equal"


# %% # This is the FP16 x INT4 part

print(f"{'FP16 x INT4':=^80}")


class qLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "qweight",
            torch.empty((self.in_features // 2, self.out_features), dtype=torch.uint8),
        )
        self.register_buffer("scales", torch.empty(self.out_features, dtype=torch.half))
        self.register_buffer("zeroes", torch.empty(self.out_features, dtype=torch.half))

    def weight(self):
        return self.scales * self.unpack_i4(self.qweight).float() - self.zeroes

    @staticmethod
    def pack_i4(q):
        # assume dim 0
        # assert torch.is_signed(q), "The tensor to be packed should be signed int"
        q_i8 = q.to(dtype=torch.int8) & 0xF
        q_i4 = q_i8[0::2] | (q_i8[1::2] << 4)  # [a, b] -> 0x{b:x}{a:x}
        return q_i4

    @staticmethod
    def unpack_i4(q):
        # assume dim 0
        q = q.to(dtype=torch.int8)
        return einops.rearrange(
            torch.stack([q & 0xF, (q >> 4) & 0xF], dim=0), "b k nq -> (k b) nq"
        )

    def forward(self, x):
        return quant_ops.mat4mul(x, self.qweight, self.scales, self.zeroes)


DEV = "cuda"
M = 2048
K = 4 * 4096
N = 4096

layer = nn.Linear(K, N, dtype=torch.half).cuda()
MAT = torch.rand(M, K, dtype=torch.half).to(DEV)

quantizer = Quantizer()
quantizer.configure(4, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)

import marlin
# qlayer = qLayer(K, N)
qlayer = marlin.Layer(K, N)

qweight = quantize(layer.weight, quantizer.scale, quantizer.zero, quantizer.maxq).T

qlayer.qweight = qLayer.pack_i4(qweight).to(torch.uint8).contiguous()
qlayer.scales = quantizer.scale.flatten().half()
qlayer.zeroes = (quantizer.zero * quantizer.scale).flatten().half()
qlayer = qlayer.to(DEV)


with torch.no_grad():
    fp16i4_tick = time.perf_counter()
    print("Kern:", qlayer(MAT))
    # torch.cuda.synchronize()
    fp16i4_tock = time.perf_counter()
    fp16fp16_tick = time.perf_counter()
    print("Simu:", layer(MAT))
    fp16fp16_tock = time.perf_counter()

    print("Kernel time:", fp16i4_tock - fp16i4_tick)
    print("Simulation time:", fp16fp16_tock - fp16fp16_tick)
# %%
# This is the INT4 X INT4 part

import quarot

print(f"{'INT4 x INT4':=^80}")

# require add `find_packages` in setup.py in QuaRot
DEV = "cuda"

A = torch.rand((1536, 11008), dtype=torch.float16, device="cuda")
B = torch.rand((4096, 11008), dtype=torch.float16, device="cuda")
BT = torch.rand((1536, 4096), dtype=torch.float16, device="cuda")
mask = torch.randint(0, 2, (11008, 4096), dtype=torch.uint8, device="cuda")
x = torch.rand((100, 1536), dtype=torch.float16, device="cuda")
weight = torch.rand((1536, 4096), dtype=torch.float16, device="cuda")

quantizer = quarot.nn.Quantizer()
Ap = quantizer(A)
Bp = quantizer(B)
pack_mask = pack(mask.clone(), 8, 1, by_rows=True, device="cuda")
pack_mask = pack_mask.to(dtype=torch.uint8)

# A @ B.T (11008, 4096)
kern_tick = time.perf_counter()
# quarot.matmul
i4i4 = quarot._CUDA.matmul(Ap.quantized_x, Bp.quantized_x)
print("i4i4 multiplication", time.perf_counter() - kern_tick)
full_tick = time.perf_counter()
c_check = A @ B.T
print("full prec", time.perf_counter() - full_tick)
print(c_check.shape, i4i4.shape)
C = quarot.sym_dequant(
    i4i4, Ap.scales_x, Bp.scales_x
)
quant_ops._nonmul_packedmask_2d(C, pack_mask) # C is modified in place
torch.cuda.synchronize()
kern_tick = time.perf_counter()
y_q = x @ C
kern_tock = time.perf_counter()

C_ref = A.float() @ B.float().T
print("Kern:", C, C.shape)
print("Simu:", C_ref, C_ref.shape)


# %%

print(f"{'Masking':=^80}")
# Kernel

# Simulation
pack_mask = pack(mask.clone(), 8, 1, by_rows=True, device="cuda")
pack_mask = pack_mask.to(dtype=torch.uint8)
unpack_mask = unpack(pack_mask, data_size=1, by_rows=True, device="cuda")
weight_ref = C_ref * unpack_mask

print("Kern:", C)
print("Simu:", weight_ref)
# print(C, weight_ref, sep="\n")
print("Max diff:", (C - weight_ref).max(), "Min diff:", (C - weight_ref).min())

# %%
fullprec_tick = time.perf_counter()
y = x @ weight
fullprec_tock = time.perf_counter()
print("i4i4m time", kern_tock - kern_tick)
print("Full precision time:", fullprec_tock - fullprec_tick)