# %%
import torch
import time
import torch.nn as nn
import numpy as np

print(f"torch version {torch.version.cuda}")
import quant_cuda

from packing import pack, unpack
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


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


import quant_cuda


## This is the mask kernel part
def generate_input():
    ## create random binary mask, shape: 4096 x 11008
    mask = torch.randint(0, 2, (4096, 11008), dtype=torch.uint8, device="cuda")

    ## create two low rank matrices
    A = torch.rand((1536, 11008), dtype=torch.float16, device="cuda")
    B = torch.rand((4096, 1536), dtype=torch.float16, device="cuda")
    return mask, A, B


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
quant_cuda._nonmul_packedmask_2d(weight_packedmask, pack_mask)
torch.cuda.synchronize()
# print(weight)
# print(weight_packedmask)
assert (weight_packedmask == weight).all(), "Panick, two masks operation are not equal"


# %% # This is the FP16 x INT4 part
import marlin

DEV = "cuda"
M = 10
K = 4 * 4096
N = 4096

layer = nn.Linear(K, N)
MAT = torch.randn(10, K, dtype=torch.half).to(DEV)

quantizer = Quantizer()
quantizer.configure(4, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = marlin.Layer(layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print("Simu:", layer(MAT))
    print("Kern:", qlayer(MAT.half()), "(faster)")

# %% # This is the INT4 X INT4 part

DEV = "cuda"

# A = torch.rand((1536, 11008), dtype=torch.float16, device="cuda")
A = torch.rand((11008, 1536), dtype=torch.float16, device="cuda")
B = torch.rand((4096, 1536), dtype=torch.float16, device="cuda")

quantizerA = Quantizer()
quantizerA.configure(4, perchannel=True, sym=False, mse=False)
quantizerA.find_params(A.T, weight=True)
A = quantize(A, quantizerA.scale, quantizerA.zero, quantizerA.maxq)

quantizerB = Quantizer()
quantizerB.configure(4, perchannel=True, sym=False, mse=False)
quantizerB.find_params(B, weight=True)
B = quantize(B, quantizerB.scale, quantizerB.zero, quantizerB.maxq)

### INT4 B * INT4 A here (A is transpose)


## This is the mask kernel + INT4 X INT4 part

### INT4 A * INT4 B first

### Then masking
