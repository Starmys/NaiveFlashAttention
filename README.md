## Prerequsites
- PyTorch
- NVCC >= 11.3

## Installation
```bash
git submodule update
cd csrc/cutlass && git checkout v3.1.0
cd ../.. && python setup.py install
```

## Quick Start
```python
import torch
from cutlass_flash_attention import FlashMultiHeadAttention

BATCH, N_HEADS, N_CTX, D_HEAD = 8, 32, 1024, 64
dtype = torch.float32
device = 'cuda'

q = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
scale = D_HEAD ** -0.5

fmha = FlashMultiHeadAttention()

o = cutlass_fmha(q, k, v, scale)
```
