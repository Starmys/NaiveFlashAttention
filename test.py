import time
import torch
from cutlass_flash_attention import FlashMultiHeadAttention


BATCH, N_HEADS, N_CTX, D_HEAD = 8, 32, 1024, 64


def attention_forward_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    p = torch.einsum(f'bmhk, bnhk -> bhmn', query, key).to(torch.float32) * sm_scale
    p_max = p.max(-1, keepdim=True).values
    p_exp = torch.exp(p - p_max)
    s = p_exp / (p_exp.sum(-1, keepdim=True) + 1e-6)
    out = torch.einsum(f'bhmn, bnhk -> bmhk', s.to(value.dtype), value)
    return out


def profile(fn, mode='fwd', warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    end = time.time()
    latency = (end - start) * 1e3 / rep
    flops_per_matmul = 2. * BATCH * N_HEADS * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    gflops = total_flops / latency * 1e-9
    print(f'{mode}: {latency:.3f} ms | {gflops:.3f} GFLOP/s')


def test_flash_attention(dtype=torch.float16, device="cuda"):
    q = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    do = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    sm_scale = D_HEAD ** -0.5

    cutlass_fmha = FlashMultiHeadAttention(training=True)

    ref_o = attention_forward_reference(q, k, v, sm_scale)
    ref_o.backward(do)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    forward_fn = lambda: cutlass_fmha(q, k, v, sm_scale)
    backward_fn = lambda: o.backward(do, retain_graph=True)
    o = forward_fn()
    backward_fn()
    dv, v.grad = v.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dq, q.grad = q.grad.clone(), None

    atol = {
        torch.float: 5e-4,
        torch.half: 9.5e-2,
    }[dtype]
    rtol = {
        torch.float: 1e-4,
        torch.half: 2e-2,
    }[dtype]
    torch.testing.assert_close(o, ref_o, atol=atol, rtol=rtol)
    torch.testing.assert_close(dq, ref_dq, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk, ref_dk, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv, ref_dv, atol=atol, rtol=rtol)

    forward_flops = profile(forward_fn, f'{dtype} fwd')
    backward_flops = profile(backward_fn, f'{dtype} bwd')
    return forward_flops, backward_flops


torch.manual_seed(2023)
forward_flops, backward_flops = test_flash_attention(torch.float16)
forward_flops, backward_flops = test_flash_attention(torch.float32)
