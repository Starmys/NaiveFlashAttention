# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Callable, Optional

import torch
import numpy as np
from pycuda.driver import function_attribute
import pytest

from flash_attn.flash_attention import FlashAttention

from sparta.kernels import FlashSparseAttentionFP32ForwardKernel, FlashSparseAttentionFP32BackwardKernel, FlashSparseAttentionFP16ForwardKernel, FlashSparseAttentionFP16BackwardKernel
from sparta.testing import block_mask, profile, sparse_multi_head_attention_forward_reference


def prepare_data(
    batch: int = 64,
    Nt: int = 1024,
    Ns: int = 1024,
    heads: int = 12,
    D: int = 64,
    transposed: bool = False,
    granularity: Tuple[int, int] = (16, 64),
    sparsity: float = 0.8,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
    random_seed: int = 2022,
):
    inputs = ['Q', 'K', 'V']
    outputs = ['O']
    shapes = {'Q': (Nt, D), 'K': (Ns, D), 'V': (Ns, D), 'O': (Nt, D)}
    if transposed:
        shapes = {name: (batch, heads, N, D) for name, (N, D) in shapes.items()}
    else:
        shapes = {name: (batch, N, heads, D) for name, (N, D) in shapes.items()}

    torch.manual_seed(random_seed)
    data: Dict[str, torch.Tensor] = {}
    for x in inputs:
        data[f'input_{x}'] = torch.randn(size=shapes[x], dtype=dtype, device='cuda')
    if requires_grad:
        for y in outputs:
            data[f'input_grad_{y}'] = torch.randn(size=shapes[x], dtype=dtype, device='cuda')

    mask = block_mask(
        shape=(Nt, Ns),
        granularity=granularity,
        sparsity=sparsity,
        device='cuda',
    )

    calc_target_data(data, mask, transposed, requires_grad)

    return data, mask


def calc_target_data(
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    transposed: bool,
    requires_grad: bool,
):
    if requires_grad:
        for k, v in data.items():
            if k.startswith('input'):
                v.requires_grad = True

    inputs = [data['input_Q'], data['input_K'], data['input_V'], mask, np.nan, transposed]
    data['target_O'] = sparse_multi_head_attention_forward_reference(*inputs)
    grad = None

    if requires_grad:
        grad = data['input_grad_O']
        data['target_O'].backward(grad)
        data['target_grad_Q'] = data['input_Q'].grad
        data['input_Q'].grad = None
        data['target_grad_K'] = data['input_K'].grad
        data['input_K'].grad = None
        data['target_grad_V'] = data['input_V'].grad
        data['input_V'].grad = None

    profile_kernel(
        forward_func=sparse_multi_head_attention_forward_reference,
        inputs=inputs,
        grad=grad,
        label='Reference',
    )


def profile_kernel(
    forward_func: Callable[[], torch.Tensor],
    inputs: List[torch.Tensor],
    grad: Optional[torch.Tensor] = None,
    backward_func: Optional[Callable] = None,
    output_size: int = 1,
    label: str = '',
):
    forward_latency = profile(forward_func, inputs)
    print(f'Forward Latency ({label}) = {forward_latency}')
    if grad is not None:
        if backward_func is None:
            if output_size == 1:
                def forward_backward():
                    output = forward_func(*inputs)
                    output.backward(grad)
            else:
                def forward_backward():
                    output = forward_func(*inputs)
                    output[0].backward(grad)
        else:
            def forward_backward():
                output = forward_func(*inputs)
                backward_func(grad, output.data, *inputs)
        backward_latency = profile(forward_backward, []) - forward_latency
        print(f'Backward Latency ({label}) = {backward_latency}')


def check_results(data: Dict[str, torch.Tensor]):
    return
    for name, out in data.items():
        if name.startswith('output_'):
            tar = data[name.replace('output', 'target')]
            print(f'==================== Checking {name} ====================')
            if out.dtype is torch.float32:
                rtol, atol = 1e-7, 1e-6
            elif out.dtype is torch.float16:
                rtol, atol = 1e-7, 2e-3
            # torch.testing.assert_close(out, tar, rtol=rtol, atol=atol)
            # for k in range(4):
            #     start = k * 32
            #     end = start + 32
            #     print(start, end)
            #     torch.testing.assert_close(out[:, start:end, :], tar[:, start:end, :], rtol=1e-7, atol=1e-6)
            abs_err = (out - tar).abs().sum()
            print(abs_err)
            if not abs_err < atol * 1024 * 1024:
                # print(out[0, :128:32, 0, :32:4])
                # print(tar[0, :128:32, 0, :32:4])
                print(out[0, :4, 0, :8] / 20)
                print(tar[0, :4, 0, :8])
                # print(out[0, -129:-1:32, -33:-1:4])
                # print(tar[0, -129:-1:32, -33:-1:4])
            # torch.testing.assert_close(out, tar, atol=1e-4, rtol=1e-4, msg=name)


def get_params(dtype: str, direction: str):
    return {
        'float32': {
            'forward': {
                '_impl': 'flash',
                'BLOCK_SIZE_T_VALUE': 64,
                'BLOCK_SIZE_S_VALUE': 64,
                'THREAD_SIZE_T_VALUE': 4,
                'THREAD_SIZE_S_VALUE': 4,
                'THREAD_SIZE_D_VALUE': 4,
            },
            'backward': {
                '_impl': 'flash',
                'BLOCK_SIZE_T_VALUE': 64,
                'BLOCK_SIZE_S_VALUE': 32,
                'THREAD_SIZE_T_VALUE': 4,
                'THREAD_SIZE_S_VALUE': 2,
                'THREAD_SIZE_D_VALUE': 4,
            },
        },
        'float16': {
            'forward': {
                '_impl': 'flash',
                'BLOCK_SIZE_T_VALUE': 64,
                'BLOCK_SIZE_S_VALUE': 128,
                'THREADS_PER_BLOCK': 256,
                'TS_WARP_SIZE_M_VALUE': 16,
                'TD_WARP_SIZE_M_VALUE': 16,
            },
            'backward': {
                '_impl': 'flash',
                'BLOCK_SIZE_T_VALUE': 64,
                'BLOCK_SIZE_S_VALUE': 64,
                'THREADS_PER_BLOCK': 256,
                'TS_WARP_SIZE_M_VALUE': 16,
                'TD_WARP_SIZE_M_VALUE': 16,
                'SD_WARP_SIZE_M_VALUE': 16,
            },
        }
    }[dtype][direction]


def test_sparse_matmul_kernel(
    batch: int = 20,
    heads: int = 12,
    Nt: int = 1024,
    Ns: int = 1024,
    # batch: int = 1,
    # Nt: int = 64,
    # Ns: int = 64,
    D: int = 64,
    transposed: bool = False,
    granularity: Tuple[int, int] = (16, 16),
    sparsity: float = 0,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
):
    data, mask = prepare_data(batch, Nt, Ns, heads, D, transposed, granularity, sparsity, dtype=dtype, requires_grad=requires_grad)

    buffer = torch.zeros(batch * heads * Nt * 2, dtype=torch.float32, device='cuda')
    if dtype is torch.float32:
        forward_kernel = FlashSparseAttentionFP32ForwardKernel(buffer=buffer, transposed=transposed)
    else:
        forward_kernel = FlashSparseAttentionFP16ForwardKernel(buffer=buffer, transposed=transposed)
    forward_kernel.attr.set_mask(mask)
    forward_kernel.compile(get_params(forward_kernel.__dtype__, 'forward'), (batch, Nt, Ns, heads, D))
    if requires_grad:
        if dtype is torch.float32:
            backward_kernel = FlashSparseAttentionFP32BackwardKernel(buffer=buffer, transposed=transposed)
        else:
            backward_kernel = FlashSparseAttentionFP16BackwardKernel(buffer=buffer, transposed=transposed)
        backward_kernel.attr.set_mask(mask)
        backward_kernel.compile(get_params(backward_kernel.__dtype__, 'backward'), (batch, Nt, Ns, heads, D))

    input_data = [data[f'input_{x}'].data for x in ['Q', 'K', 'V']]

    shared = forward_kernel._kernel.get_attribute(function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES)
    print(f'Forward Shared Memory = {shared} Bytes')
    data['output_O'] = forward_kernel(*input_data)
    # print(data['output_O'].shape, kernel.attr.indexes.block_nnz)
    # print('========== mask ==========')
    # print(mask[::16, ::64].sum(-1))
    # print('========== target ==========')
    # print(data['target_O'][0, :8, :8])
    # print('========== output ==========')
    # print(data['output_O'][0, :8, :8])
    if requires_grad:
        shared = backward_kernel._kernel.get_attribute(function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES)
        print(f'Backward Shared Memory = {shared} Bytes')
        grad_Q, grad_K, grad_V = backward_kernel(data['input_grad_O'].data, data['output_O'].data, *input_data)
        data['output_grad_Q'] = grad_Q
        data['output_grad_K'] = grad_K
        data['output_grad_V'] = grad_V
        profile_kernel(
            forward_func=forward_kernel,
            inputs=input_data,
            grad=data['input_grad_O'].data,
            backward_func=backward_kernel,
            label='SparTA-FA',
        )
        # def forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        #     q = q.swapaxes(2, 3).contiguous()
        #     k = k.swapaxes(2, 3).contiguous()
        #     v = v.swapaxes(2, 3).contiguous()
        #     o = forward_kernel(q, k, v)
        #     return o.swapaxes(2, 3).contiguous()
        # def backward_func(grad: torch.Tensor, o: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        #     grad = grad.swapaxes(2, 3).contiguous()
        #     o = o.swapaxes(2, 3).contiguous()
        #     q = q.swapaxes(2, 3).contiguous()
        #     k = k.swapaxes(2, 3).contiguous()
        #     v = v.swapaxes(2, 3).contiguous()
        #     grad_q, grad_k, grad_v = backward_kernel(grad, o, q, k, v)
        #     grad_q = grad_q.swapaxes(2, 3).contiguous()
        #     grad_k = grad_k.swapaxes(2, 3).contiguous()
        #     grad_v = grad_v.swapaxes(2, 3).contiguous()
        #     return grad_q, grad_k, grad_v
        # profile_kernel(
        #     forward_func=forward_func,
        #     inputs=input_data,
        #     grad=data['input_grad_O'].data,
        #     backward_func=backward_func,
        #     label='SparTA-FA',
        # )
    else:
        profile_kernel(
            forward_func=forward_kernel,
            inputs=input_data,
            grad=None,
            backward_func=None,
            label='SparTA-FA',
        )
    check_results(data)

    # print('========== debug ==========')
    # P = torch.mm(data['input_Q'][0], data['input_K'][0].T) / 8.0
    # print(P[0])
    # P1 = P[:, :64]
    # P2 = P[:, 64:]
    # M1 = torch.max(P1, dim=-1).values
    # M2 = torch.max(P2, dim=-1).values
    # M = torch.max(P, dim=-1).values
    # print(M1[0].item(), M2[0].item(), M[0].item())
    # coef_M1 = torch.exp(M1 - M)
    # coef_M2 = torch.exp(M2 - M)
    # S1 = torch.exp(P1 - M1.unsqueeze(-1))
    # S2 = torch.exp(P2 - M2.unsqueeze(-1))
    # S = torch.exp(P - M.unsqueeze(-1))
    # print(S[0])
    # L1 = torch.sum(S1, dim=-1)
    # L2 = torch.sum(S2, dim=-1)
    # L = torch.sum(S, dim=-1)
    # print(L1[0].item(), coef_M1[0].item())
    # print(L2[0].item(), coef_M2[0].item())
    # print(L[0].item())
    # coef_L1 = L1 / L
    # coef_L2 = 1 / L
    # O1 = torch.mm(S1 / L1.unsqueeze(-1), data['input_V'][0, :64, :])
    # O2 = torch.mm(S2, data['input_V'][0, 64:, :])
    # O = torch.mm(S / L.unsqueeze(-1), data['input_V'][0])
    # print(O1[0, :4], coef_M1[0].item(), coef_L1[0].item())
    # print(O2[0, :4], coef_M2[0].item(), coef_L2[0].item())
    # print(O[0, :4])
    # O_hat = O1 * coef_M1.unsqueeze(-1) * coef_L1.unsqueeze(-1) + O2 * coef_M2.unsqueeze(-1) * coef_L2.unsqueeze(-1)
    # print(O_hat[0, :4])

    if dtype is not torch.float16 or transposed:
        return
    qkv = torch.concat([x.unsqueeze(2) for x in input_data], dim=2).detach()
    qkv.requires_grad = True
    flash_attention = FlashAttention()
    profile_kernel(
        forward_func=flash_attention,
        inputs=[qkv],
        grad=data['input_grad_O'],
        output_size=2,
        label='FlashAttn',
    )
    data['output_O'] = flash_attention(qkv)[0]
    data['output_grad_Q'] = qkv.grad[:, :, 0, :, :]
    data['output_grad_K'] = qkv.grad[:, :, 1, :, :]
    data['output_grad_V'] = qkv.grad[:, :, 2, :, :]
    check_results(data)


if __name__ == '__main__':
    print(f'\n******************** FP32 TRANSPOSED ********************')
    test_sparse_matmul_kernel(dtype=torch.float32, requires_grad=True, transposed=True)
    print(f'\n******************** FP32 NOT TRANSPOSED ********************')
    test_sparse_matmul_kernel(dtype=torch.float32, requires_grad=True, transposed=False)
    print(f'\n******************** FP16 TRANSPOSED ********************')
    test_sparse_matmul_kernel(dtype=torch.float16, requires_grad=True, transposed=True)
    print(f'\n******************** FP16 NOT TRANSPOSED ********************')
    test_sparse_matmul_kernel(dtype=torch.float16, requires_grad=True, transposed=False)
