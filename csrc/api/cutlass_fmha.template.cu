// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include <vector>

#include <cuda_fp16.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/fast_math.h"

#include "kernel_forward.h"
#include "kernel_backward.h"


using ArchTag = cutlass::arch::Sm##;


template <typename data_t, typename scalar_t, typename ArchTag>
std::vector<at::Tensor> fmha_forward_(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  float scale,
  bool calc_lse
) {
  cudaSetDevice(Q.get_device());
  int B = Q.size(0);
  int Ns = K.size(1);
  int Nt = Q.size(1);
  int H = Q.size(2);
  int D = Q.size(3);
  auto opts = Q.options();
  at::Tensor O = torch::zeros_like(Q, opts);

  static constexpr int kMaxK = 64; // <- Decrease to 32/16 if your problem is smaller
  static int const kQueriesPerBlock = 64;
  static int const kKeysPerBlock = 64;

  using ForwardKernel = AttentionKernel<
    scalar_t,             // scalar_t
    ArchTag,              // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK,
    false,                // Supports dropout
    false                 // Supports bias
  >;

  typename ForwardKernel::Params p;
  p.query_ptr = (scalar_t*)(Q.data_ptr<data_t>());
  p.key_ptr = (scalar_t*)(K.data_ptr<data_t>());
  p.value_ptr = (scalar_t*)(V.data_ptr<data_t>());
  p.logsumexp_ptr = nullptr;
  p.output_accum_ptr = nullptr;
  if (ForwardKernel::kNeedsOutputAccumulatorBuffer) {
    cudaMalloc(&p.output_accum_ptr, B * H * Nt * D * sizeof(typename ForwardKernel::output_accum_t));
  }
  p.output_ptr = (scalar_t*)(O.data_ptr<data_t>());

  p.scale = scale;
  p.num_heads = H;
  p.num_batches = B;
  p.head_dim = D;
  p.head_dim_value = D;
  p.num_queries = Nt;
  p.num_keys = Ns;

  p.q_strideH = D;
  p.k_strideH = D;
  p.v_strideH = D;
  p.q_strideM = p.q_strideH * H;
  p.k_strideM = p.k_strideH * H;
  p.v_strideM = p.v_strideH * H;
  p.q_strideB = p.q_strideM * Nt;
  p.k_strideB = p.k_strideM * Ns;
  p.v_strideB = p.v_strideM * Ns;
  p.o_strideM = p.head_dim_value * p.num_heads;

  std::vector<at::Tensor> outputs = {O};
  if (calc_lse) {
    at::Tensor lse = torch::empty({B, H, Nt}, opts.dtype(at::kFloat));
    p.logsumexp_ptr = lse.data_ptr<float>();
    outputs.push_back(lse);
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<ForwardKernel>;

  int smem_bytes = sizeof(typename ForwardKernel::SharedStorage);
  if (smem_bytes > 0xc000) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }

  if (!ForwardKernel::check_supported(p)) {
    std::cerr << "Kernel does not support these inputs" << std::endl;
    return outputs;
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess)  {
    std::cerr << "Kernel execution error: " << cudaGetErrorString(result);
  }

  return outputs;
}


template <typename data_t, typename scalar_t, typename ArchTag>
std::vector<at::Tensor> fmha_backward_(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  at::Tensor dO, // B, Nt, H, D
  at::Tensor lse, // B, H, Nt
  at::Tensor delta, // B, H, Nt
  float scale
) {
  cudaSetDevice(Q.get_device());
  int B = Q.size(0);
  int Ns = K.size(1);
  int Nt = Q.size(1);
  int H = Q.size(2);
  int D = Q.size(3);
  at::Tensor dQ = torch::empty_like(Q, Q.options());
  at::Tensor dK = torch::empty_like(K, K.options());
  at::Tensor dV = torch::empty_like(V, V.options());

  static constexpr int kMaxK = 64;
  static constexpr bool kSupports64x128 =
      ArchTag::kMinComputeCapability >= 80 ||
      (ArchTag::kMinComputeCapability >= 70 &&
      cutlass::sizeof_bits<scalar_t>::value <= 16);
  static constexpr int kBlockSizeI = kSupports64x128 && kMaxK > 64 ? 128 : 64;
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;
  static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
  static constexpr bool kPreload = kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF;
  static constexpr int kBlockSizeJ = kPreload && kMaxK > 64 ? 128 : 64;

  using BackwardKernel = AttentionBackwardKernel<
    ArchTag,
    scalar_t,
    true,        // kIsAligned_
    false,       // kApplyDropout_
    false,       // kPreload_
    kBlockSizeI, // kBlockSizeI_,
    kBlockSizeJ, // kBlockSizeJ_,
    kMaxK,       // kMaxK
    false,       // kKeysQueriesAlignedToBlockSize
    true         // kEnableSplitKeys
  >;

  typename BackwardKernel::Params p;
  p.query_ptr = (scalar_t*)(Q.data_ptr<data_t>());
  p.key_ptr = (scalar_t*)(K.data_ptr<data_t>());
  p.value_ptr = (scalar_t*)(V.data_ptr<data_t>());
  p.output_ptr = (scalar_t*)(O.data_ptr<data_t>());
  p.logsumexp_ptr = lse.data_ptr<float>();
  p.delta_ptr = delta.data_ptr<float>();
  p.grad_output_ptr = (scalar_t*)(dO.data_ptr<data_t>());
  p.grad_query_ptr = (scalar_t*)(dQ.data_ptr<data_t>());
  p.grad_key_ptr = (scalar_t*)(dK.data_ptr<data_t>());
  p.grad_value_ptr = (scalar_t*)(dV.data_ptr<data_t>());

  p.scale = scale;
  p.num_heads = H;
  p.num_batches = B;
  p.head_dim = D;
  p.head_dim_value = D;
  p.num_queries = Nt;
  p.num_keys = Ns;

  p.q_strideH = D;
  p.k_strideH = D;
  p.v_strideH = D;
  p.o_strideH = D;
  p.gQ_strideH = D;
  p.gK_strideH = D;
  p.gV_strideH = D;
  p.gO_strideH = D;

  p.q_strideM = p.q_strideH * H;
  p.k_strideM = p.k_strideH * H;
  p.v_strideM = p.v_strideH * H;
  p.gO_strideM = p.gO_strideH * H;

  p.gQKV_strideM_multiplier = 1;
  p.q_strideB = p.q_strideM * Nt;
  p.k_strideB = p.k_strideM * Ns;
  p.v_strideB = p.v_strideM * Ns;
  p.o_strideB = p.o_strideM() * Nt;
  p.gQ_strideB = p.gQ_strideM() * Nt;
  p.gK_strideB = p.gK_strideM() * Ns;
  p.gV_strideB = p.gV_strideM() * Ns;
  p.gO_strideB = p.gO_strideM * Nt;

  p.lse_strideH = Nt;
  p.delta_strideH = Nt;
  p.lse_strideB = p.lse_strideH * H;
  p.delta_strideB = p.delta_strideH * H;

  p.num_splits_key = Ns / 64;

  if (p.workspace_size()) {
      cudaMalloc(&p.workspace, p.workspace_size());
  }

  std::vector<at::Tensor> outputs = {dQ, dK, dV};

  auto kernel_fn = attention_kernel_backward_batched_impl<BackwardKernel>;

  int smem_bytes = sizeof(typename BackwardKernel::SharedStorage);
  if (smem_bytes > 0xc000) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }

  if (!BackwardKernel::check_supported(p)) {
    std::cerr << "Kernel does not support these inputs" << std::endl;
    return outputs;
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

  // Wait for completion
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess)  {
    std::cerr << "Kernel execution error: " << cudaGetErrorString(result);
  }

  return outputs;
}



std::vector<at::Tensor> fmha_forward(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  float scale,
  bool calc_lse
) {
  if (Q.dtype() == torch::kFloat32) {
    return fmha_forward_<float, float, ArchTag>(Q, K, V, scale, calc_lse);
  } else {
    return fmha_forward_<c10::Half, cutlass::half_t, ArchTag>(Q, K, V, scale, calc_lse);
  }
}


std::vector<at::Tensor> fmha_backward(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  at::Tensor dO, // B, Nt, H, D
  at::Tensor lse, // B, H, Nt
  at::Tensor delta, // B, H, Nt
  float scale
) {
  if (Q.dtype() == torch::kFloat32) {
    return fmha_backward_<float, float, ArchTag>(Q, K, V, O, dO, lse, delta, scale);
  } else {
    return fmha_backward_<c10::Half, cutlass::half_t, ArchTag>(Q, K, V, O, dO, lse, delta, scale);
  }
}
