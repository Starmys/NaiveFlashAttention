// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include "torch/extension.h"


std::vector<at::Tensor> fmha_forward(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  float scale,
  bool calc_lse
);


std::vector<at::Tensor> fmha_backward(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  at::Tensor dO, // B, Nt, H, D
  at::Tensor lse, // B, H, Nt
  at::Tensor delta, // B, H, Nt
  float scale
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fmha_forward, "FHMA forward function");
  m.def("backward", &fmha_backward, "FHMA backward function");
}
