import os
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability()
with open('csrc/api/cutlass_fmha.template.cu') as f:
    cu_template = f.read()
os.makedirs('csrc/api/build', exist_ok=True)
with open('csrc/api/build/cutlass_fmha.cu', 'w') as f:
    f.write(cu_template.replace('##', f'{CC_MAJOR}{CC_MINOR}'))

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

ext_modules.append(
    CUDAExtension(
        name="cutlass_fmha_cpp",
        sources=[
            "csrc/api/fmha_api.cpp",
            "csrc/api/build/cutlass_fmha.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo"
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / 'csrc' / 'cutlass' / 'include',
            Path(this_dir) / 'csrc' / 'cutlass' / 'tools' / 'library' / 'include',
            Path(this_dir) / 'csrc' / 'cutlass' / 'tools' / 'util' / 'include',
            Path(this_dir) / 'csrc' / 'cutlass' / 'examples' / '41_fused_multi_head_attention',
        ],
    )
)

setup(
    name="cutlass_flash_attention",
    version="1.0",
    packages=find_packages(
        include=("cutlass_flash_attention"),
    ),
    author="Chengruidong Zhang",
    author_email="chengzhang@microsoft.com",
    description="Cutlass Flash Multi-Head Attention Python API",
    url="https://github.com/Starmys/CutlassFlashAttention",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
    ],
)
