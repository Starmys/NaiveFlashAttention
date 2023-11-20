// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda.h>

__device__ __forceinline__ float2 _add_float2(float2 x, float2 y) \
{                                                                 \
    float2 res;                                                   \
    res.x = x.x + y.x;                                            \
    res.y = x.y + y.y;                                            \
    return res;                                                   \
}

__device__ __forceinline__ float2 _scale_float2(float2 x, float y) \
{                                                                  \
    float2 res;                                                    \
    res.x = x.x * y;                                               \
    res.y = x.y * y;                                               \
    return res;                                                    \
}

__device__ __forceinline__ float4 _add_float4(float4 x, float4 y) \
{                                                                 \
    float4 res;                                                   \
    res.x = x.x + y.x;                                            \
    res.y = x.y + y.y;                                            \
    res.z = x.z + y.z;                                            \
    res.w = x.w + y.w;                                            \
    return res;                                                   \
}

__device__ __forceinline__ float4 _scale_float4(float4 x, float y) \
{                                                                  \
    float4 res;                                                    \
    res.x = x.x * y;                                               \
    res.y = x.y * y;                                               \
    res.z = x.z * y;                                               \
    res.w = x.w * y;                                               \
    return res;                                                    \
}

template<int BT, int BS, int D, int TT, int TS, int TD> // D / Td >= Bs / Ts
__global__ void BLOCK_SPARSE_FLASH_ATTENTION_FP32(
    float* Q,
    float* K,
    float* V,
    float* O,
    float* ML,
    uint Ns,
    uint Nt
) {
    const int WARP_REDUCE_SIZE = BS / TS; // <= 32
    const int THREADS_PER_BLOCK = WARP_REDUCE_SIZE * BT;
    const int SD = D / WARP_REDUCE_SIZE;

    const int SMEM_THREADS_D = D / 4;
    const int SMEM_THREADS_N = {{ THREADS_PER_BLOCK }} / SMEM_THREADS_D;

    int H = gridDim.x;
    int HEAD_IDX = (blockIdx.z * H + blockIdx.y);
    Q += blockIdx.z * Nt * H * D + blockIdx.y * D;
    K += blockIdx.z * Ns * H * D + blockIdx.y * D;
    V += blockIdx.z * Ns * H * D + blockIdx.y * D;
    O += blockIdx.z * Nt * H * D + blockIdx.y * D;
    int stride = H * D;
    ML += Nt * 2 * HEAD_IDX;

    uint WARP_OFFSET = (threadIdx.y % (32 / WARP_REDUCE_SIZE)) * WARP_REDUCE_SIZE;
    uint WARP_MASK = ((1 << WARP_REDUCE_SIZE) - 1) << WARP_OFFSET;

    extern __shared__ float shared[];
    float* shared_Q = &shared[0];
    float* shared_K = &shared_Q[BT * D];
    float* shared_V = &shared_K[BS * D];
    float* shared_O = shared_K;
    float* shared_ML = shared_Q;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int SMEM_TID_N = tid / SMEM_THREADS_D;
    int SMEM_TID_D = tid % SMEM_THREADS_D * 4;

    float4 tmp_float4;
    float frag_Q[TT][TD];
    float frag_O[TT][TD];
    float frag_KV[TS][TD];
    float frag_P[TT][TS];
    float frag_ML[TT];

    float temperature = __frsqrt_rn((float)D);
    float row_max;
    float row_sum;
    float row_sum_new;
    float seg_max;
    float seg_sum;
    float row_coef;
    float seg_coef;

    int row_offset = blockIdx.x * BT;

    // Load Q
    #pragma unroll
    for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
        *((float4*)(&shared_Q[k * D + SMEM_TID_D])) =
            *((float4*)(&Q[(row_offset + k) * stride + SMEM_TID_D]));
    }

    for (int col_offset = 0; col_offset < Ns; col_offset += BS) {

        // Load K
        #pragma unroll
        for (int k = SMEM_TID_N; k < BS; k += SMEM_THREADS_N) {
            tmp_float4 = ((float4*)(&K[(col_offset + k) * stride + SMEM_TID_D]))[0];
            shared_K[(SMEM_TID_D+0) * BS + k] = tmp_float4.x;
            shared_K[(SMEM_TID_D+1) * BS + k] = tmp_float4.y;
            shared_K[(SMEM_TID_D+2) * BS + k] = tmp_float4.z;
            shared_K[(SMEM_TID_D+3) * BS + k] = tmp_float4.w;
        }
        // Load V
        #pragma unroll
        for (int k = SMEM_TID_N; k < BS; k += SMEM_THREADS_N) {
            *((float4*)(&shared_V[k * D + SMEM_TID_D])) =
                *((float4*)(&V[(col_offset + k) * stride + SMEM_TID_D]));
        }
        __syncthreads();

        #pragma unroll
        for (int js = 0; js < TS; js++) {
            #pragma unroll
            for (int jt = 0; jt < TT; jt++) {
                frag_P[jt][js] = 0;
            }
        }

        // Calc P = Q K^T
        #pragma unroll
        for (int k = 0; k < D; k += TD) {
            #pragma unroll
            for (int i = 0; i < TD; i++) {
                #pragma unroll
                for (int jt = 0; jt < TT; jt++) {
                    frag_Q[jt][i] = shared_Q[(threadIdx.y + blockDim.y * jt) * D + k + i];
                }
            }
            #pragma unroll
            for (int i = 0; i < TD; i++) {
                #pragma unroll
                for (int js = 0; js < TS; js++) {
                    frag_KV[js][i] = shared_K[(k + i) * BS + threadIdx.x + blockDim.x * js];
                }
            }
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                #pragma unroll
                for (int jt = 0; jt < TT; jt++) {
                    #pragma unroll
                    for (int i = 0; i < TD; i++) {
                        frag_P[jt][js] += frag_Q[jt][i] * frag_KV[js][i];
                    }
                }
            }
        }
        __syncthreads();

        // Load M, L
        #pragma unroll
        for (int jt = tid; jt < BT * 2; jt += THREADS_PER_BLOCK) {
            shared_ML[jt] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int jt = 0; jt < TT; jt++) {
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                frag_P[jt][js] *= temperature;
            }
            // Calc M~ = max_j(P)
            seg_max = -100000.0;
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                seg_max = max(seg_max, frag_P[jt][js]);
            }
            #pragma unroll
            for (int offset = WARP_REDUCE_SIZE / 2; offset > 0; offset >>= 1) {
                seg_max = max(seg_max, __shfl_xor_sync(WARP_MASK, seg_max, offset));
            }
            // Calc S = exp(P - M~)
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                frag_P[jt][js] = expf(frag_P[jt][js] - seg_max);
            }
            // Calc L~ = sum_j(P)
            seg_sum = 0.0f;
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                seg_sum += frag_P[jt][js];
            }
            #pragma unroll
            for (int offset = WARP_REDUCE_SIZE / 2; offset > 0; offset >>= 1) {
                seg_sum += __shfl_down_sync(WARP_MASK, seg_sum, offset);
            }
            // Calc M' = max(M, M~), L' = exp(M - M') * L + exp(M~ - M') * L~
            if (threadIdx.x == 0) {
                int block_row_idx = (threadIdx.y + blockDim.y * jt) * 2;
                row_max = shared_ML[block_row_idx];
                row_sum = shared_ML[block_row_idx + 1];
                if (row_max < seg_max) {
                    shared_ML[block_row_idx] = seg_max;
                    row_coef = expf(row_max - seg_max);
                    row_sum_new = row_coef * row_sum + seg_sum;
                    row_coef *= row_sum / row_sum_new;
                    seg_coef = 1.0f / row_sum_new;
                } else {
                    seg_coef = expf(seg_max - row_max);
                    row_sum_new = row_sum + seg_coef * seg_sum;
                    row_coef = row_sum / row_sum_new;
                    seg_coef /= row_sum_new;
                }
                shared_ML[block_row_idx + 1] = row_sum_new;
            }
            row_coef = __shfl_sync(WARP_MASK, row_coef, WARP_OFFSET);
            seg_coef = __shfl_sync(WARP_MASK, seg_coef, WARP_OFFSET);
            // Calc O' = L / L' * exp(M - M') * O, S' = exp(M~ - M') / L' * S
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                frag_P[jt][js] *= seg_coef;
            }
            frag_ML[jt] = row_coef;
        }
        __syncthreads();

        // Save M, L
        #pragma unroll
        for (int jt = tid * 2; jt < BT; jt += THREADS_PER_BLOCK) {
            *((float4*)(&ML[(row_offset + jt) * 2])) = *((float4*)(&shared_ML[jt * 2]));
        }
        __syncthreads();

        // Calc O = O' + S' V
        #pragma unroll
        for (int kk = 0, k = threadIdx.x * TD; kk < D; k = (k + TD) % D, kk += TD) {
            #pragma unroll
            for (int jt = 0; jt < TT; jt++) {
                #pragma unroll
                for (int i = 0; i < TD; i++) {
                    frag_O[jt][i] *= frag_ML[jt];
                }
            }
            #pragma unroll
            for (int js = 0; js < TS; js++) {
                if (TD == 1) {
                    frag_KV[js][0] = shared_V[(threadIdx.x + blockDim.x * js) * D + k];
                } else if (TD == 2) {
                    *((float2*)(&frag_KV[js][0])) = *((float2*)(&shared_V[(threadIdx.x + blockDim.x * js) * D + k]));
                } else {
                    #pragma unroll
                    for (int i = 0; i < TD; i += 4) {
                        *((float4*)(&frag_KV[js][i])) = *((float4*)(&shared_V[(threadIdx.x + blockDim.x * js) * D + k + i]));
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < TD; i++) {
                #pragma unroll
                for (int jt = 0; jt < TT; jt++) {
                    #pragma unroll
                    for (int js = 0; js < TS; js++) {
                        frag_O[jt][i] += frag_P[jt][js] * frag_KV[js][i];
                    }
                }
            }
            #pragma unroll
            for (int jt = 0; jt < TT; jt++) {
                if (TD == 1) {
                    shared_Q[(threadIdx.y + blockDim.y * jt) * D + k] += frag_O[jt][0];
                } else if (TD == 2) {
                    ((float2*)(&shared_Q[(threadIdx.y + blockDim.y * jt) * D + k]))[0] =
                        _add_float2(
                            ((float2*)(&shared_Q[(threadIdx.y + blockDim.y * jt) * D + k]))[0],
                            ((float2*)(&frag_O[jt][0]))[0]
                        );
                } else {
                    #pragma unroll
                    for (int i = 0; i < TD; i += 4) {
                        ((float4*)(&shared_Q[(threadIdx.y + blockDim.y * jt) * D + k + i]))[0] =
                            _add_float4(
                                ((float4*)(&shared_Q[(threadIdx.y + blockDim.y * jt) * D + k + i]))[0],
                                ((float4*)(&frag_O[jt][i]))[0]
                            );
                    }
                }
            }
            __syncthreads();
        }

    }

    // Save O
    #pragma unroll
    for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
        *((float4*)(&O[(row_idx * BT + k) * stride + SMEM_TID_D])) =
            *((float4*)(&shared_Q[k * D + SMEM_TID_D]));
    }
}
