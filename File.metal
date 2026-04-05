#include <metal_stdlib>
using namespace metal;

kernel void matvec_naive(
    device const float* A  [[ buffer(0) ]],
    device const float* x  [[ buffer(1) ]],
    device float*       y  [[ buffer(2) ]],
    constant uint&      K  [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    for (uint j = 0; j < K; j++) {
        sum += A[id * K + j] * x[j];
    }
    y[id] = sum;
}

constant uint TILE = 64;

kernel void matvec_tiled(
    device const float* A      [[ buffer(0) ]],
    device const float* x      [[ buffer(1) ]],
    device float*       y      [[ buffer(2) ]],
    constant uint&      K      [[ buffer(3) ]],
    threadgroup float*  x_tile [[ threadgroup(0) ]],
    uint id       [[ thread_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    float sum = 0.0f;
    for (uint tile_start = 0; tile_start < K; tile_start += TILE) {
        if (local_id < TILE) {
            uint x_idx = tile_start + local_id;
            x_tile[local_id] = (x_idx < K) ? x[x_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_end = min(tile_start + TILE, K);
        for (uint j = tile_start; j < tile_end; j++) {
            sum += A[id * K + j] * x_tile[j - tile_start];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[id] = sum;
}

kernel void matvec_coalesced(
    device const float* A       [[ buffer(0) ]],
    device const float* x       [[ buffer(1) ]],
    device float*       y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float*  partial [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{

    float sum = 0.0f;
    for (uint j = local_id; j < K; j += gsize) {
        sum += A[row * K + j] * x[j];
    }

    partial[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = gsize / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            partial[local_id] += partial[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        y[row] = partial[0];
    }
}

kernel void matvec_simdgroup(
    device const float* A       [[ buffer(0) ]],
    device const float* x       [[ buffer(1) ]],
    device float*       y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float*  partial [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    const uint SGSIZE = 32;

    float sum = 0.0f;
    for (uint j = local_id; j < K; j += gsize) {
        sum += A[row * K + j] * x[j];
    }

    sum = simd_sum(sum);

    if (local_id % SGSIZE == 0) {
        partial[local_id / SGSIZE] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / SGSIZE;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        y[row] = total;
    }
}

kernel void matvec_float4(
    device const float* A       [[ buffer(0) ]],
    device const float* x       [[ buffer(1) ]],
    device float*       y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float*  partial [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    device const float4* A4 = (device const float4*)(A + row * K);
    device const float4* x4 = (device const float4*)x;
    uint K4 = K / 4;


    float4 acc = float4(0.0f);
    for (uint j = local_id; j < K4; j += gsize) {
        acc += A4[j] * x4[j];
    }

    float sum = acc.x + acc.y + acc.z + acc.w;

    sum = simd_sum(sum);

    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        y[row] = total;
    }
}

kernel void matvec_cached(
    device const float* A       [[ buffer(0) ]],
    device const float* x       [[ buffer(1) ]],
    device float*       y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float4* x_cache [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    device const float4* A4 = (device const float4*)(A + row * K);
    device const float4* x4 = (device const float4*)x;
    uint K4 = K / 4;

    for (uint j = local_id; j < K4; j += gsize) {
        x_cache[j] = x4[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = float4(0.0f);
    for (uint j = local_id; j < K4; j += gsize) {
        acc += A4[j] * x_cache[j];
    }
    float sum = acc.x + acc.y + acc.z + acc.w;

    sum = simd_sum(sum);

    threadgroup float* partial =
        (threadgroup float*)(x_cache + K4);
    
    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        y[row] = total;
    }
}

kernel void matvec_half(
    device const half*  A       [[ buffer(0) ]],
    device const float* x       [[ buffer(1) ]],
    device float*       y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float*  partial [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{

    device const half2* A2 = (device const half2*)(A + row * K);
    uint K2 = K / 2;

    float sum = 0.0f;
    for (uint j = local_id; j < K2; j += gsize) {

        float2 a = float2(A2[j]);
        float2 v = float2(x[j*2], x[j*2 + 1]);
        sum += a.x * v.x + a.y * v.y;
    }

    sum = simd_sum(sum);

    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        y[row] = total;
    }
}
