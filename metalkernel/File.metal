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


    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint j = local_id; j < K4; j += gsize * 2) {
        acc0 += A4[j] * x4[j];
        if(j + gsize < K4){
            acc1 += A4[j + gsize] * x4[j + gsize];
        }
    }

    float sum = acc0.x + acc0.y + acc0.z + acc0.w
               + acc1.x + acc1.y + acc1.z + acc1.w;

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
    constant uint&      TILE    [[ buffer(4) ]],
    threadgroup float*  x_cache [[ threadgroup(0) ]],
    uint id       [[ thread_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    device const float* row = A + id * K;
    float sum = 0.0f;

    for (uint tile_start = 0; tile_start < K; tile_start += TILE) {
        for (uint t = local_id; t < TILE; t += gsize) {
            uint x_idx = tile_start + t;
            x_cache[t] = (x_idx < K) ? x[x_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_len = min(TILE, K - tile_start);
        for (uint offset = 0; offset < tile_len; offset++) {
            sum += row[tile_start + offset] * x_cache[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    y[id] = sum;
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

    device const half4* A4 = (device const half4*)(A + row * K);
    device const float4* x4 = (device const float4*)x;
    uint K4 = K / 4;

    float sum = 0.0f;
    for (uint j = local_id; j < K4; j += gsize) {

        float4 a = float4(A4[j]);
        float4 v = x4[j];
        sum += dot(a, v);
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
//kernel void broken_sum(
//    device const float* x  [[ buffer(0) ]],
//    device float*       y  [[ buffer(1) ]],
//    constant uint&      N  [[ buffer(2) ]],
//    threadgroup float*  partial [[ threadgroup(0) ]],
//    uint local_id [[ thread_position_in_threadgroup ]],
//    uint gsize    [[ threads_per_threadgroup ]])
//{
//    float sum = 0.0f;
//    for (uint i = local_id; i < N; i += gsize) {
//        sum += x[i];
//    }
//
//    partial[local_id] = sum;
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//    
//    for (uint stride = gsize / 2; stride > 0; stride /= 2) {
//        if (local_id < stride) {
//            partial[local_id] += partial[local_id + stride];
//        }
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//    }
//
//    if (local_id == 0) {
//        y[0] = partial[0];
//    }
//}

kernel void matvec_gate_up_fused(
    device const half*  gate_weight [[ buffer(0) ]],
    device const half*  up_weight   [[ buffer(1) ]],
    device const float* x           [[ buffer(2) ]],
    device float*       gate_out    [[ buffer(3) ]],
    device float*       up_out      [[ buffer(4) ]],
    constant uint&      K           [[ buffer(5) ]],
    threadgroup float*  partial     [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]], 
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    device const half4* gate4 = (device const half4*)(gate_weight + row * K);
    device const half4* up4   = (device const half4*)(up_weight   + row * K);
    device const float4* x4   = (device const float4*)x;
    uint K4 = K / 4;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (uint j = local_id; j < K4; j += gsize * 2) {
        float4 g = float4(gate4[j]);
        float4 v = x4[j];
        gate_sum += dot(g, v);

        float4 u = float4(up4[j]);
        up_sum += dot(u, v);

        uint j2 = j + gsize;
        if (j2 < K4) {
            float4 g2 = float4(gate4[j2]);
            float4 v2 = x4[j2];
            gate_sum += dot(g2, v2);

            float4 u2 = float4(up4[j2]);
            up_sum += dot(u2, v2);
        }
    }

    gate_sum = simd_sum(gate_sum);
    if (local_id % 32 == 0) {
        partial[(local_id / 32) * 2] = gate_sum;
    }

    up_sum = simd_sum(up_sum);
    if (local_id % 32 == 0) {
        partial[(local_id / 32) * 2 + 1] = up_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        float total_gate = 0.0f;
        float total_up   = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) {
            total_gate += partial[i * 2];
            total_up   += partial[i * 2 + 1];
        }
        gate_out[row] = total_gate;
        up_out[row]   = total_up;
    }
}

kernel void matvec_down_fused(
                              device const half* down_weight [[ buffer(0) ]],
                              device const float* x [[ buffer(1) ]],
                              device float* y [[ buffer(2) ]],
                              constant uint& K [[buffer(3) ]],
                              threadgroup float* partial [[ threadgroup(0) ]],
                              uint row [[ threadgroup_position_in_grid ]],
                              uint local_id [[ thread_position_in_threadgroup ]],
                              uint gsize [[ threads_per_threadgroup ]]
                              )
{
    device const half4* down4 = (device const half4*)(down_weight + row * K);
    device const float4* x4 = (device const float4*)x;
    uint K4 = K / 4;
    
    float sum = 0.0f;
    for (uint j = local_id; j < K4; j += gsize){
        float4 d = float4(down4[j]);
        float4 v = x4[j];
        sum += dot(d, v);
    }
    
    sum = simd_sum(sum);
    if (local_id % 32 == 0){
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (local_id == 0){
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        y[row] = total;
    }
}
