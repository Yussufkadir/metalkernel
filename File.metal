//
//  File.metal
//  metalkernel
//
//  Created by Yusuf Surmen on 29/03/2026.
//

#include <metal_stdlib>
using namespace metal;

kernel void matvec_naive(
                    device const float* A [[ buffer(0) ]],
                    device const float* x [[ buffer(1) ]],
                    device float* y [[ buffer(2) ]],
                    constant uint& K [[ buffer(3) ]],
                    uint id [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    for (uint j = 0; j < K; j++){
        sum += A[id * K + j] * x[j];
    }
    y[id] = sum;
}

constant uint TILE = 64;

kernel void matvec_tiled(
                         device const float* A [[ buffer(0) ]],
                         device const float* x [[ buffer(1) ]],
                         device float* y [[ buffer(2) ]],
                         constant uint& K [[ buffer(3) ]],
                         threadgroup float* x_tile [[ threadgroup(0) ]],
                         uint id [[ thread_position_in_grid ]],
                         uint local_id [[ thread_position_in_threadgroup ]],
                         uint group_size [[ threads_per_threadgroup ]]
                         )
{
    float sum = 0.0f;
    for (uint tile_start = 0; tile_start < K; tile_start += TILE){
        if (local_id < TILE) {
            uint x_idx = tile_start + local_id;
            x_tile[local_id] = (x_idx < K) ? x[x_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tile_end = min(tile_start + TILE, K);
        for (uint j = tile_start; j < tile_end; j++){
            sum += A[id * K + j] * x_tile[j - tile_start];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[id] = sum;
}


