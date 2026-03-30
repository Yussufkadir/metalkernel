//
//  File.metal
//  metalkernel
//
//  Created by Yusuf Surmen on 29/03/2026.
//

#include <metal_stdlib>
using namespace metal;

kernel void vec_add(
                    device const float* A [[ buffer(0) ]],
                    device const float* B [[ buffer(1) ]],
                    device float* C [[ buffer(2) ]],
                    uint id [[ thread_position_in_grid ]])
{
    C[id] = A[id] + B[id];
}


