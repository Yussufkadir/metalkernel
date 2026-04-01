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


