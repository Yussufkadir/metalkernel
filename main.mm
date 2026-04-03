#pragma clang language objective-c++

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"GPU: %@", device.name);

    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) { NSLog(@"failed to load library"); return 1; }

    NSError* error = nil;

    id<MTLFunction> func_naive =
        [library newFunctionWithName:@"matvec_naive"];
    id<MTLFunction> func_tiled =
        [library newFunctionWithName:@"matvec_tiled"];
    id<MTLFunction> func_coalesced =
        [library newFunctionWithName:@"matvec_coalesced"];

    id<MTLComputePipelineState> pipeline_naive =
        [device newComputePipelineStateWithFunction:func_naive
                                             error:&error];
    id<MTLComputePipelineState> pipeline_tiled =
        [device newComputePipelineStateWithFunction:func_tiled
                                             error:&error];
    id<MTLComputePipelineState> pipeline_coalesced =
        [device newComputePipelineStateWithFunction:func_coalesced
                                             error:&error];

    id<MTLCommandQueue> queue = [device newCommandQueue];

    const uint M          = 2048;
    const uint K          = 2048;
    const uint TILE       = 64;
    const uint GROUP_SIZE = 256;

    std::vector<float> A(M * K), x(K), y_cpu(M);
    for (uint i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (uint i = 0; i < K;     i++) x[i] = (float)rand() / RAND_MAX;

    for (uint i = 0; i < M; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < K; j++) sum += A[i * K + j] * x[j];
        y_cpu[i] = sum;
    }

    id<MTLBuffer> bufA = [device newBufferWithBytes:A.data()
                          length:M * K * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufX = [device newBufferWithBytes:x.data()
                          length:K * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufY = [device newBufferWithLength:M * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [device newBufferWithBytes:&K
                          length:sizeof(uint)
                          options:MTLResourceStorageModeShared];

    auto benchmark = [&](id<MTLComputePipelineState> pipeline,
                         NSString* name,
                         int mode) {

        auto run_once = [&]() {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufX offset:0 atIndex:1];
            [enc setBuffer:bufY offset:0 atIndex:2];
            [enc setBuffer:bufK offset:0 atIndex:3];

            MTLSize grid  = MTLSizeMake(M, 1, 1);
            MTLSize group = MTLSizeMake(GROUP_SIZE, 1, 1);

            if (mode == 0) {
                if (mode == 0 && pipeline == pipeline_tiled) {
                    [enc setThreadgroupMemoryLength:TILE * sizeof(float)
                                           atIndex:0];
                }
                [enc dispatchThreads:grid
                    threadsPerThreadgroup:group];

            } else {
                [enc setThreadgroupMemoryLength:GROUP_SIZE * sizeof(float)
                                       atIndex:0];
                [enc dispatchThreadgroups:grid
                     threadsPerThreadgroup:group];
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int r = 0; r < 5; r++) run_once();

        const int REPS = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; r++) run_once();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double,
                       std::milli>(t1 - t0).count() / REPS;
        double bytes = (double)(M * K + K + M) * sizeof(float);
        double gbs   = (bytes / 1e9) / (ms / 1000.0);

        float* result = (float*)bufY.contents;
        float max_err = 0.0f;
        for (uint i = 0; i < M; i++)
            max_err = fmaxf(max_err, fabsf(result[i] - y_cpu[i]));

        NSLog(@"\n--- %@ ---", name);
        NSLog(@"  time:        %.4f ms",  ms);
        NSLog(@"  bandwidth:   %.1f GB/s", gbs);
        NSLog(@"  utilization: %.1f%%",   (gbs / 200.0) * 100.0);
        NSLog(@"  max error:   %f",       max_err);
    };

    benchmark(pipeline_naive,     @"matvec_naive",     0);
    benchmark(pipeline_tiled,     @"matvec_tiled",     0);
    benchmark(pipeline_coalesced, @"matvec_coalesced", 1);

    return 0;
}
