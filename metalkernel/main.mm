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
    NSLog(@"Max threadgroup memory: %lu bytes",
          (unsigned long)device.maxThreadgroupMemoryLength);

    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) { NSLog(@"failed to load library"); return 1; }

    NSError* error = nil;

    id<MTLFunction> func_naive =
        [library newFunctionWithName:@"matvec_naive"];
    id<MTLFunction> func_tiled =
        [library newFunctionWithName:@"matvec_tiled"];
    id<MTLFunction> func_coalesced =
        [library newFunctionWithName:@"matvec_coalesced"];
    id<MTLFunction> func_simdgroup =
        [library newFunctionWithName:@"matvec_simdgroup"];
    id<MTLFunction> func_float4 =
        [library newFunctionWithName:@"matvec_float4"];
    id<MTLFunction> func_cached =
        [library newFunctionWithName:@"matvec_cached"];
    id<MTLFunction> func_half =
        [library newFunctionWithName:@"matvec_half"];

    id<MTLComputePipelineState> pipeline_naive =
        [device newComputePipelineStateWithFunction:func_naive
                                             error:&error];
    id<MTLComputePipelineState> pipeline_tiled =
        [device newComputePipelineStateWithFunction:func_tiled
                                             error:&error];
    id<MTLComputePipelineState> pipeline_coalesced =
        [device newComputePipelineStateWithFunction:func_coalesced
                                             error:&error];
    id<MTLComputePipelineState> pipeline_simdgroup =
        [device newComputePipelineStateWithFunction:func_simdgroup
                                             error:&error];
    id<MTLComputePipelineState> pipeline_float4 =
        [device newComputePipelineStateWithFunction:func_float4
                                             error:&error];
    id<MTLComputePipelineState> pipeline_cached =
        [device newComputePipelineStateWithFunction:func_cached
                                             error:&error];
    id<MTLComputePipelineState> pipeline_half =
        [device newComputePipelineStateWithFunction:func_half
                                             error:&error];

    if (!pipeline_naive)     { NSLog(@"ERROR: matvec_naive missing");     return 1; }
    if (!pipeline_tiled)     { NSLog(@"ERROR: matvec_tiled missing");     return 1; }
    if (!pipeline_coalesced) { NSLog(@"ERROR: matvec_coalesced missing"); return 1; }
    if (!pipeline_simdgroup) { NSLog(@"ERROR: matvec_simdgroup missing"); return 1; }
    if (!pipeline_float4)    { NSLog(@"ERROR: matvec_float4 missing");    return 1; }
    if (!pipeline_cached)    { NSLog(@"ERROR: matvec_cached missing");    return 1; }
    if (!pipeline_half)      { NSLog(@"ERROR: matvec_half missing");      return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    const uint M          = 8192;
    const uint K          = 8192;
    const uint TILE       = 64;
    const uint GROUP_SIZE = 256;
    const int  REPS       = 100;
    const size_t simd_partial_bytes = (GROUP_SIZE / 32) * sizeof(float);
    const size_t max_tg_bytes = device.maxThreadgroupMemoryLength;
    const std::vector<uint> cached_tiles = {64, 128, 256, 512, 1024};

    std::vector<float> A(M * K), x(K), y_cpu(M);
    for (uint i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (uint i = 0; i < K;     i++) x[i] = (float)rand() / RAND_MAX;

    for (uint i = 0; i < M; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < K; j++) sum += A[i * K + j] * x[j];
        y_cpu[i] = sum;
    }

    std::vector<uint16_t> A_half(M * K);
    for (uint i = 0; i < M * K; i++) {
        A_half[i] = (uint16_t)(__fp16)A[i];
    }
    
    id<MTLBuffer> bufA = [device newBufferWithBytes:A.data()
                          length:M * K * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufA_half = [device newBufferWithBytes:A_half.data()
                               length:M * K * sizeof(uint16_t)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufX = [device newBufferWithBytes:x.data()
                          length:K * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufY = [device newBufferWithLength:M * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [device newBufferWithBytes:&K
                          length:sizeof(uint)
                          options:MTLResourceStorageModeShared];

    auto print_results = [&](NSString* name, double ms,
                              double bytes_moved) {
        double gbs = (bytes_moved / 1e9) / (ms / 1000.0);
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

    auto benchmark = [&](id<MTLComputePipelineState> pipeline,
                         NSString* name,
                         int mode,
                         size_t tg_memory_bytes,
                         uint runtime_tile = 0) {

        auto run_once = [&]() {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufX offset:0 atIndex:1];
            [enc setBuffer:bufY offset:0 atIndex:2];
            [enc setBuffer:bufK offset:0 atIndex:3];
            if (runtime_tile > 0) {
                [enc setBytes:&runtime_tile
                       length:sizeof(runtime_tile)
                      atIndex:4];
            }
            if (tg_memory_bytes > 0) {
                [enc setThreadgroupMemoryLength:tg_memory_bytes
                                       atIndex:0];
            }
            MTLSize grid  = MTLSizeMake(M, 1, 1);
            MTLSize group = MTLSizeMake(GROUP_SIZE, 1, 1);
            if (mode == 0) {
                [enc dispatchThreads:grid threadsPerThreadgroup:group];
            } else {
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int r = 0; r < 5; r++) run_once();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; r++) run_once();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double,
                       std::milli>(t1 - t0).count() / REPS;
        double bytes = (double)(M * K + K + M) * sizeof(float);
        print_results(name, ms, bytes);
    };

    auto benchmark_half = [&]() {
        auto run_once = [&]() {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_half];
            [enc setBuffer:bufA_half offset:0 atIndex:0];
            [enc setBuffer:bufX      offset:0 atIndex:1];
            [enc setBuffer:bufY      offset:0 atIndex:2];
            [enc setBuffer:bufK      offset:0 atIndex:3];
            [enc setThreadgroupMemoryLength:simd_partial_bytes atIndex:0];
            MTLSize grid  = MTLSizeMake(M, 1, 1);
            MTLSize group = MTLSizeMake(GROUP_SIZE, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int r = 0; r < 5; r++) run_once();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; r++) run_once();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double,
                    std::milli>(t1 - t0).count() / REPS;
        double bytes = (double)M * K * sizeof(uint16_t)
                     + (double)K * sizeof(float)
                     + (double)M * sizeof(float);
        print_results(@"matvec_half", ms, bytes);
    };

    benchmark(pipeline_naive,
              @"matvec_naive",
              0,
              0);

    benchmark(pipeline_tiled,
              @"matvec_tiled",
              0,
              TILE * sizeof(float));

    benchmark(pipeline_coalesced,
              @"matvec_coalesced",
              1,
              GROUP_SIZE * sizeof(float));

    benchmark(pipeline_simdgroup,
              @"matvec_simdgroup",
              1,
              simd_partial_bytes);

    benchmark(pipeline_float4,
              @"matvec_float4",
              1,
              simd_partial_bytes);

    for (uint tile : cached_tiles) {
        size_t cached_tg_bytes = tile * sizeof(float);
        if (cached_tg_bytes <= max_tg_bytes) {
            NSString* cached_name =
                [NSString stringWithFormat:@"matvec_cached_tile_%u", tile];

            benchmark(pipeline_cached,
                      cached_name,
                      0,
                      cached_tg_bytes,
                      tile);
        } else {
            NSLog(@"\n--- matvec_cached_tile_%u ---", tile);
            NSLog(@"  skipped: requires %zu bytes of threadgroup memory, "
                  @"device limit is %zu", cached_tg_bytes, max_tg_bytes);
        }
    }

    benchmark_half();

    return 0;
}
