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
    id<MTLFunction> func = [library newFunctionWithName:@"matvec_naive"];
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    const uint M = 2048;
    const uint K = 2048;
    
    std::vector<float> A(M * K), x(K), y_gpu(M), y_cpu(M);
    for (uint i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (uint i = 0; i < K; i++) x[i] = (float)rand() / RAND_MAX;
    
    for (uint i = 0; i < M; i++){
        float sum = 0.0f;
        for (uint j = 0; j < K; j++) sum += A[i * K + j] * x[j];
        y_cpu[i] = sum;
    }
    
    id<MTLBuffer> bufA = [device newBufferWithBytes:A.data() length: M * K * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufX = [device newBufferWithBytes:x.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufY = [device newBufferWithLength: M * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [device newBufferWithBytes:&K length:sizeof(uint) options:MTLResourceStorageModeShared];
    
    for (int run = 0; run < 6; run++){
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        [enc setBuffer:bufK offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake(M, 1, 1);
        MTLSize group = MTLSizeMake(256, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int REPS = 100;
    for (int run = 0; run < REPS; run++){
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        [enc setBuffer:bufK offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake(M, 1, 1);
        MTLSize group = MTLSizeMake(256, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / REPS;
    
    float* result = (float*)bufY.contents;
    float max_err = 0.0f;
    for (uint i = 0; i < M; i++){
        max_err = fmaxf(max_err, fabsf(result[i] - y_cpu[i]));
    }
    
    std::cout << "Matrix: " << M << " x " << K << "\n";
    std::cout << "Max error vs CPU: " << max_err << "\n";
    std::cout << "Time per matvec:  " << ms << " ms\n";

    double bytes = (M * K + K + M) * sizeof(float);
    double gbs   = (bytes / 1e9) / (ms / 1000.0);
    std::cout << "Bandwidth:        " << gbs << " GB/s\n";
    std::cout << "M1 Pro peak:      200 GB/s\n";
    std::cout << "Utilization:      " << (gbs / 200.0) * 100.0 << "%\n";

    return 0;
    }
