#pragma clang language objective-c++

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal not supported\n";
        return 1;
    }
    std::cout << "GPU: " << device.name.UTF8String << "\n";

    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        std::cerr << "Failed to load Metal library\n";
        return 1;
    }

    id<MTLFunction> func = [library newFunctionWithName:@"vec_add"];
    if (!func) {
        std::cerr << "Failed to find vec_add kernel\n";
        return 1;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:func
                                             error:&error];
    if (!pipeline) {
        std::cerr << "Pipeline error: "
                  << error.localizedDescription.UTF8String << "\n";
        return 1;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    const int N = 1024;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }
    
    id<MTLBuffer> bufA = [device newBufferWithBytes:A.data()
                                            length:N * sizeof(float)
                                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufB = [device newBufferWithBytes:B.data()
                                            length:N * sizeof(float)
                                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufC = [device newBufferWithLength:N * sizeof(float)
                                            options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufA offset:0 atIndex:0];
    [encoder setBuffer:bufB offset:0 atIndex:1];
    [encoder setBuffer:bufC offset:0 atIndex:2];


    MTLSize gridSize  = MTLSizeMake(N, 1, 1);
    MTLSize groupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize
       threadsPerThreadgroup:groupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted]; 

    float* result = (float*)bufC.contents;
    std::cout << "C[0]   = " << result[0]   << "  expected 0\n";
    std::cout << "C[1]   = " << result[1]   << "  expected 3\n";
    std::cout << "C[100] = " << result[100] << "  expected 300\n";
    std::cout << "C[512] = " << result[512] << "  expected 1536\n";

    return 0;
}
