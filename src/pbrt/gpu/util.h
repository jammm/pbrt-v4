// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_UTIL_H
#define PBRT_GPU_UTIL_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/progressreporter.h>

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef NVTX
#ifdef UNICODE
#undef UNICODE
#endif
#include <nvtx3/nvToolsExt.h>

#ifdef RGB
#undef RGB
#endif  // RGB
#endif

#define CUDA_CHECK(EXPR)                                        \
    if (EXPR != hipSuccess) {                                   \
        hipError_t error = hipGetLastError();                   \
        LOG_FATAL("CUDA error: %s", hipGetErrorString(error));  \
    } else /* eat semicolon */

namespace pbrt {

std::pair<hipEvent_t, hipEvent_t> GetProfilerEvents(const char *description);

template <typename F>
inline int GetBlockSize(const char *description, F kernel) {
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    blockSize = 64;
    //CUDA_CHECK(
    //    hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

// GPU Launch Function Declarations
template <typename F>
void GPUParallelFor(const char *description, int nItems, F func);

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;

    int blockSize = GetBlockSize(description, kernel);
    std::pair<hipEvent_t, hipEvent_t> events = GetProfilerEvents(description);

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif
    hipEventRecord(events.first);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
    hipEventRecord(events.second);

#ifdef PBRT_DEBUG_BUILD
    CUDA_CHECK(hipDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

#endif  // __NVCC__ || __HIPCC__

// GPU Synchronization Function Declarations
void GPUWait();

void ReportKernelStats();

void GPUInit();
void GPUThreadInit();

void GPUMemset(void *ptr, int byte, size_t bytes);

void GPURegisterThread(const char *name);
void GPUNameStream(hipStream_t stream, const char *name);

}  // namespace pbrt

#endif  // PBRT_GPU_UTIL_H
