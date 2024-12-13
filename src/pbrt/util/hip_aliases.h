#ifndef PBRT_GPU_HIP_ALIASES_H
#define PBRT_GPU_HIP_ALIASES_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define CUdeviceptr hipDeviceptr_t
#define cudaEvent_t hipEvent_t

#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree

#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice

#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize

#define cudaDeviceSynchronize hipDeviceSynchronize

#endif  // PBRT_GPU_HIP_ALIASES_H
