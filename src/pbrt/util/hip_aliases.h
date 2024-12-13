#ifndef PBRT_UTIL_HIP_ALIASES_H
#define PBRT_UTIL_HIP_ALIASES_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDevAttrKernelExecTimeout hipDeviceAttributeKernelExecTimeout
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion

#define cudaGraphicsMapResources hipGraphicsMapResources
#define cudaGraphicsUnmapResources hipGraphicsUnmapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsResource hipGraphicsResource 
#define cudaGraphicsGLRegisterBuffer hipGraphicsGLRegisterBuffer
#define cudaGraphicsRegisterFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard

#define cudaGLGetDevices hipGLGetDevices
#define cudaGLDeviceListAll hipGLDeviceListAll

#define CUdeviceptr hipDeviceptr_t
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree

#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemset hipMemset

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventSynchronize hipEventSynchronize

#define CUstream hipStream_t
#define cudaDeviceSynchronize hipDeviceSynchronize

#endif  // PBRT_UTIL_HIP_ALIASES_H
