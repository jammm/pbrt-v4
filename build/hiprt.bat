REM call hipcc.bat -I. -I../src -I../src/ext/hiprtSdk/ -I../src/ext/openvdb/nanovdb -x hip ../src/pbrt/gpu/hiprt/hiprt.cu -O3 -ffast-math -std=c++17 --cuda-device-only --genco --offload-arch=gfx1100 -DWARP_THREADS=32 -D__HIP_PLATFORM_AMD__ -DPBRT_BUILD_GPU_RENDERER -DPBRT_RESTRICT=__restrict__ -DPBRT_IS_WINDOWS -DNOMINMAX -DPBRT_IS_MSVC -D_CRT_SECURE_NO_WARNINGS -D_ENABLE_EXTENDED_ALIGNED_STORAGE -DBLOCK_SIZE=64 -DSHARED_STACK_SIZE=16 -o hiprt.hipfb

call hipcc.bat -I. -I../src -I../src/ext/hiprtSdk/ -I../src/ext/openvdb/nanovdb -x hip ../src/pbrt/gpu/hiprt/hiprt.cu ../src/pbrt/util/sobolmatrices.cpp ../src/pbrt/util/primes.cpp ../src/pbrt/options.cpp ../src/pbrt/shapes.cpp -O3 -ffast-math -std=c++17 --cuda-device-only --offload-arch=gfx1100 -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm -DWARP_THREADS=32 -D__HIP_PLATFORM_AMD__ -DPBRT_BUILD_GPU_RENDERER -DPBRT_RESTRICT=__restrict__ -DPBRT_IS_WINDOWS -DNOMINMAX -DPBRT_IS_MSVC -D_CRT_SECURE_NO_WARNINGS -D_ENABLE_EXTENDED_ALIGNED_STORAGE -DBLOCK_SIZE=64 -DSHARED_STACK_SIZE=16

call clang++.exe -fgpu-rdc --hip-link -Xoffload-linker --whole-archive --cuda-device-only --offload-arch=gfx1100 hiprt02004_6.0_amd_lib_win.bc hiprt-hip-amdgcn-amd-amdhsa.bc primes-hip-amdgcn-amd-amdhsa.bc sobolmatrices-hip-amdgcn-amd-amdhsa.bc options-hip-amdgcn-amd-amdhsa.bc shapes-hip-amdgcn-amd-amdhsa.bc -o hiprt.hipfb