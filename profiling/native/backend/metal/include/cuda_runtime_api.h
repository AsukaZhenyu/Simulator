// The CUDA vocabulary the generated Executor fragments speak, answered on Metal.
//
// This file is reached only when a build puts `profiling/native/backend/metal`
// ahead of the CUDA toolkit on the include path, which is what
// `-DPROBE_BACKEND=metal` does. It exists because the fragments that make up
// `steady-costs-probe` and `depth-probe` are hash-pinned: `generate_probe.py`
// records a digest of each one and the campaign refuses a directory of different
// fragments, so they cannot be edited to say "ggml-metal" instead of "ggml-cuda".
// Replacing the *header* keeps those bytes intact and puts the backend decision
// in the build, which is where it belongs.
//
// What each name means here is not "CUDA works anyway". There is one memory on
// this machine, so:
//
//   * a pointer's "device or host?" question has no second answer, and the calls
//     that ask it raise instead of returning a category the receipt would then
//     claim to have observed;
//   * host registration is a no-op by construction (nothing is ever unmapped
//     from the device), and it also raises rather than silently succeeding;
//   * `cudaEvent` timing has no counterpart that measures the same interval, and
//     the probes refuse the `events` instrumentation before reaching it.
//
// Raising is the point. A stub that returned success would let a Metal receipt
// state a pageable/registered category it never observed, which is exactly the
// kind of plausible-looking wrong number the port is trying to avoid. The
// portable modes -- `bare` and `stamped` -- never call any of these.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace portable_metal {

//: The one place a Metal build says "this question has no answer here".
[[noreturn]] inline void unsupported(const char *what) {
    throw std::runtime_error(
        std::string("this build is Metal: ") + what
        + " asks about a second memory that this machine does not have. "
          "The portable modes (bare, stamped) do not call it.");
}

}  // namespace portable_metal

using cudaError_t = int;
enum : cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorNotSupported = 801,
    cudaErrorInvalidDevice = 101,
};

enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
};

struct cudaPointerAttributes {
    cudaMemoryType type = cudaMemoryTypeUnregistered;
    int device = 0;
    void *devicePointer = nullptr;
    void *hostPointer = nullptr;
};

using cudaEvent_t = void *;
using cudaEventFlags = unsigned int;
enum : cudaEventFlags { cudaEventDefault = 0, cudaEventBlockingSync = 1 };

using cudaDeviceFlags = unsigned int;
enum : cudaDeviceFlags {
    cudaDeviceScheduleAuto = 0,
    cudaDeviceScheduleSpin = 1,
    cudaDeviceScheduleYield = 2,
    cudaDeviceScheduleBlockingSync = 4,
    cudaDeviceScheduleMask = 7,
};

enum cudaDeviceAttr { cudaDevAttrHostRegisterSupported = 100 };

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char *cudaGetErrorString(cudaError_t) { return "not queried on a Metal build"; }
inline const char *cudaGetErrorName(cudaError_t) { return "cudaErrorNotSupported"; }

inline cudaError_t cudaGetDevice(int *device) { if (device) *device = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceFlags(cudaDeviceFlags *flags) {
    if (flags) *flags = cudaDeviceScheduleAuto;
    return cudaSuccess;
}
inline cudaError_t cudaSetDeviceFlags(cudaDeviceFlags) { return cudaSuccess; }

// Unified memory: every pointer is host-accessible, so there is no category to
// report and no registration to perform.
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *, const void *) {
    return cudaErrorNotSupported;
}
inline cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int) {
    return cudaErrorNotSupported;
}
inline cudaError_t cudaMemGetInfo(std::size_t *, std::size_t *) { return cudaErrorNotSupported; }
inline cudaError_t cudaHostRegister(void *, std::size_t, unsigned int) {
    portable_metal::unsupported("cudaHostRegister");
}
inline cudaError_t cudaHostUnregister(void *) {
    portable_metal::unsupported("cudaHostUnregister");
}

// The three calls a timing event needs. Each one raises: the interval a CUDA
// event measures is "on the device, between two stream positions", and on a
// unified machine the honest instrument is the portable clock, which is what
// the `stamped` mode already uses.
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, cudaEventFlags) {
    portable_metal::unsupported("cudaEventCreateWithFlags");
}
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t) {
    portable_metal::unsupported("cudaEventElapsedTime");
}

//: Named so a receipt can say which build produced it.
#define PORTABLE_ACCELERATOR "metal"
#define PORTABLE_ACCELERATOR_HAS_CUDA 0
#define PORTABLE_DEVICE_MEMORY "shared"
