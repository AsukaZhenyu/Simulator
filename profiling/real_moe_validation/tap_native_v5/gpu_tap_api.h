// Private additive observer ABI v5. Registration/allocation are outside compute.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ggml.h"
#if defined(SIMULATOR_GPU_TAP_BUILD)
#define SIM_GPU_API __declspec(dllexport)
#else
#define SIM_GPU_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Reserve a unique pinned destination for this actual consumer invocation.
// Return null to reject. The backend supplies enqueue status separately.
typedef void * (*sim_gpu_reserve)(const struct ggml_tensor *, const struct ggml_tensor *, void *, size_t, void *);
typedef void (*sim_gpu_copy_status)(int, void *);
typedef void (*sim_gpu_graph_event)(const struct ggml_cgraph *, int, int, int, void *);
SIM_GPU_API void simulator_gpu_tap_set_v5(sim_gpu_reserve, sim_gpu_copy_status, sim_gpu_graph_event, void *);
SIM_GPU_API int simulator_gpu_tap_host_alloc_v5(size_t, void **);
SIM_GPU_API int simulator_gpu_tap_host_free_v5(void *);
#ifdef __cplusplus
}
#endif
