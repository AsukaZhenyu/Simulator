// Private additive ABI. Configure only before/after all owned graph computation.
#pragma once
#include "ggml.h"
#if defined(SIMULATOR_MOE_TAP_BUILD)
#define SIMULATOR_MOE_TAP_API __declspec(dllexport)
#else
#define SIMULATOR_MOE_TAP_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*simulator_moe_tap_callback)(const struct ggml_tensor * dst,
                                         const struct ggml_tensor * ids,
                                         void * user);
SIMULATOR_MOE_TAP_API void simulator_moe_tap_set(simulator_moe_tap_callback callback, void * user);
#ifdef __cplusplus
}
#endif
