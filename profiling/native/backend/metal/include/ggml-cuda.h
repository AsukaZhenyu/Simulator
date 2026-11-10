// `ggml-cuda.h`'s three names, answered by the Metal backend.
//
// Reached when a build puts this directory ahead of the toolkit's headers
// (`-DPROBE_BACKEND=metal`). The highlevel fragment -- which is hash-pinned and
// therefore not editable -- asks for a backend, its buffer type and its registry
// by the CUDA names. Those are three questions about "the accelerator", not about
// CUDA, so the answers are the Metal ones and no caller changes.
//
// The buffer type is the interesting one: Metal exposes no
// `ggml_backend_metal_buffer_type`, because a Metal buffer type belongs to the
// device rather than to the library. Asking the device for its default buffer
// type is the same question on every backend, which is why the wrapper goes
// through the generic API instead of a Metal-specific symbol.
#pragma once

#include "ggml-backend.h"
#include "ggml-metal.h"

//: The accelerator's backend. One device on a Mac, so the argument is ignored in
//: the same way CUDA's device argument would be on a single-GPU box.
static inline ggml_backend_t ggml_backend_cuda_init(int /*device*/) {
    return ggml_backend_metal_init();
}

static inline ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int /*device*/) {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend == nullptr) return nullptr;
    ggml_backend_buffer_type_t type = ggml_backend_dev_buffer_type(ggml_backend_get_device(backend));
    ggml_backend_free(backend);
    return type;
}

static inline ggml_backend_reg_t ggml_backend_cuda_reg(void) {
    return ggml_backend_metal_reg();
}
