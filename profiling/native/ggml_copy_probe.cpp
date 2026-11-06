// Isolated ggml copy probe: the transfer price at the same abstraction level as the
// stage probes (a ggml graph copy between backends), not raw cudaMemcpy.
//
// The measured object is exactly what the IR executor does for a `copy` node:
//   synchronize both backends, ggml_backend_tensor_copy(view_from, view_to), synchronize.
// Allocation, view construction and validation are outside the timed region.
//
// `C2C` is the control channel.  It moves the same payload between two CPU backend
// buffers through the same call and the same synchronization pair, so its intercept `a`
// is what this harness costs when no PCIe is involved.  Without it, a large `a` on H2D
// cannot be told apart from harness overhead -- and the intercept is the part a
// per-copy price for small tensors is made of.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

// The same placement discipline the stage probes and `ir_runner` follow: an explicit
// logical-CPU mask, applied to both the process and the CPU backend's threadpool, and
// echoed in the reading.  The first transfer round had none of this, and its two-state
// behaviour (a 4 KiB copy reading 13 us or 53 us with the same binary and command) has a
// standing alternative explanation -- the host thread landed on a different physical core
// -- which cannot be tested against a reading that does not say where it ran.
static std::vector<int> parse_mask(const std::string & v) {
    std::vector<int> ids;
    size_t start = 0;
    while (start <= v.size()) {
        const size_t comma = v.find(',', start);
        const std::string piece = v.substr(start, comma == std::string::npos ? std::string::npos
                                                                            : comma - start);
        if (piece.empty() || piece.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("invalid cpu mask: " + v);
        ids.push_back(std::stoi(piece));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return ids;
}

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t parse_size(const std::string & v) {
    if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid byte count: " + v);
    return static_cast<size_t>(std::stoull(v));
}

int main(int argc, char ** argv) {
    try {
        size_t bytes = 0;
        int warmup = 5, samples = 20, device = 0;
        std::string direction;
        std::vector<int> cpu_ids;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::printf("ggml-copy-probe --bytes N --direction H2D|D2H|C2C "
                            "[--warmup 5 --samples 20 --device 0 --cpu-mask 2,4]\n");
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
            std::string value = argv[++i];
            if (key == "--bytes") bytes = parse_size(value);
            else if (key == "--direction") direction = value;
            else if (key == "--warmup") warmup = std::stoi(value);
            else if (key == "--samples") samples = std::stoi(value);
            else if (key == "--device") device = std::stoi(value);
            else if (key == "--cpu-mask") cpu_ids = parse_mask(value);
            else throw std::runtime_error("unknown argument " + key);
        }
        if (!bytes || bytes > (size_t{1} << 30)) throw std::runtime_error("--bytes must be in 1..1073741824");
        if (warmup < 0 || samples < 3) throw std::runtime_error("invalid warmup/sample count");
        const bool h2d = direction == "H2D";
        const bool c2c = direction == "C2C";
        if (!h2d && direction != "D2H" && !c2c) throw std::runtime_error("invalid direction");

        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) throw std::runtime_error("cpu backend init failed");
        ggml_backend_cpu_set_n_threads(cpu, 2);
        ggml_threadpool_t pool = nullptr;
        if (!cpu_ids.empty()) {
            // As in `ir_runner`: the threadpool is strict *and* the process is pinned, then
            // the mask is read back -- a requested placement that was not applied is the
            // failure this whole mechanism exists to make visible.
            auto tpp = ggml_threadpool_params_default(static_cast<int>(cpu_ids.size()));
            tpp.strict_cpu = true;
            for (const int id : cpu_ids) {
                if (id < 0 || id >= GGML_MAX_N_THREADS) throw std::runtime_error("cpu mask out of range");
                tpp.cpumask[id] = true;
            }
#if defined(_WIN32)
            DWORD_PTR bits = 0;
            for (const int id : cpu_ids) bits |= DWORD_PTR{1} << id;
            if (!SetProcessAffinityMask(GetCurrentProcess(), bits))
                throw std::runtime_error("cannot apply process CPU affinity");
            DWORD_PTR observed = 0, system_mask = 0;
            if (!GetProcessAffinityMask(GetCurrentProcess(), &observed, &system_mask) || observed != bits)
                throw std::runtime_error("process CPU affinity readback differs from the request");
#endif
            pool = ggml_threadpool_new(&tpp);
            if (!pool) throw std::runtime_error("threadpool initialization failed");
            ggml_backend_cpu_set_threadpool(cpu, pool);
        }
#if defined(GGML_USE_CUDA)
        ggml_backend_t gpu = ggml_backend_cuda_init(device);
        if (!gpu) throw std::runtime_error("cuda backend init failed");
#else
        throw std::runtime_error("built without CUDA support");
#endif

        // One host buffer and one device buffer, each as large as the payload; the copy
        // moves the same number of bytes every sample.
        ggml_context * hctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        ggml_context * dctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        // C2C needs two CPU-side tensors, so the "device" context is allocated on the cpu
        // backend instead of the cuda one; everything else in the harness is identical.
        ggml_tensor * host = ggml_new_tensor_1d(hctx, GGML_TYPE_I8, static_cast<int64_t>(bytes));
        ggml_tensor * dev = ggml_new_tensor_1d(dctx, GGML_TYPE_I8, static_cast<int64_t>(bytes));
        ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors(hctx, cpu);
        ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(dctx, c2c ? cpu : gpu);
        if (!hbuf || !dbuf) throw std::runtime_error("buffer allocation failed");

        std::vector<uint8_t> pattern(bytes, (h2d || c2c) ? 0x5a : 0x00);
        ggml_backend_tensor_set(host, pattern.data(), 0, bytes);
        std::vector<uint8_t> other(bytes, (h2d || c2c) ? 0x00 : 0x5a);
        ggml_backend_tensor_set(dev, other.data(), 0, bytes);
        ggml_backend_synchronize(cpu);
        ggml_backend_synchronize(gpu);

        auto copy = [&]() {
            if (c2c) ggml_backend_tensor_copy(host, dev);
            else ggml_backend_tensor_copy(h2d ? host : dev, h2d ? dev : host);
        };
        for (int i = 0; i < warmup; ++i) {
            copy();
            ggml_backend_synchronize(gpu);
            ggml_backend_synchronize(cpu);
        }
        std::vector<int64_t> wall_ns;
        for (int i = 0; i < samples; ++i) {
            ggml_backend_synchronize(gpu);
            ggml_backend_synchronize(cpu);
            const int64_t t0 = now_ns();
            copy();
            ggml_backend_synchronize(gpu);
            ggml_backend_synchronize(cpu);
            wall_ns.push_back(now_ns() - t0);
        }

        // Validation outside timing: read back from the host side.
        std::vector<uint8_t> check(bytes);
        ggml_backend_tensor_get(host, check.data(), 0, bytes);
        for (size_t i = 0; i < bytes; ++i) {
            if (check[i] != 0x5a) throw std::runtime_error("copy validation failed at byte " + std::to_string(i));
        }

        std::string mask = "null";
        if (!cpu_ids.empty()) {
            mask = "[";
            for (size_t i = 0; i < cpu_ids.size(); ++i)
                mask += (i ? "," : "") + std::to_string(cpu_ids[i]);
            mask += "]";
        }
        std::printf("{\"kind\":\"transfer\",\"backend\":\"ggml\",\"direction\":\"%s\","
                    "\"size_bytes\":%zu,\"device_index\":%d,\"threads\":2,\"cpu_mask\":%s,"
                    "\"warmup\":%d,\"samples\":%d,"
                    "\"timing_scope\":\"ggml_backend_tensor_copy plus synchronize on both backends\","
                    "\"includes\":[\"copy enqueue\",\"both backend synchronizations\"],"
                    "\"excludes\":[\"allocation\",\"view construction\",\"validation\"],"
                    "\"wall_ns\":[",
                    direction.c_str(), bytes, device, mask.c_str(), warmup, samples);
        for (size_t i = 0; i < wall_ns.size(); ++i) std::printf("%s%lld", i ? "," : "", (long long) wall_ns[i]);
        std::printf("]}\n");

        ggml_backend_buffer_free(hbuf);
        ggml_backend_buffer_free(dbuf);
        ggml_free(hctx);
        ggml_free(dctx);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ggml-copy-probe: %s\n", e.what());
        return 1;
    }
}
