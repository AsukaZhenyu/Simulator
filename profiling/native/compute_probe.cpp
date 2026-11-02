// One resident GGML MUL_MAT per sample. Allocation, quantization and I/O are untimed.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef MEASURE_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string backend;
    std::string dtype;
    int64_t m = 0, k = 0, n = 0;
    int threads = 16, warmup = 5, samples = 15;
    uint32_t seed = 17;
};

static int64_t integer(const std::string & text) {
    size_t consumed = 0;
    const int64_t value = std::stoll(text, &consumed);
    if (consumed != text.size()) throw std::runtime_error("invalid integer: " + text);
    return value;
}

static Options parse(int argc, char ** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc || !seen.insert(key).second) {
            throw std::runtime_error("missing value or duplicate option: " + key);
        }
        const std::string value = argv[i + 1];
        if (key == "--backend") o.backend = value;
        else if (key == "--dtype") o.dtype = value;
        else if (key == "--m") o.m = integer(value);
        else if (key == "--k") o.k = integer(value);
        else if (key == "--n") o.n = integer(value);
        else if (key == "--seed") {
            const int64_t v = integer(value);
            if (v < 0 || v > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("seed out of range");
            o.seed = static_cast<uint32_t>(v);
        } else if (key == "--threads" || key == "--warmup" || key == "--samples") {
            const int64_t v = integer(value);
            if (v < (key == "--warmup" ? 0 : 1) || v > std::numeric_limits<int>::max()) throw std::runtime_error(key + " has an invalid integer value");
            if (key == "--threads") o.threads = static_cast<int>(v);
            if (key == "--warmup") o.warmup = static_cast<int>(v);
            if (key == "--samples") o.samples = static_cast<int>(v);
        } else throw std::runtime_error("unknown option: " + key);
    }
    if (o.backend != "CPU" && o.backend != "CUDA0") throw std::runtime_error("--backend must be CPU or CUDA0");
    if (o.dtype != "f32" && o.dtype != "f16" && o.dtype != "q4_k" && o.dtype != "q6_k") throw std::runtime_error("--dtype must be f32, f16, q4_k or q6_k");
    if (o.m <= 0 || o.k <= 0 || o.n <= 0) throw std::runtime_error("--m, --k and --n must be positive");
    if (o.threads > GGML_MAX_N_THREADS) throw std::runtime_error("too many CPU threads");
    if (o.samples < 3 || o.samples > 10000) throw std::runtime_error("--samples must be in 3..10000");
    if ((o.dtype == "q4_k" || o.dtype == "q6_k") && o.k % 256 != 0) throw std::runtime_error("K quantization requires k divisible by 256");
    return o;
}

static size_t elements(int64_t a, int64_t b) {
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float),
                                            std::numeric_limits<int64_t>::max() / sizeof(float));
    if (static_cast<uint64_t>(a) > limit / static_cast<uint64_t>(b)) throw std::runtime_error("shape overflows addressable storage");
    return static_cast<size_t>(a) * static_cast<size_t>(b);
}

struct Session {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_threadpool_t pool = nullptr;
    ~Session() {
        if (backend) ggml_backend_synchronize(backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
        if (ctx) ggml_free(ctx);
    }
};

static void fill(std::vector<float> & values, std::mt19937 & rng) {
    for (float & v : values) {
        // 24 random bits retain F32 precision beyond F16. The old j/2048
        // grid often made conversion and sampled float dots exactly representable.
        const int32_t value = static_cast<int32_t>(rng() >> 8) - 8388608;
        v = static_cast<float>(value) / 16777216.0f;
    }
}

static void compute(Session & s, ggml_cgraph * graph) {
    // ggml_backend_graph_compute includes backend synchronization in this GGML version.
    const ggml_status status = ggml_backend_graph_compute(s.backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed, status=" + std::to_string(status));
}

static int run(const Options & o) {
    // Set before backend initialization: CUDA caches this environment decision.
#ifdef _WIN32
    if (_putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1") != 0) throw std::runtime_error("cannot disable CUDA graphs");
#else
    if (setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) != 0) throw std::runtime_error("cannot disable CUDA graphs");
#endif
    const size_t nw = elements(o.k, o.m), nx = elements(o.k, o.n), ny = elements(o.m, o.n);
    const ggml_type wt = o.dtype == "f32" ? GGML_TYPE_F32 : o.dtype == "f16" ? GGML_TYPE_F16 : o.dtype == "q4_k" ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    const bool quantized = wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q6_K;
    Session s;
    const bool cpu = o.backend == "CPU";
    if (cpu) s.backend = ggml_backend_cpu_init();
    else {
#ifdef MEASURE_HAVE_CUDA
        s.backend = ggml_backend_cuda_init(0);
#else
        throw std::runtime_error("CUDA0 requested, but this probe was built without GGML CUDA support");
#endif
    }
    if (!s.backend) throw std::runtime_error("backend initialization failed");
    if (cpu) {
        ggml_threadpool_params p = ggml_threadpool_params_default(o.threads);
        s.pool = ggml_threadpool_new(&p);
        if (!s.pool) throw std::runtime_error("CPU threadpool initialization failed");
        ggml_backend_cpu_set_n_threads(s.backend, o.threads);
        ggml_backend_cpu_set_threadpool(s.backend, s.pool);
    }
    ggml_init_params init = {ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false), nullptr, true};
    s.ctx = ggml_init(init);
    if (!s.ctx) throw std::runtime_error("graph context allocation failed");
    ggml_tensor * w = ggml_new_tensor_2d(s.ctx, wt, o.k, o.m);
    ggml_tensor * x = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, o.k, o.n);
    ggml_tensor * y = ggml_mul_mat(s.ctx, w, x);
    ggml_set_name(w, "weight"); ggml_set_name(x, "input"); ggml_set_name(y, "output");
    if (!ggml_backend_supports_op(s.backend, y)) throw std::runtime_error("backend does not support this MUL_MAT");
    ggml_cgraph * graph = ggml_new_graph_custom(s.ctx, 16, false);
    ggml_build_forward_expand(graph, y);
    s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, s.backend);
    if (!s.buffer) throw std::runtime_error("backend tensor allocation failed");
    const size_t cpu_work = cpu ? ggml_graph_plan(graph, o.threads, s.pool).work_size : 0;

    std::mt19937 rng(o.seed);
    std::vector<float> wf(nw), xf(nx), yf(ny);
    fill(wf, rng); fill(xf, rng);
    std::vector<uint8_t> packed(ggml_nbytes(w));
    size_t f16_rounded_values = 0;
    double f16_max_rounding_error = 0;
    if (wt == GGML_TYPE_F32) std::memcpy(packed.data(), wf.data(), packed.size());
    else if (wt == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(wf.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()), static_cast<int64_t>(nw));
        const auto * half = reinterpret_cast<const ggml_fp16_t *>(packed.data());
        for (size_t i = 0; i < nw; ++i) {
            const double error = std::abs(static_cast<double>(ggml_fp16_to_fp32(half[i])) - wf[i]);
            f16_rounded_values += error != 0;
            f16_max_rounding_error = std::max(f16_max_rounding_error, error);
        }
    } else {
        const size_t wrote = ggml_quantize_chunk(wt, wf.data(), packed.data(), 0, o.m, o.k, nullptr);
        if (wrote != packed.size()) throw std::runtime_error("quantized weight size mismatch");
    }
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    ggml_backend_synchronize(s.backend);
    for (int i = 0; i < o.warmup; ++i) compute(s, graph);
    std::vector<int64_t> wall_ns;
    wall_ns.reserve(o.samples);
    for (int i = 0; i < o.samples; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        compute(s, graph);
        const auto end = std::chrono::steady_clock::now();
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns <= 0) throw std::runtime_error("nonpositive timer sample");
        wall_ns.push_back(ns);
    }

    // Check all outputs are finite, then sample reference dots using the stored weights.
    // Quantized kernels also quantize activations internally; this is a sanity check,
    // not proof of whole-model numerical accuracy.
    ggml_backend_tensor_get(y, yf.data(), 0, ggml_nbytes(y));
    for (const float value : yf) if (!std::isfinite(value)) throw std::runtime_error("nonfinite output");
    const size_t count = std::min<size_t>(12, ny);
    const double atol = quantized ? 0.002 * std::sqrt(static_cast<double>(o.k)) : 0.001;
    const double rtol = quantized ? 0.025 : 0.002;
    double max_abs = 0, max_relative = 0, error_squared = 0, reference_squared = 0;
    std::vector<float> row(static_cast<size_t>(o.k));
    const ggml_type_traits * traits = ggml_get_type_traits(wt);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = count == 1 ? 0 : static_cast<size_t>((static_cast<long double>(ny - 1) * i) / (count - 1));
        const size_t r = index % static_cast<size_t>(o.m), token = index / static_cast<size_t>(o.m);
        const uint8_t * bytes = packed.data() + r * ggml_row_size(wt, o.k);
        if (wt == GGML_TYPE_F32) std::memcpy(row.data(), bytes, row.size() * sizeof(float));
        else {
            if (!traits || !traits->to_float) throw std::runtime_error("missing weight dequantizer");
            traits->to_float(bytes, row.data(), o.k);
        }
        double reference = 0;
        for (size_t j = 0; j < row.size(); ++j) reference += static_cast<double>(row[j]) * xf[token * row.size() + j];
        const double error = std::abs(static_cast<double>(yf[index]) - reference);
        max_abs = std::max(max_abs, error);
        max_relative = std::max(max_relative, error / std::max(std::abs(reference), 1e-12));
        error_squared += error * error;
        reference_squared += reference * reference;
    }
    const double rmse = std::sqrt(error_squared / static_cast<double>(count));
    const double reference_rms = std::sqrt(reference_squared / static_cast<double>(count));
    const double normalized_rmse = rmse / std::max(reference_rms, 1e-12);
    const bool correct = normalized_rmse <= rtol || rmse <= atol;
    auto sorted = wall_ns;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 ? static_cast<double>(sorted[sorted.size()/2])
        : (static_cast<double>(sorted[sorted.size()/2-1]) + sorted[sorted.size()/2]) / 2;
    std::cout << std::setprecision(17)
        << "{\"schema\":\"ggml_compute_probe/1\",\"backend\":\"" << o.backend
        << "\",\"dtype\":\"" << o.dtype << "\",\"activation_dtype\":\"f32\",\"m\":" << o.m
        << ",\"k\":" << o.k << ",\"n\":" << o.n << ",\"seed\":" << o.seed
        << ",\"input_generator\":\"mt19937_uniform24_v2\""
        << ",\"threads\":" << o.threads << ",\"threads_requested\":" << o.threads << ",\"threads_applied\":";
    if (cpu) std::cout << o.threads; else std::cout << "null";
    std::cout
        << ",\"thread_management\":\"" << (cpu ? "persistent_ggml_threadpool" : "not_applicable")
        << "\",\"warmup\":" << o.warmup << ",\"samples\":" << o.samples
        << ",\"cuda_graphs_disabled\":true,\"timing\":\"steady_clock_graph_compute_including_sync\""
        << ",\"scope\":\"resident_repeated_single_task_setup_and_host_device_transfers_excluded\""
        << ",\"weight_bytes\":" << ggml_nbytes(w) << ",\"input_bytes\":" << ggml_nbytes(x)
        << ",\"output_bytes\":" << ggml_nbytes(y) << ",\"backend_buffer_bytes\":" << ggml_backend_buffer_get_size(s.buffer)
        << ",\"cpu_cplan_work_size_bytes\":";
    if (cpu) std::cout << cpu_work; else std::cout << "null";
    std::cout << ",\"workspace_bytes\":";
    if (cpu) std::cout << cpu_work; else std::cout << "null";
    std::cout << ",\"workspace_source\":\"" << (cpu ? "ggml_graph_plan" : "unknown_backend_private_allocations")
        << "\",\"gpu_workspace_bytes\":null,\"gpu_workspace_status\":\""
        << (cpu ? "not_applicable" : "unknown_backend_private_allocations")
        << "\",\"f16_rounding_diagnostic\":";
    if (wt == GGML_TYPE_F16) {
        std::cout << "{\"changed_values\":" << f16_rounded_values
                  << ",\"total_values\":" << nw << ",\"max_abs_error\":" << f16_max_rounding_error << "}";
    } else std::cout << "null";
    std::cout << ",\"all_outputs_finite\":true,\"reference_check\":{\"samples\":" << count
        << ",\"passed\":" << (correct ? "true" : "false") << ",\"max_abs_error\":" << max_abs
        << ",\"max_relative_error\":" << max_relative << ",\"atol\":" << atol << ",\"rtol\":" << rtol
        << ",\"rmse\":" << rmse << ",\"reference_rms\":" << reference_rms << ",\"normalized_rmse\":" << normalized_rmse
        << ",\"criterion\":\"normalized_rmse_le_rtol_or_rmse_le_atol\""
        << ",\"reference\":\"double_dot_of_dequantized_stored_weight_and_f32_input\"}"
        << ",\"median_wall_ns\":" << median << ",\"wall_ns\":[";
    for (size_t i = 0; i < wall_ns.size(); ++i) std::cout << (i ? "," : "") << wall_ns[i];
    std::cout << "]}\n";
    if (!correct) {
        std::cerr << "sampled output reference check failed\n";
        return 2;
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "ggml-compute-probe --backend CPU|CUDA0 --m OUTPUT_ROWS --k INPUT_FEATURES "
                     "--n TOKENS --dtype f32|f16|q4_k|q6_k "
                     "[--threads 16 --warmup 5 --samples 20 --seed 17]\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception & e) {
        std::cerr << "ggml-compute-probe: " << e.what() << '\n';
        return 1;
    }
}
