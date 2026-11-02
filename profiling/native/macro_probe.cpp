// CPU GGML macro graphs. Weights, inputs and output are resident; setup is untimed.
// This is an explicit operator contract, not a claim to reproduce a model runtime.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string macro, dtype, attention_gate;
    std::string gate_dtype, up_dtype, down_dtype;
    int64_t hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0;
    int64_t q = 0, history = 0;
    int threads = 16, warmup = 5, samples = 20;
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
        if (i + 1 == argc || !seen.insert(key).second) throw std::runtime_error("missing value or duplicate option: " + key);
        const std::string value = argv[i + 1];
        if (key == "--macro") o.macro = value;
        else if (key == "--dtype") o.dtype = value;
        else if (key == "--gate-dtype") o.gate_dtype = value;
        else if (key == "--up-dtype") o.up_dtype = value;
        else if (key == "--down-dtype") o.down_dtype = value;
        else if (key == "--attention-gate") o.attention_gate = value;
        else if (key == "--hidden") o.hidden = integer(value);
        else if (key == "--intermediate") o.intermediate = integer(value);
        else if (key == "--heads") o.heads = integer(value);
        else if (key == "--kv-heads") o.kv_heads = integer(value);
        else if (key == "--head-dim") o.head_dim = integer(value);
        else if (key == "--q") o.q = integer(value);
        else if (key == "--history") o.history = integer(value);
        else if (key == "--seed") {
            const int64_t v = integer(value);
            if (v < 0 || v > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("seed out of range");
            o.seed = static_cast<uint32_t>(v);
        } else if (key == "--threads" || key == "--warmup" || key == "--samples") {
            const int64_t v = integer(value);
            if (v < (key == "--warmup" ? 0 : 1) || v > std::numeric_limits<int>::max()) throw std::runtime_error("invalid " + key);
            if (key == "--threads") o.threads = static_cast<int>(v);
            if (key == "--warmup") o.warmup = static_cast<int>(v);
            if (key == "--samples") o.samples = static_cast<int>(v);
        } else throw std::runtime_error("unknown option: " + key);
    }
    if (o.macro != "attention" && o.macro != "dense_ffn") throw std::runtime_error("--macro must be attention or dense_ffn");
    const auto supported_dtype = [](const std::string & dtype) { return dtype == "f32" || dtype == "f16" || dtype == "q4_k" || dtype == "q6_k"; };
    const bool mixed = !o.gate_dtype.empty() || !o.up_dtype.empty() || !o.down_dtype.empty();
    if (mixed) {
        if (o.macro != "dense_ffn" || !o.dtype.empty() || !supported_dtype(o.gate_dtype) || !supported_dtype(o.up_dtype) || !supported_dtype(o.down_dtype)) throw std::runtime_error("mixed FFN requires all three matrix dtype options and no --dtype");
        o.dtype = "mixed";
    } else if (!supported_dtype(o.dtype)) throw std::runtime_error("unsupported --dtype");
    if (o.hidden <= 0 || o.q <= 0 || o.history < 0) throw std::runtime_error("hidden and q must be positive; history must be nonnegative");
    if (o.threads > GGML_MAX_N_THREADS || o.samples < 3 || o.samples > 10000) throw std::runtime_error("threads out of range or samples outside 3..10000");
    if (o.macro == "dense_ffn") {
        if (o.intermediate <= 0) throw std::runtime_error("dense_ffn requires positive --intermediate");
        if (o.heads != 0 || o.kv_heads != 0 || o.head_dim != 0) throw std::runtime_error("attention geometry does not apply to dense_ffn");
        if (!o.attention_gate.empty() && o.attention_gate != "none") throw std::runtime_error("dense_ffn cannot have an attention gate");
        o.attention_gate = "none";
        if (!mixed) o.gate_dtype = o.up_dtype = o.down_dtype = o.dtype;
    } else {
        if (o.intermediate != 0) throw std::runtime_error("intermediate does not apply to attention");
        if (o.heads <= 0 || o.kv_heads <= 0 || o.head_dim <= 0 || o.heads % o.kv_heads != 0) throw std::runtime_error("attention requires positive geometry with heads divisible by kv-heads");
        if (o.attention_gate != "none" && o.attention_gate != "sigmoid") throw std::runtime_error("attention requires explicit --attention-gate none|sigmoid");
        if (o.history > std::numeric_limits<int>::max() || o.q > std::numeric_limits<int>::max() - o.history) throw std::runtime_error("attention sequence length exceeds mask indexing limit");
    }
    const auto quantized = [](const std::string & dtype) { return dtype == "q4_k" || dtype == "q6_k"; };
    if (o.macro == "dense_ffn" && (((quantized(o.gate_dtype) || quantized(o.up_dtype)) && o.hidden % 256) || (quantized(o.down_dtype) && o.intermediate % 256))) throw std::runtime_error("quantized matrix reduction dimensions must be divisible by 256");
    if (o.macro == "attention" && quantized(o.dtype) && (o.heads > std::numeric_limits<int64_t>::max() / o.head_dim || (o.heads * o.head_dim) % 256)) throw std::runtime_error("quantized O projection reduction dimension must be divisible by 256");
    return o;
}

static size_t product(std::initializer_list<int64_t> dims) {
    size_t result = 1;
    const size_t limit = static_cast<size_t>(std::min<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float), std::numeric_limits<int64_t>::max() / sizeof(float)));
    for (const int64_t d : dims) {
        if (d <= 0 || static_cast<uint64_t>(d) > limit / result) throw std::runtime_error("shape overflows addressable storage");
        result *= static_cast<size_t>(d);
    }
    return result;
}

static size_t add_bytes(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::runtime_error("memory accounting overflow");
    return a + b;
}

static void fill(std::vector<float> & values, std::mt19937 & rng) {
    for (float & v : values) v = static_cast<float>(static_cast<int32_t>(rng() >> 8) - 8388608) / 16777216.0f;
}

struct Session {
    ggml_context * resident = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_threadpool_t pool = nullptr;
    ~Session() {
        if (backend) ggml_backend_synchronize(backend);
        if (allocator) ggml_gallocr_free(allocator);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
        if (ctx) ggml_free(ctx);
        if (resident) ggml_free(resident);
    }
};

struct Weight {
    ggml_tensor * tensor = nullptr;
    int64_t cols = 0, rows = 0;
    ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> packed;

    void initialize(std::mt19937 & rng) {
        std::vector<float> values(product({cols, rows}));
        fill(values, rng);
        packed.resize(ggml_nbytes(tensor));
        if (type == GGML_TYPE_F32) std::memcpy(packed.data(), values.data(), packed.size());
        else if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()), static_cast<int64_t>(values.size()));
        else if (ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows, cols, nullptr) != packed.size()) throw std::runtime_error("weight quantization size mismatch");
        ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
    }

    void row(int64_t index, std::vector<float> & values) const {
        values.resize(static_cast<size_t>(cols));
        const uint8_t * bytes = packed.data() + static_cast<size_t>(index) * ggml_row_size(type, cols);
        if (type == GGML_TYPE_F32) std::memcpy(values.data(), bytes, values.size() * sizeof(float));
        else {
            const auto * traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float) throw std::runtime_error("weight dequantizer unavailable");
            traits->to_float(bytes, values.data(), cols);
        }
    }
};

static ggml_tensor * input(ggml_context * ctx, const char * name, int64_t a, int64_t b, int64_t c = 1) {
    product({a, b, c});
    auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

static Weight weight(ggml_context * ctx, const char * name, ggml_type type, int64_t cols, int64_t rows) {
    product({cols, rows});
    Weight w;
    w.cols = cols; w.rows = rows; w.type = type;
    w.tensor = ggml_new_tensor_2d(ctx, type, cols, rows);
    ggml_set_name(w.tensor, name);
    ggml_set_input(w.tensor);
    return w;
}

static void set_float(ggml_tensor * tensor, std::vector<float> & values, std::mt19937 & rng) {
    values.resize(static_cast<size_t>(ggml_nelements(tensor)));
    fill(values, rng);
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}

static void compute(Session & s, ggml_cgraph * graph) {
    const ggml_status status = ggml_backend_graph_compute(s.backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed: " + std::to_string(status));
}

// Deterministic spread over output rows and token positions. Whole small outputs
// are checked; large output coverage is explicitly reported rather than implied.
static std::vector<size_t> reference_indices(const Options & o) {
    const size_t count = product({o.hidden, o.q});
    std::vector<size_t> indices;
    if (count <= 512) {
        for (size_t i = 0; i < count; ++i) indices.push_back(i);
        return indices;
    }
    const size_t nt = std::min<size_t>(3, static_cast<size_t>(o.q));
    const size_t nr = std::min<size_t>(512 / nt, static_cast<size_t>(o.hidden));
    for (size_t t = 0; t < nt; ++t) {
        const size_t token = nt == 1 ? 0 : static_cast<size_t>((static_cast<long double>(o.q - 1) * t) / (nt - 1));
        for (size_t r = 0; r < nr; ++r) {
            const size_t row = nr == 1 ? 0 : static_cast<size_t>((static_cast<long double>(o.hidden - 1) * r) / (nr - 1));
            indices.push_back(token * static_cast<size_t>(o.hidden) + row);
        }
    }
    return indices;
}

static double dot(const std::vector<float> & row, const float * x) {
    double result = 0;
    for (size_t i = 0; i < row.size(); ++i) result += static_cast<double>(row[i]) * x[i];
    return result;
}

static std::map<size_t, std::vector<double>> reference_ffn(const Options & o, const std::vector<size_t> & indices, const Weight & gate, const Weight & up, const std::vector<float> & x) {
    std::map<size_t, std::vector<double>> activations;
    for (const size_t index : indices) activations.emplace(index / static_cast<size_t>(o.hidden), std::vector<double>());
    std::vector<float> grow, urow;
    for (auto & token : activations) {
        auto & a = token.second;
        a.resize(static_cast<size_t>(o.intermediate));
        const float * values = x.data() + token.first * static_cast<size_t>(o.hidden);
        for (int64_t r = 0; r < o.intermediate; ++r) {
            gate.row(r, grow); up.row(r, urow);
            const double g = dot(grow, values), u = dot(urow, values);
            a[static_cast<size_t>(r)] = g / (1.0 + std::exp(-g)) * u;
        }
    }
    return activations;
}

static std::map<size_t, std::vector<double>> reference_attention(const Options & o, const std::vector<size_t> & indices, const std::vector<float> & query, const std::vector<float> & k, const std::vector<float> & v, const std::vector<float> & gate) {
    const size_t d = static_cast<size_t>(o.head_dim), nh = static_cast<size_t>(o.heads), length = static_cast<size_t>(o.history + o.q);
    std::map<size_t, std::vector<double>> result;
    for (const size_t index : indices) result.emplace(index / static_cast<size_t>(o.hidden), std::vector<double>());
    for (auto & token : result) {
        auto & output = token.second;
        output.assign(product({o.head_dim, o.heads}), 0.0);
        const size_t visible = static_cast<size_t>(o.history) + token.first + 1;
        std::vector<double> scores(visible);
        for (size_t h = 0; h < nh; ++h) {
            const size_t kh = h / static_cast<size_t>(o.heads / o.kv_heads);
            const float * qr = query.data() + d * (token.first + static_cast<size_t>(o.q) * h);
            double largest = -std::numeric_limits<double>::infinity();
            for (size_t s = 0; s < visible; ++s) {
                const float * kr = k.data() + d * (s + length * kh);
                double score = 0;
                for (size_t j = 0; j < d; ++j) score += static_cast<double>(kr[j]) * qr[j];
                scores[s] = score / std::sqrt(static_cast<double>(d));
                largest = std::max(largest, scores[s]);
            }
            double sum = 0;
            for (double & score : scores) { score = std::exp(score - largest); sum += score; }
            for (size_t s = 0; s < visible; ++s) {
                const float * vr = v.data() + d * (s + length * kh);
                for (size_t j = 0; j < d; ++j) output[h * d + j] += scores[s] / sum * vr[j];
            }
        }
        if (o.attention_gate == "sigmoid") {
            for (size_t j = 0; j < output.size(); ++j) output[j] /= 1.0 + std::exp(-static_cast<double>(gate[token.first * output.size() + j]));
        }
    }
    return result;
}

static int run(const Options & o) {
    const bool attention = o.macro == "attention";
    const auto type_of = [](const std::string & dtype) { return dtype == "f32" ? GGML_TYPE_F32 : dtype == "f16" ? GGML_TYPE_F16 : dtype == "q4_k" ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K; };
    const auto is_quantized = [](const std::string & dtype) { return dtype == "q4_k" || dtype == "q6_k"; };
    const bool quantized = attention ? is_quantized(o.dtype) : is_quantized(o.gate_dtype) || is_quantized(o.up_dtype) || is_quantized(o.down_dtype);
    const size_t embedding = attention ? product({o.heads, o.head_dim}) : static_cast<size_t>(o.hidden);
    const int64_t length = attention ? o.history + o.q : 0;
    Session s;
    s.backend = ggml_backend_cpu_init();
    if (!s.backend) throw std::runtime_error("CPU backend initialization failed");
    ggml_threadpool_params p = ggml_threadpool_params_default(o.threads);
    s.pool = ggml_threadpool_new(&p);
    if (!s.pool) throw std::runtime_error("threadpool initialization failed");
    ggml_backend_cpu_set_n_threads(s.backend, o.threads);
    ggml_backend_cpu_set_threadpool(s.backend, s.pool);
    s.resident = ggml_init({ggml_tensor_overhead() * 24, nullptr, true});
    s.ctx = ggml_init({ggml_tensor_overhead() * 80 + ggml_graph_overhead_custom(128, false), nullptr, true});
    if (!s.resident || !s.ctx) throw std::runtime_error("tensor context allocation failed");
    ggml_tensor * x = nullptr, * query = nullptr, * knew = nullptr, * vnew = nullptr, * kcache = nullptr, * vcache = nullptr, * agate = nullptr;
    std::vector<Weight> weights;
    if (!attention) {
        x = input(s.resident, "ffn_input", o.hidden, o.q);
        weights.push_back(weight(s.resident, "gate_weight", type_of(o.gate_dtype), o.hidden, o.intermediate));
        weights.push_back(weight(s.resident, "up_weight", type_of(o.up_dtype), o.hidden, o.intermediate));
        weights.push_back(weight(s.resident, "down_weight", type_of(o.down_dtype), o.intermediate, o.hidden));
    } else {
        query = input(s.resident, "query", o.head_dim, o.q, o.heads);
        knew = input(s.resident, "new_key", o.head_dim, o.q, o.kv_heads);
        vnew = input(s.resident, "new_value", o.head_dim, o.q, o.kv_heads);
        kcache = input(s.resident, "key_cache", o.head_dim, length, o.kv_heads);
        vcache = input(s.resident, "value_cache", o.head_dim, length, o.kv_heads);
        if (o.attention_gate == "sigmoid") agate = input(s.resident, "attention_gate_input", static_cast<int64_t>(embedding), o.q);
        weights.push_back(weight(s.resident, "output_projection_weight", type_of(o.dtype), static_cast<int64_t>(embedding), o.hidden));
    }
    auto * output = input(s.resident, "macro_output", o.hidden, o.q);
    ggml_set_output(output);
    s.buffer = ggml_backend_alloc_ctx_tensors(s.resident, s.backend);
    if (!s.buffer) throw std::runtime_error("resident allocation failed");
    auto * graph = ggml_new_graph_custom(s.ctx, 128, false);
    ggml_tensor * result = nullptr;
    if (!attention) {
        auto * gate = ggml_mul_mat(s.ctx, weights[0].tensor, x);
        auto * up = ggml_mul_mat(s.ctx, weights[1].tensor, x);
        auto * active = ggml_swiglu_split(s.ctx, gate, up);
        result = ggml_mul_mat(s.ctx, weights[2].tensor, active);
    } else {
        // Writes are in the timed graph. Each repetition overwrites the same new
        // slots; history is never extended by the measurement loop.
        const size_t offset = product({o.head_dim}) * static_cast<size_t>(o.history) * sizeof(float);
        auto * kslot = ggml_view_3d(s.ctx, kcache, o.head_dim, o.q, o.kv_heads, kcache->nb[1], kcache->nb[2], offset);
        auto * vslot = ggml_view_3d(s.ctx, vcache, o.head_dim, o.q, o.kv_heads, vcache->nb[1], vcache->nb[2], offset);
        ggml_build_forward_expand(graph, ggml_cpy(s.ctx, knew, kslot));
        ggml_build_forward_expand(graph, ggml_cpy(s.ctx, vnew, vslot));
        // Explicit graph order places the cache writes ahead of all readers.
        auto * scores = ggml_mul_mat(s.ctx, kcache, query);
        scores = ggml_scale(s.ctx, scores, 1.0f / std::sqrt(static_cast<float>(o.head_dim)));
        scores = ggml_diag_mask_inf(s.ctx, scores, static_cast<int>(o.history));
        auto * probabilities = ggml_soft_max(s.ctx, scores);
        auto * vt = ggml_cont(s.ctx, ggml_transpose(s.ctx, vcache));
        auto * attended = ggml_mul_mat(s.ctx, vt, probabilities);
        auto * merged = ggml_cont(s.ctx, ggml_permute(s.ctx, attended, 0, 2, 1, 3));
        merged = ggml_reshape_2d(s.ctx, merged, static_cast<int64_t>(embedding), o.q);
        if (agate) merged = ggml_mul(s.ctx, merged, ggml_sigmoid(s.ctx, agate));
        result = ggml_mul_mat(s.ctx, weights[0].tensor, merged);
    }
    result = ggml_cpy(s.ctx, result, output);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(s.backend, ggml_graph_node(graph, i))) throw std::runtime_error("CPU does not support an operator in the requested macro");
    }
    s.allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!s.allocator || !ggml_gallocr_alloc_graph(s.allocator, graph)) throw std::runtime_error("internal graph allocation failed");
    const size_t internal_bytes = ggml_gallocr_get_buffer_size(s.allocator, 0);
    const size_t cpu_work = ggml_graph_plan(graph, o.threads, s.pool).work_size;
    const size_t workspace = add_bytes(internal_bytes, cpu_work);
    size_t logical_internal = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (!node->view_src) logical_internal = add_bytes(logical_internal, ggml_nbytes(node));
    }

    std::mt19937 rng(o.seed);
    for (auto & w : weights) w.initialize(rng);
    std::vector<float> xf, qf, kn, vn, kh, vh, gf;
    if (x) set_float(x, xf, rng);
    if (query) set_float(query, qf, rng);
    if (knew) set_float(knew, kn, rng);
    if (vnew) set_float(vnew, vn, rng);
    if (kcache) set_float(kcache, kh, rng);
    if (vcache) set_float(vcache, vh, rng);
    if (agate) set_float(agate, gf, rng);
    ggml_backend_synchronize(s.backend);
    // Untimed execution primes the CPU backend's reusable cplan scratch even if
    // the caller asks for zero additional warmup repetitions.
    compute(s, graph);
    for (int i = 0; i < o.warmup; ++i) compute(s, graph);
    std::vector<int64_t> wall_ns;
    for (int i = 0; i < o.samples; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        compute(s, graph);
        const auto end = std::chrono::steady_clock::now();
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns <= 0) throw std::runtime_error("nonpositive timer sample");
        wall_ns.push_back(ns);
    }
    std::vector<float> yf(product({o.hidden, o.q}));
    ggml_backend_tensor_get(output, yf.data(), 0, ggml_nbytes(output));
    for (const float value : yf) if (!std::isfinite(value)) throw std::runtime_error("nonfinite macro output");
    bool append_exact = true, history_unchanged = true;
    if (attention) {
        std::vector<float> actual_k(kh.size()), actual_v(vh.size());
        ggml_backend_tensor_get(kcache, actual_k.data(), 0, ggml_nbytes(kcache));
        ggml_backend_tensor_get(vcache, actual_v.data(), 0, ggml_nbytes(vcache));
        const size_t d = static_cast<size_t>(o.head_dim), l = static_cast<size_t>(length);
        for (size_t h = 0; h < static_cast<size_t>(o.kv_heads); ++h) {
            for (size_t t = 0; t < l; ++t) {
                for (size_t j = 0; j < d; ++j) {
                    const size_t dst = j + d * (t + l * h);
                    if (t < static_cast<size_t>(o.history)) history_unchanged = history_unchanged && actual_k[dst] == kh[dst] && actual_v[dst] == vh[dst];
                    else {
                        const size_t src = j + d * (t - static_cast<size_t>(o.history) + static_cast<size_t>(o.q) * h);
                        append_exact = append_exact && actual_k[dst] == kn[src] && actual_v[dst] == vn[src];
                        kh[dst] = kn[src]; vh[dst] = vn[src];
                    }
                }
            }
        }
    }
    const auto indices = reference_indices(o);
    const auto intermediates = attention ? reference_attention(o, indices, qf, kh, vh, gf) : reference_ffn(o, indices, weights[0], weights[1], xf);
    const Weight & final_weight = attention ? weights[0] : weights[2];
    double error_squared = 0, reference_squared = 0, max_abs = 0, max_relative = 0;
    std::vector<float> row;
    for (const size_t index : indices) {
        final_weight.row(static_cast<int64_t>(index % static_cast<size_t>(o.hidden)), row);
        const auto & active = intermediates.at(index / static_cast<size_t>(o.hidden));
        double reference = 0;
        for (size_t j = 0; j < row.size(); ++j) reference += static_cast<double>(row[j]) * active[j];
        const double error = std::abs(static_cast<double>(yf[index]) - reference);
        error_squared += error * error; reference_squared += reference * reference;
        max_abs = std::max(max_abs, error);
        max_relative = std::max(max_relative, error / std::max(std::abs(reference), 1e-12));
    }
    const double rmse = std::sqrt(error_squared / static_cast<double>(indices.size()));
    const double reference_rms = std::sqrt(reference_squared / static_cast<double>(indices.size()));
    const double nrmse = rmse / std::max(reference_rms, 1e-12);
    const double atol = quantized ? 0.002 * std::sqrt(static_cast<double>(final_weight.cols)) : 0.001;
    const double rtol = quantized ? 0.025 : 0.002;
    const bool correct = (nrmse <= rtol || rmse <= atol) && append_exact && history_unchanged;
    size_t weight_bytes = 0;
    for (const auto & w : weights) weight_bytes = add_bytes(weight_bytes, ggml_nbytes(w.tensor));
    const size_t output_bytes = ggml_nbytes(output);
    size_t input_bytes = x ? ggml_nbytes(x) : ggml_nbytes(query) + ggml_nbytes(knew) + ggml_nbytes(vnew) + (agate ? ggml_nbytes(agate) : 0);
    const size_t kv_history_bytes = attention ? product({o.head_dim, o.kv_heads}) * static_cast<size_t>(o.history) * sizeof(float) * 2 : 0;
    const size_t kv_new_bytes = attention ? product({o.head_dim, o.kv_heads, o.q}) * sizeof(float) * 2 : 0;
    const size_t kv_cache_bytes = add_bytes(kv_history_bytes, kv_new_bytes);
    const size_t resident_logical = add_bytes(add_bytes(weight_bytes, input_bytes), add_bytes(output_bytes, kv_cache_bytes));
    const char * variant = attention ? "ggml_cpu_explicit_causal_gqa_f32_kv_v1" : o.dtype == "mixed" ? "ggml_cpu_swiglu_split_f32_activation_mixed_weight_dtype_v1" : "ggml_cpu_swiglu_split_f32_activation_uniform_weight_dtype_v1";
    std::cout << std::setprecision(17)
        << "{\"schema\":\"ggml_macro_probe/1\",\"kind\":\"macro\",\"macro\":\"" << o.macro << "\",\"backend\":\"CPU\",\"batch\":1"
        << ",\"hidden\":" << o.hidden << ",\"intermediate\":" << o.intermediate << ",\"heads\":" << o.heads
        << ",\"kv_heads\":" << o.kv_heads << ",\"head_dim\":" << o.head_dim << ",\"q\":" << o.q << ",\"history\":" << o.history
        << ",\"dtype\":\"" << o.dtype << "\",\"weight_dtypes\":";
    if (attention) std::cout << "{\"output\":\"" << o.dtype << "\"}";
    else std::cout << "{\"gate\":\"" << o.gate_dtype << "\",\"up\":\"" << o.up_dtype << "\",\"down\":\"" << o.down_dtype << "\"}";
    std::cout << ",\"activation_dtype\":\"f32\",\"kv_dtype\":\"" << (attention ? "f32" : "not_applicable")
        << "\",\"attention_gate\":\"" << o.attention_gate << "\",\"implementation_variant\":\"" << variant
        << "\",\"history_semantics\":\"" << (attention ? "existing_prefix_length_in_causal_attention" : "not_used_by_dense_ffn")
        << "\",\"threads\":" << o.threads << ",\"threads_requested\":" << o.threads << ",\"threads_applied\":" << o.threads
        << ",\"thread_management\":\"persistent_ggml_threadpool\",\"warmup\":" << o.warmup << ",\"setup_compute_repetitions\":1"
        << ",\"samples\":" << o.samples << ",\"seed\":" << o.seed << ",\"input_generator\":\"mt19937_uniform24_v2\""
        << ",\"timing\":\"steady_clock_graph_compute_including_sync\",\"scope\":\"resident_repeated_partial_block_macro\""
        << ",\"included\":" << (attention ? "[\"timed_kv_append\",\"causal_qk_scale_mask_softmax_pv\",\"optional_sigmoid_gate\",\"o_projection\",\"final_output_copy\"]" : "[\"gate_matmul\",\"up_matmul\",\"swiglu_split\",\"down_matmul\",\"final_output_copy\"]")
        << ",\"excluded\":[\"norm\",\"residual\",\"qkv_projection\",\"rope\",\"allocation\",\"quantization\",\"io\",\"external_concurrency\"]"
        << ",\"nodes\":[";
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        std::cout << (i ? "," : "") << "{\"op\":\"" << ggml_op_name(node->op) << "\",\"bytes\":" << ggml_nbytes(node) << ",\"view\":" << (node->view_src ? "true" : "false") << "}";
    }
    std::cout << "]"
        << ",\"weight_bytes\":" << weight_bytes << ",\"input_bytes\":" << input_bytes << ",\"output_bytes\":" << output_bytes
        << ",\"kv_history_bytes\":" << kv_history_bytes << ",\"kv_new_bytes\":" << kv_new_bytes << ",\"kv_cache_bytes\":" << kv_cache_bytes
        << ",\"internal_tensor_bytes_upper_bound\":" << std::max(internal_bytes, logical_internal)
        << ",\"cpu_cplan_work_size_bytes\":" << cpu_work << ",\"workspace_bytes\":" << workspace
        << ",\"workspace_source\":\"ggml_gallocr_internal_buffer_plus_cpu_graph_plan\""
        << ",\"memory_accounting\":{\"resident_logical_bytes\":" << resident_logical
        << ",\"resident_allocated_bytes\":" << ggml_backend_buffer_get_size(s.buffer)
        << ",\"internal_allocated_buffer_bytes\":" << internal_bytes << ",\"internal_logical_tensor_bytes_sum\":" << logical_internal
        << ",\"cpu_cplan_work_size_bytes\":" << cpu_work << ",\"workspace_bytes\":" << workspace
        << ",\"workspace_scope\":\"internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads\""
        << ",\"workspace_bound\":\"conservative_allocated_capacity\",\"kv_new_bytes_semantics\":\"appended_slots_in_cache_distinct_from_new_kv_input_tensors\"}"
        << ",\"all_outputs_finite\":true,\"kv_validation\":";
    if (attention) std::cout << "{\"append_exact\":" << (append_exact ? "true" : "false") << ",\"history_unchanged\":" << (history_unchanged ? "true" : "false") << "}";
    else std::cout << "null";
    std::cout << ",\"kv_append_check\":";
    if (attention) std::cout << "{\"passed\":" << ((append_exact && history_unchanged) ? "true" : "false") << ",\"append_exact\":" << (append_exact ? "true" : "false") << ",\"history_unchanged\":" << (history_unchanged ? "true" : "false") << "}";
    else std::cout << "null";
    std::cout << ",\"reference_check\":{\"passed\":" << (correct ? "true" : "false") << ",\"samples\":" << indices.size()
        << ",\"output_elements\":" << yf.size() << ",\"coverage\":\"" << (indices.size() == yf.size() ? "all_output_elements" : "sampled_rows_at_up_to_three_token_positions")
        << "\",\"reference\":\"independent_double_macro_with_dequantized_stored_weights_and_f32_inputs\""
        << ",\"max_abs_error\":" << max_abs << ",\"max_relative_error\":" << max_relative << ",\"rmse\":" << rmse
        << ",\"reference_rms\":" << reference_rms << ",\"normalized_rmse\":" << nrmse << ",\"atol\":" << atol << ",\"rtol\":" << rtol
        << ",\"criterion\":\"normalized_rmse_le_rtol_or_rmse_le_atol_and_exact_kv_validation\"},\"wall_ns\":[";
    for (size_t i = 0; i < wall_ns.size(); ++i) std::cout << (i ? "," : "") << wall_ns[i];
    std::cout << "]}\n";
    if (!correct) { std::cerr << "macro numerical or KV validation failed\n"; return 2; }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "ggml-macro-probe --macro attention|dense_ffn --hidden H --q Q --history HIST --dtype f32|f16|q4_k|q6_k "
                     "[instead of --dtype: --gate-dtype TYPE --up-dtype TYPE --down-dtype TYPE (mixed FFN)] "
                     "[--intermediate I (FFN)] [--heads NH --kv-heads NKV --head-dim D --attention-gate none|sigmoid (attention)] "
                     "[--threads 16 --warmup 5 --samples 20 --seed 17]; CPU only, batch=1\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception & e) { std::cerr << "ggml-macro-probe: " << e.what() << '\n'; return 1; }
}
