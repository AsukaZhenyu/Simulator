// CUDA resident decoder block with explicit materialized stage boundaries.
// Graph/reference implementation copied from frozen CPU source SHA256:
// 46ceee3d9e6135c1e082b29230b59a06e75eb9e7e288ccab8f57ddc3a6d84eba
// CPU diagnostics remain independent; no CPU backend or scheduler is created.
// This generic contract deliberately does not claim to reproduce a model runtime.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Only the separate memory diagnostic wrapper enables these synchronized observations.
// The normal performance build compiles every checkpoint to a no-op.
#ifdef MEASURE_DENSE_CUDA_MEMORY_PROBE
static void dense_cuda_memory_checkpoint(const char *label);
#define DENSE_CUDA_MEMORY_CHECKPOINT(label) dense_cuda_memory_checkpoint(label)
#else
#define DENSE_CUDA_MEMORY_CHECKPOINT(label) do { } while (false)
#endif
struct Options {
    std::string stage, attention_gate, qk_norm, rope_mode, cuda_graphs;
    std::map<std::string, std::string> dtypes;
    int64_t hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0, q = 0,
            history = -1, rope_dims = 0;
    float rope_base = 0, norm_eps = 0;
    int cuda_device = -1, warmup = -1, samples = 0;
    uint32_t seed = 0;
};
static int64_t integer(const std::string &text) {
    size_t n = 0;
    const auto v = std::stoll(text, &n);
    if (n != text.size())
        throw std::runtime_error("invalid integer: " + text);
    return v;
}
static float real(const std::string &text) {
    size_t n = 0;
    const float v = std::stof(text, &n);
    if (n != text.size() || !std::isfinite(v) || v <= 0)
        throw std::runtime_error("invalid positive float: " + text);
    return v;
}
static Options parse(int argc, char **argv) {
    Options o;
    std::set<std::string> seen;
    const std::map<std::string, std::string> dtype_keys = {
        {"--q-dtype", "q"},      {"--k-dtype", "k"},
        {"--v-dtype", "v"},      {"--gate-proj-dtype", "gate_proj"},
        {"--o-dtype", "output"}, {"--ffn-gate-dtype", "gate"},
        {"--up-dtype", "up"},    {"--down-dtype", "down"}};
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc || !seen.insert(key).second)
            throw std::runtime_error("missing value or duplicate option: " + key);
        const std::string v = argv[i + 1];
        if (dtype_keys.count(key))
            o.dtypes[dtype_keys.at(key)] = v;
        else if (key == "--stage")
            o.stage = v;
        else if (key == "--cuda-graphs")
            o.cuda_graphs = v;
        else if (key == "--attention-gate")
            o.attention_gate = v;
        else if (key == "--qk-norm")
            o.qk_norm = v;
        else if (key == "--rope-mode")
            o.rope_mode = v;
        else if (key == "--hidden")
            o.hidden = integer(v);
        else if (key == "--intermediate")
            o.intermediate = integer(v);
        else if (key == "--heads")
            o.heads = integer(v);
        else if (key == "--kv-heads")
            o.kv_heads = integer(v);
        else if (key == "--head-dim")
            o.head_dim = integer(v);
        else if (key == "--q")
            o.q = integer(v);
        else if (key == "--history")
            o.history = integer(v);
        else if (key == "--rope-dims")
            o.rope_dims = integer(v);
        else if (key == "--rope-base")
            o.rope_base = real(v);
        else if (key == "--norm-eps")
            o.norm_eps = real(v);
        else if (key == "--seed") {
            const auto t = integer(v);
            if (t < 0 || t > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("seed out of range");
            o.seed = static_cast<uint32_t>(t);
        } else if (key == "--cuda-device" || key == "--warmup" || key == "--samples") {
            const auto t = integer(v);
            if (t < 0 || t > std::numeric_limits<int>::max())
                throw std::runtime_error("option out of range: " + key);
            if (key == "--cuda-device")
                o.cuda_device = static_cast<int>(t);
            if (key == "--warmup")
                o.warmup = static_cast<int>(t);
            if (key == "--samples")
                o.samples = static_cast<int>(t);
        } else
            throw std::runtime_error("unknown option: " + key);
    }
    const std::set<std::string> stages = {"projection", "attention", "dense_ffn", "block"};
    if (!stages.count(o.stage))
        throw std::runtime_error("unsupported stage");
    if (!seen.count("--cuda-device") || o.cuda_device < 0 || o.cuda_device >= GGML_CUDA_MAX_DEVICES)
        throw std::runtime_error("requires explicit --cuda-device in CUDA device range");
    if (!seen.count("--cuda-graphs") || o.cuda_graphs != "disabled")
        throw std::runtime_error(
            "this variant requires explicit --cuda-graphs disabled; no fallback");
    if (o.hidden <= 0 || o.intermediate <= 0 || o.heads <= 0 || o.kv_heads <= 0 ||
        o.head_dim <= 0 || o.heads % o.kv_heads || o.q <= 0 || o.history < 0)
        throw std::runtime_error("invalid block geometry");
    if (o.history > std::numeric_limits<int>::max() ||
        o.q > std::numeric_limits<int>::max() - o.history)
        throw std::runtime_error("sequence indexing overflow");
    if (o.attention_gate != "none" && o.attention_gate != "sigmoid")
        throw std::runtime_error("attention_gate must be none or sigmoid");
    if (o.qk_norm != "none" && o.qk_norm != "rms")
        throw std::runtime_error("qk_norm must be none or rms");
    if (o.rope_mode != "normal" || o.rope_dims <= 0 || o.rope_dims % 2 ||
        o.rope_dims > o.head_dim || o.rope_base <= 1 || o.norm_eps <= 0)
        throw std::runtime_error("requires normal RoPE, positive even dimensions <= head_dim, "
                                 "explicit base and epsilon");
    if (o.warmup < 0 || o.samples < 3 || o.samples > 10000 || !seen.count("--seed"))
        throw std::runtime_error("requires explicit warmup/samples/seed, samples in 3..10000");
    const std::set<std::string> types = {"f32", "f16", "q4_k", "q6_k"};
    for (const auto &role : {"q", "k", "v", "output", "gate", "up", "down"})
        if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
            throw std::runtime_error(std::string("missing/unsupported matrix dtype: ") + role);
    if (o.attention_gate == "sigmoid") {
        if (!o.dtypes.count("gate_proj") || !types.count(o.dtypes.at("gate_proj")))
            throw std::runtime_error("sigmoid gate requires gate projection dtype");
    } else {
        if (o.dtypes.count("gate_proj") && o.dtypes.at("gate_proj") != "none")
            throw std::runtime_error("inactive gate projection dtype must be none");
        o.dtypes.erase("gate_proj");
    }
    o.dtypes["input_norm"] = "f32";
    o.dtypes["post_attention_norm"] = "f32";
    if (o.qk_norm == "rms") {
        o.dtypes["q_norm"] = "f32";
        o.dtypes["k_norm"] = "f32";
    }
    return o;
}
static size_t product(std::initializer_list<int64_t> dims) {
    size_t result = 1;
    const size_t limit = static_cast<size_t>(
        std::min<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float),
                           std::numeric_limits<int64_t>::max() / sizeof(float)));
    for (const int64_t d : dims) {
        if (d <= 0 || static_cast<uint64_t>(d) > limit / result)
            throw std::runtime_error("shape overflows addressable storage");
        result *= static_cast<size_t>(d);
    }
    return result;
}

static size_t add_bytes(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("memory accounting overflow");
    return a + b;
}

static void fill(std::vector<float> &values, std::mt19937 &rng) {
    for (float &v : values)
        v = static_cast<float>(static_cast<int32_t>(rng() >> 8) - 8388608) / 16777216.0f;
}
struct Session {
    ggml_context *resident = nullptr, *ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~Session() {
        if (backend)
            ggml_backend_synchronize(backend);
        if (allocator)
            ggml_gallocr_free(allocator);
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (backend)
            ggml_backend_free(backend);
        if (ctx)
            ggml_free(ctx);
        if (resident)
            ggml_free(resident);
    }
};
struct Weight {
    ggml_tensor *tensor = nullptr;
    int64_t cols = 0, rows = 0;
    ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> packed;

    void initialize(std::mt19937 &rng) {
        std::vector<float> values(product({cols, rows}));
        fill(values, rng);
        for (float &value : values)
            value /= std::sqrt(static_cast<float>(cols));
        packed.resize(ggml_nbytes(tensor));
        if (type == GGML_TYPE_F32)
            std::memcpy(packed.data(), values.data(), packed.size());
        else if (type == GGML_TYPE_F16)
            ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()),
                                  static_cast<int64_t>(values.size()));
        else if (ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows, cols, nullptr) !=
                 packed.size())
            throw std::runtime_error("weight quantization size mismatch");
        ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
    }

    void row(int64_t index, std::vector<float> &values) const {
        values.resize(static_cast<size_t>(cols));
        const uint8_t *bytes =
            packed.data() + static_cast<size_t>(index) * ggml_row_size(type, cols);
        if (type == GGML_TYPE_F32)
            std::memcpy(values.data(), bytes, values.size() * sizeof(float));
        else {
            const auto *traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float)
                throw std::runtime_error("weight dequantizer unavailable");
            traits->to_float(bytes, values.data(), cols);
        }
    }
};

static ggml_tensor *input(ggml_context *ctx, const char *name, int64_t a, int64_t b,
                          int64_t c = 1) {
    product({a, b, c});
    auto *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

static void compute(Session &s, ggml_cgraph *graph) {
    const ggml_status status = ggml_backend_graph_compute_async(s.backend, graph);
    if (status != GGML_STATUS_SUCCESS)
        throw std::runtime_error("CUDA graph compute failed: " + std::to_string(status));
    ggml_backend_synchronize(s.backend); // wall timing includes completed CUDA work
}

static double dot(const std::vector<float> &row, const float *x) {
    double result = 0;
    for (size_t i = 0; i < row.size(); ++i)
        result += static_cast<double>(row[i]) * x[i];
    return result;
}

static ggml_type type_of(const std::string &t) {
    if (t == "f32")
        return GGML_TYPE_F32;
    if (t == "f16")
        return GGML_TYPE_F16;
    if (t == "q4_k")
        return GGML_TYPE_Q4_K;
    if (t == "q6_k")
        return GGML_TYPE_Q6_K;
    throw std::runtime_error("unknown dtype");
}
struct Data {
    std::map<std::string, Weight> weights;
    std::vector<float> x, kcache, vcache, query, key, value, agate, attn_hidden, final_hidden;
    std::vector<int32_t> positions;
};
static Data make_data(const Options &o) {
    Data d;
    std::mt19937 rng(o.seed);
    const auto e = static_cast<int64_t>(product({o.head_dim, o.heads}));
    const auto kv = static_cast<int64_t>(product({o.head_dim, o.kv_heads}));
    auto add = [&](const std::string &role, int64_t cols, int64_t rows, bool gamma = false) {
        Weight w;
        w.cols = cols;
        w.rows = rows;
        w.type = type_of(o.dtypes.at(role));
        if ((w.type == GGML_TYPE_Q4_K || w.type == GGML_TYPE_Q6_K) && cols % 256)
            throw std::runtime_error("quantized reduction dimension must be divisible by 256: " +
                                     role);
        std::vector<float> f(product({cols, rows}));
        fill(f, rng);
        for (float &v : f)
            v = gamma ? 1.0f + 0.4f * v : v / std::sqrt(static_cast<float>(cols));
        w.packed.resize(ggml_row_size(w.type, cols) * static_cast<size_t>(rows));
        if (w.type == GGML_TYPE_F32)
            std::memcpy(w.packed.data(), f.data(), w.packed.size());
        else if (w.type == GGML_TYPE_F16)
            ggml_fp32_to_fp16_row(f.data(), reinterpret_cast<ggml_fp16_t *>(w.packed.data()),
                                  static_cast<int64_t>(f.size()));
        else if (ggml_quantize_chunk(w.type, f.data(), w.packed.data(), 0, rows, cols, nullptr) !=
                 w.packed.size())
            throw std::runtime_error("quantization size mismatch");
        d.weights.emplace(role, std::move(w));
    };
    add("q", o.hidden, e);
    add("k", o.hidden, kv);
    add("v", o.hidden, kv);
    if (o.attention_gate == "sigmoid")
        add("gate_proj", o.hidden, e);
    add("output", e, o.hidden);
    add("gate", o.hidden, o.intermediate);
    add("up", o.hidden, o.intermediate);
    add("down", o.intermediate, o.hidden);
    add("input_norm", o.hidden, 1, true);
    add("post_attention_norm", o.hidden, 1, true);
    if (o.qk_norm == "rms") {
        add("q_norm", o.head_dim, 1, true);
        add("k_norm", o.head_dim, 1, true);
    }
    d.x.resize(product({o.hidden, o.q}));
    fill(d.x, rng);
    d.kcache.resize(product({o.head_dim, o.history + o.q, o.kv_heads}));
    fill(d.kcache, rng);
    d.vcache.resize(d.kcache.size());
    fill(d.vcache, rng);
    for (int64_t i = 0; i < o.q; ++i)
        d.positions.push_back(static_cast<int32_t>(o.history + i));
    return d;
}
struct Stage {
    Session s;
    ggml_cgraph *graph = nullptr;
    std::map<std::string, ggml_tensor *> tensors, weights, ports;
    std::map<std::string, std::pair<int, int>> node_ranges;
    size_t input_bytes = 0, output_bytes = 0, weight_bytes = 0, kv_bytes = 0, internal_bytes = 0,
           logical_internal = 0;
};
static std::vector<std::string> roles_for(const Options &o, const std::string &stage) {
    std::vector<std::string> roles;
    if (stage == "projection" || stage == "block") {
        roles = {"input_norm", "q", "k", "v"};
        if (o.attention_gate == "sigmoid")
            roles.push_back("gate_proj");
        if (o.qk_norm == "rms") {
            roles.push_back("q_norm");
            roles.push_back("k_norm");
        }
    }
    if (stage == "attention" || stage == "block")
        roles.push_back("output");
    if (stage == "dense_ffn" || stage == "block")
        for (const auto &r : {"post_attention_norm", "gate", "up", "down"})
            roles.push_back(r);
    return roles;
}
static std::unique_ptr<Stage> build_stage(const Options &o, const Data &d,
                                          const std::string &stage) {
    auto p = std::make_unique<Stage>();
    auto &s = p->s;
    const int64_t e = static_cast<int64_t>(product({o.head_dim, o.heads})),
                  length = o.history + o.q;
    s.backend = ggml_backend_cuda_init(o.cuda_device);
    if (!s.backend || !ggml_backend_is_cuda(s.backend))
        throw std::runtime_error("requested CUDA backend initialization failed");
    s.resident = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
    s.ctx = ggml_init(
        {ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false), nullptr, true});
    if (!s.resident || !s.ctx)
        throw std::runtime_error("tensor metadata allocation failed");
    auto add = [&](const std::string &name, int64_t a, int64_t b, int64_t c, bool output,
                   bool cache = false) {
        auto *t = input(s.resident, name.c_str(), a, b, c);
        p->tensors[name] = t;
        if (cache)
            p->kv_bytes = add_bytes(p->kv_bytes, ggml_nbytes(t));
        else if (output) {
            p->output_bytes = add_bytes(p->output_bytes, ggml_nbytes(t));
            p->ports[name] = t;
            ggml_set_output(t);
        } else
            p->input_bytes = add_bytes(p->input_bytes, ggml_nbytes(t));
        return t;
    };
    const bool projection = stage == "projection" || stage == "block",
               attention = stage == "attention" || stage == "block",
               ffn = stage == "dense_ffn" || stage == "block";
    if (projection || attention)
        add("hidden_input", o.hidden, o.q, 1, false);
    if (projection) {
        auto *pos = ggml_new_tensor_1d(s.resident, GGML_TYPE_I32, o.q);
        ggml_set_name(pos, "positions");
        ggml_set_input(pos);
        p->tensors["positions"] = pos;
        p->input_bytes = add_bytes(p->input_bytes, ggml_nbytes(pos));
    }
    if (projection || attention) {
        add("query", o.head_dim, o.q, o.heads, projection);
        add("new_key", o.head_dim, o.q, o.kv_heads, projection);
        add("new_value", o.head_dim, o.q, o.kv_heads, projection);
        if (o.attention_gate == "sigmoid")
            add("attention_gate", e, o.q, 1, projection);
    }
    if (attention) {
        add("key_cache", o.head_dim, length, o.kv_heads, false, true);
        add("value_cache", o.head_dim, length, o.kv_heads, false, true);
    }
    if (attention || ffn)
        add("attention_hidden", o.hidden, o.q, 1, attention);
    if (ffn)
        add("hidden_output", o.hidden, o.q, 1, true);
    for (const auto &role : roles_for(o, stage)) {
        const auto &w = d.weights.at(role);
        auto *t = ggml_new_tensor_2d(s.resident, w.type, w.cols, w.rows);
        ggml_set_name(t, role.c_str());
        ggml_set_input(t);
        p->weights[role] = t;
        p->weight_bytes = add_bytes(p->weight_bytes, ggml_nbytes(t));
    }
    s.buffer = ggml_backend_alloc_ctx_tensors(s.resident, s.backend);
    if (!s.buffer)
        throw std::runtime_error("resident allocation failed");
    for (const auto &w : p->weights)
        ggml_backend_tensor_set(w.second, d.weights.at(w.first).packed.data(), 0,
                                ggml_nbytes(w.second));
    auto put = [&](const std::string &name, const std::vector<float> &f) {
        if (p->tensors.count(name)) {
            auto *t = p->tensors.at(name);
            if (f.size() != static_cast<size_t>(ggml_nelements(t)))
                throw std::runtime_error("unprepared stage input: " + name);
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        }
    };
    put("hidden_input", d.x);
    put("key_cache", d.kcache);
    put("value_cache", d.vcache);
    if (!projection && attention) {
        put("query", d.query);
        put("new_key", d.key);
        put("new_value", d.value);
        if (o.attention_gate == "sigmoid")
            put("attention_gate", d.agate);
    }
    if (!attention && ffn)
        put("attention_hidden", d.attn_hidden);
    if (projection)
        ggml_backend_tensor_set(p->tensors.at("positions"), d.positions.data(), 0,
                                static_cast<size_t>(o.q) * sizeof(int32_t));
    p->graph = ggml_new_graph_custom(s.ctx, 512, false);
    auto *ctx = s.ctx;
    auto materialize = [&](const std::string &name, ggml_tensor *v) {
        auto *out = ggml_cpy(ctx, v, p->tensors.at(name));
        ggml_set_output(out);
        ggml_build_forward_expand(p->graph, out);
        return out;
    };
    auto norm = [&](ggml_tensor *v, const std::string &role) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, v, o.norm_eps), p->weights.at(role));
    };
    ggml_tensor *query = nullptr, *key = nullptr, *value = nullptr, *agate = nullptr,
                *hidden = nullptr;
    if (projection) {
        auto *n = norm(p->tensors.at("hidden_input"), "input_norm");
        auto qkv = [&](const std::string &role, int64_t heads, bool rotary) {
            auto *v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, p->weights.at(role), n), o.head_dim,
                                      heads, o.q);
            if (rotary) {
                if (o.qk_norm == "rms")
                    v = norm(v, role == "q" ? "q_norm" : "k_norm");
                v = ggml_rope_ext(ctx, v, p->tensors.at("positions"), nullptr,
                                  static_cast<int>(o.rope_dims), GGML_ROPE_TYPE_NORMAL, 0,
                                  o.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            return ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        };
        query = materialize("query", qkv("q", o.heads, true));
        key = materialize("new_key", qkv("k", o.kv_heads, true));
        value = materialize("new_value", qkv("v", o.kv_heads, false));
        if (o.attention_gate == "sigmoid")
            agate = materialize("attention_gate", ggml_mul_mat(ctx, p->weights.at("gate_proj"), n));
        p->node_ranges["projection"] = {0, ggml_graph_n_nodes(p->graph)};
    } else if (attention) {
        query = p->tensors.at("query");
        key = p->tensors.at("new_key");
        value = p->tensors.at("new_value");
        if (o.attention_gate == "sigmoid")
            agate = p->tensors.at("attention_gate");
    }
    if (attention) {
        auto *kc = p->tensors.at("key_cache"), *vc = p->tensors.at("value_cache");
        const size_t offset =
            static_cast<size_t>(o.history) * static_cast<size_t>(o.head_dim) * sizeof(float);
        // SET-inplace returns the complete cache with an explicit dependency on
        // new K/V. Readers therefore depend on the writes, including in block.
        auto *kr = ggml_set_inplace(ctx, kc, key, kc->nb[1], kc->nb[2], kc->nb[3], offset);
        auto *vr = ggml_set_inplace(ctx, vc, value, vc->nb[1], vc->nb[2], vc->nb[3], offset);
        auto *scores = ggml_mul_mat(ctx, kr, query);
        scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(o.head_dim)));
        scores = ggml_diag_mask_inf(ctx, scores, static_cast<int>(o.history));
        auto *probs = ggml_soft_max(ctx, scores);
        auto *vt = ggml_cont(ctx, ggml_transpose(ctx, vr));
        auto *attended = ggml_mul_mat(ctx, vt, probs);
        auto *merged =
            ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), e, o.q);
        if (agate)
            merged = ggml_mul(ctx, merged, ggml_sigmoid(ctx, agate));
        hidden = materialize("attention_hidden",
                             ggml_add(ctx, ggml_mul_mat(ctx, p->weights.at("output"), merged),
                                      p->tensors.at("hidden_input")));
        p->node_ranges["attention"] = {projection ? p->node_ranges.at("projection").second : 0,
                                       ggml_graph_n_nodes(p->graph)};
    } else if (ffn)
        hidden = p->tensors.at("attention_hidden");
    if (ffn) {
        auto *n = norm(hidden, "post_attention_norm");
        auto *g = ggml_mul_mat(ctx, p->weights.at("gate"), n);
        auto *u = ggml_mul_mat(ctx, p->weights.at("up"), n);
        auto *active = ggml_swiglu_split(ctx, g, u);
        materialize("hidden_output",
                    ggml_add(ctx, ggml_mul_mat(ctx, p->weights.at("down"), active), hidden));
        p->node_ranges["dense_ffn"] = {attention ? p->node_ranges.at("attention").second : 0,
                                       ggml_graph_n_nodes(p->graph)};
    }
    for (int i = 0; i < ggml_graph_n_nodes(p->graph); ++i)
        if (!ggml_backend_supports_op(s.backend, ggml_graph_node(p->graph, i)))
            throw std::runtime_error("unsupported requested CUDA operator at node " +
                                     std::to_string(i) + ": " +
                                     ggml_op_name(ggml_graph_node(p->graph, i)->op));
    s.allocator = ggml_gallocr_new(ggml_backend_cuda_buffer_type(o.cuda_device));
    if (!s.allocator || !ggml_gallocr_alloc_graph(s.allocator, p->graph))
        throw std::runtime_error("graph allocation failed");
    p->internal_bytes = ggml_gallocr_get_buffer_size(s.allocator, 0);

    for (int i = 0; i < ggml_graph_n_nodes(p->graph); ++i) {
        const auto *n = ggml_graph_node(p->graph, i);
        if (!n->view_src)
            p->logical_internal = add_bytes(p->logical_internal, ggml_nbytes(n));
    }
    const auto requested = ggml_backend_cuda_buffer_type(o.cuda_device);
    auto require_cuda = [&](const ggml_tensor *tensor) {
        const ggml_tensor *base = tensor;
        while (base->view_src)
            base = base->view_src;
        if (!base->buffer || ggml_backend_buffer_get_type(base->buffer) != requested)
            throw std::runtime_error("tensor is not resident on requested CUDA device: " +
                                     std::string(tensor->name));
    };
    for (const auto &item : p->tensors)
        require_cuda(item.second);
    for (const auto &item : p->weights)
        require_cuda(item.second);
    for (int i = 0; i < ggml_graph_n_nodes(p->graph); ++i) {
        const auto *node = ggml_graph_node(p->graph, i);
        require_cuda(node);
        for (int j = 0; j < GGML_MAX_SRC; ++j)
            if (node->src[j])
                require_cuda(node->src[j]);
        if (!ggml_backend_supports_op(s.backend, node))
            throw std::runtime_error("allocated CUDA operator unsupported: " + std::to_string(i));
    }
    return p;
}
static std::vector<float> read_tensor(ggml_tensor *t) {
    std::vector<float> f(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    return f;
}
static void capture(Stage &s, Data &d) {
    d.query = read_tensor(s.ports.at("query"));
    d.key = read_tensor(s.ports.at("new_key"));
    d.value = read_tensor(s.ports.at("new_value"));
    if (s.ports.count("attention_gate"))
        d.agate = read_tensor(s.ports.at("attention_gate"));
    d.attn_hidden = read_tensor(s.ports.at("attention_hidden"));
    d.final_hidden = read_tensor(s.ports.at("hidden_output"));
}
static std::vector<float> normalized(const std::vector<float> &x, int64_t width,
                                     const Weight &gamma, float eps) {
    std::vector<float> y(x.size()), g;
    gamma.row(0, g);
    for (size_t t = 0; t < x.size() / static_cast<size_t>(width); ++t) {
        double sum = 0;
        for (int64_t j = 0; j < width; ++j) {
            const double v = x[t * static_cast<size_t>(width) + static_cast<size_t>(j)];
            sum += v * v;
        }
        const double scale = 1.0 / std::sqrt(sum / static_cast<double>(width) + eps);
        for (int64_t j = 0; j < width; ++j)
            y[t * static_cast<size_t>(width) + static_cast<size_t>(j)] =
                static_cast<float>(x[t * static_cast<size_t>(width) + static_cast<size_t>(j)] *
                                   scale * g[static_cast<size_t>(j)]);
    }
    return y;
}
static std::vector<float> matmul(const Weight &w, const std::vector<float> &x, int64_t tokens) {
    std::vector<float> y(product({w.rows, tokens})), row;
    for (int64_t r = 0; r < w.rows; ++r) {
        w.row(r, row);
        for (int64_t t = 0; t < tokens; ++t)
            y[static_cast<size_t>(r + w.rows * t)] =
                static_cast<float>(dot(row, x.data() + static_cast<size_t>(w.cols * t)));
    }
    return y;
}
static std::vector<float> projected(const Options &o, const Data &d, const std::vector<float> &norm,
                                    const std::string &role, int64_t heads, bool rotary) {
    auto f = matmul(d.weights.at(role), norm, o.q);
    if (rotary && o.qk_norm == "rms")
        f = normalized(f, o.head_dim, d.weights.at(role == "q" ? "q_norm" : "k_norm"), o.norm_eps);
    if (rotary) {
        for (int64_t t = 0; t < o.q; ++t)
            for (int64_t h = 0; h < heads; ++h) {
                const size_t offset = static_cast<size_t>(o.head_dim * (h + heads * t));
                for (int64_t j = 0; j < o.rope_dims; j += 2) {
                    const double theta =
                        static_cast<double>(o.history + t) *
                        std::pow(static_cast<double>(o.rope_base),
                                 -static_cast<double>(j) / static_cast<double>(o.rope_dims));
                    const double c = std::cos(theta), s = std::sin(theta),
                                 a = f[offset + static_cast<size_t>(j)],
                                 b = f[offset + static_cast<size_t>(j + 1)];
                    f[offset + static_cast<size_t>(j)] = static_cast<float>(a * c - b * s);
                    f[offset + static_cast<size_t>(j + 1)] = static_cast<float>(a * s + b * c);
                }
            }
    }
    std::vector<float> result(f.size());
    for (int64_t t = 0; t < o.q; ++t)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t j = 0; j < o.head_dim; ++j)
                result[static_cast<size_t>(j + o.head_dim * (t + o.q * h))] =
                    f[static_cast<size_t>(j + o.head_dim * (h + heads * t))];
    return result;
}
static std::vector<float> attention_reference(const Options &o, const Data &d,
                                              const std::vector<float> &query,
                                              const std::vector<float> &key,
                                              const std::vector<float> &value,
                                              const std::vector<float> &gate) {
    const auto len = o.history + o.q, e = o.head_dim * o.heads;
    auto k = d.kcache, v = d.vcache;
    for (int64_t h = 0; h < o.kv_heads; ++h)
        for (int64_t t = 0; t < o.q; ++t)
            for (int64_t j = 0; j < o.head_dim; ++j) {
                const size_t dst = static_cast<size_t>(j + o.head_dim * (o.history + t + len * h)),
                             src = static_cast<size_t>(j + o.head_dim * (t + o.q * h));
                k[dst] = key[src];
                v[dst] = value[src];
            }
    std::vector<float> merged(product({e, o.q}));
    for (int64_t t = 0; t < o.q; ++t)
        for (int64_t h = 0; h < o.heads; ++h) {
            const auto kh = h / (o.heads / o.kv_heads), visible = o.history + t + 1;
            std::vector<double> scores(static_cast<size_t>(visible));
            double largest = -std::numeric_limits<double>::infinity(), sum = 0;
            for (int64_t s = 0; s < visible; ++s) {
                double score = 0;
                for (int64_t j = 0; j < o.head_dim; ++j)
                    score += static_cast<double>(
                                 query[static_cast<size_t>(j + o.head_dim * (t + o.q * h))]) *
                             k[static_cast<size_t>(j + o.head_dim * (s + len * kh))];
                scores[static_cast<size_t>(s)] = score / std::sqrt(static_cast<double>(o.head_dim));
                largest = std::max(largest, scores[static_cast<size_t>(s)]);
            }
            for (auto &p : scores) {
                p = std::exp(p - largest);
                sum += p;
            }
            for (int64_t j = 0; j < o.head_dim; ++j) {
                double y = 0;
                for (int64_t s = 0; s < visible; ++s)
                    y += scores[static_cast<size_t>(s)] / sum *
                         v[static_cast<size_t>(j + o.head_dim * (s + len * kh))];
                const size_t dst = static_cast<size_t>(j + o.head_dim * h + e * t);
                if (o.attention_gate == "sigmoid")
                    y /= 1 + std::exp(-static_cast<double>(gate[dst]));
                merged[dst] = static_cast<float>(y);
            }
        }
    auto out = matmul(d.weights.at("output"), merged, o.q);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] += d.x[i];
    return out;
}
static std::vector<size_t> checked_indices(size_t n, int64_t width, int64_t tokens) {
    std::vector<size_t> ix;
    if (n <= 512) {
        for (size_t i = 0; i < n; ++i)
            ix.push_back(i);
    } else {
        const size_t nt = std::min<size_t>(3, static_cast<size_t>(tokens)),
                     nr = std::min<size_t>(512 / nt, static_cast<size_t>(width));
        for (size_t t = 0; t < nt; ++t) {
            const auto tok =
                nt == 1 ? 0
                        : static_cast<size_t>(static_cast<long double>(tokens - 1) * t / (nt - 1));
            for (size_t r = 0; r < nr; ++r) {
                const auto row =
                    nr == 1
                        ? 0
                        : static_cast<size_t>(static_cast<long double>(width - 1) * r / (nr - 1));
                ix.push_back(tok * static_cast<size_t>(width) + row);
            }
        }
    }
    return ix;
}
static std::map<size_t, double> ffn_reference(const Options &o, const Data &d,
                                              const std::vector<float> &h,
                                              const std::vector<size_t> &indices) {
    auto n = normalized(h, o.hidden, d.weights.at("post_attention_norm"), o.norm_eps);
    auto g = matmul(d.weights.at("gate"), n, o.q), u = matmul(d.weights.at("up"), n, o.q);
    std::vector<float> active(g.size());
    for (size_t i = 0; i < g.size(); ++i)
        active[i] = static_cast<float>(static_cast<double>(g[i]) /
                                       (1.0 + std::exp(-static_cast<double>(g[i]))) * u[i]);
    std::map<size_t, double> result;
    std::vector<float> row;
    for (const size_t ix : indices) {
        d.weights.at("down").row(static_cast<int64_t>(ix % static_cast<size_t>(o.hidden)), row);
        result[ix] = dot(row, active.data() + (ix / static_cast<size_t>(o.hidden)) *
                                                  static_cast<size_t>(o.intermediate)) +
                     h[ix];
    }
    return result;
}
struct Check {
    size_t samples = 0, elements = 0;
    double squared = 0, ref_squared = 0, max_abs = 0, max_relative = 0;
    void add(double actual, double ref) {
        const double e = std::abs(actual - ref);
        squared += e * e;
        ref_squared += ref * ref;
        max_abs = std::max(max_abs, e);
        max_relative = std::max(max_relative, e / std::max(std::abs(ref), 1e-12));
        ++samples;
    }
    double rmse() const {
        return std::sqrt(squared / static_cast<double>(samples));
    }
    double rms() const {
        return std::sqrt(ref_squared / static_cast<double>(samples));
    }
    double nrmse() const {
        return rmse() / std::max(rms(), 1e-12);
    }
};
static std::string digest(const std::vector<float> &f) {
    uint64_t hash = 14695981039346656037ULL;
    const auto *bytes = reinterpret_cast<const uint8_t *>(f.data());
    for (size_t i = 0; i < f.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    std::ostringstream s;
    s << std::hex << std::setfill('0') << std::setw(16) << hash;
    return s.str();
}
static void emit_dtypes(const Options &o) {
    std::cout << "{";
    bool first = true;
    for (const auto &d : o.dtypes) {
        std::cout << (first ? "" : ",") << "\"" << d.first << "\":\"" << d.second << "\"";
        first = false;
    }
    std::cout << "}";
}
static void emit_config(const Options &o) {
    std::cout << "{\"hidden\":" << o.hidden << ",\"intermediate\":" << o.intermediate
              << ",\"heads\":" << o.heads << ",\"kv_heads\":" << o.kv_heads
              << ",\"head_dim\":" << o.head_dim << ",\"attention_gate\":\"" << o.attention_gate
              << "\",\"qk_norm\":\"" << o.qk_norm << "\",\"rope_mode\":\"" << o.rope_mode
              << "\",\"rope_dims\":" << o.rope_dims << ",\"rope_base\":" << o.rope_base
              << ",\"norm_epsilon\":" << o.norm_eps << ",\"weight_dtypes\":";
    emit_dtypes(o);
    std::cout << "}";
}

static void emit_string_list(const std::vector<std::string> &items) {
    std::cout << "[";
    for (size_t i = 0; i < items.size(); ++i)
        std::cout << (i ? "," : "") << "\"" << items[i] << "\"";
    std::cout << "]";
}
static std::vector<std::string> operations(const Options &o, const std::string &kind) {
    if (kind == "projection") {
        std::vector<std::string> v = {"input_RMSNorm", "input_norm_scale", "Q_projection",
                                      "K_projection", "V_projection"};
        if (o.attention_gate == "sigmoid")
            v.push_back("gate_projection");
        if (o.qk_norm == "rms")
            for (const auto &s :
                 {"Q_per_head_RMSNorm", "Q_norm_scale", "K_per_head_RMSNorm", "K_norm_scale"})
                v.push_back(s);
        for (const auto &s :
             {"RoPE_Q", "RoPE_K", "materialize_Q", "materialize_K", "materialize_V"})
            v.push_back(s);
        if (o.attention_gate == "sigmoid")
            v.push_back("materialize_gate");
        return v;
    }
    if (kind == "attention") {
        std::vector<std::string> v = {"append_new_KV", "QK_transpose", "scale",
                                      "causal_mask",   "softmax",      "PV"};
        if (o.attention_gate == "sigmoid") {
            v.push_back("sigmoid_gate");
            v.push_back("attention_times_gate");
        }
        for (const auto &s :
             {"output_projection", "attention_residual_add", "materialize_attention_hidden"})
            v.push_back(s);
        return v;
    }
    if (kind == "dense_ffn")
        return {"post_attention_RMSNorm",
                "post_attention_norm_scale",
                "gate_projection",
                "up_projection",
                "SiLU_gate",
                "gate_times_up",
                "down_projection",
                "ffn_residual_add",
                "materialize_hidden"};
    std::vector<std::string> v;
    for (const auto &s : {"projection", "attention", "dense_ffn"}) {
        auto part = operations(o, s);
        v.insert(v.end(), part.begin(), part.end());
    }
    return v;
}
static std::string port_role(const std::string &name) {
    const std::map<std::string, std::string> names = {{"hidden_input", "hidden"},
                                                      {"query", "q"},
                                                      {"new_key", "k"},
                                                      {"new_value", "v"},
                                                      {"attention_gate", "gate"},
                                                      {"attention_hidden", "attention_hidden"},
                                                      {"hidden_output", "hidden_out"},
                                                      {"positions", "positions"},
                                                      {"key_cache", "key_cache"},
                                                      {"value_cache", "value_cache"}};
    return names.at(name);
}
static std::string tensor_digest(ggml_tensor *t) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    uint64_t hash = 14695981039346656037ULL;
    for (uint8_t v : bytes) {
        hash ^= v;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}
static void emit_ns(const std::vector<int64_t> &values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i)
        std::cout << (i ? "," : "") << values[i];
    std::cout << "]";
}
static void cuda_ok(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status) +
                                 " (status " + std::to_string(static_cast<int>(status)) + ")");
}
static void disable_cuda_graphs() {
#if defined(_WIN32)
    if (_putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1") != 0)
        throw std::runtime_error("cannot set GGML_CUDA_DISABLE_GRAPHS");
#else
    if (setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) != 0)
        throw std::runtime_error("cannot set GGML_CUDA_DISABLE_GRAPHS");
#endif
#if defined(_WIN32)
    char value[8]{};
    size_t required = 0;
    if (getenv_s(&required, value, sizeof(value), "GGML_CUDA_DISABLE_GRAPHS") != 0 ||
        required != 2 || std::string(value) != "1")
        throw std::runtime_error("CUDA graph disable environment was not applied");
#else
    const char *value = std::getenv("GGML_CUDA_DISABLE_GRAPHS");
    if (!value || std::string(value) != "1")
        throw std::runtime_error("CUDA graph disable environment was not applied");
#endif
}
static std::string json_quote(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        else
            out << c;
    }
    out << '"';
    return out.str();
}
static void emit_device(int index) {
    cudaDeviceProp properties{};
    cuda_ok(cudaGetDeviceProperties(&properties, index), "cudaGetDeviceProperties");
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            uuid << '-';
        uuid << std::setw(2)
             << static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    std::cout << "{\"index\":" << index << ",\"name\":" << json_quote(properties.name)
              << ",\"uuid\":" << json_quote(uuid.str())
              << ",\"total_memory_bytes\":" << properties.totalGlobalMem
              << ",\"compute_capability_major\":" << properties.major
              << ",\"compute_capability_minor\":" << properties.minor
              << ",\"verification_source\":\"cudaGetDeviceProperties\"}";
}
static int list_devices() {
    int count = 0;
    cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    // Query all properties before opening JSON to avoid partial success output.
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp properties{};
        cuda_ok(cudaGetDeviceProperties(&properties, i), "cudaGetDeviceProperties");
    }
    std::cout << "{\"schema\":\"ggml_dense_layer_cuda_devices/1\",\"device_count\":" << count
              << ",\"devices\":[";
    for (int i = 0; i < count; ++i) {
        if (i)
            std::cout << ",";
        emit_device(i);
    }
    std::cout << "]}\n";
    return 0;
}
static int run(const Options &o) {
    disable_cuda_graphs();
    int device_count = 0;
    cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (o.cuda_device >= device_count)
        throw std::runtime_error("requested CUDA device unavailable");
    cuda_ok(cudaSetDevice(o.cuda_device), "cudaSetDevice");
    DENSE_CUDA_MEMORY_CHECKPOINT("after_context_initialization");
    Data d = make_data(o);
    const std::string stage_kind = o.stage;
    if (stage_kind != "block") {
        auto prep = build_stage(o, d, "block");
        compute(prep->s, prep->graph);
        capture(*prep, d);
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_upstream_preparation");
    auto stage = build_stage(o, d, stage_kind);
    DENSE_CUDA_MEMORY_CHECKPOINT("after_stage_allocation");
    auto &s = stage->s;
    std::vector<int64_t> wall;
    compute(s, stage->graph); // primes CUDA backend private reusable resources
    DENSE_CUDA_MEMORY_CHECKPOINT("after_setup");
    if (stage_kind == "block")
        capture(*stage, d);
    for (int i = 0; i < o.warmup; ++i)
        compute(s, stage->graph);
    DENSE_CUDA_MEMORY_CHECKPOINT("after_warmup");
    for (int i = 0; i < o.samples; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        compute(s, stage->graph);
        const auto end = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns <= 0)
            throw std::runtime_error("nonpositive CUDA sample");
        wall.push_back(ns);
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_samples");
    std::map<std::string, std::vector<float>> outputs;
    for (const auto &p : stage->ports) {
        outputs[p.first] = read_tensor(p.second);
        for (float v : outputs[p.first])
            if (!std::isfinite(v))
                throw std::runtime_error("nonfinite output: " + p.first);
    }
    const bool p = stage_kind == "projection" || stage_kind == "block",
               a = stage_kind == "attention" || stage_kind == "block",
               f = stage_kind == "dense_ffn" || stage_kind == "block";
    std::vector<float> rq, rk, rv, rg, rh;
    std::map<std::string, Check> checks;
    auto check = [&](const std::string &name, const std::vector<float> &expected, int64_t width,
                     int64_t tokens) {
        auto &c = checks[name];
        const auto &actual = outputs.at(name);
        if (expected.size() != actual.size())
            throw std::runtime_error("reference shape disagreement");
        c.elements = actual.size();
        for (const size_t ix : checked_indices(actual.size(), width, tokens))
            c.add(actual[ix], expected[ix]);
    };
    if (p) {
        auto norm = normalized(d.x, o.hidden, d.weights.at("input_norm"), o.norm_eps);
        rq = projected(o, d, norm, "q", o.heads, true);
        rk = projected(o, d, norm, "k", o.kv_heads, true);
        rv = projected(o, d, norm, "v", o.kv_heads, false);
        check("query", rq, o.head_dim * o.heads, o.q);
        check("new_key", rk, o.head_dim * o.kv_heads, o.q);
        check("new_value", rv, o.head_dim * o.kv_heads, o.q);
        if (o.attention_gate == "sigmoid") {
            rg = matmul(d.weights.at("gate_proj"), norm, o.q);
            check("attention_gate", rg, o.head_dim * o.heads, o.q);
        }
    }
    if (a) {
        rh = attention_reference(
            o, d, stage_kind == "block" ? rq : d.query, stage_kind == "block" ? rk : d.key,
            stage_kind == "block" ? rv : d.value, stage_kind == "block" ? rg : d.agate);
        check("attention_hidden", rh, o.hidden, o.q);
    }
    if (f) {
        const auto &actual = outputs.at("hidden_output");
        auto &c = checks["hidden_output"];
        c.elements = actual.size();
        const auto ix = checked_indices(actual.size(), o.hidden, o.q);
        const auto reference = ffn_reference(o, d, stage_kind == "block" ? rh : d.attn_hidden, ix);
        for (size_t i : ix)
            c.add(actual[i], reference.at(i));
    }
    bool append_exact = true, history_unchanged = true;
    if (a) {
        auto actualk = read_tensor(stage->tensors.at("key_cache")),
             actualv = read_tensor(stage->tensors.at("value_cache"));
        const auto &kn = stage_kind == "block" ? outputs.at("new_key") : d.key;
        const auto &vn = stage_kind == "block" ? outputs.at("new_value") : d.value;
        const int64_t len = o.history + o.q;
        for (int64_t h = 0; h < o.kv_heads; ++h)
            for (int64_t t = 0; t < len; ++t)
                for (int64_t j = 0; j < o.head_dim; ++j) {
                    const size_t dst = static_cast<size_t>(j + o.head_dim * (t + len * h));
                    if (t < o.history)
                        history_unchanged = history_unchanged && actualk[dst] == d.kcache[dst] &&
                                            actualv[dst] == d.vcache[dst];
                    else {
                        const size_t src =
                            static_cast<size_t>(j + o.head_dim * (t - o.history + o.q * h));
                        append_exact =
                            append_exact && actualk[dst] == kn[src] && actualv[dst] == vn[src];
                    }
                }
    }
    const bool quant = std::any_of(o.dtypes.begin(), o.dtypes.end(), [](const auto &item) {
        return item.second == "q4_k" || item.second == "q6_k";
    });
    const double atol = quant ? 0.003 : 0.0002, rtol = quant ? 0.035 : 0.002;
    Check total;
    bool passed = append_exact && history_unchanged;
    for (const auto &entry : checks) {
        const auto &c = entry.second;
        passed = passed && (c.nrmse() <= rtol || c.rmse() <= atol);
        total.samples += c.samples;
        total.elements += c.elements;
        total.squared += c.squared;
        total.ref_squared += c.ref_squared;
        total.max_abs = std::max(total.max_abs, c.max_abs);
        total.max_relative = std::max(total.max_relative, c.max_relative);
    }
    const size_t resident = add_bytes(add_bytes(stage->input_bytes, stage->output_bytes),
                                      add_bytes(stage->weight_bytes, stage->kv_bytes));
    std::cout << std::setprecision(17)
              << "{\"schema\":\"ggml_dense_layer_cuda_probe/1\","
                 "\"contract\":\"ggml_cuda_dense_decoder_block_v1\",\"kind\":\"dense_layer\","
                 "\"stage\":\""
              << o.stage << "\",\"backend\":\"CUDA" << o.cuda_device
              << "\",\"cuda_device\":" << o.cuda_device
              << ",\"cuda_graphs\":\"disabled\",\"cuda_graphs_requested\":\"disabled\","
                 "\"cuda_graphs_applied\":\"disabled\",\"cuda_graph_disable_environment\":\"1\","
                 "\"batch\":1,\"config\":";
    emit_config(o);
    std::cout << ",\"q\":" << o.q << ",\"history\":" << o.history << ",\"warmup\":" << o.warmup
              << ",\"setup_compute_repetitions\":1,"
                 "\"upstream_setup_compute_repetitions\":"
              << (stage_kind == "block" ? 0 : 1) << ",\"samples\":" << o.samples
              << ",\"seed\":" << o.seed
              << ",\"input_generator\":\"mt19937_uniform24_scaled_weights_gamma_near_one_v1\","
                 "\"input_lineage\":\"same_seed_complete_upstream_preparation_outside_timing\","
                 "\"matmul_precision\":\"ggml_cuda_backend_default\",\"activation_dtype\":\"f32\","
                 "\"kv_dtype\":\"f32\",\"position_dtype\":\"i32\","
                 "\"implementation_variant\":\"ggml_cuda_dense_materialized_"
              << o.stage
              << "_v1\","
                 "\"timing\":\"steady_clock_cuda_graph_compute_including_sync\","
                 "\"scope\":\"resident_repeated_complete_declared_decoder_block\","
                 "\"included\":";
    emit_string_list(operations(o, stage_kind));
    std::cout << ",\"excluded\":";
    emit_string_list({"allocation", "quantization", "io", "external_concurrency"});
    std::cout << ",\"device_info\":";
    emit_device(o.cuda_device);
    std::cout << ",\"backend_execution_check\":{\"all_nodes_supported\":true,"
                 "\"all_tensors_on_requested_cuda_buffer\":true,\"cpu_fallback_used\":false,"
                 "\"node_count\":"
              << ggml_graph_n_nodes(stage->graph)
              << ",\"node_count_semantics\":\"all_ggml_graph_nodes_including_views\","
                 "\"verification_source\":\"supports_op_before_and_after_allocation_and_exact_cuda_"
                 "buft_identity_for_resident_nodes_and_sources\"}";
    std::cout << ",\"nodes\":[";
    for (int i = 0; i < ggml_graph_n_nodes(stage->graph); ++i) {
        const auto *n = ggml_graph_node(stage->graph, i);
        std::cout << (i ? "," : "") << "{\"op\":\"" << ggml_op_name(n->op)
                  << "\",\"bytes\":" << ggml_nbytes(n)
                  << ",\"view\":" << (n->view_src ? "true" : "false")
                  << ",\"supported_by_requested_backend\":true,\"buffer_type\":\"CUDA"
                  << o.cuda_device << "\"}";
    }
    std::cout << "]";
    std::cout << ",\"port_accounting\":{";
    bool first = true;
    for (const auto &t : stage->tensors) {
        const auto *v = t.second;
        const bool cache = t.first == "key_cache" || t.first == "value_cache",
                   out = stage->ports.count(t.first) > 0;
        std::cout << (first ? "" : ",") << "\"" << port_role(t.first) << "\":{\"dtype\":\""
                  << (v->type == GGML_TYPE_I32 ? "i32" : "f32") << "\",\"shape\":[" << v->ne[0]
                  << "," << v->ne[1] << "," << v->ne[2] << "],\"ne\":[" << v->ne[0] << ","
                  << v->ne[1] << "," << v->ne[2] << "," << v->ne[3]
                  << "],\"bytes\":" << ggml_nbytes(v) << ",\"category\":\""
                  << (cache ? "kv_cache"
                      : out ? "output"
                            : "input")
                  << "\"}";
        first = false;
    }
    std::cout << "}";
    std::cout << ",\"weight_roles_bytes\":{";
    first = true;
    for (const auto &w : stage->weights) {
        std::cout << (first ? "" : ",") << "\"" << w.first << "\":" << ggml_nbytes(w.second);
        first = false;
    }
    std::cout << "},\"input_ports_bytes\":{";
    first = true;
    for (const auto &t : stage->tensors)
        if (!stage->ports.count(t.first) && t.first != "key_cache" && t.first != "value_cache") {
            std::cout << (first ? "" : ",") << "\"" << port_role(t.first)
                      << "\":" << ggml_nbytes(t.second);
            first = false;
        }
    std::cout << "},\"output_ports_bytes\":{";
    first = true;
    for (const auto &p2 : stage->ports) {
        std::cout << (first ? "" : ",") << "\"" << port_role(p2.first)
                  << "\":" << ggml_nbytes(p2.second);
        first = false;
    }
    std::cout << "}";
    const size_t kv_per_token = product({o.head_dim, o.kv_heads}) * sizeof(float) * 2,
                 history_bytes = a ? static_cast<size_t>(o.history) * kv_per_token : 0,
                 new_bytes = a ? static_cast<size_t>(o.q) * kv_per_token : 0;
    std::cout << ",\"weight_bytes\":" << stage->weight_bytes
              << ",\"input_bytes\":" << stage->input_bytes
              << ",\"output_bytes\":" << stage->output_bytes
              << ",\"kv_history_bytes\":" << history_bytes << ",\"kv_new_bytes\":" << new_bytes
              << ",\"kv_cache_bytes\":" << stage->kv_bytes
              << ",\"resident_logical_bytes\":" << resident
              << ",\"workspace_bytes\":null,"
                 "\"workspace_source\":\"unavailable_cuda_backend_private_scratch_and_runtime_"
                 "overheads\","
                 "\"internal_tensor_bytes_upper_bound\":"
              << std::max(stage->internal_bytes, stage->logical_internal)
              << ",\"memory_accounting\":{\"resident_logical_bytes\":" << resident
              << ",\"resident_allocated_bytes\":" << ggml_backend_buffer_get_size(s.buffer)
              << ",\"internal_allocated_buffer_bytes\":" << stage->internal_bytes
              << ",\"internal_logical_tensor_bytes_sum\":" << stage->logical_internal
              << ",\"known_tensor_allocation_bytes\":"
              << add_bytes(ggml_backend_buffer_get_size(s.buffer), stage->internal_bytes)
              << ",\"backend_scratch_bytes\":null,\"runtime_overhead_bytes\":null,"
                 "\"workspace_bytes\":null,\"workspace_bound\":\"unknown\","
                 "\"workspace_scope\":\"known_cuda_gallocr_capacity_excludes_private_backend_"
                 "scratch_and_runtime_overheads\","
                 "\"kv_new_bytes_semantics\":\"appended_slots_in_cache_distinct_from_new_kv_input_"
                 "tensors\"}";
    std::cout
        << ",\"port_digest_algorithm\":\"fnv1a64_native_tensor_bytes\",\"stage_port_digests\":{";
    first = true;
    for (const auto &item : outputs) {
        std::cout << (first ? "" : ",") << "\"" << port_role(item.first) << "\":\""
                  << digest(item.second) << "\"";
        first = false;
    }
    std::cout << "},\"input_port_digests\":{";
    first = true;
    for (const auto &item : stage->tensors)
        if (!stage->ports.count(item.first) && item.first != "key_cache" &&
            item.first != "value_cache") {
            std::cout << (first ? "" : ",") << "\"" << port_role(item.first) << "\":\""
                      << tensor_digest(item.second) << "\"";
            first = false;
        }
    std::cout << "},\"stage_timed_output_ports\":";
    std::vector<std::string> timedports;
    for (const auto &out : outputs)
        timedports.push_back(port_role(out.first));
    emit_string_list(timedports);
    std::cout << ",\"all_outputs_finite\":true,\"kv_append_check\":";
    if (a)
        std::cout << "{\"passed\":" << ((append_exact && history_unchanged) ? "true" : "false")
                  << ",\"append_exact\":" << (append_exact ? "true" : "false")
                  << ",\"history_unchanged\":" << (history_unchanged ? "true" : "false") << "}";
    else
        std::cout << "null";
    auto emitcheck = [&](const Check &c, bool pass) {
        std::cout << "{\"passed\":" << (pass ? "true" : "false") << ",\"samples\":" << c.samples
                  << ",\"output_elements\":" << c.elements << ",\"coverage\":\""
                  << (c.samples == c.elements ? "all_output_elements"
                                              : "deterministic_output_element_sample")
                  << "\",\"reference\":\"independent_double_declared_block_with_dequantized_stored_"
                     "weights_and_f32_operator_boundaries\",\"max_abs_error\":"
                  << c.max_abs << ",\"max_relative_error\":" << c.max_relative
                  << ",\"rmse\":" << c.rmse() << ",\"reference_rms\":" << c.rms()
                  << ",\"normalized_rmse\":" << c.nrmse() << ",\"atol\":" << atol
                  << ",\"rtol\":" << rtol
                  << ",\"criterion\":\"each_port_normalized_rmse_le_rtol_or_rmse_le_atol_and_exact_"
                     "kv_validation\"}";
    };
    std::cout << ",\"reference_check\":";
    emitcheck(total, passed);
    std::cout << ",\"port_reference_checks\":{";
    first = true;
    for (const auto &entry : checks) {
        std::cout << (first ? "" : ",") << "\"" << entry.first << "\":";
        emitcheck(entry.second, entry.second.nrmse() <= rtol || entry.second.rmse() <= atol);
        first = false;
    }
    std::cout << "}";
    std::cout << ",\"wall_ns\":";
    emit_ns(wall);
    std::cout << "}\n";
    if (!passed) {
        std::cerr << "CUDA dense layer numerical or KV validation failed\n";
        return 2;
    }
    return 0;
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "ggml-dense-layer-cuda-probe --stage projection|attention|dense_ffn|block "
                         "--cuda-device index --cuda-graphs disabled; --list-devices; "
                         "explicit CUDA B1 decoder contract; no CPU fallback\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--list-devices")
            return list_devices();
        return run(parse(argc, argv));
    } catch (const std::exception &e) {
        std::cerr << "ggml-dense-layer-cuda-probe: " << e.what() << '\n';
        return 1;
    }
}
