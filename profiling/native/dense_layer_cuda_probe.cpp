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
    // The cache element type, not a weight dtype.  Defaults to f32 so the strict-F32
    // contract this probe was written for is unchanged when the flag is absent; the
    // pricing contract the CPU table declares is f16 (`dense_contract.KV_DTYPE`), and a
    // GPU price that has to sit beside a CPU one must store the cache the same way.
    std::string kv_dtype = "f32";
    std::map<std::string, std::string> dtypes;
    int64_t hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0, q = 0,
            history = -1, rope_dims = 0;
    float rope_base = 0, norm_eps = 0;
    int cuda_device = -1, warmup = -1, samples = 0, repetitions = 1, sensor_q = 0;
    int first_runs = 0, fresh_graphs = 0;
    // DIAGNOSTIC (2026-11-09): per-node cost by prefix truncation.  The graph's node list is
    // already a linear order, so running the first k nodes and differencing successive k gives
    // node k's own launch + execution without a profiler.  The device level moves inside one
    // process -- the same graph reads 294 us and 766 us minutes apart -- so each prefix sample
    // is paired with a full-graph execution taken at the same instant and only the ratio is
    // kept.  0 disables the sweep, which is what the published prices were taken with;
    // `node_profile_samples` is the number of (prefix, reference) pairs per prefix.
    int node_profile = 0, node_profile_samples = 31;
    // The sparse feed-forward.  With experts == 0 the feed-forward is the dense gate/up/down
    // block and nothing changes; with experts > 0 it is the routed expert block of
    // `moe_ffn`, whose weights are [cols, rows, experts] planes.
    int64_t experts = 0, top_k = 0, shared_intermediate = 0;
    bool normalize_topk = false;
    // How many expert planes are resident in the priced graph.  A block that streams the
    // experts it routes to holds `k` of them (one token uses exactly `k`); a block that keeps
    // the whole stack holds `E`.  The router chain is unchanged either way -- logits, softmax
    // and top-k are still over all `E` experts -- so only the planes the products address
    // change.  0 means `E`, which is the published geometry.
    int64_t stack = 0;
    // DIAGNOSTIC (2026-11-09): the block's cost depends on *which* experts the tokens land
    // on, and by default that is whatever the synthetic router logits happen to select --
    // an undeclared variable that differs from model to model.  With `--route 1` the three
    // indirect products are given uniformly scattered ids instead, so the routing is
    // declared and the same across models and across the executor.  Absent, the price is
    // the one already published, bit for bit.
    bool route = false;
    // A model may add a bias to its query/key/value projections (Qwen1.5-MoE does).  The
    // bias is part of the declared work, so a price taken without it is a different graph.
    bool attention_bias = false;
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
        {"--q-dtype", "q"},              {"--k-dtype", "k"},
        {"--v-dtype", "v"},              {"--gate-proj-dtype", "gate_proj"},
        {"--o-dtype", "output"},         {"--ffn-gate-dtype", "gate"},
        {"--up-dtype", "up"},            {"--down-dtype", "down"},
        {"--gate-exps-dtype", "gate_exps"}, {"--up-exps-dtype", "up_exps"},
        {"--down-exps-dtype", "down_exps"}, {"--shexp-gate-dtype", "gate_shexp"},
        {"--shexp-up-dtype", "up_shexp"},   {"--shexp-down-dtype", "down_shexp"},
        {"--q-bias-dtype", "q_bias"},       {"--k-bias-dtype", "k_bias"},
        {"--v-bias-dtype", "v_bias"}};
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
        else if (key == "--kv-dtype")
            o.kv_dtype = v;
        else if (key == "--attention-gate")
            o.attention_gate = v;
        else if (key == "--attention-bias")
            o.attention_bias = integer(v) != 0;
        else if (key == "--qk-norm")
            o.qk_norm = v;
        else if (key == "--rope-mode")
            o.rope_mode = v;
        else if (key == "--hidden")
            o.hidden = integer(v);
        else if (key == "--intermediate")
            o.intermediate = integer(v);
        else if (key == "--experts")
            o.experts = integer(v);
        else if (key == "--top-k")
            o.top_k = integer(v);
        else if (key == "--shared-intermediate")
            o.shared_intermediate = integer(v);
        else if (key == "--normalize-topk")
            o.normalize_topk = integer(v) != 0;
        else if (key == "--route")
            o.route = integer(v) != 0;
        else if (key == "--stack")
            o.stack = integer(v);
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
        } else if (key == "--cuda-device" || key == "--warmup" || key == "--samples" ||
                   key == "--repetitions" || key == "--sensor-q" || key == "--first-runs" ||
                   key == "--fresh-graphs" || key == "--node-profile" ||
                   key == "--node-profile-samples") {
            const auto t = integer(v);
            if (t < 0 || t > std::numeric_limits<int>::max())
                throw std::runtime_error("option out of range: " + key);
            if (key == "--cuda-device")
                o.cuda_device = static_cast<int>(t);
            if (key == "--warmup")
                o.warmup = static_cast<int>(t);
            if (key == "--samples")
                o.samples = static_cast<int>(t);
            if (key == "--repetitions")
                o.repetitions = static_cast<int>(t);
            if (key == "--sensor-q")
                o.sensor_q = static_cast<int>(t);
            if (key == "--first-runs")
                o.first_runs = static_cast<int>(t);
            if (key == "--fresh-graphs")
                o.fresh_graphs = static_cast<int>(t);
            if (key == "--node-profile")
                o.node_profile = static_cast<int>(t);
            if (key == "--node-profile-samples")
                o.node_profile_samples = static_cast<int>(t);
        } else
            throw std::runtime_error("unknown option: " + key);
    }
    const std::set<std::string> stages = {"projection", "attention", "dense_ffn", "moe_ffn",
                                          "block"};
    if (!stages.count(o.stage))
        throw std::runtime_error("unsupported stage");
    // The sparse feed-forward is its own stage: it replaces the dense one, so measuring it
    // inside `block` would declare a graph with both.  A block-level sparse measurement is
    // not implemented, and saying so is better than returning a dense number under its name.
    // Declaring the router and expert geometry is harmless on the stages that never read
    // those weights: a sparse model's projection and attention are the same graph either
    // way, and refusing them would leave those cells unpriceable for no reason.
    if (o.stage == "moe_ffn" && o.experts <= 0)
        throw std::runtime_error("moe_ffn requires experts > 0");
    if (o.stage == "dense_ffn" && o.experts > 0)
        throw std::runtime_error("a dense feed-forward cannot also declare routed experts");
    if (o.experts > 0 &&
        (o.top_k <= 0 || o.top_k > o.experts || o.shared_intermediate < 0))
        throw std::runtime_error("experts require top_k in 1..experts and "
                                 "shared_intermediate >= 0");
    if (o.stack < 0 || (o.stack > 0 && (o.stack < o.top_k || o.stack > o.experts)))
        throw std::runtime_error("--stack must be 0 (all experts) or in top_k..experts");
    if (o.experts > 0 && o.stack == 0)
        o.stack = o.experts;
    if (!seen.count("--cuda-device") || o.cuda_device < 0 || o.cuda_device >= GGML_CUDA_MAX_DEVICES)
        throw std::runtime_error("requires explicit --cuda-device in CUDA device range");
    if (!seen.count("--cuda-graphs") || o.cuda_graphs != "disabled")
        throw std::runtime_error(
            "this variant requires explicit --cuda-graphs disabled; no fallback");
    if (o.kv_dtype != "f32" && o.kv_dtype != "f16")
        throw std::runtime_error("--kv-dtype must be f32 or f16 (cache element type, not weights)");
    if (o.hidden <= 0 || o.intermediate <= 0 || o.heads <= 0 || o.kv_heads <= 0 ||
        o.head_dim <= 0 || o.heads % o.kv_heads || o.q <= 0 || o.history < 0)
        throw std::runtime_error("invalid block geometry");
    if (o.history > std::numeric_limits<int>::max() ||
        o.q > std::numeric_limits<int>::max() - o.history)
        throw std::runtime_error("sequence indexing overflow");
    if (o.attention_gate != "none" && o.attention_gate != "sigmoid")
        throw std::runtime_error("attention_gate must be none or sigmoid");
    if (o.qk_norm != "none" && o.qk_norm != "rms" && o.qk_norm != "rms_hidden")
        throw std::runtime_error("qk_norm must be none, rms, or rms_hidden (one gain per hidden "
                                 "unit, applied before the head reshape)");
    if (o.rope_mode != "normal" || o.rope_dims <= 0 || o.rope_dims % 2 ||
        o.rope_dims > o.head_dim || o.rope_base <= 1 || o.norm_eps <= 0)
        throw std::runtime_error("requires normal RoPE, positive even dimensions <= head_dim, "
                                 "explicit base and epsilon");
    if (o.warmup < 0 || o.samples < 3 || o.samples > 10000 || !seen.count("--seed"))
        throw std::runtime_error("requires explicit warmup/samples/seed, samples in 3..10000");
    if (o.repetitions < 1 || o.repetitions > 10000)
        throw std::runtime_error("repetitions must be in 1..10000");
    if (o.first_runs < 0 || o.first_runs > 1000 || o.fresh_graphs < 0 || o.fresh_graphs > 64)
        throw std::runtime_error("first-runs must be in 0..1000, fresh-graphs in 0..64");
    const std::set<std::string> types = {"f32", "f16", "q4_k", "q6_k", "q8_0"};
    // Only the roles the selected stage reads are required.  A model whose feed-forward is
    // sparse has no dense gate/up/down weights at all, and its projection and attention are
    // still measurable; demanding the full block would refuse them for lacking a stage they
    // do not run.  Everything the caller declares is emitted, because the receipt has to
    // describe the same config the case declared.
    std::map<std::string, std::set<std::string>> stage_roles = {
        {"projection", {"q", "k", "v"}},
        {"attention", {"output"}},
        {"dense_ffn", {"gate", "up", "down"}},
        {"moe_ffn", {"gate_exps", "up_exps", "down_exps"}},
        {"block", {"q", "k", "v", "output", "gate", "up", "down"}}};
    // A sparse model's block ends in the routed experts, not in a dense feed-forward, so it
    // requires those roles instead.
    if (o.experts > 0)
        stage_roles["block"] = {"q", "k", "v", "output", "gate_exps", "up_exps", "down_exps"};
    if (o.shared_intermediate > 0)
        for (const auto &role : {"gate_shexp", "up_shexp", "down_shexp"}) {
            stage_roles.at("moe_ffn").insert(role);
            stage_roles.at("block").insert(role);
        }
    for (const auto &role : stage_roles.at(o.stage))
        if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
            throw std::runtime_error(std::string("missing/unsupported matrix dtype: ") + role);
    // The router and the shared-expert gate are F32 parameters of the sparse block, like the
    // norms: the caller declares the expert geometry, never the router's type.  They are
    // recorded for every stage of such a case, because the receipt's config is the block's
    // config and the block is sparse whichever stage was timed.
    const bool sparse = o.experts > 0;
    if (sparse)
        o.dtypes["router"] = "f32";
    if (sparse && o.shared_intermediate > 0)
        o.dtypes["gate_inp_shexp"] = "f32";
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
    if (o.qk_norm == "rms" || o.qk_norm == "rms_hidden") {
        o.dtypes["q_norm"] = "f32";
        o.dtypes["k_norm"] = "f32";
    }
    // A stored bias belongs to the projection graph, so a biased case declares the three
    // rows per output channel for every stage it measures: the preparation that seeds an
    // attention stage runs that same projection.  The type is F32, like the norms.
    if (o.attention_bias)
        for (const auto &role : {"q_bias", "k_bias", "v_bias"}) {
            if (!o.dtypes.count(role))
                throw std::runtime_error(std::string("a biased projection requires a declared "
                                                     "bias weight: ") + role);
            if (o.dtypes.at(role) != "f32")
                throw std::runtime_error(std::string("bias weights are f32 parameters: ") + role);
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
    // One plane per expert.  A routed projection is the same matrix repeated `experts`
    // times along the third axis, and `ggml_mul_mat_id` addresses the plane; `row()` takes
    // a global row index, so plane e row r is `e * rows + r`.
    int64_t experts = 1;
    ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> packed;

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

// The cache element type.  One helper, so the tensor allocation, the host copy and the
// receipt cannot disagree about what the cache stores.
static ggml_type kv_cache_type(const Options &o) {
    return o.kv_dtype == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
}
// The value a half cache actually stores for an f32 input.
static float half_round(float value) {
    const ggml_fp16_t half = ggml_fp32_to_fp16(value);
    return ggml_fp16_to_fp32(half);
}
static ggml_tensor *input_typed(ggml_context *ctx, const char *name, ggml_type type,
                                int64_t a, int64_t b, int64_t c = 1) {
    product({a, b, c});
    auto *t = ggml_new_tensor_3d(ctx, type, a, b, c);
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

// Back-to-back launches followed by one synchronize.  A single graph compute at the small
// shapes costs less than the host-side dispatch, so timing one launch per sample measures
// the dispatch, not the work.  Repeating inside the timed region amortises it; dividing the
// loop time by R is the per-execution cost the price is meant to express.
static void compute_batch(Session &s, ggml_cgraph *graph, int repetitions) {
    for (int i = 0; i < repetitions; ++i) {
        const ggml_status status = ggml_backend_graph_compute_async(s.backend, graph);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA graph compute failed: " + std::to_string(status));
    }
    ggml_backend_synchronize(s.backend);
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
    if (t == "q8_0")
        return GGML_TYPE_Q8_0;
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
    // A stage builds and allocates only its own roles, so a weight the stage does not read
    // is absent from `dtypes` and must not be created here either.
    auto add_if = [&](const std::string &role, int64_t cols, int64_t rows, bool gamma = false) {
        if (o.dtypes.count(role))
            add(role, cols, rows, gamma);
    };
    add_if("q", o.hidden, e);
    add_if("k", o.hidden, kv);
    add_if("v", o.hidden, kv);
    if (o.attention_bias) {
        add_if("q_bias", e, 1);
        add_if("k_bias", kv, 1);
        add_if("v_bias", kv, 1);
    }
    add_if("gate_proj", o.hidden, e);
    add_if("output", e, o.hidden);
    add_if("gate", o.hidden, o.intermediate);
    add_if("up", o.hidden, o.intermediate);
    add_if("down", o.intermediate, o.hidden);
    if (o.experts > 0) {
        // One expert plane is one dense matrix, so the planes are generated and quantized in
        // blocks: the float staging copy stays small beside the packed bytes it is only an
        // intermediate for, and an eight-gigabyte device is not asked to hold both.
        auto add_exps = [&](const std::string &role, int64_t cols, int64_t rows) {
            if (!o.dtypes.count(role))
                return;
            Weight w;
            w.cols = cols;
            w.rows = rows;
            w.experts = o.stack;   // resident planes, not the model's expert count
            w.type = type_of(o.dtypes.at(role));
            if ((w.type == GGML_TYPE_Q4_K || w.type == GGML_TYPE_Q6_K) && cols % 256)
                throw std::runtime_error(
                    "quantized reduction dimension must be divisible by 256: " + role);
            const size_t row_bytes = ggml_row_size(w.type, cols);
            w.packed.resize(row_bytes * static_cast<size_t>(rows) *
                            static_cast<size_t>(o.stack));
            const int64_t plane = product({cols, rows});
            const int64_t block = std::max<int64_t>(1, (int64_t{8} << 20) / plane);
            std::vector<float> f(static_cast<size_t>(plane) * static_cast<size_t>(block));
            for (int64_t start = 0; start < o.stack; start += block) {
                const int64_t n = std::min(block, o.stack - start);
                f.resize(static_cast<size_t>(plane) * static_cast<size_t>(n));
                fill(f, rng);
                for (float &v : f)
                    v /= std::sqrt(static_cast<float>(cols));
                uint8_t *dst = w.packed.data() + static_cast<size_t>(start) * row_bytes *
                                                     static_cast<size_t>(rows);
                if (w.type == GGML_TYPE_F32)
                    std::memcpy(dst, f.data(), f.size() * sizeof(float));
                else if (w.type == GGML_TYPE_F16)
                    ggml_fp32_to_fp16_row(f.data(), reinterpret_cast<ggml_fp16_t *>(dst),
                                          static_cast<int64_t>(f.size()));
                else if (ggml_quantize_chunk(w.type, f.data(), dst, 0, rows * n, cols,
                                             nullptr) !=
                         row_bytes * static_cast<size_t>(rows) * static_cast<size_t>(n))
                    throw std::runtime_error("expert tensor quantization size mismatch: " + role);
            }
            d.weights.emplace(role, std::move(w));
        };
        add_exps("gate_exps", o.hidden, o.intermediate);
        add_exps("up_exps", o.hidden, o.intermediate);
        // The down projection reduces over the expert width, so its rows are the hidden
        // dimension -- the stored layout, not a transposed copy.
        add_exps("down_exps", o.intermediate, o.hidden);
        add_if("router", o.hidden, o.experts);
        if (o.shared_intermediate > 0) {
            add_if("gate_shexp", o.hidden, o.shared_intermediate);
            add_if("up_shexp", o.hidden, o.shared_intermediate);
            add_if("down_shexp", o.shared_intermediate, o.hidden);
            add_if("gate_inp_shexp", o.hidden, 1);
        }
    }
    add("input_norm", o.hidden, 1, true);
    add("post_attention_norm", o.hidden, 1, true);
    if (o.qk_norm == "rms" || o.qk_norm == "rms_hidden") {
        // One gain per head, or one per hidden unit when the normalization runs over the
        // head-concatenated projection (OLMoE).
        const int64_t width = o.qk_norm == "rms_hidden" ? o.hidden : o.head_dim;
        add("q_norm", width, 1, true);
        add("k_norm", width, 1, true);
    }
    d.x.resize(product({o.hidden, o.q}));
    fill(d.x, rng);
    d.kcache.resize(product({o.head_dim, o.history + o.q, o.kv_heads}));
    fill(d.kcache, rng);
    d.vcache.resize(d.kcache.size());
    fill(d.vcache, rng);
    if (kv_cache_type(o) == GGML_TYPE_F16) {
        // The host vectors are what the independent reference and the exact KV append check
        // compare against.  When the cache tensor stores halves, the host copies are rounded
        // the same way first: then the reference reads the history the graph actually sees,
        // the append can still be compared exactly, and nothing is compared across dtypes.
        for (auto *cache : {&d.kcache, &d.vcache}) {
            std::vector<ggml_fp16_t> halves(cache->size());
            ggml_fp32_to_fp16_row(cache->data(), halves.data(),
                                  static_cast<int64_t>(cache->size()));
            ggml_fp16_to_fp32_row(halves.data(), cache->data(),
                                  static_cast<int64_t>(cache->size()));
        }
    }
    for (int64_t i = 0; i < o.q; ++i)
        d.positions.push_back(static_cast<int32_t>(o.history + i));
    return d;
}
struct Stage {
    Session s;
    ggml_cgraph *graph = nullptr;
    std::map<std::string, ggml_tensor *> tensors, weights, ports;
    // Scatter offsets for an f16 cache.  They are constants of the block, but they live in
    // the resident buffer rather than in the graph: the graph allocator frees a leaf's slot
    // once its last reader has run, so a constant written into the graph is overwritten by
    // the next sample's intermediates.  They are not ports, so they stay out of the port
    // accounting; the resident byte total below includes them.
    ggml_tensor *key_offsets = nullptr, *value_offsets = nullptr;
    // DIAGNOSTIC (2026-11-09): explicit routing for the sparse block.  See `Options::route`.
    ggml_tensor *route_ids = nullptr;
    std::map<std::string, std::pair<int, int>> node_ranges;
    size_t input_bytes = 0, output_bytes = 0, weight_bytes = 0, kv_bytes = 0, internal_bytes = 0,
           logical_internal = 0;
};
static std::vector<std::string> roles_for(const Options &o, const std::string &stage) {
    std::vector<std::string> roles;
    if (stage == "projection" || stage == "block") {
        roles = {"input_norm", "q", "k", "v"};
        if (o.attention_bias)
            for (const auto &r : {"q_bias", "k_bias", "v_bias"})
                roles.push_back(r);
        if (o.attention_gate == "sigmoid")
            roles.push_back("gate_proj");
        if (o.qk_norm == "rms" || o.qk_norm == "rms_hidden") {
            roles.push_back("q_norm");
            roles.push_back("k_norm");
        }
    }
    if (stage == "attention" || stage == "block")
        roles.push_back("output");
    if (stage == "dense_ffn" || (stage == "block" && o.experts == 0))
        for (const auto &r : {"post_attention_norm", "gate", "up", "down"})
            roles.push_back(r);
    if (stage == "moe_ffn" || (stage == "block" && o.experts > 0)) {
        for (const auto &r : {"post_attention_norm", "router", "gate_exps", "up_exps", "down_exps"})
            roles.push_back(r);
        if (o.shared_intermediate > 0)
            for (const auto &r : {"gate_shexp", "up_shexp", "down_shexp", "gate_inp_shexp"})
                roles.push_back(r);
    }
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
                   bool cache = false, ggml_type type = GGML_TYPE_F32) {
        auto *t = input_typed(s.resident, name.c_str(), type, a, b, c);
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
               ffn = stage == "dense_ffn" || stage == "moe_ffn" || stage == "block";
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
        add("key_cache", o.head_dim, length, o.kv_heads, false, true, kv_cache_type(o));
        add("value_cache", o.head_dim, length, o.kv_heads, false, true, kv_cache_type(o));
        if (kv_cache_type(o) == GGML_TYPE_F16) {
            // SET-inplace accepts an f32 destination only, so a half cache cannot be
            // appended that way.  SET_ROWS writes a destination of any element type, takes
            // f32 values (the rule the CUDA backend's set_rows path requires anyway) and
            // returns the complete cache, so readers still depend on the write.
            p->key_offsets = ggml_new_tensor_2d(s.resident, GGML_TYPE_I64, o.q, o.kv_heads);
            ggml_set_name(p->key_offsets, "key_offsets");
            ggml_set_input(p->key_offsets);
            p->value_offsets = ggml_new_tensor_2d(s.resident, GGML_TYPE_I64, o.q, o.kv_heads);
            ggml_set_name(p->value_offsets, "value_offsets");
            ggml_set_input(p->value_offsets);
            // Deliberately not counted as an input port: the contract's input accounting
            // names the tensors the mathematics consumes, and the offsets are addressing.
        }
    }
    if (attention || ffn)
        add("attention_hidden", o.hidden, o.q, 1, attention);
    if (ffn)
        add("hidden_output", o.hidden, o.q, 1, true);
    for (const auto &role : roles_for(o, stage)) {
        const auto &w = d.weights.at(role);
        // An expert plane is the third axis of the stored matrix, which is where
        // `ggml_mul_mat_id` reads it from.
        auto *t = w.experts > 1
                      ? ggml_new_tensor_3d(s.resident, w.type, w.cols, w.rows, w.experts)
                      : ggml_new_tensor_2d(s.resident, w.type, w.cols, w.rows);
        ggml_set_name(t, role.c_str());
        ggml_set_input(t);
        p->weights[role] = t;
        p->weight_bytes = add_bytes(p->weight_bytes, ggml_nbytes(t));
    }
    // An explicit routing arm needs an ids tensor the graph cannot overwrite, so it lives
    // in the resident buffer beside the weights (the same reason as the cache offsets).
    // A stack smaller than the model's expert count needs explicit ids too: the router's
    // own top-k addresses planes 0..E-1, and a graph that holds only `k` planes has to
    // name which of them it holds.
    const bool explicit_ids = o.route || (o.experts > 0 && o.stack < o.experts);
    if (ffn && o.experts > 0 && explicit_ids) {
        p->route_ids = ggml_new_tensor_2d(s.resident, GGML_TYPE_I32, o.top_k, o.q);
        ggml_set_name(p->route_ids, "route_ids");
    }
    s.buffer = ggml_backend_alloc_ctx_tensors(s.resident, s.backend);
    if (!s.buffer)
        throw std::runtime_error("resident allocation failed");
    for (const auto &w : p->weights)
        ggml_backend_tensor_set(w.second, d.weights.at(w.first).packed.data(), 0,
                                ggml_nbytes(w.second));
    if (p->route_ids) {
        // The same formula `ir_runner.cpp` gives its executor, so the two graphs see one
        // routing: every expert receives the same number of tokens, which is what routing
        // over a real corpus averages to and what a batched kernel has to pay for.
        const int64_t stride = std::max<int64_t>(1, o.stack / std::max<int64_t>(1, o.top_k));
        std::vector<int32_t> ids(static_cast<size_t>(o.top_k) * static_cast<size_t>(o.q));
        for (int64_t t = 0; t < o.q; ++t)
            for (int64_t k = 0; k < o.top_k; ++k)
                ids[static_cast<size_t>(k + o.top_k * t)] =
                    static_cast<int32_t>((t + k * stride) % o.stack);
        ggml_backend_tensor_set(p->route_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    auto put = [&](const std::string &name, const std::vector<float> &f) {
        if (p->tensors.count(name)) {
            auto *t = p->tensors.at(name);
            if (f.size() != static_cast<size_t>(ggml_nelements(t)))
                throw std::runtime_error("unprepared stage input: " + name);
            // The host vector is always f32; a cache tensor stores its own element type.
            // Converting is the difference between storing half the values and storing
            // four-byte floats reinterpreted as two-byte halves.
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> halves(f.size());
                ggml_fp32_to_fp16_row(f.data(), halves.data(), static_cast<int64_t>(f.size()));
                ggml_backend_tensor_set(t, halves.data(), 0, ggml_nbytes(t));
            } else {
                throw std::runtime_error("unsupported input element type for " + name);
            }
        }
    };
    put("hidden_input", d.x);
    put("key_cache", d.kcache);
    put("value_cache", d.vcache);
    if (p->key_offsets) {
        // Row r of the cache is the position r's head slice, and the scatter addresses the
        // row and the head plane separately (set-rows.cu: dst_row*s1 + i02*s2), so one
        // offset per (new position, kv head) is the whole specification.
        std::vector<int64_t> ko(static_cast<size_t>(o.q) * static_cast<size_t>(o.kv_heads));
        for (int64_t h = 0; h < o.kv_heads; ++h)
            for (int64_t t = 0; t < o.q; ++t)
                ko[static_cast<size_t>(t + o.q * h)] = o.history + t;
        for (auto *t : {p->key_offsets, p->value_offsets})
            ggml_backend_tensor_set(t, ko.data(), 0, ko.size() * sizeof(int64_t));
    }
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
            auto *flat = ggml_mul_mat(ctx, p->weights.at(role), n);
            // llama.cpp adds the stored bias to the projection, then applies QK norm and
            // RoPE; the order is part of the declared graph.
            if (o.attention_bias)
                flat = ggml_add(ctx, flat, p->weights.at(role + "_bias"));
            // The per-hidden variant normalizes the flat projection before the head split.
            if (rotary && o.qk_norm == "rms_hidden")
                flat = norm(flat, role == "q" ? "q_norm" : "k_norm");
            auto *v = ggml_reshape_3d(ctx, flat, o.head_dim, heads, o.q);
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
        // Both appends return the complete cache with an explicit dependency on
        // new K/V. Readers therefore depend on the writes, including in block.
        // SET-inplace can only target an f32 cache; a half cache is appended with SET_ROWS,
        // which is the scatter the CPU contract uses for any element type.
        ggml_tensor *kr = nullptr, *vr = nullptr;
        if (kc->type == GGML_TYPE_F32) {
            kr = ggml_set_inplace(ctx, kc, key, kc->nb[1], kc->nb[2], kc->nb[3], offset);
            vr = ggml_set_inplace(ctx, vc, value, vc->nb[1], vc->nb[2], vc->nb[3], offset);
        } else {
            kr = ggml_set_rows(ctx, kc, key, p->key_offsets);
            vr = ggml_set_rows(ctx, vc, value, p->value_offsets);
        }
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
    if (ffn && o.experts == 0) {
        auto *n = norm(hidden, "post_attention_norm");
        auto *g = ggml_mul_mat(ctx, p->weights.at("gate"), n);
        auto *u = ggml_mul_mat(ctx, p->weights.at("up"), n);
        auto *active = ggml_swiglu_split(ctx, g, u);
        materialize("hidden_output",
                    ggml_add(ctx, ggml_mul_mat(ctx, p->weights.at("down"), active), hidden));
        p->node_ranges["dense_ffn"] = {attention ? p->node_ranges.at("attention").second : 0,
                                       ggml_graph_n_nodes(p->graph)};
    } else if (ffn) {
        // The sparse block as llama.cpp builds it: router scores, a softmax over all
        // experts, top-k selection, three indirect products, the weighted reduction over
        // the selected experts, then the shared expert (if the checkpoint has one) and the
        // residual.
        auto *n = norm(hidden, "post_attention_norm");
        auto *logits = ggml_mul_mat(ctx, p->weights.at("router"), n);  // [experts, tokens]
        auto *probs = ggml_soft_max(ctx, logits);
        auto *selected = ggml_argsort_top_k(ctx, probs, static_cast<int>(o.top_k));
        // The ids the three indirect products use.  Absent an explicit routing arm this is
        // the router's own top-k, so the published graph is unchanged; with one, the id is
        // both the plane that is read and the probability it is weighted by, which is the
        // only reading an explicit-ids graph has.
        auto *routed = p->route_ids ? p->route_ids : selected;
        auto *weights = ggml_get_rows(
            ctx, ggml_reshape_3d(ctx, probs, 1, o.experts, o.q), routed);  // [1, k, tokens]
        if (o.normalize_topk) {
            // Renormalizing the selected weights is what the declared variant does; the
            // clamp is llama.cpp's own guard against a zero sum.
            auto *w2 = ggml_reshape_2d(ctx, weights, o.top_k, o.q);
            auto *sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY);
            weights = ggml_reshape_3d(ctx, ggml_div(ctx, w2, sum), 1, o.top_k, o.q);
        }
        auto *cur = ggml_reshape_3d(ctx, n, o.hidden, 1, o.q);
        // The router's own chain stays in the graph either way, so the priced op set does
        // not change; only the ids the three indirect products are given do.
        auto *up = ggml_mul_mat_id(ctx, p->weights.at("up_exps"), cur, routed);
        auto *gate = ggml_mul_mat_id(ctx, p->weights.at("gate_exps"), cur, routed);
        auto *active = ggml_swiglu_split(ctx, gate, up);
        auto *experts = ggml_mul_mat_id(ctx, p->weights.at("down_exps"), active, routed);
        experts = ggml_mul(ctx, experts, weights);  // [hidden, k, tokens]
        auto *moe = ggml_view_2d(ctx, experts, o.hidden, o.q, experts->nb[2], 0);
        for (int64_t i = 1; i < o.top_k; ++i)
            moe = ggml_add(ctx, moe,
                           ggml_view_2d(ctx, experts, o.hidden, o.q, experts->nb[2],
                                        i * experts->nb[1]));
        if (o.shared_intermediate > 0) {
            auto *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, p->weights.at("gate_inp_shexp"), n));
            auto *shared =
                ggml_swiglu_split(ctx, ggml_mul_mat(ctx, p->weights.at("gate_shexp"), n),
                                  ggml_mul_mat(ctx, p->weights.at("up_shexp"), n));
            shared = ggml_mul(ctx, ggml_mul_mat(ctx, p->weights.at("down_shexp"), shared), sg);
            moe = ggml_add(ctx, moe, shared);
        }
        materialize("hidden_output", ggml_add(ctx, moe, hidden));
        p->node_ranges["moe_ffn"] = {attention ? p->node_ranges.at("attention").second : 0,
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
    const auto elements = static_cast<size_t>(ggml_nelements(t));
    std::vector<float> f(elements);
    if (t->type == GGML_TYPE_F16) {
        // Read the stored halves, then widen: `ggml_nbytes` is two bytes per element
        // here, so a direct copy into a float buffer would fill only half of it.
        std::vector<ggml_fp16_t> halves(elements);
        ggml_backend_tensor_get(t, halves.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(halves.data(), f.data(), static_cast<int64_t>(elements));
        return f;
    }
    ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    return f;
}
static void capture(Stage &s, Data &d) {
    // A preparation strip produces only the ports of the stages it contains, so a port that
    // is absent is simply not part of what the strip computes.
    auto take = [&](const std::string &name, std::vector<float> &into) {
        if (s.ports.count(name))
            into = read_tensor(s.ports.at(name));
    };
    take("query", d.query);
    take("new_key", d.key);
    take("new_value", d.value);
    take("attention_gate", d.agate);
    take("attention_hidden", d.attn_hidden);
    take("hidden_output", d.final_hidden);
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
    if (o.attention_bias) {
        std::vector<float> bias;
        d.weights.at(role + "_bias").row(0, bias);
        const size_t width = bias.size();
        for (size_t i = 0; i < f.size(); ++i)
            f[i] += bias[i % width];
    }
    if (rotary && o.qk_norm != "none")
        f = normalized(f, o.qk_norm == "rms_hidden" ? o.hidden : o.head_dim,
                       d.weights.at(role == "q" ? "q_norm" : "k_norm"), o.norm_eps);
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
static std::map<size_t, double> moe_reference(const Options &o, const Data &d,
                                              const std::vector<float> &h,
                                              const std::vector<size_t> &indices) {
    // The independent double-precision path through the same declared sparse block: router
    // scores, softmax over all experts, the declared top-k rule, the three expert products,
    // the weighted reduction, the shared expert and the residual.  It reads the stored
    // (possibly quantized) weights, so a disagreement is a graph disagreement, not a
    // storage one.
    auto n = normalized(h, o.hidden, d.weights.at("post_attention_norm"), o.norm_eps);
    std::map<size_t, double> result;
    std::vector<float> row;
    std::set<size_t> tokens;
    for (const size_t ix : indices)
        tokens.insert(ix / static_cast<size_t>(o.hidden));
    for (const size_t t : tokens) {
        const float *x = n.data() + t * static_cast<size_t>(o.hidden);
        std::vector<double> logits(static_cast<size_t>(o.experts)), probs(logits.size());
        for (int64_t expert = 0; expert < o.experts; ++expert) {
            d.weights.at("router").row(expert, row);
            logits[static_cast<size_t>(expert)] = dot(row, x);
        }
        const double largest = *std::max_element(logits.begin(), logits.end());
        double total = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - largest);
            total += probs[i];
        }
        for (double &p : probs)
            p /= total;
        std::vector<size_t> order(probs.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::partial_sort(order.begin(), order.begin() + o.top_k, order.end(),
                          [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        if (o.route || o.stack < o.experts) {
            // The same declared routing the graph is given, so the check stays a check of
            // the graph and not of a routing the graph did not run.  With a `k`-plane
            // stack the id is both the probability it weights by and the plane it reads,
            // which is what an explicit-ids graph does.
            const int64_t stride =
                std::max<int64_t>(1, o.stack / std::max<int64_t>(1, o.top_k));
            for (int64_t i = 0; i < o.top_k; ++i)
                order[static_cast<size_t>(i)] = static_cast<size_t>(
                    (static_cast<int64_t>(t) + i * stride) % o.stack);
        }
        double selected_sum = 0;
        for (int64_t i = 0; i < o.top_k; ++i)
            selected_sum += probs[order[static_cast<size_t>(i)]];
        std::vector<double> out(static_cast<size_t>(o.hidden), 0.0);
        std::vector<double> gate(static_cast<size_t>(o.intermediate)),
            up(static_cast<size_t>(o.intermediate)), active(gate.size());
        for (int64_t i = 0; i < o.top_k; ++i) {
            const size_t expert = order[static_cast<size_t>(i)];
            const double weight =
                o.normalize_topk ? probs[expert] / std::max(selected_sum, 6.103515625e-5)
                                 : probs[expert];
            for (int64_t j = 0; j < o.intermediate; ++j) {
                d.weights.at("gate_exps").row(static_cast<int64_t>(expert) * o.intermediate + j,
                                            row);
                gate[static_cast<size_t>(j)] = dot(row, x);
                d.weights.at("up_exps").row(static_cast<int64_t>(expert) * o.intermediate + j, row);
                up[static_cast<size_t>(j)] = dot(row, x);
                const double g = gate[static_cast<size_t>(j)];
                active[static_cast<size_t>(j)] =
                    static_cast<double>(static_cast<float>(g / (1.0 + std::exp(-g)))) *
                    up[static_cast<size_t>(j)];
            }
            for (int64_t k = 0; k < o.hidden; ++k) {
                d.weights.at("down_exps").row(static_cast<int64_t>(expert) * o.hidden + k, row);
                double value = 0;
                for (int64_t j = 0; j < o.intermediate; ++j)
                    value += static_cast<double>(row[static_cast<size_t>(j)]) *
                             active[static_cast<size_t>(j)];
                out[static_cast<size_t>(k)] += weight * value;
            }
        }
        if (o.shared_intermediate > 0) {
            d.weights.at("gate_inp_shexp").row(0, row);
            const double g = 1.0 / (1.0 + std::exp(-dot(row, x)));
            std::vector<double> sh(static_cast<size_t>(o.shared_intermediate)),
                shup(static_cast<size_t>(o.shared_intermediate));
            for (int64_t j = 0; j < o.shared_intermediate; ++j) {
                d.weights.at("gate_shexp").row(j, row);
                sh[static_cast<size_t>(j)] = dot(row, x);
                d.weights.at("up_shexp").row(j, row);
                shup[static_cast<size_t>(j)] = dot(row, x);
                const double v = sh[static_cast<size_t>(j)];
                sh[static_cast<size_t>(j)] =
                    static_cast<double>(static_cast<float>(v / (1.0 + std::exp(-v)))) *
                    shup[static_cast<size_t>(j)];
            }
            for (int64_t k = 0; k < o.hidden; ++k) {
                d.weights.at("down_shexp").row(k, row);
                double value = 0;
                for (int64_t j = 0; j < o.shared_intermediate; ++j)
                    value += static_cast<double>(row[static_cast<size_t>(j)]) *
                             sh[static_cast<size_t>(j)];
                out[static_cast<size_t>(k)] += g * value;
            }
        }
        for (int64_t k = 0; k < o.hidden; ++k)
            out[static_cast<size_t>(k)] +=
                h[t * static_cast<size_t>(o.hidden) + static_cast<size_t>(k)];
        for (const size_t ix : indices)
            if (ix / static_cast<size_t>(o.hidden) == t)
                result[ix] = out[ix % static_cast<size_t>(o.hidden)];
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
              << ",\"norm_epsilon\":" << o.norm_eps;
    // A property of the model's projections, emitted whenever it was declared so the receipt
    // and the case describe the same config; the stage-scoped `included` list is what says
    // whether the timed graph itself contains the bias adds.
    if (o.experts > 0) {
        std::cout << ",\"experts\":" << o.experts << ",\"top_k\":" << o.top_k;
        // How many expert planes the priced graph holds; a plan's resident requirement is
        // this many `E0`, not one.  Omitted when it is the whole stack, so every published
        // config (and therefore every published contract hash) stays byte-identical.
        if (o.stack != o.experts)
            std::cout << ",\"expert_stack\":" << o.stack;
        std::cout << ",\"shared_intermediate\":" << o.shared_intermediate
                  << ",\"normalize_topk\":" << (o.normalize_topk ? "true" : "false")
                  << ",\"attention_bias\":" << (o.attention_bias ? "true" : "false");
    }
    else if (o.attention_bias)
        std::cout << ",\"attention_bias\":true";
    std::cout << ",\"weight_dtypes\":";
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
        if (o.attention_bias)
            for (const auto &s : {"Q_bias_add", "K_bias_add", "V_bias_add"})
                v.push_back(s);
        if (o.attention_gate == "sigmoid")
            v.push_back("gate_projection");
        if (o.qk_norm == "rms")
            for (const auto &s :
                 {"Q_per_head_RMSNorm", "Q_norm_scale", "K_per_head_RMSNorm", "K_norm_scale"})
                v.push_back(s);
        else if (o.qk_norm == "rms_hidden")
            for (const auto &s : {"Q_RMSNorm", "Q_norm_scale", "K_RMSNorm", "K_norm_scale"})
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
    if (kind == "moe_ffn") {
        std::vector<std::string> v = {"post_attention_RMSNorm",
                                      "post_attention_norm_scale",
                                      "router_projection",
                                      "router_softmax",
                                      "top_k_selection",
                                      "top_k_weight_gather"};
        if (o.normalize_topk)
            v.push_back("top_k_weight_normalize");
        for (const auto &s : {"expert_gate_projection", "expert_up_projection", "SiLU_gate",
                              "gate_times_up", "expert_down_projection", "expert_weighting"})
            v.push_back(s);
        for (int64_t i = 1; i < o.top_k; ++i)
            v.push_back("expert_accumulate");
        if (o.shared_intermediate > 0)
            for (const auto &s :
                 {"shared_expert_gate_inp", "shared_sigmoid", "shared_expert_gate_projection",
                  "shared_expert_up_projection", "SiLU_gate_shared", "gate_times_up_shared",
                  "shared_expert_down_projection", "shared_gate_scale"})
                v.push_back(s);
        v.push_back("ffn_residual_add");
        v.push_back("materialize_hidden");
        return v;
    }
    std::vector<std::string> v;
    for (const auto &s : {"projection", "attention", o.experts == 0 ? "dense_ffn" : "moe_ffn"}) {
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
    // The upstream preparation only has to produce the ports the measured stage reads:
    // `hidden_input` for a projection, the three projections for an attention, a whole block
    // for a feed-forward.  Building the strip that is needed instead of always a block keeps
    // a stage measurable when the rest of the graph has no weights declared -- a sparse
    // model's projection is still a projection, and its feed-forward is not a dense one.
    const std::string prep_kind =
        stage_kind == "attention"      ? "projection"
        : stage_kind == "dense_ffn" ||
                stage_kind == "moe_ffn" ? "block"
                                        : "";
    if (!prep_kind.empty()) {
        auto prep = build_stage(o, d, prep_kind);
        compute(prep->s, prep->graph);
        capture(*prep, d);
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_upstream_preparation");
    auto stage = build_stage(o, d, stage_kind);
    DENSE_CUDA_MEMORY_CHECKPOINT("after_stage_allocation");
    auto &s = stage->s;
    std::vector<int64_t> wall;
    const auto time_once = [](Session &ss, ggml_cgraph *g) {
        const auto begin = std::chrono::steady_clock::now();
        compute(ss, g);
        const auto end = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns <= 0)
            throw std::runtime_error("nonpositive CUDA sample");
        return ns;
    };
    // The first execution of this graph, timed rather than thrown away.  Everything the backend
    // does once per graph -- module loading, kernel selection, scratch growth -- lands here, and
    // nothing below can see it, because every sample re-runs this same, already primed graph.
    // The IR executor never gets that discount: it builds a graph per node and runs it once
    // (`ir_runner.cpp:1232`), so this is the price it actually pays.
    const auto priming_ns = time_once(s, stage->graph);
    std::vector<int64_t> first_run;
    for (int i = 0; i < o.first_runs; ++i)
        first_run.push_back(time_once(s, stage->graph));
    DENSE_CUDA_MEMORY_CHECKPOINT("after_setup");
    if (stage_kind == "block")
        capture(*stage, d);
    for (int i = 0; i < o.warmup; ++i)
        compute(s, stage->graph);
    DENSE_CUDA_MEMORY_CHECKPOINT("after_warmup");
    for (int i = 0; i < o.samples; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        compute_batch(s, stage->graph, o.repetitions);
        const auto end = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns <= 0)
            throw std::runtime_error("nonpositive CUDA sample");
        wall.push_back(ns);
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_samples");
    // Per-node sweep: run the graph truncated to its first k nodes, for every k.  The median of
    // the timed batches at k, divided by the timed repetitions, is the cost of that prefix; the
    // difference between successive prefixes is the marginal cost of node k.  This is the
    // measurement the primitive key needs: it attributes time to a ggml node -- which carries its
    // own op name, shape and dtypes -- instead of to a stage, which carries none of them.
    std::vector<int64_t> prefix_ns;
    std::vector<double> prefix_ratio, prefix_pair_spread;
    std::vector<int> prefix_nodes;
    std::vector<int64_t> reference_ns;
    int64_t node_profile_reference = 0;
    if (o.node_profile > 0) {
        const int total_nodes = ggml_graph_n_nodes(stage->graph);
        // Prefix graphs are built by expanding nodes 0..k-1 in the stage's own topological
        // order.  `ggml_build_forward_expand` pulls in a node's parents, and every parent sits
        // at a lower index, so the result is exactly the prefix.  The node objects are shared
        // with the stage graph and are already resident, so nothing is reallocated: the bytes
        // timed here are the bytes the published price was taken on.
        ggml_context *graph_ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        if (!graph_ctx)
            throw std::runtime_error("could not allocate the node-profile graph context");
        for (int k = 1; k <= total_nodes; ++k) {
            ggml_cgraph *prefix = ggml_new_graph_custom(graph_ctx, total_nodes + 1, false);
            for (int j = 0; j < k; ++j)
                ggml_build_forward_expand(prefix, ggml_graph_node(stage->graph, j));
            compute(s, prefix); // first run of this prefix pays its own per-graph setup
            // Every pair is (prefix, full graph) back to back, so a level change between
            // pairs cancels in the ratio instead of landing on whichever prefix happened to
            // be running.  The full graph is the price's own object, not a rebuilt copy.
            // The order alternates because the first execution after a gap is slower -- with
            // a fixed order that transient lands on the subject every time and shows up as a
            // self-ratio above one.  The pair right after priming is dropped for the same
            // reason.
            std::vector<double> ratio;
            std::vector<int64_t> ref;
            for (int i = 0; i < o.node_profile_samples; ++i) {
                int64_t subject = 0, reference = 0;
                if (i % 2 == 0) {
                    subject = time_once(s, prefix);
                    reference = time_once(s, stage->graph);
                } else {
                    reference = time_once(s, stage->graph);
                    subject = time_once(s, prefix);
                }
                if (i == 0)
                    continue;
                ratio.push_back(static_cast<double>(subject) /
                                static_cast<double>(reference));
                ref.push_back(reference);
            }
            const size_t mid = ratio.size() / 2;
            std::vector<double> spread = ratio;
            std::sort(spread.begin(), spread.end());
            prefix_pair_spread.push_back(
                spread[spread.size() - 1 - spread.size() / 10] - spread[spread.size() / 10]);
            std::nth_element(ratio.begin(), ratio.begin() + mid, ratio.end());
            std::nth_element(ref.begin(), ref.begin() + mid, ref.end());
            prefix_ratio.push_back(ratio[mid]);
            reference_ns.push_back(ref[mid]);
            prefix_nodes.push_back(ggml_graph_n_nodes(prefix));
        }
        ggml_free(graph_ctx);
        // One absolute scale for the whole sweep: the median full-graph execution across every
        // pair.  `prefix_ns` is therefore the cost of the prefix in the units the published
        // price is stated in, with the level divided out by the ratio.
        std::vector<int64_t> scale = reference_ns;
        std::nth_element(scale.begin(), scale.begin() + scale.size() / 2, scale.end());
        const double reference = static_cast<double>(scale[scale.size() / 2]);
        node_profile_reference = scale[scale.size() / 2];
        prefix_ns.assign(prefix_ratio.size(), 0);
        for (size_t i = 0; i < prefix_ratio.size(); ++i)
            prefix_ns[i] = static_cast<int64_t>(std::llround(prefix_ratio[i] * reference));
        compute(s, stage->graph); // leave the stage full and its outputs valid for the checks
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_node_profile");
    // In-process sensor: a fixed workload timed in the same process, so that a per-process
    // device-state band can be divided out instead of being mistaken for a shape effect.
    std::vector<int64_t> sensor_wall;
    if (o.sensor_q > 0) {
        Options so = o;
        so.stage = "projection";
        so.q = o.sensor_q;
        so.history = 0;
        const auto saved_x = d.x;
        const auto saved_positions = d.positions;
        d.x.assign(static_cast<size_t>(product({o.hidden, so.q})), 0.5f);
        d.positions.assign(static_cast<size_t>(so.q), 0);
        auto sensor = build_stage(so, d, "projection");
        compute(sensor->s, sensor->graph);
        for (int i = 0; i < 10; ++i)
            compute(sensor->s, sensor->graph);
        for (int i = 0; i < 10; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            compute_batch(sensor->s, sensor->graph, o.repetitions);
            const auto end = std::chrono::steady_clock::now();
            const auto ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            if (ns <= 0)
                throw std::runtime_error("nonpositive sensor sample");
            sensor_wall.push_back(ns);
        }
        d.x = saved_x;
        d.positions = saved_positions;
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_sensor");
    // A graph the backend has never seen, with its weight uploads flushed before the clock
    // starts.  This is one step further than `priming_ns`: the process is warm (modules loaded,
    // handles created) but this graph is not, which is exactly the executor's standing state on
    // node k > 1 of a plan.  The second execution of the same fresh graph is recorded as well,
    // so a per-graph setup cost separates from a per-buffer one.
    std::vector<int64_t> fresh_first, fresh_second;
    for (int i = 0; i < o.fresh_graphs; ++i) {
        auto fresh = build_stage(o, d, stage_kind);
        ggml_backend_synchronize(fresh->s.backend);
        fresh_first.push_back(time_once(fresh->s, fresh->graph));
        fresh_second.push_back(time_once(fresh->s, fresh->graph));
    }
    DENSE_CUDA_MEMORY_CHECKPOINT("after_fresh_graphs");
    std::map<std::string, std::vector<float>> outputs;
    for (const auto &p : stage->ports) {
        outputs[p.first] = read_tensor(p.second);
        for (float v : outputs[p.first])
            if (!std::isfinite(v))
                throw std::runtime_error("nonfinite output: " + p.first);
    }
    const bool p = stage_kind == "projection" || stage_kind == "block",
               a = stage_kind == "attention" || stage_kind == "block",
               f = stage_kind == "dense_ffn" || stage_kind == "moe_ffn" ||
                   stage_kind == "block";
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
        const auto &hidden_in = stage_kind == "block" ? rh : d.attn_hidden;
        const auto reference = o.experts == 0 ? ffn_reference(o, d, hidden_in, ix)
                                              : moe_reference(o, d, hidden_in, ix);
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
                        // A half cache stores the value the graph was given, rounded once.
                        // The comparison is still exact -- it is against the stored value,
                        // not against a tolerance.
                        if (kv_cache_type(o) == GGML_TYPE_F16) {
                            append_exact = append_exact &&
                                           actualk[dst] == half_round(kn[src]) &&
                                           actualv[dst] == half_round(vn[src]);
                        } else
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
                 "\"contract\":\""
              << (o.experts > 0 ? "ggml_cuda_moe_decoder_block_v1"
                                : "ggml_cuda_dense_decoder_block_v1")
              << "\",\"kind\":\"dense_layer\","
                 "\"stage\":"
              << "\"" << o.stage << "\",\"backend\":\"CUDA" << o.cuda_device
              << "\",\"cuda_device\":" << o.cuda_device
              << ",\"cuda_graphs\":\"disabled\",\"cuda_graphs_requested\":\"disabled\","
                 "\"cuda_graphs_applied\":\"disabled\",\"cuda_graph_disable_environment\":\"1\","
                 "\"batch\":1,\"config\":";
    emit_config(o);
    std::cout << ",\"q\":" << o.q << ",\"history\":" << o.history << ",\"warmup\":" << o.warmup
              << ",\"setup_compute_repetitions\":1,"
                 "\"upstream_setup_compute_repetitions\":"
              << (stage_kind == "block" ? 0 : 1) << ",\"samples\":" << o.samples
              << ",\"timed_repetitions\":" << o.repetitions
              << ",\"seed\":" << o.seed
              << ",\"input_generator\":\"mt19937_uniform24_scaled_weights_gamma_near_one_v1\","
                 "\"input_lineage\":\"same_seed_complete_upstream_preparation_outside_timing\","
                 "\"matmul_precision\":\"ggml_cuda_backend_default\",\"activation_dtype\":\"f32\","
                 "\"kv_dtype\":\""
              << o.kv_dtype << "\",\"position_dtype\":\"i32\","
                 "\"implementation_variant\":\"ggml_cuda_"
              << (o.experts > 0 ? "moe" : "dense") << "_materialized_" << o.stage
              << (kv_cache_type(o) == GGML_TYPE_F16 ? "_v2" : "_v1") << "\","
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
        int src_count = 0;
        for (int j = 0; j < GGML_MAX_SRC; ++j)
            src_count += n->src[j] ? 1 : 0;
        std::cout << (i ? "," : "") << "{\"op\":\"" << ggml_op_name(n->op)
                  << "\",\"bytes\":" << ggml_nbytes(n)
                  << ",\"view\":" << (n->view_src ? "true" : "false")
                  << ",\"name\":" << json_quote(n->name)
                  << ",\"type\":\"" << ggml_type_name(n->type) << "\""
                  << ",\"ne\":[" << n->ne[0] << "," << n->ne[1] << "," << n->ne[2] << ","
                  << n->ne[3] << "],\"src_count\":" << src_count
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
                  << (v->type == GGML_TYPE_I32 ? "i32"
                      : v->type == GGML_TYPE_F16 ? "f16"
                                                 : "f32")
                  << "\",\"shape\":[" << v->ne[0]
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
    const size_t kv_per_token = product({o.head_dim, o.kv_heads}) *
                                ggml_type_size(kv_cache_type(o)) * 2,
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
    // The same graph, executed once per number above.  `steady_per_execution_ns` is what the 4c
    // protocol calls the price; `priming_ns` is execution 1 of that graph and `first_run_ns`
    // executions 2..N+1, all before the warm-up.  `fresh_graph_*` are graphs built after the
    // samples, each run once (then twice) with its uploads already flushed.
    std::vector<int64_t> sorted_wall = wall;
    std::nth_element(sorted_wall.begin(), sorted_wall.begin() + sorted_wall.size() / 2,
                     sorted_wall.end());
    const int64_t steady_median = sorted_wall[sorted_wall.size() / 2];
    std::cout << ",\"priming_ns\":" << priming_ns << ",\"steady_median_ns\":" << steady_median
              << ",\"steady_per_execution_ns\":"
              << std::setprecision(17)
              << (static_cast<double>(steady_median) / static_cast<double>(o.repetitions))
              << ",\"first_runs\":" << o.first_runs << ",\"first_run_ns\":";
    emit_ns(first_run);
    std::cout << ",\"fresh_graphs\":" << o.fresh_graphs << ",\"fresh_graph_first_run_ns\":";
    emit_ns(fresh_first);
    std::cout << ",\"fresh_graph_second_run_ns\":";
    emit_ns(fresh_second);
    if (o.sensor_q > 0) {
        std::cout << ",\"sensor_q\":" << o.sensor_q << ",\"sensor_wall_ns\":";
        emit_ns(sensor_wall);
    }
    if (!prefix_ns.empty()) {
        std::cout << ",\"node_profile\":{\"enabled\":true"
                  << ",\"samples_per_prefix\":" << o.node_profile_samples
                  << ",\"method\":\"prefix_truncation_paired_with_full_graph_ratio\","
                  << "\"reference_ns\":" << node_profile_reference
                  << ",\"prefix_nodes\":[";
        for (size_t i = 0; i < prefix_nodes.size(); ++i)
            std::cout << (i ? "," : "") << prefix_nodes[i];
        std::cout << "],\"prefix_ratio\":[";
        for (size_t i = 0; i < prefix_ratio.size(); ++i)
            std::cout << (i ? "," : "") << std::fixed << std::setprecision(6) << prefix_ratio[i];
        std::cout << std::defaultfloat << "],\"prefix_ns\":[";
        for (size_t i = 0; i < prefix_ns.size(); ++i)
            std::cout << (i ? "," : "") << prefix_ns[i];
        std::cout << "],\"prefix_pair_spread\":[";
        for (size_t i = 0; i < prefix_pair_spread.size(); ++i)
            std::cout << (i ? "," : "") << std::fixed << std::setprecision(6)
                      << prefix_pair_spread[i];
        std::cout << std::defaultfloat << "],\"delta_ns\":[";
        for (size_t i = 0; i < prefix_ns.size(); ++i)
            std::cout << (i ? "," : "") << (prefix_ns[i] - (i ? prefix_ns[i - 1] : 0));
        std::cout << "],\"reference_samples_ns\":[";
        for (size_t i = 0; i < reference_ns.size(); ++i)
            std::cout << (i ? "," : "") << reference_ns[i];
        std::cout << "]}";
    }
    std::cout << "}\n";
    if (!passed) {
        std::cerr << "CUDA dense layer numerical or KV validation failed\n";
        return 2;
    }
    return 0;
}
// One case per line, arguments separated by tabs.  Same protocol as the CPU probe: each case
// is a full parse()+run(), its record goes to its own stream, and it is followed by a
// "#case <index> <code>" sentinel.  Building the CUDA context once and reusing it across the
// cases is the point -- a price table that re-pays context creation per case would price the
// context, not the layer.
static std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out(1);
    for (const char c : line) {
        if (c == '\t')
            out.emplace_back();
        else if (c != '\r')
            out.back().push_back(c);
    }
    return out;
}

static int serve_cases(std::istream &in) {
    std::string line;
    std::vector<std::string> owned;
    std::vector<char *> argv;
    size_t index = 0, failed = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        owned.clear();
        argv.clear();
        owned.emplace_back("ggml-dense-layer-cuda-probe");
        for (std::string &token : split_tabs(line))
            if (!token.empty())
                owned.push_back(std::move(token));
        for (std::string &arg : owned)
            argv.push_back(const_cast<char *>(arg.c_str()));
        argv.push_back(nullptr);

        std::ostringstream record;
        std::streambuf *saved = std::cout.rdbuf(record.rdbuf());
        std::string error;
        int code = 1;
        try {
            code = run(parse(static_cast<int>(argv.size()) - 1, argv.data()));
        } catch (const std::exception &e) {
            error = e.what();
        }
        std::cout.rdbuf(saved);
        // A case that fails its own numerical/KV check still produced a record (with
        // "passed":false); keep it next to the failure code.
        if (!record.str().empty())
            std::cout << record.str();
        std::cout << "#case " << index << ' ' << code << '\n';
        std::cout.flush();
        if (code != 0) {
            ++failed;
            std::cerr << "ggml-dense-layer-cuda-probe case " << index << " failed (" << code << ')';
            if (!error.empty())
                std::cerr << ": " << error;
            std::cerr << '\n';
        }
        ++index;
    }
    std::cerr << "ggml-dense-layer-cuda-probe: " << index << " case(s), " << failed << " failed\n";
    return failed == 0 ? 0 : 3;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--cases-from-stdin") {
        try {
            return serve_cases(std::cin);
        } catch (const std::exception &e) {
            std::cerr << "ggml-dense-layer-cuda-probe stdin: " << e.what() << '\n';
            return 1;
        }
    }
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "ggml-dense-layer-cuda-probe --stage projection|attention|dense_ffn|block "
                         "--cuda-device index --cuda-graphs disabled; optional --kv-dtype f32|f16 "
                         "(cache element type, default f32); --list-devices; standalone "
                         "--cases-from-stdin runs one case per input line (tab-separated "
                         "arguments, '#' starts a comment line) and prints each case's record "
                         "followed by a '#case <index> <code>' sentinel, continuing after a "
                         "failing case so a whole price table can be measured in one process; "
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
