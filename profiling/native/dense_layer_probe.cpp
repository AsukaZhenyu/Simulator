// CPU resident decoder block with explicit materialized stage boundaries.
// This generic contract deliberately does not claim to reproduce a model runtime.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#if defined(_WIN32)
#define NOMINMAX
// tlhelp32 requires Windows types; preserve this dependent include order.
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#endif
struct Options {
    std::string stage, attention_gate, qk_norm, rope_mode, compute_mode, kv_dtype;
    // Experts are opt-in: with --experts the feed-forward is the sparse MoE block
    // and --intermediate is the width of one expert instead of the dense width.
    int64_t experts = 0, top_k = 0, shared_intermediate = 0;
    bool normalize_topk = false;
    // Qwen2-MoE stores an F32 bias vector for each of Q, K and V; llama.cpp adds
    // it after the projection and before RoPE. The add is opt-in so a checkpoint
    // that has no biases measures exactly the graph it declares.
    bool attention_bias = false;
    std::map<std::string, std::string> dtypes;
    int64_t hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0, q = 0,
            history = -1, rope_dims = 0, vocab = 0;
    float rope_base = 0, norm_eps = 0;
    int threads = 0, warmup = -1, samples = 0;
    int64_t min_warmup_ms = -1;
    std::vector<int> cpu_ids;
    uint32_t seed = 0;
    // Declared residency regime of the timed samples. Zero means the sample
    // starts with whatever the previous sample left in the caches, which is what
    // a resident interactive decode looks like. A positive value streams that
    // many bytes of scratch immediately before each warmup iteration and each
    // timed sample, so the stage reads its own operands from memory instead of
    // from a cache the previous sample primed -- what a deployment does when the
    // block is one of many tenants. The stream is outside the timed region: the
    // sample is priced cold, the act of making it cold is not.
    int64_t evict_bytes = 0;
    // Opt-in: also record how many CPU cycles the whole process burned inside
    // each timed sample. Cycles are what the work costs independently of the
    // clock frequency the cores happened to run at, so a sample that is slow but
    // used the same cycles as a fast one was taken at a lower frequency tier.
    // QueryProcessCycleTime sums every thread in the process, which is what is
    // needed here because the graph runs on the threadpool workers, not on the
    // main thread. Additive JSON key; default output is unchanged.
    bool thread_cycles = false;
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
static std::vector<int> cpu_ids(const std::string &text) {
    std::vector<int> values;
    std::set<int> seen;
    if (text.empty() || text.back() == ',')
        throw std::runtime_error("--cpu-mask requires a nonempty comma-separated logical CPU list");
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto value = integer(item);
        if (value < 0 || value >= GGML_MAX_N_THREADS ||
            !seen.insert(static_cast<int>(value)).second)
            throw std::runtime_error("invalid or duplicate logical CPU in --cpu-mask");
        values.push_back(static_cast<int>(value));
    }
    std::sort(values.begin(), values.end());
    return values;
}
static Options parse(int argc, char **argv) {
    Options o;
    std::set<std::string> seen;
    const std::map<std::string, std::string> dtype_keys = {
        {"--q-dtype", "q"},      {"--k-dtype", "k"},
        {"--v-dtype", "v"},      {"--gate-proj-dtype", "gate_proj"},
        {"--o-dtype", "output"}, {"--ffn-gate-dtype", "gate"},
        {"--up-dtype", "up"},    {"--down-dtype", "down"},
        {"--lm-head-dtype", "lm_head"}, {"--token-embd-dtype", "token_embd"},
        {"--gate-exps-dtype", "gate_exps"}, {"--up-exps-dtype", "up_exps"},
        {"--down-exps-dtype", "down_exps"}, {"--shexp-gate-dtype", "gate_shexp"},
        {"--shexp-up-dtype", "up_shexp"},   {"--shexp-down-dtype", "down_shexp"}};
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc || !seen.insert(key).second)
            throw std::runtime_error("missing value or duplicate option: " + key);
        const std::string v = argv[i + 1];
        if (dtype_keys.count(key))
            o.dtypes[dtype_keys.at(key)] = v;
        else if (key == "--stage")
            o.stage = v;
        else if (key == "--compute-mode")
            o.compute_mode = v;
        else if (key == "--kv-dtype")
            o.kv_dtype = v;
        else if (key == "--min-warmup-ms")
            o.min_warmup_ms = integer(v);
        else if (key == "--evict-bytes")
            o.evict_bytes = integer(v);
        else if (key == "--thread-cycles") {
            const auto t = integer(v);
            if (t != 0 && t != 1)
                throw std::runtime_error("--thread-cycles takes 0 or 1");
            o.thread_cycles = t == 1;
        }
        else if (key == "--cpu-mask")
            o.cpu_ids = cpu_ids(v);
        else if (key == "--attention-gate")
            o.attention_gate = v;
        else if (key == "--qk-norm")
            o.qk_norm = v;
        else if (key == "--rope-mode")
            o.rope_mode = v;
        else if (key == "--hidden")
            o.hidden = integer(v);
        else if (key == "--vocab")
            o.vocab = integer(v);
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
        else if (key == "--attention-bias")
            o.attention_bias = integer(v) != 0;
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
        } else if (key == "--threads" || key == "--warmup" || key == "--samples") {
            const auto t = integer(v);
            if (t < 0 || t > std::numeric_limits<int>::max())
                throw std::runtime_error("option out of range: " + key);
            if (key == "--threads")
                o.threads = static_cast<int>(t);
            if (key == "--warmup")
                o.warmup = static_cast<int>(t);
            if (key == "--samples")
                o.samples = static_cast<int>(t);
        } else
            throw std::runtime_error("unknown option: " + key);
    }
    const std::set<std::string> stages = {"projection", "attention", "dense_ffn", "moe_ffn", "block",
                                          "paired", "embedding", "head"};
    if (!stages.count(o.stage))
        throw std::runtime_error("unsupported stage");
    // The two boundary stages multiply a vocabulary-wide matrix, so they are the
    // only stages that declare one. Everything else must not, or a mis-typed
    // invocation would allocate hundreds of megabytes the block never reads.
    const bool word = o.stage == "embedding" || o.stage == "head";
    if (word != (o.vocab > 0))
        throw std::runtime_error(word ? "embedding/head requires an explicit positive --vocab"
                                      : "--vocab is only valid for embedding/head");
    if (o.stage == "embedding" || o.stage == "head")
        if (!seen.count("--lm-head-dtype") || !seen.count("--token-embd-dtype"))
            throw std::runtime_error("embedding/head require --lm-head-dtype and --token-embd-dtype");
    if (o.stage == "paired") {
        if (!seen.count("--compute-mode") ||
            (o.compute_mode != "graph" && o.compute_mode != "plan"))
            throw std::runtime_error("paired stage requires explicit --compute-mode graph|plan");
        if (!seen.count("--min-warmup-ms") || o.min_warmup_ms < 0 || o.min_warmup_ms > 600000)
            throw std::runtime_error("paired stage requires explicit --min-warmup-ms in 0..600000");
    } else if (seen.count("--compute-mode") || seen.count("--min-warmup-ms"))
        throw std::runtime_error("paired runtime options are only valid for paired diagnostics");
    if (o.hidden <= 0 || o.intermediate <= 0 || o.heads <= 0 || o.kv_heads <= 0 ||
        o.head_dim <= 0 || o.heads % o.kv_heads || o.q <= 0 || o.history < 0)
        throw std::runtime_error("invalid block geometry");
    if (o.history > std::numeric_limits<int>::max() ||
        o.q > std::numeric_limits<int>::max() - o.history)
        throw std::runtime_error("sequence indexing overflow");
    if (o.attention_gate != "none" && o.attention_gate != "sigmoid")
        throw std::runtime_error("attention_gate must be none or sigmoid");
    if (o.qk_norm != "none" && o.qk_norm != "rms" && o.qk_norm != "rms_hidden")
        throw std::runtime_error("qk_norm must be none, rms (one gain per head) or "
                                 "rms_hidden (one gain per hidden unit, applied before "
                                 "the head split)");
    if (o.rope_mode != "normal" || o.rope_dims <= 0 || o.rope_dims % 2 ||
        o.rope_dims > o.head_dim || o.rope_base <= 1 || o.norm_eps <= 0)
        throw std::runtime_error("requires normal RoPE, positive even dimensions <= head_dim, "
                                 "explicit base and epsilon");
    if (o.threads < 1 || o.threads > GGML_MAX_N_THREADS || o.warmup < 0 || o.samples < 3 ||
        o.samples > 10000 || !seen.count("--seed"))
        throw std::runtime_error(
            "requires explicit threads/warmup/samples/seed, samples in 3..10000");
    if (o.evict_bytes < 0 || o.evict_bytes > (int64_t{1} << 34))
        throw std::runtime_error("--evict-bytes must be 0..17179869184");
    // The paired diagnostic alternates two paths over one resident graph to
    // separate serial from joint launch cost. Flushing between the arms would
    // change both of them and destroy exactly the comparison it exists to make,
    // so the cold regime is refused rather than silently ignored.
    if (o.evict_bytes > 0 && o.stage == "paired")
        throw std::runtime_error("--evict-bytes is only valid for single-stage measurements");
    if (!o.cpu_ids.empty() && o.cpu_ids.size() != static_cast<size_t>(o.threads))
        throw std::runtime_error("strict --cpu-mask must contain exactly --threads logical CPUs");
    const std::set<std::string> types = {"f32", "f16", "q4_k", "q6_k", "q8_0"};
    if (o.kv_dtype != "f32" && o.kv_dtype != "f16")
        throw std::runtime_error("--kv-dtype must be f32 or f16 (cache element type, not weights)");
    // MoE is a declared variant, not a flag on the dense block: the two use
    // different weights, so a stage that names one must not be measured with the
    // other's weights present.
    if (o.experts != 0 || o.top_k != 0) {
        if (o.experts <= 0 || o.top_k <= 0 || o.top_k > o.experts || o.shared_intermediate < 0)
            throw std::runtime_error("--experts and --top-k must be positive, top-k <= experts, "
                                     "--shared-intermediate >= 0");
        // Every stage of a sparse contract declares the expert tensors, but only
        // the sparse feed-forward (and a joint block) reads them, so projection,
        // attention and the word stages are measured without generating them.
        if (o.stage == "dense_ffn")
            throw std::runtime_error("a dense feed-forward cannot be measured with an expert block");
        for (const auto &role : {"gate_exps", "up_exps", "down_exps"})
            if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
                throw std::runtime_error(std::string("missing/unsupported expert matrix dtype: ") +
                                         role);
        if (o.shared_intermediate > 0) {
            for (const auto &role : {"gate_shexp", "up_shexp", "down_shexp"})
                if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
                    throw std::runtime_error(
                        std::string("missing/unsupported shared expert matrix dtype: ") + role);
        } else {
            for (const auto &role : {"gate_shexp", "up_shexp", "down_shexp"})
                if (o.dtypes.count(role))
                    throw std::runtime_error("shared expert dtype declared without --shared-intermediate");
        }
        o.dtypes["router"] = "f32";
        if (o.shared_intermediate > 0)
            o.dtypes["gate_inp_shexp"] = "f32";
    } else {
        if (o.shared_intermediate != 0 || o.normalize_topk || o.dtypes.count("gate_exps") ||
            o.dtypes.count("up_exps") || o.dtypes.count("down_exps") ||
            o.dtypes.count("gate_shexp"))
            throw std::runtime_error("expert options declared without an expert block");
        if (o.stage == "moe_ffn")
            throw std::runtime_error("moe_ffn requires --experts and --top-k");
    }
    for (const auto &role : {"q", "k", "v", "output"})
        if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
            throw std::runtime_error(std::string("missing/unsupported matrix dtype: ") + role);
    // Exactly one feed-forward is declared: the dense matrices or the expert
    // tensors, never both.
    for (const auto &role : {"gate", "up", "down"})
        if (o.experts == 0) {
            if (!o.dtypes.count(role) || !types.count(o.dtypes.at(role)))
                throw std::runtime_error(std::string("missing/unsupported matrix dtype: ") + role);
        } else if (o.dtypes.count(role)) {
            throw std::runtime_error("dense feed-forward dtype declared for an expert block");
        }
    // The three Q/K/V biases are F32 parameters of the projection, like the
    // norms: the caller says whether they exist, never what type they are.
    if (o.attention_bias)
        for (const auto &role : {"q_bias", "k_bias", "v_bias"})
            o.dtypes[role] = "f32";
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
    if (o.vocab > 0) {
        if (!types.count(o.dtypes.at("lm_head")) || !types.count(o.dtypes.at("token_embd")))
            throw std::runtime_error("unsupported word matrix dtype");
        o.dtypes["final_norm"] = "f32";
    }
    if (o.qk_norm != "none") {
        o.dtypes["q_norm"] = "f32";
        o.dtypes["k_norm"] = "f32";
    }
    return o;
}
static ggml_type kv_cache_type(const Options &o) {
    return o.kv_dtype == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
}
// The value the cache actually holds: an f16 cache rounds every appended value,
// so the append check and the scalar reference must compare against the rounded
// value, not against the f32 source.
static float stored(const Options &o, float v) {
    return kv_cache_type(o) == GGML_TYPE_F16
               ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v))
               : v;
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
static std::string mask_hex(uint64_t mask) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << mask;
    return out.str();
}
static std::vector<int> mask_ids(uint64_t mask) {
    std::vector<int> values;
    for (int i = 0; i < 64; ++i)
        if (mask & (uint64_t{1} << i))
            values.push_back(i);
    return values;
}
static void emit_ids(const std::vector<int> &ids) {
    std::cout << "[";
    for (size_t i = 0; i < ids.size(); ++i)
        std::cout << (i ? "," : "") << ids[i];
    std::cout << "]";
}
#if defined(_WIN32)
struct CoreMask {
    WORD group;
    uint64_t mask;
};
struct PhysicalCore {
    BYTE efficiency_class, flags;
    std::vector<CoreMask> masks;
};
struct CpuTopology {
    WORD group_count = 0;
    std::vector<USHORT> process_groups;
    uint64_t process_mask = 0, system_mask = 0;
    std::vector<PhysicalCore> cores;
};
static CpuTopology cpu_topology() {
    CpuTopology result;
    result.group_count = GetActiveProcessorGroupCount();
    if (!result.group_count)
        throw std::runtime_error("GetActiveProcessorGroupCount failed");
    USHORT count = result.group_count;
    result.process_groups.resize(count);
    if (!GetProcessGroupAffinity(GetCurrentProcess(), &count, result.process_groups.data()))
        throw std::runtime_error("GetProcessGroupAffinity failed: " +
                                 std::to_string(GetLastError()));
    result.process_groups.resize(count);
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask))
        throw std::runtime_error("GetProcessAffinityMask failed: " +
                                 std::to_string(GetLastError()));
    result.process_mask = process_mask;
    result.system_mask = system_mask;
    DWORD size = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || !size)
        throw std::runtime_error("GetLogicalProcessorInformationEx size query failed");
    std::vector<uint8_t> bytes(size);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(bytes.data()), &size))
        throw std::runtime_error("GetLogicalProcessorInformationEx failed: " +
                                 std::to_string(GetLastError()));
    size_t offset = 0;
    while (offset < size) {
        const auto *info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
            bytes.data() + offset);
        if (info->Size < sizeof(DWORD) * 2 || info->Size > size - offset ||
            info->Relationship != RelationProcessorCore)
            throw std::runtime_error("invalid Windows physical core topology record");
        const auto &processor = info->Processor;
        PhysicalCore core{processor.EfficiencyClass, processor.Flags, {}};
        for (WORD group = 0; group < processor.GroupCount; ++group)
            core.masks.push_back({processor.GroupMask[group].Group,
                                  static_cast<uint64_t>(processor.GroupMask[group].Mask)});
        result.cores.push_back(std::move(core));
        offset += info->Size;
    }
    return result;
}
static void validate_cpu_mask(const Options &o) {
    const auto topology = cpu_topology();
    if (topology.group_count != 1 || topology.process_groups.size() != 1 ||
        topology.process_groups[0] != 0)
        throw std::runtime_error("explicit CPU masks require one Windows processor group 0; "
                                 "local GGML Windows affinity does not support multiple groups");
    uint64_t active = 0;
    for (const auto &core : topology.cores)
        for (const auto &mask : core.masks)
            if (mask.group == 0)
                active |= mask.mask;
    for (const int id : o.cpu_ids)
        if (id >= 64 || !(active & (uint64_t{1} << id)) ||
            !(topology.process_mask & (uint64_t{1} << id)))
            throw std::runtime_error("requested logical CPU is unavailable to this process: " +
                                     std::to_string(id));
}
static std::set<DWORD> process_threads() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
        throw std::runtime_error("CreateToolhelp32Snapshot failed: " +
                                 std::to_string(GetLastError()));
    std::set<DWORD> threads;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (!Thread32First(snapshot, &entry)) {
        const DWORD error = GetLastError();
        CloseHandle(snapshot);
        throw std::runtime_error("Thread32First failed: " + std::to_string(error));
    }
    do {
        if (entry.th32OwnerProcessID == GetCurrentProcessId())
            threads.insert(entry.th32ThreadID);
        entry.dwSize = sizeof(entry);
    } while (Thread32Next(snapshot, &entry));
    CloseHandle(snapshot);
    return threads;
}
#else
static void validate_cpu_mask(const Options &) {
    throw std::runtime_error("explicit CPU affinity verification is unsupported on this platform");
}
#endif
static int emit_cpu_topology() {
    std::cout << "{\"schema\":\"ggml_cpu_topology/1\",";
#if defined(_WIN32)
    const auto topology = cpu_topology();
    std::cout << "\"platform\":\"windows\",\"supported\":true,\"processor_group_count\":"
              << topology.group_count << ",\"process_groups\":[";
    for (size_t i = 0; i < topology.process_groups.size(); ++i)
        std::cout << (i ? "," : "") << topology.process_groups[i];
    std::cout << "],\"process_allowed_mask_hex\":\"" << mask_hex(topology.process_mask)
              << "\",\"system_mask_hex\":\"" << mask_hex(topology.system_mask)
              << "\",\"allowed_logical_ids\":";
    emit_ids(mask_ids(topology.process_mask));
    std::cout << ",\"cores\":[";
    for (size_t i = 0; i < topology.cores.size(); ++i) {
        const auto &core = topology.cores[i];
        std::cout << (i ? "," : "") << "{\"core_index\":" << i
                  << ",\"efficiency_class\":" << static_cast<int>(core.efficiency_class)
                  << ",\"flags\":" << static_cast<int>(core.flags) << ",\"group_masks\":[";
        for (size_t j = 0; j < core.masks.size(); ++j) {
            const auto &mask = core.masks[j];
            std::cout << (j ? "," : "") << "{\"group\":" << mask.group << ",\"mask_hex\":\""
                      << mask_hex(mask.mask) << "\",\"logical_ids\":";
            emit_ids(mask_ids(mask.mask));
            std::cout << "}";
        }
        std::cout << "]}";
    }
    std::cout << "],\"source\":\"GetLogicalProcessorInformationEx(RelationProcessorCore);"
                 "GetProcessAffinityMask;GetProcessGroupAffinity\","
                 "\"explicit_affinity_supported\":"
              << ((topology.group_count == 1 && topology.process_groups.size() == 1 &&
                   topology.process_groups[0] == 0)
                      ? "true"
                      : "false");
#else
    std::cout << "\"platform\":\"unknown\",\"supported\":false,"
                 "\"reason\":\"this probe implements Windows topology queries only\","
                 "\"explicit_affinity_supported\":false";
#endif
    std::cout << "}\n";
    return 0;
}

struct Session {
    ggml_context *resident = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_threadpool_t pool = nullptr;
#if defined(_WIN32)
    std::set<DWORD> threads_before_pool;
    GROUP_AFFINITY original_main_affinity{};
    bool restore_main_affinity = false;
    bool explicit_main_binding = false;
#endif
    ~Session() {
        if (backend)
            ggml_backend_synchronize(backend);
        if (allocator)
            ggml_gallocr_free(allocator);
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (backend)
            ggml_backend_free(backend);
        if (pool)
            ggml_threadpool_free(pool);
#if defined(_WIN32)
        if (restore_main_affinity &&
            !SetThreadGroupAffinity(GetCurrentThread(), &original_main_affinity, nullptr))
            std::cerr << "could not restore main thread affinity: " << GetLastError() << '\n';
#endif
        if (ctx)
            ggml_free(ctx);
        if (resident)
            ggml_free(resident);
    }
};

struct Weight {
    ggml_tensor *tensor = nullptr;
    // Expert tensors are [cols, rows, experts]; a dense matrix is experts == 1.
    int64_t cols = 0, rows = 0, experts = 1;
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

static ggml_tensor *typed_input(ggml_context *ctx, const char *name, ggml_type type, int64_t a,
                                int64_t b, int64_t c) {
    product({a, b, c});
    auto *t = ggml_new_tensor_3d(ctx, type, a, b, c);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

static void compute(Session &s, ggml_cgraph *graph) {
    const ggml_status status = ggml_backend_graph_compute(s.backend, graph);
    if (status != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed: " + std::to_string(status));
}

// Declared cold-start support: read a scratch buffer far larger than any
// last-level cache so the next timed sample finds its own operands evicted
// instead of resident from the sample before it. Called between samples and
// never inside one -- the act of making the sample cold is not part of its
// price. The volatile sink keeps the loop from being optimised away; one load
// per cache line is enough to displace it.
struct CacheFlush {
    std::vector<unsigned char> scratch;
    explicit CacheFlush(int64_t bytes) : scratch(static_cast<size_t>(bytes), 1) {}
    void operator()() const {
        if (scratch.empty())
            return;
        volatile unsigned long long sink = 0;
        for (size_t i = 0; i < scratch.size(); i += 64)
            sink = sink + scratch[i];
    }
};

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
    std::vector<int32_t> positions, ids;
};
static Data make_data(const Options &o) {
    Data d;
    std::mt19937 rng(o.seed);
    const auto e = static_cast<int64_t>(product({o.head_dim, o.heads}));
    const auto kv = static_cast<int64_t>(product({o.head_dim, o.kv_heads}));
    // Every decoder stage is measured on top of the joint block's prepared
    // inputs, so each one needs the expert tensors resident -- the projection and
    // attention stages read none of them, but the preparation that produces their
    // inputs runs the sparse feed-forward.  The word stages are the only ones
    // that touch no feed-forward weight at all.
    const bool expert_weights =
        o.experts > 0 && o.stage != "embedding" && o.stage != "head" && o.stage != "dense_ffn";
    auto add = [&](const std::string &role, int64_t cols, int64_t rows, bool gamma = false) {
        Weight w;
        w.cols = cols;
        w.rows = rows;
        w.type = type_of(o.dtypes.at(role));
        if (cols % ggml_blck_size(w.type))
            throw std::runtime_error("quantized reduction dimension must be a multiple of " +
                                     std::to_string(ggml_blck_size(w.type)) + ": " + role);
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
    if (o.attention_bias && o.stage != "embedding" && o.stage != "head") {
        // One row per output channel: ggml_add broadcasts the vector across the
        // token axis, which is what llama.cpp does with the stored bias.
        add("q_bias", e, 1);
        add("k_bias", kv, 1);
        add("v_bias", kv, 1);
    }
    if (o.attention_gate == "sigmoid")
        add("gate_proj", o.hidden, e);
    add("output", e, o.hidden);
    if (o.experts == 0) {
        add("gate", o.hidden, o.intermediate);
        add("up", o.hidden, o.intermediate);
        add("down", o.intermediate, o.hidden);
    } else if (expert_weights) {
        // One expert plane is one dense matrix, so the whole tensor is generated
        // and quantized in blocks of experts: the float staging copy has to stay
        // small beside the packed bytes it is only an intermediate for.
        auto add_exps = [&](const std::string &role, int64_t cols, int64_t rows) {
            Weight w;
            w.cols = cols;
            w.rows = rows;
            w.experts = o.experts;
            w.type = type_of(o.dtypes.at(role));
            if (cols % ggml_blck_size(w.type))
                throw std::runtime_error("quantized reduction dimension must be a multiple of " +
                                         std::to_string(ggml_blck_size(w.type)) + ": " + role);
            const size_t row_bytes = ggml_row_size(w.type, cols);
            w.packed.resize(row_bytes * static_cast<size_t>(rows) *
                            static_cast<size_t>(o.experts));
            const int64_t plane = cols * rows;
            const int64_t block = std::max<int64_t>(1, (int64_t{8} << 20) / std::max<int64_t>(1, plane));
            std::vector<float> f(static_cast<size_t>(plane) * static_cast<size_t>(block));
            for (int64_t start = 0; start < o.experts; start += block) {
                const int64_t n = std::min(block, o.experts - start);
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
                else if (ggml_quantize_chunk(w.type, f.data(), dst, 0, rows * n, cols, nullptr) !=
                         row_bytes * static_cast<size_t>(rows) * static_cast<size_t>(n))
                    throw std::runtime_error("expert tensor quantization size mismatch: " + role);
            }
            d.weights.emplace(role, std::move(w));
        };
        add_exps("gate_exps", o.hidden, o.intermediate);
        add_exps("up_exps", o.hidden, o.intermediate);
        // The down projection reduces over the expert width, so its rows are the
        // hidden dimension -- the stored GGUF layout, not a transposed copy.
        add_exps("down_exps", o.intermediate, o.hidden);
        add("router", o.hidden, o.experts);
        if (o.shared_intermediate > 0) {
            add("gate_shexp", o.hidden, o.shared_intermediate);
            add("up_shexp", o.hidden, o.shared_intermediate);
            add("down_shexp", o.shared_intermediate, o.hidden);
            add("gate_inp_shexp", o.hidden, 1);
        }
    }
    add("input_norm", o.hidden, 1, true);
    add("post_attention_norm", o.hidden, 1, true);
    if (o.vocab > 0) {
        // The two vocabulary-wide matrices are the real boundary of a token step:
        // llama.cpp reads the token embedding once and the LM head once per step.
        // They are generated in row blocks: materializing the whole float matrix
        // would be several gigabytes for an 8B geometry and would not fit beside
        // the packed copy it is only an intermediate for.
        auto add_word = [&](const std::string &role) {
            Weight w;
            w.cols = o.hidden;
            w.rows = o.vocab;
            w.type = type_of(o.dtypes.at(role));
            if ((w.type == GGML_TYPE_Q4_K || w.type == GGML_TYPE_Q6_K) && w.cols % 256)
                throw std::runtime_error("quantized reduction dimension must be divisible by "
                                         "256: " + role);
            const size_t row_bytes = ggml_row_size(w.type, w.cols);
            w.packed.resize(row_bytes * static_cast<size_t>(w.rows));
            const int64_t block = 512;
            std::vector<float> f(static_cast<size_t>(w.cols) * static_cast<size_t>(block));
            for (int64_t start = 0; start < w.rows; start += block) {
                const int64_t n = std::min(block, w.rows - start);
                f.resize(static_cast<size_t>(w.cols) * static_cast<size_t>(n));
                fill(f, rng);
                for (float &v : f)
                    v /= std::sqrt(static_cast<float>(w.cols));
                uint8_t *dst = w.packed.data() + static_cast<size_t>(start) * row_bytes;
                if (w.type == GGML_TYPE_F32)
                    std::memcpy(dst, f.data(), f.size() * sizeof(float));
                else if (w.type == GGML_TYPE_F16)
                    ggml_fp32_to_fp16_row(f.data(), reinterpret_cast<ggml_fp16_t *>(dst),
                                          static_cast<int64_t>(f.size()));
                else if (ggml_quantize_chunk(w.type, f.data(), dst, 0, n, w.cols, nullptr) !=
                         row_bytes * static_cast<size_t>(n))
                    throw std::runtime_error("word matrix quantization size mismatch: " + role);
            }
            d.weights.emplace(role, std::move(w));
        };
        add_word("token_embd");
        add_word("lm_head");
        add("final_norm", o.hidden, 1, true);
        d.ids.resize(static_cast<size_t>(o.q));
        for (int64_t i = 0; i < o.q; ++i)
            d.ids[static_cast<size_t>(i)] = static_cast<int32_t>(rng() % static_cast<uint32_t>(o.vocab));
    }
    if (o.qk_norm != "none") {
        // One gain per head, or one per hidden unit when the normalization runs
        // over the head-concatenated projection (OLMoE).
        const int64_t width = o.qk_norm == "rms_hidden" ? o.hidden : o.head_dim;
        add("q_norm", width, 1, true);
        add("k_norm", width, 1, true);
    }
    d.x.resize(product({o.hidden, o.q}));
    fill(d.x, rng);
    // Host copies in the physical cache layouts: K is [head, position, kv head],
    // V is the transposed [position, head, kv head]. Round them through the cache
    // element type so the reference sees exactly what the cache stores.
    d.kcache.resize(product({o.head_dim, o.history + o.q, o.kv_heads}));
    fill(d.kcache, rng);
    d.vcache.resize(d.kcache.size());
    fill(d.vcache, rng);
    if (kv_cache_type(o) == GGML_TYPE_F16)
        for (auto *cache : {&d.kcache, &d.vcache})
            for (float &value : *cache)
                value = stored(o, value);
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
           cpu_work = 0, logical_internal = 0;
};
static std::vector<std::string> roles_for(const Options &o, const std::string &stage) {
    std::vector<std::string> roles;
    if (stage == "embedding") {
        roles = {"token_embd"};
        return roles;
    }
    if (stage == "head") {
        roles = {"final_norm", "lm_head"};
        return roles;
    }
    if (stage == "projection" || stage == "block") {
        roles = {"input_norm", "q", "k", "v"};
        if (o.attention_bias)
            for (const auto &r : {"q_bias", "k_bias", "v_bias"})
                roles.push_back(r);
        if (o.attention_gate == "sigmoid")
            roles.push_back("gate_proj");
        if (o.qk_norm != "none") {
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
    s.backend = ggml_backend_cpu_init();
    if (!s.backend)
        throw std::runtime_error("CPU backend initialization failed");
    auto tpp = ggml_threadpool_params_default(o.threads);
    if (!o.cpu_ids.empty()) {
        validate_cpu_mask(o);
        tpp.strict_cpu = true;
        for (const int id : o.cpu_ids)
            tpp.cpumask[id] = true;
#if defined(_WIN32)
        {
            // Confine the whole process first.  Per-thread binding only reaches
            // threads that enter a ggml region; with an OpenMP runtime that
            // leaves idle pool threads free to migrate, and it leaves the
            // guarantee dependent on which runtime supplied the workers.
            DWORD_PTR bits = 0;
            for (const int id : o.cpu_ids)
                bits |= DWORD_PTR{1} << id;
            if (!SetProcessAffinityMask(GetCurrentProcess(), bits))
                throw std::runtime_error("cannot apply process CPU affinity: " +
                                         std::to_string(GetLastError()));
            DWORD_PTR observed = 0, system_mask = 0;
            if (!GetProcessAffinityMask(GetCurrentProcess(), &observed, &system_mask) ||
                observed != bits)
                throw std::runtime_error("process CPU affinity readback differs from the request");
        }
        s.threads_before_pool = process_threads();
        if (!GetThreadGroupAffinity(GetCurrentThread(), &s.original_main_affinity))
            throw std::runtime_error("could not query original main thread affinity: " +
                                     std::to_string(GetLastError()));
        s.restore_main_affinity = true;
        if (o.threads == 1) {
            // The local GGML OpenMP single-thread branch skips worker affinity
            // application. Bind the only executing thread explicitly instead.
            GROUP_AFFINITY requested{};
            requested.Group = 0;
            requested.Mask = DWORD_PTR{1} << o.cpu_ids.at(0);
            if (!SetThreadGroupAffinity(GetCurrentThread(), &requested, nullptr))
                throw std::runtime_error("could not bind single main thread to requested CPU: " +
                                         std::to_string(GetLastError()));
            s.explicit_main_binding = true;
        }
#endif
    }
    s.pool = ggml_threadpool_new(&tpp);
    if (!s.pool)
        throw std::runtime_error("threadpool initialization failed");
    ggml_backend_cpu_set_n_threads(s.backend, o.threads);
    ggml_backend_cpu_set_threadpool(s.backend, s.pool);
    s.resident = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
    s.ctx = ggml_init({ggml_tensor_overhead() * 512 +
                           ggml_graph_overhead_custom(512, false) * (o.stage == "paired" ? 4 : 1),
                       nullptr, true});
    if (!s.resident || !s.ctx)
        throw std::runtime_error("tensor metadata allocation failed");
    auto add = [&](const std::string &name, int64_t a, int64_t b, int64_t c, bool output,
                   bool cache = false, ggml_type type = GGML_TYPE_F32) {
        auto *t = type == GGML_TYPE_F32 ? input(s.resident, name.c_str(), a, b, c)
                                        : typed_input(s.resident, name.c_str(), type, a, b, c);
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
               ffn = stage == "dense_ffn" || stage == "moe_ffn" || stage == "block",
               embedding = stage == "embedding", head = stage == "head";
    if (projection || attention || head)
        add("hidden_input", o.hidden, o.q, 1, false);
    if (embedding) {
        auto *ids = ggml_new_tensor_1d(s.resident, GGML_TYPE_I32, o.q);
        ggml_set_name(ids, "token_ids");
        ggml_set_input(ids);
        p->tensors["token_ids"] = ids;
        p->input_bytes = add_bytes(p->input_bytes, ggml_nbytes(ids));
    }
    if (head)
        add("logits", o.vocab, o.q, 1, true);
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
    ggml_tensor *key_offsets = nullptr, *value_offsets = nullptr;
    if (attention) {
        // K keeps the row-major [head, position, kv head] layout mul_mat wants.
        // V is stored transposed as [position, head, kv head]: llama.cpp reads V
        // through this layout so P*V consumes the cache directly, while the plain
        // layout has to materialize a full transpose of the whole cache per step.
        add("key_cache", o.head_dim, length, o.kv_heads, false, true, kv_cache_type(o));
        add("value_cache", length, o.head_dim, o.kv_heads, false, true, kv_cache_type(o));
        // The scatter offsets are constants of the block, but they have to live
        // in the resident buffer, not in the graph: the graph allocator frees a
        // leaf's slot as soon as its last reader has run, so a constant written
        // once is overwritten by the next sample's intermediates (observed as
        // index garbage on the third compute). They are not ports, so they stay
        // out of the port accounting; the resident buffer size below includes them.
        key_offsets = ggml_new_tensor_1d(s.resident, GGML_TYPE_I64, o.q);
        ggml_set_name(key_offsets, "key_offsets");
        value_offsets = ggml_new_tensor_1d(
            s.resident, GGML_TYPE_I64, static_cast<int64_t>(o.q) * o.head_dim * o.kv_heads);
        ggml_set_name(value_offsets, "value_offsets");
    }
    if (attention || ffn)
        add("attention_hidden", o.hidden, o.q, 1, attention);
    // The embedding stage's product is the hidden activation the first block
    // consumes, so it is a materialized output exactly like the FFN's.
    if (ffn || embedding)
        add("hidden_output", o.hidden, o.q, 1, true);
    for (const auto &role : roles_for(o, stage)) {
        const auto &w = d.weights.at(role);
        auto *t = w.experts > 1
                      ? ggml_new_tensor_3d(s.resident, w.type, w.cols, w.rows, w.experts)
                      : ggml_new_tensor_2d(s.resident, w.type, w.cols, w.rows);
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
            if (t->type == GGML_TYPE_F32)
                ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
            else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(f.size());
                ggml_fp32_to_fp16_row(f.data(), half.data(),
                                      static_cast<int64_t>(f.size()));
                ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
            } else
                throw std::runtime_error("unsupported stage input dtype: " + name);
        }
    };
    put("hidden_input", d.x);
    put("key_cache", d.kcache);
    put("value_cache", d.vcache);
    if (embedding)
        ggml_backend_tensor_set(p->tensors.at("token_ids"), d.ids.data(), 0,
                                static_cast<size_t>(o.q) * sizeof(int32_t));
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
    if (embedding) {
        materialize("hidden_output", ggml_get_rows(ctx, p->weights.at("token_embd"),
                                                   p->tensors.at("token_ids")));
    } else if (head) {
        // Final RMSNorm and the LM head are the last two operations of a token
        // step; together with the embedding they are what a decoder-block-only
        // prediction leaves out.
        auto *n = norm(p->tensors.at("hidden_input"), "final_norm");
        materialize("logits", ggml_mul_mat(ctx, p->weights.at("lm_head"), n));
    }
    ggml_tensor *query = nullptr, *key = nullptr, *value = nullptr, *agate = nullptr,
                *hidden = nullptr;
    if (projection) {
        auto *n = norm(p->tensors.at("hidden_input"), "input_norm");
        auto qkv = [&](const std::string &role, int64_t heads, bool rotary) {
            auto *projected = ggml_mul_mat(ctx, p->weights.at(role), n);
            // llama.cpp adds the stored bias to the projection, then applies
            // QK norm and RoPE; the order is part of the declared graph.
            if (o.attention_bias)
                projected = ggml_add(ctx, projected, p->weights.at(role + "_bias"));
            // OLMoE normalizes the flat projection output (one gain per hidden
            // unit) before the head split; the per-head variant normalizes
            // after it.  The order relative to RoPE is the declared graph.
            if (rotary && o.qk_norm == "rms_hidden")
                projected = norm(projected, role == "q" ? "q_norm" : "k_norm");
            auto *v = ggml_reshape_3d(ctx, projected, o.head_dim, heads, o.q);
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
        // SET_ROWS is the only scatter that writes a cache of any element type
        // *and* returns the complete cache, so readers depend on the write even
        // in the joint graph. SET-inplace accepts an f32 destination only.
        // K: one row per (kv head, new position); the head plane is dim 2.
        auto *kr = ggml_set_rows(ctx, kc, key, key_offsets);
        // V: the transposed cache interleaves heads along the slowest axis, so
        // the append is one element per (head, position, channel) triple.
        auto *vflat = ggml_reshape_2d(ctx, vc, 1, static_cast<int64_t>(ggml_nelements(vc)));
        auto *newflat = ggml_reshape_2d(ctx, value, 1,
                                        static_cast<int64_t>(ggml_nelements(value)));
        auto *vr = ggml_reshape_3d(ctx, ggml_set_rows(ctx, vflat, newflat, value_offsets),
                                   length, o.head_dim, o.kv_heads);
        auto *scores = ggml_mul_mat(ctx, kr, query);
        scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(o.head_dim)));
        scores = ggml_diag_mask_inf(ctx, scores, static_cast<int>(o.history));
        auto *probs = ggml_soft_max(ctx, scores);
        auto *attended = ggml_mul_mat(ctx, vr, probs);
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
        // The sparse block in the sequence llama.cpp builds it: router scores,
        // top-k, three indirect matrix products, the weighted reduction over the
        // selected experts, then the shared expert (if the checkpoint has one)
        // and the residual.
        auto *n = norm(hidden, "post_attention_norm");
        auto *logits = ggml_mul_mat(ctx, p->weights.at("router"), n); // [experts, tokens]
        auto *probs = ggml_soft_max(ctx, logits);                     // over all experts
        auto *selected = ggml_argsort_top_k(ctx, probs, static_cast<int>(o.top_k));
        auto *weights = ggml_get_rows(
            ctx, ggml_reshape_3d(ctx, probs, 1, o.experts, o.q), selected); // [1, k, tokens]
        if (o.normalize_topk) {
            // Renormalizing the selected weights is what the declared variant
            // does; the clamp is llama.cpp's own guard against a zero sum.
            auto *w2 = ggml_reshape_2d(ctx, weights, o.top_k, o.q);
            auto *sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY);
            weights = ggml_reshape_3d(ctx, ggml_div(ctx, w2, sum), 1, o.top_k, o.q);
        }
        auto *cur = ggml_reshape_3d(ctx, n, o.hidden, 1, o.q);
        auto *up = ggml_mul_mat_id(ctx, p->weights.at("up_exps"), cur, selected);
        auto *gate = ggml_mul_mat_id(ctx, p->weights.at("gate_exps"), cur, selected);
        auto *active = ggml_swiglu_split(ctx, gate, up);
        auto *experts = ggml_mul_mat_id(ctx, p->weights.at("down_exps"), active, selected);
        experts = ggml_mul(ctx, experts, weights); // [hidden, k, tokens]
        auto *moe = ggml_view_2d(ctx, experts, o.hidden, o.q, experts->nb[2], 0);
        for (int64_t i = 1; i < o.top_k; ++i)
            moe = ggml_add(ctx, moe,
                           ggml_view_2d(ctx, experts, o.hidden, o.q, experts->nb[2],
                                        i * experts->nb[1]));
        if (o.shared_intermediate > 0) {
            auto *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, p->weights.at("gate_inp_shexp"), n));
            auto *shared = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, p->weights.at("gate_shexp"), n),
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
            throw std::runtime_error("unsupported requested CPU operator");
    s.allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!s.allocator || !ggml_gallocr_alloc_graph(s.allocator, p->graph))
        throw std::runtime_error("graph allocation failed");
    // Write the scatter offsets once, into the resident buffer. This has to
    // happen after the resident tensors are allocated, before the first compute.
    // The readback below is the guard: an offset that never reached the buffer
    // produces a corrupt cache, not a crash.
    if (key_offsets) {
        std::vector<int64_t> ko(static_cast<size_t>(o.q)),
            vo(static_cast<size_t>(o.q) * o.head_dim * o.kv_heads);
        for (int64_t t = 0; t < o.q; ++t)
            ko[static_cast<size_t>(t)] = o.history + t;
        for (int64_t h = 0; h < o.kv_heads; ++h)
            for (int64_t t = 0; t < o.q; ++t)
                for (int64_t j = 0; j < o.head_dim; ++j) {
                    const size_t src = static_cast<size_t>(j + o.head_dim * (t + o.q * h));
                    vo[src] = static_cast<int64_t>(o.history + t) +
                              length * (j + o.head_dim * h);
                }
        ggml_backend_tensor_set(key_offsets, ko.data(), 0,
                                static_cast<size_t>(o.q) * sizeof(int64_t));
        ggml_backend_tensor_set(value_offsets, vo.data(), 0,
                                vo.size() * sizeof(int64_t));
        {   // the scatter is only correct if the offsets really reached the buffer
            std::vector<int64_t> back(vo.size());
            ggml_backend_tensor_get(value_offsets, back.data(), 0,
                                    vo.size() * sizeof(int64_t));
            std::vector<int64_t> backk(static_cast<size_t>(o.q));
            ggml_backend_tensor_get(key_offsets, backk.data(), 0,
                                    static_cast<size_t>(o.q) * sizeof(int64_t));
            if (back != vo || backk != ko || !key_offsets->buffer || !value_offsets->buffer)
                throw std::runtime_error("scatter offsets did not reach the buffer");
        }
    }
    p->internal_bytes = ggml_gallocr_get_buffer_size(s.allocator, 0);
    p->cpu_work = ggml_graph_plan(p->graph, o.threads, s.pool).work_size;
    for (int i = 0; i < ggml_graph_n_nodes(p->graph); ++i) {
        const auto *n = ggml_graph_node(p->graph, i);
        if (!n->view_src)
            p->logical_internal = add_bytes(p->logical_internal, ggml_nbytes(n));
    }
    return p;
}
static std::vector<float> read_tensor(ggml_tensor *t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32)
        ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(static_cast<size_t>(n));
        ggml_backend_tensor_get(t, half.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(half.data(), f.data(), n);
    } else
        throw std::runtime_error("cannot read back a tensor of this element type");
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
    // k and v are the host copies of the two caches in their physical layouts:
    // K as [head, position, kv head], V transposed as [position, head, kv head].
    auto k = d.kcache, v = d.vcache;
    for (int64_t h = 0; h < o.kv_heads; ++h)
        for (int64_t t = 0; t < o.q; ++t)
            for (int64_t j = 0; j < o.head_dim; ++j) {
                const size_t dst = static_cast<size_t>(j + o.head_dim * (o.history + t + len * h)),
                             vdst = static_cast<size_t>(o.history + t) +
                                    static_cast<size_t>(len) *
                                        static_cast<size_t>(j + o.head_dim * h),
                             src = static_cast<size_t>(j + o.head_dim * (t + o.q * h));
                k[dst] = stored(o, key[src]);
                v[vdst] = stored(o, value[src]);
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
                         v[static_cast<size_t>(s) +
                           static_cast<size_t>(len) *
                               static_cast<size_t>(j + o.head_dim * kh)];
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
    auto n = normalized(h, o.hidden, d.weights.at("post_attention_norm"), o.norm_eps);
    std::map<size_t, double> result;
    std::vector<float> row;
    std::set<size_t> tokens;
    for (const size_t ix : indices)
        tokens.insert(ix / static_cast<size_t>(o.hidden));
    for (const size_t t : tokens) {
        const float *x = n.data() + t * static_cast<size_t>(o.hidden);
        // Router scores and the softmax the graph computes over all experts.
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
                d.weights.at("gate_exps").row(static_cast<int64_t>(expert) * o.intermediate + j, row);
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
            out[static_cast<size_t>(k)] += h[t * static_cast<size_t>(o.hidden) +
                                             static_cast<size_t>(k)];
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
    if (o.vocab > 0)
        std::cout << ",\"vocab\":" << o.vocab << ",\"token_embd_dtype\":\""
                  << o.dtypes.at("token_embd") << "\",\"lm_head_dtype\":\""
                  << o.dtypes.at("lm_head") << "\"";
    if (o.experts > 0)
        std::cout << ",\"experts\":" << o.experts << ",\"top_k\":" << o.top_k
                  << ",\"shared_intermediate\":" << o.shared_intermediate
                  << ",\"normalize_topk\":" << (o.normalize_topk ? "true" : "false")
                  << ",\"attention_bias\":" << (o.attention_bias ? "true" : "false");
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
            for (const auto &s : {"shared_expert_gate_inp", "shared_sigmoid",
                                  "shared_expert_gate_projection", "shared_expert_up_projection",
                                  "SiLU_gate_shared", "gate_times_up_shared",
                                  "shared_expert_down_projection", "shared_gate_scale"})
                v.push_back(s);
        v.push_back("ffn_residual_add");
        v.push_back("materialize_hidden");
        return v;
    }
    if (kind == "embedding")
        return {"token_embedding_lookup", "materialize_hidden"};
    if (kind == "head")
        return {"final_RMSNorm", "final_norm_scale", "lm_head_projection", "materialize_logits"};
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
                                                      {"value_cache", "value_cache"},
                                                      {"token_ids", "token_ids"},
                                                      {"logits", "logits"}};
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
struct PathSnapshot {
    std::map<std::string, std::string> ports, inputs, caches;
    std::map<std::string, std::vector<uint8_t>> port_bytes, input_bytes, cache_bytes;
    bool finite = true, append_exact = true, history_unchanged = true;
};
struct ObservedThread {
    uint64_t thread_id, mask;
    unsigned group;
    int logical_id;
};
struct CpuAffinity {
    bool requested = false, verified = false;
    bool explicit_main_binding = false;
    std::vector<int> applied_ids;
    std::vector<ObservedThread> threads;
};
static CpuAffinity verify_cpu_affinity(const Options &o, const Session &s) {
    CpuAffinity result;
    if (o.cpu_ids.empty())
        return result;
    result.requested = true;
#if defined(_WIN32)
    result.explicit_main_binding = s.explicit_main_binding;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    std::string reason;
    do {
        // Every thread of this process must be confined to exactly one of the
        // requested logical CPUs, and every requested CPU must be covered.
        //
        // Counting "newly created worker threads" only works for a persistent
        // ggml threadpool.  When GGML is built with OpenMP there are no
        // persistent workers: each parallel region borrows the OpenMP runtime's
        // threads and applies the same cpumask inside the region.  Confinement
        // is the property that makes the price comparable, so it is what gets
        // verified here, whichever runtime supplies the threads.
        result.threads.clear();
        std::set<int> single;
        uint64_t requested = 0;
        for (const int id : o.cpu_ids)
            requested |= uint64_t{1} << id;
        uint64_t covered = 0;
        bool valid = true;
        reason.clear();
        for (const DWORD id : process_threads()) {
            HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, id);
            if (!thread) {
                valid = false;
                reason = "OpenThread query failed: " + std::to_string(GetLastError());
                continue;
            }
            GROUP_AFFINITY affinity{};
            const bool queried = GetThreadGroupAffinity(thread, &affinity) != 0;
            const DWORD error = GetLastError();
            CloseHandle(thread);
            if (!queried) {
                valid = false;
                reason = "GetThreadGroupAffinity failed: " + std::to_string(error);
                continue;
            }
            const uint64_t mask = static_cast<uint64_t>(affinity.Mask);
            if (affinity.Group != 0 || (mask & ~requested) != 0 || mask == 0) {
                valid = false;
                reason = "thread " + std::to_string(id) + " mask " + mask_hex(mask) +
                         " escapes the requested logical CPUs";
                continue;
            }
            covered |= mask;
            const auto ids = mask_ids(mask);
            if (ids.size() == 1)
                single.insert(ids[0]);
            result.threads.push_back({id, mask, affinity.Group, ids.empty() ? -1 : ids[0]});
        }
        // Every requested CPU must actually be used by a single-CPU thread; an
        // idle runtime thread may still carry the whole process mask, which is
        // confined but is not one of the computing threads.
        result.applied_ids.assign(single.begin(), single.end());
        if (valid && covered == requested && result.applied_ids == o.cpu_ids) {
            result.verified = true;
            return result;
        }
        Sleep(1); // outside setup/warmup/timed measurements, only waits for startup
    } while (std::chrono::steady_clock::now() < deadline);
    {
        std::string seen;
        for (const auto &t : result.threads)
            seen += (seen.empty() ? "" : ",") + std::to_string(t.thread_id) + ":" + mask_hex(t.mask);
        reason += " [confined " + std::to_string(result.threads.size()) + ": " + seen + "]";
    }
    throw std::runtime_error("could not verify requested CPU affinity: " + reason);
#else
    (void)s;
    throw std::runtime_error("CPU affinity verification is unsupported on this platform");
#endif
}
static void emit_cpu_affinity(const Options &o, const CpuAffinity &affinity) {
    std::cout << "{\"requested_logical_ids\":";
    if (affinity.requested)
        emit_ids(o.cpu_ids);
    else
        std::cout << "null";
    std::cout << ",\"applied_logical_ids\":";
    if (affinity.requested)
        emit_ids(affinity.applied_ids);
    else
        std::cout << "null";
    std::cout << ",\"strict_cpu\":" << (affinity.requested ? "true" : "false") << ",\"verified\":"
              << (affinity.requested ? (affinity.verified ? "true" : "false") : "null")
              << ",\"verified_before_and_after_timing\":" << (affinity.requested ? "true" : "null")
              << ",\"observed_threads\":[";
    for (size_t i = 0; i < affinity.threads.size(); ++i) {
        const auto &thread = affinity.threads[i];
        std::cout << (i ? "," : "") << "{\"thread_id\":" << thread.thread_id
                  << ",\"group\":" << thread.group << ",\"mask_hex\":\"" << mask_hex(thread.mask)
                  << "\",\"logical_id\":" << thread.logical_id << "}";
    }
    std::cout << "],\"binding_source\":\""
              << (affinity.requested ? (affinity.explicit_main_binding
                                            ? "Windows_SetThreadGroupAffinity_for_single_thread"
                                            : "GGML_threadpool_strict_cpu_mask")
                                     : "not_requested")
              << "\",\"verification_source\":\""
              << (affinity.requested ? (affinity.explicit_main_binding
                                            ? "Windows_explicit_main_thread_binding_"
                                              "GetThreadGroupAffinity_after_setup_and_samples"
                                            : "Windows_new_process_threads_plus_main_"
                                              "GetThreadGroupAffinity_after_setup_and_samples")
                                     : "not_requested;actual_thread_placement_unknown")
              << "\"}";
}
static PathSnapshot path_snapshot(const Options &o, const Data &d, Stage &stage) {
    PathSnapshot result;
    for (const auto &item : stage.ports) {
        result.ports[port_role(item.first)] = tensor_digest(item.second);
        auto &bytes = result.port_bytes[port_role(item.first)];
        bytes.resize(ggml_nbytes(item.second));
        ggml_backend_tensor_get(item.second, bytes.data(), 0, bytes.size());
        for (float v : read_tensor(item.second))
            result.finite = result.finite && std::isfinite(v);
    }
    for (const auto &item : stage.tensors)
        if (!stage.ports.count(item.first) && item.first != "key_cache" &&
            item.first != "value_cache") {
            result.inputs[port_role(item.first)] = tensor_digest(item.second);
            auto &bytes = result.input_bytes[port_role(item.first)];
            bytes.resize(ggml_nbytes(item.second));
            ggml_backend_tensor_get(item.second, bytes.data(), 0, bytes.size());
        }
    const auto key = read_tensor(stage.tensors.at("key_cache"));
    const auto value = read_tensor(stage.tensors.at("value_cache"));
    const auto new_key = read_tensor(stage.ports.at("new_key"));
    const auto new_value = read_tensor(stage.ports.at("new_value"));
    result.caches["key_cache"] = digest(key);
    result.caches["value_cache"] = digest(value);
    for (const auto &name : {"key_cache", "value_cache"}) {
        auto *tensor = stage.tensors.at(name);
        auto &bytes = result.cache_bytes[name];
        bytes.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    }
    const int64_t length = o.history + o.q;
    for (int64_t h = 0; h < o.kv_heads; ++h)
        for (int64_t t = 0; t < length; ++t)
            for (int64_t j = 0; j < o.head_dim; ++j) {
                const size_t dst = static_cast<size_t>(j + o.head_dim * (t + length * h)),
                             vdst = static_cast<size_t>(t) +
                                    static_cast<size_t>(length) *
                                        static_cast<size_t>(j + o.head_dim * h);
                if (t < o.history)
                    result.history_unchanged = result.history_unchanged &&
                                               key[dst] == d.kcache[dst] &&
                                               value[vdst] == d.vcache[vdst];
                else {
                    const size_t src =
                        static_cast<size_t>(j + o.head_dim * (t - o.history + o.q * h));
                    result.append_exact =
                        result.append_exact && key[dst] == stored(o, new_key[src]) &&
                        value[vdst] == stored(o, new_value[src]);
                }
            }
    return result;
}
struct PairedResult {
    std::map<std::string, std::vector<int64_t>> stage_wall;
    std::vector<int64_t> stage_sum, serial_wall, joint_wall;
    std::vector<std::string> order;
    std::map<std::string, size_t> work_sizes;
    size_t backend_capacity = 0, cached_work_total = 0;
    int64_t warmup_rounds = 0, warmup_elapsed_ns = 0;
    CpuAffinity affinity;
    PathSnapshot serial, joint;
    bool inputs_unchanged = false;
};
struct CachedPlans {
    ggml_backend_t backend;
    std::map<std::string, ggml_backend_graph_plan_t> plans;
    explicit CachedPlans(ggml_backend_t b) : backend(b) {
    }
    ~CachedPlans() {
        for (const auto &p : plans)
            ggml_backend_graph_plan_free(backend, p.second);
    }
};
static PairedResult measure_paired(const Options &o, const Data &d, Stage &stage) {
    PairedResult result;
    std::map<std::string, std::vector<uint8_t>> original_inputs;
    for (const auto &item : stage.tensors)
        if (!stage.ports.count(item.first) && item.first != "key_cache" &&
            item.first != "value_cache") {
            auto &bytes = original_inputs[port_role(item.first)];
            bytes.resize(ggml_nbytes(item.second));
            ggml_backend_tensor_get(item.second, bytes.data(), 0, bytes.size());
        }
    std::map<std::string, ggml_cgraph *> graphs;
    int next = 0;
    for (const auto &name : {"projection", "attention", "dense_ffn"}) {
        const auto range = stage.node_ranges.at(name);
        if (range.first != next || range.second <= range.first)
            throw std::runtime_error("invalid contiguous block stage partition");
        auto *graph = ggml_new_graph_custom(stage.s.ctx, 512, false);
        for (int n = range.first; n < range.second; ++n)
            ggml_graph_add_node(graph, ggml_graph_node(stage.graph, n));
        graphs[name] = graph;
        next = range.second;
        result.stage_wall[name].resize(static_cast<size_t>(o.samples));
    }
    if (next != ggml_graph_n_nodes(stage.graph))
        throw std::runtime_error("stage partitions do not cover complete block graph");
    graphs["joint"] = stage.graph;
    CachedPlans cached(stage.s.backend);
    for (const auto &item : graphs) {
        const size_t work = ggml_graph_plan(item.second, o.threads, stage.s.pool).work_size;
        result.work_sizes[item.first] = work;
        result.backend_capacity = std::max(result.backend_capacity, work);
        if (o.compute_mode == "plan") {
            auto *plan = ggml_backend_graph_plan_create(stage.s.backend, item.second);
            if (!plan)
                throw std::runtime_error("cached CPU graph plan allocation failed");
            cached.plans[item.first] = plan;
            result.cached_work_total = add_bytes(result.cached_work_total, work);
        }
    }
    if (o.compute_mode == "plan")
        result.backend_capacity = 0; // graph_compute was never called on this backend
    auto execute = [&](const std::string &name) {
        if (o.compute_mode == "graph")
            compute(stage.s, graphs.at(name));
        else {
            const auto status =
                ggml_backend_graph_plan_compute(stage.s.backend, cached.plans.at(name));
            if (status != GGML_STATUS_SUCCESS)
                throw std::runtime_error("cached graph plan compute failed: " +
                                         std::to_string(status));
        }
    };
    auto serial = [&]() {
        for (const auto &name : {"projection", "attention", "dense_ffn"})
            execute(name);
    };
    // Each path receives one setup pass. These and warmup are outside timing.
    serial();
    execute("joint");
    result.affinity = verify_cpu_affinity(o, stage.s);
    const auto warmup_begin = std::chrono::steady_clock::now();
    do {
        result.warmup_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - warmup_begin)
                                       .count();
        if (result.warmup_rounds >= o.warmup &&
            result.warmup_elapsed_ns >= o.min_warmup_ms * 1000000)
            break;
        if (result.warmup_rounds % 2 == 0) {
            serial();
            execute("joint");
        } else {
            execute("joint");
            serial();
        }
        ++result.warmup_rounds;
    } while (true);
    result.stage_sum.resize(static_cast<size_t>(o.samples));
    result.serial_wall.resize(static_cast<size_t>(o.samples));
    result.joint_wall.resize(static_cast<size_t>(o.samples));
    auto duration = [](const std::chrono::steady_clock::time_point &start) {
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
        if (ns <= 0)
            throw std::runtime_error("nonpositive paired sample");
        return ns;
    };
    for (int i = 0; i < o.samples; ++i) {
        const size_t ix = static_cast<size_t>(i);
        auto timed_serial = [&]() {
            const auto outer = std::chrono::steady_clock::now();
            int64_t sum = 0;
            for (const auto &name : {"projection", "attention", "dense_ffn"}) {
                const auto begin = std::chrono::steady_clock::now();
                execute(name);
                const int64_t ns = duration(begin);
                result.stage_wall.at(name)[ix] = ns;
                if (ns > std::numeric_limits<int64_t>::max() - sum)
                    throw std::runtime_error("stage timing sum overflow");
                sum += ns;
            }
            result.serial_wall[ix] = duration(outer);
            result.stage_sum[ix] = sum;
        };
        auto timed_joint = [&]() {
            const auto begin = std::chrono::steady_clock::now();
            execute("joint");
            result.joint_wall[ix] = duration(begin);
        };
        if (i % 2 == 0) {
            result.order.push_back("AB");
            timed_serial();
            timed_joint();
        } else {
            result.order.push_back("BA");
            timed_joint();
            timed_serial();
        }
    }
    result.affinity = verify_cpu_affinity(o, stage.s);
    // Reading ports is deliberately outside timed pairs; it would otherwise
    // perturb the cache between the compared paths.
    serial();
    result.serial = path_snapshot(o, d, stage);
    execute("joint");
    result.joint = path_snapshot(o, d, stage);
    result.inputs_unchanged =
        result.serial.input_bytes == original_inputs && result.joint.input_bytes == original_inputs;
    return result;
}
static void emit_digest_map(const std::map<std::string, std::string> &values) {
    std::cout << "{";
    bool first = true;
    for (const auto &item : values) {
        std::cout << (first ? "" : ",") << "\"" << item.first << "\":\"" << item.second << "\"";
        first = false;
    }
    std::cout << "}";
}
static void emit_path_snapshot(const PathSnapshot &s) {
    std::cout << "{\"stage_port_digests\":";
    emit_digest_map(s.ports);
    std::cout << ",\"input_port_digests\":";
    emit_digest_map(s.inputs);
    std::cout << ",\"kv_cache_digests\":";
    emit_digest_map(s.caches);
    std::cout << ",\"all_outputs_finite\":" << (s.finite ? "true" : "false")
              << ",\"kv_append_check\":{\"passed\":"
              << ((s.append_exact && s.history_unchanged) ? "true" : "false")
              << ",\"append_exact\":" << (s.append_exact ? "true" : "false")
              << ",\"history_unchanged\":" << (s.history_unchanged ? "true" : "false") << "}}";
}
static void emit_ns(const std::vector<int64_t> &values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i)
        std::cout << (i ? "," : "") << values[i];
    std::cout << "]";
}
// Cumulative cycles burned by every thread of this process. Read outside the
// wall-clock bracket on both sides, so the cycle bracket is slightly wider; the
// extra is two cheap query calls. A failed read is reported as -1 rather than
// silently dropping the sample.
static bool process_cycles(uint64_t &value) {
#ifdef _WIN32
    ULONG64 cycles = 0;
    if (!QueryProcessCycleTime(GetCurrentProcess(), &cycles))
        return false;
    value = static_cast<uint64_t>(cycles);
    return true;
#else
    (void)value;
    return false;
#endif
}
static int run(const Options &o) {
    Data d = make_data(o);
    const bool paired = o.stage == "paired";
    const std::string stage_kind = paired ? "block" : o.stage;
    const bool decoder_stage = stage_kind == "projection" || stage_kind == "attention" ||
                               stage_kind == "dense_ffn" || stage_kind == "moe_ffn";
    if (decoder_stage) {
        auto prep = build_stage(o, d, "block");
        compute(prep->s, prep->graph);
        capture(*prep, d);
    }
    auto stage = build_stage(o, d, stage_kind);
    auto &s = stage->s;
    std::vector<int64_t> wall;
    std::vector<int64_t> cycles;
    std::unique_ptr<PairedResult> pair;
    // Declared before either branch so the single-stage record can report what
    // the workers were actually bound to; an unrequested mask reports nulls.
    CpuAffinity affinity;
    if (paired) {
        pair = std::make_unique<PairedResult>(measure_paired(o, d, *stage));
        capture(*stage, d);
    } else {
        compute(s, stage->graph); // primes reusable cplan, separate from upstream prep
        affinity = verify_cpu_affinity(o, s);
        if (stage_kind == "block")
            capture(*stage, d);
        const CacheFlush flush(o.evict_bytes);
        for (int i = 0; i < o.warmup; ++i) {
            flush();
            compute(s, stage->graph);
        }
        for (int i = 0; i < o.samples; ++i) {
            flush();
            uint64_t cycles_before = 0, cycles_after = 0;
            const bool cycle_bracket = o.thread_cycles && process_cycles(cycles_before);
            const auto begin = std::chrono::steady_clock::now();
            compute(s, stage->graph);
            const auto end = std::chrono::steady_clock::now();
            const auto ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            if (ns <= 0)
                throw std::runtime_error("nonpositive sample");
            wall.push_back(ns);
            if (o.thread_cycles) {
                const bool closed = cycle_bracket && process_cycles(cycles_after) &&
                                    cycles_after >= cycles_before;
                cycles.push_back(closed ? static_cast<int64_t>(cycles_after - cycles_before) : -1);
            }
        }
    }
    std::map<std::string, std::vector<float>> outputs;
    for (const auto &p : stage->ports) {
        outputs[p.first] = read_tensor(p.second);
        for (float v : outputs[p.first])
            if (!std::isfinite(v))
                throw std::runtime_error("nonfinite output: " + p.first);
    }
    const bool p = stage_kind == "projection" || stage_kind == "block",
               a = stage_kind == "attention" || stage_kind == "block",
               f = stage_kind == "dense_ffn" || stage_kind == "moe_ffn" || stage_kind == "block";
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
        const auto reference =
            o.experts == 0
                ? ffn_reference(o, d, stage_kind == "block" ? rh : d.attn_hidden, ix)
                : moe_reference(o, d, stage_kind == "block" ? rh : d.attn_hidden, ix);
        for (size_t i : ix)
            c.add(actual[i], reference.at(i));
    }
    if (stage_kind == "embedding") {
        // The lookup is exact: every checked element is one stored weight row.
        const auto &actual = outputs.at("hidden_output");
        auto &c = checks["hidden_output"];
        c.elements = actual.size();
        std::vector<float> row;
        for (const size_t ix : checked_indices(actual.size(), o.hidden, o.q)) {
            const int64_t r = static_cast<int64_t>(ix) % o.hidden,
                          t = static_cast<int64_t>(ix) / o.hidden;
            d.weights.at("token_embd").row(d.ids.at(static_cast<size_t>(t)), row);
            c.add(actual[ix], row.at(static_cast<size_t>(r)));
        }
    }
    if (stage_kind == "head") {
        // Only sampled rows are materialized into the reference: the full logits
        // reference would repeat the vocabulary-wide product the stage already
        // performed, in double precision, for every token.
        const auto &actual = outputs.at("logits");
        auto &c = checks["logits"];
        c.elements = actual.size();
        const auto final = normalized(d.x, o.hidden, d.weights.at("final_norm"), o.norm_eps);
        std::vector<float> row;
        for (const size_t ix : checked_indices(actual.size(), o.vocab, o.q)) {
            const int64_t r = static_cast<int64_t>(ix) % o.vocab,
                          t = static_cast<int64_t>(ix) / o.vocab;
            d.weights.at("lm_head").row(r, row);
            c.add(actual[ix],
                  static_cast<float>(dot(row, &final.at(static_cast<size_t>(t) * o.hidden))));
        }
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
                    const size_t dst = static_cast<size_t>(j + o.head_dim * (t + len * h)),
                                 vdst = static_cast<size_t>(t) +
                                        static_cast<size_t>(len) *
                                            static_cast<size_t>(j + o.head_dim * h);
                    if (t < o.history)
                        history_unchanged = history_unchanged && actualk[dst] == d.kcache[dst] &&
                                            actualv[vdst] == d.vcache[vdst];
                    else {
                        const size_t src =
                            static_cast<size_t>(j + o.head_dim * (t - o.history + o.q * h));
                        append_exact = append_exact && actualk[dst] == stored(o, kn[src]) &&
                                       actualv[vdst] == stored(o, vn[src]);
                    }
                }
    }
    const bool quant = std::any_of(o.dtypes.begin(), o.dtypes.end(), [](const auto &item) {
        return item.second == "q4_k" || item.second == "q6_k";
    });
    const double atol = quant ? 0.003 : 0.0002, rtol = quant ? 0.035 : 0.002;
    Check total;
    bool passed = append_exact && history_unchanged;
    if (paired)
        passed = passed && pair->inputs_unchanged && pair->serial.finite && pair->joint.finite &&
                 pair->serial.append_exact && pair->serial.history_unchanged &&
                 pair->joint.append_exact && pair->joint.history_unchanged &&
                 pair->serial.port_bytes == pair->joint.port_bytes &&
                 pair->serial.cache_bytes == pair->joint.cache_bytes;
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
    const size_t workspace =
                     add_bytes(stage->internal_bytes,
                               paired ? add_bytes(pair->backend_capacity, pair->cached_work_total)
                                      : stage->cpu_work),
                 resident = add_bytes(add_bytes(stage->input_bytes, stage->output_bytes),
                                      add_bytes(stage->weight_bytes, stage->kv_bytes));
    std::cout << std::setprecision(17) << "{\"schema\":\""
              << (paired ? "ggml_dense_layer_paired_probe/1" : "ggml_dense_layer_probe/1")
              << "\",\"contract\":\""
              << (o.experts > 0
                      ? "ggml_cpu_moe_decoder_block_v1"
                      : kv_cache_type(o) == GGML_TYPE_F16 ? "ggml_cpu_dense_decoder_block_v2"
                                                          : "ggml_cpu_dense_decoder_block_v1")
              << "\",\"kind\":\"dense_layer\","
                 "\"stage\":\""
              << o.stage << "\",\"backend\":\"CPU\",\"batch\":1,\"config\":";
    emit_config(o);
    std::cout << ",\"q\":" << o.q << ",\"history\":" << o.history << ",\"threads\":" << o.threads
              << ",\"threads_requested\":" << o.threads << ",\"threads_applied\":" << o.threads
              << ",\"thread_management\":\"persistent_ggml_threadpool\",\"cpu_affinity\":";
    if (paired)
        emit_cpu_affinity(o, pair->affinity);
    else
        emit_cpu_affinity(o, affinity);
    std::cout << ",\"warmup\":" << o.warmup
              << ",\"setup_compute_repetitions\":1,\"upstream_setup_compute_repetitions\":"
              << (decoder_stage ? 1 : 0) << ",\"samples\":" << o.samples
              << ",\"seed\":" << o.seed
              << ",\"evict_bytes\":" << o.evict_bytes
              << ",\"residency_regime\":\""
              << (o.evict_bytes > 0 ? "streamed_cold_cache_before_each_sample"
                                    : "resident_warm_cache_shared_across_samples")
              << "\","
                 "\"input_generator\":\"mt19937_uniform24_scaled_weights_gamma_near_one_v1\","
                 "\"input_lineage\":\""
              << (paired ? "same_seed_shared_resident_graph_partitions"
                         : "same_seed_complete_upstream_preparation_outside_timing")
              << "\","
                 "\"activation_dtype\":\"f32\",\"kv_dtype\":\""
              << o.kv_dtype << "\",\"position_dtype\":\"i32\","
                 "\"implementation_variant\":\""
              << (o.experts > 0 ? "ggml_cpu_moe_materialized_" : "ggml_cpu_dense_materialized_")
              << o.stage << (paired ? "_" + o.compute_mode : "")
              << (o.experts > 0 ? "_v1" : kv_cache_type(o) == GGML_TYPE_F16 ? "_v2" : "_v1")
              << "\",\"timing\":\""
              << (paired ? "steady_clock_paired_same_resident_graph_partitions_including_sync"
                         : o.evict_bytes > 0
                               ? "steady_clock_graph_compute_including_sync_after_declared_cache_eviction"
                               : "steady_clock_graph_compute_including_sync")
              << "\",\"scope\":\""
              << (stage_kind == "embedding"
                      ? "resident_repeated_declared_token_embedding"
                      : stage_kind == "head"
                            ? "resident_repeated_declared_final_norm_and_lm_head"
                            : "resident_repeated_complete_declared_decoder_block")
              << "\",\"included\":";
    emit_string_list(operations(o, stage_kind));
    std::cout << ",\"excluded\":";
    emit_string_list({"allocation", "quantization", "io", "external_concurrency"});
    std::cout << ",\"nodes\":[";
    for (int i = 0; i < ggml_graph_n_nodes(stage->graph); ++i) {
        const auto *n = ggml_graph_node(stage->graph, i);
        std::cout << (i ? "," : "") << "{\"op\":\"" << ggml_op_name(n->op)
                  << "\",\"bytes\":" << ggml_nbytes(n)
                  << ",\"view\":" << (n->view_src ? "true" : "false") << "}";
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
                      : v->type == GGML_TYPE_F16 ? "f16" : "f32") << "\",\"shape\":[" << v->ne[0]
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
    const size_t kv_per_token =
                     product({o.head_dim, o.kv_heads}) *
                     static_cast<size_t>(ggml_type_size(kv_cache_type(o))) * 2,
                 history_bytes = a ? static_cast<size_t>(o.history) * kv_per_token : 0,
                 new_bytes = a ? static_cast<size_t>(o.q) * kv_per_token : 0;
    std::cout
        << ",\"weight_bytes\":" << stage->weight_bytes << ",\"input_bytes\":" << stage->input_bytes
        << ",\"output_bytes\":" << stage->output_bytes << ",\"kv_history_bytes\":" << history_bytes
        << ",\"kv_new_bytes\":" << new_bytes << ",\"kv_cache_bytes\":" << stage->kv_bytes
        << ",\"resident_logical_bytes\":" << resident << ",\"workspace_bytes\":" << workspace
        << ",\"workspace_source\":\""
        << (paired ? "shared_gallocr_plus_execution_mode_live_cpu_work_buffers"
                   : "ggml_gallocr_internal_buffer_plus_cpu_graph_plan")
        << "\",\"internal_"
           "tensor_bytes_upper_bound\":"
        << std::max(stage->internal_bytes, stage->logical_internal)
        << ",\"cpu_cplan_work_size_bytes\":" << stage->cpu_work
        << ",\"memory_accounting\":{\"resident_logical_bytes\":" << resident
        << ",\"resident_allocated_bytes\":" << ggml_backend_buffer_get_size(s.buffer)
        << ",\"internal_allocated_buffer_bytes\":" << stage->internal_bytes
        << ",\"internal_logical_tensor_bytes_sum\":" << stage->logical_internal
        << ",\"cpu_cplan_work_size_bytes\":" << stage->cpu_work
        << ",\"workspace_bytes\":" << workspace
        << ",\"workspace_scope\":\"internal_reusable_allocator_capacity_including_alignment_plus_"
           "cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads\","
           "\"workspace_bound\":\"conservative_allocated_capacity\",\"kv_new_bytes_semantics\":"
           "\"appended_slots_in_cache_distinct_from_new_kv_input_tensors\"}";
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
    if (paired) {
        std::cout << ",\"compute_mode\":\"" << o.compute_mode
                  << "\",\"min_warmup_ms\":" << o.min_warmup_ms
                  << ",\"warmup_rounds_requested\":" << o.warmup
                  << ",\"warmup_rounds_effective\":" << pair->warmup_rounds
                  << ",\"actual_warmup_elapsed_ns\":" << pair->warmup_elapsed_ns
                  << ",\"diagnostic_only\":true,\"m2_cost_export_eligible\":false,"
                     "\"paired_order\":";
        emit_string_list(pair->order);
        std::cout << ",\"pair_order_semantics\":\"A=projection_then_attention_then_dense_ffn;"
                     "B=joint_block;sample_parity_alternates_AB_BA\","
                     "\"sample_timing_scope\":\"one_graph_execution_per_path_per_pair;"
                     "serial_outer_includes_nested_stage_clock_and_loop_overhead\","
                     "\"setup_path_repetitions\":{\"serial\":1,\"joint\":1},"
                     "\"warmup_path_repetitions\":{\"serial\":"
                  << pair->warmup_rounds << ",\"joint\":" << pair->warmup_rounds
                  << "},\"stage_wall_ns\":{";
        first = true;
        for (const auto &name : {"projection", "attention", "dense_ffn"}) {
            std::cout << (first ? "" : ",") << "\"" << name << "\":";
            emit_ns(pair->stage_wall.at(name));
            first = false;
        }
        std::cout << "},\"stage_sum_wall_ns\":";
        emit_ns(pair->stage_sum);
        std::cout << ",\"serial_wall_ns\":";
        emit_ns(pair->serial_wall);
        std::cout << ",\"joint_wall_ns\":";
        emit_ns(pair->joint_wall);
        std::cout << ",\"stage_node_ranges\":{";
        first = true;
        for (const auto &name : {"projection", "attention", "dense_ffn"}) {
            const auto range = stage->node_ranges.at(name);
            std::cout << (first ? "" : ",") << "\"" << name << "\":{\"begin\":" << range.first
                      << ",\"end\":" << range.second << ",\"ops\":[";
            for (int n = range.first; n < range.second; ++n)
                std::cout << (n == range.first ? "" : ",") << "\""
                          << ggml_op_name(ggml_graph_node(stage->graph, n)->op) << "\"";
            std::cout << "]}";
            first = false;
        }
        const size_t physical_tensor_and_work_bytes =
            add_bytes(add_bytes(ggml_backend_buffer_get_size(s.buffer), stage->internal_bytes),
                      add_bytes(pair->backend_capacity, pair->cached_work_total));
        std::cout << "},\"shared_memory_inventory\":{\"resident_logical_bytes\":" << resident
                  << ",\"resident_allocated_bytes\":" << ggml_backend_buffer_get_size(s.buffer)
                  << ",\"internal_allocated_buffer_bytes\":" << stage->internal_bytes
                  << ",\"per_graph_cplan_work_size_bytes\":{";
        first = true;
        for (const auto &item : pair->work_sizes) {
            std::cout << (first ? "" : ",") << "\"" << item.first << "\":" << item.second;
            first = false;
        }
        std::cout << "},\"backend_shared_cplan_capacity_bytes\":" << pair->backend_capacity
                  << ",\"cached_plans_total_work_bytes\":" << pair->cached_work_total
                  << ",\"physical_tensor_and_work_allocated_bytes\":"
                  << physical_tensor_and_work_bytes
                  << ",\"timed_paths_share_tensor_addresses\":true,"
                     "\"timed_paths_share_gallocr_allocation\":true,"
                     "\"timed_paths_share_backend_and_threadpool\":true,"
                     "\"host_metadata_reference_and_runtime_overheads_included\":false,"
                     "\"stage_workspace_is_independent_cost_bound\":false,"
                     "\"cplan_work_allocation_basis\":\"local_ggml_cpu_backend_new_uint8_work_size;"
                     "graph_mode_reuses_max_capacity;plan_mode_owns_four_separate_work_buffers\"},"
                     "\"paired_validation\":{\"serial\":";
        emit_path_snapshot(pair->serial);
        std::cout << ",\"joint\":";
        emit_path_snapshot(pair->joint);
        std::cout << ",\"ports_exact\":"
                  << (pair->serial.port_bytes == pair->joint.port_bytes ? "true" : "false")
                  << ",\"kv_caches_exact\":"
                  << (pair->serial.cache_bytes == pair->joint.cache_bytes ? "true" : "false")
                  << ",\"input_ports_unchanged\":" << (pair->inputs_unchanged ? "true" : "false")
                  << ",\"checked_after_timing\":true,"
                     "\"validation_scope\":\"untimed_path_equivalence_after_samples;"
                     "joint_reference_check_applies_to_bitwise_equal_serial_ports\"}";
    } else {
        std::cout << ",\"wall_ns\":";
        emit_ns(wall);
        if (o.thread_cycles) {
            std::cout << ",\"process_cycles\":";
            emit_ns(cycles);
            std::cout << ",\"process_cycles_scope\":\"raw_QueryProcessCycleTime_delta_summed_over_"
                         "process_threads;bracket_wider_than_wall_by_two_counter_reads;"
                         "no_conversion_to_nanoseconds;-1_means_the_counter_failed\"";
        }
    }
    std::cout << "}\n";
    if (!passed) {
        std::cerr << "dense layer numerical or KV validation failed\n";
        return 2;
    }
    return 0;
}
// One case per line, arguments separated by tabs.  Each case is the same
// parse()+run() pair as a standalone invocation, so a record is byte-identical
// to what that command line prints on its own; only the process it runs in is
// shared.  Records are framed so a case that dies half-way cannot be mistaken
// for the next one: every record is captured into its own stream and is
// followed by a "#case <index> <code>" sentinel line.
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
        owned.emplace_back("ggml-dense-layer-probe");
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
        // A case that fails its own numerical/KV check still produced a record
        // (with "passed":false); keep it next to the failure code.
        if (!record.str().empty())
            std::cout << record.str();
        std::cout << "#case " << index << ' ' << code << '\n';
        std::cout.flush();
        if (code != 0) {
            ++failed;
            std::cerr << "ggml-dense-layer-probe case " << index << " failed (" << code << ')';
            if (!error.empty())
                std::cerr << ": " << error;
            std::cerr << '\n';
        }
        ++index;
    }
    std::cerr << "ggml-dense-layer-probe: " << index << " case(s), " << failed << " failed\n";
    return failed == 0 ? 0 : 3;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--cases-from-stdin") {
        try {
            return serve_cases(std::cin);
        } catch (const std::exception &e) {
            std::cerr << "ggml-dense-layer-probe stdin: " << e.what() << '\n';
            return 1;
        }
    }
    if (argc == 2 && std::string(argv[1]) == "--cpu-topology") {
        try {
            return emit_cpu_topology();
        } catch (const std::exception &e) {
            std::cerr << "ggml-dense-layer-probe topology: " << e.what() << '\n';
            return 1;
        }
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "ggml-dense-layer-probe --stage projection|attention|dense_ffn|moe_ffn|block|"
                     "paired|embedding|head; embedding/head additionally require --vocab and both "
                     "--lm-head-dtype and --token-embd-dtype; moe_ffn/block with --experts and "
                     "--top-k additionally require the three --*-exps-dtype options and treat "
                     "--intermediate as one expert's width, with --shared-intermediate and the "
                     "three --shexp-*-dtype options when the checkpoint has a shared expert; "
                     "paired additionally requires --compute-mode graph|plan and --min-warmup-ms; "
                     "optional --cpu-mask logical_id,logical_id (one per thread, applied to "
                     "workers and main and read back before timing); optional --evict-bytes N "
                     "streams N bytes of scratch outside the timed region before each warmup "
                     "iteration and each sample, declaring a cold-cache residency regime "
                     "(default 0, resident warm cache); standalone --cpu-topology; "
                     "standalone --cases-from-stdin runs one case per input line (tab-separated "
                     "arguments, '#' starts a comment line) and prints each case's record followed "
                     "by a '#case <index> <code>' sentinel, continuing after a failing case so a "
                     "whole price table can be measured inside one process; "
                     "explicit CPU B1 decoder contract; see dense_contract.py native_arguments\n";
        return 0;
    }
    try {
        return run(parse(argc, argv));
    } catch (const std::exception &e) {
        std::cerr << "ggml-dense-layer-probe: " << e.what() << '\n';
        return 1;
    }
}
