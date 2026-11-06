// IR -> ggml executor for the two-dimension evaluation (20261106-ir-ggml-evaluation).
//
// Input is an *instantiated* topology emitted by instantiate.py: every IR node is
// already bound to a real model geometry, a token count and a weight dtype.  This
// program never reads a price table -- dimension B exists exactly because the
// compiler plans with prices while the plan is judged by measurement.
//
// Execution model follows the IR, not a single stream:
//   * one worker thread per machine (cpu, gpu, pcie);
//   * a node runs when all of its deps have finished;
//   * compute nodes are one ggml graph each, built while the node waits (build
//     time is recorded separately and never folded into the action time);
//   * copy nodes move the declared bytes between a host and a device buffer.
// The CUDA backend is synchronized at the end of every node so a downstream
// consumer on another machine always sees completed data.  Launch overhead is
// therefore inside the action time; gate 2 measures it on an empty graph
// (--empty-overhead) instead of leaving it implicit.
//
// Usage:
//   ggml-ir-runner --case path.txt [--replicas 3] [--out out.json]
//   ggml-ir-runner --empty-overhead [--threads 2]
//
// Case file (one directive per line, '#' starts a comment):
//   case <name> | model <name> | threads <n> | tokens <n> | history <n>
//   gpu <0|1> | experts <n> | top_k <n>
//   dtype_<role> <name>          # q k v o gate up down router kv
//   node <idx> <op> <machine> <deps|-> <k=v,...>
//
// Edge-tier declarations (all optional; absent = the plain two-tier executor, so a case
// written before they existed executes exactly as it did):
//   vram_bytes <n>   dram_bytes <n>   ssd_bytes <n>   # device budgets, 0 = unbounded
//   reclaim <release_clean_weights|decline>           # what to do when a budget is full
//   release_order <lru|largest_first|history_frequency>
//   uma <0|1>                        # host and device share one address space
// `ssd_bytes > 0` gives the case a third tier: copy nodes whose machine is `ssd` read and
// write a file (--ssd-dir) instead of a device buffer.  `uma 1` turns every cpu<->gpu copy
// into an alias: the bytes are declared but not moved, which is what an integrated GPU
// does.  A capacity breach is reported as `evict_bytes` on the node that caused it, and
// under `decline` it fails the node -- an infeasible plan is a failed execution, not a
// silently cheaper one.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock_type::now().time_since_epoch()).count();
}

struct Options {
    std::string case_path, out_path, ssd_dir;
    int replicas = 1, threads = 2, cuda_device = 0;
    bool empty_overhead = false, verbose = false, no_gpu = false;
};

// ---------------------------------------------------------------- case model

struct Node {
    int index = -1;
    std::string id, op, machine;
    std::vector<int> deps;
    std::map<std::string, std::string> p;

    std::string str(const char * k, const std::string & f = "") const {
        auto it = p.find(k); return it == p.end() ? f : it->second;
    }
    int64_t num(const char * k, int64_t f = 0) const {
        auto it = p.find(k); return it == p.end() ? f : std::stoll(it->second);
    }
};

struct Case {
    std::string name, model;
    int threads = 2, tokens = 1, history = 0, cuda_device = 0;
    int64_t experts = 0, top_k = 0;
    bool gpu = true;
    std::vector<int> cpu_ids;          // strict placement, one per thread, as the tables used
    std::map<std::string, std::string> dtype;
    std::vector<Node> nodes;
    // Edge-tier declarations.  A budget of 0 means "unbounded", which is what every case
    // in the corpus declared until the capacity family existed; the ledger is then never
    // consulted and the copy path is the one that was there before.
    int64_t vram_bytes = 0, dram_bytes = 0, ssd_bytes = 0;
    std::string reclaim;               // "" | release_clean_weights | decline
    std::string release_order = "lru"; // lru | largest_first | history_frequency
    bool uma = false;
};

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static Case read_case(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open case " + path);
    Case c;
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#'); if (hash != std::string::npos) line = line.substr(0, hash);
        auto w = split(line, ' ');
        if (w.empty()) continue;
        const std::string & d = w[0];
        if (d == "case") c.name = w.size() > 1 ? w[1] : "";
        else if (d == "model") c.model = w.size() > 1 ? w[1] : "";
        else if (d == "threads") c.threads = std::stoi(w[1]);
        else if (d == "tokens") c.tokens = std::stoi(w[1]);
        else if (d == "history") c.history = std::stoi(w[1]);
        else if (d == "gpu") c.gpu = std::stoi(w[1]) != 0;
        else if (d == "cuda_device") c.cuda_device = std::stoi(w[1]);
        else if (d == "cpu_mask") for (auto & x : split(w[1], ',')) c.cpu_ids.push_back(std::stoi(x));
        else if (d == "experts") c.experts = std::stoll(w[1]);
        else if (d == "top_k") c.top_k = std::stoll(w[1]);
        else if (d == "vram_bytes") c.vram_bytes = std::stoll(w[1]);
        else if (d == "dram_bytes") c.dram_bytes = std::stoll(w[1]);
        else if (d == "ssd_bytes") c.ssd_bytes = std::stoll(w[1]);
        else if (d == "reclaim") c.reclaim = w[1];
        else if (d == "release_order") c.release_order = w[1];
        else if (d == "uma") c.uma = std::stoi(w[1]) != 0;
        else if (d.rfind("dtype_", 0) == 0) c.dtype[d.substr(6)] = w[1];
        else if (d == "node") {
            if (w.size() < 5) throw std::runtime_error("short node line: " + line);
            Node n;
            n.index = std::stoi(w[1]); n.op = w[2]; n.machine = w[3];
            if (w[4] != "-") for (auto & x : split(w[4], ',')) n.deps.push_back(std::stoi(x));
            std::string rest;
            for (size_t i = 5; i < w.size(); ++i) rest += (rest.empty() ? "" : " ") + w[i];
            for (auto & kv : split(rest, ',')) {
                auto eq = kv.find('=');
                if (eq != std::string::npos) n.p[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
            n.id = n.str("id", std::to_string(n.index));
            c.nodes.push_back(n);
        }
    }
    if (c.nodes.empty()) throw std::runtime_error("case has no nodes");
    std::sort(c.nodes.begin(), c.nodes.end(), [](const Node & a, const Node & b) { return a.index < b.index; });
    return c;
}

// ---------------------------------------------------------------- geometry

struct Geometry {
    int64_t hidden = 1024, intermediate = 3072;
    int64_t heads = 16, kv_heads = 8, head_dim = 128;
    int64_t rope_dims = 128;
    double rope_base = 1e6, norm_eps = 1e-6;
    std::string qk_norm = "rms", attention_gate = "none", key_cache_layout = "head_major";
    int64_t q_dim() const { return heads * head_dim; }
    int64_t kv_dim() const { return kv_heads * head_dim; }

    static ggml_type dtype_of(const std::string & n) {
        if (n.empty() || n == "f32") return GGML_TYPE_F32;
        if (n == "f16") return GGML_TYPE_F16;
        if (n == "bf16") return GGML_TYPE_BF16;
        if (n == "q4_0") return GGML_TYPE_Q4_0;
        if (n == "q4_k" || n == "q4_K") return GGML_TYPE_Q4_K;
        if (n == "q5_k") return GGML_TYPE_Q5_K;
        if (n == "q6_k") return GGML_TYPE_Q6_K;
        if (n == "q8_0") return GGML_TYPE_Q8_0;
        throw std::runtime_error("unsupported dtype " + n);
    }
};

static void fill_bytes(void * dst, size_t bytes, uint32_t seed, ggml_type type) {
    auto * p = static_cast<uint8_t *>(dst);
    std::mt19937 rng(seed);
    for (size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(rng() & 0xFF);
    if (type == GGML_TYPE_F32) {
        auto * f = static_cast<float *>(dst);
        for (size_t i = 0; i < bytes / 4; ++i) f[i] = 0.5f + 0.01f * float(int(rng() % 100) - 50);
    } else if (type == GGML_TYPE_F16) {
        auto * h = static_cast<ggml_fp16_t *>(dst);
        for (size_t i = 0; i < bytes / 2; ++i) h[i] = ggml_fp32_to_fp16(0.05f);
    } else if (ggml_is_quantized(type)) {
        const size_t ts = ggml_type_size(type), blocks = bytes / ts;
        for (size_t b = 0; b < blocks; ++b) {
            auto * s = reinterpret_cast<ggml_fp16_t *>(p + b * ts);
            s[0] = ggml_fp32_to_fp16(0.02f);
        }
    }
}

// ---------------------------------------------------------------- session

struct Port {
    ggml_tensor * cpu = nullptr;
    ggml_tensor * gpu = nullptr;
    int writer = 0;         // 0 unwritten, 1 cpu, 2 gpu
};

// Scatter offsets of one (tokens, history) shape: where a node of that shape writes its
// new keys/values and which positions its tokens occupy.  They are per-shape because the
// value offsets are laid out with that shape's own total length L.
struct KVScatter {
    ggml_tensor * koff = nullptr, * voff = nullptr, * positions = nullptr;
    ggml_tensor * koff_gpu = nullptr, * voff_gpu = nullptr, * positions_gpu = nullptr;
};

// ---------------------------------------------------------------- residency
//
// An edge device has a *budget*, not an address space: an accelerator holds a few hundred
// megabytes of weights and admitting one more block can require another block to leave.
// The ledger is consulted only when the case declares a budget, so a case that declares
// none runs the copy path unchanged.
//
// What leaves a tier when it is full is the strategy the simulator declares in
// `reconfiguration.ReconfigurationSpec` -- `release_clean_weights` with an order of
// `lru` / `largest_first` / `history_frequency`, or `decline`.  Executing those here is
// what makes the declaration a comparison unit instead of a label: the same three words
// pick the victim on both sides.
enum Dev { DEV_CPU = 0, DEV_GPU = 1, DEV_SSD = 2, DEV_COUNT = 3 };

struct Resident {
    int64_t bytes = 0;
    int64_t last_touch = 0;     // admission tick -- the `lru` key
    int64_t touches = 1;        // -- the `history_frequency` key
    int64_t offset = -1;        // byte offset in the SSD file (DEV_SSD only)
};

struct Ledger {
    std::mutex mu;
    std::map<std::string, Resident> live[DEV_COUNT];
    int64_t used[DEV_COUNT] = {0, 0, 0};
    int64_t peak[DEV_COUNT] = {0, 0, 0};
    int64_t cap[DEV_COUNT] = {0, 0, 0};
    int64_t tick = 0;
    int64_t evictions = 0, evicted_bytes = 0;
    int64_t ssd_next = 0;                              // bump allocator inside the SSD file
    std::vector<std::pair<int64_t, int64_t>> ssd_free;  // (offset, size) returned by eviction
    // A replica is a fresh process's worth of state on the same hardware: the tier files
    // and the buffers are the same, what is resident is not.  Peaks and eviction counts
    // are cumulative so a replica cannot hide the worst one.
    void reset_state() {
        for (int d = 0; d < DEV_COUNT; ++d) { live[d].clear(); used[d] = 0; }
        tick = 0; ssd_next = 0; ssd_free.clear();
    }
};

static int dev_of(const std::string & machine) {
    if (machine == "gpu") return DEV_GPU;
    if (machine == "ssd") return DEV_SSD;
    return DEV_CPU;
}

static const char * dev_name(int dev) {
    return dev == DEV_GPU ? "gpu" : dev == DEV_SSD ? "ssd" : "cpu";
}

// The victim the declared ordering picks.  `spare` is the tensor being admitted: evicting
// it would be a no-op transfer of the bytes that are about to arrive.
static std::string pick_victim(const std::map<std::string, Resident> & live,
                               const std::string & order, const std::string & spare) {
    std::string best;
    int64_t best_score = 0;
    for (const auto & kv : live) {
        if (kv.first == spare) continue;
        const int64_t score = order == "largest_first" ? kv.second.bytes
                            : order == "history_frequency" ? kv.second.touches
                            : kv.second.last_touch;             // lru
        // largest_first keeps the biggest victim; the other two keep the smallest key
        const bool better = best.empty()
                         || (order == "largest_first" ? score > best_score : score < best_score);
        if (better) { best = kv.first; best_score = score; }
    }
    return best;
}

struct Session {
    Case c;
    Geometry g;
    bool verbose = false;
    ggml_backend_t cpu = nullptr, gpu = nullptr;
    ggml_threadpool_t pool = nullptr;
    ggml_context * resident_cpu = nullptr, * resident_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr, buf_gpu = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_cpu_buf = nullptr, weights_gpu_buf = nullptr;
    std::map<std::string, ggml_tensor *> w_cpu, w_gpu;
    // One port set per distinct (tokens, history).  Every graph writes its outputs with
    // `ggml_cpy`, whose destination must have exactly the source's shape, so a pd case --
    // a prefill round at t tokens and a decode round at t=1 against the prefill's KV --
    // cannot share one port set.  Ports only ever carry data within one phase; the only
    // thing that crosses a phase boundary is the KV cache, which is shared on purpose.
    std::vector<std::pair<int64_t, int64_t>> shapes;   // in order of first use
    std::map<std::pair<int64_t, int64_t>, std::map<std::string, Port>> ports;
    std::map<std::pair<int64_t, int64_t>, KVScatter> scatter;
    size_t cur_shape = 0;                              // set while a node graph is built
    std::pair<int64_t, int64_t> shape() const { return shapes[cur_shape]; }
    int64_t T() const { return shapes[cur_shape].first; }    // tokens executed at once
    int64_t H() const { return shapes[cur_shape].second; }   // visible history before them
    int64_t L() const { return T() + H(); }
    ggml_tensor * kcache_cpu = nullptr, * vcache_cpu = nullptr;
    ggml_tensor * kcache_gpu = nullptr, * vcache_gpu = nullptr;
    ggml_tensor * stage_host = nullptr, * stage_dev = nullptr;
    int64_t stage_bytes = 0;
    int verify_failures = 0;
    // third tier: a file the `ssd` machine reads and writes.  Opened only when the case
    // declares one; an unbuffered handle so a repeated stream cannot be served from the
    // OS page cache and read back as device bandwidth.
    void * ssd_handle = nullptr;
    void * ssd_buf = nullptr;          // aligned bounce buffer, one 1 MiB chunk
    int64_t ssd_buf_bytes = 0;
    int64_t ssd_file_bytes = 0;
    Ledger ledger;
    // The third tier's worker is a second thread, and it must not share the PCIe thread's
    // staging pool: two tiers moving bytes through one buffer would make both readings a
    // measure of the interleaving.  Its own pair costs one max_copy tensor.
    ggml_tensor * ssd_stage_host = nullptr, * ssd_stage_dev = nullptr;

    ~Session() {
        if (cpu) ggml_backend_synchronize(cpu);
        if (gpu) ggml_backend_synchronize(gpu);
        if (buf_cpu) ggml_backend_buffer_free(buf_cpu);
        if (buf_gpu) ggml_backend_buffer_free(buf_gpu);
        if (weights_cpu_buf) ggml_backend_buffer_free(weights_cpu_buf);
        if (weights_gpu_buf) ggml_backend_buffer_free(weights_gpu_buf);
        if (resident_cpu) ggml_free(resident_cpu);
        if (resident_gpu) ggml_free(resident_gpu);
        if (weights_ctx) ggml_free(weights_ctx);
        if (cpu) ggml_backend_free(cpu);
        if (gpu) ggml_backend_free(gpu);
        close_ssd();
    }

    void close_ssd() {
#if defined(_WIN32)
        if (ssd_handle) { CloseHandle(static_cast<HANDLE>(ssd_handle)); ssd_handle = nullptr; }
        if (ssd_buf) { VirtualFree(ssd_buf, 0, MEM_RELEASE); ssd_buf = nullptr; }
#endif
    }
};

// ---------------------------------------------------------------- tiers

// Unbuffered I/O requires sector-aligned offsets and lengths.  A tensor is rounded up to
// one sector when it is given a slot, so at most 4095 bytes per tensor are ever touched
// beyond its size -- well under a thousandth for the smallest weight matrix in the corpus.
static constexpr int64_t SSD_SECTOR = 4096;
static int64_t round_up(int64_t n) { return (n + SSD_SECTOR - 1) / SSD_SECTOR * SSD_SECTOR; }

#if defined(_WIN32)
static void open_ssd(Session & s, const std::string & path) {
    // The layout of the SSD tier is a file the executor owns; the path is ASCII.
    std::wstring wide(path.begin(), path.end());
    HANDLE h = CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING
                           | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open the ssd tier file " + path);
    LARGE_INTEGER size; size.QuadPart = s.c.ssd_bytes;
    if (!SetFilePointerEx(h, size, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) {
        CloseHandle(h);
        throw std::runtime_error("cannot size the ssd tier file to " + std::to_string(s.c.ssd_bytes));
    }
    s.ssd_handle = h;
    s.ssd_file_bytes = s.c.ssd_bytes;
    s.ssd_buf_bytes = int64_t(1) << 20;
    s.ssd_buf = VirtualAlloc(nullptr, size_t(s.ssd_buf_bytes), MEM_COMMIT | MEM_RESERVE,
                             PAGE_READWRITE);
    if (!s.ssd_buf) throw std::runtime_error("ssd bounce buffer allocation failed");
}
#else
static void open_ssd(Session & s, const std::string &) {
    s.ssd_file_bytes = 0;
    throw std::runtime_error("the ssd tier needs the Windows unbuffered path in this build");
}
#endif

// One leg of a copy, host <-> device, through the staging pool.  This is the body the
// two-tier path has always used; it is factored out so eviction and the third tier move
// bytes the same way instead of inventing a second one.
static void move_pair(Session & s, ggml_tensor * host, ggml_tensor * dev, bool to_gpu,
                      int64_t bytes) {
    if (bytes > s.stage_bytes) throw std::runtime_error("copy larger than staging pool");
    ggml_context * vctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto * from = ggml_view_1d(vctx, to_gpu ? host : dev, bytes, 0);
    auto * to = ggml_view_1d(vctx, to_gpu ? dev : host, bytes, 0);
    if (!from->buffer && ggml_backend_view_init(from) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("stage view init failed");
    if (!to->buffer && ggml_backend_view_init(to) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("stage view init failed");
    if (s.gpu) ggml_backend_synchronize(s.gpu);
    ggml_backend_synchronize(s.cpu);
    ggml_backend_tensor_copy(from, to);
    if (s.gpu) ggml_backend_synchronize(s.gpu);
    ggml_backend_synchronize(s.cpu);
    ggml_free(vctx);
}

// `alt` selects the third tier's own staging pair, so the PCIe thread and the SSD thread
// never touch the same buffer.
static void move_staging(Session & s, bool to_gpu, int64_t bytes, bool alt = false) {
    if (alt) move_pair(s, s.ssd_stage_host, s.ssd_stage_dev, to_gpu, bytes);
    else move_pair(s, s.stage_host, s.stage_dev, to_gpu, bytes);
}

#if defined(_WIN32)
// One leg of a copy through the third tier.  `bytes` and `offset` are sector multiples.
static void move_ssd(Session & s, bool write, int64_t offset, int64_t bytes) {
    if (!s.ssd_handle) throw std::runtime_error("ssd copy with no third tier opened");
    if (offset + round_up(bytes) > s.ssd_file_bytes)
        throw std::runtime_error("ssd copy past the end of the tier file");
    HANDLE h = static_cast<HANDLE>(s.ssd_handle);
    LARGE_INTEGER pos; pos.QuadPart = offset;
    if (!SetFilePointerEx(h, pos, nullptr, FILE_BEGIN))
        throw std::runtime_error("ssd seek failed");
    char * host = static_cast<char *>(s.ssd_stage_host->data);
    int64_t done = 0;
    const int64_t total = round_up(bytes);
    while (done < total) {
        const int64_t n = std::min(s.ssd_buf_bytes, total - done);
        DWORD got = 0;
        if (write) {
            std::memcpy(s.ssd_buf, host + done, size_t(n));
            if (!WriteFile(h, s.ssd_buf, DWORD(n), &got, nullptr) || int64_t(got) != n)
                throw std::runtime_error("ssd write failed");
        } else {
            if (!ReadFile(h, s.ssd_buf, DWORD(n), &got, nullptr) || int64_t(got) != n)
                throw std::runtime_error("ssd read failed");
            std::memcpy(host + done, s.ssd_buf, size_t(n));
        }
        done += n;
    }
}
#else
static void move_ssd(Session &, bool, int64_t, int64_t) {
    throw std::runtime_error("the ssd tier needs the Windows unbuffered path in this build");
}
#endif

struct Admission {
    int64_t evicted = 0;    // bytes that had to leave to make room
    int64_t offset = -1;    // where the tensor lives in the tier file, DEV_SSD only
};

static int64_t resident_offset(Session & s, const std::string & key) {
    std::lock_guard<std::mutex> lock(s.ledger.mu);
    auto it = s.ledger.live[DEV_SSD].find(key);
    return it == s.ledger.live[DEV_SSD].end() ? -1 : it->second.offset;
}

static int64_t ssd_slot(Ledger & L, int64_t bytes) {
    const int64_t want = round_up(bytes);
    for (auto & block : L.ssd_free) {
        if (block.second >= want) {
            const int64_t off = block.first;
            block.first += want;
            block.second -= want;
            return off;
        }
    }
    const int64_t off = round_up(L.ssd_next);
    L.ssd_next = off + want;
    return off;
}

// Admit `bytes` of `key` into `dev`, evicting per the case's declared policy.
//
// Lock order is ledger -> staging: an eviction may move bytes, and the copy path takes
// only the staging lock once it is past this call.  Nothing else takes the ledger.
static Admission acquire(Session & s, int dev, const std::string & key, int64_t bytes,
                         bool alt = false) {
    Ledger & L = s.ledger;
    std::lock_guard<std::mutex> lock(L.mu);
    Admission out;
    auto it = L.live[dev].find(key);
    if (it != L.live[dev].end()) {
        it->second.last_touch = ++L.tick;
        it->second.touches += 1;
        if (dev == DEV_SSD) out.offset = it->second.offset;
        return out;                                  // already resident: nothing leaves
    }
    while (L.cap[dev] > 0 && L.used[dev] + bytes > L.cap[dev]) {
        if (s.c.reclaim.empty() || s.c.reclaim == "decline")
            throw std::runtime_error(std::string("capacity exceeded on ") + dev_name(dev)
                                     + ": " + std::to_string(bytes) + " bytes, short by "
                                     + std::to_string(std::max<int64_t>(
                                           0, bytes - (L.cap[dev] - L.used[dev])))
                                     + " bytes");
        const std::string victim = pick_victim(L.live[dev], s.c.release_order, key);
        if (victim.empty())
            throw std::runtime_error(std::string("capacity exceeded on ") + dev_name(dev)
                                     + " with nothing evictable -- the plan does not fit");
        const Resident r = L.live[dev][victim];
        L.live[dev].erase(victim);
        L.used[dev] -= r.bytes;
        L.evictions += 1;
        L.evicted_bytes += r.bytes;
        out.evicted += r.bytes;
        if (dev == DEV_GPU) {
            // the victim really leaves the accelerator, on the same copy engine
            move_staging(s, false, r.bytes, alt);
        } else if (dev == DEV_SSD) {
            // the third tier is the last one: dropping a resident costs no transfer, and
            // the cost lands on the next read -- which is where the policy should show.
            L.ssd_free.push_back({r.offset, round_up(r.bytes)});
        }
    }
    Resident res;
    res.bytes = bytes; res.last_touch = ++L.tick; res.touches = 1;
    if (dev == DEV_SSD) {
        res.offset = ssd_slot(L, bytes);
        if (res.offset + round_up(bytes) > s.ssd_file_bytes)
            throw std::runtime_error("the case's working set does not fit the ssd tier file");
        out.offset = res.offset;
    }
    L.live[dev][key] = res;
    L.used[dev] += bytes;
    L.peak[dev] = std::max(L.peak[dev], L.used[dev]);
    return out;
}


static const std::string & dtype_of_role(const Case & c, const std::string & role,
                                         const char * fallback) {
    auto it = c.dtype.find(role);
    if (it != c.dtype.end() && !it->second.empty()) return it->second;
    static const std::string fb[8] = {};
    (void) fb;
    static std::string scratch;
    scratch = fallback;
    return scratch;
}

static void init_session(Session & s) {
    const Case & c = s.c;
    Geometry & g = s.g;
    if (!c.model.empty()) {
        // The instantiator writes the geometry into dtype_* and the model name only;
        // the numeric geometry comes from the case file's own geometry_* lines.
    }
    s.cpu = ggml_backend_cpu_init();
    if (!s.cpu) throw std::runtime_error("cpu backend init failed");
    if (!c.cpu_ids.empty()) {
        // The frozen tables were measured with explicit_strict_cpu_mask; on this machine a
        // free-running process lands on a different band, so the placement is part of the
        // instrument and is echoed in the output, not silently assumed.
        if (static_cast<int>(c.cpu_ids.size()) != c.threads)
            throw std::runtime_error("cpu_mask must name exactly `threads` logical CPUs");
        auto tpp = ggml_threadpool_params_default(c.threads);
        tpp.strict_cpu = true;
        for (const int id : c.cpu_ids) {
            if (id < 0 || id >= GGML_MAX_N_THREADS) throw std::runtime_error("cpu_mask out of range");
            tpp.cpumask[id] = true;
        }
#if defined(_WIN32)
        {
            DWORD_PTR bits = 0;
            for (const int id : c.cpu_ids) bits |= DWORD_PTR{1} << id;
            if (!SetProcessAffinityMask(GetCurrentProcess(), bits))
                throw std::runtime_error("cannot apply process CPU affinity");
            DWORD_PTR observed = 0, system_mask = 0;
            if (!GetProcessAffinityMask(GetCurrentProcess(), &observed, &system_mask) || observed != bits)
                throw std::runtime_error("process CPU affinity readback differs from the request");
        }
#endif
        s.pool = ggml_threadpool_new(&tpp);
        if (!s.pool) throw std::runtime_error("threadpool initialization failed");
        ggml_backend_cpu_set_threadpool(s.cpu, s.pool);
    }
    ggml_backend_cpu_set_n_threads(s.cpu, c.threads);
#ifdef GGML_USE_CUDA
    if (c.gpu) {
        s.gpu = ggml_backend_cuda_init(c.cuda_device);
        if (!s.gpu) throw std::runtime_error("cuda backend init failed");
    }
#else
    if (c.gpu) throw std::runtime_error("built without CUDA support but the case asks for gpu");
#endif
    (void) dtype_of_role;

    const int64_t nports = 16;
    // A node carries its own shape in `t=`/`h=` when that shape differs from the case
    // header's (instantiate.py writes it for the pd family's decode round only), so one
    // execution runs a prefill round and a decode round at their own shapes.
    auto shape_of = [&](const Node & n) {
        return std::make_pair(n.num("t", c.tokens), n.num("h", c.history));
    };
    int64_t max_L = 1;
    for (const auto & n : c.nodes) {
        if (n.op == "copy" || n.op == "arrival" || n.op == "request_arrival") continue;
        const auto key = shape_of(n);
        if (std::find(s.shapes.begin(), s.shapes.end(), key) == s.shapes.end())
            s.shapes.push_back(key);
        max_L = std::max(max_L, key.first + key.second);
    }
    if (s.shapes.empty()) s.shapes.push_back({c.tokens, c.history});
    for (const auto & key : s.shapes)
        max_L = std::max(max_L, key.first + key.second);

    const int64_t nmeta = nports * static_cast<int64_t>(s.shapes.size()) + 32;
    s.resident_cpu = ggml_init({ggml_tensor_overhead() * nmeta, nullptr, true});
    if (s.gpu) s.resident_gpu = ggml_init({ggml_tensor_overhead() * nmeta, nullptr, true});
    if (!s.resident_cpu) throw std::runtime_error("resident context allocation failed");

    for (const auto & key : s.shapes) {
        const int64_t T = key.first, L = key.first + key.second;
        auto & set = s.ports[key];
        auto make_port = [&](const std::string & name, ggml_type type, int64_t a, int64_t b,
                             int64_t c3) {
            Port p;
            auto * t = ggml_new_tensor_3d(s.resident_cpu, type, a, b, c3);
            ggml_set_name(t, name.c_str());
            p.cpu = t;
            if (s.gpu) {
                auto * u = ggml_new_tensor_3d(s.resident_gpu, type, a, b, c3);
                ggml_set_name(u, name.c_str());
                p.gpu = u;
            }
            set[name] = p;
        };
        make_port("x", GGML_TYPE_F32, g.hidden, T, 1);
        make_port("query", GGML_TYPE_F32, g.head_dim, T, g.heads);
        make_port("new_key", GGML_TYPE_F32, g.head_dim, T, g.kv_heads);
        make_port("new_value", GGML_TYPE_F32, g.head_dim, T, g.kv_heads);
        make_port("scores", GGML_TYPE_F32, L, T, g.heads);
        make_port("probs", GGML_TYPE_F32, L, T, g.heads);
        make_port("attended", GGML_TYPE_F32, g.q_dim(), T, 1);
        make_port("attention_hidden", GGML_TYPE_F32, g.hidden, T, 1);
        make_port("hidden_output", GGML_TYPE_F32, g.hidden, T, 1);
        make_port("normed", GGML_TYPE_F32, g.hidden, T, 1);
        make_port("moe_probs", GGML_TYPE_F32, c.experts > 0 ? c.experts : 1, T, 1);
        make_port("expert_out", GGML_TYPE_F32, g.hidden, T, 1);

        KVScatter sc;
        sc.koff = ggml_new_tensor_1d(s.resident_cpu, GGML_TYPE_I64, T);
        sc.voff = ggml_new_tensor_1d(s.resident_cpu, GGML_TYPE_I64, T * g.head_dim * g.kv_heads);
        sc.positions = ggml_new_tensor_1d(s.resident_cpu, GGML_TYPE_I32, T);
        ggml_set_name(sc.positions, "positions");
        if (s.gpu) {
            sc.koff_gpu = ggml_new_tensor_1d(s.resident_gpu, GGML_TYPE_I64, T);
            sc.voff_gpu = ggml_new_tensor_1d(s.resident_gpu, GGML_TYPE_I64,
                                             T * g.head_dim * g.kv_heads);
            sc.positions_gpu = ggml_new_tensor_1d(s.resident_gpu, GGML_TYPE_I32, T);
            ggml_set_name(sc.positions_gpu, "positions");
        }
        s.scatter[key] = sc;
    }

    // KV caches: K is [head_dim, L, kv_heads], V is [L, head_dim, kv_heads].  One cache
    // for the whole execution, sized for the longest chain any node reaches into: that is
    // what lets a decode round read the keys its prefill round wrote.
    const ggml_type kv_type = Geometry::dtype_of(dtype_of_role(c, "kv", "f16"));
    s.kcache_cpu = ggml_new_tensor_3d(s.resident_cpu, kv_type, g.head_dim, max_L, g.kv_heads);
    s.vcache_cpu = ggml_new_tensor_3d(s.resident_cpu, kv_type, max_L, g.head_dim, g.kv_heads);
    ggml_set_name(s.kcache_cpu, "key_cache");
    ggml_set_name(s.vcache_cpu, "value_cache");
    if (s.gpu) {
        s.kcache_gpu = ggml_new_tensor_3d(s.resident_gpu, kv_type, g.head_dim, max_L, g.kv_heads);
        s.vcache_gpu = ggml_new_tensor_3d(s.resident_gpu, kv_type, max_L, g.head_dim, g.kv_heads);
        ggml_set_name(s.kcache_gpu, "key_cache");
        ggml_set_name(s.vcache_gpu, "value_cache");
    }
    // staging pool for copy nodes: one flat byte tensor on each side
    int64_t max_copy = 1;
    for (const auto & n : c.nodes) if (n.op == "copy") max_copy = std::max(max_copy, n.num("bytes", 1));
    s.stage_bytes = max_copy;
    s.stage_host = ggml_new_tensor_1d(s.resident_cpu, GGML_TYPE_I8, max_copy);
    if (s.gpu) s.stage_dev = ggml_new_tensor_1d(s.resident_gpu, GGML_TYPE_I8, max_copy);
    if (c.ssd_bytes > 0) {
        s.ssd_stage_host = ggml_new_tensor_1d(s.resident_cpu, GGML_TYPE_I8, max_copy);
        if (s.gpu) s.ssd_stage_dev = ggml_new_tensor_1d(s.resident_gpu, GGML_TYPE_I8, max_copy);
    }

    s.buf_cpu = ggml_backend_alloc_ctx_tensors(s.resident_cpu, s.cpu);
    if (!s.buf_cpu) throw std::runtime_error("resident cpu buffer allocation failed");
    if (s.gpu) {
        s.buf_gpu = ggml_backend_alloc_ctx_tensors(s.resident_gpu, s.gpu);
        if (!s.buf_gpu) throw std::runtime_error("resident gpu buffer allocation failed");
    }

    // zero the caches and ports so a reader that forgot a producer is not lucky
    for (auto & set : s.ports) {
        for (auto & kv : set.second) {
            std::vector<uint8_t> zero(ggml_nbytes(kv.second.cpu), 0);
            ggml_backend_tensor_set(kv.second.cpu, zero.data(), 0, zero.size());
            if (kv.second.gpu) ggml_backend_tensor_set(kv.second.gpu, zero.data(), 0, zero.size());
        }
    }
    {
        std::vector<uint8_t> zero(ggml_nbytes(s.kcache_cpu), 0);
        ggml_backend_tensor_set(s.kcache_cpu, zero.data(), 0, zero.size());
        zero.assign(ggml_nbytes(s.vcache_cpu), 0);
        ggml_backend_tensor_set(s.vcache_cpu, zero.data(), 0, zero.size());
        if (s.gpu) {
            zero.assign(ggml_nbytes(s.kcache_gpu), 0);
            ggml_backend_tensor_set(s.kcache_gpu, zero.data(), 0, zero.size());
            zero.assign(ggml_nbytes(s.vcache_gpu), 0);
            ggml_backend_tensor_set(s.vcache_gpu, zero.data(), 0, zero.size());
        }
    }
    // scatter offsets, one set per shape: the value offsets are laid out with the shape's
    // own total length L, so a prefill round and a decode round cannot share one table
    for (const auto & key : s.shapes) {
        const int64_t T = key.first, H = key.second, L = T + H;
        KVScatter & sc = s.scatter.at(key);
        std::vector<int64_t> ko(static_cast<size_t>(T));
        std::vector<int64_t> vo(static_cast<size_t>(T) * g.head_dim * g.kv_heads);
        for (int64_t t = 0; t < T; ++t) ko[size_t(t)] = H + t;
        for (int64_t h = 0; h < g.kv_heads; ++h)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t j = 0; j < g.head_dim; ++j)
                    vo[size_t(j + g.head_dim * (t + T * h))] = (H + t) + L * (j + g.head_dim * h);
        ggml_backend_tensor_set(sc.koff, ko.data(), 0, ko.size() * sizeof(int64_t));
        ggml_backend_tensor_set(sc.voff, vo.data(), 0, vo.size() * sizeof(int64_t));
        std::vector<int32_t> pos(static_cast<size_t>(T));
        for (int64_t t = 0; t < T; ++t) pos[size_t(t)] = static_cast<int32_t>(H + t);
        ggml_backend_tensor_set(sc.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        if (s.gpu) {
            ggml_backend_tensor_set(sc.koff_gpu, ko.data(), 0, ko.size() * sizeof(int64_t));
            ggml_backend_tensor_set(sc.voff_gpu, vo.data(), 0, vo.size() * sizeof(int64_t));
            ggml_backend_tensor_set(sc.positions_gpu, pos.data(), 0,
                                    pos.size() * sizeof(int32_t));
        }
    }
    // a deterministic activation, one per shape
    for (auto & set : s.ports) {
        auto & x = set.second.at("x");
        std::vector<uint8_t> data(ggml_nbytes(x.cpu));
        fill_bytes(data.data(), data.size(), 11, GGML_TYPE_F32);
        ggml_backend_tensor_set(x.cpu, data.data(), 0, data.size());
        if (x.gpu) ggml_backend_tensor_set(x.gpu, data.data(), 0, data.size());
        x.writer = s.gpu ? 2 : 1;
    }

    // weights
    s.weights_ctx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
    auto add2 = [&](const std::string & role, ggml_type type, int64_t cols, int64_t rows) {
        auto * t = ggml_new_tensor_2d(s.weights_ctx, type, cols, rows);
        ggml_set_name(t, role.c_str());
        s.w_cpu[role] = t;
    };
    auto add1 = [&](const std::string & role, int64_t n) {
        auto * t = ggml_new_tensor_1d(s.weights_ctx, GGML_TYPE_F32, n);
        ggml_set_name(t, role.c_str());
        s.w_cpu[role] = t;
    };
    add1("input_norm", g.hidden);
    add1("post_attention_norm", g.hidden);
    add1("final_norm", g.hidden);
    add1("q_norm", g.head_dim);
    add1("k_norm", g.head_dim);
    add2("q", Geometry::dtype_of(dtype_of_role(c, "q", "q4_k")), g.hidden, g.q_dim());
    add2("k", Geometry::dtype_of(dtype_of_role(c, "k", "q4_k")), g.hidden, g.kv_dim());
    add2("v", Geometry::dtype_of(dtype_of_role(c, "v", "q6_k")), g.hidden, g.kv_dim());
    add2("output", Geometry::dtype_of(dtype_of_role(c, "o", "q4_k")), g.q_dim(), g.hidden);
    if (c.experts > 0) {
        // top_k expert matrices, not one: the executor must pay what the IR declares and
        // what the frozen `moe_ffn` price was measured on (see routed_experts)
        const int64_t slots = c.top_k > 0 ? c.top_k : 1;
        add2("router", GGML_TYPE_F32, g.hidden, c.experts);
        add2("gate_exps", Geometry::dtype_of(dtype_of_role(c, "gate", "q4_k")), g.hidden,
             g.intermediate * slots);
        add2("up_exps", Geometry::dtype_of(dtype_of_role(c, "up", "q4_k")), g.hidden,
             g.intermediate * slots);
        add2("down_exps", Geometry::dtype_of(dtype_of_role(c, "down", "q6_k")), g.intermediate,
             g.hidden * slots);
    } else {
        add2("gate", Geometry::dtype_of(dtype_of_role(c, "gate", "q4_k")), g.hidden, g.intermediate);
        add2("up", Geometry::dtype_of(dtype_of_role(c, "up", "q4_k")), g.hidden, g.intermediate);
        add2("down", Geometry::dtype_of(dtype_of_role(c, "down", "q6_k")), g.intermediate, g.hidden);
    }
    s.weights_cpu_buf = ggml_backend_alloc_ctx_tensors(s.weights_ctx, s.cpu);
    if (!s.weights_cpu_buf) throw std::runtime_error("weight cpu buffer allocation failed");
    std::vector<uint8_t> tmp;
    uint32_t seed = 17;
    for (auto & kv : s.w_cpu) {
        tmp.assign(ggml_nbytes(kv.second), 0);
        fill_bytes(tmp.data(), tmp.size(), seed++, kv.second->type);
        ggml_backend_tensor_set(kv.second, tmp.data(), 0, tmp.size());
    }
    if (s.gpu) {
        ggml_context * wg = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
        for (auto & kv : s.w_cpu) {
            auto * t = ggml_new_tensor_4d(wg, kv.second->type, kv.second->ne[0], kv.second->ne[1],
                                          kv.second->ne[2], kv.second->ne[3]);
            ggml_set_name(t, kv.first.c_str());
            s.w_gpu[kv.first] = t;
        }
        s.weights_gpu_buf = ggml_backend_alloc_ctx_tensors(wg, s.gpu);
        if (!s.weights_gpu_buf) throw std::runtime_error("weight gpu buffer allocation failed");
        for (auto & kv : s.w_cpu) {
            tmp.assign(ggml_nbytes(kv.second), 0);
            ggml_backend_tensor_get(kv.second, tmp.data(), 0, tmp.size());
            ggml_backend_tensor_set(s.w_gpu[kv.first], tmp.data(), 0, tmp.size());
        }
        static std::vector<ggml_context *> keep;   // freed with the process; one run per process
        keep.push_back(wg);
    }
}

// ---------------------------------------------------------------- node graphs

struct Built {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    size_t shape_index = 0;                 // which (tokens, history) this graph was built at
    std::vector<std::string> writes;
};

static ggml_tensor * P(Session & s, const std::string & name, bool cuda) {
    const auto & set = s.ports.at(s.shape());   // the port set of the shape being built
    auto it = set.find(name);
    if (it == set.end()) throw std::runtime_error("unknown port " + name);
    ggml_tensor * t = cuda ? it->second.gpu : it->second.cpu;
    if (!t) throw std::runtime_error("port " + name + " has no device copy");
    return t;
}

static ggml_tensor * W(Session & s, const std::string & role, bool cuda) {
    auto & m = cuda ? s.w_gpu : s.w_cpu;
    auto it = m.find(role);
    if (it == m.end()) throw std::runtime_error("weight role not allocated: " + role);
    return it->second;
}

static ggml_tensor * KVK(Session & s, bool cuda) {
    ggml_tensor * t = cuda ? s.kcache_gpu : s.kcache_cpu;
    if (!t) throw std::runtime_error("no KV cache on this machine");
    return t;
}
static ggml_tensor * KVV(Session & s, bool cuda) {
    ggml_tensor * t = cuda ? s.vcache_gpu : s.vcache_cpu;
    if (!t) throw std::runtime_error("no KV cache on this machine");
    return t;
}

// Projection stage, exactly the operation list the frozen table prices:
// input RMSNorm, Q/K/V projections, per-head Q/K RMSNorm, RoPE, materialize.
static void build_projection(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                             std::vector<std::string> & writes) {
    const auto & g = s.g;
    const int64_t T = s.T();
    auto * x = P(s, "x", cuda);
    const KVScatter & sc = s.scatter.at(s.shape());
    auto * pos = cuda ? sc.positions_gpu : sc.positions;
    auto * n = ggml_mul(ctx, ggml_rms_norm(ctx, x, float(g.norm_eps)), W(s, "input_norm", cuda));
    auto qkv = [&](const char * role, int64_t heads, bool rotary) {
        auto * projected = ggml_mul_mat(ctx, W(s, role, cuda), n);
        auto * v = ggml_reshape_3d(ctx, projected, g.head_dim, heads, T);
        if (rotary) {
            if (g.qk_norm == "rms")
                v = ggml_mul(ctx, ggml_rms_norm(ctx, v, float(g.norm_eps)),
                             W(s, role[0] == 'q' ? "q_norm" : "k_norm", cuda));
            v = ggml_rope_ext(ctx, v, pos, nullptr, static_cast<int>(g.rope_dims),
                              GGML_ROPE_TYPE_NORMAL, 0, float(g.rope_base), 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);
        }
        return ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    };
    auto * q = ggml_cpy(ctx, qkv("q", g.heads, true), P(s, "query", cuda));
    auto * k = ggml_cpy(ctx, qkv("k", g.kv_heads, true), P(s, "new_key", cuda));
    auto * v = ggml_cpy(ctx, qkv("v", g.kv_heads, false), P(s, "new_value", cuda));
    ggml_build_forward_expand(gr, q);
    ggml_build_forward_expand(gr, k);
    ggml_build_forward_expand(gr, v);
    writes = {"query", "new_key", "new_value"};
}

static void build_attn_qk(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                          std::vector<std::string> & writes) {
    const auto & g = s.g;
    const int64_t L = s.L();
    const KVScatter & sc = s.scatter.at(s.shape());
    auto * ko = cuda ? sc.koff_gpu : sc.koff;
    // The cache is sized for the longest chain in the case; this node sees exactly its own
    // L slots.  Without the window a prefill round inside a pd case would attend over its
    // decode round's slot too, and its scores would not have the shape the frozen
    // attention price describes.
    auto * kv = ggml_view_3d(ctx, KVK(s, cuda), g.head_dim, L, g.kv_heads,
                             KVK(s, cuda)->nb[1], KVK(s, cuda)->nb[2], 0);
    auto * kr = ggml_set_rows(ctx, kv, P(s, "new_key", cuda), ko);
    auto * scores = ggml_mul_mat(ctx, kr, P(s, "query", cuda));
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(float(g.head_dim)));
    scores = ggml_diag_mask_inf(ctx, scores, static_cast<int>(s.H()));
    auto * out = ggml_cpy(ctx, scores, P(s, "scores", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"scores"};
}

static void build_attn_softmax(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                               std::vector<std::string> & writes) {
    auto * out = ggml_cpy(ctx, ggml_soft_max(ctx, P(s, "scores", cuda)), P(s, "probs", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"probs"};
}

static void build_attn_pv(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                          std::vector<std::string> & writes) {
    const auto & g = s.g;
    const int64_t T = s.T(), L = s.L();
    const KVScatter & sc = s.scatter.at(s.shape());
    auto * vo = cuda ? sc.voff_gpu : sc.voff;
    auto * vc = KVV(s, cuda);
    // The cache is sized for the longest chain any node reaches into, which can be longer
    // than this node's own L (a prefill round inside a pd case sits below its decode
    // round).  Only the first L*head_dim*kv_heads slots belong to this shape, and this
    // shape's own `vo` offsets are laid out inside exactly that many slots.
    auto * vflat = ggml_reshape_2d(ctx, ggml_view_1d(ctx, vc, L * g.head_dim * g.kv_heads, 0),
                                   1, L * g.head_dim * g.kv_heads);
    auto * nv = P(s, "new_value", cuda);
    auto * newflat = ggml_reshape_2d(ctx, nv, 1, static_cast<int64_t>(ggml_nelements(nv)));
    auto * vr = ggml_reshape_3d(ctx, ggml_set_rows(ctx, vflat, newflat, vo), L, g.head_dim,
                                g.kv_heads);
    auto * attended = ggml_mul_mat(ctx, vr, P(s, "probs", cuda));
    auto * merged = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), g.q_dim(), T);
    auto * out = ggml_cpy(ctx, merged, P(s, "attended", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"attended"};
}

static void build_attn_output(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                              std::vector<std::string> & writes) {
    auto * hidden = ggml_add(ctx, ggml_mul_mat(ctx, W(s, "output", cuda), P(s, "attended", cuda)),
                             P(s, "x", cuda));
    auto * out = ggml_cpy(ctx, hidden, P(s, "attention_hidden", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"attention_hidden"};
}

static void build_attention_block(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda) {
    const auto & g = s.g;
    const int64_t T = s.T(), L = s.L();
    const KVScatter & sc = s.scatter.at(s.shape());
    auto * ko = cuda ? sc.koff_gpu : sc.koff;
    auto * vo = cuda ? sc.voff_gpu : sc.voff;
    auto * kv = ggml_view_3d(ctx, KVK(s, cuda), g.head_dim, L, g.kv_heads,
                             KVK(s, cuda)->nb[1], KVK(s, cuda)->nb[2], 0);
    auto * kr = ggml_set_rows(ctx, kv, P(s, "new_key", cuda), ko);
    auto * scores = ggml_mul_mat(ctx, kr, P(s, "query", cuda));
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(float(g.head_dim)));
    scores = ggml_diag_mask_inf(ctx, scores, static_cast<int>(s.H()));
    auto * probs = ggml_soft_max(ctx, scores);
    auto * vc = KVV(s, cuda);
    // The cache is sized for the longest chain any node reaches into, which can be longer
    // than this node's own L (a prefill round inside a pd case sits below its decode
    // round).  Only the first L*head_dim*kv_heads slots belong to this shape, and this
    // shape's own `vo` offsets are laid out inside exactly that many slots.
    auto * vflat = ggml_reshape_2d(ctx, ggml_view_1d(ctx, vc, L * g.head_dim * g.kv_heads, 0),
                                   1, L * g.head_dim * g.kv_heads);
    auto * nv = P(s, "new_value", cuda);
    auto * newflat = ggml_reshape_2d(ctx, nv, 1, static_cast<int64_t>(ggml_nelements(nv)));
    auto * vr = ggml_reshape_3d(ctx, ggml_set_rows(ctx, vflat, newflat, vo), L, g.head_dim,
                                g.kv_heads);
    auto * attended = ggml_mul_mat(ctx, vr, probs);
    auto * merged = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), g.q_dim(), T);
    auto * hidden = ggml_add(ctx, ggml_mul_mat(ctx, W(s, "output", cuda), merged),
                             P(s, "x", cuda));
    auto * out = ggml_cpy(ctx, hidden, P(s, "attention_hidden", cuda));
    ggml_build_forward_expand(gr, out);
}

static void build_dense_ffn(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                            std::vector<std::string> & writes) {
    auto * h = P(s, "attention_hidden", cuda);
    auto * n = ggml_mul(ctx, ggml_rms_norm(ctx, h, float(s.g.norm_eps)),
                        W(s, "post_attention_norm", cuda));
    auto * gate = ggml_mul_mat(ctx, W(s, "gate", cuda), n);
    auto * up = ggml_mul_mat(ctx, W(s, "up", cuda), n);
    auto * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    auto * down = ggml_mul_mat(ctx, W(s, "down", cuda), act);
    auto * out = ggml_cpy(ctx, ggml_add(ctx, down, h), P(s, "hidden_output", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"hidden_output"};
}

static void build_router(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                         std::vector<std::string> & writes) {
    auto * h = P(s, "attention_hidden", cuda);
    auto * n = ggml_mul(ctx, ggml_rms_norm(ctx, h, float(s.g.norm_eps)),
                        W(s, "post_attention_norm", cuda));
    auto * ncpy = ggml_cpy(ctx, n, P(s, "normed", cuda));
    auto * logits = ggml_mul_mat(ctx, W(s, "router", cuda), n);
    auto * p = ggml_cpy(ctx, ggml_soft_max(ctx, logits), P(s, "moe_probs", cuda));
    ggml_build_forward_expand(gr, ncpy);
    ggml_build_forward_expand(gr, p);
    writes = {"normed", "moe_probs"};
}

// One expert's slice of a `*_exps` weight: the tensor holds `slots` expert matrices stacked
// along the second dimension, so slot i is rows [i*rows, (i+1)*rows).
static ggml_tensor * expert_slice(ggml_context * ctx, ggml_tensor * w, int64_t slot, int64_t rows) {
    return ggml_view_2d(ctx, w, w->ne[0], rows, w->nb[1], size_t(slot) * size_t(rows) * size_t(w->nb[1]));
}

static ggml_tensor * expert_path(Session & s, ggml_context * ctx, bool cuda, const char * tag,
                                 int64_t slot) {
    auto * n = P(s, "normed", cuda);
    const int64_t ffn = s.g.intermediate;
    auto * gate = ggml_mul_mat(ctx, expert_slice(ctx, W(s, std::string(tag) + "gate_exps", cuda),
                                                 slot, ffn), n);
    auto * up = ggml_mul_mat(ctx, expert_slice(ctx, W(s, std::string(tag) + "up_exps", cuda),
                                               slot, ffn), n);
    auto * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat(ctx, expert_slice(ctx, W(s, std::string(tag) + "down_exps", cuda),
                                          slot, s.g.hidden), act);
}

// The case header declares `top_k`, and the frozen `moe_ffn` price was measured on the
// model's real routed set.  Executing one expert set would under-execute by exactly top_k.
static ggml_tensor * routed_experts(Session & s, ggml_context * ctx, bool cuda, const char * tag) {
    const int64_t slots = s.c.top_k > 0 ? s.c.top_k : 1;
    ggml_tensor * sum = expert_path(s, ctx, cuda, tag, 0);
    for (int64_t slot = 1; slot < slots; ++slot)
        sum = ggml_add(ctx, sum, expert_path(s, ctx, cuda, tag, slot));
    return sum;
}

static void build_expert(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                         std::vector<std::string> & writes) {
    auto * out = ggml_cpy(ctx, routed_experts(s, ctx, cuda, ""), P(s, "expert_out", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"expert_out"};
}

static void build_combine(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda,
                          std::vector<std::string> & writes) {
    auto * out = ggml_cpy(ctx,
                          ggml_add(ctx, P(s, "expert_out", cuda), P(s, "attention_hidden", cuda)),
                          P(s, "hidden_output", cuda));
    ggml_build_forward_expand(gr, out);
    writes = {"hidden_output"};
}

static void build_moe_ffn(Session & s, ggml_context * ctx, ggml_cgraph * gr, bool cuda) {
    auto * h = P(s, "attention_hidden", cuda);
    auto * n = ggml_mul(ctx, ggml_rms_norm(ctx, h, float(s.g.norm_eps)),
                        W(s, "post_attention_norm", cuda));
    auto * logits = ggml_mul_mat(ctx, W(s, "router", cuda), n);
    ggml_soft_max(ctx, logits);   // routing probabilities; the priced weight of the
                                  // expert path is the routed top_k, executed in full
    auto * expert = routed_experts(s, ctx, cuda, "");
    auto * out = ggml_cpy(ctx, ggml_add(ctx, expert, h), P(s, "hidden_output", cuda));
    ggml_build_forward_expand(gr, out);
}

static void build_node_graph(Session & s, const Node & n, bool cuda, ggml_backend_t backend,
                             Built & b, int64_t & build_ns) {
    const int64_t t0 = now_ns();
    // The node's own shape selects its port set and its scatter offsets; the builders read
    // it through s.T()/s.H()/s.L() instead of one case-wide shape.  Building is single
    // threaded, so this cursor never races with the executor.
    const std::pair<int64_t, int64_t> key{n.num("t", s.c.tokens), n.num("h", s.c.history)};
    const auto it = std::find(s.shapes.begin(), s.shapes.end(), key);
    if (it == s.shapes.end()) throw std::runtime_error("node shape was never allocated");
    s.cur_shape = static_cast<size_t>(it - s.shapes.begin());
    b.shape_index = s.cur_shape;
    b.ctx = ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false),
                       nullptr, true});
    if (!b.ctx) throw std::runtime_error("node context allocation failed");
    b.graph = ggml_new_graph_custom(b.ctx, 256, false);

    if (n.op == "proj" || n.op == "projection") build_projection(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "attn" || n.op == "attention") build_attention_block(s, b.ctx, b.graph, cuda);
    else if (n.op == "attn.qk") build_attn_qk(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "attn.softmax") build_attn_softmax(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "attn.pv") build_attn_pv(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "attn.output") build_attn_output(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "ffn" || n.op == "dense_ffn") build_dense_ffn(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "router") build_router(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "expert" || n.op.rfind("expert.", 0) == 0)
        build_expert(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "combine") build_combine(s, b.ctx, b.graph, cuda, b.writes);
    else if (n.op == "moe_ffn") build_moe_ffn(s, b.ctx, b.graph, cuda);
    else throw std::runtime_error("unsupported op " + n.op);

    b.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!b.alloc || !ggml_gallocr_alloc_graph(b.alloc, b.graph))
        throw std::runtime_error("graph allocation failed for op " + n.op);
    build_ns = now_ns() - t0;
}

// ---------------------------------------------------------------- scheduler

struct NodeRun {
    int64_t begin_ns = 0, end_ns = 0, build_ns = 0, bytes = 0;
    int64_t evicted = 0;        // bytes this node forced out of a tier to make room
    int64_t moved = 0;          // bytes actually moved (0 under uma)
    bool ok = false;
    std::string status;
};

struct Schedule {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<char> done;
    int remaining = 0;
    int64_t start_ns = 0, end_ns = 0;
};

static int machine_id(const std::string & m) {
    if (m == "cpu") return 1;
    if (m == "gpu") return 2;
    return 3;
}

static void run_machine(Session & s, Schedule & sc, const std::string & machine,
                        std::vector<NodeRun> & runs,
                        const std::vector<Built> & prepared,
                        const std::vector<int64_t> & built_build_ns) {
    const bool cuda = machine == "gpu";
    ggml_backend_t backend = cuda ? s.gpu : s.cpu;
    while (true) {
        int pick = -1;
        {
            std::unique_lock<std::mutex> lock(sc.mu);
            sc.cv.wait(lock, [&] {
                if (sc.remaining == 0) return true;
                for (const auto & n : s.c.nodes) {
                    if (n.machine != machine || sc.done[size_t(n.index)]) continue;
                    bool ready = true;
                    for (int d : n.deps) if (!sc.done[size_t(d)]) { ready = false; break; }
                    if (ready) { pick = n.index; return true; }
                }
                return false;
            });
            if (pick < 0) { sc.cv.notify_all(); return; }
        }
        const Node & n = s.c.nodes[size_t(pick)];
        NodeRun & r = runs[size_t(pick)];
        if (s.verbose) std::fprintf(stderr, "[%s] node %d op=%s\n", machine.c_str(), n.index, n.op.c_str());
        int64_t build = 0;
        try {
            if (n.op == "arrival" || n.op == "request_arrival") {
                r.begin_ns = now_ns();
                r.end_ns = r.begin_ns;
                r.ok = true;
                r.status = "arrival";
            } else if (n.op == "copy") {
                const int64_t bytes = n.num("bytes", 0);
                const std::string dir = n.str("dir", "cpu>gpu");
                const std::string key = n.str("tensor", n.id);
                const bool to_gpu = dir == "cpu>gpu";
                const size_t arrow = dir.find('>');
                const std::string src = arrow == std::string::npos ? "cpu" : dir.substr(0, arrow);
                const std::string dst = arrow == std::string::npos ? "gpu" : dir.substr(arrow + 1);
                // The ledger is only consulted when the case declares a budget: a case that
                // does not is the plain two-tier executor and must time exactly as before.
                const bool ledger_on = s.c.vram_bytes || s.c.dram_bytes || s.c.ssd_bytes;
                if (s.c.uma && (dir == "cpu>gpu" || dir == "gpu>cpu")) {
                    // One address space.  The tensor is already where it is going, so the
                    // node declares its bytes and moves none of them -- what an integrated
                    // GPU does, and the whole reason an iGPU can hold a model that will not
                    // fit an accelerator with its own memory.
                    const int64_t b0 = now_ns();
                    const int64_t b1 = now_ns();
                    r.begin_ns = b0; r.end_ns = b1; r.bytes = bytes; r.moved = 0;
                    r.ok = true;
                    r.status = "uma_alias";
                } else if (n.machine == "ssd") {
                    const int64_t b0 = now_ns();
                    const Admission adm = acquire(s, dev_of(dst), key, bytes, true);
                    if (src == "ssd") {
                        const int64_t off = resident_offset(s, key);
                        if (off < 0)
                            throw std::runtime_error("ssd read of a tensor never written: " + key);
                        move_ssd(s, false, off, bytes);
                    } else if (src == "gpu") {
                        move_staging(s, false, bytes, true);
                    }
                    if (dst == "gpu") {
                        move_staging(s, true, bytes, true);
                    } else if (dst == "ssd") {
                        move_ssd(s, true, adm.offset, bytes);
                    }
                    const int64_t b1 = now_ns();
                    r.begin_ns = b0; r.end_ns = b1; r.bytes = bytes;
                    r.moved = bytes; r.evicted = adm.evicted;
                    r.ok = true;
                    r.status = "copy";
                } else {
                if (bytes > s.stage_bytes) throw std::runtime_error("copy larger than staging pool");
                ggml_context * vctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
                auto * from = ggml_view_1d(vctx, to_gpu ? s.stage_host : s.stage_dev, bytes, 0);
                auto * to = ggml_view_1d(vctx, to_gpu ? s.stage_dev : s.stage_host, bytes, 0);
                // a view of a device buffer is created with a null buffer on purpose;
                // ggml_backend_view_init derives it from the parent again
                if (!from->buffer && ggml_backend_view_init(from) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("stage view init failed");
                if (!to->buffer && ggml_backend_view_init(to) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("stage view init failed");
                if (s.gpu) ggml_backend_synchronize(s.gpu);
                ggml_backend_synchronize(s.cpu);
                const int64_t b0 = now_ns();
                // admission happens inside the timed window: an eviction is work the copy
                // engine does, so a plan that does not fit pays for it here
                if (ledger_on) r.evicted = acquire(s, dev_of(dst), key, bytes).evicted;
                ggml_backend_tensor_copy(from, to);
                if (s.gpu) ggml_backend_synchronize(s.gpu);
                ggml_backend_synchronize(s.cpu);
                const int64_t b1 = now_ns();
                ggml_free(vctx);
                r.begin_ns = b0; r.end_ns = b1; r.bytes = bytes; r.moved = bytes;
                r.ok = true;
                r.status = "copy";
                }
            } else {
                // Graphs are built once, ahead of the timed region, exactly as a compiler
                // would emit them; graph construction is the compiler's cost and is
                // reported as build_ns, never as part of the makespan.
                const Built & built = prepared[size_t(pick)];
                build = built_build_ns[size_t(pick)];
                ggml_backend_synchronize(backend);
                const int64_t b0 = now_ns();
                ggml_backend_graph_compute(backend, built.graph);
                ggml_backend_synchronize(backend);
                const int64_t b1 = now_ns();
                for (const auto & w : built.writes)
                    s.ports.at(s.shapes[built.shape_index]).at(w).writer = cuda ? 2 : 1;
                r.begin_ns = b0; r.end_ns = b1; r.build_ns = build;
                r.ok = true;
                r.status = "ok";
            }
        } catch (const std::exception & e) {
            r.ok = false;
            r.status = e.what();
            r.end_ns = now_ns();
        }
        {
            std::lock_guard<std::mutex> lock(sc.mu);
            sc.done[size_t(pick)] = 1;
            --sc.remaining;
            sc.end_ns = std::max(sc.end_ns, r.end_ns);
        }
        sc.cv.notify_all();
    }
}

static void prepare_graphs(Session & s, std::vector<Built> & prepared,
                           std::vector<int64_t> & build_ns) {
    const size_t n = s.c.nodes.size();
    prepared.assign(n, Built());
    build_ns.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const Node & node = s.c.nodes[i];
        if (node.op == "arrival" || node.op == "request_arrival" || node.op == "copy") continue;
        const bool cuda = node.machine == "gpu";
        ggml_backend_t backend = cuda ? s.gpu : s.cpu;
        if (!backend) throw std::runtime_error("machine " + node.machine + " has no backend");
        build_node_graph(s, node, cuda, backend, prepared[i], build_ns[i]);
    }
}

static void release_graphs(std::vector<Built> & prepared) {
    for (auto & b : prepared) {
        if (b.alloc) ggml_gallocr_free(b.alloc);
        if (b.ctx) ggml_free(b.ctx);
        b.alloc = nullptr;
        b.ctx = nullptr;
    }
}

static bool run_schedule(Session & s, std::vector<NodeRun> & runs, int64_t & wall_ns,
                         const std::vector<Built> & prepared,
                         const std::vector<int64_t> & built_build_ns) {
    runs.assign(s.c.nodes.size(), NodeRun());
    s.ledger.cap[DEV_CPU] = s.c.dram_bytes;
    s.ledger.cap[DEV_GPU] = s.c.vram_bytes;
    s.ledger.cap[DEV_SSD] = s.c.ssd_bytes;
    s.ledger.reset_state();
    Schedule sc;
    sc.done.assign(s.c.nodes.size(), 0);
    sc.remaining = static_cast<int>(s.c.nodes.size());
    sc.start_ns = now_ns();
    sc.end_ns = sc.start_ns;
    const bool has_gpu = std::any_of(s.c.nodes.begin(), s.c.nodes.end(),
                                     [](const Node & n) { return n.machine == "gpu"; });
    const bool has_pcie = std::any_of(s.c.nodes.begin(), s.c.nodes.end(),
                                      [](const Node & n) { return n.machine == "pcie"; });
    // The third tier is its own machine with its own worker: an SSD leg and a PCIe leg are
    // different hardware and must be able to overlap, exactly as the two-tier path already
    // lets cpu and gpu work overlap.
    const bool has_ssd = std::any_of(s.c.nodes.begin(), s.c.nodes.end(),
                                     [](const Node & n) { return n.machine == "ssd"; });
    if (has_ssd && !s.ssd_handle) throw std::runtime_error("this case needs a third tier; pass --ssd-dir");
    std::vector<std::thread> threads;
    threads.emplace_back(run_machine, std::ref(s), std::ref(sc), "cpu", std::ref(runs),
                         std::cref(prepared), std::cref(built_build_ns));
    if (has_gpu) threads.emplace_back(run_machine, std::ref(s), std::ref(sc), "gpu", std::ref(runs),
                                      std::cref(prepared), std::cref(built_build_ns));
    if (has_pcie) threads.emplace_back(run_machine, std::ref(s), std::ref(sc), "pcie", std::ref(runs),
                                       std::cref(prepared), std::cref(built_build_ns));
    if (has_ssd) threads.emplace_back(run_machine, std::ref(s), std::ref(sc), "ssd", std::ref(runs),
                                      std::cref(prepared), std::cref(built_build_ns));
    for (auto & t : threads) t.join();
    wall_ns = sc.end_ns - sc.start_ns;
    for (const auto & r : runs) if (!r.ok) return false;
    return true;
}

// ---------------------------------------------------------------- gate 2 instrument

static int64_t empty_overhead(ggml_backend_t backend, int n_ops) {
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
                                    nullptr, true});
    auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_set_input(t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> ones(1024, 1.0f);
    ggml_backend_tensor_set(t, ones.data(), 0, ggml_nbytes(t));
    ggml_cgraph * gr = ggml_new_graph_custom(ctx, 64, false);
    ggml_tensor * out = t;
    for (int i = 0; i < n_ops; ++i) {
        out = ggml_scale(ctx, out, 1.0f);
        ggml_set_output(out);
    }
    ggml_build_forward_expand(gr, out);
    ggml_backend_synchronize(backend);
    const int64_t a = now_ns();
    ggml_backend_graph_compute(backend, gr);
    ggml_backend_synchronize(backend);
    const int64_t b = now_ns();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return b - a;
}

}  // namespace

int main(int argc, char ** argv) {
    Options o;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char * w) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + w);
                return argv[++i];
            };
            if (a == "--case") o.case_path = next("--case");
            else if (a == "--out") o.out_path = next("--out");
            else if (a == "--replicas") o.replicas = std::stoi(next("--replicas"));
            else if (a == "--threads") o.threads = std::stoi(next("--threads"));
            else if (a == "--cuda-device") o.cuda_device = std::stoi(next("--cuda-device"));
            else if (a == "--empty-overhead") o.empty_overhead = true;
            else if (a == "--verbose") o.verbose = true;
            else if (a == "--no-gpu") o.no_gpu = true;
            else if (a == "--ssd-dir") o.ssd_dir = next("--ssd-dir");
            else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
        }
        if (o.empty_overhead) {
            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, o.threads);
            std::printf("{\"cpu_2op_ns\": %lld, \"cpu_10op_ns\": %lld",
                        (long long) empty_overhead(cpu, 2), (long long) empty_overhead(cpu, 10));
            ggml_backend_free(cpu);
            std::printf("}\n");
            return 0;
        }
        if (o.case_path.empty()) { std::fprintf(stderr, "--case is required\n"); return 2; }

        Session s;
        s.c = read_case(o.case_path);
        s.verbose = o.verbose;
        // the third tier is an instrument setting (one file, reused by every case), not a
        // property of the plan: the case says *how much* disk it wants, the caller says
        // where the disk is
        if (s.c.ssd_bytes > 0) {
            if (o.ssd_dir.empty())
                throw std::runtime_error("case declares ssd_bytes but --ssd-dir is missing");
            open_ssd(s, o.ssd_dir);
        }
        if (o.no_gpu) s.c.gpu = false;
        s.c.cuda_device = o.cuda_device;
        if (o.threads > 0) s.c.threads = o.threads;
        // geometry_* lines are parsed by the instantiator into dtype_* only; the
        // numeric geometry lives in the same file under geometry_<field>.
        {
            std::ifstream in(o.case_path);
            std::string line;
            while (std::getline(in, line)) {
                auto hash = line.find('#');
                if (hash != std::string::npos) line = line.substr(0, hash);
                auto w = split(line, ' ');
                if (w.size() != 2) continue;
                const auto & k = w[0];
                if (k == "geometry_hidden") s.g.hidden = std::stoll(w[1]);
                else if (k == "geometry_intermediate") s.g.intermediate = std::stoll(w[1]);
                else if (k == "geometry_heads") s.g.heads = std::stoll(w[1]);
                else if (k == "geometry_kv_heads") s.g.kv_heads = std::stoll(w[1]);
                else if (k == "geometry_head_dim") s.g.head_dim = std::stoll(w[1]);
                else if (k == "geometry_rope_dims") s.g.rope_dims = std::stoll(w[1]);
                else if (k == "geometry_rope_base") s.g.rope_base = std::stod(w[1]);
                else if (k == "geometry_norm_eps") s.g.norm_eps = std::stod(w[1]);
                else if (k == "geometry_qk_norm") s.g.qk_norm = w[1];
                else if (k == "geometry_attention_gate") s.g.attention_gate = w[1];
            }
        }
        init_session(s);

        std::vector<Built> prepared;
        std::vector<int64_t> built_build_ns;
        prepare_graphs(s, prepared, built_build_ns);
        std::vector<std::vector<NodeRun>> replica_runs;
        std::vector<int64_t> walls;
        bool all_ok = true;
        for (int rep = 0; rep < o.replicas; ++rep) {
            std::vector<NodeRun> runs;
            int64_t wall = 0;
            const bool ok = run_schedule(s, runs, wall, prepared, built_build_ns);
            all_ok = all_ok && ok;
            replica_runs.push_back(runs);
            walls.push_back(wall);
            for (const auto & r : runs) if (!r.ok) {
                if (o.verbose) std::fprintf(stderr, "node failed: %s\n", r.status.c_str());
                break;
            }
        }
        release_graphs(prepared);
        int64_t build_total = 0;
        for (int64_t b : built_build_ns) build_total += b;
        std::ostringstream js;
        js << "{\n  \"case\": \"" << s.c.name << "\",\n  \"model\": \"" << s.c.model
           << "\",\n  \"threads\": " << s.c.threads
           << ",\n  \"tokens\": " << s.c.tokens
           << ",\n  \"cpu_mask\": [";
        for (size_t i = 0; i < s.c.cpu_ids.size(); ++i) js << (i ? ", " : "") << s.c.cpu_ids[i];
        js << "],\n  \"thread_placement\": \""
           << (s.c.cpu_ids.empty() ? "not_requested;actual_thread_placement_unknown"
                                   : "explicit_strict_cpu_mask")
           << "\",\n  \"history\": " << s.c.history << ",\n  \"gpu\": " << (s.gpu ? "true" : "false")
           // every shape this execution actually built a graph at, so the driver can
           // check the executed shapes against the ones it priced
           << ",\n  \"shapes\": [";
        for (size_t i = 0; i < s.shapes.size(); ++i)
            js << (i ? ", " : "") << "{\"tokens\": " << s.shapes[i].first
               << ", \"history\": " << s.shapes[i].second << "}";
        js << "],\n  \"tiers\": {\"uma\": " << (s.c.uma ? "true" : "false")
           << ", \"vram_bytes\": " << s.c.vram_bytes
           << ", \"dram_bytes\": " << s.c.dram_bytes
           << ", \"ssd_bytes\": " << s.c.ssd_bytes
           << ", \"reclaim\": \"" << s.c.reclaim << "\""
           << ", \"release_order\": \"" << s.c.release_order << "\""
           << ", \"vram_peak\": " << s.ledger.peak[DEV_GPU]
           << ", \"dram_peak\": " << s.ledger.peak[DEV_CPU]
           << ", \"ssd_peak\": " << s.ledger.peak[DEV_SSD]
           << ", \"evictions\": " << s.ledger.evictions
           << ", \"evicted_bytes\": " << s.ledger.evicted_bytes << "}"
           << ",\n  \"all_ok\": " << (all_ok ? "true" : "false")
           << ",\n  \"build_ns_total\": " << build_total
           << ",\n  \"wall_ns\": [";
        for (size_t i = 0; i < walls.size(); ++i) js << (i ? ", " : "") << walls[i];
        js << "],\n  \"nodes\": [";
        // A failed node keeps its reason in `status`, and until this was emitted the only
        // way to read it was `--verbose` on a hand-run -- so a whole campaign could record
        // `all_ok=false` with no auditable cause.  `decline` plans fail *by design*, so the
        // reason is a result, not a diagnostic.
        auto js_escape = [](const std::string & in) {
            std::string out;
            for (char ch : in) {
                if (ch == '\n') { out += ' '; continue; }
                if (ch == '"' || ch == '\\') out += '\\';
                out += ch;
            }
            return out;
        };
        for (size_t i = 0; i < s.c.nodes.size(); ++i) {
            const Node & n = s.c.nodes[i];
            js << (i ? ",\n    " : "\n    ") << "{\"index\": " << n.index << ", \"op\": \"" << n.op
               << "\", \"machine\": \"" << n.machine << "\", \"id\": \"" << n.id << "\", \"dur_ns\": [";
            for (size_t r = 0; r < replica_runs.size(); ++r) {
                const NodeRun & run = replica_runs[r][i];
                js << (r ? ", " : "") << (run.end_ns - run.begin_ns);
            }
            js << "], \"build_ns\": " << replica_runs[0][i].build_ns
               << ", \"bytes\": " << replica_runs[0][i].bytes
               << ", \"moved\": " << replica_runs[0][i].moved
               << ", \"evicted_bytes\": " << replica_runs[0][i].evicted
               // the shape this node actually executed at, echoed so the driver can
               // price it at the same one rather than at the case header's
               << ", \"tokens\": " << n.num("t", s.c.tokens)
               << ", \"history\": " << n.num("h", s.c.history)
               << ", \"ok\": "
               << (replica_runs[0][i].ok ? "true" : "false")
               << ", \"status\": \"" << js_escape(replica_runs[0][i].status) << "\"}";
        }
        js << "\n  ]\n}\n";
        if (!o.out_path.empty()) {
            std::ofstream f(o.out_path);
            f << js.str();
        }
        std::printf("%s", js.str().c_str());
        return all_ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
