// Resident F32 MoE FFN trace replay. No model weights, attention, or offload
// timing.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Options {
  int h = 0, i = 0, e = 0, k = 0, q = 0, threads = 0, warmup = -1, samples = 0;
  uint32_t seed = 0;
  std::vector<std::vector<int32_t>> routes;
  std::string sampling;
};
static int integer(const std::string &v) {
  size_t n = 0;
  long long x = std::stoll(v, &n);
  if (n != v.size() || x < 0 || x > 1000000)
    throw std::runtime_error("invalid integer");
  return int(x);
}
static Options parse(int argc, char **argv) {
  Options o;
  std::set<std::string> seen;
  std::string routes;
  for (int a = 1; a < argc; a += 2) {
    std::string key = argv[a];
    if (a + 1 == argc || !seen.insert(key).second)
      throw std::runtime_error("missing or duplicate option");
    std::string v = argv[a + 1];
    if (key == "--hidden")
      o.h = integer(v);
    else if (key == "--intermediate")
      o.i = integer(v);
    else if (key == "--experts")
      o.e = integer(v);
    else if (key == "--top-k")
      o.k = integer(v);
    else if (key == "--tokens")
      o.q = integer(v);
    else if (key == "--threads")
      o.threads = integer(v);
    else if (key == "--warmup")
      o.warmup = integer(v);
    else if (key == "--samples")
      o.samples = integer(v);
    else if (key == "--seed")
      o.seed = uint32_t(integer(v));
    else if (key == "--sampling-mode")
      o.sampling = v;
    else if (key == "--routes")
      routes = v;
    else if (key == "--backend") {
      if (v != "CPU")
        throw std::runtime_error("unsupported backend; CPU only, no fallback");
    } else
      throw std::runtime_error("unknown option: " + key);
  }
  for (auto key : {"--hidden", "--intermediate", "--experts", "--top-k",
                   "--tokens", "--threads", "--warmup", "--samples", "--seed",
                   "--routes", "--backend"})
    if (!seen.count(key))
      throw std::runtime_error("all execution options must be explicit");
  if (o.h < 1 || o.i < 1 || o.e < 1 || o.k < 1 || o.k > o.e || o.q < 1 ||
      o.threads < 1 || o.threads > GGML_MAX_N_THREADS || o.warmup < 0 ||
      o.samples < 3 || o.samples > 10000 || o.e > 1024 || o.q > 4096 ||
      o.h > 32768 || o.i > 65536)
    throw std::runtime_error("invalid or unsupported geometry/runtime budget");
  if (o.sampling != "isolated_warm" &&
      o.sampling != "interleaved_joint_context")
    throw std::runtime_error("unsupported sampling mode");
  std::istringstream rs(routes);
  std::string row;
  while (std::getline(rs, row, ';')) {
    std::vector<int32_t> r;
    std::set<int> unique;
    std::istringstream vs(row);
    std::string v;
    while (std::getline(vs, v, ',')) {
      int id = integer(v);
      if (id >= o.e || !unique.insert(id).second)
        throw std::runtime_error("invalid or duplicate expert");
      r.push_back(id);
    }
    if (int(r.size()) != o.k)
      throw std::runtime_error("route top-k mismatch");
    o.routes.push_back(r);
  }
  if (int(o.routes.size()) != o.q)
    throw std::runtime_error("route token coverage mismatch");
  return o;
}
struct Data {
  std::vector<float> x, router, scores, hidden;
  std::map<int, std::vector<float>> gate, up, down, expert;
  std::map<int, std::vector<int32_t>> rows, slots;
  std::vector<int32_t> route_ids, discovered;
};
static std::vector<float> matrix(const std::vector<float> &w,
                                 const std::vector<float> &x, int in, int out,
                                 int n) {
  std::vector<float> y(size_t(out) * n);
  for (int t = 0; t < n; t++)
    for (int r = 0; r < out; r++) {
      double s = 0;
      for (int c = 0; c < in; c++)
        s += double(w[size_t(r) * in + c]) * x[size_t(t) * in + c];
      y[size_t(t) * out + r] = float(s);
    }
  return y;
}
static Data data(const Options &o) {
  Data d;
  std::mt19937 rng(o.seed);
  std::uniform_real_distribution<float> rnd(-.43f, .43f);
  auto fill = [&](size_t n, float scale = 1.f) {
    std::vector<float> v(n);
    for (auto &x : v)
      x = rnd(rng) * scale;
    return v;
  };
  d.x = fill(size_t(o.h) * o.q);
  d.router = fill(size_t(o.h) * o.e, 1.f / std::sqrt(float(o.h)));
  for (int e = 0; e < o.e; e++) {
    bool active = false;
    for (auto &route : o.routes)
      active =
          active || std::find(route.begin(), route.end(), e) != route.end();
    if (!active)
      continue;
    d.gate[e] = fill(size_t(o.h) * o.i, 1.f / std::sqrt(float(o.h)));
    d.up[e] = fill(size_t(o.h) * o.i, 1.f / std::sqrt(float(o.h)));
    d.down[e] = fill(size_t(o.i) * o.h, 1.f / std::sqrt(float(o.i)));
  }
  auto logits = matrix(d.router, d.x, o.h, o.e, o.q);
  d.scores.resize(size_t(o.k) * o.q);
  d.discovered.resize(size_t(o.k) * o.q);
  for (int t = 0; t < o.q; t++) {
    std::vector<int32_t> order(o.e);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return logits[size_t(t) * o.e + a] > logits[size_t(t) * o.e + b];
    });
    double max = *std::max_element(logits.begin() + size_t(t) * o.e,
                                   logits.begin() + size_t(t + 1) * o.e);
    double sum = 0;
    for (int s = 0; s < o.k; s++)
      sum += std::exp(double(logits[size_t(t) * o.e + o.routes[t][s]]) - max);
    for (int s = 0; s < o.k; s++) {
      int e = o.routes[t][s];
      d.route_ids.push_back(e);
      d.scores[size_t(t) * o.k + s] =
          float(std::exp(double(logits[size_t(t) * o.e + e]) - max) / sum);
      d.discovered[size_t(t) * o.k + s] = order[s];
      d.rows[e].push_back(t);
      d.slots[e].push_back(t * o.k + s);
    }
  }
  d.hidden = d.x;
  for (auto &item : d.rows) {
    int e = item.first, n = int(item.second.size());
    std::vector<float> gathered(size_t(o.h) * n);
    for (int t = 0; t < n; t++)
      std::copy_n(d.x.begin() + size_t(item.second[t]) * o.h, o.h,
                  gathered.begin() + size_t(t) * o.h);
    auto g = matrix(d.gate[e], gathered, o.h, o.i, n),
         u = matrix(d.up[e], gathered, o.h, o.i, n);
    for (size_t j = 0; j < g.size(); j++)
      g[j] = float(double(g[j]) / (1. + std::exp(-double(g[j])))) * u[j];
    d.expert[e] = matrix(d.down[e], g, o.i, o.h, n);
    for (int r = 0; r < n; r++)
      for (int c = 0; c < o.h; c++)
        d.hidden[size_t(item.second[r]) * o.h + c] +=
            d.scores[item.second[r] * o.k + (d.slots[e][r] % o.k)] *
            d.expert[e][size_t(r) * o.h + c];
  }
  return d;
}
struct Stage {
  ggml_context *resident = nullptr, *ctx = nullptr, *slice_ctx = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_gallocr_t allocator = nullptr;
  ggml_threadpool_t pool = nullptr;
  ggml_cgraph *graph = nullptr;
  std::map<std::string, ggml_tensor *> tensors;
  std::map<std::string, std::string> categories;
  std::map<std::string, std::pair<int, int>> ranges;
  size_t internal = 0, work = 0, resident_logical = 0;
  std::vector<uint8_t> work_data;
  ggml_cplan plan{};
  ~Stage() {
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
    if (slice_ctx)
      ggml_free(slice_ctx);
    if (ctx)
      ggml_free(ctx);
    if (resident)
      ggml_free(resident);
  }
  void compute() {
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("graph compute failed");
  }
};
static std::unique_ptr<Stage> build(const Options &o, const Data &d,
                                    const std::string &kind) {
  auto p = std::make_unique<Stage>();
  auto &s = *p;
  s.backend = ggml_backend_cpu_init();
  if (!s.backend)
    throw std::runtime_error("CPU backend unavailable");
  auto tp = ggml_threadpool_params_default(o.threads);
  s.pool = ggml_threadpool_new(&tp);
  if (!s.pool)
    throw std::runtime_error("threadpool initialization failed");
  ggml_backend_cpu_set_n_threads(s.backend, o.threads);
  ggml_backend_cpu_set_threadpool(s.backend, s.pool);
  size_t count = size_t(o.e) * 20 + 100;
  s.resident = ggml_init({ggml_tensor_overhead() * count, nullptr, true});
  s.ctx = ggml_init({ggml_tensor_overhead() * count * 4 +
                         ggml_graph_overhead_custom(count * 4, false),
                     nullptr, true});
  if (!s.ctx || !s.resident)
    throw std::runtime_error("metadata allocation failed");
  s.graph = ggml_new_graph_custom(s.ctx, count * 4, false);
  auto add = [&](std::string name, ggml_type type, int64_t a, int64_t b,
                 std::string category) {
    auto *t = ggml_new_tensor_2d(s.resident, type, a, b);
    ggml_set_name(t, name.c_str());
    if (category == "output")
      ggml_set_output(t);
    else
      ggml_set_input(t);
    s.tensors[name] = t;
    s.categories[name] = category;
    s.resident_logical += ggml_nbytes(t);
    return t;
  };
  bool joint = kind == "joint", router = kind == "router" || joint,
       combine = kind == "combine" || joint;
  int selected = -1;
  if (kind.rfind("expert.", 0) == 0)
    selected = integer(kind.substr(7));
  auto *x = add("hidden_input", GGML_TYPE_F32, o.h, o.q, "input");
  if (router) {
    add("router_weight", GGML_TYPE_F32, o.h, o.e, "weight");
    add("route_ids", GGML_TYPE_I32, o.k, o.q, "input");
    add("selected_scores", GGML_TYPE_F32, o.k, o.q, "output");
    add("discovered_topk", GGML_TYPE_I32, o.k, o.q, "output");
  } else if (combine)
    add("selected_scores", GGML_TYPE_F32, o.k, o.q, "input");
  for (auto &item : d.rows) {
    int e = item.first, n = int(item.second.size());
    if (joint || selected == e) {
      for (auto role : {"gate", "up"})
        add(std::string(role) + "." + std::to_string(e), GGML_TYPE_F32, o.h,
            o.i, "weight");
      add("down." + std::to_string(e), GGML_TYPE_F32, o.i, o.h, "weight");
      add("rows." + std::to_string(e), GGML_TYPE_I32, n, 1, "input");
      add("expert_output." + std::to_string(e), GGML_TYPE_F32, o.h, n,
          "output");
    } else if (combine)
      add("expert_output." + std::to_string(e), GGML_TYPE_F32, o.h, n, "input");
    if (combine) {
      if (!s.tensors.count("rows." + std::to_string(e)))
        add("rows." + std::to_string(e), GGML_TYPE_I32, n, 1, "input");
      add("slots." + std::to_string(e), GGML_TYPE_I32, n, 1, "input");
    }
  }
  if (combine)
    add("hidden_output", GGML_TYPE_F32, o.h, o.q, "output");
  s.buffer = ggml_backend_alloc_ctx_tensors(s.resident, s.backend);
  if (!s.buffer)
    throw std::runtime_error("resident allocation failed");
  ggml_backend_buffer_clear(s.buffer, 0);
  auto write = [&](const std::string &n, const auto &v) {
    if (s.tensors.count(n))
      ggml_backend_tensor_set(s.tensors.at(n), v.data(), 0,
                              ggml_nbytes(s.tensors.at(n)));
  };
  write("hidden_input", d.x);
  write("router_weight", d.router);
  write("route_ids", d.route_ids);
  if (!router)
    write("selected_scores", d.scores);
  for (auto &item : d.rows) {
    int e = item.first;
    std::string suffix = "." + std::to_string(e);
    write("gate" + suffix, d.gate.at(e));
    write("up" + suffix, d.up.at(e));
    write("down" + suffix, d.down.at(e));
    write("rows" + suffix, item.second);
    write("slots" + suffix, d.slots.at(e));
    if (!joint && selected != e)
      write("expert_output" + suffix, d.expert.at(e));
  }
  auto materialize = [&](std::string name, ggml_tensor *node) {
    auto *copy = ggml_cpy(s.ctx, node, s.tensors.at(name));
    ggml_build_forward_expand(s.graph, copy);
    return copy;
  };
  ggml_tensor *scores = s.tensors.count("selected_scores")
                            ? s.tensors.at("selected_scores")
                            : nullptr;
  if (router) {
    int begin = ggml_graph_n_nodes(s.graph);
    auto *logits = ggml_mul_mat(s.ctx, s.tensors.at("router_weight"), x);
    auto *probs = ggml_soft_max(s.ctx, logits);
    materialize("discovered_topk", ggml_argsort_top_k(s.ctx, probs, o.k));
    auto *prob3 = ggml_reshape_3d(s.ctx, probs, 1, o.e, o.q);
    auto *pick = ggml_get_rows(s.ctx, prob3, s.tensors.at("route_ids"));
    auto *pick2 = ggml_cont(s.ctx, ggml_transpose(s.ctx, pick));
    auto *renorm = ggml_div(s.ctx, pick2, ggml_sum_rows(s.ctx, pick2));
    scores = materialize("selected_scores",
                         ggml_reshape_2d(s.ctx, renorm, o.k, o.q));
    s.ranges["router"] = {begin, ggml_graph_n_nodes(s.graph)};
  }
  std::map<int, ggml_tensor *> outputs;
  for (auto &item : d.rows) {
    int e = item.first;
    std::string suffix = "." + std::to_string(e);
    if (joint || selected == e) {
      int begin = ggml_graph_n_nodes(s.graph);
      auto *group = ggml_get_rows(s.ctx, x, s.tensors.at("rows" + suffix));
      auto *g = ggml_mul_mat(s.ctx, s.tensors.at("gate" + suffix), group);
      auto *u = ggml_mul_mat(s.ctx, s.tensors.at("up" + suffix), group);
      auto *a = ggml_swiglu_split(s.ctx, g, u);
      outputs[e] =
          materialize("expert_output" + suffix,
                      ggml_mul_mat(s.ctx, s.tensors.at("down" + suffix), a));
      s.ranges["expert" + suffix] = {begin, ggml_graph_n_nodes(s.graph)};
    } else if (combine)
      outputs[e] = s.tensors.at("expert_output" + suffix);
  }
  if (combine) {
    int begin = ggml_graph_n_nodes(s.graph);
    ggml_tensor *sum = x;
    auto *score_rows = ggml_reshape_2d(s.ctx, scores, 1, o.k * o.q);
    for (auto &item : d.rows) {
      int e = item.first;
      std::string suffix = "." + std::to_string(e);
      auto *coeff =
          ggml_get_rows(s.ctx, score_rows, s.tensors.at("slots" + suffix));
      auto *weighted = ggml_mul(s.ctx, outputs.at(e), coeff);
      auto *scatter =
          ggml_get_rows_back(s.ctx, weighted, s.tensors.at("rows" + suffix), x);
      sum = ggml_add(s.ctx, sum, scatter);
    }
    materialize("hidden_output", sum);
    s.ranges["combine"] = {begin, ggml_graph_n_nodes(s.graph)};
  }
  for (int j = 0; j < ggml_graph_n_nodes(s.graph); j++)
    if (!ggml_backend_supports_op(s.backend, ggml_graph_node(s.graph, j)))
      throw std::runtime_error("unsupported CPU operation");
  s.allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (!s.allocator || !ggml_gallocr_alloc_graph(s.allocator, s.graph))
    throw std::runtime_error("graph allocation failed");
  s.internal = ggml_gallocr_get_buffer_size(s.allocator, 0);
  s.plan = ggml_graph_plan(s.graph, o.threads, s.pool);
  s.work = s.plan.work_size;
  s.work_data.resize(s.work);
  s.plan.work_data = s.work_data.data();
  return p;
}
static std::string digest(ggml_tensor *t) {
  std::vector<uint8_t> b(ggml_nbytes(t));
  ggml_backend_tensor_get(t, b.data(), 0, b.size());
  uint64_t h = 14695981039346656037ULL;
  for (auto v : b) {
    h ^= v;
    h *= 1099511628211ULL;
  }
  std::ostringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(16) << h;
  return ss.str();
}
static std::vector<float> read(ggml_tensor *t) {
  std::vector<float> v(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
  return v;
}
static void
output_stage(const Options &o, const Data &d, const std::string &name, Stage &s,
             const std::vector<int64_t> *supplied = nullptr,
             const std::map<std::string, std::string> *visible = nullptr,
             size_t stage_work = 0) {
  const auto &categories = visible ? *visible : s.categories;
  size_t logical = 0;
  for (auto &item : categories)
    logical += ggml_nbytes(s.tensors.at(item.first));
  s.compute();
  double max_abs = 0, max_rel = 0;
  bool finite = true, passed = true;
  for (auto &item : categories)
    if (item.second == "output" &&
        s.tensors.at(item.first)->type == GGML_TYPE_F32) {
      auto actual = read(s.tensors.at(item.first));
      const std::vector<float> *ref = nullptr;
      if (item.first == "selected_scores")
        ref = &d.scores;
      else if (item.first == "hidden_output")
        ref = &d.hidden;
      else if (item.first.rfind("expert_output.", 0) == 0)
        ref = &d.expert.at(integer(item.first.substr(14)));
      if (!ref || actual.size() != ref->size())
        throw std::runtime_error("reference coverage mismatch");
      for (size_t j = 0; j < actual.size(); j++) {
        double a = std::abs(double(actual[j]) - (*ref)[j]),
               r = a / std::max(1.e-6, std::abs(double((*ref)[j])));
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, r);
        finite = finite && std::isfinite(actual[j]);
        passed = passed && a <= 1.e-4 + 1.e-4 * std::abs(double((*ref)[j]));
      }
    }
  if (s.tensors.count("discovered_topk")) {
    std::vector<int32_t> got(size_t(o.k) * o.q);
    ggml_backend_tensor_get(s.tensors.at("discovered_topk"), got.data(), 0,
                            got.size() * 4);
    passed = passed && got == d.discovered;
  }
  if (!supplied)
    for (int j = 0; j < o.warmup; j++)
      s.compute();
  std::vector<int64_t> samples = supplied ? *supplied : std::vector<int64_t>{};
  if (!supplied)
    for (int j = 0; j < o.samples; j++) {
      auto start = std::chrono::steady_clock::now();
      s.compute();
      auto end = std::chrono::steady_clock::now();
      auto ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      if (ns <= 0)
        throw std::runtime_error("nonpositive measured time");
      samples.push_back(ns);
    }
  std::cout << "\"" << name << "\":{\"samples_ns\":[";
  for (size_t j = 0; j < samples.size(); j++) {
    if (j)
      std::cout << ',';
    std::cout << samples[j];
  }
  std::cout << "],\"workspace_bytes\":" << s.internal + s.work
            << ",\"workspace_source\":\""
            << (supplied
                    ? "shared_ggml_gallocr_internal_buffer_plus_cpu_plan_arena"
                    : "ggml_gallocr_internal_buffer_plus_cpu_graph_plan")
            << "\",\"memory_accounting\":{\"resident_logical_bytes\":"
            << logical << ",\"resident_allocated_bytes\":"
            << ggml_backend_buffer_get_size(s.buffer)
            << ",\"internal_buffer_bytes\":" << s.internal
            << ",\"cpu_graph_plan_work_bytes\":" << s.work
            << ",\"shared_resident_logical_bytes\":" << s.resident_logical
            << ",\"shared_resident_allocated_bytes\":"
            << ggml_backend_buffer_get_size(s.buffer)
            << ",\"stage_cpu_graph_plan_required_bytes\":" << stage_work
            << ",\"shared_allocator\":" << (supplied ? "true" : "false")
            << ",\"metadata_bytes_included\":false},\"all_outputs_finite\":"
            << (finite ? "true" : "false")
            << ",\"reference_check\":{\"passed\":"
            << (passed && finite ? "true" : "false")
            << ",\"max_abs_error\":" << max_abs
            << ",\"max_rel_error\":" << max_rel
            << ",\"atol\":0.0001,\"rtol\":0.0001},\"ports\":{";
  bool first = true;
  for (auto &item : categories) {
    if (!first)
      std::cout << ',';
    first = false;
    auto *t = s.tensors.at(item.first);
    std::cout << "\"" << item.first << "\":{\"dtype\":\""
              << (t->type == GGML_TYPE_F32 ? "f32" : "i32") << "\",\"ne\":["
              << t->ne[0] << ',' << t->ne[1]
              << ",1,1],\"bytes\":" << ggml_nbytes(t) << ",\"category\":\""
              << item.second << "\",\"digest\":\"" << digest(t) << "\"}";
  }
  std::cout << "}}";
  if (!passed || !finite)
    throw std::runtime_error("numerical reference failed");
}

static std::map<std::string, std::string>
stage_categories(const Data &d, const std::string &name) {
  std::map<std::string, std::string> v{{"hidden_input", "input"}};
  bool router = name == "router" || name == "joint",
       combine = name == "combine" || name == "joint";
  int selected = name.rfind("expert.", 0) == 0 ? integer(name.substr(7)) : -1;
  if (router) {
    v["router_weight"] = "weight";
    v["route_ids"] = "input";
    v["selected_scores"] = "output";
    v["discovered_topk"] = "output";
  } else if (combine)
    v["selected_scores"] = "input";
  for (auto &item : d.rows) {
    int e = item.first;
    std::string suffix = "." + std::to_string(e);
    if (name == "joint" || selected == e) {
      for (auto role : {"gate", "up", "down"})
        v[std::string(role) + suffix] = "weight";
      v["rows" + suffix] = "input";
      v["expert_output" + suffix] = "output";
    } else if (combine)
      v["expert_output" + suffix] = "input";
    if (combine) {
      v["rows" + suffix] = "input";
      v["slots" + suffix] = "input";
    }
  }
  if (combine)
    v["hidden_output"] = "output";
  return v;
}
struct SharedSamples {
  std::map<std::string, std::vector<int64_t>> samples;
  std::map<std::string, size_t> required_work;
};
static SharedSamples sample_shared(const Options &o, Stage &s,
                                   const std::vector<std::string> &names) {
  SharedSamples result;
  s.slice_ctx = ggml_init(
      {ggml_graph_overhead_custom(512, false) * names.size(), nullptr, true});
  if (!s.slice_ctx)
    throw std::runtime_error("slice metadata allocation failed");
  std::map<std::string, ggml_cgraph *> graphs;
  std::map<std::string, ggml_cplan> plans;
  int next = 0;
  size_t max_work = 0;
  for (auto &name : names) {
    ggml_cgraph *g = s.graph;
    if (name != "joint") {
      auto range = s.ranges.at(name);
      if (range.first != next || range.second <= range.first)
        throw std::runtime_error("noncontiguous joint stage partition");
      g = ggml_new_graph_custom(s.slice_ctx, 512, false);
      for (int j = range.first; j < range.second; j++)
        ggml_graph_add_node(g, ggml_graph_node(s.graph, j));
      next = range.second;
    }
    graphs[name] = g;
    plans[name] = ggml_graph_plan(g, o.threads, s.pool);
    result.required_work[name] = plans[name].work_size;
    max_work = std::max(max_work, plans[name].work_size);
  }
  if (next != ggml_graph_n_nodes(s.graph))
    throw std::runtime_error("stage partition omits joint nodes");
  s.work_data.resize(max_work);
  s.work = s.work_data.capacity();
  for (auto &item : plans)
    item.second.work_data = s.work_data.data();
  s.plan.work_data = s.work_data.data();
  auto execute = [&](const std::string &name) {
    if (ggml_graph_compute(graphs.at(name), &plans.at(name)) !=
        GGML_STATUS_SUCCESS)
      throw std::runtime_error("slice compute failed");
  };
  for (int round = 0; round < o.warmup; round++)
    for (auto &name : names)
      execute(name);
  for (int round = 0; round < o.samples; round++)
    for (auto &name : names) {
      auto begin = std::chrono::steady_clock::now();
      execute(name);
      auto end = std::chrono::steady_clock::now();
      auto ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count();
      if (ns <= 0)
        throw std::runtime_error("nonpositive slice sample");
      result.samples[name].push_back(ns);
    }
  return result;
}

int main(int argc, char **argv) {
  try {
    Options o = parse(argc, argv);
    Data d = data(o);
    Data reference;
    reference.scores = d.scores;
    reference.expert = d.expert;
    reference.hidden = d.hidden;
    reference.discovered = d.discovered;
    auto setup = build(o, d, "joint");
    setup->compute();
    d.scores = read(setup->tensors.at("selected_scores"));
    for (auto &item : d.rows)
      d.expert[item.first] = read(setup->tensors.at(
          "expert_output." +
          std::to_string(item.first))); // exact materialized stage lineage for
                                        // isolated inputs
    const bool shared = o.sampling == "interleaved_joint_context";
    if (!shared)
      setup.reset(); // The v1 protocol measured separate resident graphs.
    std::vector<std::string> names = {"router"};
    for (auto &item : d.rows)
      names.push_back("expert." + std::to_string(item.first));
    names.push_back("combine");
    names.push_back("joint");
    SharedSamples paired;
    if (shared)
      paired = sample_shared(o, *setup, names);
    std::cout
        << std::setprecision(17) << "{\"schema_version\":\""
        << (shared ? "resident_moe_ffn_native_v2"
                   : "resident_moe_ffn_native_v1")
        << "\",\"contract\":\""
        << (shared ? "ggml_cpu_resident_moe_ffn_f32_v2"
                   : "ggml_cpu_resident_moe_ffn_f32_v1")
        << "\",\"sampling_mode\":\"" << o.sampling
        << "\","
           "\"backend\":\"CPU\",\"routing_mode\":\"prescribed_trace_"
           "replay\",\"scope\":\"resident_moe_ffn_subgraph\",\"random_"
           "weights_not_model_weights\":true,\"router_discovery_used_for_"
           "grouping\":false,\"dtype\":\"f32\",\"config\":{\"hidden\":"
        << o.h << ",\"intermediate\":" << o.i << ",\"experts\":" << o.e
        << ",\"top_k\":" << o.k << ",\"tokens\":" << o.q
        << "},\"threads_requested\":" << o.threads
        << ",\"threads_applied\":" << o.threads << ",\"warmup\":" << o.warmup
        << ",\"samples\":" << o.samples << ",\"seed\":" << o.seed
        << ",\"timing_scope\":\""
        << (shared
                ? "interleaved_materialized_stages_and_joint_on_shared_"
                  "resident_graph"
                : "steady_clock_completed_cpu_graph_with_persistent_threadpool")
        << "\",\"port_digest_algorithm\":\"fnv1a64_"
           "native_tensor_bytes\",\"routes\":[";
    for (int t = 0; t < o.q; t++) {
      if (t)
        std::cout << ',';
      std::cout << '[';
      for (int k = 0; k < o.k; k++) {
        if (k)
          std::cout << ',';
        std::cout << o.routes[t][k];
      }
      std::cout << ']';
    }
    std::cout << "],\"discovered_routes\":[";
    for (int t = 0; t < o.q; t++) {
      if (t)
        std::cout << ',';
      std::cout << '[';
      for (int k = 0; k < o.k; k++) {
        if (k)
          std::cout << ',';
        std::cout << d.discovered[size_t(t) * o.k + k];
      }
      std::cout << ']';
    }
    std::cout << "],\"stages\":{";
    for (size_t j = 0; j < names.size(); j++) {
      if (j)
        std::cout << ',';
      if (shared) {
        auto visible = stage_categories(d, names[j]);
        output_stage(o, reference, names[j], *setup,
                     &paired.samples.at(names[j]), &visible,
                     paired.required_work.at(names[j]));
      } else {
        auto stage = build(o, d, names[j]);
        output_stage(o, reference, names[j], *stage);
      }
    }
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
