#!/usr/bin/env python3
"""One-shot patch: strict CPU placement in ir_runner, matching dense_layer_probe."""
import pathlib

p = pathlib.Path(__file__).with_name("ir_runner.cpp")
t = p.read_text(encoding="utf-8")

if "cpu_ids" in t:
    print("already patched")
    raise SystemExit(0)

t = t.replace('#include <random>', '#include <random>\n#if defined(_WIN32)\n#include <windows.h>\n#endif', 1)

t = t.replace(
    "    bool gpu = true;\n    std::map<std::string, std::string> dtype;",
    "    bool gpu = true;\n"
    "    std::vector<int> cpu_ids;          // strict placement, one per thread, as the tables used\n"
    "    std::map<std::string, std::string> dtype;", 1)

t = t.replace(
    '        else if (d == "cuda_device") c.cuda_device = std::stoi(w[1]);',
    '        else if (d == "cuda_device") c.cuda_device = std::stoi(w[1]);\n'
    '        else if (d == "cpu_mask") for (auto & x : split(w[1], \',\')) c.cpu_ids.push_back(std::stoi(x));', 1)

old = (
    "    s.cpu = ggml_backend_cpu_init();\n"
    "    if (!s.cpu) throw std::runtime_error(\"cpu backend init failed\");\n"
    "    ggml_backend_cpu_set_n_threads(s.cpu, c.threads);")
new = (
    "    s.cpu = ggml_backend_cpu_init();\n"
    "    if (!s.cpu) throw std::runtime_error(\"cpu backend init failed\");\n"
    "    if (!c.cpu_ids.empty()) {\n"
    "        // The frozen tables were measured with explicit_strict_cpu_mask; on this machine a\n"
    "        // free-running process lands on a different band, so the placement is part of the\n"
    "        // instrument and is echoed in the output, not silently assumed.\n"
    "        if (static_cast<int>(c.cpu_ids.size()) != c.threads)\n"
    "            throw std::runtime_error(\"cpu_mask must name exactly `threads` logical CPUs\");\n"
    "        auto tpp = ggml_threadpool_params_default(c.threads);\n"
    "        tpp.strict_cpu = true;\n"
    "        for (const int id : c.cpu_ids) {\n"
    "            if (id < 0 || id >= GGML_MAX_N_THREADS) throw std::runtime_error(\"cpu_mask out of range\");\n"
    "            tpp.cpumask[id] = true;\n"
    "        }\n"
    "#if defined(_WIN32)\n"
    "        {\n"
    "            DWORD_PTR bits = 0;\n"
    "            for (const int id : c.cpu_ids) bits |= DWORD_PTR{1} << id;\n"
    "            if (!SetProcessAffinityMask(GetCurrentProcess(), bits))\n"
    "                throw std::runtime_error(\"cannot apply process CPU affinity\");\n"
    "            DWORD_PTR observed = 0, system_mask = 0;\n"
    "            if (!GetProcessAffinityMask(GetCurrentProcess(), &observed, &system_mask) || observed != bits)\n"
    "                throw std::runtime_error(\"process CPU affinity readback differs from the request\");\n"
    "        }\n"
    "#endif\n"
    "        s.pool = ggml_threadpool_new(&tpp);\n"
    "        if (!s.pool) throw std::runtime_error(\"threadpool initialization failed\");\n"
    "        ggml_backend_cpu_set_threadpool(s.cpu, s.pool);\n"
    "    }\n"
    "    ggml_backend_cpu_set_n_threads(s.cpu, c.threads);")
assert old in t
t = t.replace(old, new, 1)

# Session keeps the pool alive for the whole run
t = t.replace(
    "    ggml_backend_t cpu = nullptr, gpu = nullptr;",
    "    ggml_backend_t cpu = nullptr, gpu = nullptr;\n    ggml_threadpool_t pool = nullptr;", 1)

# echo the placement next to the other config
old_echo = '           << "\\",\\n  \\"threads\\": " << s.c.threads << ",\\n  \\"tokens\\": " << s.c.tokens'
assert old_echo in t, "echo anchor missing"
new_echo = (
    '           << "\\",\\n  \\"threads\\": " << s.c.threads\n'
    '           << ",\\n  \\"tokens\\": " << s.c.tokens\n'
    '           << ",\\n  \\"cpu_mask\\": [";\n'
    '        for (size_t i = 0; i < s.c.cpu_ids.size(); ++i) js << (i ? ", " : "") << s.c.cpu_ids[i];\n'
    '        js << "],\\n  \\"thread_placement\\": "\n'
    '           << (s.c.cpu_ids.empty() ? "not_requested;actual_thread_placement_unknown"\n'
    '                                   : "explicit_strict_cpu_mask")')
t = t.replace(old_echo, new_echo, 1)

p.write_text(t, encoding="utf-8")
print("patched")
