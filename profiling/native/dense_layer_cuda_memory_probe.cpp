// Device-wide synchronized CUDA memory observations; no process peak or bound claim.
// Reuse the declared Dense graph and validation rather than maintaining a second graph.
#define MEASURE_DENSE_CUDA_MEMORY_PROBE 1
#define main dense_cuda_performance_main_unused
#include "dense_layer_cuda_probe.cpp"
#undef main

struct DenseCudaMemorySnapshot {
    std::string label;
    int64_t elapsed_ns = 0;
    int device_index = -1;
    size_t free_bytes = 0, total_bytes = 0, used_bytes = 0;
};

static const auto memory_process_origin = std::chrono::steady_clock::now();
static int memory_requested_device = -1;
static std::vector<DenseCudaMemorySnapshot> memory_snapshots;

static void dense_cuda_memory_checkpoint(const char *label) {
    int current_device = -1;
    cuda_ok(cudaGetDevice(&current_device), "memory checkpoint cudaGetDevice");
    if (current_device != memory_requested_device)
        throw std::runtime_error(
            "memory checkpoint current device differs from requested CUDA device");
    cuda_ok(cudaDeviceSynchronize(), "memory checkpoint cudaDeviceSynchronize");
    size_t free_bytes = 0, total_bytes = 0;
    cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "memory checkpoint cudaMemGetInfo");
    if (total_bytes == 0 || free_bytes > total_bytes)
        throw std::runtime_error("invalid device-wide CUDA memory availability observation");
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - memory_process_origin)
                                .count();
    memory_snapshots.push_back(
        {label, elapsed_ns, current_device, free_bytes, total_bytes, total_bytes - free_bytes});
}

// Restore stdout even if validation or a CUDA API throws before its JSON is complete.
struct DenseCudaCaptureStdout {
    std::ostringstream stream;
    std::streambuf *previous = nullptr;
    DenseCudaCaptureStdout() : previous(std::cout.rdbuf(stream.rdbuf())) {
    }
    ~DenseCudaCaptureStdout() {
        std::cout.rdbuf(previous);
    }
};

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout
                << "ggml-dense-layer-cuda-memory-probe: same explicit Dense CUDA arguments "
                   "as ggml-dense-layer-cuda-probe; synchronized device-wide availability "
                   "snapshots only; timing is not a cost; workspace and process peak unknown\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--list-devices")
            return list_devices();
        const Options options = parse(argc, argv);
        memory_requested_device = options.cuda_device;
        memory_snapshots.clear();
        int validation_exit_code = 0;
        std::string dense_validation;
        {
            DenseCudaCaptureStdout capture;
            validation_exit_code = run(options);
            dense_validation = capture.stream.str();
        }
        // run() has returned, so its stage Session has released its allocator,
        // resident buffer, backend pools, and handle. Runtime state may remain.
        dense_cuda_memory_checkpoint("after_stage_release");
        if (memory_snapshots.size() != 7)
            throw std::runtime_error(
                "memory probe must collect exactly seven declared checkpoints");
        size_t max_observed_used_bytes = 0;
        for (const auto &snapshot : memory_snapshots)
            max_observed_used_bytes = std::max(max_observed_used_bytes, snapshot.used_bytes);
        std::cout << "{\"schema\":\"ggml_dense_layer_cuda_memory_probe/1\","
                     "\"purpose\":\"memory_diagnostics_only\","
                     "\"timing_usable_as_cost\":false,\"hardware_compute_performed\":true,"
                     "\"workspace_bytes\":null,\"workspace_bound\":\"unknown\","
                     "\"snapshot_scope\":\"device_wide_cuda_memory_availability\","
                     "\"source\":\"cudaMemGetInfo_after_cudaDeviceSynchronize\","
                     "\"process_peak_bytes\":null,\"capacity_verified\":false,"
                     "\"peak_measurement_claimed\":false,\"process_attribution_verified\":false,"
                     "\"backend\":\"CUDA"
                  << options.cuda_device << "\",\"cuda_device\":" << options.cuda_device
                  << ",\"cuda_graphs\":\"disabled\",\"device_info\":";
        emit_device(options.cuda_device);
        std::cout << ",\"snapshot_count\":" << memory_snapshots.size()
                  << ",\"max_observed_used_bytes\":" << max_observed_used_bytes
                  << ",\"max_observed_used_bytes_scope\":\"maximum_of_seven_device_wide_snapshots_"
                     "not_process_peak\","
                     "\"upstream_setup_compute_repetitions\":"
                  << (options.stage == "block" ? 0 : 1)
                  << ",\"dense_validation_exit_code\":" << validation_exit_code
                  << ",\"snapshots\":[";
        for (size_t i = 0; i < memory_snapshots.size(); ++i) {
            const auto &snapshot = memory_snapshots[i];
            std::cout << (i ? "," : "") << "{\"label\":" << json_quote(snapshot.label)
                      << ",\"elapsed_ns\":" << snapshot.elapsed_ns
                      << ",\"device_index\":" << snapshot.device_index
                      << ",\"free_bytes\":" << snapshot.free_bytes
                      << ",\"total_bytes\":" << snapshot.total_bytes
                      << ",\"used_bytes\":" << snapshot.used_bytes << "}";
        }
        std::cout << "],\"dense_validation\":" << dense_validation << "}\n";
        return validation_exit_code;
    } catch (const std::exception &error) {
        std::cerr << "ggml-dense-layer-cuda-memory-probe: " << error.what() << '\n';
        return 1;
    }
}
