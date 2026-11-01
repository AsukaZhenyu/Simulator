// Optional Windows NVML PCIe link observations. No instance may be constructed
// by a metadata-only self-test: construction initializes the actual NVML runtime.
// There are no CUDA events, CUPTI, per-copy queries, or theoretical fallbacks.
#pragma once

#ifndef _WIN32
#error "PCIeNVML requires Windows module identity and QPC APIs"
#endif

#include <windows.h>
#include <winver.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <cwchar>
#include <cwctype>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace simulator_pcie_nvml {
using json = nlohmann::json;
struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st *;
using nvmlReturn_t = int;
using Init = nvmlReturn_t(__cdecl *)(void);
using Shutdown = nvmlReturn_t(__cdecl *)(void);
using HandleByIndex = nvmlReturn_t(__cdecl *)(unsigned int, nvmlDevice_t *);
using HandleByBusId = nvmlReturn_t(__cdecl *)(const char *, nvmlDevice_t *);
using LinkQuery = nvmlReturn_t(__cdecl *)(nvmlDevice_t, unsigned int *);

inline std::wstring wide(const std::string &value) {
    if (value.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
        value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) return {};
    std::wstring result(static_cast<size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), size) != size) return {};
    return result;
}

inline std::string utf8(const std::wstring &value) {
    if (value.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
        value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), size, nullptr, nullptr) != size) return {};
    return result;
}

inline bool absolute(const std::wstring &path) {
    return (path.size() >= 3 && std::iswalpha(path[0]) && path[1] == L':'
            && (path[2] == L'\\' || path[2] == L'/'))
        || (path.size() > 2 && path[0] == L'\\' && path[1] == L'\\');
}

inline std::wstring system_file(const wchar_t *name, DWORD &error) {
    std::vector<wchar_t> buffer(32768);
    const UINT size = GetSystemDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
    if (!size || size >= buffer.size()) { error = GetLastError(); return {}; }
    return std::wstring(buffer.data(), size) + L"\\" + name;
}

inline std::wstring module_file(HMODULE module, DWORD &error) {
    std::vector<wchar_t> buffer(32768);
    const DWORD size = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!size || size >= buffer.size()) { error = GetLastError(); return {}; }
    return std::wstring(buffer.data(), size);
}

inline std::wstring final_file(const std::wstring &path, DWORD &error, uint64_t *bytes = nullptr) {
    const HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { error = GetLastError(); return {}; }
    LARGE_INTEGER size{};
    if (bytes && !GetFileSizeEx(file, &size)) {
        error = GetLastError(); CloseHandle(file); return {};
    }
    std::vector<wchar_t> buffer(32768);
    const DWORD used = GetFinalPathNameByHandleW(file, buffer.data(),
        static_cast<DWORD>(buffer.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    const DWORD last = used && used < buffer.size() ? ERROR_SUCCESS : GetLastError();
    CloseHandle(file);
    if (last != ERROR_SUCCESS || !used || used >= buffer.size()) { error = last; return {}; }
    if (bytes) *bytes = static_cast<uint64_t>(size.QuadPart);
    return std::wstring(buffer.data(), used);
}

inline bool bus_id_valid(const std::string &value) {
    // CUDA can print four- or eight-digit PCI domains. No NVML index fallback.
    const size_t colon = value.find(':');
    if ((colon != 4 && colon != 8) || value.size() != colon + 8
        || value[colon + 3] != ':' || value[colon + 6] != '.'
        || value[colon + 7] < '0' || value[colon + 7] > '7') return false;
    for (size_t i = 0; i < value.size(); ++i) {
        if (i == colon || i == colon + 3 || i == colon + 6) continue;
        const char c = value[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    }
    return true;
}

inline json qpc_value(bool valid, int64_t value) { return valid ? json(value) : json(nullptr); }
inline bool tick(int64_t &value) {
    LARGE_INTEGER sample{};
    if (!QueryPerformanceCounter(&sample)) return false;
    value = sample.QuadPart; return true;
}
}  // namespace simulator_pcie_nvml

class PCIeNVML {
    using json = nlohmann::json;
    using Device = simulator_pcie_nvml::nvmlDevice_t;
    HMODULE module_ = nullptr;
    simulator_pcie_nvml::Shutdown shutdown_ = nullptr;
    Device device_ = nullptr;
    bool initialized_ = false, ready_ = false, closed_ = false;
    std::array<simulator_pcie_nvml::LinkQuery, 4> queries_{};
    const std::array<const char *, 4> symbols_{{
        "nvmlDeviceGetCurrPcieLinkGeneration", "nvmlDeviceGetMaxPcieLinkGeneration",
        "nvmlDeviceGetCurrPcieLinkWidth", "nvmlDeviceGetMaxPcieLinkWidth"}};
    const std::array<const char *, 4> fields_{{
        "pcie_current_gen", "pcie_max_gen", "pcie_current_width", "pcie_max_width"}};
    json metadata_ = json::object();
    std::string unavailable_;

    void fail(const std::string &reason, DWORD error = ERROR_SUCCESS) {
        ready_ = false; unavailable_ = reason;
        metadata_["unavailable_reason"] = reason;
        metadata_["windows_error"] = error == ERROR_SUCCESS ? json(nullptr) : json(error);
    }

    template<class Function> Function symbol(const char *name, bool required) {
        const FARPROC address = GetProcAddress(module_, name);
        metadata_["symbols"][name] = json{{"available", address != nullptr}, {"required", required}};
        return reinterpret_cast<Function>(address);
    }

    template<class Call> static json call_status(const char *name, Call call) {
        int64_t begin = 0, end = 0;
        const bool before = simulator_pcie_nvml::tick(begin);
        const int status = call();
        const bool after = simulator_pcie_nvml::tick(end);
        return json{{"symbol", name}, {"nvml_status", status},
            {"status", status == 0 ? "ok" : "unknown"},
            {"qpc_begin", simulator_pcie_nvml::qpc_value(before, begin)},
            {"qpc_end", simulator_pcie_nvml::qpc_value(after, end)},
            {"qpc_valid", before && after && end >= begin}};
    }

    static json version(const std::wstring &file) {
        using namespace simulator_pcie_nvml;
        DWORD error = ERROR_SUCCESS;
        const auto path = system_file(L"version.dll", error);
        if (path.empty()) return json{{"status", "unknown"}, {"reason", "system_version_module_path_unavailable"}, {"windows_error", error}};
        HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) return json{{"status", "unknown"}, {"reason", "system_version_module_load_failed"}, {"windows_error", GetLastError()}};
        using Size = DWORD(WINAPI *)(LPCWSTR, LPDWORD);
        using Info = BOOL(WINAPI *)(LPCWSTR, DWORD, DWORD, LPVOID);
        using Query = BOOL(WINAPI *)(LPCVOID, LPCWSTR, LPVOID *, PUINT);
        const auto get_size = reinterpret_cast<Size>(GetProcAddress(module, "GetFileVersionInfoSizeW"));
        const auto get_info = reinterpret_cast<Info>(GetProcAddress(module, "GetFileVersionInfoW"));
        const auto query = reinterpret_cast<Query>(GetProcAddress(module, "VerQueryValueW"));
        json result{{"status", "unknown"}, {"reader_requested_path", utf8(path)}};
        const auto actual = module_file(module, error);
        result["reader_actual_module_path"] = actual.empty() ? json(nullptr) : json(utf8(actual));
        if (!get_size || !get_info || !query) result["reason"] = "version_reader_symbols_missing";
        else {
            DWORD ignored = 0; const DWORD bytes = get_size(file.c_str(), &ignored);
            if (!bytes) { result["reason"] = "file_version_resource_unavailable"; result["windows_error"] = GetLastError(); }
            else {
                std::vector<uint8_t> storage(bytes); VS_FIXEDFILEINFO *fixed = nullptr; UINT size = 0;
                if (!get_info(file.c_str(), 0, bytes, storage.data())
                    || !query(storage.data(), L"\\", reinterpret_cast<LPVOID *>(&fixed), &size)
                    || !fixed || size < sizeof(VS_FIXEDFILEINFO) || fixed->dwSignature != 0xfeef04bd) {
                    result["reason"] = "file_version_resource_read_failed"; result["windows_error"] = GetLastError();
                } else {
                    const auto a = HIWORD(fixed->dwFileVersionMS), b = LOWORD(fixed->dwFileVersionMS);
                    const auto c = HIWORD(fixed->dwFileVersionLS), d = LOWORD(fixed->dwFileVersionLS);
                    result["status"] = "ok";
                    result["value"] = std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(c) + "." + std::to_string(d);
                }
            }
        }
        FreeLibrary(module); return result;
    }

    void initialize(const std::string &explicit_dll_path, const std::string &cuda_bus_id) {
        using namespace simulator_pcie_nvml;
        metadata_ = json{{"schema", "native-pcie-nvml-provider-1"},
            {"requested_dll_path", explicit_dll_path.empty() ? json(nullptr) : json(explicit_dll_path)},
            {"dll_path_source", explicit_dll_path.empty() ? "GetSystemDirectoryW/nvml.dll" : "explicit_absolute_request_path"},
            {"cuda_pci_bus_id", cuda_bus_id}, {"handle_selection", "nvmlDeviceGetHandleByPciBusId_v2"},
            {"NVML_index_assumed_equal_to_CUDA_index", false}, {"index_fallback_used", false},
            {"symbols", json::object()}, {"initialization_calls", json::array()},
            {"sha256", nullptr}, {"SHA_binding_owner", "parent collection must hash the actual module file"},
            {"link_state_is_effective_bandwidth", false}, {"theoretical_fallback_used", false},
            {"sampling_scope", "four read-only driver link queries per explicit sample, intended once before/after each burst; never per copy"}};
        LARGE_INTEGER frequency{};
        metadata_["qpc_frequency_hz"] = QueryPerformanceFrequency(&frequency) && frequency.QuadPart > 0 ? json(frequency.QuadPart) : json(nullptr);
        DWORD error = ERROR_SUCCESS;
        const std::wstring path = explicit_dll_path.empty() ? system_file(L"nvml.dll", error) : wide(explicit_dll_path);
        if (path.empty() || !absolute(path)) { fail("absolute_NVML_module_path_unavailable_or_invalid_UTF8", error); return; }
        metadata_["resolved_requested_dll_path"] = utf8(path);
        module_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module_) { fail("explicit_NVML_module_load_failed", GetLastError()); return; }
        const auto actual = module_file(module_, error);
        if (actual.empty()) { fail("actual_loaded_NVML_module_path_unavailable", error); return; }
        metadata_["actual_module_path"] = utf8(actual);
        uint64_t bytes = 0;
        const auto actual_final = final_file(actual, error, &bytes);
        if (actual_final.empty()) { fail("actual_loaded_NVML_file_identity_unavailable", error); return; }
        metadata_["actual_module_final_path"] = utf8(actual_final);
        metadata_["actual_module_bytes"] = bytes;
        const auto requested_final = final_file(path, error);
        if (requested_final.empty() || _wcsicmp(requested_final.c_str(), actual_final.c_str()) != 0) {
            fail("loaded_NVML_file_differs_from_explicit_requested_file", error); return;
        }
        metadata_["loaded_requested_file_identity_equal"] = true;
        metadata_["file_version"] = version(actual);
        auto init = symbol<Init>("nvmlInit_v2", true);
        shutdown_ = symbol<Shutdown>("nvmlShutdown", true);
        auto handle = symbol<HandleByBusId>("nvmlDeviceGetHandleByPciBusId_v2", true);
        // Resolve for explicit inventory only. Never call or fall back to it.
        (void)symbol<HandleByIndex>("nvmlDeviceGetHandleByIndex_v2", false);
        for (size_t i = 0; i < queries_.size(); ++i) queries_[i] = symbol<LinkQuery>(symbols_[i], false);
        if (!init || !shutdown_ || !handle) { fail("required_NVML_initialization_or_PCI_bus_handle_symbol_missing"); return; }
        if (!bus_id_valid(cuda_bus_id)) { fail("explicit_CUDA_PCI_bus_id_required_or_invalid_no_index_fallback"); return; }
        auto init_status = call_status("nvmlInit_v2", [&]() { return init(); });
        metadata_["initialization_calls"].push_back(init_status);
        if (init_status["nvml_status"] != 0) { fail("NVML_initialization_failed"); return; }
        initialized_ = true;
        auto handle_status = call_status("nvmlDeviceGetHandleByPciBusId_v2", [&]() { return handle(cuda_bus_id.c_str(), &device_); });
        metadata_["initialization_calls"].push_back(handle_status);
        if (handle_status["nvml_status"] != 0 || !device_) { fail("NVML_handle_by_explicit_CUDA_PCI_bus_id_failed"); return; }
        ready_ = true;
    }

public:
    explicit PCIeNVML(const std::string &explicit_dll_path, const std::string &cuda_bus_id) {
        try { initialize(explicit_dll_path, cuda_bus_id); }
        catch (const std::exception &error) { fail(std::string("provider_initialization_exception: ") + error.what()); }
        catch (...) { fail("provider_initialization_unknown_exception"); }
    }
    PCIeNVML(const PCIeNVML &) = delete;
    PCIeNVML &operator=(const PCIeNVML &) = delete;
    ~PCIeNVML() {
        // Destruction must not throw, including after a partially failed init.
        if (initialized_ && shutdown_) shutdown_();
        if (module_) FreeLibrary(module_);
    }

    json describe() const {
        json result = metadata_;
        result["ready"] = ready_ && !closed_;
        result["closed"] = closed_;
        result["unavailable_reason"] = unavailable_.empty() ? json(nullptr) : json(unavailable_);
        return result;
    }

    json sample() const {
        using namespace simulator_pcie_nvml;
        json rows = json::array(), values = json::object(), unknown = json::object();
        unsigned successes = 0;
        for (size_t i = 0; i < queries_.size(); ++i) {
            json row{{"field", fields_[i]}, {"symbol", symbols_[i]},
                {"status", "unknown"}, {"nvml_status", nullptr}, {"value", nullptr},
                {"qpc_begin", nullptr}, {"qpc_end", nullptr}, {"qpc_valid", false}};
            if (!ready_ || closed_) row["unknown_reason"] = closed_ ? "provider_closed" : unavailable_;
            else if (!queries_[i]) row["unknown_reason"] = "NVML_query_symbol_missing";
            else {
                unsigned int value = 0;
                auto status = call_status(symbols_[i], [&]() { return queries_[i](device_, &value); });
                for (auto it = status.begin(); it != status.end(); ++it) row[it.key()] = it.value();
                if (row["nvml_status"] == 0) {
                    row["value"] = value; values[fields_[i]] = value; ++successes;
                } else row["unknown_reason"] = "NVML_query_failed";
            }
            if (row["status"] != "ok") unknown[fields_[i]] = row.value("unknown_reason", "unidentified_NVML_query_failure");
            rows.push_back(std::move(row));
        }
        return json{{"schema", "native-pcie-nvml-sample-1"},
            {"status", successes == 4 ? "ok" : successes ? "partial" : "unavailable"},
            {"clock", "windows_qpc"}, {"qpc_frequency_hz", metadata_.value("qpc_frequency_hz", json(nullptr))},
            {"cuda_pci_bus_id", metadata_.value("cuda_pci_bus_id", json(nullptr))},
            {"queries", rows}, {"values", values}, {"unknown", unknown},
            {"effective_bandwidth_measured", false}, {"theoretical_fallback_used", false}};
    }

    json close() {
        if (closed_) return json{{"status", "already_closed"}, {"nvml_status", nullptr}};
        closed_ = true; ready_ = false;
        if (!initialized_ || !shutdown_) return json{{"status", "not_initialized"}, {"nvml_status", nullptr}};
        auto result = call_status("nvmlShutdown", [&]() { return shutdown_(); });
        initialized_ = false;
        metadata_["shutdown_call"] = result;
        return result;
    }
};
