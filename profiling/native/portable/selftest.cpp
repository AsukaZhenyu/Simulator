// Proof that the POSIX half of platform.hpp behaves, on the machine in front of you.
//
// The probes cannot be built on a host without CUDA, ggml and the pinned
// Executor fragments, but everything this checks is in the two headers they
// include first. Building and running this is therefore the cheapest honest
// answer to "does the port work here":
//
//     c++ -std=c++17 -O2 -o portable-selftest portable/selftest.cpp && ./portable-selftest
//
// It exercises the parts whose failure would be silent rather than loud: a
// digest that is not SHA-256, an affinity mask that round-trips but covers the
// wrong cores, a CPU-time query that never advances, a stopwatch that is not
// monotonic. Each check prints one line and the exit status is the number of
// failures, so a script can gate on it.
#include "platform.hpp"
#include "sha256.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool passed, const std::string &what, const std::string &detail = "") {
    std::printf("%s  %s%s%s\n", passed ? "ok  " : "FAIL", what.c_str(),
                detail.empty() ? "" : "  (", detail.empty() ? "" : (detail + ")").c_str());
    if (!passed) ++failures;
}

void check_sha256() {
    struct Vector { const char *message; std::size_t size; const char *digest; const char *name; };
    // NIST's vectors. The third one is the one that matters: its 0x80 and length
    // do not fit in the final block, so it exercises the two-block padding path.
    const std::string two_block = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const Vector vectors[] = {
        {"", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty"},
        {"abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc"},
        {two_block.c_str(), two_block.size(),
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "56-byte padding boundary"},
    };
    for (const auto &vector : vectors)
        check(portable_sha256::digest_hex(vector.message, vector.size) == vector.digest,
              std::string("sha256 one-shot: ") + vector.name);
    for (const auto &vector : vectors) {
        // The same bytes fed in pieces must hash the same: `initial_assets.hpp`
        // streams a multi-gigabyte file through exactly this path.
        for (std::size_t piece : {std::size_t(1), std::size_t(7), std::size_t(64), std::size_t(65)}) {
            portable_sha256::Hasher hasher;
            for (std::size_t offset = 0; offset < vector.size; offset += piece)
                hasher.update(vector.message + offset,
                              vector.size - offset < piece ? vector.size - offset : piece);
            check(hasher.hex() == vector.digest,
                  std::string("sha256 streamed in ") + std::to_string(piece)
                      + "-byte pieces: " + vector.name);
            check(hasher.hex() == hasher.hex(), "sha256 hex() is idempotent: " + std::string(vector.name));
        }
    }
}

void check_clock() {
    LARGE_INTEGER frequency{};
    const BOOL known = QueryPerformanceFrequency(&frequency);
    check(known != 0 && frequency.QuadPart > 0, "QPC frequency is positive",
          std::to_string(frequency.QuadPart));
    LARGE_INTEGER first{}, second{};
    check(QueryPerformanceCounter(&first) != 0, "QPC counter reads");
    const auto started = first.QuadPart;
    Sleep(20);
    check(QueryPerformanceCounter(&second) != 0 && second.QuadPart > first.QuadPart,
          "QPC is strictly monotonic across a sleep");
    const double milliseconds = double(second.QuadPart - started) * 1000.0 / double(frequency.QuadPart);
    // Sleep(20) may overshoot; it must not return early by more than a scheduler tick.
    check(milliseconds >= 19.0, "Sleep(20) waited at least 19 ms",
          std::to_string(milliseconds) + " ms");
}

void check_environment() {
    check(_putenv_s("PORTABLE_SELFTEST", "value") == 0, "_putenv_s sets");
    const char *observed = std::getenv("PORTABLE_SELFTEST");
    check(observed != nullptr && std::string(observed) == "value", "_putenv_s value is readable");
    check(_putenv_s("PORTABLE_SELFTEST", "") == 0, "_putenv_s clears with an empty value");
    check(std::getenv("PORTABLE_SELFTEST") == nullptr, "a cleared variable is absent");
    check(_putenv_s(nullptr, "x") != 0, "_putenv_s rejects a null name");
}

void check_affinity() {
#if PORTABLE_AFFINITY
    DWORD_PTR initial = 0, system = 0;
    check(GetProcessAffinityMask(GetCurrentProcess(), &initial, &system) != 0,
          "process affinity is observable");
    check(system != 0 && (initial & system) == initial,
          "the process mask is a subset of the online mask");
    unsigned online = 0;
    for (int cpu = 0; cpu < 64; ++cpu) online += unsigned((system >> cpu) & 1);
    check(online != 0, "the online mask names at least one CPU", std::to_string(online) + " online");
    // Pin one core and read it back: this is exactly what the probes assert.
    const DWORD_PTR wanted = 4;
    const DWORD_PTR previous = SetThreadAffinityMask(GetCurrentThread(), wanted);
    check(previous != 0, "thread affinity can be set on core 4");
    GROUP_AFFINITY group{};
    const BOOL observed = GetThreadGroupAffinity(GetCurrentThread(), &group);
    check(observed != 0 && group.Mask == wanted, "the pinned mask reads back unchanged",
          std::to_string(static_cast<unsigned long long>(group.Mask)));
    check(SetThreadAffinityMask(GetCurrentThread(), previous) != 0, "the previous mask is restored");
    GROUP_AFFINITY restored{};
    const BOOL back = GetThreadGroupAffinity(GetCurrentThread(), &restored);
    check(back != 0 && restored.Mask == previous, "the restore reads back unchanged",
          std::to_string(static_cast<unsigned long long>(restored.Mask)));
    check(SetThreadAffinityMask(GetCurrentThread(), 0) == 0, "an empty mask is refused");
#else
    std::printf("skip  affinity: this platform exposes no enforced CPU affinity\n");
    DWORD_PTR initial = 0, system = 0;
    check(GetProcessAffinityMask(GetCurrentProcess(), &initial, &system) == 0,
          "the affinity queries report unavailability rather than a made-up mask");
#endif
}

unsigned long long ticks_of(const FILETIME &value) {
    return (static_cast<unsigned long long>(value.dwHighDateTime) << 32) | value.dwLowDateTime;
}

void check_accounting() {
    FILETIME created{}, exited{}, kernel{}, user{};
    check(GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user) != 0,
          "thread CPU time is available");
    const unsigned long long thread_before = ticks_of(kernel) + ticks_of(user);
    volatile double accumulator = 0.0;
    for (int spin = 0; spin < 2000000; ++spin) accumulator += double(spin) * 1e-9;
    FILETIME after_kernel{}, after_user{};
    check(GetThreadTimes(GetCurrentThread(), &created, &exited, &after_kernel, &after_user) != 0,
          "thread CPU time reads twice");
    const unsigned long long thread_after = ticks_of(after_kernel) + ticks_of(after_user);
    check(thread_after > thread_before, "thread CPU time advanced across work",
          std::to_string(thread_after - thread_before) + " x100ns");
    FILETIME process_kernel{}, process_user{};
    check(GetProcessTimes(GetCurrentProcess(), &created, &exited, &process_kernel, &process_user) != 0,
          "process CPU time is available");
    check(ticks_of(process_kernel) + ticks_of(process_user) >= thread_after,
          "process CPU time covers at least this thread");
    ULONG64 cycles = 12345;
    const BOOL cycles_known = QueryThreadCycleTime(GetCurrentThread(), &cycles);
#if defined(_WIN32)
    check(cycles_known != 0, "thread cycles are available on this platform");
#else
    check(cycles_known == 0 && cycles == 0,
          "thread cycles report unavailability instead of a fabricated count");
#endif

    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    const BOOL faults = GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    check(faults != 0, "page-fault count is available", std::to_string(counters.PageFaultCount));
}

void check_system() {
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    const BOOL known = GlobalMemoryStatusEx(&memory);
#if defined(_WIN32) || defined(__linux__)
    check(known != 0, "system memory is observable");
    if (known) {
        check(memory.ullTotalPhys > 0 && memory.ullAvailPhys <= memory.ullTotalPhys,
              "available memory does not exceed total",
              std::to_string(memory.ullAvailPhys >> 20) + " of " + std::to_string(memory.ullTotalPhys >> 20) + " MiB");
        check(memory.dwMemoryLoad <= 100, "memory load is a percentage",
              std::to_string(memory.dwMemoryLoad) + "%");
    }
#else
    // Darwin: the shim refuses rather than inventing an "available" figure that
    // a preflight would then trust.
    check(known == 0, "system memory reports unavailability instead of a made-up total");
#endif
    FILETIME idle{}, kernel{}, user{};
    const BOOL times = GetSystemTimes(&idle, &kernel, &user);
#if defined(_WIN32) || defined(__linux__)
    check(times != 0, "cumulative system times are observable");
    if (times) {
        const unsigned long long kernel_ticks = (static_cast<unsigned long long>(kernel.dwHighDateTime) << 32)
                                                | kernel.dwLowDateTime;
        const unsigned long long idle_ticks = (static_cast<unsigned long long>(idle.dwHighDateTime) << 32)
                                              | idle.dwLowDateTime;
        check(kernel_ticks >= idle_ticks, "kernel time includes idle time");
    }
#else
    check(times == 0, "cumulative system times report unavailability rather than zeroes");
#endif
}

void check_modules() {
    // The probes find the library they are executing in by name and bind their
    // observer through it. A lookup that silently returns null would turn the
    // measurement into "the tap was never registered", which is the failure this
    // shim must not introduce quietly.
#if defined(_WIN32)
    const HMODULE kernel = GetModuleHandleA("kernel32.dll");
    check(kernel != nullptr, "a module already loaded is found by name");
    check(GetProcAddress(kernel, "Sleep") != nullptr, "an exported symbol is found");
    check(GetModuleHandleA("no-such-module-here.dll") == nullptr,
          "an unloaded module is reported missing rather than loaded");
#else
    // No translation for a name that is already a POSIX library name.
    check(portable_library_name("libc.so.6") == "libc.so.6",
          "a POSIX library name is passed through unchanged");
#if defined(__APPLE__)
    check(portable_library_name("ggml-cpu.dll") == "libggml-cpu.dylib",
          "a Win32 library name maps to the host's");
#else
    check(portable_library_name("ggml-cpu.dll") == "libggml-cpu.so",
          "a Win32 library name maps to the host's");
#endif
#if defined(__APPLE__)
    // macOS keeps no `libc.so`; libSystem is the library every process maps.
    const HMODULE libc = GetModuleHandleA("libSystem.B.dylib");
    check(libc != nullptr, "a library already mapped is found by name");
    check(GetProcAddress(libc, "malloc") != nullptr, "an exported symbol is found");
#else
    const HMODULE libc = GetModuleHandleA("libc.so.6");
    check(libc != nullptr, "a library already mapped is found by name");
    check(GetProcAddress(libc, "malloc") != nullptr, "an exported symbol is found");
#if defined(__linux__)
    // A library opened by absolute path is only matchable through the map when
    // it carries no DT_SONAME; this is the lookup that covers that case.
    const std::string mapped = portable_mapped_path("libc.so.6");
    const std::string suffix = "/libc.so.6";
    check(mapped.size() > suffix.size()
              && mapped.compare(mapped.size() - suffix.size(), suffix.size(), suffix) == 0,
          "the map names the path a library was loaded from", mapped);
    check(portable_mapped_path("libno-such-module-here.so").empty(),
          "a library that is not mapped has no path in the map");
#endif
#endif
    check(GetModuleHandleA("no-such-module-here.dll") == nullptr,
          "an unmapped module is reported missing rather than loaded");
    // The miss must leave the process unchanged: a lookup that quietly loaded
    // the library would bind the tap to a second copy.
#if defined(__linux__)
    check(portable_mapped_path("libno-such-module-here.so").empty(),
          "a failed lookup did not load anything");
#endif
#endif
}

void check_files() {
    // The probes write their result with CREATE_NEW, so a second result can never
    // overwrite a first one. That guarantee is the whole point of the shim here,
    // and it is silent when wrong: an ordinary create would succeed and the
    // earlier receipt would disappear.
    const std::string path = "portable-selftest-" + std::to_string(GetCurrentProcessId()) + ".tmp";
    std::remove(path.c_str());
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                    CREATE_NEW, FILE_ATTRIBUTE_NORMAL, 0);
    check(file != INVALID_HANDLE_VALUE, "CREATE_NEW opens a file that does not exist");
    DWORD written = 0;
    const char payload[] = "first result\n";
    // Evaluated first, so the message reports the write this check is about
    // rather than the initial value of `written`.
    const BOOL wrote = WriteFile(file, payload, sizeof(payload) - 1, &written, nullptr);
    check(wrote != 0 && written == sizeof(payload) - 1,
          "WriteFile reports every byte", std::to_string(written) + " bytes");
    check(CloseHandle(file) != 0, "CloseHandle closes");
    const HANDLE again = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                     CREATE_NEW, FILE_ATTRIBUTE_NORMAL, 0);
    check(again == INVALID_HANDLE_VALUE, "CREATE_NEW refuses an existing file");
    const HANDLE reader = CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    char read_back[sizeof(payload)] = {0};
    DWORD got = 0;
    check(reader != INVALID_HANDLE_VALUE && ReadFile(reader, read_back, sizeof(payload) - 1, &got, nullptr) != 0
              && got == sizeof(payload) - 1 && std::string(read_back) == payload,
          "the bytes read back are the bytes written");
    check(CloseHandle(reader) != 0, "the reader closes");
    check(std::remove(path.c_str()) == 0, "the temporary file is removed");
}

}  // namespace

int main() {
    std::printf("platform: %s\n", PORTABLE_CLOCK_ID);
    check_sha256();
    check_clock();
    check_environment();
    check_files();
    check_modules();
    check_affinity();
    check_accounting();
    check_system();
    std::printf("%d failure(s)\n", failures);
    return failures;
}
