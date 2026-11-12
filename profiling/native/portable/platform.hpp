// The Win32 surface the native probes call, supplied on a POSIX host.
//
// The measurement probes were written against Win32 for one machine, and most of
// the generated Executor already branches on `_WIN32` (its clock, its exclusive
// output creation and its private wait all have a POSIX side). What remained
// unguarded was a handful of calls in the hand-written files: affinity,
// thread/process CPU accounting, the process and system counters, `Sleep`,
// `_putenv_s` and a SHA-256 provider. This header supplies exactly those names
// and nothing else, so the same sources compile into the same instrument on
// Linux, macOS and Windows instead of into a second implementation of it.
//
// Two calls cannot be emulated and say so rather than inventing a number:
//
//   * QueryThreadCycleTime -- POSIX has no per-thread cycle counter. It reports
//     unavailability (returns 0), so every caller records `cycles_ok=false` and
//     the receipt carries null where a Windows receipt carries a cycle count.
//     A fabricated value would be indistinguishable from a measurement.
//
//   * Affinity on macOS -- `sched_setaffinity` does not exist there and
//     THREAD_AFFINITY_POLICY is a scheduling hint, not an enforcement. So
//     PORTABLE_AFFINITY is 0 on Darwin and the probes refuse the protocol with
//     a message instead of reporting `"enforced": true` about something that is
//     not enforced. See `apply_process_mask` in depth_probe.cpp.
//
// The mask width is 64 bits everywhere. A host with more than 64 online CPUs
// cannot be named by one `DWORD_PTR`, so the affinity calls fail instead of
// truncating: a partial mask silently measured on the wrong set of cores is
// exactly the class of error these probes exist to refuse.
#pragma once

#if defined(_WIN32)

// The generated Executor fragments call `std::max`. A real <windows.h>
// defines `max` and `min` as function-like macros, which turns the
// qualified call into `std::` followed by an expansion. The macros are
// suppressed here, at the first inclusion, so every later include is
// covered by the header's own #ifndef NOMINMAX.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#define PORTABLE_AFFINITY 1
#define PORTABLE_CLOCK_ID "windows_qpc"
#define PORTABLE_PHYSICAL_CLOCK_ID "windows_qpc_floor_ns"
#define CLOCK_ORIGIN "absolute_qpc_zero"
#define PORTABLE_AFFINITY_SCOPE "owned process only"

#else

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <string>

#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <dlfcn.h>
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#else
#include <sched.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace portable_win32 {

using DWORD = std::uint32_t;
using DWORD_PTR = std::uintptr_t;
using ULONG = unsigned long;
using ULONG64 = unsigned long long;
using ULONGLONG = unsigned long long;
using UCHAR = unsigned char;
using PUCHAR = unsigned char *;
using BOOL = int;
using HANDLE = std::intptr_t;
using HMODULE = void *;
using WORD = std::uint16_t;

#if !defined(TRUE)
constexpr BOOL TRUE = 1;
#endif
#if !defined(FALSE)
constexpr BOOL FALSE = 0;
#endif

// The Win32 structures these probes read. Only the fields a caller touches are
// present; every caller sets `cb`/`dwLength` first and every implementation
// ignores it, which is also what Win32 does.
struct LARGE_INTEGER { long long QuadPart = 0; };
struct FILETIME {
    DWORD dwLowDateTime = 0;
    DWORD dwHighDateTime = 0;
};
struct GROUP_AFFINITY {
    DWORD_PTR Mask = 0;
    WORD Group = 0;
    WORD Reserved[3] = {0, 0, 0};
};
struct PROCESS_MEMORY_COUNTERS {
    DWORD cb = 0;
    DWORD PageFaultCount = 0;
    std::size_t PeakWorkingSetSize = 0;
    std::size_t WorkingSetSize = 0;
    std::size_t QuotaPeakPagedPoolUsage = 0;
    std::size_t QuotaPagedPoolUsage = 0;
    std::size_t QuotaPeakNonPagedPoolUsage = 0;
    std::size_t QuotaNonPagedPoolUsage = 0;
    std::size_t PagefileUsage = 0;
    std::size_t PeakPagefileUsage = 0;
};
struct MEMORYSTATUSEX {
    DWORD dwLength = 0;
    DWORD dwMemoryLoad = 0;
    unsigned long long ullTotalPhys = 0;
    unsigned long long ullAvailPhys = 0;
    unsigned long long ullTotalPageFile = 0;
    unsigned long long ullAvailPageFile = 0;
    unsigned long long ullTotalVirtual = 0;
    unsigned long long ullAvailVirtual = 0;
    unsigned long long ullAvailExtendedVirtual = 0;
};

inline HANDLE GetCurrentProcess() { return static_cast<HANDLE>(-1); }
inline HANDLE GetCurrentThread() { return static_cast<HANDLE>(-2); }
inline DWORD GetCurrentProcessId() { return static_cast<DWORD>(::getpid()); }
inline DWORD GetCurrentThreadId() {
#if defined(__linux__)
    return static_cast<DWORD>(::syscall(SYS_gettid));
#elif defined(__APPLE__)
    return static_cast<DWORD>(reinterpret_cast<std::uintptr_t>(::pthread_self()));
#else
    return static_cast<DWORD>(::getpid());
#endif
}

// QPC's contract here is "a monotonic counter whose frequency the caller reads
// once". CLOCK_MONOTONIC at a declared 1 ns tick is that, and it is the same
// clock the generated Executor already uses on POSIX, so action stamps and
// block stamps stay on one timeline.
inline BOOL QueryPerformanceFrequency(LARGE_INTEGER *value) {
    if (value == nullptr) return FALSE;
    value->QuadPart = 1000000000LL;
    return TRUE;
}
inline BOOL QueryPerformanceCounter(LARGE_INTEGER *value) {
    if (value == nullptr) return FALSE;
    timespec now{};
    if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0) return FALSE;
    value->QuadPart = static_cast<long long>(now.tv_sec) * 1000000000LL + static_cast<long long>(now.tv_nsec);
    return TRUE;
}

inline void Sleep(DWORD milliseconds) {
    timespec request{static_cast<time_t>(milliseconds / 1000),
                     static_cast<long>(milliseconds % 1000) * 1000000L};
    while (::nanosleep(&request, &request) != 0 && errno == EINTR) {}
}

// `_putenv_s(name, "")` deletes the variable on Win32; setenv/unsetenv split that
// in two, so the empty case is spelled out rather than left to setenv's.
inline int _putenv_s(const char *name, const char *value) {
    if (name == nullptr) return EINVAL;
    if (value == nullptr || *value == '\0')
        return ::unsetenv(name) == 0 ? 0 : errno;
    return ::setenv(name, value, 1) == 0 ? 0 : errno;
}

// --- exclusive output creation ---------------------------------------------
//
// The native probes write their result with CreateFileW(CREATE_NEW), so a file
// that already exists is never overwritten: a second result can never be
// mistaken for the first run's. `O_CREAT|O_EXCL` is the same guarantee, so the
// three calls a caller makes are supplied rather than the call site branching.
// `path.c_str()` is `wchar_t*` on Windows and `char*` here, which is why the
// shim takes the narrow spelling: it is the same expression at both call sites.
constexpr DWORD GENERIC_READ = 0x80000000UL;
constexpr DWORD GENERIC_WRITE = 0x40000000UL;
constexpr DWORD CREATE_NEW = 1;
constexpr DWORD CREATE_ALWAYS = 2;
constexpr DWORD OPEN_EXISTING = 3;
constexpr DWORD FILE_ATTRIBUTE_NORMAL = 0x80;
constexpr HANDLE INVALID_HANDLE_VALUE = static_cast<HANDLE>(-1);

// The template parameter is the template handle: Win32's seventh argument is an
// `HANDLE` (a pointer, so `nullptr` types), and every call site in the probes
// passes null. Naming it `HANDLE` here would reject the very spelling the
// Windows build accepts.
template <typename TemplateHandle>
inline HANDLE CreateFileW(const char *path, DWORD access, DWORD, void *,
                          DWORD disposition, DWORD, TemplateHandle) {
    if (path == nullptr) return INVALID_HANDLE_VALUE;
    int flags = (access & GENERIC_WRITE) ? O_WRONLY : O_RDONLY;
    if (disposition == CREATE_NEW) flags |= O_CREAT | O_EXCL;
    else if (disposition == CREATE_ALWAYS) flags |= O_CREAT | O_TRUNC;
    const int descriptor = ::open(path, flags, 0644);
    return descriptor < 0 ? INVALID_HANDLE_VALUE : static_cast<HANDLE>(descriptor);
}

inline BOOL ReadFile(HANDLE handle, void *data, DWORD count, DWORD *read, void *) {
    const ssize_t got = ::read(static_cast<int>(handle), data, count);
    if (read != nullptr) *read = got < 0 ? 0 : static_cast<DWORD>(got);
    return got < 0 ? FALSE : TRUE;
}

inline BOOL WriteFile(HANDLE handle, const void *data, DWORD count, DWORD *written, void *) {
    const ssize_t put = ::write(static_cast<int>(handle), data, count);
    if (written != nullptr) *written = put < 0 ? 0 : static_cast<DWORD>(put);
    return put < 0 ? FALSE : TRUE;
}

inline BOOL CloseHandle(HANDLE handle) {
    // The -1/-2 pseudo-handles name the process and the thread; they were never
    // opened and Win32 does not close them either.
    if (handle == static_cast<HANDLE>(-1) || handle == static_cast<HANDLE>(-2)) return TRUE;
    return ::close(static_cast<int>(handle)) == 0 ? TRUE : FALSE;
}

// How many CPUs the machine reports as online. This is *not* the set the
// affinity calls use -- see `GetProcessAffinityMask` below for why a host with
// cgroups, `taskset` or a container cpuset answers a different question -- but a
// receipt that records it lets a reader tell "20 usable cores" from "20 of 72".
inline unsigned long long portable_online_cpus() {
    const long count = ::sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? static_cast<unsigned long long>(count) : 0;
}

// --- process and thread CPU accounting ------------------------------------
//
// The POSIX analogue of GetProcessTimes/GetThreadTimes is the per-id CPU clock,
// which is a total in nanoseconds rather than the 100 ns tick Win32 reports.
// Each caller sums `kernel + user`, so the total goes in `user` and `kernel`
// carries zero: the sum is exact and no caller can read a split that was never
// measured.
inline BOOL portable_cpu_time(clockid_t id, FILETIME *kernel, FILETIME *user) {
    timespec now{};
    if (::clock_gettime(id, &now) != 0) return FALSE;
    const unsigned long long ticks = (static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL
                                      + static_cast<unsigned long long>(now.tv_nsec)) / 100ULL;
    if (kernel != nullptr) {
        kernel->dwLowDateTime = 0;
        kernel->dwHighDateTime = 0;
    }
    if (user != nullptr) {
        user->dwLowDateTime = static_cast<DWORD>(ticks & 0xFFFFFFFFULL);
        user->dwHighDateTime = static_cast<DWORD>(ticks >> 32);
    }
    return TRUE;
}
inline BOOL GetThreadTimes(HANDLE, FILETIME *created, FILETIME *exited, FILETIME *kernel, FILETIME *user) {
    for (FILETIME *field : {created, exited})
        if (field != nullptr) { field->dwLowDateTime = 0; field->dwHighDateTime = 0; }
    return portable_cpu_time(CLOCK_THREAD_CPUTIME_ID, kernel, user);
}
inline BOOL GetProcessTimes(HANDLE, FILETIME *created, FILETIME *exited, FILETIME *kernel, FILETIME *user) {
    for (FILETIME *field : {created, exited})
        if (field != nullptr) { field->dwLowDateTime = 0; field->dwHighDateTime = 0; }
    return portable_cpu_time(CLOCK_PROCESS_CPUTIME_ID, kernel, user);
}

// No per-thread cycle counter on POSIX. Reporting 0 (failure) makes the caller
// record null; reporting a converted nanosecond count as "cycles" would publish
// a unit that does not exist on this machine.
inline BOOL QueryThreadCycleTime(HANDLE, ULONG64 *cycles) {
    if (cycles != nullptr) *cycles = 0;
    return FALSE;
}

inline BOOL GetProcessMemoryInfo(HANDLE, PROCESS_MEMORY_COUNTERS *counters, DWORD) {
    if (counters == nullptr) return FALSE;
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) return FALSE;
    counters->PageFaultCount = static_cast<DWORD>(usage.ru_minflt + usage.ru_majflt);
    return TRUE;
}

inline BOOL GlobalMemoryStatusEx(MEMORYSTATUSEX *memory) {
    if (memory == nullptr) return FALSE;
#if !defined(__linux__)
    // Darwin has no /proc and does not implement `_SC_AVPHYS_PAGES`; reporting a
    // total with an invented "available" would feed a preflight that decides
    // whether a run is safe. It says unavailable instead.
    (void)memory;
    return FALSE;
#else
    unsigned long long total = 0, available = 0;
    // MemAvailable is the kernel's own estimate of what a new workload can take;
    // the free-page count alone excludes essentially all reclaimable cache and
    // would make every preflight look starved.
    std::ifstream info("/proc/meminfo");
    std::string line;
    while (info.good() && std::getline(info, line)) {
        unsigned long long kilobytes = 0;
        if (std::sscanf(line.c_str(), "MemTotal: %llu kB", &kilobytes) == 1) total = kilobytes * 1024ULL;
        else if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kilobytes) == 1) available = kilobytes * 1024ULL;
        if (total != 0 && available != 0) break;
    }
    if (total == 0) {
        const long pages = ::sysconf(_SC_PHYS_PAGES);
        const long page = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && page > 0) total = static_cast<unsigned long long>(pages) * static_cast<unsigned long long>(page);
        if (total != 0) available = total;   // unresolvable availability is not zero
    }
    if (total == 0) return FALSE;
    memory->ullTotalPhys = total;
    memory->ullAvailPhys = available < total ? available : total;
    memory->dwMemoryLoad = static_cast<DWORD>((total - memory->ullAvailPhys) * 100ULL / total);
    return TRUE;
#endif
}

inline void portable_filetime(FILETIME *field, unsigned long long ticks) {
    if (field == nullptr) return;
    field->dwLowDateTime = static_cast<DWORD>(ticks & 0xFFFFFFFFULL);
    field->dwHighDateTime = static_cast<DWORD>(ticks >> 32);
}
inline BOOL GetSystemTimes(FILETIME *idle, FILETIME *kernel, FILETIME *user) {
#if defined(__linux__)
    std::ifstream stat("/proc/stat");
    if (!stat.good()) return FALSE;
    std::string label;
    unsigned long long field[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    stat >> label;
    if (label != "cpu") return FALSE;
    for (unsigned long long &value : field) stat >> value;
    // USER_HZ is 100 on every configuration these probes run on, so a jiffy is
    // 10 ms and therefore 100000 units of the 100 ns Win32 reports.
    portable_filetime(idle, field[3] * 100000ULL);
    portable_filetime(user, (field[0] + field[1]) * 100000ULL);
    // Win32's kernel time includes idle, not iowait; iowait is idle-while-blocked
    // so it belongs on this side of the split, and the receipt names the scope.
    portable_filetime(kernel, (field[2] + field[3] + field[4] + field[5] + field[6] + field[7]) * 100000ULL);
    return TRUE;
#else
    (void)idle; (void)kernel; (void)user;
    return FALSE;
#endif
}

#if defined(__linux__)

#define PORTABLE_AFFINITY 1
#define PORTABLE_AFFINITY_SCOPE \
    "calling thread before any worker exists; GGML pool threads inherit the mask"

inline bool portable_mask_of(const cpu_set_t &set, DWORD_PTR *mask) {
    unsigned long long value = 0;
    for (int cpu = 64; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) return false;   // wider than one DWORD_PTR
    for (int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) value |= (1ULL << cpu);
    if (mask != nullptr) *mask = static_cast<DWORD_PTR>(value);
    return true;
}
inline bool portable_set_of(DWORD_PTR mask, cpu_set_t *set) {
    if (mask == 0) return false;
    CPU_ZERO(set);
    for (int cpu = 0; cpu < 64; ++cpu)
        if ((mask >> cpu) & 1) CPU_SET(cpu, set);
    return true;
}

// The mask a process may run on, and the set of online CPUs behind it.
//
// On Linux the CPUs a process may run on are a subset -- and sometimes a
// non-contiguous one -- of the CPUs the machine reports online: a cgroup cpuset,
// `taskset`, a scheduler reservation, WSL's own CPU limit and a container's
// `--cpuset-cpus=24-47` all produce that. `_SC_NPROCESSORS_ONLN` answers the
// other question, so a mask built as "the low N bits" can name a CPU this
// process is not allowed on; `sched_setaffinity` then returns EINVAL and the
// probe reports "cannot set owned process mask", which reads as a broken
// measurement rather than as a host whose CPU set is not 0..N-1.
//
// So both masks are the allowed set (`sched_getaffinity`): the protocol asks for
// "every CPU this host offers this process", and this is that set, gaps and all.
// On a host where nothing is restricted it is `(1 << n) - 1`, exactly what the
// published receipts carry.
inline BOOL GetProcessAffinityMask(HANDLE, DWORD_PTR *process, DWORD_PTR *system) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) return FALSE;
    DWORD_PTR current = 0;
    if (!portable_mask_of(set, &current)) return FALSE;
    if (process != nullptr) *process = current;
    if (system != nullptr) *system = current;
    return TRUE;
}

// Process-wide on Win32, thread-wide here. Callers apply this before any
// backend exists (the receipt says so), and a POSIX thread's workers inherit its
// mask, so the two agree in the only window this is used.
inline BOOL SetProcessAffinityMask(HANDLE, DWORD_PTR mask) {
    cpu_set_t set;
    if (!portable_set_of(mask, &set)) return FALSE;
    return ::sched_setaffinity(0, sizeof(set), &set) == 0 ? TRUE : FALSE;
}
inline BOOL GetThreadGroupAffinity(HANDLE, GROUP_AFFINITY *group) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) return FALSE;
    DWORD_PTR mask = 0;
    if (!portable_mask_of(set, &mask)) return FALSE;
    if (group != nullptr) {
        group->Group = 0;   // POSIX has one affinity group
        group->Mask = mask;
    }
    return TRUE;
}
inline DWORD_PTR SetThreadAffinityMask(HANDLE, DWORD_PTR mask) {
    cpu_set_t previous;
    CPU_ZERO(&previous);
    if (::sched_getaffinity(0, sizeof(previous), &previous) != 0) return 0;
    DWORD_PTR before = 0;
    if (!portable_mask_of(previous, &before)) return 0;
    cpu_set_t wanted;
    if (!portable_set_of(mask, &wanted)) return 0;
    if (::sched_setaffinity(0, sizeof(wanted), &wanted) != 0) return 0;
    return before;
}

#else   // !__linux__: Darwin reports that the protocol cannot be honoured.

#define PORTABLE_AFFINITY 0
#define PORTABLE_AFFINITY_SCOPE "unavailable: this platform exposes no enforced CPU affinity"

inline BOOL GetProcessAffinityMask(HANDLE, DWORD_PTR *, DWORD_PTR *) { return FALSE; }
inline BOOL SetProcessAffinityMask(HANDLE, DWORD_PTR) { return FALSE; }
inline BOOL GetThreadGroupAffinity(HANDLE, GROUP_AFFINITY *) { return FALSE; }
inline DWORD_PTR SetThreadAffinityMask(HANDLE, DWORD_PTR) { return 0; }

#endif  // __linux__

// --- module lookup ---------------------------------------------------------
//
// The probes bind their observers with `GetModuleHandle("ggml-cpu.dll")` and
// `GetProcAddress`: the tap has to be registered on the library the process is
// actually executing, so opening a second copy by path would observe nothing.
// `dlopen` with RTLD_NOLOAD is that same query -- it fails rather than loading
// -- and the name it is asked for is the Windows one, so it is translated.
inline std::string portable_library_name(const char *name) {
    std::string text = name ? name : "";
    const std::string suffix = ".dll";
    if (text.size() > suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
        text.resize(text.size() - suffix.size());
#if defined(__APPLE__)
        return "lib" + text + ".dylib";
#else
        return "lib" + text + ".so";
#endif
    }
    return text;
}

// The path a library was mapped from, if this process has it mapped. Win32's
// `GetModuleHandle` answers by module *name*, and a loader that opened the file
// by absolute path registers no basename for `dlopen(RTLD_NOLOAD)` to match
// unless the library carries a DT_SONAME. Reading the map is how the same
// question is answered without asking the loader to guess.
#if defined(__linux__)
// A library whose SONAME carries a major version is mapped as
// `libggml-base.so.0` while the probe asks for the Windows name and gets
// `libggml-base.so`: one module, two spellings, and NOLOAD by the unversioned
// name finds nothing. A name that merely starts the same way (`.so.1` beside
// `.so.0`) is a *different* module, so the match is the version suffix alone.
inline bool portable_same_library(const std::string &mapped, const std::string &library) {
    if (mapped == library) return true;
    if (mapped.size() <= library.size() + 1) return false;
    if (mapped.compare(0, library.size(), library) != 0) return false;
    if (mapped[library.size()] != '.') return false;
    return mapped.find_first_not_of("0123456789.", library.size() + 1) == std::string::npos;
}

inline std::string portable_mapped_path(const std::string &library) {
    FILE *maps = std::fopen("/proc/self/maps", "r");
    if (maps == nullptr) return std::string();
    std::string found;
    bool ambiguous = false;
    char line[4096];
    while (!ambiguous && std::fgets(line, sizeof(line), maps) != nullptr) {
        // "7f..-7f.. r-xp 00000000 fd:01 12345 /path/to/libggml-cpu.so.0"
        const char *path = std::strchr(line, '/');
        if (path == nullptr) continue;
        std::string mapped(path);
        while (!mapped.empty() && (mapped.back() == '\n' || mapped.back() == ' ')) mapped.pop_back();
        // A library replaced on disk while it was mapped is reported "(deleted)".
        const std::string deleted = " (deleted)";
        if (mapped.size() > deleted.size()
            && mapped.compare(mapped.size() - deleted.size(), deleted.size(), deleted) == 0)
            mapped.resize(mapped.size() - deleted.size());
        const std::size_t slash = mapped.find_last_of('/');
        const std::string base = slash == std::string::npos ? mapped : mapped.substr(slash + 1);
        if (!portable_same_library(base, library)) continue;
        // Two different paths carrying the same name is a question this cannot
        // answer; picking one would bind the tap to a copy that may not be the
        // one computing, which is the failure mode this whole query exists to
        // avoid. Nothing is returned and the caller reports the miss.
        if (found.empty()) found = mapped;
        else if (found != mapped) { found.clear(); ambiguous = true; }
    }
    std::fclose(maps);
    return found;
}
#endif

inline HMODULE GetModuleHandleA(const char *name) {
    const std::string library = portable_library_name(name);
    if (library.empty()) return nullptr;
    // RTLD_NOLOAD keeps Win32's promise ("already loaded, never load one"). The
    // reference it adds is never released: the process itself holds the library
    // for as long as the probe can be asking.
    void *module = ::dlopen(library.c_str(), RTLD_LAZY | RTLD_NOLOAD);
#if defined(__linux__)
    if (module == nullptr) {
        const std::string mapped = portable_mapped_path(library);
        if (!mapped.empty()) module = ::dlopen(mapped.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    }
#endif
    // Deliberately no loading fallback. Loading the library here would succeed
    // and bind the tap to a *second* copy, which measures nothing while looking
    // like a working measurement; Win32 reports the miss, and so does this.
    return module;
}

// The names the probes pass are ASCII literals in wide form; the low byte is the
// character. A name that is not ASCII never reaches here.
inline HMODULE GetModuleHandleW(const wchar_t *name) {
    std::string narrow;
    for (; name != nullptr && *name != L'\0'; ++name)
        narrow.push_back(static_cast<char>(*name & 0xFF));
    return GetModuleHandleA(narrow.c_str());
}

inline void *GetProcAddress(HMODULE module, const char *name) {
    if (module == nullptr || name == nullptr) return nullptr;
    return ::dlsym(module, name);
}

}  // namespace portable_win32

// The probes call the Win32 spellings unqualified, exactly as they do on
// Windows, so the whole shim is brought into the global namespace the way
// <windows.h> brings its own. A named list here would drift: every function
// added below would have to be remembered twice.
using namespace portable_win32;

#define PORTABLE_CLOCK_ID "posix_clock_monotonic"
#define PORTABLE_PHYSICAL_CLOCK_ID "posix_clock_monotonic_floor_ns"
#define CLOCK_ORIGIN "absolute_monotonic_zero"

#endif  // _WIN32
