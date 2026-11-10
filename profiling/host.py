"""What this machine is, in the few respects the measurement protocol depends on.

The V-A columns were written on one 32-CPU Windows machine, and three of their
constants were really properties of that machine wearing a protocol's clothes:
the process affinity mask ``0xFFFFFFFF``, the shared-library suffix ``.dll``, and
the virtualenv's ``Scripts``/``bin`` split. Written that way, a second device did
not fail with "this box has 20 cores"; it failed with "requested process mask
must name every online CPU", or with a probe that loaded no libraries, which
reads like a measurement fault rather than a wrong assumption.

Each helper here turns one of those into a question the machine answers, and
each answer is written into the campaign receipt instead of being assumed. The
one that matters most is `process_affinity_mask`: on the machine the published
table came from it returns exactly ``4294967295``, so nothing measured there
changes, and on a 20-core device it returns ``1048575`` instead of refusing to
run at all.

It lives here rather than beside one column because all three V-A columns --
the model column, the dense column and the e2a column -- start their child the
same way and ask the host the same questions.

Nothing here measures anything, and nothing here is imported by the simulator's
cost model: this is the instrumentation's view of its host.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# The protocol pins the main thread to one fixed CPU so that a price is the
# price of a core rather than of a migration. The index is a choice, not a
# property of the machine -- but it can only be made on a machine that has it.
# The published cohorts wrote this as the mask `4`, which is the bit for CPU 2;
# both spellings name the same core and `main_thread_mask()` returns `4` again.
MAIN_THREAD_CPU = 2

# One DWORD_PTR names 64 CPUs on both hosts the probes build for.
MAX_MASK_BITS = 64


class UnsupportedHost(RuntimeError):
    """The host cannot honour the measurement protocol, with the reason named."""


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def online_cpus() -> int:
    """The number of CPUs the machine reports as online.

    This is the *machine's* answer, not necessarily the set this process may run
    on: see `usable_cpus` for the difference and why the protocol needs the
    second one.
    """
    count = os.cpu_count() or 0
    if count <= 0:
        raise UnsupportedHost("this host reports no online CPUs")
    return count


def usable_cpus() -> tuple[int, ...]:
    """The CPUs this process may run on, as indices -- which is not the same set.

    A 20-core device answers 20; a container started with `--cpuset-cpus=24-47`
    answers 24..47 even though the machine reports 64 online, and WSL answers its
    configured subset of the host's CPUs. The protocol's mask is "every CPU this
    host offers this process", so this is the set it has to name: a mask built as
    "the low N bits" would name CPUs `sched_setaffinity` refuses, and the probe
    would report a failed pin on a host that is perfectly able to be pinned.

    The probes ask the same question the same way -- `GetProcessAffinityMask` in
    `profiling/native/portable/platform.hpp` reads `sched_getaffinity` on Linux
    and the Win32 call on Windows -- so the number this returns is the number the
    probe will accept.
    """
    allowed = getattr(os, "sched_getaffinity", None)
    if allowed is not None and not is_windows():
        try:
            cpus = sorted(allowed(0))
        except OSError:                                   # pragma: no cover - defensive
            cpus = []
        if cpus:
            return tuple(cpus)
    return tuple(range(online_cpus()))


def process_affinity_mask() -> int:
    """Every CPU this host offers this process, as the bitmask the probe enforces.

    The probe refuses a mask that is not exactly the set it observes, so this is
    not a hint: it is the number the request must carry for the run to proceed.
    """
    cpus = usable_cpus()
    if cpus[-1] >= MAX_MASK_BITS:
        raise UnsupportedHost(
            "this host offers %d CPUs, numbered up to %d, and one affinity mask names "
            "at most %d; the probes would have to truncate it, which would measure on "
            "a smaller core set than the receipt claims"
            % (len(cpus), cpus[-1], MAX_MASK_BITS))
    return sum(1 << cpu for cpu in cpus)


def main_thread_mask() -> int:
    """The mask pinning the main thread, and a refusal if this host lacks the core.

    Returns the protocol's own constant (`4`) on every host that has CPU 2, which
    is what makes the published receipts comparable with a new device's. A host
    whose usable set starts above 2 -- a container pinned to cores 24..47, say --
    has no CPU 2 to pin, and `sched_setaffinity` would fail with EINVAL; it is
    refused here with that named instead.
    """
    cpus = usable_cpus()
    if MAIN_THREAD_CPU not in cpus:
        raise UnsupportedHost(
            "the protocol pins the main thread to CPU %d and this host offers %s; "
            "run on a host that includes it, or change MAIN_THREAD_CPU for the cohort"
            % (MAIN_THREAD_CPU, "CPUs %d-%d" % (cpus[0], cpus[-1]) if cpus else "no CPUs"))
    return 1 << MAIN_THREAD_CPU


def shared_library_pattern() -> str:
    """The glob that finds the native libraries a probe loads from a directory."""
    if is_windows():
        return "*.dll"
    if is_macos():
        return "*.dylib"
    return "*.so*"


def shared_library_variable() -> str:
    """The environment variable the loader searches for the libraries."""
    if is_windows():
        return "PATH"
    if is_macos():
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def executable_name(stem: str) -> str:
    """`depth-probe` on POSIX, `depth-probe.exe` on Windows."""
    return stem + ".exe" if is_windows() else stem


# Suffixes that belong to a loader other than this host's. A file with one of
# them is not a program here whatever its name says.
_FOREIGN_LIBRARY_SUFFIXES = (".dll", ".dylib")


def _is_shared_library(name: str) -> bool:
    return name.endswith(_FOREIGN_LIBRARY_SUFFIXES) or ".so" in name


def windows_image(path) -> bool:
    """Whether the file starts with the MS-DOS header every Windows image has."""
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"MZ"
    except OSError:
        return False


def foreign_artifact_reason(path) -> str | None:
    """Why the columns could not use this artifact on this host, or None.

    A build directory and an executable are portable in name only. A Windows
    build unpacked on a Linux host -- a shared drive, a copied artifact
    directory, a WSL checkout whose `build/` came from the machine that made the
    table -- holds every file a preflight checks for and not one thing the
    loader can open, so the run dies inside a probe at its first dlopen with a
    message about a missing DLL. Naming the platform here turns that into a
    refusal that says which artifact and why, before anything is timed.

    Only positive evidence is refused: an absent or not-yet-built directory
    answers None, because "missing" is the preflight's other sentence and the
    two must not be confused.
    """
    path = Path(path)
    if path.is_file():
        if not is_windows() and windows_image(path):
            return "a Windows executable image; this host has no loader for it"
        if is_windows() and _is_shared_library(path.name) and not path.name.endswith(".dll"):
            return "a POSIX shared library; this host has no loader for it"
        return None
    if not path.is_dir():
        return None
    pattern = shared_library_pattern()
    if any(path.rglob(pattern)):
        return None
    kinds = set()
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if ".so" in child.name:
            kinds.add(".so")
        elif child.name.endswith(".dll"):
            kinds.add(".dll")
        elif child.name.endswith(".dylib"):
            kinds.add(".dylib")
    if not kinds:
        return None
    return ("it holds " + ", ".join(sorted(kinds)) + " libraries and this host loads "
            + pattern + ": it was built on another platform")


def measurement_clock_id() -> str:
    """The clock name a probe's receipt carries on this host.

    `profiling/native/portable/platform.hpp` supplies `QueryPerformanceCounter`
    as CLOCK_MONOTONIC at a declared 1 ns tick on POSIX, and names that clock
    `posix_clock_monotonic`; on Windows the same source is QPC and the name is
    `windows_qpc`. A reader that checks a receipt against a clock name has to ask
    this, not assume the machine that published the table.
    """
    return "windows_qpc" if is_windows() else "posix_clock_monotonic"


def measurement_executables() -> tuple[str, ...]:
    """Process names that time something, in the spelling this host reports.

    A leftover probe keeps the machine in the wrong state even after its parent
    is gone, so the campaign refuses to start while one is resident. Comparing a
    POSIX process name against a list of `.exe` names would find nothing and
    would let exactly that case through.
    """
    return tuple(executable_name(stem) for stem in (
        "real-moe-runtime-window-probe", "depth-probe", "steady-costs-probe",
        "llama-server", "llama-cli"))


def cuda_toolkit_roots() -> list[str]:
    """Where an installed CUDA toolkit is normally found, best guess first."""
    if is_windows():
        return ["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"]
    if is_macos():
        return []          # CUDA has not been available on macOS since 10.13
    return ["/usr/local", "/opt/cuda", "/usr/lib/cuda"]


def installed_toolkits() -> list[Path]:
    """Every CUDA toolkit root on this host, newest first.

    Windows keeps them as `CUDA/v12.6`; Linux as `/usr/local/cuda-12.6` with
    `/usr/local/cuda` a link to one of them; a distribution package installs at
    `/opt/cuda` or `/usr/lib/cuda` itself. All three are the same question -- a
    directory with a `bin` and a version somewhere in its name -- so one loop
    over `cuda_toolkit_roots` answers it on every host.
    """
    found = set()
    for base in cuda_toolkit_roots():
        directory = Path(base)
        if not directory.is_dir():
            continue
        if (directory / "bin").is_dir():
            found.add(directory)                       # /opt/cuda, /usr/local/cuda
        for child in directory.iterdir():
            if child.is_dir() and (child / "bin").is_dir():
                found.add(child)                       # CUDA/v12.6, cuda-12.6
    return sorted(found,
                  key=lambda p: (tuple(int(n) for n in re.findall(r"\d+", p.name)), str(p)),
                  reverse=True)


def newest_toolkit() -> Path | None:
    installed = installed_toolkits()
    return installed[0] if installed else None


def cupti_library_dir(root) -> Path:
    """The directory holding CUPTI's import library for this toolkit layout.

    x86_64 keeps it in `extras/CUPTI/lib64` on both Windows and Linux; an aarch64
    install (Jetson, and DGX Spark's Grace part) also ships it under
    `targets/aarch64-linux/lib`.
    """
    root = Path(root)
    candidates = [root / "extras" / "CUPTI" / "lib64",
                  root / "targets" / "aarch64-linux" / "lib",
                  root / "targets" / "x86_64-linux" / "lib",
                  root / "lib64"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*cupti*")):
            return candidate
    return candidates[0]


def cuda_library_dir(root) -> Path | None:
    """The directory holding a toolkit's shared libraries, on layouts that split them.

    Windows keeps a toolkit's DLLs beside its executables in `bin`, so the
    question has the same answer as `cuda_dirs` and nothing extra is needed
    there: this returns `None`. Linux separates them -- `bin` holds `nvcc` and
    the CUDA runtime while cuBLAS, cuDNN and cuFFT sit in `lib64` -- and an
    aarch64 install puts them under `targets/aarch64-linux/lib`. A probe that
    links ggml-cuda loads those before its first action, so a runtime list
    naming only `bin` lets every child die in the loader with
    `libcublas.so.12: cannot open shared object file`.
    """
    if is_windows():
        return None
    root = Path(root)
    candidates = [root / "targets" / (arch() + "-linux") / "lib",
                  root / "lib64", root / "lib"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("libcublas.so*")):
            return candidate
    return None


def available_memory_bytes() -> int:
    """Bytes an allocating process can still get, asked of the host in its own terms.

    A campaign preflight refuses a run whose column needs more than this, so an
    unavailable answer raises instead of returning a guess: a made-up figure
    would let a run start and then fail by being swapped rather than refused.
    """
    if is_windows():
        import ctypes

        class _Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    'total', 'available', 'page_total', 'page_available',
                    'virtual_total', 'virtual_available', 'ext')]

        state = _Memory()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError('GlobalMemoryStatusEx failed')
        return state.available
    if is_linux():
        # MemAvailable is the kernel's own estimate of what a new allocation can
        # get without swapping; MemFree alone would understate it by the
        # reclaimable page cache.
        with open('/proc/meminfo', encoding='ascii', errors='replace') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
        raise OSError('/proc/meminfo has no MemAvailable')
    # macOS reports free pages, not an available estimate. That is the
    # conservative side of the same question, and refusing a run slightly early
    # is better than admitting one that then swaps.
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


def cuda_available(root=None) -> bool:
    """Whether a toolkit with a `bin` directory looks installed here.

    A build question, not a measurement one: the link line needs `nvcc` and the
    CUPTI import library, and those are installed by a toolkit rather than by a
    driver. Whether a *device* will answer is `nvidia_smi_devices`, and the two
    come apart on exactly one common host -- a container or WSL image where the
    toolkit was installed for building and no GPU was passed through.
    """
    if is_macos():
        return False
    if root is not None:
        return (Path(root) / "bin").is_dir()
    return bool(installed_toolkits())


def nvidia_smi_path() -> str | None:
    """Where the driver's own tool is, or `None` if this host has no such file."""
    name = "nvidia-smi.exe" if is_windows() else "nvidia-smi"
    directories = os.environ.get("PATH", "").split(os.pathsep)
    # `nvidia-smi.exe` normally is on PATH, but a service or a scheduled task can
    # start with a PATH that was never given System32, and then a machine with a
    # card reads as a machine without one.
    if is_windows():
        directories += [str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")]
    else:
        directories += ["/usr/bin", "/usr/sbin", "/usr/local/bin", "/bin"]
    for directory in directories:
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    return None


def nvidia_smi_devices() -> list[str] | None:
    """The GPUs the driver reports, or `None` when this host cannot be asked.

    `None` is *unknown*, not *none*, and the distinction is the whole point: an
    image with a toolkit and no device is a real configuration (this repository
    builds on one), but a machine whose `nvidia-smi` is absent or erroring is
    usually a working GPU host whose PATH or driver is in an unusual state, and
    refusing it would turn a diagnostic into a broken checkout. Only a driver
    that answers, and answers with no GPU, is evidence of no GPU.
    """
    tool = nvidia_smi_path()
    if tool is None:
        return None
    try:
        done = subprocess.run([tool, "-L"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line.strip() for line in done.stdout.splitlines()
            if line.strip().startswith("GPU ")]


def arch() -> str:
    """The instruction set name this build targets, in ggml's spelling.

    It matters in two places and both are about names rather than speed: the
    CUDA toolkit keeps its aarch64 pieces under `targets/aarch64-linux/lib`
    (`cupti_library_dir` asks that separately), and a build receipt that says
    `x86_64` on a Grace box would describe the wrong machine.
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine or "unknown"


#: The ggml backend directory each accelerator's library is named after. One
#: name per accelerator, because it is also the suffix of `libggml-<name>` and
#: the target CMake has to be asked for.
BACKEND_LIBRARY = {"cuda": "ggml-cuda", "metal": "ggml-metal", "vulkan": "ggml-vulkan"}


def backend(root=None) -> str | None:
    """The ggml backend that computes on this host's accelerator, or `None`.

    Deliberately not "does this machine have a GPU": a Mac has one and it is not
    CUDA, which is the whole of the UMA port. The answer names the *backend*,
    because that is what the three probes link, what `runtime_libraries` asks the
    directory for, and what a receipt has to record before anyone compares two
    devices' tables.

    A CUDA toolkit alone does not answer it. The toolkit says what can be
    *built*; a probe needs a device to compute on, and an image that installed
    the toolkit for its link line -- a CI container, a WSL distribution with no
    GPU passed through -- would otherwise be described as a CUDA host right up
    until a probe failed to enumerate one. So the driver is asked too, and only
    a driver that answers with no GPU turns a toolkit into `None`.

    `None` means "no accelerator this repository builds a probe against", which
    is a CPU-only box and a legitimate answer: the software side runs there.
    """
    if is_macos():
        # Every Mac that can run a current Python has Metal; there has been no
        # CUDA since 10.13, so asking for a toolkit first would answer `None`.
        return "metal"
    if not cuda_available(root):
        return None
    devices = nvidia_smi_devices()
    if devices is not None and not devices:
        return None
    return "cuda"


def backend_reason() -> str:
    """Why `backend()` answered what it did, in the sentence a report can print.

    The interesting case is a toolkit with no device: a host that can compile
    every CUDA probe here and is still not one of the machines a V-A column can
    be measured on. Naming both halves is what keeps that from reading as a
    broken install.
    """
    if is_macos():
        return "macOS: the accelerator is the SoC GPU through ggml's Metal backend"
    if not cuda_available():
        return "no CUDA toolkit and no Metal: nothing to build a probe against"
    devices = nvidia_smi_devices()
    if devices is None:
        return ("a CUDA toolkit is installed and `nvidia-smi -L` could not be asked; "
                "the toolkit is taken to mean a device is present")
    if not devices:
        return ("a CUDA toolkit is installed but the driver reports no GPU: "
                "the toolkit can build a probe, this host cannot run one")
    return "the driver reports " + "; ".join(devices)


def unified_memory(accelerator: str | None, machine: str) -> bool | None:
    """Whether that accelerator draws on the host's DRAM instead of its own VRAM.

    ``True``/``False`` when the host's shape decides it, ``None`` when it does not.

    * A Mac's accelerator is the SoC's own GPU, so there is one pool -- on an ARM
      Mac and on an Intel one alike.
    * A CUDA device with an x86_64 host is a card on the far side of PCIe: copies
      move bytes.
    * CUDA on aarch64 is where the inference stops being safe. It is true that the
      parts this repository has been written against put the GPU on the package --
      Jetson Orin, and GB10 in a DGX Spark -- but a Grace-Blackwell *host* can pair
      an aarch64 CPU with a discrete card (GB300 in a DGX Station, on PCIe), where
      every copy does move bytes. No such machine has been observed here, so the
      answer is unknown rather than shared: a wrong ``shared`` prices every copy at
      zero, which invents an absence of work, while ``unknown`` only says the host
      has not been asked yet.
    """
    if accelerator == "metal":
        return True
    if accelerator == "cuda" and machine == "x86_64":
        return False
    if accelerator == "cuda" and machine == "aarch64":
        return None
    return False


def device_memory() -> str:
    """Whether the accelerator has memory of its own, shares the host's, or is unknown.

    Apple Silicon has no separate VRAM: a "copy to the GPU" moves nothing and
    occupies no link. A probe or a scenario that inserts one anyway is not merely
    mispriced, it invents an action -- so the question is asked of the host and
    written into the receipt rather than assumed from the architecture's name.
    ``unknown`` is a real answer: it is what an architecture cannot decide, and a
    caller that would price a copy from it must refuse rather than guess. A host
    whose owner has checked the topology states it once with ``IR_DEVICE_MEMORY``
    (``shared`` or ``separate``); the declaration then enters the receipt as the
    declared input it is.
    """
    answer = unified_memory(backend(), arch())
    if answer is None:
        declared = os.environ.get("IR_DEVICE_MEMORY", "").strip().lower()
        if declared in ("shared", "separate"):
            return declared
        return "unknown"
    return "shared" if answer else "separate"


def accelerator_device_name() -> str:
    """The device name a probe asks ggml for, in this backend's spelling.

    `real_moe_probe` used to spell this `"CUDA0"` inline. That is not a CUDA
    constant but a *device name*, and ggml's own name for the same card on a Mac
    is `"Metal0"`; keeping it in one function is what lets the probe be built
    against either backend without a second source file.
    """
    name = backend()
    if name == "cuda":
        return "CUDA0"
    if name == "metal":
        return "Metal0"
    if name == "vulkan":
        return "Vulkan0"
    raise UnsupportedHost("this host has no ggml accelerator device to name")


def require_measurement_host(root=None) -> None:
    """Refuse before anything long starts, not in the middle of a build.

    Every V-A column compares an accelerator placement with a CPU one, and each
    probe links the host's own backend: CUDA on a Linux or Windows box, Metal on
    a Mac. What is left to refuse is a host with no accelerator at all, where no
    column has a second placement to compare against.

    Saying so here rather than letting CMake fail three layers down is the point:
    the message is the difference between "this machine is a CPU-only host" and
    "this checkout looks broken".

    A column whose *observer* is CUDA -- the model column's llama.cpp taps, the
    CUPTI instrumentation, the strict-F32 router patch -- still refuses on its
    own, by name, in `run_accuracy_campaign.preflight(backend="cuda")`. That is
    a property of that column, not of the host.
    """
    if backend(root) is not None:
        return
    if not cuda_available(root):
        raise UnsupportedHost(
            "no accelerator ggml can compute on was found: no CUDA toolkit at any of "
            + ", ".join(cuda_toolkit_roots())
            + ", and this is not a Mac. Every V-A column needs a GPU placement beside "
              "the CPU one. Pass --cuda-dir to name a toolkit.")
    # A toolkit is installed and still there is no backend, which is one host
    # with one cause: the toolkit builds a probe and the driver gives it no
    # device to compute on. Telling that apart from "nothing installed" is the
    # difference between a machine that needs a GPU passed through and a
    # checkout that looks broken.
    raise UnsupportedHost(
        backend_reason() + ". Every V-A column needs a GPU placement beside the CPU "
        "one, and a toolkit alone cannot provide it: the probe has to find a device.")
