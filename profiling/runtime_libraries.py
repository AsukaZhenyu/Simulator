"""Which of a probe's runtime libraries a directory (or several) holds.

Every V-A native probe links ggml in three pieces and loads them by name at
launch: the shared base, the CPU kernels, and the kernels of whichever
accelerator the probe was built for. A runtime directory has to hold all of
them. The check is worth having in one place and worth keeping importable from
anywhere: the campaign's preflight asks it before a cohort exists, and the two
`prepare` adapters ask it again with the exact directory list they will hand
the probe.

"Whichever accelerator" is the part that used to be spelled `ggml-cuda` here and
therefore was a CUDA machine's answer wearing a general name. On a Mac the third
library is `ggml-metal` and on a CUDA box `ggml-cuda`; a CPU-only box has no
third library at all. The list is derived from `host.backend()` so that the same
refusal works on every device, and so that a Mac is not refused for lacking a
library its backend never had.

Nothing here imports numpy, the model package or a campaign module: the
campaign asks this question in its own process, before it has built the
environment its child steps run in.
"""
from __future__ import annotations

from pathlib import Path

from profiling import host


#: "Ask this host" -- the default. ``None`` is not the default because it is a
#: real answer too: a CPU-only host builds a probe with no accelerator kernels,
#: and a test has to be able to say so without depending on the machine.
THIS_HOST = object()


def components(backend=THIS_HOST) -> tuple[str, ...]:
    """The libraries a probe built for `backend` loads, in link order.

    `backend` defaults to the host's own, and is a parameter so that the two
    answers no single device can both give -- "this Mac needs ggml-metal" and
    "this GPU box needs ggml-cuda" -- can each be stated and tested here.
    ``None`` means no accelerator at all.
    """
    name = host.backend() if backend is THIS_HOST else backend
    libraries = ["ggml-base", "ggml-cpu"]
    if name:
        libraries.append(host.BACKEND_LIBRARY.get(name, "ggml-" + name))
    return tuple(libraries)


def missing(directories, backend=THIS_HOST) -> list[str]:
    """Which libraries these directories do not hold between them.

    Holding *one* of them is the half-built case: an existence check passes, the
    probe starts far enough to print, and then dies on an unresolved symbol after
    a campaign has spent its first minutes. The union is what matters -- the
    campaign passes the runtime, the toolkit's `bin` and CUPTI's `lib` as three
    separate directories -- and the glob is the host's own, so a directory of
    another platform's libraries never answers for this one.
    """
    wanted = components(backend)
    names = []
    for directory in directories:
        for path in Path(directory).glob(host.shared_library_pattern()):
            # `libggml-cuda.so`, `libggml-cuda.so.0` and `ggml-cuda.dll` all
            # answer for `ggml-cuda`.
            names.append(path.name[3:] if path.name.startswith("lib") else path.name)
    return [component for component in wanted
            if not any(name.startswith(component) for name in names)]


def require(directories, backend=THIS_HOST) -> None:
    """Raise unless the directories hold every library, naming the ones that are not."""
    wanted = components(backend)
    absent = missing(directories, backend)
    if absent:
        raise ValueError(
            "the runtime directories hold no " + ", ".join(absent)
            + " of the " + str(len(wanted)) + " libraries the probe links ("
            + ", ".join(wanted) + "): "
            + "; ".join(str(Path(directory).resolve()) for directory in directories))
