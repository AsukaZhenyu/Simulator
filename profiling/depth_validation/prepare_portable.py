"""Prepare the depth cohort from explicit paths, without the archived campaign.

`campaign.prepare` reads one legacy descriptor -- the H2D history campaign's
manifest -- for three things only: the protocol constants that describe how an
action is timed, the runtime DLLs the probe is launched against, and a hardware
snapshot. Everything else it needs it builds itself.

The historical entry point reused that manifest because both campaigns ran on one
machine on one afternoon. On a second machine the runtime paths do not exist and
the hardware snapshot would be someone else's, so this adapter supplies the same
three fields from explicit arguments and the current device:

  * `runtime_dirs` are the directories given on the command line, first one first
    (it is the one holding the probe's own DLLs);
  * `runtime` is a receipt of every DLL in them, which the run re-checks before
    and after every measurement;
  * `baseline_context` keeps the protocol constants verbatim -- threads 1, F32,
    CUDA graphs disabled, `GGML_PREC_F32`, allocation untimed, the same response
    context and the same 500 ms warmup floor -- and replaces the machine-specific
    fields with this device's: the probe's hash, the runtime DLL hashes, and
    `hardware_identity()`.

Nothing here reads a prior profile, a frozen prediction or a heldout result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import campaign
from . import runtime_dirs
from profiling import host
from profiling.dac_validation import e2a
from profiling.state_response.collect import hardware_identity

LEGACY = campaign.ROOT / 'build/h2d-history-20261026/campaign-v1/manifest.json'

# The measurement protocol the published depth campaign ran. These describe the
# probe and the response it is asked for, not the machine, so they are carried
# over unchanged rather than re-derived.
PROTOCOL = dict(dtype='F32', cuda_graphs='disabled', precision='GGML_PREC_F32',
                threads=1, poll='0',
                boundary='unchanged completed Executor.action',
                allocation_timed=False, minimum_active_primitive_warmup_ms=500,
                response_context='independent hot action mean; Dense reset plus one complete prime; fixed input')


def machine_fields() -> dict:
    """The two entries that were machine properties written as protocol.

    `process_affinity_mask` was `4294967295` and `main_thread_mask` was `4`
    because the machine they were measured on had 32 CPUs. A 20-core device has
    no such mask to send, and the probe's refusal read as a measurement fault.
    Both are now asked of the host and written into the receipt beside the
    hardware snapshot, where a machine property belongs. On the 32-CPU Windows
    machine that produced the published table they come out unchanged.
    """
    return dict(process_affinity_mask=host.process_affinity_mask(),
                main_thread_mask=host.main_thread_mask(),
                online_cpus=host.online_cpus())


#: Re-exported: the derivation is used by `descriptor` below and by the
#: campaign's preflight, which cannot import this module (it pulls the campaign
#: and the model package in).
ggml_runtime_dirs = runtime_dirs.ggml_runtime_dirs


def descriptor(binary: Path, runtime_dirs: list[Path]) -> dict:
    binary = binary.resolve()
    e2a.require(binary.is_file(), 'explicit probe binary missing: ' + str(binary))
    runtime_dirs = [Path(d) for d in runtime_dirs]
    derived = [d for d in ggml_runtime_dirs(binary)
               if d.resolve() not in {r.resolve() for r in runtime_dirs}]
    runtime_dirs = runtime_dirs + derived
    for directory in runtime_dirs:
        e2a.require(Path(directory).is_dir(), 'explicit runtime directory missing: ' + str(directory))
    dlls = [path for directory in runtime_dirs
            for path in sorted(Path(directory).glob(host.shared_library_pattern()))]
    e2a.require(dlls, 'the runtime directories hold no '
                       + host.shared_library_pattern() + ' files; the probe would not load')
    # `depth-probe` links the same three components and loads them by name at
    # launch: a runtime with only the first one passes the check above and then
    # exits on an unresolved symbol, one phase into a prepared cohort.
    e2a.require_components(runtime_dirs)
    context = dict(PROTOCOL, **machine_fields(),
                   binary_sha256=e2a.sha(binary),
                   runtime_dlls={str(path): e2a.sha(path) for path in dlls},
                   hardware=hardware_identity())
    return dict(baseline_context=context,
                runtime=[dict(path=str(path), sha256=e2a.sha(path)) for path in dlls],
                runtime_dirs=[str(Path(directory).resolve()) for directory in runtime_dirs],
                derived_runtime_dirs=[str(d.resolve()) for d in derived])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, action='append', required=True,
                        help='directory searched at launch for the probe and GGML DLLs; repeatable')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cache', type=Path,
                        help='an earlier cohort whose inputs/*/program.json may be re-seeded instead of '
                             'composed again; no timing, profile or held-out result is read from it')
    args = parser.parse_args(argv)

    supplied = descriptor(args.binary, args.runtime_dir)
    original = campaign.read
    legacy = LEGACY.resolve()

    def explicit_read(path):
        if Path(path).resolve() == legacy:
            return supplied
        return original(path)

    campaign.read = explicit_read
    try:
        result = campaign.prepare(args.out, args.binary, pilot=args.pilot, workers=args.workers,
                                  cache=args.cache)
    finally:
        campaign.read = original
    e2a.write(args.out / 'portable-preparation-receipt.json',
              dict(schema='depth-explicit-path-preparation-1',
                   source=e2a.receipt(Path(__file__).resolve()),
                   campaign=e2a.receipt(Path(campaign.__file__).resolve()),
                   replaced_descriptor=str(legacy),
                   supplied_fields=sorted(supplied),
                   protocol_constants=PROTOCOL,
                   hardware=supplied['baseline_context']['hardware'],
                   runtime_dirs=supplied['runtime_dirs'],
                   prior_profile_or_result_read=False,
                   prepared_manifest=e2a.receipt(args.out / 'manifest.json')))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
