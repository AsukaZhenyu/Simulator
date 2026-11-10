"""Prepare the real-MoE window cohort on this machine, from explicit paths.

`runtime_window_campaign.prepare` reads two inputs that are build artifacts of the
machine that ran the published campaign:

  * the numerical campaign's manifest, for six fields only -- the model receipt,
    the capacity policy, the two host-memory guards, the resident peak limit and
    the frozen request suite;
  * the request template derived from that run's tokenizer pass.

Everything else it builds itself: the request files, the placement labels, the
execution receipts and the source archive. On a second machine neither path
exists, so this adapter supplies them from explicit arguments and the current
device, exactly as `profiling/depth_validation/prepare_portable.py` does for the
depth cohort:

  * `--model` is this device's GGUF, verified by SHA-256 and byte count against
    the pinned manifest (`profiling/multimodel/models.json`); the template is
    rewritten to point at it, and nothing else in the template changes;
  * `--suite` and `--capacity-policy` default to the copies shipped under
    `published/`, whose hashes are the ones the published cohort froze;
  * the two host-memory guards come from the capacity policy's own budget block
    (model bytes plus the declared context workspace and system margin), and may
    be raised -- never silently guessed -- with `--native-guard-bytes`;
  * the runtime search path is `--runtime-dir`, repeated.

Nothing here reads a prior profile, a frozen prediction or a heldout result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import runtime_window_campaign as campaign
from .dataset import read, write
from .runtime_window_io import receipt, require, sha

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "published"
SHIPPED_SUITE = PUBLISHED / "request-suite-v3.json"
SHIPPED_POLICY = PUBLISHED / "RESIDENT_CAPACITY_POLICY-v2.json"
SHIPPED_TEMPLATE = PUBLISHED / "request-template-normal-SSLS.json"


def model_file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_model(identifier: str = "olmoe-1b-7b") -> dict:
    manifest = json.loads((HERE.parent / "multimodel" / "models.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["id"] == identifier:
            return model
    raise SystemExit(f"{identifier} is not in profiling/multimodel/models.json")


def verify_model(path: Path, pinned: dict) -> dict:
    path = Path(path).resolve()
    require(path.is_file(), f"model missing: {path}")
    size = path.stat().st_size
    require(size == pinned["bytes"],
            f"{path.name} is {size} bytes, the pinned model is {pinned['bytes']}")
    digest = model_file_sha(path)
    require(digest == pinned["sha256"],
            f"{path.name} hashes to {digest}; the pinned model is {pinned['sha256']}")
    return dict(path=str(path), sha256=digest, bytes=size)


def budget(policy: dict) -> dict:
    block = policy.get("budget")
    require(isinstance(block, dict), "capacity policy has no budget block")
    for field in ("model_bytes", "context_workspace_budget_bytes", "system_margin_bytes"):
        require(isinstance(block.get(field), int), f"capacity policy budget lacks {field}")
    native_guard = (block["model_bytes"] + block["context_workspace_budget_bytes"]
                    + block["system_margin_bytes"])
    parent_preflight = native_guard + block.get("parent_extra_margin_bytes", 0) + block.get(
        "observed_startup_peak_wset_bytes", 0)
    resident = policy.get("resident_peak_limit_bytes", block["model_bytes"])
    return dict(native_guard_bytes=native_guard, parent_preflight_bytes=parent_preflight,
                resident_peak_limit_bytes=resident)


def template_for(model: dict, template: dict) -> dict:
    """The published template with only the device-specific fields replaced."""
    out = dict(template)
    out["model_path"] = model["path"]
    out["model_sha256"] = model["sha256"]
    out["model_bytes"] = model["bytes"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True,
                        help="compiled runtime-windows build (holds BUILD_COMPLETE.json)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True,
                        help="directory searched at launch for CUDA runtime DLLs; repeatable")
    parser.add_argument("--suite", type=Path, default=SHIPPED_SUITE)
    parser.add_argument("--capacity-policy", type=Path, default=SHIPPED_POLICY)
    parser.add_argument("--template", type=Path, default=SHIPPED_TEMPLATE)
    parser.add_argument("--model-id", default="olmoe-1b-7b")
    parser.add_argument("--native-guard-bytes", type=int,
                        help="raise the derived host-memory guard for this device")
    parser.add_argument("--heldout-index", type=int, default=1)
    parser.add_argument("--labels", help="comma-separated placement labels")
    args = parser.parse_args(argv)

    for directory in args.runtime_dir:
        require(Path(directory).is_dir(), f"runtime directory missing: {directory}")
    model = verify_model(args.model, pinned_model(args.model_id))
    policy = read(args.capacity_policy)
    suite = read(args.suite)
    template = template_for(model, read(args.template))
    guards = budget(policy)
    if args.native_guard_bytes is not None:
        require(args.native_guard_bytes > guards["native_guard_bytes"],
                "the guard may only be raised; it is a floor against thrashing")
        guards["parent_preflight_bytes"] += args.native_guard_bytes - guards["native_guard_bytes"]
        guards["native_guard_bytes"] = args.native_guard_bytes

    supplied_descriptor = dict(
        model=model,
        capacity_policy=receipt(args.capacity_policy),
        source_suite=receipt(args.suite),
        **guards,
    )
    require(suite.get("requests"), "the frozen request suite is empty")

    original = campaign.read
    legacy_descriptor = campaign.LEGACY_DESCRIPTOR.resolve()
    legacy_template = campaign.LEGACY_TEMPLATE.resolve()

    def explicit_read(path):
        resolved = Path(path).resolve()
        if resolved == legacy_descriptor:
            return supplied_descriptor
        if resolved == legacy_template:
            return template
        return original(path)

    campaign.read = explicit_read
    try:
        result = campaign.prepare(
            args.out, args.build,
            heldout_index=args.heldout_index,
            labels=tuple(x for x in args.labels.split(",") if x) if args.labels else None,
            runtime_dirs=[str(Path(d).resolve()) for d in args.runtime_dir],
        )
    finally:
        campaign.read = original

    write(args.out / "portable-preparation-receipt.json", dict(
        schema="real-moe-explicit-path-preparation-1",
        source=receipt(Path(__file__).resolve()),
        campaign=receipt(Path(campaign.__file__).resolve()),
        replaced_descriptor=dict(path=str(legacy_descriptor), supplied_fields=sorted(supplied_descriptor)),
        replaced_template=dict(path=str(legacy_template), changed_fields=["model_path", "model_sha256", "model_bytes"]),
        model=model, guards=guards,
        runtime_dirs=[str(Path(d).resolve()) for d in args.runtime_dir],
        prior_profile_or_result_read=False,
        prepared_manifest=receipt(args.out / "manifest.json"),
    ))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
