"""Pick the declared decoder-block contract a probe document belongs to.

The probe, the measurement runner and the exporter all accept either a dense
block or a sparse (mixture-of-experts) block.  Which one is never a guess and
never a flag the caller can get wrong: a suite declares its ``schema_version``,
a layer document declares its ``physical.contract``, and a measurement manifest
declares its ``contract``.  This module turns any of those into the contract
module that owns the rest of the declarations.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

MODELING = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))

from llm_infer_model.m2 import dense_contract, moe_contract  # noqa: E402

CONTRACTS = {module.CONTRACT: module for module in (dense_contract, moe_contract)}
# The manifest a measurement run writes, per contract.
PROFILE_SCHEMAS = {dense_contract.CONTRACT: "dense_layer_measurement_profile_v1",
                   moe_contract.CONTRACT: moe_contract.PROFILE_SCHEMA}


def for_suite(suite) -> object:
    schema = suite.get("schema_version") if isinstance(suite, dict) else None
    for module in CONTRACTS.values():
        if schema == module.SUITE_SCHEMA:
            return module
    raise ValueError(f"unknown measurement suite schema {schema!r}; "
                     f"known: {sorted(module.SUITE_SCHEMA for module in CONTRACTS.values())}")


def for_contract(name) -> object:
    if name not in CONTRACTS:
        raise ValueError(f"unknown block contract {name!r}; known: {sorted(CONTRACTS)}")
    return CONTRACTS[name]


def for_manifest(manifest) -> object:
    name = manifest.get("contract") if isinstance(manifest, dict) else None
    return for_contract(name)


def for_layer(layer) -> object:
    return for_contract(getattr(getattr(layer, "physical", None), "contract", None))


def load_layer(path: Path) -> object:
    """Load whichever kind of layer descriptor this file declares.

    A dense descriptor goes through the mapping's own m2-layer-0.3 parser, so
    all of its existing checks still run.  The probe's sparse descriptor has its
    own format (``moe-layer-0.1``) and its own parser, which verifies the same
    thing for a sparse block: the declared geometry and the declared GGML bytes.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("format_version") == moe_contract.LAYER_FORMAT_VERSION:
        return moe_contract.load_layer_document(document)
    from llm_infer_model.m2.spec import load_layer as load_dense_layer

    return load_dense_layer(path)
