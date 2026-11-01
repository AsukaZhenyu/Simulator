# Prepared transfer and complete Dense validation

This additive study checks host-memory registration and measurement preparation
without changing the historical measurements or the simulator's default model.
The reusable modeling API is
`llm_infer_model.performance.prepared_transfer.PreparedTransferModel`.
The campaign driver is a bounded paper experiment that requires the retained
mathematical programs and explicit local GGML/CUDA runtime assets.

## One response model with explicit inputs

The model uses the existing nonnegative affine or continuous one-hinge family.
Its inputs identify transfer bytes, route, actual host-memory type, completed
API boundary, preparation, and hardware/runtime scope. Each supported memory
type has independently measured coefficients within this same family.

The price is the inclusive host-completed response of
`ggml_backend_tensor_copy`, GPU synchronization and CPU synchronization.
Copy-call phase intervals and CUDA event diagnostics are never added to that
inclusive price. A byte-response coefficient does not identify physical PCIe
bandwidth by itself.

Unknown contracts, changed API semantics, unmeasured host-memory categories and
sizes outside the calibration envelope are rejected. There is no implicit
pageable-to-registered substitution or end-to-end residual correction.

```python
import json
from pathlib import Path
from llm_infer_model.performance.prepared_transfer import PreparedTransferModel

profile = json.loads(Path("transfer-pageable-frozen.json").read_text())
model = PreparedTransferModel.from_profile(profile)
prediction = model.prediction(262144, contract=profile["contract"],
                              allow_unqualified=True)  # Explicit validation use.
print(prediction["duration_ns"])
```

An unqualified frozen profile requires explicit `allow_unqualified=True` for
validation. A qualified profile is written separately only after independent
evidence auditing and strict heldout predicates; its prices and contract remain
identical to the pre-heldout freeze.

The 2026-10-26 short-window campaign remains unqualified. A separate followup
validates a finite size-dependent measurement grid without refitting its
coefficients. Use `ValidatedTransferCatalog` for that independently checked
grid, rather than changing the original frozen profiles' qualification flags:

```python
from copy import deepcopy
from llm_infer_model.performance.validated_transfer import ValidatedTransferCatalog

manifest = json.loads(Path("validated-transfer-catalog.json").read_text())
catalog = ValidatedTransferCatalog.from_manifest(manifest)
entry = manifest["entries"]["pageable"]
size = 262144
contract = deepcopy(entry["source_contract"])
contract["preparation_contract"]["timed_replays_per_batch"] = entry["validated_replays"][str(size)]
prediction = catalog.prediction(size, contract=contract)
```

The accepted grid uses 4096 completed copies per block at 64, 1024, 2048 and
16384 bytes, and 256 at 262144, 2097152, 8388608 and 33554432 bytes. It is a
qualification of warm block-average costs. The catalog rejects single-copy
requests, unvalidated sizes, cold preparation and other runtime contracts.
The original short-window fluctuations remain in the evidence.

## Measurement preparation

The native executable verifies its compiled source SHA against the request.
When `expected_process_mask` is supplied, it applies and verifies that mask
before backend initialization, and verifies it again after measurement.
GGML uses one CPU compute thread; the main thread is pinned to mask 4. The
campaign explicitly records the broader process mask allowed to runtime helper
threads. These are three separate settings.

Both Dense primary and observer blocks use exactly one reset, one untimed
complete prime, and eight timed fixed-input complete replays. The observer
changes only interval stamping, using balanced ABBA/BAAB blocks. Every block
retains its preparation contract and preceding system-memory/CPU-counter/CUDA
free-memory observations. These observations are diagnostic; future target
utilization and target durations are not predictor inputs.

The transfer probes retain 8 trials, balanced bare/stamped blocks and 256
completed copies per block, with equal warmup and one untimed copy prime. The
paired diagnostic registers and unregisters the same owned CPU destination,
queries its actual CUDA pointer category, checks all copied bytes, and restores
the original category. Registration is outside steady timing and is not a free
initialization or allocation benefit.

## Acquisition and freezing

Build `native/CMakeLists.txt` with explicit `LLAMA_ROOT`, `GGML_BUILD_DIR`,
`STEADY_GENERATED_DIR`, and `CUDA_ROOT`; no assets are downloaded or installed.
Use a new Release build directory. The native `--self-test` does not construct a
backend or start an acquisition.

The paired diagnostic has its own immutable six-process manifest:

```text
python -m profiling.dac_accuracy_repair.diagnostic prepare --out NEW_DIAG --binary EXPLICIT_EXE
python -m profiling.dac_accuracy_repair.diagnostic collect --out NEW_DIAG
python -m profiling.dac_accuracy_repair.diagnostic summarize --out NEW_DIAG
```

The complete campaign uses a new output directory and explicit process mask:

```text
python -m profiling.dac_accuracy_repair.campaign prepare --out NEW --binary EXPLICIT_EXE --process-mask EXPLICIT_MASK
python -m profiling.dac_accuracy_repair.campaign transfer-profile --out NEW
python -m profiling.dac_accuracy_repair.campaign profile --out NEW
python -m profiling.dac_accuracy_repair.campaign freeze --out NEW
python -m profiling.dac_accuracy_repair.campaign transfer-heldout --out NEW
python -m profiling.dac_accuracy_repair.campaign dense --out NEW
python -m profiling.dac_accuracy_repair.campaign evaluate --out NEW
```

Set `PYTHONPATH` to the repository's `src`, `modeling`, `mapping`, and `profiling`
directories. The driver checks archived and current execution sources,
executable and runtime hashes. It serializes hardware acquisition with the
existing measurement lock. Each owned child has a timeout, numerical/byte and
QPC verification, explicit exit evidence, and retained failure output. Failed
jobs are never retried or removed from the declared matrix.

Training uses eight sizes, three fresh processes per size and host-memory mode.
Training-only leave-one-size-out selection chooses the mathematical family.
The parameters and complete-program predictions are frozen before the first
heldout process. The eight heldout sizes have three new processes per mode.
The six Dense programs have four primary and four observer processes each.

## Qualification scope

The original transfer predicate requires maximum size-mean APE at most 10%,
all 24 process trial CVs at most 10%, and all 24 observer differences at most 5%.
The stricter reported predicate additionally requires every process-mean APE
at most 10% and pooled trial-average P90 APE at most 20%.

Dense qualification separately requires matching primary/observer preparation,
every process-mean APE at most 10%, every process CV at most 10%, pooled
trial-average P90 APE at most 20%, and all observer differences at most 5%.
The prespecified primary provider is the independently measured semantic hot
action table. Replacing its D2H entries with an isolated transfer batch mean is
an explicitly assumption-bound secondary result, including both the replay
window and in-program preparation/arena assumptions. It is not exact-contract
COPY qualification.

The results concern warm fixed-input repeated responses on the identified local
CPU/GPU system. They do not certify cold single copies, arbitrary background
loads, real UMA, physical-link bandwidth or a production autoregressive LLM.
The old single-core-process measurements remain separate evidence.
