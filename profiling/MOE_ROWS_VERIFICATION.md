# Selected-row MoE numerical verification

This delivery verifies the arithmetic and logical data ranges of an explicit
selected-row MoE FFN. It does not measure CPU/GPU execution, calibrate a stage
price or run a real model. The reference uses only Python's standard library.

```powershell
python profiling/verify_moe_rows.py `
  --case profiling/cases/moe_rows_reference.json `
  --out profiling/results/my-moe-rows-check
```

All shapes, scored token routes, shared-expert presence, seed and tolerances
must be explicit in the case. An absent or empty output directory is required;
existing evidence is never overwritten. Unsupported dtypes, missing scores,
duplicate token rows or expert IDs, nonfinite numbers and unknown fields are
rejected. Exit codes are `0` for numerical equivalence, `3` for numerical
failure and `2` for invalid input or file errors.

## What is compared

The independent reference evaluates every token separately: gate projection,
up projection, SwiGLU, down projection, each provided route score, the optional
shared expert and exactly one residual. The materialized path groups routed
rows across requests and executes gather → gate/up → SwiGLU → down → weighted
scatter-add. The first path uses high precision scalar reductions and rounds
materialized ports to F32; the grouped path rounds each multiply/add to F32.
This intentionally uses distinct reduction paths, so bit-identical outputs
are not required. No attention, router discovery or score normalization is
performed by either path.

The fixture has six tokens across three requests with nonzero starting
positions, unequal top-2 scores, experts shared across requests, an inactive
expert and a shared expert. Hidden/intermediate widths are 5/7. Random values
use non-power-of-two scaling. Wrong gather rows, wrong score slots, omitted
shared contributions and dropped stages have separate rejecting checks.

`raw.json` records generated tensors, every group port and both outputs.
`report.json` records error tolerances, selected versus whole-request bytes,
F32 output digests and explicit scope flags. `manifest.json` binds exact source
bytes and output artifacts with SHA-256. Logical buffer bytes do not include
allocator padding, backend scratch or thread runtime memory. Workspace stays
`null`, accompanied by `not_measured_by_numerical_oracle`.

## Checking a Modeling expansion

`validate_row_execution_document(workload_document, physical_boundary,
expected_groups=compiled.expert_work_items)` checks the new
`physical_boundary.row_execution` metadata without importing Modeling. It
checks compact rows, stage order and dependencies, selected input/intermediate/
output bytes, per-request scatter coverage and combine inputs. True route rows
can be supplied explicitly for exact coverage comparison. The test suite also
checks actual Modeling opaque and staged expansions when that package is
installed.

The CLI can check exported documents with `--workload workload.json --boundary
physical_boundary.json`. These two inputs must be supplied together. Its
numerical fixture and the Modeling expansion are separate evidence scopes;
passing their checks does **not** make the synthetic Modeling graph a numerical
kernel. The report retains `numeric_execution_verified: false` for Modeling
row accounting. In particular, a synthetic 64-byte expert allocation is not
the same object as the reference's three explicit F32 matrices.

## What the existing native probe can and cannot provide

In `native/moe_probe.cpp`, each `expert.N` region starts with
`ggml_get_rows`, then gate/up, SwiGLU, down and output materialization. The
existing expert time includes gather. Its `combine` region includes score
selection, multiplication, scatter-back, residual accumulation and output
materialization. Existing measurements cannot be relabelled as isolated
`gather_rows`, `expert_gate`, `expert_up`, `expert_swiglu`, `expert_down`,
`scatter_rows` or `combine_rows` prices. Adding a separate gather while retaining
this measured expert time would count gather twice.

Physical profiling of the new stages still needs matching explicit port and
runtime contracts: full input and selected output shapes, row/slot index
storage, route score representation, exact weight matrix or quantized-block
layout, allocated scratch, backend, threads, completion synchronization and
fusion/materialization scope. Weight residency and any materialized subweight
partitions must match the particular lowering's allocation contract; this
numerical oracle does not establish physical aliases or residency benefits.
It exports no costs.

Run the hardware-free contract suite with:

```powershell
python -B -m unittest discover `
  -s profiling/tests -p test_moe_rows.py -v
```
