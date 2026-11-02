# Context response refinement

This optional experiment tests a specific accuracy hypothesis on H512/P64
Dense S0/S1/S2, with one and two connected layers. It does not replace old
campaigns, claim all runtime state is explained, or validate expert caching.

`modeling/llm_infer_model/performance/context_response.py` owns the optional
preceding-service response coordinate and serial composition. Current and
previous physical action semantics, rather than strategy names or graph IDs,
index independently measured prices. Unknown contexts fail explicitly. The
original isolated-response model stays available unchanged.

The new native executable includes the immutable historical Executor and
completion helpers. Calibration measures each target action independently:
an untimed full working-set sweep and exact prefix precede its QPC interval.
Logical controls use hot batches. An adjacent calibration measurement obtains
hot repeated-action prices as the prespecified ablation. Numerical seeds are
different for calibration and complete heldout targets. No full-target duration
or residual fits either table. Where two-layer history coordinates require
their own calibration, that is declared; this is not arbitrary unseen-layer or
unprofiled-shape prediction.

D2H retains the historical 8 calibration and 8 different heldout sizes, three
fresh processes per size, 8 trials and 256 replays. Each process contains
ABBA/BAAB bare/stamped blocks. Only bare blocks fit the transfer response.
Stamped blocks separately observe the GGML copy API, GPU synchronization and
CPU synchronization responses. The copy call itself can contain staging and
internal synchronization; its time is not physical PCIe active service.
`cudaPointerGetAttributes` records the actual host-pointer registration category.

All source/binary/runtime bindings and inputs are checked. Failed native
processes retain output and remain in the declared denominator. Each owned
process has a 240-second cap; timeout cleanup touches only that child. The
shared measurement lock prevents overlapping GPU/CPU acquisitions.

## Reproduction

Supply explicit existing GGML source/import libraries, strict-F32 runtime,
generated Executor include directory and installed CUDA toolkit; there are no
downloads or system-wide changes. Build the new `native/CMakeLists.txt` in a
new directory using Release configuration. A Windows installation with the
Debug MSPDB mismatch can use `-DCMAKE_TRY_COMPILE_CONFIGURATION=Release`.

The bounded paper campaign driver intentionally reads the retained original
native-asset manifest at
`build/dac-followup-20261025/multilayer-v1/campaign-v1/manifest.json` for its
hardware identity context and explicit runtime search directories. It does
not read its old action prices for calibration. The separate historical
reference report needs the retained E2A profile. This artifact convenience
driver is not a clean-install automatic hardware-discovery command. The native
probe, request schema, context coordinate and response-table API can be used
with newly generated programs and explicit resources independently.

With `PYTHONPATH=modeling;mapping;profiling`, run:

```text
python -m profiling.dac_refinement.campaign prepare --out NEW --binary EXPLICIT_EXE
python -m profiling.dac_refinement.campaign profile --out NEW
python -m profiling.dac_refinement.campaign freeze --out NEW
python -m profiling.dac_refinement.campaign heldout --out NEW
python -m profiling.dac_refinement.campaign observer --out NEW
python -m profiling.dac_refinement.campaign transfer-profile --out NEW
python -m profiling.dac_refinement.campaign transfer-freeze --out NEW
python -m profiling.dac_refinement.campaign transfer-heldout --out NEW
python -m profiling.dac_refinement.campaign evaluate --out NEW
```

The manifest fixes the exact process count before timing. Dense qualification
requires all four processes' mean APE at most 10% and CV at most 10%, pooled
trial-average P90 APE at most 20%, and all four observer effects at most 5%.
D2H qualification requires maximum heldout size-mean APE at most 10%, all 24
process CVs at most 10%, and all 24 bare/stamped effects at most 5%. These are
bounded engineering targets, not conference acceptance thresholds.

## Recorded result and applicability

The 2026-10-26 campaign completes all 111 processes. The primary reset-only
matrix qualifies 5/6 programs with fresh-hot and 2/6 with the optional preceding
response table. The latter improves two-layer S2 mean error but overpredicts
other programs, so it must not replace the original default or be selected
case by case after observing targets. D2H remains unqualified: maximum heldout
size-mean APE is 19.08%, CV passes 19/24, observer effects pass 24/24. Actual host
pointer queries identify unregistered pageable memory.

A post hoc secondary analysis retains all six programs and uses already
collected observer bare blocks under reset plus an additional complete prime
pass. Under this different preparation contract, each table qualifies 4/6
programs, covering different cases. The preceding table passes both mixed S2
programs (maximum process APE 3.04% and 7.54%), but still overpredicts the GPU-only
programs. Between-phase/process state remains confounded with priming. These
records support preparation-contract sensitivity, not a universal physical
cause or a replacement of primary qualification. The old 0/6 group result stays
unchanged and refers to a different matrix.

The authoritative local outputs are `summary.json` for the primary result,
`reset-prime-secondary.json` for this explicitly secondary analysis, and
`REPORT_zh_v3.html`. `final-evidence-v2.json` binds raw hashes, executed source
snapshots, row-level CSVs and software-test receipts. No target time is used to
refit parameters or introduce a working-set/layer axis.
