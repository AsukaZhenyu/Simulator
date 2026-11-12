# Pre-registration: the model column without an engine

Written before the registered run named in §5.  The only execution that preceded this
document was a functional one (does the lowering load a 28-layer file's own bytes and
finish), and its numbers are reproduced in §6 so they cannot be presented as a fresh
confirmation later.

## 1. What this column replaces

The published model column timed another engine's request and asked the simulator to
predict that engine.  The project's claim is about a plan, not about one runtime's
scheduler, and the dependency made every reading contingent on that runtime's headers and
release cadence.  This column lowers the same plan ourselves (`profiling/native/
ir_runner.cpp`, linking `ggml-base`/`ggml-cpu`/`ggml-cuda` only) and executes it against
the model file's own per-layer weight bytes.

The executed object is therefore *our lowering of a declared plan*, not the model running
under somebody's runtime.  What is compared is the frozen table's price of each action
against what that action costs when our graph runs it.

## 2. The cells

Five placements `g ∈ {0, L/4, L/2, 3L/4, L}` (`g` blocks of the `L`-block stack on the
accelerator), each case being prefill at 64 tokens followed by two decode rounds.  The
decode rounds run at visible history **128**, a point the frozen table prices; a chain
history (65, 66, …) is not a grid point and its `no_price_at` is a coverage fact, not an
accuracy reading.

A1 is measured over each priced action unit, split by phase.  A2 is the case's measured
wall time over the sum of its actions' prices; copies carry no table price and are
declared, not silently included.

## 3. Acceptance

| cell | band |
|---|---|
| state gate: all-CPU placement, prefill units | `[0.85, 1.20]` (unchanged) |
| A1, prefill: attention / dense_ffn / projection | measured/predicted within ±10% |
| A1, decode: attention / dense_ffn / projection | measured/predicted within ±10% |
| A2, the all-CPU case | within ±10% |

The ±10% band is the one the paper already applies to this column; it is not chosen from
this reading.  No cell is added, dropped or re-banded after a number is seen.

## 4. Declared limitations, stated before the numbers

* **GPU decode prices do not exist in the frozen table** (the accelerator export prices
  prefill only).  Accelerator decode units are therefore `no_price_for_device` and are
  excluded from A1, counted in `unpriceable`.  A reading that omits them is a reading
  about the CPU decode path.
* **The executor times one ggml graph per plan node.**  A priced unit the lowering splits
  into *n* nodes pays the graph's fixed cost *n* times.  The report therefore gives A1 by
  (unit, phase, node count) as well as pooled, so a fixed cost that is invisible on a
  6-millisecond prefill unit and visible on a 0.2-millisecond decode unit cannot be read
  as a pricing error.
* **Copies have no table price.**  A2 on a case with copies is a lower bound.
* **The first unit of a phase on the accelerator** absorbs buffer allocation for the
  whole GPU stretch.  It is reported in the outlier tail, not corrected.

## 5. The registered run

```
python -m profiling.ir_ggml.model_column --model qwen3-0.6b \
    --gguf ../llama.cpp-implement/validation-models/Qwen3-0.6B-Q4_K_M.gguf \
    --rounds 2 --replicas 4 --out build/model-ggml/registered-qwen3
```

Replicas 4 is the published protocol; replica 0 pays the CUDA module load and is never a
reading.  The model file's sha256 is recorded in `column.json`.

## 6. The functional run (exploratory, not a confirmation)

`build/model-ggml/column-qwen3`, 2 replicas, same cells except one case fewer:
prefill attention 0.897 / dense_ffn 0.922 / projection 0.887; decode 1.122 / 0.957 /
1.117; state gate in band (all-CPU control 0.896); A2 0.908.  Verdict under §3:
`off_band`, with the prefill cells missing by 0.3–1.3 points and decode by 1.7–2.2.
