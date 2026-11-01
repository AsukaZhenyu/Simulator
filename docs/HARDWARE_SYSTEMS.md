# Representative edge hardware configurations

This catalogue describes nine selected hardware configurations, not every
available product or the latest flagship. It supplies named, reproducible inputs
for strategy comparisons. Product identity, memory options and the cited nominal
bandwidth are vendor facts. **Every effective task rate, launch cost, usable
memory budget, link rate and storage rate used by this experiment is an explicit
author assumption. No machine was measured.**

The executable catalogue is
[`hardware_system_catalogue.py`](../experiments/hardware_system_catalogue.py).
It exports immutable `SYSTEMS`, `systems_by_id()`, `system_rows()`,
`hardware_kwargs(system_or_id)` and `hardware_for(system_or_id)`. Each descriptor
also has `.hardware()`. It requires no downloaded profiles or model weights.

## Configurations and storage topology

Memory values below are nominal GB configurations. The experiment deliberately
represents each as GiB; rates in GB/s use decimal bytes. The reserve is a declared
scenario input, not a claim about what an operating system reserves. For separate
memory, the public Hardware capacities are the resulting usable budgets. For
shared memory, the physical capacity and the reserve are both recorded and passed
to Hardware; DRAM and VRAM logical views must never be added.

| Configuration | Modeled physical memory | Available budget | Additional accelerator | Primary source |
|---|---|---|---|---|
| RTX 4060 Laptop with selected 32 GB host | Separate: host 32 + GPU 8 | Host 24 + GPU 6 | None in the declared configuration | [NVIDIA laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/40-series/) |
| RTX 5090 desktop with selected 64 GB host | Separate: host 64 + GPU 32 | Host 56 + GPU 28 | None in the declared configuration | [NVIDIA RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) |
| Mac mini M4 Pro, selected 14 CPU / 20 GPU cores | Shared 64 | Shared 56 | 16-core Neural Engine | [Apple 2024 technical specifications](https://support.apple.com/en-us/121555) |
| DGX Spark GB10 | Coherent shared 128 | Shared 112 | No separate NPU modeled | [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) |
| Ryzen AI Max+ 395, selected 128 GB | Shared 128 | Shared 112 | XDNA 2 NPU | [AMD specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html), [unified-memory description](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html) |
| Intel Core Ultra 7 258V | Shared 32 | Shared 24 | Intel AI Boost | [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html) |
| Snapdragon X Elite, selected 32 GB laptop | Shared 32 | Shared 24 | Hexagon NPU | [Qualcomm specifications](https://www.qualcomm.com/laptops/products/snapdragon-x-elite) |
| Jetson AGX Orin 64 GB | Shared 64 | Shared 56 | Two NVDLA v2 engines, excluded from generic NPU experiments | [NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) |
| OnePlus 13, Snapdragon 8 Elite, selected 16 GB / 512 GB | Shared 16 | Shared 10 | Hexagon NPU | [OnePlus device specifications](https://www.oneplus.com/us/13/specs), [Qualcomm platform specifications](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform) |

The x86 CPU SKU and host RAM in the two discrete systems are author-selected
configuration placeholders. Snapdragon X Elite RAM is an OEM configuration
assumption. The OnePlus, Apple, AMD, Spark, Intel and Orin memory sizes correspond
to sourced device configurations or capacity options; this is not evidence that
the full capacity is allocatable by every runtime.

## Assumed effective compute and data rates

The existing F32 analytical task formula remains unchanged. These values are
**hypothetical task-effective inputs**, not vendor peaks, measured throughput,
or conversion from advertised AI TOPS. They allow experiments to ask how a policy
responds to a faster compute lane, a slower link, or a shared pool. Replacing them
with measured task profiles is a later calibration activity.

| Configuration ID | CPU TFLOP/s / GB/s | GPU TFLOP/s / GB/s | Real copy GB/s | Storage read / write GB/s | Cited nominal system-memory GB/s |
|---|---|---|---|---|---|
| `rtx4060_laptop` | 0.40 / 40 | 6 / 180 | 10 | 2 / 1 | Not used |
| `rtx5090_desktop` | 1.00 / 70 | 35 / 1000 | 24 | 4 / 2 | Not used |
| `m4pro_mini` | 0.80 / 80 | 5 / 190 | No copy for same buffer | 3 / 2 | 273 |
| `dgx_spark` | 0.60 / 80 | 10 / 190 | No copy for same buffer | 3 / 2 | 273 |
| `ryzen_ai_max395` | 1.00 / 100 | 6 / 180 | No copy for same buffer | 3 / 2 | 256 |
| `core_ultra258v` | 0.30 / 35 | 2 / 85 | No copy for same buffer | 2 / 1 | Not asserted |
| `snapdragon_x_elite` | 0.25 / 40 | 1.2 / 85 | No copy for same buffer | 2 / 1 | 135 |
| `jetson_agx_orin` | 0.15 / 35 | 2 / 140 | No copy for same buffer | 1 / 0.5 | 204.8 |
| `snapdragon8elite_phone` | 0.15 / 25 | 1 / 55 | No copy for same buffer | 1 / 0.5 | Not asserted |

CPU and GPU launch assumptions are respectively 500 ns and 5,000 ns per modeled
task for every configuration. Separate memory uses a 1,000 ns copy latency.
Shared-buffer visibility is represented by the existing zero-payload SHARE
action. We do not provide a fabricated positive-bandwidth copy quote for an
unused shared-memory copy. A strategy requesting an actual COPY still needs an
explicit real-copy cost and distinct allocation.

All shared configurations assume the selected buffer is accessible to both
modeled compute lanes without a format conversion. This is a controlled
abstraction. Driver carve-outs, pinning, cache maintenance, command queues,
layout conversions and platform-specific ownership transitions are not inferred
from the product name. AMD's graphics carve-out is part of the same physical
memory, not extra capacity. Jetson storage assumes an added NVMe drive. Phone
storage uses the legacy SSD-rate fields to price average UFS payload access; it
does not imply that the phone contains an NVMe SSD.

Nominal shared-memory bandwidth is descriptive metadata, not an additional
performance multiplier. Effective CPU/GPU memory rates price individual tasks;
the current model does not automatically arbitrate aggregate CPU+GPU DRAM
bandwidth. Concurrent shared-memory throughput must therefore not be presented
as a calibrated UMA result.

## NPU proxy experiments

The runtime still has one CPU and one accelerator execution slot. A proxy
experiment may substitute a hypothetical NPU task cost table for the accelerator
slot:

```text
t_proxy_NPU(task, shape) = alpha * t_same_platform_GPU(task, shape)
alpha in {0.5, 1.0, 2.0}
```

This brackets a faster/equal/slower task lane while preserving the same graph,
tensor bytes, dependencies and memory budget. It is not a third independently
concurrent CPU+GPU+NPU lane, an NPU benchmark, or proof of NPU operator/precision
support. In particular, the F32 tensor abstraction cannot be advertised as
evidence that a vendor's NPU accepts those tasks in F32. There is no INT8/FP4 TOPS
to F32 conversion.

Five catalogue configurations are eligible for this explicitly hypothetical
proxy: Apple M4 Pro, AMD Ryzen AI Max+ 395, Intel Core Ultra 258V, Snapdragon X
Elite and Snapdragon 8 Elite. Actual experiment selection may use a smaller
subset and must record which platforms were run. DGX Spark is evaluated through
its GPU. Orin's DLA has a different operator/support boundary and is excluded
until that boundary is declared and validated.

## What the comparison can establish

The same workload and finite strategy candidates can be replayed under every
declared configuration. Useful outputs are feasibility, finite cold-sequence
end-to-end simulated latency, physical memory peaks, real transfer bytes and
strategy regret relative to the best tested feasible candidate. Per-round first
and last compute timestamps are labeled compute boundaries; they exclude
mandatory copy/output completion and are not TTFT, TPOT or complete round latency.
Infeasible strategies must remain in the denominator and carry their failure
reason. Identical policies can legitimately tie under UMA when their only
difference is handling redundant CPU/GPU views.

Larger graphs should exercise the declared memory budgets. Artificially shrinking
a product's capacity to fit a toy graph creates a different scenario and must be
reported as such. Results using this catalogue support claims about modeled
policy mechanisms and sensitivity to assumptions. They do not establish product
rankings, deployment tokens/s, energy efficiency, real NPU benefits, or global
optimality. A winner that changes when assumed task times change is a finding
about sensitivity, not a hardware recommendation.

## Reuse

```python
from experiments.hardware_system_catalogue import hardware_for, system_rows

hardware = hardware_for("m4pro_mini")
metadata = system_rows()  # Includes source URLs, assumptions and exact inputs.
```

Run `python experiments/hardware_system_catalogue.py` to inspect all JSON rows.
Reading the catalogue does not access the network. Sources were inspected on
2026-10-23; the selected product generations remain fixed even if their current
marketing pages change.
