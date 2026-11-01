"""Named configurations with sourced identities and explicitly assumed costs.

No rate here is a measurement. This catalogue is for structural/sensitivity
experiments, not purchase advice, a device benchmark, or a capability database.
Vendor GB capacities are interpreted as GiB by an explicit experiment convention;
bandwidth GB/s is decimal. Neural-engine TOPS are never converted to F32 FLOP/s.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


GIB = 1024 ** 3
NPU_TIME_MULTIPLIERS = (0.5, 1.0, 2.0)
CATALOGUE_DATE = "2026-10-23"
WARNING = "Hypothetical effective task costs; not measured hardware performance."


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    supports: str


@dataclass(frozen=True)
class HardwareSystem:
    id: str
    name: str
    topology: str
    cpu_name: str
    gpu_name: str
    accelerator_name: str | None
    host_memory_gib: int
    device_memory_gib: int
    shared_memory_gib: int
    available_host_gib: int
    available_device_gib: int
    available_shared_gib: int
    cpu_tflops: float
    cpu_gbps: float
    gpu_tflops: float
    gpu_gbps: float
    cpu_launch_ns: int
    gpu_launch_ns: int
    link_gbps: float | None
    link_latency_ns: int
    ssd_read_gbps: float
    ssd_write_gbps: float
    spec_memory_bandwidth_gbps: float | None
    npu_proxy_eligible: bool
    sources: tuple[Source, ...]
    notes: tuple[str, ...]

    def __post_init__(self):
        if self.topology not in ("separate", "shared"):
            raise ValueError("invalid storage topology")
        pairs = ((self.host_memory_gib, self.available_host_gib),
                 (self.device_memory_gib, self.available_device_gib),
                 (self.shared_memory_gib, self.available_shared_gib))
        if any(not 0 <= usable <= total for total, usable in pairs):
            raise ValueError("available memory must fit its declared physical pool")
        if self.topology == "shared" and (self.host_memory_gib or self.device_memory_gib
                                           or not self.available_shared_gib):
            raise ValueError("shared memory must be represented exactly once")
        if self.topology == "separate" and (self.shared_memory_gib
                or not self.available_host_gib or not self.available_device_gib
                or self.link_gbps is None):
            raise ValueError("separate mode needs two budgets and a real-copy rate")

    @property
    def memory_mode(self):
        return self.topology

    @property
    def has_npu(self):
        return self.npu_proxy_eligible

    def hardware(self):
        """Materialize only the existing CPU+GPU model; never create an NPU."""
        from llm_simulator import Device, Hardware

        kwargs = hardware_kwargs(self)
        kwargs["cpu"] = Device(**kwargs["cpu"])
        kwargs["gpu"] = Device(**kwargs["gpu"])
        return Hardware(**kwargs)


NVIDIA_LAPTOP = Source("NVIDIA RTX 40 Series Laptop specifications",
    "https://www.nvidia.com/en-us/geforce/laptops/40-series/",
    "RTX 4060 Laptop GPU identity and 8 GB GDDR6; not a CPU/RAM configuration or measured rate")
NVIDIA_5090 = Source("NVIDIA RTX 5090 specifications",
    "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
    "RTX 5090 identity and 32 GB GDDR7; desktop CPU/RAM and effective rates are assumptions")
APPLE = Source("Apple Mac mini (2024) technical specifications",
    "https://support.apple.com/en-us/121555",
    "M4 Pro CPU/GPU, 16-core Neural Engine, configurable 64 GB unified memory and 273 GB/s bandwidth")
SPARK = Source("NVIDIA DGX Spark specifications",
    "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
    "GB10 CPU/Blackwell GPU, selected 128 GB coherent unified-memory configuration, 273 GB/s")
AMD = Source("AMD Ryzen AI Max+ 395 processor specifications",
    "https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html",
    "16-core CPU, Radeon 8060S, maximum 128 GB, 256-bit LPDDR5x-8000, NPU presence")
AMD_UNIFIED = Source("AMD Ryzen AI Max+ 395 memory description",
    "https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html",
    "Unified memory configurations up to 128 GB and 256 GB/s; up to 96 GB graphics carve-out is not free extra memory")
INTEL = Source("Intel Core Ultra 7 258V specifications",
    "https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html",
    "8-core CPU, Arc 140V GPU, 32 GB maximum memory, Intel AI Boost NPU; no F32 rate inferred from INT8 TOPS")
QUALCOMM_X = Source("Qualcomm Snapdragon X Elite specifications",
    "https://www.qualcomm.com/laptops/products/snapdragon-x-elite",
    "Oryon CPU, Adreno GPU, Hexagon NPU and LPDDR5x 135 GB/s; 32 GB is our selected OEM configuration")
ORIN = Source("NVIDIA Jetson Orin specifications",
    "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
    "AGX Orin 64 GB, 12-core Cortex-A78AE, Ampere GPU, 204.8 GB/s and two NVDLA v2 engines")
ONEPLUS = Source("OnePlus 13 North America specifications",
    "https://www.oneplus.com/us/13/specs",
    "Snapdragon 8 Elite, Oryon CPU, Adreno 830, selected 16 GB LPDDR5X/512 GB UFS 4.0 configuration")
QUALCOMM_8 = Source("Qualcomm Snapdragon 8 Elite specifications",
    "https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform",
    "Hexagon NPU presence; listed INT4/INT8/INT16/FP16 support is not evidence of F32 task support")


# All CPU/GPU TFLOP/s, effective bandwidths, launch, copy, storage rates,
# reservations and accessibility assumptions below belong to the experiment.
# Product identities/capacity options are sourced separately above.
SYSTEMS = (
    HardwareSystem("rtx4060_laptop", "RTX 4060 Laptop / configured 32 GB host", "separate",
        "Configured x86 laptop CPU (SKU unspecified)", "NVIDIA RTX 4060 Laptop GPU", None,
        32, 8, 0, 24, 6, 0, .40, 40, 6, 180, 500, 5000, 10, 1000, 2.0, 1.0,
        None, False, (NVIDIA_LAPTOP,),
        ("Host RAM/CPU and 8 GiB host + 2 GiB device reserve are author-selected.",)),
    HardwareSystem("rtx5090_desktop", "RTX 5090 desktop / configured 64 GB host", "separate",
        "Configured x86 desktop CPU (SKU unspecified)", "NVIDIA RTX 5090", None,
        64, 32, 0, 56, 28, 0, 1.0, 70, 35, 1000, 500, 5000, 24, 1000, 4.0, 2.0,
        None, False, (NVIDIA_5090,),
        ("Host RAM/CPU and 8 GiB host + 4 GiB device reserve are author-selected.",)),
    HardwareSystem("m4pro_mini", "Mac mini M4 Pro / configured 64 GB", "shared",
        "Apple M4 Pro 14-core CPU", "Apple M4 Pro 20-core GPU", "Apple 16-core Neural Engine",
        0, 0, 64, 0, 0, 56, .80, 80, 5, 190, 500, 5000, None, 0, 3.0, 2.0,
        273, True, (APPLE,),
        ("Selected configurable 14 CPU / 20 GPU core model; 8 GiB reserve is assumed.",)),
    HardwareSystem("dgx_spark", "NVIDIA DGX Spark GB10 / 128 GB", "shared",
        "20-core Arm CPU", "NVIDIA GB10 Blackwell GPU", None,
        0, 0, 128, 0, 0, 112, .60, 80, 10, 190, 500, 5000, None, 0, 3.0, 2.0,
        273, False, (SPARK,),
        ("One GB10 only; no network/multi-node experiment; 16 GiB reserve assumed.",
         "FP4 sparse AI peak is not used as F32 performance.")),
    HardwareSystem("ryzen_ai_max395", "AMD Ryzen AI Max+ 395 / configured 128 GB", "shared",
        "AMD Ryzen AI Max+ 395 16-core CPU", "AMD Radeon 8060S", "AMD XDNA 2 NPU",
        0, 0, 128, 0, 0, 112, 1.0, 100, 6, 180, 500, 5000, None, 0, 3.0, 2.0,
        256, True, (AMD, AMD_UNIFIED),
        ("Ideal shared-buffer accessibility; OS/driver graphics carve-outs are not enforced.",
         "16 GiB reserve is assumed; graphics carve-out is not a second physical pool.")),
    HardwareSystem("core_ultra258v", "Intel Core Ultra 7 258V / 32 GB", "shared",
        "Intel Core Ultra 7 258V 8-core CPU", "Intel Arc 140V", "Intel AI Boost NPU",
        0, 0, 32, 0, 0, 24, .30, 35, 2, 85, 500, 5000, None, 0, 2.0, 1.0,
        None, True, (INTEL,),
        ("Ideal shared-buffer accessibility and 8 GiB reserve assumed; no INT8 TOPS conversion.",)),
    HardwareSystem("snapdragon_x_elite", "Snapdragon X Elite / configured 32 GB laptop", "shared",
        "Qualcomm Oryon 12-core CPU", "Qualcomm Adreno X1", "Qualcomm Hexagon NPU",
        0, 0, 32, 0, 0, 24, .25, 40, 1.2, 85, 500, 5000, None, 0, 2.0, 1.0,
        135, True, (QUALCOMM_X,),
        ("32 GiB OEM RAM configuration and 8 GiB reserve assumed; no exact OEM SKU claimed.",)),
    HardwareSystem("jetson_agx_orin", "Jetson AGX Orin / 64 GB", "shared",
        "12-core Arm Cortex-A78AE", "NVIDIA Ampere 2048-core GPU", "2 x NVDLA v2 (not evaluated)",
        0, 0, 64, 0, 0, 56, .15, 35, 2, 140, 500, 5000, None, 0, 1.0, .5,
        204.8, False, (ORIN,),
        ("Optional external NVMe is assumed; 8 GiB reserve assumed.",
         "DLA is not treated as a generic F32 LLM NPU; its operator coverage is not established.")),
    HardwareSystem("snapdragon8elite_phone", "OnePlus 13 / Snapdragon 8 Elite / 16 GB", "shared",
        "Qualcomm Oryon mobile CPU", "Qualcomm Adreno 830", "Qualcomm Hexagon NPU",
        0, 0, 16, 0, 0, 10, .15, 25, 1, 55, 500, 5000, None, 0, 1.0, .5,
        None, True, (ONEPLUS, QUALCOMM_8),
        ("Selected 16 GB / 512 GB product configuration; 6 GiB reserve assumed.",
         "Storage rate models UFS payload through legacy SSD-rate fields, not an NVMe claim.")),
)


def systems_by_id():
    return {system.id: system for system in SYSTEMS}


def _resolve(system):
    return systems_by_id()[system] if isinstance(system, str) else system


def hardware_kwargs(system):
    """JSON-safe Hardware fields; nested Device dictionaries need materializing."""
    s = _resolve(system)
    result = {
        "cpu": dict(compute_flops_per_second=s.cpu_tflops * 1e12,
                    memory_bandwidth_bytes_per_second=s.cpu_gbps * 1e9,
                    launch_overhead_ns=s.cpu_launch_ns),
        "gpu": dict(compute_flops_per_second=s.gpu_tflops * 1e12,
                    memory_bandwidth_bytes_per_second=s.gpu_gbps * 1e9,
                    launch_overhead_ns=s.gpu_launch_ns),
        "dram_capacity_bytes": s.available_host_gib * GIB,
        "vram_capacity_bytes": s.available_device_gib * GIB,
        "transfer_bandwidth_bytes_per_second": None if s.link_gbps is None else s.link_gbps * 1e9,
        "transfer_latency_ns": s.link_latency_ns,
        "ssd_read_bandwidth_bytes_per_second": s.ssd_read_gbps * 1e9,
        "ssd_write_bandwidth_bytes_per_second": s.ssd_write_gbps * 1e9,
    }
    if s.topology == "shared":
        capacity = s.shared_memory_gib * GIB
        result.update(memory_mode="shared", dram_capacity_bytes=capacity,
                      vram_capacity_bytes=capacity, shared_capacity_bytes=capacity,
                      shared_runtime_reserved_bytes=(s.shared_memory_gib - s.available_shared_gib) * GIB)
    return result


def hardware_for(system):
    return _resolve(system).hardware()


def system_rows():
    """Portable provenance with no silent 'measured' or precision conversions."""
    rows = []
    for system in SYSTEMS:
        row = asdict(system)
        row.update(catalogue_date=CATALOGUE_DATE, calibrated=False,
                   hardware_measurement=False, numerical_llm_execution=False,
                   effective_rates_provenance="author_selected_hypothetical_F32_task_rates",
                   budget_provenance="author_selected_available_memory_after_reserve",
                   capacity_unit_convention="vendor GB represented as GiB for this experiment",
                   shared_accessibility="ideal_aliasable_shared_buffer" if system.topology == "shared" else None,
                   warning=WARNING, hardware=hardware_kwargs(system))
        rows.append(row)
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(system_rows(), indent=2, ensure_ascii=False, allow_nan=False))
