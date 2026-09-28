# M1 设计：CPU/GPU 混合推理与 DRAM/SSD 存储层次

日期：2026-09-26。状态：待 Claude 实施。验收见 [M1_ACCEPTANCE.md](M1_ACCEPTANCE.md)。

本次修订：M1.3 从只读权重扩展为 SSD 读写，增加 `WRITE_SSD`、`dram_to_ssd` 和运行时 SSD 副本。共享读写资源统一称为 `ssd_io`，配置字段为 `ssd_io_slots`，替代此前草案的 `ssd_read_slots`。0.2 尚在开发，按本文更新实现、示例和测试；0.1 兼容要求不变。

## 1. 目标与范围

M1 在现有小计算图上，联合搜索**算子在哪个设备执行、张量何时搬运、内存副本何时释放、张量何时写入 SSD 和读回**，以请求输出就绪的时间为目标。

本轮沿用 GGML 的固定形状、连续 F32、独立存储、单输出 MUL_MAT / RELU / ADD。单请求、图只执行一次，不重计算、不原地修改。先保证小图规则与精确搜索正确，再扩展真实 dense、多层重复结构。本文是用户确定的新 M1 定义，替代早期路线中“M1 仅指实机校准”的称法。

### 分三步交付

| 子阶段 | 功能 | 进入下一步的条件 |
|---|---|---|
| M1.1 | CPU/GPU 放置、显式 DRAM 副本、H2D/D2H、输出目标位置 | CPU/GPU 选择、混合链和并行分支验收通过 |
| M1.2 | 有限 DRAM 与 VRAM，分域记账、源副本锁定和释放 | 容量、工作区、保留数据和不可行性验收通过 |
| M1.3 | SSD 权重读取、中间张量溢写/恢复、DRAM 缓存、两段搬运流水 | 读写、有限容量与联合放置场景通过，GUI 可解释结果 |

M1.1 可在开发时使用充足 DRAM 容量，最终 M1 场景必须给出有限容量。每步先完成固定计划回放，再开放对应搜索动作。不要同时另写 CPU 模拟器、SSD 模拟器和第三套时间循环。

本轮不包含：操作系统页缓存、mmap/缺页、NVMe 队列深度、DMA/CPU 共享带宽竞争、pageable/pinned 双缓冲、NUMA、多 GPU、统一内存、抢占、量化、动态 KV 追加/原地更新、跨 token 生命周期、多请求和完整 LLM。它们不能被暗中当作“已模拟”。

M1 的写入对象是不可变张量快照，先用小图验证保存和恢复规则。这提供未来 KV cache offload 所需的基本动作。真实 LLM 阶段可把每层、每段 KV 表示为不同 ID 的不可变块，另补跨 token 依赖和生命周期；不要现在引入可变文件、dirty bit 或把整个增长中的 KV 当成每步重写的同一个张量。

## 2. 现有资产与模块分工

- 扩展 `modeling/llm_infer_model/tensor/spec.py` 与 `engine.py`；这里继续是描述、合法性、状态转移、时间推进和内存记账的唯一实现。
- `mapping/tensor_mapping/mapper.py` 继续用同一内核做 uniform-cost 搜索；策略仅产生动作。
- 保留旧 `llm_infer_model` 入口、层级标定、校准数据与窗口模型；已有 `layer_costs.py` 只在粒度真实对应时复用，不能把整层成本硬填给任意算子。
- 复用 CLI、四份产物、窗口策略与现有浏览器查看器；GUI 根据资源和存储域展示，不维护物理规则。
- M1 可以增加小辅助函数，但不增加通用插件框架、新求解器、网络服务或另一套包体系。

实施前记录工作区差异和测试基线。已有查看器、启动器、样式和 GUI 修改应保留。

## 3. 固定的硬件抽象

| 资源 | 首版能力 |
|---|---|
| `cpu_compute` | 一个计算槽；耗时对应固定 CPU 线程配置，不模拟每个核心 |
| `gpu_compute` | 一个计算槽 |
| `pcie_copy` | 一个共享槽；H2D 与 D2H 互斥 |
| `ssd_io` | 一个共享读写槽；SSD→DRAM 与 DRAM→SSD 互斥 |

不同资源可并行，但动作还必须满足输入就绪、容量与锁定条件。共享 PCIe 槽是**明确的首版假设**，不代表所有硬件只有一个双向复制引擎。CPU、GPU、I/O 的并行成本暂时不互相减速，后续以实测判断是否需要竞争模型。

DRAM、VRAM 分别有容量、静态预留和对齐。每个存储域最多保存一份同名张量副本；同一个张量在两域同时存在时，两边各计一次。SSD 保存初始叶子张量及运行时写出的不可变副本，首版假设空间充足，不设置 SSD 容量约束、空间分配队列或 SSD 驱逐动作。运行时 SSD 副本是否存在、是否写完仍必须进入状态。

共享 SSD 槽是首版的明确假设，避免把各自测得的平均读写带宽同时用满。SSD 的 READY 表示在本模型中可重新读取，不承诺崩溃持久性；不模拟文件系统或 fsync。写入方法、缓存等系统优化通过测量得到的有效成本体现；选择张量、写入/读取时机及驻留位置属于调度决策。

CPU 算子成本包含该次执行的本地 DRAM 访问，GPU 成本包含本地 VRAM 访问；不再额外叠加一笔“输入字节/DRAM 带宽”的服务，否则会重复计算。SSD→DRAM 和跨设备复制是独立动作。第一版 DRAM 缓冲就是被计费的目标副本，不另造一个未记账的 staging buffer。

## 4. 输入格式：版本 0.2，保留 0.1

M1 使用 `schema_version="0.2"`，旧 0.1 场景、mapping、产物及语义继续可用。M1 场景可引用旧 0.1 GGML workload；带 SSD 初始位置的 workload 使用 0.2。只扩展位置与调度描述，不改张量形状约定或算子数值语义。

0.2 的新增定义：

- `initial_locations` 可含 `dram`、`vram`、`ssd`；初始 SSD 允许 weight 或 input 叶子，不允许有生产者的计算结果预存在任意域。初始跨域副本视为同一个不可变值。
- `goal.output_locations` 为每个 workload 输出指定唯一最终位置 `dram`、`vram` 或 `ssd`，键必须与 outputs 完全对应。SSD 目标必须等待写入完成。
- `mapspace.allowed_devices` 给每个算子一个非空设备集合，集合与对应 compute 成本一致。
- `mapspace.allowed_transfers` 给四条路线 `dram_to_vram`、`vram_to_dram`、`ssd_to_dram`、`dram_to_ssd` 列明可搬运的张量 ID。计算得到的中间值也可列入，不再要求全是初始 DRAM 叶子。未使用路线给空集合；旧 A～F 明确禁用写入。
- compute 成本按算子和设备索引；transfer 成本按路线和张量索引。只要求被允许的项具有成本，未知 ID、缺项、设备不支持和不正的成本明确拒绝。

以下是验收 F 的 96 B DRAM 场景示意；workload 在验收文档定义：

```json
{
  "schema_version": "0.2",
  "id": "m1-joint-dram96",
  "workload_file": "m1-single-ssd.workload.json",
  "architecture": {
    "dram_capacity_bytes": 96,
    "vram_capacity_bytes": 96,
    "dram_runtime_reserved_bytes": 0,
    "vram_runtime_reserved_bytes": 0,
    "dram_alignment_bytes": 1,
    "vram_alignment_bytes": 1,
    "cpu_compute_slots": 1,
    "gpu_compute_slots": 1,
    "pcie_copy_slots": 1,
    "ssd_io_slots": 1
  },
  "goal": {"output_locations": {"y": "dram"}},
  "costs": {
    "source": "synthetic_fixed",
    "compute": {
      "c1": {
        "cpu": {"duration_ns": 2000000, "workspace_bytes": 0},
        "gpu": {"duration_ns": 2000000, "workspace_bytes": 0}
      }
    },
    "transfer": {
      "dram_to_vram": {
        "x": {"duration_ns": 1000000},
        "W": {"duration_ns": 3000000}
      },
      "vram_to_dram": {"y": {"duration_ns": 1000000}},
      "ssd_to_dram": {"W": {"duration_ns": 4000000}},
      "dram_to_ssd": {}
    }
  },
  "mapspace": {
    "allowed_devices": {"c1": ["cpu", "gpu"]},
    "allowed_transfers": {
      "dram_to_vram": ["x", "W"],
      "vram_to_dram": ["y"],
      "ssd_to_dram": ["W"],
      "dram_to_ssd": []
    },
    "allow_eviction": true,
    "allow_copy_compute_overlap": true,
    "allow_cpu_gpu_overlap": true,
    "allow_recomputation": false
  },
  "mapper": {
    "algorithm": "uniform_cost",
    "max_expanded_states": 0,
    "wall_time_limit_s": 0
  }
}
```

容量为正整数，静态预留非负且不超过容量，对齐为正整数；四种槽数固定为 1，其他值本轮 unsupported。时间为正整数 ns，workspace 为非负字节；不得用 0 表示“快速设备”。

GUI 的默认搜索预算必须有限；上例的 0 仅供小图穷尽验收。GUI 修改输入后走同一加载/校验与指纹流程。

### 成本来源

内核只消费上述固定表，不在执行过程中根据并发动态修改耗时。表可以来自合成数据、实测覆盖，或明确标注的推导模型。例如生成搬运成本时可用：

`duration_ns = latency_ns + ceil(storage_bytes × 10^9 / bandwidth_bytes_per_s)`。

SSD 读写分别用平均有效带宽 `B_read` 和 `B_write` 生成成本；不假定二者相同。固定延迟可取 0，但生成后的动作时长必须为正。无需模拟扇区、FTL 或闪存内部行为。

带宽、延迟和 CPU 线程配置记录在成本说明中；固定表和来源进入场景身份，单位转换仅发生在输入边界。合成测试通过不等于真实推理预测准确。SSD 实测要注明读写方法和冷/热缓存条件，不能把缓存结果直接称为 SSD 介质带宽。

## 5. 动作与状态

0.2 mapping 的动作形式：

```json
[
  {"kind": "READ_SSD", "tensor_id": "W"},
  {"kind": "COPY", "tensor_id": "x", "src": "dram", "dst": "vram"},
  {"kind": "ADVANCE"},
  {"kind": "COMPUTE", "operation_id": "c1", "device": "gpu"},
  {"kind": "WRITE_SSD", "tensor_id": "y"},
  {"kind": "EVICT", "tensor_id": "W", "location": "dram"}
]
```

上例仅说明字段，**不是一条可执行计划**。0.2 的 COMPUTE、EVICT 必须给 device/location，不暗中默认 GPU。COPY 只支持 DRAM↔VRAM；READ_SSD 固定为 SSD→DRAM，WRITE_SSD 固定为 DRAM→SSD。SSD↔VRAM 直传和 EVICT SSD 本轮拒绝。GPU 结果溢写需要先 D2H；恢复到 GPU 需要 READ_SSD 后 H2D。

状态至少包含：

- 操作 NOT_STARTED / RUNNING / DONE；每个操作只在一个设备执行一次。
- DRAM/VRAM 副本状态 ABSENT / RESERVED_COPY / RESERVED_OUTPUT / READY；SSD 副本状态 ABSENT / RESERVED_COPY / READY。
- 正在运行任务的资源、对象、设备或 src/dst、剩余时间。
- 各任务对应的输入/源副本锁、目标/输出预留、workspace；能唯一推导的字段不另设独立账本。

CPU 计算在 DRAM 读取输入并产生输出，GPU 在 VRAM。初始 SSD 副本可静态记录，运行时写入状态必须动态记录；DRAM 不再是永远存在的源。completed_device 可记录到报告，但仅作为历史信息时不必进入搜索键。

### 内存账本

对域 d 分别满足：

`used(d) = reserved(d) + Σ READY/RESERVED 副本的 aligned_size(d) + Σ 在该域运行任务的 aligned_workspace ≤ capacity(d)`。

上述容量式只用于 DRAM/VRAM。复制和计算开始时就预留目标/输出空间，不能等待完成才计费，也不能提前释放被读取输入来凑容量。复制不会自动删源；完成只使目标 READY，源释放必须另有 EVICT。workspace 在完成时释放。

WRITE_SSD 从已有 DRAM 副本读取，不新增第二份 DRAM 缓冲；该源持续占内存并锁定至写入完成。SSD 目标标记 RESERVED_COPY，不计入 DRAM/VRAM 容量。将 GPU 副本先 D2H 到 DRAM 时，D2H 目标仍必须完整计费。

传输字节按 `storage_bytes` 记，对齐只影响分配。不要把“SSD 读取到 DRAM”与“DRAM→VRAM”合并为一笔，也不要把临时 DRAM 的同一副本算两次。

## 6. 精确动作语义

### COMPUTE(op, device)

前置：操作 NOT_STARTED，设备被允许且空闲，全部输入在对应域 READY，输出在所有域尚未产生，输出和 workspace 可同时分配。建立输入副本读锁；同一输入重复出现只锁/计费一次。

完成：操作 DONE，本域输出 READY，释放 workspace、资源和读锁；保留输入。CPU/GPU 可以执行不同的独立节点，不能同时执行同一个节点。

### COPY(tensor, src, dst) / READ_SSD(tensor) / WRITE_SSD(tensor)

前置：路线和对象被允许，源 READY，目标 ABSENT，资源空闲，目标容量足够。对象至少还有 NOT_STARTED 消费者，或是尚未在目标位置交付的请求输出；不为已无用途的数据新增副本。

READ_SSD 要求该张量在 SSD 为 READY，既可读取初始叶子，也可读回此前写出的中间值。WRITE_SSD 要求 DRAM 源 READY、SSD 目标 ABSENT；已存在或正在写入的同名 SSD 副本不能重复写入或覆盖。复制可以为未来输入保留备份，不强制只搬“下一个节点”的输入。

启动：目标立即 RESERVED_COPY，锁定源副本并占用资源；建立正时长任务。完成：目标 READY，解除源锁、释放资源。源处于 RESERVED 时不能读取，目标处于 RESERVED 时不能计算、再次复制或驱逐。

H2D/D2H 共用 PCIe 槽，READ_SSD/WRITE_SSD 共用 SSD I/O 槽。SSD 与 PCIe 可重叠，CPU 与 GPU 可重叠，前提是资源、内存和输入条件均允许。

WRITE_SSD 完成后 SSD 副本才可作为恢复来源；在途写入不能作为可释放唯一 DRAM 活值的理由。源 DRAM 在完成后仍 READY，需显式 EVICT 才腾出空间。以后 READ_SSD 恢复要再次计时和记读取字节，不得免费恢复。

`allow_copy_compute_overlap=false` 时，任意 CPU/GPU 计算与任意 PCIe/SSD 任务互斥；它不禁止 SSD 与 PCIe 互相重叠。`allow_cpu_gpu_overlap=false` 仅禁止 CPU/GPU 同时计算。

### EVICT(tensor, location)

前置：allow_eviction=true，location 为 dram/vram，该域副本 READY，无任务正在读它，不是请求输出必须保留的目标域副本。锁按副本而非整个 tensor 判定：CPU 读 DRAM 时，可释放未锁定的 VRAM 副本。

若仍有未来消费者或待交付输出，释放后必须保留至少一份有效副本，且通过已允许的路线仍能把值送到所需设备/输出域。初始和运行时写出的 READY SSD 副本均可作为备份，但需要相应的读回路线；RESERVED 目标不能作为备份。所有消费者已完成、没有待交付目标时，非目标副本可以释放。

具体检查：按该张量被允许的四条搬运路线构造三个位置间的可达关系，以释放后剩余 READY 副本为起点。每个 NOT_STARTED 消费者至少有一个 allowed_device 的输入域可达；待交付输出的指定域必须可达。未来设备尚未选定时，不要求保留所有设备选择。此检查忽略容量和忙闲，不把局部可达当作全局可执行证明。

这只检查值可恢复，不保证剩余策略一定有足够容量；容量可行性仍由后续动作和搜索判断。不能删除仍需使用的唯一中间值；不能因为“所有算子 DONE”就删掉尚未 D2H 交付的唯一 y。

效果：释放该域 allocation，置 ABSENT；不搬数据、不推进时间、不影响另一域副本。

### ADVANCE / goal

ADVANCE 仍推进到最早完成事件，批量完成同时结束的全部任务；允许在还有启动动作时主动等待，不强制 ASAP。没有运行任务时禁止 ADVANCE，不支持任意 sleep。

goal：全部必需操作 DONE，所有输出在指定目标域 READY，没有运行任务。完成计算但输出仍在另一域不算成功；SSD 目标的写入任务尚未完成也不算成功。成功后后续动作拒绝。

固定计划用尽但非 goal 为 incomplete_mapping；一个死路不等于全局 infeasible；窗口/手写策略失败不能冒充搜索无解。

## 7. 搜索与 GUI 的一致性

继续采用统一成本搜索，启动/释放边代价为 0，ADVANCE 为时间增量。只优化 makespan，不宣称同时优化峰值或搬运量。

新 key 必须含操作状态、所有动态域的副本状态（含运行时 SSD ABSENT/RESERVED/READY）、运行任务的资源/对象/设备/源目标/剩余时间，以及决定未来的锁和分配信息或等价充分表示。相同 GPU 状态但不同 DRAM 驻留、SSD 备份状态的状态不能合并。固定成本、无外部到达下仍可排除绝对时间 t；若未来加入缓存或竞争，须重新审查。

搜索的实际耗时和预算使用单调高分辨率壁钟，例如 `time.perf_counter()`；它与模拟时间 ns 分开。预算测试继续允许注入假时钟，避免依赖某台机器恰好跑多快。

严格小于才更新 best_cost；稳定打破平局；first popped goal 才证 optimal，穷尽才证 infeasible，预算截止按已知可行计划返回 feasible 或 unknown。所有结果回放同一内核。有限副本配置和正时长任务允许去重处理加载/驱逐循环，不加入未经证明的贪心剪枝。

现有 `viewer_payload.py:explore()` 为画树重走 Dijkstra。M1 开始时给正式 `search()` 增加**可选被动过程记录**，查看器消费这次搜索的记录，移除第二份搜索循环。不开记录时原调用和开销基本不变。GUI 记录上限可截断展示，但不能改变搜索决策或伪造“完整树”。

GUI 至少同步：CPU/GPU/PCIe/SSD 泳道、DRAM/VRAM 两条容量曲线、每个张量各域副本状态、目标输出位置、动作拒绝理由及搜索终止强度。SSD 泳道标出读/写方向，能看清写完、释放、读回的顺序；SSD 不画成有容量上限的内存曲线。CPU/GPU 选择来自搜索结果，不由前端重新算。纯 M0 场景继续可查看。

## 8. 兼容、产物与交付

0.1 默认维持 GPU 计算、无限且不可驱逐的静态 DRAM 源、H2D 单槽、目标在 VRAM。旧 COMPUTE/EVICT/COPY_H2D 含义不变，旧指纹必须保持原算法，不能因内部迁移改变历史计划身份。可以在输入边界规范化到公共内核，不复制整套 M0 引擎。

0.2 报告新增：`peak_dram_bytes`、各内存域当前占用、`h2d_bytes`、`d2h_bytes`、`ssd_read_bytes`、`ssd_write_bytes`、CPU/GPU 计算与各方向服务时间、目标域，以及事件的 device/src/dst。状态产物包括运行时 SSD 副本。仍使用 stats/events/states/mapping 四份文件；0.2 字段扩展明确版本化，0.1 产物保持兼容。保留动作序号，不能仅按时间排序合并同刻状态。

旧 h2d_bytes 语义不变，总搬运量若展示必须定义为四种方向字节之和，不能称其为唯一数据大小。读回已溢写张量仍逐次计费。峰值由回放计算，并包含对应域静态预留。

新指纹包含设备可选集、路线、各域容量/对齐/预留、目标位置、固定成本和有效配置。GUI 对成本或位置的修改必须更新身份。人工无指纹计划仍可完整验证，但不得标成身份已核验。

预期改动：tensor/spec、engine、公共导出；mapper 的状态与过程记录；CLI/artifacts 的 0.2 支持；M1 示例与语义/搜索测试；viewer 的数据装配和现有视图；README。不要以 GUI 演示替代内核验收。

交付顺序：记录 M0/旧模型/GUI 基线 → M1.1 固定计划与搜索 → M1.2 双域容量 → M1.3 SSD → 更新 GUI → 汇总验证。窗口策略若尚不支持 M1，应明确拒绝，不静默当作旧 GPU 策略执行。

核心合成验收不要求 GPU 或 SSD 文件。另提供至少一种成本表生成示例及实测采集说明；有环境时记录 CPU、GPU、双向复制、SSD 读写样本，注明形状、线程、后端、缓存和重复次数。缺少实测时明确标注“语义验收完成，硬件精度未验证”，不阻止独立构建，也不声称真实预测准确。

最终交付：可运行的 M1 核心、[验收规则](M1_ACCEPTANCE.md)中的计划/搜索结果、更新后的 GUI、复跑命令和一份简短实际验证记录。不要为了增加测试数量堆报告格式用例；优先覆盖设备放置、数据保留、容量和最优值。
