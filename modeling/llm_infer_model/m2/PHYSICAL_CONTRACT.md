# M2 物理执行与成本合同

## 当前范围

本文说明 `m2-layer-0.1`～`0.7` 的单次调用格式、物化端口和成本边界。更高层的组合策略及顺序执行变换见[策略表达与编译入口](../../../mapping/README.md)；它们仍须匹配具体 lowering 的实测合同，不能由格式支持推出性能校准通过。

保留 `m2-layer-0.1` 的 CPU/full causal/Dense 合成三段链。`m2-layer-0.2` 为 CPU、B=1 的物化 Q/K/V/可选 gate 输入 → Attention（含 KV 追加和 O projection）→ SwiGLU Dense FFN。`m2-layer-0.3` 为 CPU 完整声明 decoder block，`m2-layer-0.4` 为同一数学边界的单 CUDA 设备合同，见下文。0.5 接入 Dense CPU/GPU 串行设备分配；0.6 接入符号级 Full Causal MoE；0.7 保持 0.5 的宏合同，把磁盘变成有预算的存储层。范围见下文。层与 invocation 必须显式使用同一版本，旧 hidden 输入不会被重新解释成 QKV。

0.2 是**部分 decoder block**。Q/K 已完成位置变换，QKV projection、normalization、RoPE、residual add 在该版本边界外，不算免费任务。0.3 把这些计算明确纳入三个宏任务，该 CPU 合同仍不声称复现原始 Qwen 完整路径或 Flash Attention。GPU 使用独立的 0.4 合同，不重用 CPU 成本。几何示例使用明确的 CPU 显式 Attention/F32 KV 变体，不声称复现原模型的 Flash/F16 KV 路径。

## 0.3 完整声明 block

`dense_contract.py` 是测量与模型共用的配置、权重形状、端口、字节、操作清单和签名定义。输入为 F32 hidden `[H,q,1,1]` 和 I32 绝对 positions `[q,1,1,1]`。三个任务为：

1. projection：RMSNorm/scale、QKV/可选 gate projection、可选 Q/K per-head norm、normal RoPE；输出物化 Q/K/V/可选 gate。
2. attention：KV 追加、显式 causal GQA、可选 gate、O projection、第一条 residual；原 hidden 持续存活至这里。
3. dense_ffn：RMSNorm/scale、SwiGLU gate/up/down、第二条 residual；输出 hidden。

每个矩阵和 norm gamma 单独绑定任务。Norm 为真实的一维 F32 权重，不用二维占位矩阵。两个 residual 在所属宏内部计时；内部激活由 workspace 收费。RoPE base/epsilon 绑定 GGML 实际 F32 参数值，其他 RoPE 模式不静默替代。

额外测量 `block` 保留相同的物化阶段输出，用于核对三个孤立阶段中位数之和。联合图不作为第四个调度任务，也不覆盖独立 workspace。端口摘要核对相同 seed 的阶段输入来源和对应输出；独立数值参考及全量 KV 检查另保存证据。

完整用法见 [DENSE_BLOCK_MEASUREMENT.md](../../../profiling/DENSE_BLOCK_MEASUREMENT.md)，描述符为 `mapping/examples/m2/dense-block.cpu.layer.json` 和 `qwen35-geometry.declared-block.cpu.layer.json`。0.3 只接受 CPU/B=1/Full Causal/Dense FFN；`complete_declared_decoder_block` 不等于复现原模型融合实现。

## 0.4 单 CUDA 设备合同

`dense_cuda_contract.py` 复用 0.3 的数学配置、端口、权重布局、操作清单和 resident 字节计算，使用独立的 `ggml_cuda_dense_decoder_block_v1` 执行签名。物理层由 `CudaDensePhysical` 表示，必须显式给出 `cuda_device`（非负整数）和 `cuda_graphs="disabled"`。没有 CPU `threads` 字段；CPU 0.1/0.2/0.3 文档和签名保持原义。

三个可调度宏任务仍是 projection → attention → dense_ffn；联合 `block` 仅作数值与时间验证。签名绑定 `device="gpu"`、`backend="CUDA{cuda_device}"`、`implementation_variant="ggml_cuda_dense_materialized_{kind}_v1"` 和 `timing_scope="single_completed_cuda_graph_wall_ns"`。它要求 `backend_execution="single_cuda_backend_no_cpu_fallback"`。端口 F32 表示存储格式；`matmul_precision="ggml_cuda_backend_default"` 明确保留 CUDA 内部精度选择，不声称所有中间计算均为纯 F32。

当前只接入 Full Causal + Dense、B=1、一个 GPU 计算槽。权重、hidden/positions 输入、输出和完整历史/新增 KV 必须在 VRAM，`devices` 必须为 `["gpu"]`；CPU/GPU 混合、跨域搬运和启用 CUDA graphs 均显式拒绝。核心引擎只有一个 `gpu/vram` 域，`cuda_device` 指定该域对应的实际测量设备。单层物理配置跨调用固定，不表示多 GPU，也不模拟 GPU0 到 GPU1 的 KV 迁移。状态仍用通用 KV 的身份/位置/版本格式，不能据此宣称跨 GPU 状态迁移已建模。

GPU 宏任务的完整 workspace 必须同时覆盖内部张量 allocator 和后端额外 scratch。gallocr 容量不是 CUDA pool/cuBLAS 所有分配的证明。无法完整观测时，成本保存 `workspace_bytes=null`，模型拒绝执行和容量结论；不能以 0 替代。实测 GPU 成本表的观测时间仍可保存和分析，但不等于可执行、已校准的容量模型。

`mapping/examples/m2/cuda-interface-synthetic.*` 是明确的合成接口验收：用于核对 GPU 槽、VRAM 账目、绝对位置与 prefill→decode 的 KV 延续。其中 duration 与 scratch 为合成测试值，不是 GPU 实测性能或真实 workspace。实际 GPU 观测成本的未知 scratch 不会被该 fixture 补齐。

## 张量与权重

物理层必须声明 `physical` 的全部字段，不能从模型名补默认值。Q `[D,q,heads,1]`、K/V `[D,q,kv_heads,1]`、gate `[D×heads,q,1,1]` 都是独立、连续的 F32 输入；任务输出为 `[hidden,q,1,1]`。Q/G 交错视图与别名暂不支持，会明确拒绝。

权重组中，Attention 的 `output` 对应 O projection；FFN 的 `gate/up/down` 对应三个独立矩阵。每个矩阵的 `shape=[reduction,output]`、dtype、storage_bytes 均显式给出且相互校验。F32/F16 按 4/2 字节计数；Q4_K/Q6_K 按每 256 元素 144/210 字节计数，量化行必须完整。底层保留真实 GGML dtype、ne、nb，不再伪装成 F32 字节向量。

这些 dtype 仅开放给 M2 宏任务存储。原有普通 MUL_MAT/ADD/RELU 路径仍然只支持 F32。JSON 和直接构造 dataclass 都经过同样的物理边界验证。

## 容量与 workspace

- 新 K/V 输入与追加的缓存槽是两份存储，分别计费。
- KV 使用不可变逻辑块，大小为 `2×kv_heads×D×4` 字节/token。当前不表示原模型的 padded/preallocated cache，不自动缩减已有物理缓存。
- 后续状态需要的全部历史 KV 与新槽均受目标约束保护，不能以驱逐旧历史换取虚假的容量可行性。
- 独立 resident 张量按 scenario 的 alignment 计数，运行中的宏任务增加自身 scratch。GGML resident allocator 的实际 padding、host reference、线程栈和其他 runtime overhead 不在此模型中；导出报告保留测到的 padding 数字。
- CPU 0.2/0.3 成本的 `workspace_bytes` 是 GGML internal allocator capacity + CPU cplan scratch，包含内部激活。内部激活不再另建调度张量重复计数。
- 新物理层可声明 `workspace_limit_bytes`，它是 scratch 上限，不是额外常驻分配，也不是整层内存峰值。
- 旧层非零 `workspace_bytes` 按串行 scratch 最大值校验，不求和。本文件的物理宏任务入口采用单 CPU 计算槽；增加并发时须重新定义共享与独立 scratch，不能沿用单槽最大值。

0.1–0.5 的通用 `state_out` 保留 byte-explicit KV 身份/位置/版本格式，仍保存为 `m2-layer-0.1` 状态文档。0.6 的权重缓存快照使用独立 `m2-state-0.2` 格式；layer/invocation 与状态文档的版本分别校验。

## 成本绑定与质量边界

旧合成 B>1 成本明确表示 batch 上下文中的**单请求价格**。当前图按请求分别展开；整批实测成本不能复制到每条链。物理宏任务暂只接受 B=1。

物理成本必须携带完整 `physical_signature` 和规范 JSON 的 `contract_hash`。几何、矩阵 dtype、phase/q/history、CPU threads 或 CUDA device/graphs/precision、实现变体、计时协议、操作清单任一不匹配，都拒绝查表；不插值、不跨 history 复用 FFN，不默认替换。未知 workspace 可作为 `None` 保存，但不能参与执行或容量判断。

实测表必须显式给出 `validation_only`。`true` 只允许 invocation 显式设置 `allow_validation_only_costs=true` 后用于验证。`false` 表示没有波动豁免，**仍不证明完整层已校准**。成本 source 包含 manifest 和记录的内容身份；完整原始样本、质量标记、二进制/文件哈希与内存核对证据保留在 export_report。

## 可复用入口

从 Simulator 根目录运行；测量导出脚本自行定位 modeling。默认拒绝不稳定数据；`--phase prefill` 是显式选择，遗漏条目会写入报告。

```powershell
python profiling/export_m2_macros.py --measurements "<matched-macro-profile>/measurements.json" --layer mapping/examples/m2/qwen35-geometry.cpu-core.layer.json --phase prefill --output "<new-prefill-export-directory>"
```

需要显式允许不稳定数据进行诊断时，加 `--allow-unstable`；结果保持 validation-only，并选择新的空输出目录。不会覆写合成 fixture 或运行硬件测量。也可调用 `export_cost_table(...)`。

安装 modeling 后，通过模块入口保存确定性回放：

```powershell
python -m llm_infer_model.m2 --invocation "<matched-invocation.json>" --output "<new-replay-directory>"
```

源码运行可先设置 `PYTHONPATH` 指向 `modeling`。输出 workload/scenario、完整事件和状态回放、state_out、summary；summary 保留部分 block 边界、成本 source/hash、验证标记及输入文件哈希。这个入口不运行硬件，也不进行映射搜索。

## 与当前策略和测量的关系

合成 fixture 只验证格式、状态和容量规则；缺少质量标记或匹配物理签名的记录不能进入实测成本表。独立原生实验、组件模型与策略编译链的实测绑定是三层不同证据。当前完整性能质量门尚未通过；可复用测量入口、计时合同与独立核验步骤见[执行变换校准工具](../../../profiling/execution_calibration/README.md)。本机历史结果不随库分发。

## CPU 执行与线程配置

本文件的 CPU 宏任务入口使用一个计算槽；`threads` 是 GGML 内核的执行配置，绑定成本与物理签名，不表示多个请求可并行占用 CPU。内部线程分工、同步和运行时开销体现在匹配的完整任务耗时中，不逐线程模拟操作系统调度。外部应用影响未测时保持未知，不推断可用算力比例。

功能、数值与时间质量分别验收。波动数据的拒绝/validation-only 规则不变；性能准确性及策略收益仍须有匹配配置的独立执行证据。

## 0.5 单层 CPU/CUDA 混合与串行设备搜索

`HybridDensePhysical` 使用相同 Dense 配置和物化端口，按每个宏任务所选设备调用原 0.3 CPU 或 0.4 CUDA 签名。它没有第三套算子耗时模型。范围为 B=1、Full Causal、Dense、一个 CPU 槽和一个 GPU 槽；CUDA graphs 仍禁用。

`devices` 显式声明各宏任务可选择的 CPU/GPU 集合。每份权重的 initial_location、hidden/positions 输入位置、hidden 输出位置、KV state_location 与历史块副本位置均显式给出，只支持 DRAM/VRAM。跨域通过核心 COPY 动作执行，不把搬运隐藏在 COMPUTE 中。CPU projection → GPU attention 时，Q/K/V/gate 和第一条 residual 所需原 hidden 都必须就绪；历史 KV 也逐块列为输入。新增 KV 和保留历史须交付到声明状态域。

0.5 成本表新增必填 `transfer_entries` 数组，例如：

```json
{"route": "dram_to_vram", "storage_bytes": 64, "duration_ns": 10, "provenance": "synthetic_transfer"}
```

搬运按 (route, storage_bytes) 精确查表；缺尺寸、错误方向、重复条目均拒绝，不插值。合成条目写 `synthetic_transfer`；这一版也接受 `measured_transfer`，但必须有 `timing_scope`，并把 `validation_only` 显式写出（规则见 0.7「实测搬运条目」；DRAM↔VRAM 没有页缓存，所以那一路不得写 `cache_policy`）。pinned/pageable、设备身份与 staging 的实测合同仍未绑定。COPY 流量计原 storage_bytes，目标域容量按其 alignment 计占用，并保留源副本；两域 runtime_reserved_bytes 和正在运行的宏任务 scratch 仍由同一个内核计费。COMPUTE 成本已含本地访存，不能再加一遍本地带宽成本。

`plan_for(compiled, placements={operation_id: device})` 建立完整显式串行计划；必须给出全部宏任务，缺项不会猜设备。`run_invocation(compiled, actions=actions)` 可回放搜索计划并读回实际 READY 副本。原 0.1–0.4 的默认回放行为不变。

安装 modeling 与 mapping 后，从 Simulator 根目录运行：

```powershell
python -m tensor_mapping.m2_hybrid mapping/examples/m2/hybrid-interface-synthetic-prefill-3.json --out mapping/outputs/hybrid-prefill
python -m tensor_mapping.m2_hybrid mapping/examples/m2/hybrid-interface-synthetic-decode-1.json --out mapping/outputs/hybrid-decode
```

输出目录必须不存在或为空。输出 summary.json、workload.json/scenario.json、每个候选的映射/事件/状态、可直接加载的 best_mapping.json 与 state_out.json，保留输入文件 SHA256。

搜索枚举三个宏任务的设备分配，双设备为 8 个候选。每个候选采用固定的“输入就绪复制 → 计算 → 最终目标交付”串行计划，保留副本；不搜索预取、重叠、搬运顺序或驱逐。因此最优性只在 `serial_materialized_macro_device_assignments` 空间内成立，`global_schedule_optimality_proven=false`。全部候选失败也只表示此受限空间没有可行解，不能判定完整调度空间无解。展开后的 scenario 可供原 UCS 研究更大动作空间；需显式设置状态/时间预算。

合成接口示例只验证模型和搜索通路，不代表真实设备性能、混合执行数值正确性、策略收益或物理容量。实测 GPU workspace=null 仍拒绝进入可执行模型，不能用外部显存预留补成零。更广的组合和因果策略见[mapping 用法](../../../mapping/README.md)，其搜索/执行边界须分别说明。

## 0.6 符号级单层 Full Causal MoE

`m2-layer-0.6` 的 layer、invocation、成本表须同版，且明确声明合成来源。它复用张量内核的依赖、搬运、资源槽与容量规则，支持不同请求长度的 Prefill，以及每请求一个 token 的 Decode。当前范围是单层 Full Causal + MoE；没有真实 MoE 数值核或性能标定。

展开链为 projection → attention → router → 按专家分组的 expert_ffn → combine。每个专家将本次调用中分配给它的 token 跨请求合成一组，只执行一次；未选中的专家不加载。可选 shared expert 覆盖全部 token，独立于 routed top-k，不占 top-k 名额。

这层使用 **whole-request hidden 输入、grouped expert 输出与 row metadata**：专家任务读参与请求的完整 hidden，组输出按分配 token 数分配，`expert_work_items` 记录每行对应的 request_id 与绝对位置。它未把 token gather/scatter、融合 router 或加权聚合实现成真实内核；跨域搬运也按所声明的完整张量计费，不能读作仅搬了选中行。

### 路由输入和揭示门

`route_trace` 必须精确覆盖本调用全部 token，逐条给出 request_id、layer_id、absolute_position 和 top_k 个不同 expert_ids。`route_source` 必须明确。可选 scores 仅是有限数值元数据；没有数值执行验证其路由排序或加权 combine 结果。

`information_mode=causal_replay` 时，选中专家权重的 COPY、SSD 读写须等待该专家组的全部关联 router 完成，违规返回 `TRANSFER_DEPENDENCY_NOT_READY`。这是保守的 grouped router barrier；已驻留副本可继续使用，shared expert 无需此选择门。`offline_oracle` 允许已知路由下的提前搬运。

该格式的两种固定回放都先从完整 trace 展开 DAG。它的受限设备枚举工具即使使用 causal_replay，仍已知全部未来路由，标记 `search_information_mode=offline_oracle`、`online_policy_claimed=false`；不能据此宣称实现了未知路由的在线决策。Mapping 的因果 policy 入口使用另行定义的公开观察与决策合同。

### 成本、SSD 与缓存快照

projection、attention、router、combine 的成本为 `accounting=per_request`；expert_ffn 和 shared_expert_ffn 必须明确 `accounting=per_expert_group`，tokens 为该组实际 token 数，整个组只计费一次，不再乘 batch。专家 FFN 的 visible_history 明确为 0，表示该合成成本不适用历史轴。所有宏任务成本目前只接受 synthetic_layer；搬运接受 synthetic_transfer，或 0.7 定义的 measured_transfer；缺形状、字节数或方向时拒绝，不插值。

0.6 在 DRAM↔VRAM 外增加 DRAM↔SSD 的精确字节成本。SSD→VRAM 必须经 SSD→DRAM→VRAM 两段；输出或 KV 交付到 SSD 须先位于 DRAM，再完成 WRITE_SSD。SSD 没有容量账；DRAM/VRAM 的副本分配、alignment、runtime reserve 和运行中 workspace 仍由内核统一计费与加锁。

`state_in.weight_copies` 必须精确覆盖全部层权重，显式给出 id、不可变 version、storage_bytes 与当前 locations。当前快照位置优先于 layer.initial_location，不能每次 Decode 重置权重。回放成功后，KV 与权重的实际 READY 副本都写入 state_out；未完成或非法计划没有 state_out。不可变权重的版本保持原值，KV 追加时更新请求版本。inactive expert 仍保留声明的持久副本，但本调用不会选择性加载它。

带缓存的独立状态使用 `m2-state-0.2`，包含 weight_copies 和来源 layer。`state_in_from_state_out(...)` 保留 KV、权重版本、副本位置与 layer；`to_state_in_document(...)` 可生成下一次调用的内嵌状态。跨层缓存即使同名同大小也拒绝；初次内嵌 state_in 省略 layer 时，绑定本次显式 invocation.layer。旧执行边界拒绝带权重缓存的状态，避免静默丢弃。本文的容量结论只覆盖声明张量与合成 scratch，不表示已证明实际 MoE 物理容量。

### 回放与受限设备搜索

示例文件位于 `mapping/examples/m2/`：

- `moe-interface-synthetic.layer.json`、`moe-interface-synthetic.costs.json`；
- `moe-interface-synthetic-prefill-2.json`、`moe-interface-synthetic-decode-1.json`。

从 Simulator 根目录、已安装 modeling/mapping 的环境运行，输出目录须不存在或为空：

```powershell
python -m llm_infer_model.m2 --invocation mapping/examples/m2/moe-interface-synthetic-prefill-2.json --output mapping/outputs/moe-prefill-replay
python -m tensor_mapping.m2_moe mapping/examples/m2/moe-interface-synthetic-prefill-2.json --out mapping/outputs/moe-prefill-search --max-candidates 256
python -m tensor_mapping.m2_moe mapping/examples/m2/moe-interface-synthetic-decode-1.json --out mapping/outputs/moe-decode-search --max-candidates 256
```

每个调用读取其显式 state_in。示例 Decode 已嵌入验收中 Prefill 最佳计划的缓存快照；CLI 不会自动读取刚运行的 Prefill。继续自己的 Prefill→Decode 时，用 `to_state_in_document(state_in_from_state_out(previous_state_out), layer=layer_id)` 传入下一次 state_in，并匹配新 token 的绝对位置和 route_trace。

搜索只枚举展开宏任务的 CPU/GPU 分配，每个候选使用固定任务顺序、输入就绪搬运和最终目标交付的串行计划。不搜索 overlap、prefetch、eviction、recomputation、搬运顺序或路由选择。默认 max_candidates=256，只访问确定顺序的候选前缀：

| 枚举结果 | summary.status | 可得结论 |
|---|---|---|
| 全部候选枚举且有解 | optimal_within_declared_space | 仅此串行设备分配空间内最优 |
| 全部候选枚举且无解 | infeasible_within_declared_space | 仅此受限空间无解 |
| 被 max_candidates 截断且有解 | feasible | 已找到可行计划，未证最优 |
| 被 max_candidates 截断且无解 | unknown | 未访问候选的可行性未知 |

`global_schedule_optimality_proven=false` 始终保留。产物包括全部已比较候选的指标、拒绝原因、可重放 mapping/事件/状态，以及有解时的 best_mapping.json、state_out.json。summary 记录输入文件身份、route_source 与规范化 trace SHA256；这些合成实验不代表真实推理加速或准确性能预测。

## 0.7 单层 Dense 的声明式存储层

`m2-layer-0.7` 与 0.5 共用 `HybridDensePhysical`、同一组 CPU/CUDA 宏合同、同一套物化端口和同一份 `transfer_entries` 规则。它只改两件事：权重可以从磁盘起步，磁盘可以是一块有预算的域。0.5 的文档仍按原样解释，不从「0.7 支持磁盘」推出「0.5 也支持」。

### 声明就是开关

`invocation.hardware.ssd_capacity_bytes` 是唯一开关，且 0.7 必须显式给出；`ssd_alignment_bytes`、`ssd_runtime_reserved_bytes` 可选。没有它，磁盘仍是历史语义里的无界场景内容，0.7 直接拒绝，而不是悄悄退回无界。0.1–0.6 的文档不得**声明**磁盘：`ssd_capacity_bytes` 给非 `null` 值、`ssd_runtime_reserved_bytes` 非 0、或 `ssd_alignment_bytes` 非 `null` 都算声明，直接拒绝；但显式写出「没有盘」的默认值（`null`/`0`/`null`）与省略这三个键是同一台机器，照样接受——这与张量内核 `Architecture` 的判断一致，也让把 `Hardware` 数据类整份序列化出来的文档不会因为多出几个 `null` 就解析失败。

有了预算，磁盘就成为内核记账的第三个存储域：初始布局按 alignment 计入容量，运行中每次写入都要放得下，可以 EVICT，`EvaluationResult.peak_ssd_bytes` 与场景产物里的 `peak_ssd_bytes` 都报告峰值。没有预算时这三件事都不存在，历史产物的字节不变。

### 位置与搬运

layer 权重的 `initial_location` 可以是 `ssd`，其余输入、输出、KV 与历史块位置规则同 0.6。没有 `ssd_to_vram`：SSD→VRAM 必经 SSD→DRAM→VRAM，两段各自按精确字节查价并各自计 staging。这两个域里的 staging 都是显式的 COPY/WRITE_SSD/READ_SSD 动作，不隐藏在 COMPUTE 里。

`transfer_entries` 对 SSD 路线同样按 `(route, storage_bytes)` 精确查表；缺一条就拒绝整个展开，不会因为「方向对了」而复用另一条。

### 实测搬运条目

`provenance="measured_transfer"` 从 `.5` 起就允许进入成本表，`.6`/`.7` 的 storage 路线另有更严的要求，但都必须自报测的是什么：

```json
{"route": "ssd_to_dram", "storage_bytes": 1048576, "duration_ns": 579500,
 "provenance": "measured_transfer", "timing_scope": "single_io_wall", "cache_policy": "unbuffered"}
```

- `timing_scope` 必填：这一条计的是哪段区间。
- storage 路线必填 `cache_policy`；DRAM↔VRAM 不得声明（那里没有页缓存）。
- 有 measured 搬运条目时，表的 `validation_only` 必须显式写出。
- `cache_policy` 不在 `unbuffered` / `fadvise_dontneed` / `f_nocache` 里，说明这是「文件 I/O + 页缓存」的混合耗时，表必须是 `validation_only=true`，只能用于验证调用。

这些字段只说明作者声明了什么，不构成对盘控制器缓存、队列深度或写放大的核验；测量协议见 `profiling/storage/README.md`。

### 写缓存耗尽后的第二档价格

消费级 SSD 的写入不是一个价格。缓存和空闲块池还吃得下时是缓冲速度，用完之后是介质本身的速度；本机实测为 2.5 GB/s 对 0.46 GB/s，**同一次搬运差 5.4 倍**。所以 `dram_to_ssd` 允许第二档价，并由表级门限说明从什么时候开始收它：

```json
{"route": "dram_to_ssd", "storage_bytes": 1073741824, "duration_ns": 398000,
 "sustained_duration_ns": 2160000000, "provenance": "measured_transfer",
 "timing_scope": "single_aligned_write_wall_excluding_flush", "cache_policy": "unbuffered"}
```
```json
"storage_write_regime": {"after_cumulative_bytes": 42949672960}
```

- 两件事必须成对出现：有 `sustained_duration_ns` 没有门限，内核永远不知道该在哪一次切换，直接拒绝；有门限而没有任何写条目带第二档价，这个门限不改任何价格，同样拒绝。
- 第二档价只属于 `dram_to_ssd`。写缓存是磁盘的属性，读和显存搬运没有第二档；在那些路线上声明会被拒绝，而不是被忽略。
- `sustained_duration_ns` 不得小于 `duration_ns`；小于说明两列写反了。
- 内核在 `State` 里记 `storage_write_debt_bytes`（本次场景累计写入，封顶在门限），**从起写时已经越过门限的那一次开始**收第二档价，场景内不回退——服务循环里没有让盘歇过来的空闲窗口。这份负债进状态指纹：写过 40 GiB 与写过 60 GiB 是两个不同的未来。
- 没有 `storage_write_regime` 的场景行为、价格和指纹都不变。GC 压力下的读取变慢（实测 3.75 → 3.06 GB/s）不折进这个门限，它是另一个机制。
- 门限是逐次测量的量（本机同一条命令跑出过 12 / 35 / 47 GiB），不是器件常数；导出时必须由测量方写进表里，不能由导出器猜。

### 与 UMA 共存

`memory_mode: "shared"` 把 DRAM 与 VRAM 变成同一块物理池的两个名字，但**磁盘不在这块池里**：
`ssd_capacity_bytes` 仍然是它自己的预算，容量检查只数盘上的字节，不会拿内存用量去比磁盘预算。
于是同一份 `.7` layer 与成本表可以不加改动地跑在三类机器上，差别只在 `hardware` 那一段：
笔记本与台式机是两个分开的内存域，UMA 设备是一个共享池。

`mapping/examples/m2/storage-interface-synthetic-uma-prefill-3.json` 是同一层的 UMA 调用
（2 MiB 共享池 + 1 MiB 磁盘预算）：它与分开池版本得到相同的 `peak_ssd_bytes` 和
`ssd_read_bytes`；makespan 可以更短，因为同一份数据的 DRAM/VRAM 两个名字之间是
零代价的 SHARE，而分开池版本要为那次搬运付 `transfer_entries` 里的钱（示例里 7480 对
7580 ns，差的就是一条 `dram_to_vram` 的价格）。这不改变磁盘账：盘上的字节、读写量与
峰值两边一致。
共享模式仍要求 `dram_capacity_bytes` / `vram_capacity_bytes` 两个键，但它们被忽略——容量由
`shared_capacity_bytes` 决定，这里没有单独的两块内存。

### 示例

`mapping/examples/m2/storage-interface-synthetic.*` 是同一 Dense block 的 0.7 版本：矩阵权重从磁盘起步，norm 向量留在 DRAM，磁盘 1 MiB、4096 对齐，全部时间为合成常数。它验证模型与展开通路，不代表存储性能。

```powershell
python -m tensor_mapping.m2_hybrid mapping/examples/m2/storage-interface-synthetic-prefill-3.json --out mapping/outputs/storage-prefill
```

### 不建模的部分

同步单次 I/O 的精确字节耗时之外的一切：队列深度与并发 IOPS、写放大与 FTL/GC、控制器缓存、页缓存命中率、预取与异步 I/O、掉电持久性，以及 SSD 内部条带化带来的随机/顺序差异。`ssd_io_slots` 仍是一个槽，表示一次只有一笔 I/O 在途。

写缓存耗尽只建模成「一个门限 + 两档价」。不建模的是盘在两者之间的过渡（实测有一次中途掉到 594 MB/s 又回到 2.5 GB/s 再彻底掉下去），以及门限本身随盘的剩余空间和此前历史漂移的部分（同一条命令同一天跑出 12 GiB、35 GiB、47 GiB 三个门限）；这些都靠门限取测量值时的裕度来吸收。

**空闲恢复也不建模，但它确实会发生，而且很快**：第 12 GiB 掉档之后约 2.5 分钟重跑，16 GiB 满速；此后 5 s、30 s 各再追 16 GiB，仍满速——掉档后 3.6 分钟内盘又吃下 48 GiB 满速写。所以「场景内不回退」只在**连续写**的服务循环里成立；离线换模型、换专家集这种中间有间隔的负载会自己回到第一档，不能拿一次掉档给整个会话定价。

## 独立的实测 MoE FFN 子图

`moe_ffn_contract.py` 定义物理配置、端口、专家分组、工作区和计时签名，测量侧 `moe_contract.py` 引用同一份合同。`moe_measured.py` 把通过合同校验的 Router → active expert FFN → Combine 宏任务展开为既有张量内核的 CPU 串行计划；CPU 线程、精度、完整路由顺序、二进制和采样模式均须匹配。

当前真实数值核为 F32：Router linear/softmax 与 top-k 诊断；专家 gather/gate/up/SwiGLU/down；概率加权 scatter/sum/residual。分组由外部 trace 指定，probe 使用可复现随机权重，其自身 top-k 不改变指定专家。原 trace 的路由分数作为来源证据保留，计算使用随机 Router 权重生成的概率，不能声称复现原模型输出、精度或吞吐。未包含 Attention、QKV、KV、shared expert、GPU 计算或真实 offload 联合执行。

可执行成本只取各宏任务的原始样本中位数向上取整；joint 样本保留为独立验证目标，不进入价格。硬件容量、对齐和运行时预留必须全部显式给出。容量账只证明声明端口加有出处的内部工作区，尚未证明完整进程物理容量。缺价格、未知工作区、数值/端口不一致和错误签名均拒绝。CV > 0.1 的阶段默认拒绝；显式 `allow_unstable=True` 只能做标记为 validation_only 的诊断。

`profiling/validate_moe_prediction.py` 从独立 profile/holdout 目录核验原始 JSON SHA256、统计、同 seed 的所有物化端口、runtime 与签名；随后导出 workload/scenario/mapping/bindings/replay，并用指纹重新加载映射验证。`--max-relative-error` 必须由调用方显式指定，不隐式给出“已校准”。证据对应源码与二进制可选用 `--verify-live-sources` 核验；复制到另一台机器的历史证据仍可按 portable 模式读取。

注意缓存访问顺序也是测量合同的一部分。逐个专家反复执行的孤立测量会使其权重热驻留，不能直接相加预测轮换访问大量专家的整图。采样必须记录阶段与 joint 是否共用驻留张量、执行顺序与内部分配；不同口径的记录不能混用，禁止用 joint 时间拟合后冒充独立阶段测量。

v2 的 `sampling_mode=interleaved_joint_context` 共用 joint 图的 resident 张量、权重、端口和 gallocr，每轮按 Router→按 ID 排序的专家→Combine 分段计时，然后独立计时 joint。工作区为共享 gallocr 加真实 CPU plan arena，resident 的 stage 端口库存与共享总库存分开保存，不能把别的专家权重当成本阶段的 padding。签名与 `isolated_warm` v1 不兼容，导入时分别核对合同。

由于这些阶段价格依赖先前访问上下文，v2 展开图用前一专家输出增加纯 readiness 控制边，锁定测得的串行顺序。它不增加张量存储，也不表示数值 kernel 读取前一专家输出。`bindings.context_sensitive_prices=true`、`cost_execution_order=fixed_serial_slice_order` 明确记录限制；不能据此搜索任意专家换序、跨设备或并发后仍宣称成本有效。
