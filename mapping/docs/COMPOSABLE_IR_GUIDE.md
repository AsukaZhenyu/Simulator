# 把四种策略选择放进一份 IR

现在可以在一份 `unified-strategy-0.2` 文件里描述：**计算任务怎样拆、权重怎样缓存、怎样提前搬下一层的数据、每次搬运怎样选择引擎**。编译器把这些选择变成具体动作，性能器负责执行动作、推进时间和检查依赖与容量。

这次解决的是策略机制之间的组合。它没有搜索最优策略，也没有证明这些机制组合起来会在真实设备上更快。

## 先分清四类输入

| 输入 | 回答的问题 | 目前在哪里 |
| --- | --- | --- |
| IR | 我准备采用什么调度策略？ | `.strategy.json` 中的 `execution`、`policy`、`cache`、`prefetch`、`transport` |
| 模型 | 要执行哪些层、哪些轮次、哪些张量和依赖？ | `model.file` 引用的 M2 invocation、global model 或 block model |
| 性能数据 | 给定形状在给定设备上要多久、需要多少工作区？ | 模型引用的成本表；部分阶段成本仍写在 `execution.spec.entries` 内 |
| 轨迹 | 本次评估发生了什么？ | 模型内的路由轨迹、`routing_scores_file`、`environment_file`，以及可选的历史记录 |

“一份 IR”指调度选择集中在一份文件里，不要求把模型、实测数据和设备轨迹都复制进去。当前 JSON 还保留了阶段展开所需的精确成本条目，所以文件层面尚未把 IR 和 profile 完全分离。成本的来源和适用形状仍必须明确；不能把一块 FFN 的时间随意分摊给 gate/up/down。

模型和未来轨迹保留在 modeling 一侧。策略每次只收到当前公开观察：已揭示的任务、当前库存、设备当前状态、已完成 router 的结果和历史。完整调试图、未来 router 分数和未来设备事件都不作为策略输入。

## 第一类选择：怎样拆任务，在哪里计算

`execution` 按 `scope_id` 选择阶段展开方式。全局模型中，prefill 的第 0 层与 decode 的第 0 层是不同 scope，因此可以使用不同执行绑定。未选中的 scope 保留模型原有、经过校验的展开方式。

当前有两种主要展开：

- Dense 的 `pure_attention`：把纯 Attention 核心与输出投影/残差部分分开。核心接收物理 Q/K/V 端口，包含 causal QK、softmax、PV 和对应的逻辑 KV 追加。
- MoE 的 `selected_rows`：只收集当前 router 已选中的 token 行，再执行专家和 scatter/combine。专家可以保持 `opaque`，也可以用 `swiglu_staged` 展开为 gate、up、SwiGLU、down。

`policy.placements` 指定任务种类的设备，`request_placements` 可覆盖某个请求。新版本还支持：

```json
"attention_core": "execution_binding"
```

它表示“使用这个 scope 的 `execution.spec.subbatches` 明确绑定的 Attention 设备”。例如 [Dense 例子](../examples/composition/dense.strategy.json) 的 prefill 核心绑定 GPU，decode 核心绑定 CPU；投影、输出和 FFN 仍用 GPU。`execution_binding` 必须命中已拆出的 pure Attention 核心及唯一设备价格，不能作为猜测默认设备的捷径。

CPU Attention 目前只支持 decode。subbatch 是独立 B=1 请求的组织方式，不代表把多个请求变成一个融合 batch 内核。

MoE 的动态专家选择也能作用于拆开的专家。`selection_unit: "expert_region"` 表示，选择器比较完整专家区域的 CPU/GPU 路径，计入该区域的各阶段、输入、权重和输出返回，再固定这个区域的设备。`fiddler_paths`、`hybrimoe_queues` 可以与 staged 专家组合。它们不会只比较 gate 的时间就声称比较了完整 FFN。

动态区域选择要求 GPU gather/scatter 边界、完整且一致的已揭示专家行，以及精确 CPU/GPU 阶段成本。固定设备规则则可以明确指定各阶段的不同设备。动态规则当前固定整个专家区域的设备，不自动寻找区域内部的任意 CPU/GPU 切分。

## 第二类选择：缓存谁，以什么为单位计预算

`cache` 把“淘汰顺序”和“预算单位”分开。例如：

```json
{
  "rule": "lru",
  "unit": "tensor",
  "budget_bytes": 128,
  "alpha": null,
  "top_p": null
}
```

`tensor` 按物理权重张量计预算；拆开的 gate/up/down 可以分别驻留和驱逐。`expert_group` 把同一层、同一专家的子权重作为一个缓存组。只要组里仍有一个成员驻留或在途，预算就保留整个组的字节数；逐条驱逐时，不能先释放尚未完成驱逐的组预算。

预算包括声明的全部 GPU 权重缓存，不只预取数据。固定槽使某次搬运的新增物理分配为零，也不免除权重有效载荷的缓存预算。缓存预算和 VRAM 总容量是两项独立检查：VRAM 还包含 KV、激活、工作区和槽的预留空间。

规则可选 `none`、`lru`、`history_frequency`、`mrs`。MRS 需要完整 router 分数证据；只有 top-k ID 不足以冒充完整分数。分数由 modeling 校验并在对应 router 完成后揭示。普通权重没有 router 分数，它们仍受预算管理。

## 第三类选择：提前搬哪一层的数据

`prefetch` 明确写下一层范围、层数、候选数、排序依据，以及是否包含普通 Dense 权重。例如：

```json
{
  "enabled": true,
  "scope": "next_layer",
  "lookahead_layers": 1,
  "max_count": 2,
  "ranking": "history_frequency",
  "include_dense_weights": true
}
```

策略知道静态层顺序和权重目录，可以提前搬下一层的普通权重。预测专家时，只使用目标层已有的激活历史或已揭示分数。没有历史证据就不会读取下一层未来的真实 top-k 来“预测”。预测未命中的搬运仍消耗时间、带宽和容量。

这是一个显式的 `next_layer` 启发式规则。它不会枚举所有预取位置，不搜索最佳预取距离，也不自动跨最后一层预取下一轮第一层。`max_count` 按当前来源调用限制预取缓存单位；进入新的层调用或生成轮次时，使用该新调用的独立额度，已经驱逐的下一层权重也可以重新预取。审计记录明确保存来源调用和计数范围。

## 第四类选择：怎样搬运，怎样重用暂存槽

`transport.rules` 同时匹配张量角色、可选阶段和方向：

```json
{
  "role": "weight",
  "stage": "down",
  "src": "dram",
  "dst": "vram",
  "engine": "sm"
}
```

同一份 IR 可以规定普通权重用 CE，而 down 子权重用 SM；反方向也要显式指定规则。`stage: null` 是该角色和方向的通用规则，更具体的阶段规则优先；相同匹配条件的重复规则被拒绝。服务开启时，engine 必须是 `ce` 或 `sm`，不能静默推断。未开启服务时应明确使用 `null`。

SSD↔DRAM 读写沿用精确传输成本表，不由 CE/SM 规则控制。两种 PCIe 引擎共用服务合同中的总带宽；SM 与 GPU 计算互斥是当前保守模型。用于选择路径的单独引擎时间只是估计，实际竞争后的完成时间由性能器决定。

固定槽由 `services_file` 声明物理容量和允许成员。`slot_reuse` 只能产生显式驱逐动作，并保留当前读者、在途副本和必要来源；槽不会无成本覆盖一个仍在使用的张量。`boundary_release` 可释放已证明用完的中间值，以及有可恢复来源的非目标 KV 副本，相关理由留在审计中。

## block 也使用同一套接口

[block 例子](../examples/composition/block.strategy.json) 使用相同 policy、cache、prefetch 和 transport 对象。model 额外要求：

```json
{
  "kind": "block",
  "file": "block.model.json",
  "override_model_policy": true
}
```

这个声明明确让新 IR 接管旧模型生成的固定程序和权重释放规则。旧 block 的 GPU 精确计算成本仍是模型约束，不会自动补出 CPU 成本。模型按指定的 layer-major/batch-major 顺序开放计算，搬运与预取可以重叠。

block 的 home 描述初始来源和最终目标；中间 hidden 是否写回 SSD，由新策略实际决定。当前会保留写入 SSD 后的 DRAM 暂存副本，所以这不是完整的有界 offload buffer 系统。

## 怎样运行和看证据

从 `Simulator` 目录运行，使用已安装两个本地包的 Python 环境。源码开发环境也可以显式设置路径：

```powershell
$env:PYTHONPATH = "$PWD/mapping;$PWD/modeling"
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/dense.strategy.json --out mapping/outputs/composed-dense
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/moe.strategy.json --out mapping/outputs/composed-moe
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --out mapping/outputs/composed-block
```

输出目录必须是新的或空的。先看 `summary.json` 的状态和边界，再看 `policy_audit.json` 中的区域设备、预算驱逐、跨层预取及搬运理由。`services.json` 给出实际提交的引擎和服务竞争；`execution.json`、`events.json` 给出对应动作和时间。

`decisions.json` 保存策略实际收到的观察。`compiled_graph.json` 和完整 `states.json` 是 modeling 私有调试证据，不应作为策略输入。`persistent_state.json` 是库存导出；恢复要使用 `checkpoint.json`，不能仅把库存文件当作下一次的起始状态。

```powershell
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --max-decisions 15 --out mapping/outputs/composed-block-partial
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --resume mapping/outputs/composed-block-partial/checkpoint.json --out mapping/outputs/composed-block-resumed
```

预算耗尽但未完成时，CLI 返回码为 3，仍可产生合法 checkpoint。恢复重新执行已保存的因果命令前缀，验证时钟、库存、环境和在途服务一致，再继续；来源或运行时代码改变时拒绝恢复。主机侧重建开销未计入模拟时间。

## 当前边界

这是有限阶段的策略组合实现，尚不支持任意 tile、任意切片和通用别名变换。模型使用有限的层/轮次图，调用边界保守开放；MoE 还等待该调用的 router 完成后才开放动态专家任务。它不具备完整的 token 流水、请求到达/退出或融合 batch 服务系统。

这些例子的阶段成本、路由分数和带宽是合成参数。它们用于检查策略表达和执行语义，不证明硬件精度，不用于论文之间性能排名。未知工作区不能写成零；新阶段仍需要对应 profile 才能校准。统一表达部分 Fiddler、HybriMoE、NEO、FlexGen、ATSInfer、KTransformers 机制，也不等于复现了它们的完整系统、控制器、内核或论文结果。

兼容的旧 schema 仍可通过原入口解析，新版本不静默迁移策略。当前回归直接运行 `mapping/tests/`；执行变换与 Python 入口见[当前执行指南](EXECUTION_TRANSFORMS_GUIDE.md)。
