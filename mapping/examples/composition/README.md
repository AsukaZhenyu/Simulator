# 一份 IR 组合任务拆分、缓存、预取和搬运

本目录的 `unified-strategy-0.2` 示例把所有调度选择写在同一份策略文件里。模型、服务合同和轨迹仍作为可复用输入引用。完整说明见 [通俗设计指南](../../docs/COMPOSABLE_IR_GUIDE.md)。

| 文件 | 组合的机制 |
| --- | --- |
| `dense.strategy.json` | 两层 Dense prefill→decode；纯 Attention 拆分；prefill 核心用 GPU、decode 核心用 CPU；`execution_binding`；普通权重缓存和下一层预取；CE/SM |
| `moe.strategy.json` | 两层、多轮 MoE；selected rows；gate/up/SwiGLU/down 子任务和真实子权重范围；完整专家区域路径选择；expert-group MRS 缓存；因果下一层预取；固定槽及 CE/SM |
| `moe-queues.strategy.json` | 相同 staged 专家图，区域选择改用 GPU/CPU/搬运三个队列规则；继续组合 group 缓存、下一层预取和搬运选择 |
| `block.strategy.json` | 两层、两个 microbatch、两次 generation 的静态 block 遍历；GPU；SSD 权重/KV/初始及最终 hidden；LRU、下一层权重预取及 CE/SM |

`moe.scores.json` 是 modeling 侧的合成完整 router 分数证据，在对应 router 完成后才揭示给策略。`moe.services.json` 声明三个子权重固定槽。`block.model.json` 是独立的合成 block 模型；其新策略明确 `override_model_policy:true`，不执行旧固定程序。

## 从哪里开始看

先看 `block.strategy.json`。它没有额外的阶段展开条目，最容易看到同一份文件中四种选择的关系：

- `execution` 与 `policy`：选展开方式和计算设备。block 已有有限任务，所以 `execution` 是空数组。
- `cache`：指定淘汰规则、张量/专家组单位和 GPU 权重字节预算。
- `prefetch`：使用静态层顺序与已揭示历史提前搬下一层的数据；候选数量按当前来源调用计数，新调用/生成轮次可重新预取已驱逐的权重。
- `transport`：按角色、阶段、方向选择 CE/SM，以及槽重用和副本释放规则。

再看 Dense：`attention_core: "execution_binding"` 沿用每个 scope 的明确设备绑定。最后看 MoE：动态规则在 router 揭示后比较整个专家区域，不会把一个孤立 matmul 的成本冒充完整 FFN。

模型定义任务、依赖、张量和输入；成本表或 inline 阶段条目定义精确成本；轨迹定义这次评估的路由和设备扰动。策略观察没有未来路由和未来环境事件。inline 阶段条目目前仍包含成本，所以文件层面还没有完全分离 IR 与 profile。

## 运行

从 `Simulator` 目录运行，输出目录必须是新的或空的：

```powershell
$env:PYTHONPATH = "$PWD/mapping;$PWD/modeling"
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/dense.strategy.json --out mapping/outputs/composed-dense
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/moe.strategy.json --out mapping/outputs/composed-moe
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/moe-queues.strategy.json --out mapping/outputs/composed-moe-queues
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --out mapping/outputs/composed-block
```

先读 `summary.json`，再读 `policy_audit.json`。前者确认是否完成及声明边界，后者说明区域设备选择、缓存预算、预取和显式释放。`services.json` 保留实际引擎选择和共享服务记录，`execution.json`/`events.json` 给出动作和执行时间。

`decisions.json` 只有策略收到的公开观察；`compiled_graph.json`、完整 `states.json` 是 modeling 私有调试产物。不要把它们喂给策略。

暂停和恢复也使用同一个入口：

```powershell
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --max-decisions 15 --out mapping/outputs/composed-block-partial
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/composition/block.strategy.json --resume mapping/outputs/composed-block-partial/checkpoint.json --out mapping/outputs/composed-block-resumed
```

未完成而预算耗尽时返回码为 3，不表示该合法前缀失败。恢复以同一绝对时钟重建命令前缀，再继续。模型、策略、数据或运行时代码改变后会拒绝恢复。单独的 `persistent_state.json` 不具备 checkpoint 语义；主机重建开销未计入模拟时间。

## 使用时的边界

- 所有这些例子的时间、router 分数和服务速率都是合成参数，不能用作实测性能或论文性能对比。
- 任务拆分限于已有明确阶段，不是任意 tile/slice 编译器。动态专家规则选择整个专家区域的设备；任意内部异构切分尚未搜索。
- Attention 保留独立 B=1 请求口径，CPU 核心只支持 decode。prefill/decode 示例不是融合 batch 或连续请求服务系统。
- `next_layer` 是固定启发式，不是搜索算法；专家预取只使用已揭示的目标层历史/分数。调用边界和 router 开放采用保守屏障。
- 固定槽按物理容量计预留，缓存按有效载荷另计预算；部分专家组驻留也保留全组预算。SM 与 GPU 计算互斥是当前保守服务模型。
- block 使用旧 GPU 精确成本并保留指定 compute 遍历。home 是初始/最终目标，中间写回由新策略决定；SSD 写入后仍保留 DRAM 暂存副本，SSD 容量未建模。
- 支持组合论文中的部分机制，不能据此声称完整复现 Fiddler、HybriMoE、NEO、FlexGen、ATSInfer 或 KTransformers。

旧 `.1` 示例仍保留兼容入口。当前规则由 `mapping/tests/test_composition_ir.py`、`test_composition_block_ir.py` 持续回归；输出按本页命令重新生成。
