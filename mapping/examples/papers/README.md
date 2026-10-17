# 论文机制的小示例

这是 2026-10-08 的旧入口说明。当前统一 `.2` 的六项论文机制与七份配置见 [paper-ir](../paper-ir/README.md) 和 [快速对照](../../docs/PAPER_IR_QUICK_GUIDE.md)。旧资产保持历史行为。

这些配置用于检查策略机制能否被明确表达并产生可追查的动作。它们使用小尺寸和合成成本，不是论文性能复现。本文给出预期运行方式；最终完成状态、计数和证据以本轮验收报告为准。

[论文机制快速对照](../../docs/PAPER_IR_QUICK_GUIDE.md)解释每个机制和未完成的边界。

## 从哪里开始

在 `Simulator` 目录运行下面的 PowerShell 命令。输出目录必须是新的或空的；再次运行时使用另一个目录名，避免覆盖已有证据。

```powershell
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.paper_strategy mapping/examples/papers/fiddler.strategy.json --out mapping/outputs/papers-fiddler
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.paper_strategy mapping/examples/papers/hybrimoe.strategy.json --out mapping/outputs/papers-hybrimoe
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.paper_strategy mapping/examples/papers/neo.strategy.json --out mapping/outputs/papers-neo
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.block_strategy mapping/examples/papers/flexgen-layer-major.blocks.json --out mapping/outputs/papers-flexgen-layer
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.block_strategy mapping/examples/papers/flexgen-batch-major.blocks.json --out mapping/outputs/papers-flexgen-batch
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.service_program mapping/examples/papers/atsinfer.program.json --out mapping/outputs/papers-atsinfer
```

这几个入口是不同机制的具体实现，不是调用论文原仓库。`paper_strategy` 用因果观察驱动策略；`block_strategy` 将有限循环编译成动作；`service_program` 回放明确的槽位和共享服务程序。**三类入口尚未统一成一个完整策略语言**，目前不能用同一份通用配置任意组合它们；这些示例用于逐项补足表达合同，并非已经完成统一 IR。

## 每个例子看什么

| 入口 | 配置表达什么 | 应检查的证据 |
|---|---|---|
| `fiddler.strategy.json` | GPU gather/scatter 边界，CPU/GPU 两条完整专家路径 | 审计中的输入搬运、权重搬运、计算、CPU 结果返回；实际事件与所选设备一致 |
| `hybrimoe.strategy.json` | 驻留 GPU 队列按负载降序，未驻留 CPU 队列按负载升序，搬运队列按负载降序；MRS 缓存 | 队列次序、任务认领、完整 Router 分数仅在 Router 后揭示、MRS 更新前后值 |
| `neo.strategy.json` | 三个独立请求，CPU 子批一个请求、GPU 子批两个请求；纯 Attention 与输出阶段分开 | 每个请求的 Q/K/V、context、旧/新 KV；CPU context 返回；每个 B=1 算子独立计费 |
| `flexgen-layer-major.blocks.json` | 两生成步、两层、两个 microbatch；按层优先遍历 | 固定权重身份、多 consumer、实际读写和 KV 新版本存储 |
| `flexgen-batch-major.blocks.json` | 相同工作量，改为按 batch 优先遍历 | 对照权重加载/复用窗口、执行顺序及内存峰值；不能直接称为真实性能结论 |
| `atsinfer.program.json` | 固定临时槽；CE/SM copy 共同分享 PCIe；SM 占用 GPU stream | 槽位空闲仍预留、owner 不能提前覆盖、流量服务不超过预算、GPU kernel 等待 SM copy |

## 配置之间的关系

`paper-strategy-0.1` 入口引用 invocation、execution、policy、environment、history 与完整 Router 分数文件。路径相对入口文件解析，来源文件的 SHA-256 随产物保存。没有历史或分数时要明确写 `null`，不自动填未知信息。

MoE 示例共用 `moe.invocation.json`、`rows.execution.json` 和 `router.scores.json`。后者是 Modeling 的私有评估输入；策略不能因为文件在磁盘上就读取未来路由。`causal-policy-0.2` 增加 `fiddler_paths`、`hybrimoe_queues`、`mrs` 和按请求覆盖设备的 `request_placements`。原 `causal-policy-0.1` 的入口继续保留。

NEO 示例引用 `neo.execution.json` 和三个 `r_*.invocation.json`。CPU/GPU 子批划分和 policy 的 `attention_core` 设备绑定必须一致。示例分组不是融合 batch kernel：每个源请求的 batch 都是 1。

FlexGen 的两个 `.blocks.json` 明确给出循环范围、home、张量大小、计算费用和搬运费用。Compiler 物化实际图和动作，不使用“单层耗时 × 层数”的代算。SSD home 的写回要实际搬运，但当前保留 DRAM 暂存副本，后续可以直接复用它；还没有受限 DRAM 下逐块换入换出的完整 buffer 策略。

ATSInfer 的 `.program.json` 引用工作负载、场景和服务合同。临时槽容量、owner 生命周期、单流上限与共享链路带宽必须来自显式配置。服务时间按实际剩余 payload 推进，不能从旧 COPY 价格直接推出免费的 SM copy。

## 如何读结果

先看 `summary.json` 的完成状态和边界，再对照动作、事件、内存与审计。因果入口还输出 `policy_audit.json`、`history_out.json` 和 `compiled_graph.json`；完整图属于私有调试产物，不是提供给策略的观察。

这些例子不自动校准物理时间，也不测策略宿主程序开销。Fiddler 的完整路径选择是本项目扩展；HybriMoE 还没有三层预测和整体分配优化；NEO 还没有融合 batch、准入队列和多层流水线；FlexGen 没有 LP、压缩和 CPU delegation，SSD 容量视为无限；ATSInfer 没有 DP 或在线控制器，当前是 payload 共享和独占 GPU stream 合同。

修改配置时，缺少精确阶段价格、搬运价格、workspace 或输入范围应被拒绝。不要用旧宏任务价格随意拆分，不要用 top-k 分数补齐完整 Router 向量，也不要把一次合成示例成功当成真实硬件校准通过。
