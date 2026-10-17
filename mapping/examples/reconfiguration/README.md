# 资源变化响应：一份 IR，三类模型

2026-10-12。这里是可运行的 `unified-strategy-0.3` 示例，复用现有的单层 M2、连续多层/多轮 M2 和有限 block 模型。没有策略搜索，也没有真实硬件收益声明。旧 `unified-strategy-0.2` 保持原有字段和策略行为。

## 先看发生了什么

- **m2.strategy.json**：GPU 正在读取投影权重时收到显存回收通知。策略等读锁结束，只释放 Modeling 认证的可恢复权重，再提交 ACK；投影权重和真实版本化 KV 没有被偷偷删除。之后显存恢复，后续任务继续。
- **global.strategy.json**：两层、prefill/decode 在同一时钟和副本账本下运行。CPU 黑屏暂停已经提交的 CPU 任务；跨层预取的权重遇到显存回收时也要服从安全释放和容量约束。后续尚未提交任务按当前可用设备与精确价格选择。
- **block.strategy.json**：两层、两微批、两轮的合成 block。当前 GPU 权重被锁住时先释放其他安全权重，等 kernel 完成后才能释放被锁的权重。ACK 后显存实际外部占用增加；后续重载仍有字节和耗时账本。
- **moe.strategy.json**：连续多层/多轮 staged MoE，CPU 在某个已经绑定 CPU 的专家区域执行 gate 时黑屏。gate 暂停，后续 up/SwiGLU/down 保留同一 CPU 绑定；其他 GPU 专家照常推进，真实路由仍由 Modeling 在 router 完成后公开。

三个回收例子以及额外的 MoE CPU 黑屏例子都使用合成算子/搬运成本以及显式非零控制成本：DECIDE 2 ns，ACK 3 ns，DECLINE 1 ns。这些小数字只用于验证时间语义，**不是测量值，也不是端侧延迟预测**。

## IR 新增的唯一策略对象

```json
"reconfiguration": {
  "reclaim_response": "release_clean_weights",
  "release_order": "lru",
  "cpu_response": "current_cost"
}
```

| 字段 | 可选值 | 通俗解释 |
|---|---|---|
| reclaim_response | release_clean_weights / decline | 尝试释放安全权重并接纳，或者明确拒绝。拒绝仍可以完成推理，但不算满足外部资源请求。 |
| release_order | lru / largest_first / history_frequency | 在 Modeling 给出的安全清单内选择释放顺序。LRU 是已有实际使用历史，频率仅使用已公开专家路由。 |
| cpu_response | fixed / current_cost | 维持已有设备规则，或为尚未提交的任务/专家区域按当前精确计算和输入搬运价格选设备。 |

`current_cost` 只是当前孤立路径的规则，不会预测 CPU 恢复时刻，也没有声称计算了跨层复用收益。第一次为需求提交搬运/SSD读取/计算之后保留设备绑定；正在 CPU 上暂停的任务不会改到 GPU 重新算。纯 Attention 等只提供一个合法设备的执行绑定仍遵守该限制。缺少 CPU 算子、精确成本或 workspace 时不会自动补零或生成新算子。

## 环境输入与策略分开

```json
{
  "format_version": "environment-trace-0.2",
  "events": [{"t_ns": 16000, "vram_external_bytes": 0}],
  "requests": [{
    "id": "other-app",
    "domain": "vram",
    "t_notice_ns": 3400,
    "target_external_bytes": 1046000,
    "deadline_ns": 14000
  }],
  "control_costs": {
    "ack_ns": 3,
    "decline_ns": 1,
    "decision_ns": 2,
    "source": "synthetic_control_cost_not_hardware_measured",
    "timing_scope": "dedicated_serial_control_resource"
  }
}
```

`target_external_bytes` 是接纳后外部实际占用的**绝对总量**。通知本身不占走这些显存；ACK 真正完成时才检查完整物理账本并原子生效。恢复事件由 Modeling 私有持有，在 16000 ns 发生之前不会告诉策略。公开 deadline 可以作为等待上限。

控制资源是串行的独立抽象，CPU blackout 不会暂停它；非 WAIT 控制运行时，策略等待其完成，已在途计算/搬运继续。pending 回收期间此最小 controller 优先完成回收响应，暂停新需求/预取提交。它没有模拟真实 OS、allocator、CUDA graph 重建或百分比 CPU 利用率。

Compiler 把 `ControlDecision` 编译成独立 `ControlCommand`，与 COPY/COMPUTE 分开。ACK/DECLINE 必须引用当前公开 pending ID；猜未来 ID 会被拒绝。安全权重清单来自 Modeling，M2/global/block wrapper 只做可见性过滤，不自行猜测锁或版本是否合法。固定槽 owner 释放后槽的静态 reservation 仍存在，不能把逻辑释放说成物理容量回收。

## 运行与暂停恢复

在 `Simulator` 目录的 PowerShell：

```powershell
$env:PYTHONPATH = 'mapping;modeling'
& modeling/.venv/Scripts/python.exe -m tensor_mapping.unified_strategy mapping/examples/reconfiguration/m2.strategy.json --out mapping/outputs/my-reconfiguration-full
& modeling/.venv/Scripts/python.exe -m tensor_mapping.unified_strategy mapping/examples/reconfiguration/m2.strategy.json --max-decisions 21 --out mapping/outputs/my-reconfiguration-partial
& modeling/.venv/Scripts/python.exe -m tensor_mapping.unified_strategy mapping/examples/reconfiguration/m2.strategy.json --resume mapping/outputs/my-reconfiguration-partial/checkpoint.json --out mapping/outputs/my-reconfiguration-resumed
```

输出目录必须不存在或为空。21 个 decision 的位置正处在非零 DECIDE 控制成本中，checkpoint 包括当前 pending 请求和控制账本，并通过同一绝对时钟的确定性命令重放恢复。runtime 或输入文件改变后旧 checkpoint 被拒绝。

## 产物如何读

先看 `summary.json`：`inference_goal_reached` 与 `resource_compliant_success` 是两件事。`all_published_reclaim_requests_satisfied=false` 时不能把更快的推理时间拿来宣称同等资源条件下获胜。只有已经公开的请求进入此统计，推理完成后未来尚未通知的请求不强制消耗时间。

再看 `reconfiguration.json`：逐项列出 notice/request 结果及 DECIDE、等待、ACK/DECLINE 的实际起止、成本、拒绝原因、来源 decision。`execution.json`/`events.json` 仍只列张量动作，搬运和控制成本不会重复算作 COPY。`control_cost_ns` 是已经消耗的非 WAIT 控制时间；`control_committed_cost_ns` 还包含当前已提交、尚未消耗完的成本。等待计入真实模拟时钟，与可重叠的 kernel 一起由 runtime 推进，不把所有等待简单相加到 makespan。

`decisions.json` 是策略当时真正得到的不可变观察；`compiled_graph.json`/`states.json` 是 Modeling 的私有调试产物，不能作为策略输入。真实性能校准、dirty KV 写回、运行中任务迁移、固定槽 resize 和更长期的驻留优化仍需后续独立验收。
