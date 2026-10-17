# 共同执行接口：因果策略、固定程序和服务模型

这层解决的是“策略决定之后，所有动作由同一个运行时执行”。它不产生最优策略，也不把链路仲裁交给策略。

## 一个执行封套，保留旧动作

`llm_infer_model.tensor.runtime.ExecutionCommand` 包含：

- `action`：原有 `Action`，描述计算、复制、SSD 读写、释放或时间推进。
- `transfer_engine`：PCIe COPY 的明确 `ce` 或 `sm`；其他动作必须为 `None`。

`dispatch_command(session, command, source_index=...)` 是固定程序和因果决策共同的提交接口。它只分发，不选引擎、不改变计算图、不替策略驱逐数据。原有动作序列化没有新增引擎字段；引擎证据保存在决策和服务执行记录中。

| 运行模式 | PCIe COPY |
|---|---|
| 原有 `SimulationSession` | 使用原有显式固定搬运成本；不能提交 CE/SM 元数据 |
| `ServiceSession` | 必须明确选择 CE/SM；遗漏引擎会响亮失败 |

## 因果会话如何组合服务模型

`CausalM2Session(compiled, services=contract, environment_events=events, execution_spec=spec)` 将模型拥有的可见性投影交给 `ServiceSession`。未提供 `services` 时保留原来的运行路径。

策略仍然只收到冻结的 `CausalObservation`。`Decision.transfer_engine` 保留明确的引擎选择，`compile_command` 用当前公开对象编译成封套。原有 `compile_decision` 返回 `Action` 的合同保留，但拒绝含引擎的决策，防止元数据被悄悄丢失。

同一服务会话继续负责计算依赖、数据副本、在途预留、读锁、固定槽容量、共享链路的字节进度及外部占用。旧槽的释放必须由策略明确提交；运行时拒绝覆盖仍有所有者的槽，也拒绝释放正在被读取的数据。

## 公开服务视图与价格边界

Observation 的 `services` 字段包含链路速率、引擎上限、仲裁合同、当前流和固定槽。它没有未来环境事件。槽的成员与所有者、流的张量 ID，都经过当前张量可见性过滤；未来才产生的专家临时输出不会因槽配置提前泄露。

PCIe `TransferPrice.duration_ns` 在服务模式下是 `None`。改为明确提供 `engine_durations_ns`，同时声明 `isolated_payload_service_subject_to_contention`：这是指定引擎的独立负载估计，不是并发后的真实完成时间。实际链路时间由运行时共享带宽计算。SSD 的固定平均读写成本仍使用原合同。

模型目前采用精确分数记录字节进度、整数纳秒向上取整完成时间、显式 max-min 公平共享。SM COPY 与 GPU 计算流互斥，CE 可以按 mapspace 合同重叠。这些是声明的模型语义；没有声称已测量真实启动延迟、pinned/pageable 转换或部分 SM 占用。

## 子权重库存不会合成父副本

`PublicExpert.weight_tensor_ids` 明确列出可搬运权重。旧 opaque 专家仍有 `weight_tensor_id`；物化 gate/up/down 子权重后，父张量不存在，因此这个字段为 `None`。三个子权重的驻留分别记录。

旧 `state_out()` 无法描述部分子权重驻留，会明确拒绝此类执行。`persistent_state_document()` 导出独立 `causal-persistent-state-0.1`：KV/hidden 状态、版本化子权重库存以及未拆分权重库存。这个封套不能当作旧完整 `state_in` 直接解析，也不能把某个 gate 在 GPU 上说成整份专家在 GPU 上。多层驱动需保留自己的完整库存和检查点。

短预算停止时，同一接口导出当前时钟、各副本的 READY/RESERVED 状态和运行任务；`kv_state` 明确为 `null`，不会把正在运行的推理伪装成已完成的下一轮输入。复现恢复由顶层 driver 的绑定源码与运行时版本的命令回放检查点负责；库存快照本身明确不是检查点。

## 直接验收

- `mapping/tests/test_unified_execution.py`：因果 CPU/GPU 执行、明确引擎、服务争用和环境共用时钟、隐藏槽成员的前缀一致性、真实子权重库存。
- `modeling/tests/test_execution_runtime.py`：固定程序用同一封套执行、手算 154 ns / 144 B 搬运、缺引擎拒绝、最后读者锁。
- 原有 `mapping/tests/test_causal_mapping.py` 保持旧 CPU 基线 34,000,000 ns，并检查未来路由与未来环境不进入策略观察。

这些验收证明接口和声明语义成立，不证明论文完整算法已复现，也不证明真实机器性能准确。
