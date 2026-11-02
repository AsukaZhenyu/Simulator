# 执行变换、阶段放置与数据准备

本文说明当前策略表达接口；示例的时间、workspace 和传输成本均为明确声明的合成数据，不代表真实硬件加速或预测准确度。

## 1. 现在可以做什么

可以在同一份 Python 计划中组合这些选择：

| 要表达的策略 | 当前接口与行为 |
|---|---|
| Attention 内部异构执行 | 展开 QK、softmax、PV、输出投影，再分别放置到 CPU/GPU |
| Dense FFN 协作 | 沿中间维度切分 gate/up/down，CPU/GPU 分别计算，在指定设备求和 |
| MoE 专家内部协作 | 先根据模型披露的路由形成专家 token 行，再对各专家使用相同 FFN 切分机制 |
| Prefill/Decode 使用不同设备 | `PhasePlacement` 创建按 phase 选择的放置规则，KV 延续并按需实际搬运 |
| 计算前预载权重 | `DataPreparation(trigger="startup")`，搬运结束后才启动模型计算 |
| 利用空闲资源提前准备 | `DataPreparation(trigger="spare")`，没有剩余预算/容量则跳过，不为预取驱逐权重 |

模块仍是选择范围。`with plan.module(...)` 不会额外增加整模块结束屏障。以上机制共用一个 modeling runtime，所有计算、COPY、SSD I/O、驻留和释放都进入同一账本。

## 2. 先运行两个完整例子

在 `Simulator/` 目录、安装 modeling 与 mapping 后运行：

```powershell
# Dense：Attention 阶段异构 + FFN 分片 + 启动预载
# MoE：selected rows + Attention 阶段 + 专家 FFN 分片 + 预载/预取
python mapping/examples/python/execution_transforms_demo.py --case both --out Simulator/mapping/outputs/execution-transforms-demo

# 两层 Prefill GPU / Decode CPU，保留同一份 KV 和时钟
python mapping/examples/python/execution_phase_demo.py --out Simulator/mapping/outputs/execution-phase-demo

# 同一个两层、多轮运行同时组合全部机制
python mapping/examples/python/execution_combined_demo.py --out Simulator/mapping/outputs/execution-combined-demo
```

`--out` 必须是新目录或空目录。已有产物不会被覆盖。上述脚本可加 `--mode scope_barriers` 比较保守的整 scope 门控；默认 `data_dependencies` 保留真实输入依赖和跨轮 token 门控。

完整脚本：

- [execution_transforms_demo.py](../examples/python/execution_transforms_demo.py)
- [execution_phase_demo.py](../examples/python/execution_phase_demo.py)
- [execution_combined_demo.py](../examples/python/execution_combined_demo.py)
- [执行变换小例的模型与独立成本文件](../examples/python/fixtures/execution-transforms)

## 3. Python、IR、任务与性能模拟如何衔接

```text
Python ModelPlan
  ├─ module / place：选择对象与放置规则
  ├─ transform / transform_file：按顺序追加执行变换
  ├─ PhasePlacement：按 Prefill/Decode 创建放置规则
  └─ prepare：明确选择待准备的权重
             ↓
层次化策略 IR
  execution[scope].spec = execution-pipeline-0.1
  transforms = [Attention 变换, FFN 变换, ...]
  placements / data_preparations 保留策略选择
             ↓
Compiler
  校验模型接口、形状、分片覆盖、精确成本、放置和公开信息边界
  用细任务替换相应宏任务，连接输入、输出、KV 与部分结果
             ↓
同一 modeling runtime
  运行时生成准备/计算/释放动作
  检查依赖、驻留、容量、资源与版本，产生实际事件时间线
```

这里的 pipeline 是“依次应用图变换”，不是已经实现任意多设备流水服务。`plan.transform(scope, spec)` 追加变换并保留该 scope 已有的合法执行细化，不会悄悄覆盖另一个变换。重复变换同一家族、缺失成本或不满足输入合同会失败。

`scope` 指具体 invocation。例如同一层的 Prefill 和 Decode 可以是两个不同 scope，需要分别绑定适用的变换和成本。Python 循环可批量绑定；编译器不会把一轮的成本自动外推到另一轮。

## 4. Dense FFN：切的是同一个中间维度

示例中 hidden 为 16，FFN 中间维度为 32：

```text
输入 → RMSNorm（一次）
         ├─ CPU：[0,8)  gate/up → SwiGLU → down → 部分输出
         └─ GPU：[8,32) gate/up → SwiGLU → down → 部分输出
                                              ↓
                                     GPU 求和 → 残差相加（一次）
```

gate/up 的输出列和 down 的输入行必须对应相同切片。每片 down 输出的是完整 hidden 大小的部分和，最终使用 SUM 合并。不是各算一个完整 FFN 再相加。

在已有 `plan` 上绑定的最小写法：

```python
from dataclasses import replace
import json
from llm_infer_model.m2.ffn_execution import FFNShard, parse_ffn_execution

spec = parse_ffn_execution(json.loads(cost_file.read_text(encoding="utf-8")))
spec = replace(spec, shards=(
    FFNShard("host", 0, 8, "cpu"),
    FFNShard("accelerator", 8, 32, "gpu"),
), preprocess_device="gpu", merge_device="gpu")
plan.transform("dense-decode", spec)
```

`cost_file` 对应示例的 `dense.ffn-costs.json`。每个阶段需要与设备、phase、batch、tokens、visible_history、hidden 和片宽匹配的精确成本及 workspace。将切分点改为 16，需要相应片宽的成本条目；缺失时会拒绝，不会把旧 FFN 时间按比例除开。

权重分片是明确声明的初始物理表示，替换原矩阵；它不是模拟运行中免费的 slice。后续将片权重搬入 GPU 会产生真实 COPY 和容量占用。当前也没有自动补充一个未测量的运行时重排/重新分片实现。

## 5. MoE：复用 FFN 变换，路由仍归 modeling

MoE 示例的执行变换顺序是：

```python
plan.transform_file("moe-prefill", fixtures / "moe.selected-rows.json")
plan.transform_file("moe-prefill", fixtures / "moe.attention-stages.json")
plan.transform("moe-prefill", expert_ffn_spec)
```

第一步采用已有 `selected_rows + swiglu_staged` 合同，形成 gather、专家 gate/up/down、scatter 和 combine。第二步细化独立的 Attention 区域。第三步把专家内部 gate/up/down 替换为分片 FFN：中间维度 8，CPU `[0,2)`、GPU `[2,8)`，GPU 合并部分结果。MoE 小例为完整 F32 Q/K/V/O 包显式声明了 192 B 权重，原先 32 B 的 opaque 投影不能冒充该物理结构。

专家路由决定“哪些 token 交给哪些专家”，分片策略决定“已形成的一个专家任务内部如何协作”。这两件事分开表达。gather/scatter/combine 保留原有角色，Dense 的 RMSNorm/残差也不会被错误复制到每个专家。

运行时仍需先完成 router，才向 policy 披露相应专家任务。inactive 专家的静态权重、ID 和布局可以公开，未来激活序列不可公开。提前准备专家 0 或专家 3 是显式的静态选择，不能据此声称已经实现专家预测算法。

## 6. Attention：拆阶段，再决定放在哪里

已有 Dense 的物化 Q/K/V 接口可以继续展开为：

```text
Q / K / 历史 KV → QK → softmax → PV → 输出投影/原有输出处理
                                      └─ 按模型合同写入新增 KV
```

示例写法：

```python
plan.transform_file("dense-decode", fixtures / "dense.attention-stages.json")
plan.place("attention_qk", "cpu")
plan.place("attention_softmax", "cpu")
plan.place("attention_pv", "gpu")
plan.place("attention_output", "gpu")
```

CPU 生成、GPU 消费的中间张量需要搬运，模拟器依据真实依赖和独立传输成本执行，不会因 Python 里连续调用而跳过。QK、softmax、PV 都有自己的 exact cost；没有把融合 Attention 时间拆成比例。

新合同是 F32 full-causal/GQA。Dense 使用已有完整 post-RoPE QKV 接口；合成 MoE 需要显式的 F32 QKV/RoPE/输出权重 pack 和几何定义。一个旧的 opaque projection 并不足以推出这些物理端口，不能自动解释成真实模型。

## 7. P/D 分离：先实现 phase 放置与真实 KV 交接

```python
from tensor_mapping.phase_scheduling import PhasePlacement

kinds = ("projection", "attention", "dense_ffn")
PhasePlacement(
    prefill={kind: "gpu" for kind in kinds},
    decode={kind: "cpu" for kind in kinds},
).apply(plan)
```

它生成带 `Select(phase="prefill")` / `Select(phase="decode")` 的规则。原模型继续拥有 KV 的 ID、版本、位置和 token 依赖。Prefill 产生在 VRAM 的 KV，需要 Decode 在 DRAM 使用时，会生成计时、计量的 D2H COPY；不会把 standalone Decode fixture 的初始位置重新注入持续运行的状态。

当前实现的是同一 CPU/GPU 上的阶段放置。没有新增独立 P/D 服务实例、队列、请求迁移协议、额外 compute slot 或集群网络。跨请求服务流水与网络 KV 交接属于后续能力，不能用本例的“分离”一词宣称已经完成。

## 8. 数据预载与空闲预取

```python
from tensor_mapping.phase_scheduling import DataPreparation

# 指定层的各目标权重先完成一次准备，再开始计算；计入端到端时间。
plan.prepare(DataPreparation(layer_id="denseL0", trigger="startup"))

# 最多选择两个权重张量；只使用剩余空间，不为此驱逐权重。
plan.prepare(DataPreparation(layer_id="denseL1", trigger="spare", max_count=2))

# 也可以精确指定物化后的片权重或专家/阶段。
plan.prepare(DataPreparation(
    tensor_ids=("w:hybrid-interface-synthetic:W_gate:ffn:accelerator",),
    destination="vram", trigger="startup"))
```

选择条件是 `layer_id`、`stage`、`tensor_ids`、`expert_id` 的交集，目标限于公开静态 weight catalogue。至少指定一个条件。`destination` 支持 DRAM/VRAM；`stage` 必须由物化元数据提供，不会猜测 opaque 宏任务内部结构。显式张量不存在或标签矛盾会报错。

`max_count` 统计张量数，不是专家数。候选按 ID 排序并一次选定；达到上限后不会滚动选择下一批。这是显式准备规则，不是自动的最优预取算法。已有按层 lookahead、历史频率/MRS 排序的 `Scheduling.prefetch` 仍可与之组合。

startup 的缓存或物理容量不足时会明确失败；它不能静默转成按需执行。spare 不驱逐来获取空间；没有剩余空间时跳过。两者都使用声明的搬运引擎、公共成本与全局账本。SSD 到 VRAM 需要 READ_SSD→COPY 两跳。已经启动的预取必须结算，不能在推理输出到达时隐藏未完成的搬运尾巴。

启动准备是一次性的完成义务，不是持续驻留约束。目标完成准备后，正常 reclaim 和全局调度仍可将它驱逐；不会因此自动再次执行 startup 准备。因此不承诺启动阶段结束时所有目标仍同时驻留，更不承诺永久同时驻留。要表达持续驻留，需另外提供明确的驻留合同。

startup 需要内置资源响应流程，当前不能与 `RuntimePolicy(..., allow_control_override=True)` 组合；编译时会明确拒绝，避免自定义控制器被静默抢占。只有 spare 准备时，仍允许显式授权的自定义 control 决策；回调可以处理已公开的 reclaim 请求，再以 `DEFER` 交回正常执行。未授权的 control 决策仍按原有接口拒绝。

## 9. 如何确认编译和执行忠实

不要只看终端打印的时间。结果目录中重点查看：

| 产物 | 要核对的内容 |
|---|---|
| `plan.json` | transform 顺序、模块范围、phase 放置与 `data_preparations` |
| `compiled_graph.json` | 细化后的任务、权重分片、阶段元数据、真实输入输出连接 |
| `policy_audit.json` | 准备目标、启动时机、缓存和传输选择的理由 |
| `events.json` | COPY/READ_SSD/COMPUTE 的设备、字节数、开始和结束时间 |
| `states.json` | 单一账本中的占用、驻留和运行状态；此文件属于模型侧调试数据 |
| `persistent_state.json` | 持续时钟、KV ID 与最终位置，没有每轮重置 |
| `summary.json` | 是否有效完成、缺失成本/容量等错误、来源指纹与能力声明 |

`states.json`、`compiled_graph.json` 是调试证据，不能作为在线 policy 偷看未来任务或路由的输入。Python callback 是可信代码接口，不是安全沙箱。导出的 `plan.json` 目前仍用于审阅，不能据此假定已经支持脱离 Python 入口的独立加载执行。

## 10. 验证方式

在 `Simulator/` 目录运行：

```powershell
python -B -m unittest discover -s mapping/tests -t mapping -p test_phase_scheduling.py -v
python -B -m unittest discover -s mapping/tests -t mapping -p test_execution_transforms.py -v
python -B -m unittest discover -s mapping/tests -t mapping -p test_execution_phase_composition.py -v
```

测试覆盖声明校验、真实分片与求和动作、因果可见性、持续 KV/时钟、预载预算、SSD 两跳、资源响应和 checkpoint 来源绑定。查看完整例子时应重新运行命令，不依赖旧验收输出目录。

## 11. 当前边界

FFN 分片权重、Attention 的激活/KV/score/probability 当前限于 F32。Dense 原输出投影权重继续受原物理 dtype 合同约束。新增阶段均为明确声明的 synthetic execution，没有把旧实测宏任务成本转换成新阶段成本。小例的数值参考验证分解语义与 F32 舍入边界，没有加载生产模型权重、运行生产内核或校准硬件时间。量化分块、FP16、真实融合 kernel、运行时重分片和任意通信 collective 尚需独立合同与测量。

Attention 当前物化完整 score/probability 矩阵（包括最终被 mask 的位置），不是 FlashAttention 的存储/融合模型。FFN shard 的设备是固定约束；资源变化不会自动触发未声明的重新分片。其他未绑定任务仍可使用既有当前成本规则。整专家 `hybrimoe_queues` 与这种固定分片合同冲突，编译时明确拒绝。

当前是 CPU/GPU 两个设备和既有资源模型上的策略表达。不是任意多 GPU TP、独立 P/D 服务系统、任意模块融合、通用多请求微批流水或搜索器。改变硬件和策略后，仍需相应模型、合法布局与匹配成本；不存在免费推导未知性能的路径。

这些接口证明“可以用同一套编译和运行规则组合新的执行/存储选择”。论文所需的真实误差、硬件对比和策略收益，仍须单独测量和验收。

## 12. 同一个多轮运行的组合示例

`execution_combined_demo.py` 进一步把能力放到同一个两层 Prefill(3)→Decode(1) 运行：

1. 四个 invocation 都按顺序应用 AttentionStages 与 FFN 分片。
2. projection/QK/softmax/PV/输出投影从 Prefill GPU 切换到 Decode CPU。
3. FFN 的 CPU `[0,8)` / GPU `[8,32)` 绑定保持不变，合并输入数明确为 2。
4. 第 0 层三个 GPU 片权重执行 startup，第 1 层三个 GPU 片权重执行 spare。
5. 六个准备目标各发生一次 1536-byte COPY，相同叶子 ID 同时用于 Prefill 和 Decode。
6. 八个 KV ID 跨轮延续；Prefill 生成的 KV 经实际 D2H 搬运后再交给 Decode QK；跨轮 token gate 与单一时钟保留。

原物化 Dense `.5` 模型与状态来自既有 global fixtures。新阶段的 Prefill、Decode 成本在
[独立夹具](../examples/python/fixtures/execution-phase-transforms/README.md) 中分别声明，没有按 decode 时间乘 token 数。
新传输条目对应真实物化的 144/288/1536 bytes，已有 route/bytes 键值保持不变。

该组合由 `test_execution_phase_composition.py` 持续回归，命令见第10节。它检查一次运行中的机制组合与账本一致性；当前完整性能质量门尚未通过，不能由这些测试推导真实加速。真实执行的采集边界与独立核验命令见[执行变换校准工具](../../profiling/execution_calibration/README.md)，本机历史实验结果不随仓库分发。
