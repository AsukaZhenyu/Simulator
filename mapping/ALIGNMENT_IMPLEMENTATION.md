# modeling / mapping 对齐：给 Claude 的实施任务

日期：2026-09-25。状态：待实施；本文是修改任务，不代表修改已经完成。

## 1. 本轮目标

保留原模拟器及其标定能力，把已经实现的张量模拟内核归入 `modeling`，让 `mapping` 负责策略生成和搜索。旧窗口策略与搜索策略都能使用同一个张量模拟内核进行评估。

**直接复用现有代码，不重写模拟器。本轮只完成公共内核归属、窗口策略接入和小范围验证，不迁移旧模型的全部功能。**

完成后，用户应能回答：旧能力保留在哪里？搜索调用哪个模拟器？同一策略是否只有一套时间和内存规则？

本文更新本轮模块分工与安装方式，优先于 `../MODELING_MAPPING_BRIEF.md` 的架构建议，以及旧文档中“两个目录保持独立、mapping 零安装运行”的安排。`DESIGN.md` / `ACCEPTANCE.md` 中已有 M0 动作语义和验收数字继续有效。发现文档过时应直接更新，不把“冻结契约”当成不能修正文档的理由。

## 2. 现状与需要修改的部分

| 现有资产 | 本轮处理 |
|---|---|
| `modeling/llm_infer_model/model.py` 的层、硬件、策略定义与成本方法 | 保留；复用计算和传输耗时方法 |
| `analytical.py`、`simulator.py`、校准、GGUF 提取、Nsight 分析 | 保留现有入口及行为，不整包替换 |
| 标定配置、测量数据、旧测试和报告生成流程 | 保留，继续作为已有能力与回归依据 |
| `mapping/tensor_mapping/spec.py` | 移入 modeling 的张量子包，作为双方共用的描述与校验 |
| `mapping/tensor_mapping/engine.py` | 移入同一子包，作为张量模拟规则的唯一实现 |
| `mapping/tensor_mapping/mapper.py` | 保留搜索职责，直接调用 modeling 的张量内核 |
| 现有 CLI、GGML 导出器、结果格式、绘图脚本 | 保持可用，只调整必要导入及说明 |

目前 `search()` 已经调用同一个 `transition()`，并回放结果；不要重新发明这层一致性。问题在于模拟内核的归属不清、旧策略尚未接入，以及两个目录没有明确的依赖关系。

实施前先查看工作区差异。编写本文时 `mapping/README.md`、`mapping/tools/` 已有未提交修改，属于现有工作，应保留并在其上做必要调整。

## 3. 目标结构与依赖方向

```text
Simulator/
  modeling/llm_infer_model/
    model.py, analytical.py, simulator.py, ...  # 保留旧能力
    tensor/
      __init__.py
      spec.py             # 从 tensor_mapping/spec.py 迁入
      engine.py           # 从 tensor_mapping/engine.py 迁入
      layer_costs.py      # 小范围复用旧层成本的方法
  mapping/tensor_mapping/
    spec.py               # 旧导入路径的薄兼容层
    engine.py             # 旧导入路径的薄兼容层
    mapper.py             # 搜索
    policies.py           # 首先只实现窗口策略
    cli.py, artifacts.py  # 保留现有入口与结果输出
```

依赖方向只有 `tensor_mapping → llm_infer_model`。modeling 不依赖 mapping，也不通过它读取动作或评估策略。两个兼容文件只转出原来的类与函数，不保留另一份实现、不重新定义同名数据类。

不新增第三个公共包、插件框架、抽象基类体系或配置语言。现有 `Scenario` 中的 `mapper` 配置可以暂时保留，模型求值不解释搜索预算；本轮不为分层纯粹性重做 JSON schema。

公共接口沿用已有语义，至少提供：

```python
from llm_infer_model.tensor import (
    load_and_validate, load_mapping,
    initial_state, legal_actions, transition, is_goal,
    evaluate_mapping,
)

# 手写策略、窗口策略和搜索结果均以相同 Action 序列交给它。
result = evaluate_mapping(scenario, actions)
```

`mapper.py` 继续调用公共内核，不自行实现合法性、显存记账或时间推进。以后新增动作/资源时，先在 modeling 中定义和验证其语义，再让 mapping 搜索它；“能评估 mapping 的策略”限定在双方明确支持的语义范围内。

### 安装与兼容

- 在 `mapping/pyproject.toml` 声明对本仓库 `llm-infer-model` 包的依赖。两包一起从本地安装；不引入额外第三方运行库。
- `modeling/pyproject.toml` 当前显式只打包 `llm_infer_model`，必须让新增 `tensor` 子包也被打包。
- 在 Simulator 根目录提供唯一的开发安装步骤：`python -m pip install -e ./modeling -e ./mapping`。
- 保留 `python -m tensor_mapping model/search` 与 `python -m llm_infer_model` 的既有命令行为；本轮不增加第二套评估 CLI。
- 安装完成后旧的 `tensor_mapping` 公开导入路径继续有效。不要用源码中的 `sys.path.insert`、硬编码本机路径、符号链接或复制代码解决依赖。
- 更新安装说明中“无需安装即可在 mapping 单独运行”的旧表述；这是有意的本地包依赖调整。

## 4. 接入旧窗口策略：先做确定的一小步

在 `mapping/tensor_mapping/policies.py` 中提供一个简单函数：

```python
window_mapping(scenario, window_size) -> tuple[Action, ...]
```

首版只支持显式的线性链：每个计算节点使用一份独有权重与前一个节点的输出，首节点使用输入 x。根据依赖确定顺序，不把任意 DAG 的节点数组当作层序。对残差、分叉、共享权重等暂不支持的结构明确拒绝；通用搜索和评估仍保留原有支持。

窗口策略只决定动作顺序，规则如下：

1. 按链顺序加载权重、计算；一个 GPU 槽和一个 H2D 槽。
2. K 限制同时占用窗口的权重份数，在途复制、已就绪和计算使用中的权重都计数。激活和 workspace 仍由统一内核计入显存。
3. 每次完成事件后，显式释放已无用途的权重及中间值；不释放目标输出或仍有消费者的数据。
4. 尽可能启动下一项合法计算，以及窗口和容量允许的下一项加载；无法启动时通过 `ADVANCE` 等待。动作检查与推进均调用公共内核。
5. 初始位置由 scenario 决定，不因为 K 足够大就偷偷把 DRAM 权重改成 GPU 就绪。

K 必须为正整数，初始权重占用不能超过 K；策略不得绕过场景的 overlap/eviction 限制。若当前策略无法推进，应报告该策略失败，不能据此宣布整个调度空间无解。

这是旧窗口规则在张量模型中的接入，不声称复制了 v0.8 的全部 KV、控制和 staging 语义。函数输出动作序列，评估结果统一由 `evaluate_mapping` 产生；不要在策略里另算 makespan。

## 5. 复用旧成本，明确适用范围

`layer_costs.py` 只做小范围桥接：接收明确的一一对应关系（算子 ID → LayerSpec、权重张量 ID → 同一 LayerSpec），调用已有的：

```python
layer.compute_seconds(hardware)
layer.transfer_seconds(hardware)
```

这样直接复用逐层实测覆盖值优先、否则用 FLOPs/有效算力及字节数/带宽/延迟的现有规则。使用现有 `Costs` 返回，不增加成本插件体系。

- 首版用于“一层就是这个计算节点、一份权重就是这个权重张量”的小链示例；核对 ID 对应、覆盖范围和权重字节数，不按数组下标猜。
- 这不是任意 `ModelConfig` 到 GGML 图的转换器。真实 Qwen 层包含多种算子，不能把整层成本直接填给其中一个小矩阵乘，也不能平均摊给所有算子。
- 仅在入口将秒按 `round(seconds * 1_000_000_000)` 转成整数纳秒。拒绝非有限、非正或取整后非正的时长；不改变旧 API 的秒单位。workspace 由新场景明确提供，演示设为 0。
- 用 `costs.source` 标明来自旧层级成本模型；复用拟合参数不等于完成逐算子实测校准。
- 本轮桥接不迁移 KV、SSM 状态、控制开销、host staging、存储服务等额外项。演示配置明确将它们关闭；若提供接受完整旧配置的便利入口，遇到非零额外服务必须明确拒绝，不能静默丢弃或加一个任意总开销。

## 6. 验收：同一条件比较，保留旧能力

### A. 原有能力不退化

修改前记录两套测试的结果；修改后重跑。保留原来的断言，不能通过删测试、改预期数字或扩大 skip 掩盖回归。GGUF 可选依赖、GGML 编译产物缺失等情况记录实际原因，不把跳过称为通过。

从各自目录执行：

```text
modeling/: python -m unittest discover -s tests
mapping/:  python -m unittest discover -s tests -t .
```

mapping 在本次前序审查中 181 项通过、无跳过。实施时以新的基线记录为准。原链式 160 B / 96 B / 95 B 场景仍分别是 8 ms / 10 ms / 不可行，残差和分叉验收也保持不变。

### B. 建立三层链对照，验证旧策略接入

新建一个小示例：三个 4×4 F32 权重各 64 B，输入和各层输出各 16 B；`h1=W1*x, h2=W2*h1, y=W3*h2`。每次 H2D 为 3 ms、计算为 2 ms，显存 1024 B，alignment=1，workspace/reserve=0。没有 KV、控制、staging 或其他附加服务。

旧侧构造三个 `LayerSpec(weight_bytes=64, flops=32)`，用 measured 字段设定上述时长；新侧用同样的 LayerSpec 经成本桥接生成耗时。再补一个成本桥接检查，验证未提供 measured 值时确实复用原成本公式。

| 条件 | 旧 simulate_decode | 新窗口策略回放 | H2D 字节 |
|---|---:|---:|---:|
| K=1，权重初始在 DRAM | 15 ms | 15 ms | 192 |
| K=2，权重初始在 DRAM | 11 ms | 11 ms | 192 |
| K=3，三份权重初始已在 GPU | 6 ms | 6 ms | 0 |

旧侧以上数字已在写本文时直接运行验证。前两行比较加载/计算事件顺序和区间（统一为 ns），新侧允许额外的 EVICT/ADVANCE 事件。第三行必须给新 workload 设置真实的初始驻留条件：旧模型 K≥层数使用稳态驻留假设，不能拿它的 6 ms 与新模型冷启动比较。

旧 `peak_streamed_weight_bytes` 只统计权重，新 `peak_vram_bytes` 还统计激活等占用，二者不能直接要求相等。分别核对：新状态中的权重峰值应为 64 / 128 / 192 B；总显存峰值应包含额外张量且不超过 1024 B。不要把激活大小设成 0 来凑相等。

### C. 搜索与评估共用规则

- 新三层链冷启动场景的无窗口限制搜索应得到 11 ms：最后一份权重最早 9 ms 加载完，再计算 2 ms；K=2 的计划达到此界。K=1 的 15 ms 是策略限制，搜索不必复制它。
- 搜索产生的动作直接交给 `llm_infer_model.tensor.evaluate_mapping`，时间、峰值显存、搬运量和事件应与搜索返回的评估结果一致。
- 另保留手写计划与手算预期，避免只用搜索自我回放证明模拟规则正确。
- 在现有对称 fork 场景中，交换两个分支的加载/计算顺序，合法方案可以同为 9 ms。测试最优值和合法性，不要求最优动作序列唯一。
- 覆盖窗口 K 约束、容量不足、未支持图结构的明确拒绝，以及新公共 API 在不安装 mapping 时可独立导入、加载并回放计划。

用一个简短演示脚本汇总“旧窗口结果 / 新窗口回放 / 搜索结果”。复用现有结果对象或输出表格即可，不增加报告框架或 GUI。

## 7. 实施顺序与交付边界

1. **记录基线。** 查看未提交修改、运行已有测试，保留结果。
2. **迁移内核。** 移动 spec/engine、建立单向包依赖与兼容导入；先确认原 M0 测试和 CLI 行为不变。
3. **接入窗口与成本。** 实现上述受限窗口策略和成本桥接，完成三层链及 fork 对照。
4. **同步文档。** 更新两边 README、DESIGN 的模块归属/安装段落，以及旧简报；保留 M0 规则和数字。给用户一个主入口，不再追加多份互相矛盾的架构简报。

旧简报中的“必须先改造旧事件循环才能标定”“当前 fork 倒序必然非最优”“浮点与整数意味着精度和最优性只能二选一”等结论应修正。旧模型保留秒单位，新内核使用纳秒，边界转换即可；模型内的最优不等于真实硬件最优。

本轮不开发 GUI，不扩展 MoE、多请求、多 GPU、CPU 计算、D2H 或完整 dense，不改 GGML 算子支持范围，不重做指纹/状态码/产物格式。已新增的绘图脚本继续可用即可。

**完成标准：** 新 mapping 支持的策略全部由 modeling 的张量内核求值；旧窗口规则在上述共同范围内可以接入；旧入口、标定和回归能力保留；测试与演示可复现。交付说明只需列改动文件、运行命令、实际结果和尚未迁移的功能。

保留旧 `simulate_decode` 是分阶段迁移措施，不是把旧模拟器直接丢弃，也不是宣布两套模型已经全面统一。后续在张量内核支持某项旧能力、并通过对应回归后，再逐项接管旧入口的实现；本轮无需为尚未支持的能力预建通用框架。
