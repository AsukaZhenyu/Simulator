# modeling：程序、状态与性能模拟

本模块提供策略执行时共用的计算图、依赖、时钟、资源和容量账本。`mapping` 调用本模块执行给定策略；`profiling` 提供与具体执行边界匹配的成本。建模规则和测量数据分开维护。

## 当前代码分层

| 位置 | 职责 |
|---|---|
| `llm_infer_model/components/` | Attention、FFN、路由等模型族的语义检查 |
| `llm_infer_model/m2/` | 层描述、Prefill/Decode、KV、MoE、跨层/跨轮图，以及具体执行变换和成本绑定 |
| `llm_infer_model/tensor/` | 张量和设备契约、依赖、动作转移、容量、在途任务、环境和服务状态 |
| `llm_infer_model/units.py` | 秒到整数纳秒的严格转换，供不同成本入口共用 |
| 顶层 `model.py`、`analytical.py`、`simulator.py` 等 | 保留的逐层窗口分析模型及其校准、CLI 和 GGUF 工具 |

新 LLM 策略的主要路径是 `mapping → m2 → tensor`。`components` 定义模型语义，`m2` 决定具体物化图，`tensor` 决定动作是否合法和状态怎样变化。旧逐层模型继续被 CLI、公共 API 和成本桥接使用，尚不能整体删除。

## 常用入口

安装时在 `Simulator/` 根目录将两个本地包一起安装：

```powershell
python -m pip install -e ./modeling -e ./mapping
```

只使用 modeling 可以单独安装 `./modeling`。核心模拟用标准库；GGUF 提取需要额外的 `gguf` 可选依赖，未安装时对应功能明确拒绝或测试带原因跳过。

- 运行完整策略：从 [mapping 的 Python 示例](../mapping/examples/python/README.md) 开始，避免自己重复编写动作与容量规则。
- 单次 M2 invocation 回放：`python -m llm_infer_model.m2 --help`。
- 逐层分析与校准工具：`python -m llm_infer_model --help`；各子命令提供输入、参数和输出说明，示例配置在 `configs/`。

## 关键实现

| 需要查看的问题 | 位置 |
|---|---|
| 张量、硬件、成本、输入校验 | `tensor/spec.py` |
| COMPUTE/COPY/SSD/EVICT 等动作 | `tensor/engine.py` |
| 因果命令、资源服务和设备环境 | `tensor/runtime.py`、`services.py`、`environment.py` |
| 固定动作程序 | `tensor/program.py` |
| 单层定义、展开与成本查表 | `m2/spec.py`、`expand.py`、`costs.py` |
| 多层、多轮与持续状态 | `m2/global_execution.py` |
| 顺序执行变换 | `m2/execution_pipeline.py` |
| Attention/FFN 的可执行细化 | `m2/attention_execution.py`、`attention_stages.py`、`ffn_execution.py` |
| 专家路由、权重与实际状态 | `m2/moe.py`、`moe_spec.py`、`subweights.py` |

## 成本与能力边界

张量内核使用整数纳秒，旧层模型使用浮点秒。转换函数保留严格的正值和有限值检查；旧 `tensor.layer_costs.seconds_to_ns` 导入路径继续可用。`tensor/layer_costs.py` 只负责旧层成本到张量成本的显式桥接，M2 成本表不再为了转换单位而依赖该桥接模块。

任务拆分需要独立的物化节点、权重/中间结果和匹配成本，不能按比例拆一个宏任务时间。未知项不静默补值：未知 GPU 私有 workspace 仍不能用于容量验收。

逻辑模拟支持显式 Dense/MoE、多层/多轮以及声明的设备状态变化；这不表示任意真实模型、通用多 GPU 并行或操作系统都已经准确建模。当前完整性能质量门尚未通过。可在目标硬件按[执行变换校准工具](../profiling/execution_calibration/README.md)重新采集并检验；本机旧实验结果不随库分发，也不是运行前提。

详细合同见 [物理成本合同](llm_infer_model/m2/PHYSICAL_CONTRACT.md)、[资源响应运行时](RECLAIM_RUNTIME.md) 和 [代码地图](../docs/CODE_LAYOUT.md)。

## 测试和产物

在 `Simulator/` 根目录运行：

```powershell
python -B -m unittest discover -s modeling/tests
```

测试覆盖执行语义、成本合同、手算资源/容量检查和离线 wheel 安装；可选依赖不满足时保留明确 skip。没有本机 Nsight 捕获时，两项外部 Trace 回放检查也会跳过；它们不要求普通使用者保留历史捕获。`tests/fixtures/`、`tests/data/` 是固定输入，`outputs/` 是运行输出；它们不属于运行包。`.venv/` 和 `.egg-info/` 是当前本地环境，不作为生产代码。

旧层模型的代码与公开接口仍在使用，因此保留；过期规划、历史 README、旧 validation 和 outputs 已删除。原先独立验收中的11项有效检查已进入常规 `tests/test_performance_guards.py`。
