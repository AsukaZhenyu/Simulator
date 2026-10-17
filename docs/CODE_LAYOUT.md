# 代码地图与维护边界

## 三个模块，一套执行规则

```text
Simulator/
├── pyproject.toml             统一发行包与 llm-sim 安装入口
├── src/llm_simulator/         对外 API、显式分析成本、无 profile 示例
├── tests/                    公开 API 与干净发行包安装验收
├── examples/library/         安装后可移出源码树运行的教程
├── mapping/
│   ├── tensor_mapping/         策略前端、IR、编译、policy、有限搜索
│   ├── examples/               当前与历史策略输入
│   ├── tests/                  逻辑与端到端回归
│   ├── tools/、ggml/           查看器、绘图和计算图导出
│   └── docs/                  当前策略、执行和小图合同
├── modeling/
│   ├── llm_infer_model/
│   │   ├── components/        模型族语义
│   │   ├── m2/                LLM图、状态、物化执行和成本绑定
│   │   ├── tensor/            共用状态机、资源、容量与时钟
│   │   ├── units.py           共用单位转换
│   │   └── *.py               旧层模型与公开工具，保留兼容
│   ├── tests/、configs/
│   └── README.md              当前入口与成本边界
├── profiling/
│   ├── execution_calibration/ 最新执行变换校准
│   ├── matched_mechanisms/    端口、搬运与组合机制实验
│   ├── strategy_calibration/  完整声明Dense边界校准
│   ├── reconfiguration/       资源回收/重载实验
│   ├── native/                对应原生测量程序
│   ├── tests/、cases/
│   ├── data/                  可分发的公开路由输入与出处
│   └── results/               本机原始测量与报告，不随仓库分发
└── docs/                      当前导航和维护记录
```

对外从 `llm_simulator` 使用统一入口；内部三个模块继续共用原有执行规则。`tensor_mapping` 与 `llm_infer_model` 导入路径保持兼容。可安装库入口见[使用指南](LIBRARY_GUIDE.md)，源码维护工具与独立 profiling 按下面的模块边界保留。

## 当前策略怎样运行

1. `tensor_mapping.python_frontend.ModelPlan` 保存作用域、执行变换和调度声明。
2. `composition_ir.compile_composition_document()` 绑定模型与文档，调用 modeling 的执行变换，得到组合策略。
3. `run_unified_strategy()` 驱动策略；`CompositionPolicy` 或用户 `RuntimePolicy` 根据当前公开观察产生决策。
4. `causal.compile_command()` 将决策检查并翻译成具体命令。
5. `modeling.tensor.runtime` 使用同一状态账执行命令，记录时钟、资源、容量和依赖。

固定动作 `strategy_ir/strategy_compiler` 是另一条保留入口。新主线复用它的读取辅助函数，不是先生成旧固定程序再执行一次。搜索器可以生成方案；手写方案运行不要求先搜索。

## 新工作应该放在哪里

| 改动内容 | 放置位置 |
|---|---|
| 新的用户策略语义、选择器、声明 | mapping 前端和 IR |
| 决策合法性、策略到命令的转换 | mapping 编译层 |
| 新模型的数学依赖或物化图 | modeling components / m2 |
| 容量、在途任务、资源、时间规则 | modeling tensor |
| 新的真实成本与验证合同 | profiling，独立版本和明确边界 |
| 一次性运行结果 | 对应 outputs，按需清理 |
| 当前真实性能证据 | profiling/results，保留原始样本与来源 |

不要在 `mapping/tensor_mapping/engine.py` 兼容转出层增加状态机；不要把设备计时写死进 IR；不要从整块耗时按比例制造分片成本。

## 为什么有些旧代码还保留

- `mapping/tensor_mapping/spec.py` 和 `engine.py` 被旧调用方及测试使用，且只转出同一实现。
- `m2_*` 搜索适配器、旧 IR 和论文机制入口仍有明确用例，名称较早不等于死代码。
- modeling 的旧层模型被公开 CLI/API 以及 `tensor/layer_costs.py` 使用。
- profiling 的 v1、dual、context 程序分别对应不同实验合同，源文件和程序哈希被测量证据绑定；最新入口依然包含旧 matched helper。

因此本轮不做大规模目录重命名。未来真正移包时，要同步修改相对导入、setuptools 的显式包列表、CLI、示例与离线安装测试，并保留必要的兼容入口。

## 什么可以清，什么应保留

缓存 `__pycache__/`、`.pyc` 和未绑定的编译 `.obj` 可以再生。模型 KV backing file 是运行时临时文件，应确认路径和是否正在使用后清理。可选的本机清理记录在 `docs/maintenance/20261017-cleanup/REPORT.md`，不随仓库分发，也不是安装或阅读前提。

旧规划、历史 README、research/validation 副本与过期 outputs 已清理，不再作为阅读入口。7份仍需使用的固定计划输入已迁到 `mapping/tests/fixtures/fixed_plans/`，11项手算检查已纳入 modeling 常规测试。

`profiling/results/` 在开发者本机保留实验依据，默认被 Git 忽略。可分发的公开 MoE 路由输入已放到 [profiling/data/public-routes](../profiling/data/public-routes/README.md)，不依赖本机结果目录。已有实验绑定的 native 源码、程序和运行库应原样保留；部分 `native/build*` 中连 CMake 探测文件也可能在绑定清单中，不能按目录名整删。

当前本地 `.venv/` 与 `.egg-info/` 保持原位；测量证据的来源文件、运行库、可执行程序和冻结快照保持原位与原字节。旧实验的原始路径是验证合同的一部分。

## 测试分组

从 `Simulator/` 根目录运行主 README 的四组命令。专题无硬件测试：

```powershell
python -B -m unittest discover -s profiling/matched_mechanisms/tests
python -B -m unittest discover -s profiling/strategy_calibration/tests
python -B -m unittest discover -s profiling/reconfiguration/tests
```

真实测量只按各 profiling 工具的显式命令运行，不能把单元测试通过写成真实推理或性能验收通过。当前接口说明集中在模块 README 和 docs，不再要求阅读各阶段历史报告。
