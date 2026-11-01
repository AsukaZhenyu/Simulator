# 代码地图与维护边界

## 目录

```text
Simulator/
├── pyproject.toml             统一发行包与 llm-sim 安装入口
├── src/llm_simulator/         对外 API、显式分析成本、示例与 CLI
├── tests/                     公开 API 与干净发行包安装验收
├── examples/library/          安装后可移出源码树运行的教程
├── tools/run_tests.py         唯一测试入口，可生成 Markdown 报告
├── docs/
│   ├── CODE_LAYOUT.md         本文件
│   └── REPRODUCTION.md        从干净克隆到跑通测试的步骤
├── mapping/
│   ├── tensor_mapping/        策略前端、IR、编译、policy、有限搜索
│   ├── examples/              当前与历史策略输入
│   ├── tests/                 逻辑与端到端回归
│   ├── tools/、ggml/          查看器、绘图和计算图导出
│   └── docs/                  当前策略、执行和小图合同
├── modeling/
│   ├── llm_infer_model/
│   │   ├── components/        模型族语义
│   │   ├── m2/                LLM 图、状态、物化执行和成本绑定
│   │   ├── tensor/            共用状态机、资源、容量与时钟
│   │   ├── units.py           共用单位转换
│   │   └── *.py               旧层模型与公开工具，保留兼容
│   ├── tests/、configs/
│   └── README.md              当前入口与成本边界
└── profiling/
    └── llm_infer_profile/     被 modeling 直接 import 的库代码：校准参数、
                               存储合同、统计与 nsys 分析
```

对外从 `llm_simulator` 使用统一入口；内部模块继续共用原有执行规则。`tensor_mapping` 与 `llm_infer_model` 导入路径保持兼容。

**本仓库只有模拟器实现和它的测试。** 测量装置（native 探针、各类 calibration campaign）、实验脚本、结果文档与原始输出都不在这里，它们随论文保留。

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
| 已测量参数的读取与转换 | profiling/llm_infer_profile |
| 新的真实成本合同或测量装置 | 不进本仓库，随论文保留 |
| 一次性运行结果 | 输出目录，不入库 |

不要在 `mapping/tensor_mapping/engine.py` 兼容转出层增加状态机；不要把设备计时写死进 IR；不要从整块耗时按比例制造分片成本。

## 为什么有些旧代码还保留

- `mapping/tensor_mapping/spec.py` 和 `engine.py` 被旧调用方及测试使用，且只转出同一实现。
- `m2_*` 搜索适配器、旧 IR 和机制入口仍有明确用例，名称较早不等于死代码。
- modeling 的旧层模型被公开 CLI/API 以及 `tensor/layer_costs.py` 使用。
- `profiling/llm_infer_profile/` 看着像测量代码，实际是被 `calibration.py`、`m2/moe_measured.py`、`m2/moe_ffn_contract.py`、`nsys_analysis.py` 直接 import 的库代码；删掉仓库就无法安装。

因此不做大规模目录重命名。未来真正移包时，要同步修改相对导入、setuptools 的显式包列表、CLI、示例与离线安装测试，并保留必要的兼容入口。

## 什么可以清，什么应保留

缓存 `__pycache__/`、`.pyc` 和未绑定的编译 `.obj` 可以再生。模型 KV backing file 是运行时临时文件，应确认路径和是否正在使用后清理。本地 `.venv/`、`*.egg-info/` 与 `mapping/ggml/build/` 被 Git 忽略，保持原位。

`mapping/tests/fixtures/fixed_plans/` 是需要使用的固定计划输入；`modeling/tests/data/` 是测试用的冻结小样本。两者都是测试的一部分，不能按目录名清理。

## 测试分组

从 `Simulator/` 根目录运行：

```powershell
python tools/run_tests.py              # 全部套件
python tools/run_tests.py --list       # 套件清单
python -m unittest discover -s modeling/tests
python -m unittest discover -s mapping/tests -t mapping
```

套件表在 `tools/run_tests.py` 的 `SUITES` 里，与 CI 的 `regression` 作业是同一份。各套件覆盖什么、失败意味着什么，见 [REPRODUCTION.md](REPRODUCTION.md)。

单元测试通过只说明模拟器与测试里编码的参考实现一致，不能写成真实推理或硬件性能验收通过。
