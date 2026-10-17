# mapping：策略表达与编译

本模块负责把执行策略和存储调度规则转换为可检查、可追溯的动作，交给 `modeling` 的同一个运行时执行。它不另建一套时间和容量规则。当前推荐从模块化 Python 计划入手；小图精确搜索和旧 JSON 入口仍可使用。

## 当前主线

```text
ModelPlan / PlanModule / Scheduling
    ↓ compile()
composition_ir：组合表达、约束检查和来源绑定
    ↓ modeling.m2.execution_pipeline
图细化、Attention 阶段、Dense/MoE FFN 分片与阶段放置
    ↓ run_unified_strategy()
CompositionPolicy / RuntimePolicy → Decision
    ↓ causal.compile_command()
modeling.tensor.runtime：执行动作、推进时间、更新状态
```

模块分组用于选择作用域和记录来源，本身不增加屏障。MoE 的真实路由属于 modeling，策略只读取当前观察与已揭示历史。旧 `strategy_compiler.compile_strategy()` 服务显式固定动作 IR；它与当前逐决策编译入口分开，新主线会复用其中的文档读取辅助函数。

## 推荐阅读和运行

1. [Python 策略入门](examples/python/README.md)：先理解模块放置，再看因果回调。
2. [执行变换指南](docs/EXECUTION_TRANSFORMS_GUIDE.md)：Attention 细化、FFN 分片、P/D 放置和数据准备。
3. [组合 IR 指南](docs/COMPOSABLE_IR_GUIDE.md)：缓存、预取、传输与执行变换如何共用一份策略。

在 `Simulator/` 根目录运行：

```powershell
python -m pip install -e ./modeling -e ./mapping
python mapping/examples/python/execution_combined_demo.py --out mapping/outputs/combined-demo
python mapping/examples/python/modular_moe.py --variant swiglu_staged --out mapping/outputs/moe-demo
```

`--out` 必须是新目录或空目录。示例使用合成成本，证明表达和执行规则，不证明实机加速。当前完整性能质量门尚未通过；需要实机成本时，按[执行变换校准工具](../profiling/execution_calibration/README.md)在目标硬件重新采集、独立验证。运行示例不要求已有 profile 或本机实验结果。

## 代码位置

| 职责 | 模块 |
|---|---|
| 人类可读的 Python 前端 | `python_frontend.py`、`phase_scheduling.py` |
| 组合 IR 与执行绑定 | `composition_ir.py`、`composition_block.py`、`unified_strategy.py` |
| 因果观察、决策与规则 | `causal.py`、`composition_observation.py`、`composition_policy.py`、`unified_policy.py` |
| 模型适配与持续调用 | `service_program.py`、`schedule.py`、`m2_schedule.py`、`m2_moe.py` |
| 固定动作 IR 与因果入口 | `strategy_ir.py`、`strategy_compiler.py`、`policy_ir.py`、`causal_strategy.py` |
| 小图/有限调用链搜索 | `mapper.py`、`schedule_search.py`、`m2_*_search.py` |
| GGML 图导出、可视化工具 | `ggml/`、`tools/` |

完整分层与依赖说明见 [代码地图](../docs/CODE_LAYOUT.md)。`spec.py` 和 `engine.py` 只是旧导入路径的兼容转出，实际内核位于 modeling，不能再往这两个文件新增第二份实现。

## 其他受支持入口

机器可读的组合策略：

```powershell
python -m tensor_mapping.unified_strategy --help
```

格式与可运行配置见 [组合示例](examples/composition/README.md)、[资源响应示例](examples/reconfiguration/README.md)。论文对照入口见 [论文机制表达](docs/PAPER_IR_QUICK_GUIDE.md)；它们表达部分核心机制，不等于完整复现论文系统。

小图评价与精确搜索：

```powershell
python -m tensor_mapping search --scenario mapping/examples/chain-cap160.json --out mapping/outputs/small-search
python -m tensor_mapping model --scenario mapping/examples/chain-cap160.json --mapping mapping/outputs/small-search/mapping.json --out mapping/outputs/small-replay
```

搜索的最优性仅适用于给定图、资源、成本、动作空间和搜索预算，不是通用 LLM 的最优策略声明。原合同见 [DESIGN.md](docs/DESIGN.md) 和 [ACCEPTANCE.md](docs/ACCEPTANCE.md)。

查看器：

```powershell
python mapping/tools/viewer.py
python mapping/tools/viewer.py --self-check
```

也可以双击 `viewer.cmd`。查看器保留原有小图/窗口模型入口，不能据此声称已可视化所有新 Python 策略。可选 Node.js 用于前端烟测。

## 验证与历史

从 `Simulator/` 根目录运行：

```powershell
python -B -m unittest discover -s mapping/tests -t mapping
```

`-t mapping` 用于正确加载带相对导入的测试包。固定计划输入位于 `tests/fixtures/fixed_plans/`，测试不再读取历史验收目录。逻辑功能测试与性能准确度验收分开。

当前技术文档集中在 `docs/`。旧规划、历史 README、research/validation 副本和过期输出已清理；可选本机清理记录位于 `docs/maintenance/20261017-cleanup/REPORT.md`，不随仓库分发。当前回归请运行上述测试命令，不依赖历史日志。
