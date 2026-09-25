# mapping：GGML 计算图的执行映射与搜索

状态：M0 核心（数据契约、状态转移、固定 mapping 评价、精确搜索）与 GGML 导出器已实现并有测试；`cli.py` / `__main__.py` 与 stats/events/states/mapping 产物落盘未做，**M0 尚未整体完成**（`ACCEPTANCE.md` §7）。日期：2026-09-25。

目标：给定 GGML 小计算图、硬件资源、固定动作成本和受限动作空间，检查一种执行映射是否合法，计算其执行时间，并搜索最短的合法映射。

本目录名称由用户确定为 `mapping`，替代此前规划中的 `scheduling`。旧 `../modeling/` 保留原位作为层窗口模型和实验资产。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：首版范围、数据契约、状态转移、搜索和实现边界。
2. [ACCEPTANCE.md](ACCEPTANCE.md)：带明确数字和推导的验收例子、错误用例和交付要求。
3. 有需要时阅读 [资产梳理](../ASSET_MAP_AND_START_PLAN.md) 与 [中期规划](../PROJECT_PLAN_2026-09-25.md)。首版具体语义以本目录为准。

## 给 Claude 的实施任务

请按 DESIGN.md 和 ACCEPTANCE.md 实现 M0：GGML 小图导出、固定成本状态模型、固定 mapping 回放、小图精确搜索及验收测试。

- 先完成链式图，再完成残差和分叉图；按 DESIGN.md 的 M0.1～M0.4 顺序推进。
- 使用 Python 实现模拟与搜索，C/C++ 调用固定版本 GGML 建图并导出 JSON。
- 先完成无需 GPU 的核心闭环。首版成本为合成值，不能标成真实硬件预测；实际 GGML 图来源与时间成本来源分别记录。
- `model` 与 `search` 共用状态转移，不各自编写一套容量或依赖规则。
- 输出可回放 mapping、事件记录、统计、求解状态和最优性说明。
- 工作代码放在本目录。GGML 路径由构建参数指定，优先利用本地版本；不修改旧模拟器、历史实验数据或现有推理引擎。
- 不把本任务扩大到完整 LLM、CPU/GPU 联合计算、MoE、Agent、CUDA 预取执行器、复杂前端或 Timeloop 集成。
- 跑完本目录验收；若构建环境缺失，明确报告阻塞项，不能用手写图冒充已运行的 GGML 导出。
- 最终汇报实现内容、实际运行命令、通过/失败/跳过项、已知限制。搜索停止于预算上限时，不得宣称无解或已证明最优。

这些是首版的实现约束；后续阶段由用户另行安排。

## 与 Timeloop 的关系

参考本地 Timeloop 的 workload、architecture、mapping、mapspace 和 mapper 分工：固定方案由 evaluator 评价，搜索器负责产生和比较方案。这里的 mapping 表示整个小图中数据加载、计算、驻留和释放的执行安排；M0 不搜索循环分块、算子内部数据流或空间阵列映射。

参考入口：[Timeloop 分析](../reference-analysis/TIMELOOP.md)、[本地 mapper 文档](../reference/timeloop/doc/mapper.md)。本项目不依赖编译或运行 Timeloop。

## 计划结构

现状（✓ 已实现，· 未做）：

```text
mapping/
├── README.md
├── DESIGN.md
├── ACCEPTANCE.md
├── pyproject.toml
├── ggml/                       ✓ 建图、导出程序与独立构建入口（见 ggml/README.md）
│   ├── CMakeLists.txt          ✓ 以 GGML_SOURCE_DIR 参数接本地 ggml，不写死路径
│   └── src/
│       ├── graphs.h/.cpp       ✓ 四张正图 + 四个只用于验证拒绝路径的负图
│       └── export_workload.cpp ✓ 闭包遍历、布局校验、JSON 落盘、CLI
├── tensor_mapping/
│   ├── __init__.py             ✓
│   ├── __main__.py             ·
│   ├── spec.py                 ✓ 图、资源、成本、mapping 与输入校验
│   ├── engine.py               ✓ 状态、合法动作、转移、固定 mapping 评价
│   ├── mapper.py               ✓ uniform-cost 精确搜索与状态去重
│   └── cli.py                  · model / search
├── examples/                   ✓ 小图、scenario、固定 mapping
│   └── ggml/                   ✓ 真实导出产物与其 scenario（与同名 fixture 逐字段等价）
└── tests/                      ✓ 语义、搜索、导入与导出验证
```

模块可随实现小幅调整，但保持上述职责。核心 Python 包不导入 llama.cpp 或 GPU 库；GGML 导出是独立前端，缺构建环境时导出器相关测试 skip 并注明未验证。

## 后续路线

M0 小图语义与精确搜索 → M1 小图实机校准与异构资源 → M2 Transformer block / dense → M3 可扩展搜索与论文验证 → MoE → 多请求/Agent。

完成 M0 不等于已实现真实 GPU 调度执行器，也不等于已验证硬件性能预测。
