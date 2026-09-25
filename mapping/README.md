# mapping：GGML 计算图的执行映射与搜索

状态：M0 全部组件（数据契约、状态转移、固定 mapping 评价、精确搜索、GGML 导出器、`model`/`search` 两个入口与四份产物）已实现并有测试，共 181 项。测试全绿**不等于** M0 整体验收：`DESIGN.md` §9 的集成验证需要 GGML 构建环境，`ACCEPTANCE.md` §7 的六条要逐条有交代。日期：2026-09-25。

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

现状（全部 ✓ 已实现）：

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
│   ├── __main__.py             ✓ python -m tensor_mapping
│   ├── spec.py                 ✓ 图、资源、成本、mapping 与输入校验
│   ├── engine.py               ✓ 状态、合法动作、转移、固定 mapping 评价
│   ├── mapper.py               ✓ uniform-cost 精确搜索与状态去重
│   ├── artifacts.py            ✓ 结果对象 → stats/events/states/mapping 四份产物
│   └── cli.py                  ✓ model / search：参数、退出码、错误映射
├── examples/                   ✓ 小图、scenario、固定 mapping
│   └── ggml/                   ✓ 真实导出产物与其 scenario（与同名 fixture 逐字段等价）
└── tests/                      ✓ 语义、搜索、导入、导出验证与 CLI 端到端
```

模块可随实现小幅调整，但保持上述职责。核心 Python 包不导入 llama.cpp 或 GPU 库；GGML 导出是独立前端，缺构建环境时导出器相关测试 skip 并注明未验证。

## 命令行

```bash
python -m tensor_mapping model  --scenario S.json --mapping M.json --out DIR
python -m tensor_mapping search --scenario S.json --out DIR \
                                [--max-expanded-states N] [--wall-time-limit-s F]
```

`model` 回放一份固定 mapping，写出 `stats.json` / `events.json` / `states.json`；`search` 搜一份最短 mapping，另加 `mapping.json`（§8 的原话是「search 另外产出 mapping.json」）。`--out` 目录不存在就建，只覆盖本工具自己的文件名，**不删任何已有文件**。

两个 `--max-expanded-states` / `--wall-time-limit-s` 是契约外的附加开关，直接透传给 `search()`（0 表示不限）。它们存在的理由：`ACCEPTANCE.md` §6:110 要求「预算不足必须是 `unknown` 而不是 `infeasible`」，有了它们这条能在命令行上复现；产物里的求解预算也据此记**实际生效值**而非 scenario 默认值。

### 复现（从干净输出目录）

```bash
cd mapping
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping examples/chain.mapping.json \
                                --out outputs/chain-replay
python -m tensor_mapping search --scenario examples/chain-cap160.json \
                                --out outputs/chain-search
# M0.4 门槛：把 search 写出的 mapping 独立回放一遍，数字与事件应与上面一致
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping outputs/chain-search/mapping.json \
                                --out outputs/chain-replay2
# 真实 GGML 导出产物上同样走一遍（origin.kind = "ggml"）
python -m tensor_mapping search --scenario examples/ggml/chain-cap160.json \
                                --out outputs/ggml-search
```

`outputs/` 已被根 `.gitignore` 忽略（`mapping/outputs/`），产物不入库，全量可复跑重建。

### 跑测试

```bash
cd mapping && python -m unittest discover -s tests -t .
```

**必须带 `-t .`**。`DESIGN.md:260` 写的是 `python -m unittest discover -s tests`，但本目录的 `tests/` 是一个包（有 `__init__.py`、用相对 import），少了 `-t .` 时发现器只找到 5 个用例且全部报错。冻结契约不改，所以把差异记在这里。

### 退出码（契约未规定，本仓库自行定下）

| 码 | 含义 |
|---|---|
| 0 | 拿到可用结果：model `valid`；search `optimal` 或 `feasible`（feasible 也是一份可回放的 mapping） |
| 1 | 输入层面被拒：文件读不了、JSON 坏了、schema 多/少键、`unsupported` 语义、mapping 与 scenario 身份不符 |
| 2 | 跑完了但没有可用 mapping：model `invalid_mapping` / `incomplete_mapping` / `initial_capacity_exceeded`；search `infeasible` / `unknown` |

1 与 2 分开是为了让 shell 能区分「你给错了输入」和「答案就是做不到」——这正是 §8:270 把 `initial_capacity_exceeded` 放在裁决一侧、而不是和 JSON/schema 错误并列的原因。

**不变量**：退出 1 时**一个文件都不写**（连 `stats.json` 都没有），因为没发生过运行，一份只有身份键、没有指纹的半截 stats.json 比空目录更糟——下一个人分不出它和一次完成的运行。退出 0 和 2 都写出同一组文件名，下游不必区分「文件缺失」和「确实没有动作」。`events.json` / `states.json` 为空时带一句 `note` 说明为什么为空。

一句话摘要走 stdout（退出 2 也算「有结果」，所以照打）；诊断（错误码、动作索引、模拟时间、读不了的原因）走 stderr。

### 产物判读

四份文件都带 `schema_version` 与 scenario id + 两个指纹，所以任何一份单独拿出来都还知道自己是谁的产物。

- `stats.json`：`status` / `reason` / `mode` / 两个指纹 / `graph_origin`（图来源）/ `cost_source`（成本来源，与图来源**分开**，`DESIGN.md:271` 要求两者分别记录）/ `makespan_ns` / `peak_vram_bytes` / `h2d_bytes` / `action_count` / `declared_action_count` / `limits` / `assumptions`，失败时另有 `error`，search 另有 `search` 块。
- `action_count` 是**实际施加**的动作数，`declared_action_count` 是 mapping 里**声明**的条数：`_failure` 把前者设成 `len(events)`，所以「文件里写了 10 条、第 3 条就炸了」要靠两个字段的差看出来（`ACCEPTANCE.md:41` 正是这种情形：3 / 10）。
- `error` 块在**裁决不是 `valid` 时**出现，而不是在 `error_code` 非空时——`incomplete_mapping` 的 `error_code` 是 `None`，但它的 `action_index` / `t_ns` 恰恰是全部诊断。
- `limits` 是限制（数字，从 scenario 推导，改 scenario 就跟着变，不会过期）；`assumptions` 是假设（散文，每句只在它断言的事实成立时才出现）。两者在 §8 里合称「限制与假设」。
- **`mapping.json` 里没有状态字段**——`load_mapping` 的 allow-list 会拒绝多余键。判断一份 mapping 是否已证最优，必须读 `stats.json` 的 `search.optimality_proven`。`search` 写出的 `mapping.json` 会把这句话写进自己的 `comment`（最优/未证最优 + 终止原因 + 生效预算），因为这份文件最容易被单独复制出去被别人回放。

### 命令行上的已知限制

- **`search.wall_time_s` 在本机被时钟粒度量化**：Windows 上 `time.monotonic()` 实测是 `GetTickCount64()`，分辨率 **15.625 ms**，连续 2000 次调用返回同一个值。所以小图搜索的 `wall_time_s` 常常正好是 `0.0`（不是搜索瞬时的意思），`--wall-time-limit-s` 小于一个 tick 时**永远不会触发**。这不影响任何 M0 验收数字（那些全是模拟时间 `_ns`），但用时间预算时要心里有数。要修得动 `mapper.WallClock` 的时钟源（`perf_counter`），属核心改动，留待后续。
- **`examples/chain.mapping.json` 与 `examples/ggml/chain.mapping.json` 都不带指纹**，所以它们只靠 `scenario_id` 做身份检查，而 fixture 与 ggml 两个 scenario 的 id 恰好相同（`chain-cap160`），互相回放不会被拦。这不影响正确性——两份 workload 按设计逐字段等价，回放数字本来就一样——但说明**手写 mapping 没有跨图保护**；`search` 产出的 mapping 带两个指纹，互相回放会被 `identity_mismatch` 拒绝（退出 1）。

## 后续路线

M0 小图语义与精确搜索 → M1 小图实机校准与异构资源 → M2 Transformer block / dense → M3 可扩展搜索与论文验证 → MoE → 多请求/Agent。

完成 M0 不等于已实现真实 GPU 调度执行器，也不等于已验证硬件性能预测。
