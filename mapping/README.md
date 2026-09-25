# mapping：GGML 计算图的执行映射与搜索

状态：M0 全部组件（数据契约、状态转移、固定 mapping 评价、精确搜索、GGML 导出器、`model`/`search` 两个入口与四份产物）已实现并有测试，共 202 项。`ACCEPTANCE.md` §7 的六条逐条对照见 [M0_VERIFICATION.md](M0_VERIFICATION.md)（该报告记录验收当时的 181 项；此后新增的 21 项见下）。日期：2026-09-25。

其后一轮「对照轮」把旧的窗口规则接进了 `modeling` 的张量内核（[ALIGNMENT_IMPLEMENTATION.md](ALIGNMENT_IMPLEMENTATION.md)）：`modeling` 侧新增 `llm_infer_model.tensor` 子包（`spec` / `engine` / `layer_costs`），本目录新增 `tensor_mapping/policies.py`（窗口策略）与 `tools/demo_window_vs_search.py`（三份结果并排对照）。新加的 21 项测试落在 `tests/test_policies.py`，`modeling/tests/test_tensor_layer_costs.py` 另有 19 项。

测试全绿**不等于** M0 整体验收，所以验收按 `ACCEPTANCE.md` §7.6 要求分三类留了证据。`DESIGN.md` §9:285 的集成验证「编译 → 导出 → 模拟/搜索」已从零走通：编译腿在全新空目录 configure + build（32.7 s + 37.6 s），导出腿用这个新二进制重导出四张图，产物与已入库文件**逐字节相同**，`tests.test_ggml` 13 项零 skip。剩下的环境限制只有一条：`--wall-time-limit-s` 在本机因时钟粒度（15.625 ms）无法端到端验证，详见验证报告 3.2。

目标：给定 GGML 小计算图、硬件资源、固定动作成本和受限动作空间，检查一种执行映射是否合法，计算其执行时间，并搜索最短的合法映射。

本目录名称由用户确定为 `mapping`，替代此前规划中的 `scheduling`。旧 `../modeling/` 保留原位作为层窗口模型和实验资产。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：首版范围、数据契约、状态转移、搜索和实现边界。
2. [ACCEPTANCE.md](ACCEPTANCE.md)：带明确数字和推导的验收例子、错误用例和交付要求。
3. [ALIGNMENT_IMPLEMENTATION.md](ALIGNMENT_IMPLEMENTATION.md)：**当前的模块分工与安装方式**，以及旧窗口规则接进张量内核的范围。它与本 README 一起优先于更早的架构简报；`DESIGN.md` / `ACCEPTANCE.md` 里的 M0 动作语义和验收数字继续有效。
4. 有需要时阅读 [资产梳理](../ASSET_MAP_AND_START_PLAN.md) 与 [中期规划](../PROJECT_PLAN_2026-09-25.md)——那是立项期的计划文本，模块分工以第 3 条为准，语义以本目录为准。

## 给 Claude 的实施任务

> 下面这段是 M0 立项时的任务书，**M0 已完成**，保留它是为了让后来的人看得到当初定下的边界。当前该做什么，看 [ALIGNMENT_IMPLEMENTATION.md](ALIGNMENT_IMPLEMENTATION.md)。

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
├── M0_VERIFICATION.md          ✓ §7.6 要求的验证报告（自动测试 / 人工核对 / 未运行项）
├── pyproject.toml
├── ggml/                       ✓ 建图、导出程序与独立构建入口（见 ggml/README.md）
│   ├── CMakeLists.txt          ✓ 以 GGML_SOURCE_DIR 参数接本地 ggml，不写死路径
│   └── src/
│       ├── graphs.h/.cpp       ✓ 四张正图 + 四个只用于验证拒绝路径的负图
│       └── export_workload.cpp ✓ 闭包遍历、布局校验、JSON 落盘、CLI
├── tensor_mapping/
│   ├── __init__.py             ✓
│   ├── __main__.py             ✓ python -m tensor_mapping
│   ├── spec.py                 ⟳ 兼容层：只是转出 llm_infer_model.tensor.spec 的名字
│   ├── engine.py               ⟳ 兼容层：只是转出 llm_infer_model.tensor.engine 的名字
│   ├── mapper.py               ✓ uniform-cost 精确搜索与状态去重
│   ├── artifacts.py            ✓ 结果对象 → stats/events/states/mapping 四份产物
│   ├── cli.py                  ✓ model / search：参数、退出码、错误映射
│   └── policies.py             ✓ 旧窗口 K 规则 → 新内核的动作序列（对照轮新增）
├── examples/                   ✓ 小图、scenario、固定 mapping
│   ├── three_layer_chain*      ✓ 三链对照场景（窗口策略与 demo 的输入）
│   └── ggml/                   ✓ 真实导出产物与其 scenario（与同名 fixture 逐字段等价）
├── tools/
│   ├── plot_results.py         读产物出瀑布图与显存曲线（**包外、不计入 M0 验收**）
│   └── demo_window_vs_search.py 旧模型 / 新窗口策略 / 搜索三份结果并排对照（包外）
└── tests/                      ✓ 语义、搜索、导入、导出验证、CLI 端到端与窗口策略
```

模块可随实现小幅调整，但保持上述职责。张量级内核（`spec` / `engine` / `layer_costs`）在对照轮迁到了 `modeling/llm_infer_model/tensor/`，本目录的 `spec.py` / `engine.py` 只剩兼容层——**它们只是转出名字，不保留第二份实现**（两份同名数据类会让 `isinstance` 与 `dataclasses.replace` 悄悄失效）。`tensor_mapping` 自己的模块直接从 `llm_infer_model.tensor` 导入，所以兼容层没有调用方之后可以整个删掉。

核心 Python 包不导入 llama.cpp 或 GPU 库；GGML 导出是独立前端，缺构建环境时导出器相关测试 skip 并注明未验证。

## 安装（开发）

两个包**一起**从本地安装，在 Simulator 根目录执行一次：

```powershell
python -m pip install -e ./modeling -e ./mapping
```

**两个 `-e` 必须在同一次 pip 调用里给出**：pip 先汇总要求再解析，只装 `./mapping` 会去 PyPI 找 `llm-infer-model` 然后失败。`llm-infer-model` 是本仓库的 `modeling/` 包，不是第三方依赖——它持有张量模拟内核 `llm_infer_model.tensor`，本目录的搜索与窗口策略导入的是同一份规则，所以两边只有一套时间推进和显存记账。

装完就可以直接跑 `python -m tensor_mapping` 和 `tests/`，不需要设 `PYTHONPATH`。本目录**不再支持「零安装即可跑测试」**：那是两个目录各自独立时期的说法，现在 `tensor_mapping/__init__.py` 直接导入 `llm_infer_model.tensor`，必须先装。只跑旧模型时的单包装法保留在 `modeling/README.md` 的「安装」一节。

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
cd mapping && python -m unittest discover -s tests -t .    # 202 项
```

这里的 `python` 指装好两个包的开发环境（见上文「安装（开发）」）；本仓库自用的那个在 `modeling/.venv`。没装就只会跑到 6 个 import 错误，报 `No module named 'llm_infer_model'`。

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

### 看结果：瀑布图与显存曲线

```bash
cd mapping
python tools/plot_results.py outputs/chain-search               # 文本视图（无依赖）
python tools/plot_results.py outputs/chain-search --png         # 另出两张 PNG
python tools/plot_results.py outputs/chain-search --png --out fig --dpi 160
```

读一个结果目录里的 `stats.json` / `events.json` / `states.json`，给出两样东西：

- **瀑布图**：按资源泳道排的甘特图——什么时候搬运、什么时候计算、**是否重叠**。`events.json` 的 `resource` 决定泳道，`start_ns`/`end_ns` 决定区间。文本版还会直接报出两条资源道的重叠毫秒数（`零重叠` 说明容量卡死、装不下预取）。
- **显存曲线**：`states.json` 的 `used_vram_bytes` 对时间画阶梯，容量上限画虚线、超出的区域涂红。文本版另附**逐状态**的 `action_index / t_ns / used_vram / Δ` 表，能直接看出哪一步分配、哪一步释放。
- **一致性检查**：顺手核 7 条产物自洽性（`peak ≤ cap`、每个状态不超容量、`t_ns` 单调、`action_index` 连续、`len(states) == len(events)+1`、最晚事件不超 makespan），任一失败**退出码 1**。

**默认只打印文本**，因为文本可 diff、可入库、可贴进验收报告，而 PNG 都不能。PNG 是 `--png` 按需加的。

**`tools/` 在 `tensor_mapping/` 包外，也不被 `tests/` 导入**——`pyproject.toml` 的 `dependencies` 只有本仓库的 `llm-infer-model`、`dev` 是空列表，两边都**不引入第三方运行库**（这正是「文本视图只用标准库」能成立的原因，也是 `modeling/llm_infer_model/tensor` 必须只用标准库的原因），所以 matplotlib 只在 `--png` 分支里按需导入：没有 matplotlib 时文本视图照常工作，`--png` 报一句提示并退出 3。本脚本**只读产物**，不写任何结果文件、不碰 `DESIGN.md` §8 的任何语义。

### 看对照：旧窗口模型 / 新窗口策略 / 搜索

```bash
cd mapping
python tools/demo_window_vs_search.py                       # 五段对照，全对才退出 0
python tools/demo_window_vs_search.py --windows 1 2 3       # 换个窗口集合
```

把三份结果并排打出来：左边是 `llm_infer_model.simulator.simulate_decode`（v0.8 的窗口规则），右边是 `tensor_mapping.policies.window_mapping` + `llm_infer_model.tensor.evaluate_mapping`（新内核），最后一段是 `tensor_mapping.mapper.search` 的通用解。五段依次是：成本桥接（示例里写死的成本必须等于经旧层模型闭式求值再换算出来的值）、窗口 K 逐行对照、K 小于初始驻留份数时由策略自己拒绝、哪些图形状窗口策略读不了、搜索结果与它的独立回放。

任何一项对不上就以非零码退出，所以它也可以直接当验收命令跑。和 `plot_results.py` 一样**只读**、包外、不被 `tests/` 导入。四处对照口径（成本走公式不走实测覆盖、权重峰值要单独数、K = 层数对应旧模型的「全部常驻」分支、拒绝类型 `WindowPolicyFailed` 与 `Unsupported` 都不是 `infeasible`）写在脚本自己的 docstring 里。

### 窗口策略（`tensor_mapping/policies.py`）

旧 v0.8 的「窗口 K」规则现在有了一条接进新内核的路径：

```python
from tensor_mapping.policies import window_mapping
from llm_infer_model.tensor import evaluate_mapping

actions = window_mapping(scenario, window_size=2)   # 只决定「先做哪一件」
result = evaluate_mapping(scenario, actions)        # 合法性与时间推进一律问内核
```

策略**不解释结果**：makespan、峰值显存、搬运量全部由 `evaluate_mapping` 产生，所以本模块没有第二套容量或依赖规则。它只支持**显式线性链**（每环恰好一份独有权重 + 一个激活输入），残差、分叉、共享权重明确拒绝（`Unsupported`）。拒绝分三类，调用方必须区别对待：

| 异常 | 含义 | 是不是「场景无解」 |
|---|---|---|
| `Unsupported` | 图的形状不在窗口策略范围内 | 不是；通用搜索照常支持 |
| `InvalidInput` | `window_size` 本身不合法 | 不是 |
| `WindowPolicyFailed` | 图形状没问题，但这条策略走不下去（含容量不够） | **不是**，请改用 `search` 拿结论 |

最后一条是刻意留的余地：策略走不下去只说明**这条策略**被自己的限制挡住了，`mapper.search` 可能仍然找得到可行计划，所以调用方不得把它当成 `infeasible`。

本轮只做「窗口 K」这一条策略，**未迁移**的仍是旧模型独有的那些服务：KV 分层存储、SSM 状态往返、每 token 控制开销、host staging、静态卸载、CPU 数学、D2H。成本桥接 `llm_infer_model.tensor.layer_costs` 遇到这些非零值会**逐项点名并拒绝**，而不是静默丢弃——静默丢弃会让时延偏小且不留痕迹。

### 命令行上的已知限制

- **`search.wall_time_s` 在本机被时钟粒度量化**：Windows 上 `time.monotonic()` 实测是 `GetTickCount64()`，分辨率 **15.625 ms**，连续 2000 次调用返回同一个值。所以小图搜索的 `wall_time_s` 常常正好是 `0.0`（不是搜索瞬时的意思），`--wall-time-limit-s` 小于一个 tick 时**永远不会触发**。这不影响任何 M0 验收数字（那些全是模拟时间 `_ns`），但用时间预算时要心里有数。要修得动 `mapper.WallClock` 的时钟源（`perf_counter`），属核心改动，留待后续。
- **`examples/chain.mapping.json` 与 `examples/ggml/chain.mapping.json` 都不带指纹**，所以它们只靠 `scenario_id` 做身份检查，而 fixture 与 ggml 两个 scenario 的 id 恰好相同（`chain-cap160`），互相回放不会被拦。这不影响正确性——两份 workload 按设计逐字段等价，回放数字本来就一样——但说明**手写 mapping 没有跨图保护**；`search` 产出的 mapping 带两个指纹，互相回放会被 `identity_mismatch` 拒绝（退出 1）。

## 后续路线

M0 小图语义与精确搜索 → M1 小图实机校准与异构资源 → M2 Transformer block / dense → M3 可扩展搜索与论文验证 → MoE → 多请求/Agent。

完成 M0 不等于已实现真实 GPU 调度执行器，也不等于已验证硬件性能预测。

对照轮（把旧窗口规则接进张量内核）的范围**到此为止**：不开发 GUI，不扩展 MoE、多请求、多 GPU、CPU 计算、D2H 或完整 dense，不改 GGML 算子支持范围，不重做指纹/状态码/产物格式。已有的绘图脚本与 demo 继续可用即可。下一条策略（KV 分层存储、每 token 控制开销等）各自进来时，都应像窗口 K 这样只决定动作顺序、把合法性与时间推进留给内核。
