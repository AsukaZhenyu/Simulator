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
├── tools/                      **包外**（刻意没有 __init__.py：不可导入 = 不会被包 import）
│   ├── plot_results.py         读产物出瀑布图与显存曲线（不计入 M0 验收）
│   ├── demo_window_vs_search.py 旧模型 / 新窗口策略 / 搜索三份结果并排对照
│   ├── viewer.py               本地查看器：起服务、快照导出、--self-check
│   ├── viewer_payload.py       查看器的纯数据装配（Scenario → 载荷，含走查与时长来源）
│   ├── viewer_smoke.mjs        前端 JS 的落地烟测：DOM 垫片 + 扫 NaN（**需要 node，可选**）
│   └── viewer_static/          index.html + app.js + style.css（手写 SVG，无框架无 CDN）
├── viewer.cmd                  **双击即用**：找 venv、用默认 scenario 与默认来源、开浏览器
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

### 看图与操作：查看器

最省事的方式：**双击 `mapping/viewer.cmd`**。它自己找到 `modeling/.venv` 的解释器、用默认的 scenario 与 config 起服务、并打开浏览器——一个参数都不用给。（这个文件刻意写成纯 ASCII + CRLF：`cmd.exe` 按 OEM 码页读 `.cmd`，UTF-8 中文注释会碎成标点混进命令行。自检专门钉了这两条，见下。CRLF 也钉进了根目录 `.gitattributes`——那里的兜底规则是 `* text=auto eol=lf`，没有 `*.cmd text eol=crlf` 这一行的话，**克隆下来就是 LF**，而自检是在本机跑的、不会有任何症状。）

要从命令行起也可以，用哪个 python 都不挑——解释器里没有那两个本地包时，`viewer.py` 会自己换到 venv 重跑一遍并打一行说明：

```bash
cd mapping
python tools/viewer.py                     # 默认 examples/chain-cap160.json，时长来源默认 modeling 推导
python tools/viewer.py --scenario examples/three_layer_chain-cap1024.json
#   → 终端打印实际地址（默认 --port 0，由内核挑一个空闲端口）
#     页面上有「导出快照」→ viewer-snapshot.html
python tools/viewer.py --scenario examples/chain-cap160.json \
                       --config ../modeling/configs/toy_dense_decode.json
#   ↑ 给了 --config 才会看到门 1 那一类拒绝（随包 8 个 config 全过不了桥）。
#     不给就是「时长来源 = modeling 推导」那条默认路，这是两件不同的事，别混。
python tools/viewer.py --export out.html   # 不起服务，直接导出快照
python tools/viewer.py --self-check        # 无头自检，退出 0/1（装了 node 会多跑一组 JS 烟测）
```

起一个**只绑 `127.0.0.1`** 的本地服务（`--port 0` 让内核挑端口，`--no-browser` 不弹窗口）。五块屏**全部常驻在同一页上**，栏上那排是**跳转条不是标签页**——点一下只是 `scrollIntoView`，`aria-current` 跟着滚动位置走。屏序是 **搜索树 → 计算图 → 时间线 → 时长来源 → 参数**：先看策略长什么样，再看它把算子与张量排成什么形状，再看它花掉多少时间，最后才问那些时长是哪来的。

**顶栏上能直接换计算图**，不用回命令行重起：名字右边那一格列出**当前这份 scenario 所在目录**里的其它 scenario（`examples/` 下是 5 份），选一下就换过去。几件刻意的事：**旋钮不动**——VRAM、预算、桥的开关都跟着走，「同一套设置换一张图看看」正是它存在的理由；**名单只列同一目录下的兄弟文件**，`*.workload.json` / `*.mapping.json` 是同一个 scenario 的另两种形态、不是另一张图，排除掉，而载不进来的那份**照样列着**（它就在那个目录里，从屏上抹掉它才是骗人，点下去会得到一次带原因的拒绝）；换图前后那一格会**改口叫「换图前后」并写明两张图不可比大小**，而不是继续挂着「重跑前后·同一套代码路径」；导出的快照里它是**禁用的**并写着为什么。换个名字（名字走白名单，不拼路径——这个键是从 HTTP 进来的）或那份载不进来，都是一次 422 + 横幅，且**这一侧退回原来那份**：屏上是哪张图，选择器就写着哪张。

- **搜索树**：每个创建过的状态一个节点，解路径高亮，被支配剪掉的边点线。**节点可点**——点任一个，下面的计算图与时间线一起搬到那条路径上（走带随之切到单步）。
- **计算图**：**横轴是拓扑层、纵轴是先后**。算子按拓扑层分列，张量**贴在消费它的算子左边一列**——于是同一个算子的父张量上下叠着（不是前后排开），同一列里先加载的在左上；算子的行取它父张量里最靠下那一行，**它产出的张量与它同一行**，所以依赖边只会朝右下走（`residual` 那条残差边 `x → add` 是唯一的跨列例外：`x` 被 `c1` 与 `add` 共用，只能贴在最早那个消费者左边）。紧凑是从这几条里掉出来的，不是另调的参数：五份用例都落到 2~3 行（`residual` 从 602px 高降到 146px），方块占画布 28%~33%。**张量用琥珀、算子用蓝**，与时间线那两条泳道同色；状态在族内用底色深浅与边线虚实表达，不借别的色系。排布基准取**解路径**而非当前游标——按游标实时重排会让拖动走带时整张图不停跳位。
- **时间线**：甘特 + 显存阶梯 + 状态时间线。点某个节点后，下面还会列出**从该状态出发被引擎拒掉的候选与理由**——搜索树上的节点全是走得到的，不可行只表现为「从某状态出发，某些候选被拒」。状态时间线每一行右边那句说明（**这一步在干什么 · 什么时候 · 用了多少**）**紧跟在它自己那根条后面**，右列宽度按这一份里最长的那条现算——同一时刻的几个状态因此自动对齐成一列，字的位置本身就是时刻。三张图上下之间多留了 10px（`#pane-timeline .canvas-wrap + .canvas-wrap`）：各自的标注都贴着自家画布的边（甘特图底下一行时刻、显存曲线上头「容量 xxx」），只靠 `.pane` 的间距会挤成一片。
- **时长来源**：这条时间线上的每一根柱子是**怎么算出来的**，以及这一份到底用的是推导值还是声明值（见下）。
- **参数**：改旋钮重跑搜索并与改动前逐项对比。

**「联动」落在哪里**：时间线的宽度全部由 `costs_from_model_config` 推出来的那份 `Costs` 决定，而不是 `examples/*.json` 里手写的 `synthetic_fixed` 常数。默认就走这条——`--config` 现在默认 `None`，此时**现造**一份「除硬件外一切服务都关着」的 `ModelConfig`（`_effective_config()`；写成 `replace(config, …)` 会在 `None` 上当场 `TypeError`）。于是默认那份 `chain` 的推导值恰好与声明值**逐字节相同**（所以冻结的 8000000/160/128 一个都没动），而 `matvec` 不一样：推导 2.5 ms ≠ 声明 3 ms，makespan 因此从 5 ms 变成 4.5 ms——屏上要**用红字说清这一点**，因为 `M0_VERIFICATION.md:138` 记的是 5 ms。（`examples/*.json` 一个字节都没改，换的只是内存里那一份。）

时长来源那一屏的拒绝分**四类，且措辞上刻意不给第四类编号**：前三类是 `costs_from_model_config` 那道 guard 的三道门（`Unsupported` 未迁移 / `InvalidInput` 单位分辨率与图形状 / `ValueError` 参数不变式），第四类是「配置根本没读进来」——它发生在三道门**之前**，叫「门 0」会被读成比门 1 更早的一道 guard 判定，而它不是。相应地，`--config` 指的文件读不出来时**不退出**：scenario 一侧是好的，图 / 时间线 / 搜索树照常能看，失败原因原样带上时长来源那一屏（只 `print` 的话，`--export` 和双击启动器都把 stdout 丢了，用户看到的是「这一屏是空的」而原因滚过去了）。

来源面板写的是**溯源**，不是一行字：`kind`（`model` / `declared`）、`swapped`、`label`、`message`、`remedy`，以及 `reason_class`。退回声明值有**两条**路，理由完全不同、补救方向也相反，所以分开写：`edited`（你在参数页手改过某个 `duration_ns`——那些改动写在 `scenario.costs` 上，推导出的 `Costs` 会把它整份换掉，不退回的话「改了参数」和「改了没反应」在屏上长得一模一样）与结构性的推不出来（门 1/2/3、或配置没载入）。屏上还有两处可验：**逐项对照表**每行挂着算式（`延迟 + 字节 ÷ 带宽`，所以 `h2d` 行特地带上了**字节**——`matvec` 的权重是 48 字节，`demo_window_vs_search.py:81` 写死的那个 64 只对另外四个示例成立），以及**三个硬件旋钮**（`gpu_effective_flops` / `h2d_bandwidth_bytes_per_s` / `h2d_latency_s`），改一下推导出的 ns 与 makespan 一起变——那正是「联动非空转」的操作性证据。硬件旋钮走的是既有的 `POST /api/params` 通路，**不新增第三个参数名**。

游标有两种模式，**视觉上刻意区分**，混同就是这类工具开始骗人的起点：**回放**走烤好的计划（只读）；**单步**走一条自己点出来的分支路径——游标是一条 id 数组而不是一个整数，因为整数表达不了「已经离开计划分叉出去」。切到单步时甘特与显存**整体替换**成该路径的，并挂一条「正在探索：这不是搜索出的计划」的横幅。

同一套前端既是活页面也是导出物：`app.js` 末尾只有一处分叉（`window.__BAKED__` 有就用它，没有才 `fetch('api/payload')`），所以「导出当前视图」就是读三个文件拼一次，把载荷和 CSS/JS 内联进 `index.html`。**导出的快照双击即可离线打开**，没有 CDN、没有 `http://` 外链，此时参数页与顶栏的换图选择器**一起禁用**并说明原因（没有 Python 侧可调）。

一处刻意的取舍：两条甘特**不画在同一条时间轴上**。mapping 的 `makespan_ns` 是极小图上单 token 的排程，modeling 的 `makespan_seconds` 是 v0.8 的逐层窗口模型——它们不只是单位不同，是两个宇宙；画一条轴等于断言一个只在**逐算子**层面建立过的等价。唯一的定量联系在时长来源那一屏的表格里。

**`--self-check` 是这个工具唯一的测试**（`tools/` 不进 `tests/`），**496 项分十组**：走查与 `mapper.search()` 逐字段对账（11 个用例，覆盖五种终局）、成本桥三道门（外加「配置根本没载入」这第四类拒绝）、**时长来源**（推导值 == 声明值逐项相等、`fork`/`residual` 回落并带原因、改硬件参数后 ns 真的变了、改过 `compute_ns` 时 `reason_class == "edited"`）、载荷自洽（含独立重算 `used_vram_bytes` 这本内存账、以及换图名单的那几条：名单 == 目录里的非产物 json、每一份都真载得进来、当前那份一定在里面、不在时补在最前）、快照完整性（含五块屏的顺序与「跳转条每一项都落得下去」）、**JS 落地烟测**、房子规矩、双击启动器的三条硬约束、参数提交的合并与回滚、真实 HTTP 路由（含**换图那一下**：换过去 / 换回来 / 换个不存在的名字 / 名字写成路径的样子，四种都要求 200/422 而不是白屏，且**失败之后不许停在半路**）。任何一条 ✗ 退出 1。

**JS 落地烟测**（`tools/viewer_smoke.mjs`）是唯一**执行**前端的组：前面每一组看的都是静态的东西（载荷的字段、`index.html` 里有几个标签），而「计算图一条边都没有」那次，载荷是对的、`index.html` 是对的、`render()` 也没抛异常——错的是 JS 算出来的每个坐标都是 `NaN`，**SVG 对无效属性又是静默的**（`<rect x="NaN">` 的 `x` 被忽略、`<path d="…NaN…">` 整条元素被丢弃）。所以这组把 `app.js` 放进一个 ~300 行的 DOM 垫片里跑，然后：五个渲染器**都真的跑了**（每块屏都要有「父节点不是 pane 自己、也不在 `.pane-head` 里」的元素——空 `<svg>` 与空 `#params-body` 一个都过不了，那正是 v1「五块并排、其中四块空着」的形状）；扫**整棵文档树的每个属性与每段文字**，出现 `NaN` 就红；计数**卡相等**而不是大于零（边数 == `graph.links.length`、命中圈数 == `explore.nodes.length`、甘特柱数 == 有 resource 的事件数）；再点一次树上的终点、点一次跳转条，确认交互真接上了。两条是最能说明「计数相等不够」的：**计算图的方块互不遮盖**（坐标一律**从渲染出来的属性读**，不按布局公式重算——重算出来的是「我以为画在哪」）与**一个渲染器一个 `try`**（八个挤在一个 `try` 里时，第一个抛出去会把后面还没跑的全带走）。还有几条几何网，都是被真 bug 逼出来的、都**不针对某个用例**：**依赖边一律指向右边**（横轴是拓扑层，父在左、子在右——把任何一条边画反都一定违反它；开局就在显存里的输入张量 `inp_embd` 曾经被丢到最右边，于是 `x → c1` 从右往左画）；**每条依赖边都不许往上画**（判据取**方块的上沿**而不是箭头端点的 y：一个方块上挂多条边时锚点要摊开，摊开幅度在一个方块高 40px 以内，而行间距 62px，所以行级违规必然 ≥ 22px、摊开抖动必然 < 20px，取上沿就不用往网里塞魔法容差。这条取代了旧的「从左上走到右下」：越靠下越是「后来」的，同行的边现在很常见，卡严格更低已经不成立）；**每个算子的父张量都贴在它左边一列**（共用的父张量显式列为例外并报出来，不许静默放行）；**算子的输出张量与它同一行**；**同一列的输入/权重按加载先后自上而下**（先后从载荷的动作表现推，只比被加载进来的输入与权重——算子的行由父张量定、激活的行由产出算子定，它们服从的是结构不是加载次序）；**方块占画布面积 ≥ 18%**（用户的原话里有「保证拓扑图适当紧凑」，「太难看了」那次正是**看见**了不紧凑：`residual` 602px 高只装 11 个方块、占 5%；阈值取得远低于现在的 28%，它是个**下限**，防止哪天又滑回按层各占一行）；**锚点互不重合**（一个结点的入边原先全钉在左边中点，`c1` 的 `W1→c1` 与 `x→c1` 打在同一个点上，`fork`/`residual` 里连起点都重合，所以锚点按条数摊开在方块边上）。边那几条几何网按**画出来的 `d`** 认领边，认领是**一次性**的、还要比对两端是否落在方块的竖直范围内——只比 x 的话，同一列不同行的两个方块左边缘相同，`W1 → c1` 会把 `x → c1` 的路径认成自己的（第一版就是这么把自己报红的）；认不回来的边另有一条网报出来，不许静默跳过。盯**说明本身**的那条网（横轴那句「最左 N 列是计划开始前就已就绪的张量，不占步」里的 N 必须等于画出来的起始列数，证伪时只把说明里的数字 +1、几何一个字节不动，5 条红全落在它上面）**随这次排布一起下掉了**：横轴已经不是执行顺序，那句话在图上不再成立。教训留着——屏幕上的**主张**必须和被说明的东西对得上；这一版两条轴的那句话由上面几条几何网**无条件**担保（「不往上画」担保纵轴、「父张量贴左边一列 + 输出同行」担保横轴），所以没有再单设一条盯说明的网。还有一条：**压住了连线的方块必须半透明**——边先画、方块后画，跨列的边会被沿途的方块盖住（实测五份图 38 条边里**只剩 1 条**——按执行顺序排 x 的那一版是 15 条；同一点最多仍只叠 1 个方块，所以 α=0.5 仍然够。改一次排布就要重量一次），这条断言沿贝塞尔采样算出**哪些方块压住了边**，再要求它们的 `fill-opacity` < 1。它断言的是**性质本身**，不是「某处写了 0.5」：改 α 不误报，把填充改回不透明才报。还有三条盯的是**时间线那三张图的文字**，都是用户看出来的毛病、计数与扫 NaN 一律抓不到：**每行字都离画布上下边够远**（基线离上边 ≥ 12px、离下边 ≥ 4px。整幅 SVG 的 `overflow` 是 `hidden`，贴边的字被**静默裁掉**——用户报的「容量 xxx 被第一张图覆盖了」就是它：容量线画在绘图区上沿（`peak` 取的就是容量上限），标注只能放在线上方，旧基线落在 y=9，而 14px 的中文从基线往上要占约 12.3px，「容量」两个字的上半截被切了，看着就像被上面那张图压住）；**状态时间线每行的说明都紧挨着它自己那根条**（用户的原话是「把字放到阶段的傍边」——旧写法右对齐钉死在 x=892，实测第一条离它要说明的那个阶段 **813px**，靠后的几条才勉强挨上，图上「谁说明谁」只能靠数行；网盯的是**性质**：字的起点落在条右沿之后 1~12px 内，不卡「某处写了 7」，间距调了不误报）；**状态时间线的字都在画布内**（右列宽度现在是按内容现算的，算窄了会被右边缘裁；两份宽估各写一份是**有意的**——`app.js` 那份决定**留多宽**，网里这份决定**够不够**，合成一份等于自己验自己）。第一条网当场抓出一个**写它时并不知道的 bug**：无解那两份载荷里 `states` 是空的，画布高 `padT + 0 × rowH + 6 = 14`，而那句「没有状态可画…」画在 y=19——整句落在画布之外被裁掉，屏上就是个空盒子，「说了等于没说」；改成 `Math.max(1, states.length)` 留出一行的高度。三条网各自被自己的 bug 证伪过（改一份 `app.js` 副本再用 `--app` 指过去，不碰仓库）：`vpadT` 退回 14 → 9 条红、精确点名「`#vram-svg` 的「容量 160 B」y=9」；说明退回右对齐 → 7 条红（两份无解载荷没有条，不适用）；`padR` 收成 35px → 7 条红；无解不留行高 → 2 条红。用例除了五份有解的载荷，还有**两份没有解的**（容量 80 / 容量 1）：它们走的正是「没有解、先后退回拓扑秩」那条分支，而那条分支在浏览器里只要把 VRAM 调小就进得去。无解时树上没有终点圈，交互那一项**记为跳过**（`skip`），不进通过计数、挂在总结行上——**「没跑」不算「跑过了」**。另有两份专门的：**导出快照**那份（`mode="snapshot"`）验的是「只读」这一支——顶栏选择器与参数表里 8 个输入框的 `disabled` 必须**恰好**等于「是不是快照」；**载荷里没有名单**那份（内存里构造的、`scenario_path=None`，也正是「打开一份更早导出的快照」的形状）验的是名单空着时整格收起来、选中值空着，**绝不退而显示第一个选项**（`<select>` 的 value 不在选项里时浏览器就是那么干的，于是顶栏会理直气壮地写着另一张图的名字）。**换计算图**那一下是这一组里唯一需要服务端的：只有那一个用例配一个 fetch 桩（其余用例走的仍是那个会炸的 fetch——「不小心走到 fetch」要当场暴露），桩里返回的是 `viewer.py` **现装配出来**的那份新载荷，然后在选择器上派发一次 `change`、等 promise 落地，要求「一次 POST + body 里带 scenario」「顶栏换成新那份」「计算图重画成新图的方块数（并先确认两张图的方块数本来就不同，否则这条是空转）」「换完没有 NaN」。它**自带证伪**（`--falsify`）：往 `GRAPH.boxW` 里注入一个 NaN，那张网必须变红——不做这一步，「扫 NaN」就没被证明过。**`node` 不是运行依赖**（`DESIGN.md` 禁的是依赖，`tools/` 里这个脚本是开发工具）：没有 `node` 就跳过这一组，95 项 + 总结那一行的一句说明，而不是假装验证过。

最后两组之外的收尾：唯一绑套接字的组绑定失败只**跳过并说明**，不计 ✗——沙箱不给套接字是环境问题，不是逻辑问题。启动器那一组是写完之后立刻抓到真问题的一组：头一版 `viewer.cmd` 的注释里带了一句中文，27 个非 ASCII 字节，双击时正是会静默出错的那种。

**v1 明确不做**：真实 Qwen 配置的**图**（仓库里没有它们的 `.workload.json`；它们只作硬件/策略来源，选择器标签就叫「硬件 / 策略来源」）；接隔壁 `..\visualization-llama.cpp-tensor\`（另一个仓库、另一套数据格式）；拖拽编辑图（图是从 `producer_of`/`consumers_of` 推的，没有编辑语义可给）；MoE / 多请求 / 多 GPU（内核直接 `Unsupported`，查看器**显示**这条拒绝而不是建模它）；参数扫描与 N 路对比（前后两方案够用，扫描是另一个工具）；搜索树的展开动画；把编辑写回磁盘（查看器只写用户显式指定路径的快照，绝不重写 `examples/*.json`）。另外两处实现中发现载荷里**没有**的东西：`bridge.calibration` 那条标定对照行、以及桥 4 的「并排 K 表」——都只存在于计划里，v1 没做，也没留占位。**两条时间线仍然不画在同一条轴上**：推导出的仍是张量内核的**整数纳秒**，不是 `llm_infer_model.simulator` 那条浮点秒、逐层的时间线。

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
