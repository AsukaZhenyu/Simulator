# M0 验证报告

本文件是 `ACCEPTANCE.md` §7 第 6 条要求的交付物：「实际执行命令和验证报告，区分自动测试、人工核对、未运行项与环境限制。」

报告日期 2026-09-25，代码状态见文末「本次对应的代码状态」。所有命令都在 `mapping/` 目录下执行，解释器为本机 `python`。

**结论先说**：`ACCEPTANCE.md` §7 六条中，1～5 条已具备并有本轮实测证据；第 6 条即本文件。`DESIGN.md` §9:285 要求的集成验证「GGML 编译 → 实际导出 → 模拟/搜索」**本轮从零走通**：编译腿在全新空目录里 configure + build，导出腿用这个新构建的二进制重新导出四张图，产物与已入库文件逐字节相同。已知的环境限制只剩时钟粒度那一条（第三节 3.2），它不影响任何 M0 验收数字。

---

## 1. 自动测试

### 1.1 全量测试

```bash
cd mapping
python -m unittest discover -s tests -t .
```

结果：

```text
Ran 181 tests in 3.233s

OK
```

**必须带 `-t .`**。`DESIGN.md:260` 写的是 `python -m unittest discover -s tests`，但本目录的 `tests/` 是包（有 `__init__.py`、内部用相对 import），少了 `-t .` 时发现器只找到 5 个用例且全部报错。这是既有事实，冻结契约不改，差异记在 `README.md`。

181 项的分布：

| 文件 | 覆盖 |
|---|---|
| `tests/test_spec.py` | 输入契约、schema、拒绝路径、指纹 |
| `tests/test_engine.py` | 状态转移、四个动作、固定 mapping 回放、`ACCEPTANCE.md` §5 语义表 |
| `tests/test_mapper.py` | uniform-cost 搜索、状态去重、去重键不含绝对 `t`、终止原因 |
| `tests/test_ggml.py` | 真实导出产物、与 fixture 逐字段等价、导出器逐字节重现 |
| `tests/test_cli.py` | 两个入口、退出码、四份产物、M0.4 回放一致 |

> **此后新增**：对照轮（`ALIGNMENT_IMPLEMENTATION.md`）加了 `tests/test_policies.py` 21 项（窗口策略、旧模型与新内核的逐行/逐事件对照、成本桥接、拒绝路径），`modeling/tests/test_tensor_layer_costs.py` 另有 19 项。全量现在是 **202 项**。本节上面的 181 与「Ran 181 tests」是**验收当时**的记录，按 §7「保留 M0 规则和数字」原样留着；新增项不动本报告的任何一条 M0 结论，链式 160 B / 96 B / 95 B 三个场景仍分别是 8 ms / 10 ms / 不可行。

### 1.2 导出腿（真实 GGML 二进制）

```bash
python -m unittest tests.test_ggml
```

结果：

```text
Ran 13 tests in 0.347s

OK
```

**13 项全部通过，零 skip**，其中 `test_the_exporter_reproduces_the_committed_artifacts` 是把 `ggml-export-workload.exe` 真实跑一遍、与仓库里已提交的导出产物逐字节比对（不只是语义等价）。所以 `DESIGN.md` §9:285 那条链的「实际导出」这一环是真实执行的，不是靠 fixture 顶替；同一组测试用**本轮从零构建**的 exe 再跑一次也是 13 项零 skip，见 3.1。

按 `DESIGN.md` §9:287（不为机械细节写镜像测试），测试集中在语义，不逐字段镜像文件结构。

---

## 2. 人工核对

以下每一条都是本轮实际敲下去的命令与看到的输出，不是从测试代码反推的。

### 2.1 四条文档命令，从干净输出目录

```bash
cd mapping
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping examples/chain.mapping.json \
                                --out outputs/chain-replay
python -m tensor_mapping search --scenario examples/chain-cap160.json \
                                --out outputs/chain-search
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping outputs/chain-search/mapping.json \
                                --out outputs/chain-replay2
python -m tensor_mapping search --scenario examples/ggml/chain-cap160.json \
                                --out outputs/ggml-search
```

四条全部退出 0，产出 14 个文件（`chain-replay`/`chain-replay2` 各 3 个，两个 search 各 4 个）。这同时验证 `--out` 指向不存在的目录时会自行创建，即 §9 的 M0.4 门槛「从干净输出目录复跑」。

### 2.2 M0.4 门槛：映射独立回放一致（`ACCEPTANCE.md` §6:106）

比较 `outputs/chain-search/`（search 写出）与 `outputs/chain-replay2/`（把那份 mapping 用 model 独立回放）：

| 项 | chain-search | chain-replay2 |
|---|---|---|
| `status` | `optimal` | `valid` |
| `makespan_ns` | 8000000 | 8000000 |
| `peak_vram_bytes` | 160 | 160 |
| `h2d_bytes` | 128 | 128 |

`events.json` 的 **`events` 列表本身逐字节相同**；两文件顶层的唯一差异键是 `mode`（`search` vs `model`），这是两个入口各自的标记，不是执行差异。

再单独用核心 API 核对一次身份验证：

```python
from tensor_mapping.engine import load_mapping
from tensor_mapping import load_and_validate
sc = load_and_validate('examples/chain-cap160.json')
d = load_mapping('outputs/chain-search/mapping.json', sc)
# fingerprint_verified == True, len(d.actions) == 9
```

整个 `chain-search`/`chain-replay2` 这一对在 `examples/ggml/` 上同样成立。

### 2.3 `ACCEPTANCE.md` §2～§4 的全部容量判据

容量变体（96/95/112/111）**没有** checked-in 的 example 文件，这是刻意的（`DESIGN.md` §9:287 不为机械变体堆文件，`tests/support.py` 的模块 docstring 也引了这一条）。所以命令行核对要先按容量写一份 scenario 副本：

```python
import json, pathlib
ex = pathlib.Path("examples").resolve()
d = json.loads((ex / "chain-cap160.json").read_text(encoding="utf-8"))
d["architecture"]["vram_capacity_bytes"] = 96
d["workload_file"] = str(ex / pathlib.Path(d["workload_file"]).name)  # ← 见下
pathlib.Path("outputs/chain-cap96.json").write_text(
    json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
```

**坑**：`workload_file` 是**相对 scenario 文件**解析的，副本换了目录就必须改成绝对路径，否则报 `invalid_input: cannot read .../chain.workload.json`。

写好后逐条跑 `search`，实测结果与契约数字**逐条一致**：

| 图 | 容量 | 契约要求 | 实测 `status` | 实测 makespan | 终止原因 |
|---|---|---|---|---|---|
| chain | 160 | 8 ms | `optimal` | 8 ms | `goal_popped` |
| chain | 96 | 10 ms | `optimal` | 10 ms | `goal_popped` |
| chain | 95 | infeasible | `infeasible` | — | `search_space_exhausted` |
| residual | 160 | 9 ms | `optimal` | 9 ms | `goal_popped` |
| residual | 112 | 11 ms | `optimal` | 11 ms | `goal_popped` |
| residual | 111 | infeasible | `infeasible` | — | `search_space_exhausted` |
| fork | 160 | 9 ms | `optimal` | 9 ms | `goal_popped` |
| fork | 112 | 11 ms | `optimal` | 11 ms | `goal_popped` |
| fork | 111 | infeasible | `infeasible` | — | `search_space_exhausted` |
| matvec | 160 | （§5 表内） | `optimal` | 5 ms | `goal_popped` |

退出码也符合 `README.md` 的约定：`optimal` 退 0，`infeasible` 退 2。

### 2.4 禁用 copy/compute 重叠

`ACCEPTANCE.md` §5 表后的性质：「链式 160 B 禁用 copy/compute 重叠时应得到 10 ms」。

```bash
# scenario 副本，mapspace.allow_copy_compute_overlap = false
python -m tensor_mapping search --scenario outputs/chain-nooverlap.json --out outputs/o-nooverlap
```

实测 `optimal`，10 ms。与 2.3 表中 chain/160 的 8 ms 相差恰好 2 ms，即重叠被拿掉后两条 COPY_H2D 无法与 COMPUTE 并行的那部分。

### 2.5 预算不足必须是 `unknown`（`ACCEPTANCE.md` §6:110）

```bash
python -m tensor_mapping search --scenario examples/chain-cap160.json \
                                --max-expanded-states 1 \
                                --out outputs/o-budget
```

实测：退出 2，`status = unknown`，`termination_reason = state_budget_exhausted`，`search.optimality_proven = false`，且 `limits.max_expanded_states = 1`——**记的是实际生效的预算，不是 scenario 的默认值**。这就是「不得把预算不足写成无解」的端到端版本。

### 2.6 身份检查真的会拦（`ACCEPTANCE.md` §6:112）

把 `examples/chain.mapping.json` 拿去回放一份 id 不同的 scenario：

```text
identity_mismatch: examples/chain.mapping.json: chain.mapping.json names scenario
'chain-cap160' but was replayed against 'chain-align8'
```

退出 1（输入层面被拒，不是裁决），且**一个文件都没写**。

已知限制（`README.md` 也记着）：`examples/chain.mapping.json` 与 `examples/ggml/chain.mapping.json` **都不带指纹**，只靠 `scenario_id` 做身份检查，而两个 scenario 的 id 恰好相同，所以互相回放不会被拦。两份 workload 按设计逐字段等价，回放数字本来就一样，不影响正确性；但**手写 mapping 没有跨图保护**，`search` 产出的 mapping 带两个指纹才有。

### 2.7 对齐语义（`DESIGN.md`:190）

「COPY_H2D 的 `bytes_transferred` 是未对齐的 `storage_bytes`，EVICT 的 `bytes_released` 是已对齐的 allocation」——在 `allocation_alignment_bytes = 1` 的 fixture 上两者恰好相等，所以必须换一个 alignment > 1 的场景才看得出来。用 matvec（`W` 的 `storage_bytes` 是 48，不是 32 的倍数）、alignment = 32：

```text
COPY_H2D W  bytes_transferred = 48     ← 未对齐的 storage_bytes
status=valid  peak=128  h2d=48         ← 分配是 64（48 对齐到 32），搬运量记 48
```

`h2d_bytes` 记的是 48 而**不是** 64，正是契约要的。`tests/test_engine.py:391` 对同一性质有断言。

### 2.8 每态容量不变式

`states.json` 每一条都满足 `used_vram_bytes <= limits.vram_capacity_bytes`，在 2.1 的四份产物与 2.7 的 alignment=32 产物上各核一次，全部成立。

---

## 3. 集成验证、未运行项与环境限制

### 3.1 GGML 编译腿已在全新目录从零跑通

`DESIGN.md` §9:285 要求「GGML 编译→实际导出→模拟/搜索」的集成验证。本轮三段都真实执行了，且编译腿用的是**全新的空构建目录**（无任何 `CMakeCache.txt` 残留）：

**编译腿**——本节新增的实测：

```bash
cmake -S mapping/ggml -B <全新空目录> -G "Visual Studio 17 2022" -A x64 \
      -DGGML_SOURCE_DIR="<上游 ggml 目录>" -DGGML_CUDA=OFF
cmake --build <全新空目录> --config Release --target ggml-export-workload
```

| 步骤 | 耗时 | 说明 |
|---|---|---|
| configure | **32.7 s** | 空目录，无 cache；输出 `ggml version: 0.9.11` / `ggml commit: unknown` |
| build | **37.6 s** | 含 ggml 本身（`ggml.dll` / `ggml-base.dll` / `ggml-cpu.dll`）与导出器 |

耗时与 `mapping/ggml/README.md` 记的「约 31 s + 38 s」一致。唯一告警是 MSBuild `MSB8029`（中间目录位于临时目录下可能影响增量生成），只影响增量构建，与产物无关。

**导出腿**——用上面新构建的二进制重新导出四张图：

```text
ggml-export-workload: ggml 0.9.11 commit unknown
  context: tensor_overhead=368 graph_overhead=82608 mem_size=1589936
  chain     ggml reports 2 graph nodes, exporter declares 2
  residual  ggml reports 4 graph nodes, exporter declares 4
  fork      ggml reports 3 graph nodes, exporter declares 3
  matvec    ggml reports 1 graph nodes, exporter declares 1
```

四份产物与已入库的 `examples/ggml/*.workload.json` **逐字节相同**：

| 图 | sha256（前 16 位） |
|---|---|
| chain | `cab9b1cc0f82539b` |
| residual | `9ba9961699c2ae2b` |
| fork | `310dce40abb248eb` |
| matvec | `8d3280b1168282c9` |

再把这四份产物与**新二进制**一起交给导出腿测试：用 `TENSOR_MAPPING_GGML_EXPORTER` 指向新构建的 exe，

```bash
TENSOR_MAPPING_GGML_EXPORTER=<全新目录>/bin/Release/ggml-export-workload.exe \
  python -m unittest tests.test_ggml
```

得到 **13 项通过、零 skip**——包括 `test_the_exporter_reproduces_the_committed_artifacts`（真实跑一遍导出器、与仓库产物逐字节比对）。

一处需要说明的细节：新构建的 `ggml-export-workload.exe` 与既有 `mapping/ggml/build/bin/Release/` 下的那份 **SHA-256 不同**。`cmp -l` 显示 103424 B 中只有 **4 个字节**不同，符合嵌入的构建元数据（时间戳一类）差异，不是代码差异。有意义的比对因此不是二进制相等，而是**产物逐字节相同**——上表即是。

**仍受限于仓库之外的东西**：编译腿的 `-DGGML_SOURCE_DIR=` 指向的上游 ggml 源码树（`LLM-Infer/llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml`）在 Simulator 仓库之外，仓库不携带它。本报告证明的是「用这份上游源码 + 本机 VS 2022 工具链，按文档命令可以从零构建出产物正确的导出器」，**不**证明「任意环境、任意上游版本都能构建」。`mapping/ggml/CMakeLists.txt` 对不存在的路径按设计 `FATAL_ERROR`、不自动拉版本，这一条要的正是这种显式失败。

`GGML_CUDA:BOOL=OFF`，导出全程 `no_alloc = true`，从不分配张量数据也不初始化后端，符合 `ACCEPTANCE.md` §7.5「GGML 前端使用 CPU 构建依赖，不要求 CUDA」。

### 3.2 `--wall-time-limit-s` 在本机端到端不可测

Windows 上 `time.monotonic()` 实测是 `GetTickCount64()`，分辨率 **15.625 ms**（连续 2000 次调用返回同一个值）。后果：

- 小图搜索的 `search.wall_time_s` 常常正好是 `0.0`，这不表示搜索瞬时完成；
- `--wall-time-limit-s` 取小于一个 tick 的值时**永远不会触发**，所以 `ACCEPTANCE.md` §6:110 的「预算不足 → unknown」只能用 `--max-expanded-states` 复现（见 2.5），时间预算那条在本机没有端到端证据。

这不影响任何 M0 验收数字——那些全是模拟时间 `_ns`。要修得动 `mapper.WallClock` 的时钟源（换 `perf_counter`），属核心改动，留待后续；`README.md` 的「命令行上的已知限制」记着同一条。

### 3.3 明确不属于 M0 的部分

以下按 `DESIGN.md` §1 与 `ACCEPTANCE.md` §7 的末句属于后续阶段，本报告不声称、也不用于替代本阶段的正确性验收：真实硬件校准、CPU 计算、D2H（设备到主机）复制、SSD、带宽模型、完整 dense、`DESIGN.md:274` 的可视化 adapter。

`assumptions` 字段会逐条声明这些边界（只在对应事实成立时才出现），所以产物自身也带着这些限定。

---

## 4. `ACCEPTANCE.md` §7 六条逐条对照

| # | 要求（原文摘） | 状态 | 证据 |
|---|---|---|---|
| 1 | 可以独立构建的小图导出器，支持 chain/residual/fork，记录 GGML 版本与构建方法 | 具备 | `mapping/ggml/`，`CMakeLists.txt` 以 `GGML_SOURCE_DIR` 接入；`ggml/README.md` 记构建方法与版本；覆盖 chain/residual/fork/matvec 四张正图 + 四张负图；**本轮从零构建通过，见 3.1** |
| 2 | 三个图的实际导出产物 + 对应 scenario + 至少一个固定 mapping | 具备 | `examples/ggml/` 下四张图各一组（导出产物、scenario、mapping）；`examples/` 另有四组 fixture 版本；产物出处见 3.1 |
| 3 | Python 核心、model/search 两个入口和语义/搜索测试 | 具备 | `spec.py`/`engine.py`/`mapper.py`/`cli.py`/`__main__.py`；`tests/` 181 项（对照轮后 202 项，见 1.1 的注）；见 1.1 |
| 4 | 输出最优计划、状态/事件、指标、求解预算与终止原因 | 具备 | 四份产物 `stats.json`/`events.json`/`states.json`/`mapping.json`；预算与终止原因在 `stats.search`，实际生效预算在 `limits`；见 2.1--2.5 |
| 5 | 无 GPU 情况下可运行核心示例；GGML 前端用 CPU 构建依赖，不要求 CUDA | 具备 | 核心纯标准库、无第三方依赖，但要先装本仓库的两个本地包（`python -m pip install -e ./modeling -e ./mapping`，对照轮起 `tensor_mapping` 导入 `llm_infer_model.tensor`）；`GGML_CUDA:BOOL=OFF` 且导出全程 `no_alloc`，见 3.1 |
| 6 | 实际执行命令和验证报告，区分自动测试、人工核对、未运行项与环境限制 | 本文件 | — |

第 1～5 条中「具备」的含义是「组件存在且有本轮实测证据」。第 1、5 条依赖的构建环节已在 3.1 从零走通；剩下唯一未经本轮验证的是**上游 ggml 源码树之外的任意环境**（见 3.1 末段），这是环境边界而非本仓库的未完成项。

---

## 5. 复现本报告的完整命令序列

```bash
cd mapping

# 集成验证：编译腿（需要用全新空目录，见 3.1）
cmake -S ggml -B <全新空目录> -G "Visual Studio 17 2022" -A x64 \
      -DGGML_SOURCE_DIR="<上游 ggml 目录>" -DGGML_CUDA=OFF
cmake --build <全新空目录> --config Release --target ggml-export-workload
"<全新空目录>/bin/Release/ggml-export-workload.exe" --graph all --out-dir <临时目录>

# 自动测试（需先按 README「安装（开发）」装好两个本地包）
python -m unittest discover -s tests -t .        # 202 项（验收当时 181）
python -m unittest tests.test_ggml               # 13 项，零 skip
TENSOR_MAPPING_GGML_EXPORTER="<全新空目录>/bin/Release/ggml-export-workload.exe" \
  python -m unittest tests.test_ggml             # 13 项，零 skip（指向新构建的 exe）

# 人工核对：四条文档命令（干净输出目录）
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping examples/chain.mapping.json --out outputs/chain-replay
python -m tensor_mapping search --scenario examples/chain-cap160.json --out outputs/chain-search
python -m tensor_mapping model  --scenario examples/chain-cap160.json \
                                --mapping outputs/chain-search/mapping.json --out outputs/chain-replay2
python -m tensor_mapping search --scenario examples/ggml/chain-cap160.json --out outputs/ggml-search

# 人工核对：预算不足 → unknown
python -m tensor_mapping search --scenario examples/chain-cap160.json \
                                --max-expanded-states 1 --out outputs/o-budget

# 人工核对：§2～§4 的容量判据（需按 2.3 先写 scenario 副本）
```

`outputs/` 已被根 `.gitignore` 忽略（`mapping/outputs/`），全部产物可从 `examples/` 复跑重建，不入库。

## 本次对应的代码状态

本轮工作分两个提交：

| 提交 | 内容 |
|---|---|
| `fb23c36` | `feat(mapping): model/search 两个入口与 stats/events/states/mapping 四份产物` —— `artifacts.py` / `cli.py` / `__main__.py` / `test_cli.py` 新增，`.gitignore`、`mapping/README.md`、`tensor_mapping/__init__.py` 改动 |
| 本提交 | 这份验证报告，以及 `mapping/README.md` 中引用它的两处 |

3.1 的编译腿构建在**仓库之外的临时目录**里完成，验证后已删除，所以工作树没有因此多出未跟踪文件；`mapping/ggml/build/` 下的既有构建被 `.gitignore` 的 `build/` 覆盖，两份二进制产出相同的产物（见 3.1）。

本轮**未修改** `mapping/DESIGN.md`、`mapping/ACCEPTANCE.md`（冻结契约），也**未修改** `spec.py` / `engine.py` / `mapper.py` 的任何语义，未触碰 `ggml/` 与上游 llama.cpp 树。
