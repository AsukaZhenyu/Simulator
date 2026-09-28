# M1 验证记录（`schema_version = "0.2"`：混合计算 · 有限内存 · SSD）

这份是 `M1_ACCEPTANCE.md` §12.5 要的那份记录：**实际跑过的命令、基线/新增的测试结果、
随包每一份 scenario 的实数、环境限制与尚未实测项**。规则出处是 `M1_DESIGN.md`，
逐条验收是 `M1_ACCEPTANCE.md`。

一句话结论：**11 份随包 0.2 scenario 全部 `optimal` / `goal_popped`，与 §2～§9 表里的
最优值逐字段相同；`mapping` 侧测试 202 → 346 项（时钟修正又加 1 项，见第 7 节）、`modeling` 侧 57 项、查看器自检
975 项 + 4 处跳过、`demo_window_vs_search.py` 全部对照一致，全绿。**

**全部时长都是合成的。** 下面每一个 `makespan_ns` 都来自 scenario 自带的成本表
（`costs.source = "synthetic_fixed"`），不是任何真机的实测，也不能当预测读。
本记录里没有一条结论是靠「截了一张图看着正常」下的——每条要么是搜索自己报的数、
要么是回放账本逐字段比出来的。

---

## 1. 跑什么

工作目录一律 `Simulator/mapping`；`python` 指装好两个包的开发环境（本机自用的在
`modeling/.venv`）。

```bash
# 1) 两份测试
cd modeling && ../modeling/.venv/Scripts/python.exe -m unittest discover -s tests
cd ../mapping && ../modeling/.venv/Scripts/python.exe -m unittest discover -s tests -t .

# 2) 随包 11 份 0.2 scenario（逐份换 --scenario）
../modeling/.venv/Scripts/python.exe -m tensor_mapping search \
    --scenario examples/m1-g-spill-dram96.json --out /tmp/out-m1-g

# 3) 查看器自检（DOM 垫片真跑 app.js）
../modeling/.venv/Scripts/python.exe tools/viewer.py --self-check

# 4) 窗口策略 × 搜索的独立对照
../modeling/.venv/Scripts/python.exe tools/demo_window_vs_search.py
```

`-t .` **不能省**：`mapping/tests/` 是个包，少了它发现器只找到 5 个用例且全部报错。
`DESIGN.md:260` 写的是不带 `-t .` 的形式，冻结契约不改，差异记在 `README.md` 里。

## 2. 基线 vs 新增

基线是 HEAD `0cf687c`（M1 之前），实测数逐个文件点过：

| 测试文件 | 基线 | 现在 | Δ |
|---|---:|---:|---:|
| `test_cli.py` | 30 | 30 | 0 |
| `test_engine.py` | 54 | 54 | 0 |
| `test_ggml.py` | 13 | 13 | 0 |
| `test_mapper.py` | 25 | 30 | +4（0.2 的映射同一性/指纹）+1（时钟，见第 7 节） |
| `test_policies.py` | 21 | 21 | 0 |
| `test_spec.py` | 59 | 59 | 0 |
| `test_m1.py` | —（不存在） | **139** | +139 |
| **合计** | **202** | **346** | **+144** |

**0.1 那 202 项一个都没删、一个都没改。** 验法是 `git diff -U0 -- mapping/tests` 里
`^-.*def test_` 的条数为 **0**。`M1_ACCEPTANCE.md` 那句「不能改旧预期数字来适配 M1」
因此是查得出来的，不是靠自觉。

其余三处：

| 检查 | 结果 |
|---|---|
| `modeling` 侧 | `Ran 57 tests ... OK`（与 M1 之前相同，M1 没有动建模侧） |
| 查看器自检 | `✓ 全部 975 项通过（前端交互一项跳过 4 处）`——4 处跳过是因为 `chain-cap160` 的计划里没有 SSD 访问，那四条网改问「有则验、无则明说跳过」 |
| `demo_window_vs_search.py` | `✓ 全部对照一致`；`search` 自身评估与独立回放逐字段一致（含 `events`） |

测试组织按 `M1_ACCEPTANCE.md` §12.2 的要求：**一个类管一节验收**（`ComputeDeviceChoice`
对 §2、`HybridDelivery` 对 §3、`OverlapSwitches` 对 §4、`DramCapacity` 对 §5、
`TwoStageWeightPipeline` 对 §6、`CapacityDrivenPlacement` 对 §7、`SpillAndRecover`
对 §8、`SsdOutputDelivery` 对 §9、`Refusals` + `RefusalPairs` + `LedgerDetail`
对 §10），每张验收表在同一个类里逐行钉住。容量/设备变体**不各存一份 JSON**，
在测试里换一个旋钮现造。

## 3. 随包 11 份 scenario 的实数

就是第 1 节第 2 条那条命令，逐份换 `--scenario`。字节单位为 B。

| scenario | status | makespan_ns | peak_vram | peak_dram | h2d | d2h | ssd_r | ssd_w | expanded |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m1-a-cpu6`（A） | `optimal` | 6000000 | 0 | 96 | 0 | 0 | 0 | 0 | 35 |
| `m1-b-hybrid`（B） | `optimal` | 5000000 | 32 | 48 | 16 | 16 | 0 | 0 | 85 |
| `m1-c-fork`（C） | `optimal` | 6000000 | 32 | 64 | 0 | 16 | 0 | 0 | 58 |
| `m1-c-two-outputs`（C） | `optimal` | 6000000 | 48 | 48 | 16 | 16 | 0 | 0 | 91 |
| `m1-d-dram96`（D） | `optimal` | 7000000 | 0 | 96 | 0 | 0 | 64 | 0 | 11 |
| `m1-e-dram128`（E） | `optimal` | 13000000 | 160 | 128 | 128 | 0 | 128 | 0 | 301 |
| `m1-f-dram96`（F） | `optimal` | 6000000 | 0 | 96 | 0 | 0 | 64 | 0 | 34 |
| `m1-g-spill-dram96`（G） | `optimal` | 10000000 | 0 | 96 | 0 | 0 | 80 | 16 | 37 |
| `m1-h-ssd-cpu`（H） | `optimal` | 3000000 | 0 | 32 | 0 | 0 | 0 | 16 | 6 |
| `m1-h-ssd-gpu`（H） | `optimal` | 4000000 | 32 | 16 | 0 | 16 | 0 | 16 | 14 |
| `m1-recovery-shared` | `optimal` | 9000000 | 0 | 144 | 0 | 0 | 64 | 0 | 37 |

十一份**全是 `goal_popped`**（`optimality_proven=True`），逐字段对得上
`M1_ACCEPTANCE.md` §2～§9 那张「最优」列。`m1-recovery-shared` 不在 A～H 表里，
是 §10 末尾要的「共享权重释放 DRAM 缓存后从 SSD 重读恢复」那个额外用例。
`m1-g-spill-dram96` 是唯一同时有 SSD 读与写的：中间值 `h` 写 16 B、之后读回 80 B，
而 `h2d=0 / d2h=0`——它全程没走 PCIe，这正是「DRAM 逼出来的放置反转」。

**人工计划也回放过。** §5～§9 的「人工方案 / 人工计划」不存 JSON，在测试里逐动作
构造再 `model` 回放、与账本逐字段对齐（例如 `test_the_hand_plan_replays_to_the_stated_ledger`
把 §8 那份 10 ms 计划按动作对齐到 96 / 80 / 16 这几个数）。

## 4. §10 覆盖审计：10 个缺口与它们的闭合方式

按 §10 那张 32 行的表逐行查过一遍，**10 行在 M1 落地后仍然没有用例**——它们有共同
的形状：0.1 只有一个计算设备、一个宿主域、一个对齐，所以这些行的「另一半」在 0.1 里
**根本写不出来**，而 M1 加了 CPU / DRAM / SSD 之后才第一次可达。

| §10 那一行 | 缺的是哪一半 | 补在哪（`tests/test_m1.py`） |
|---|---|---|
| CPU 输入仅在 VRAM，或 GPU 输入仅在 DRAM | VRAM→CPU 这个方向 | `RefusalPairs.test_an_input_only_in_vram_is_not_ready_for_the_cpu` |
| COPY/READ/WRITE 的源不存在或未就绪 | WRITE 这一半 | `RefusalPairs.test_a_write_whose_source_is_absent_is_illegal` |
| 两个 H2D、H2D+D2H、两个 SSD 读、SSD 读+写、两个 SSD 写 | H2D+D2H、两个 SSD 写 | `RefusalPairs.test_the_bus_excludes_an_h2d_and_a_d2h`、`test_two_writes_collide_on_the_ssd_slot` |
| CPU/GPU 同时尝试同一算子 | 「即使设备不同」那半 | `RefusalPairs.test_the_same_operation_cannot_start_on_the_other_device` |
| 写入目标在途，尚无 READY 备份 | 整行 | `RefusalPairs.test_an_in_flight_write_is_not_a_backup`（见下「够不着」） |
| CPU/GPU workspace=16 B | 整行（随包 fixture 的 workspace 全是 0） | `LedgerDetail.test_a_cpu_workspace_is_charged_to_dram` + `…_gpu_…_to_vram` |
| 域 reserve=16 B | 「峰值包含 reserve」那半 | `LedgerDetail.test_a_runtime_reserve_is_inside_the_dram_peak` |
| alignment=32 B 的 16 B 向量 | 「两域不同 alignment 分别应用」那半 | `LedgerDetail.test_alignment_is_applied_per_domain` |
| 两域初始副本 / 初始超容量报告域及 required/capacity | 整行 | `LedgerDetail.test_a_leaf_in_two_domains_is_charged_in_each`、`test_an_impossible_initial_layout_names_the_domain_and_the_numbers` |
| 0/负/非整数持续时间、非有限推导值 | 负、非有限（0 与非整数在 `test_spec`） | `LedgerDetail.test_a_negative_or_non_finite_duration_is_refused` |

补的这 13 项落在**两个新类**上：`RefusalPairs`（6 项，0.1 写不出来的那几对拒绝）、
`LedgerDetail`（7 项，两个账本自己，不是计划）。`tests/test_m1.py` 因此 126 → 139 项。
没有为凑数加报告格式用例——`M1_DESIGN.md:249` 那句「优先覆盖设备放置、数据保留、
容量和最优值」就是照这个办的。

两处**够不着**，一并记下，别把它们读成「验过了」：

* **`test_an_in_flight_write_is_not_a_backup`**——`h` 是 `c3` 唯一的输入来源，
  所以**不**开写的时候 `EVICT h dram` 一样被拒（可达性：`leaves no way to get it to c3`）。
  这行钉住的是「拒绝」，钉不住「因为写还在途所以拒绝」这个理由：两者的错误码都是
  `TENSOR_IN_USE`，锁优先级在前。这个 bug（把没落地的目标当成备份）在引擎里没有
  任何一条代码路径会犯，所以也没有东西可以拿来证伪。
* **`test_a_gpu_workspace_is_charged_to_vram`** 与 CPU 那条是**差分**测的
  （同一个算子跑两遍，`workspace_bytes` 取 0 与 16 相减），不是拿「算子开始前的
  账本」直接比：算子启动时**同时**给输出分配空间，直接比会把 `输出 + workspace`
  两件事一起断言了。差分也顺带把「释放」钉死了（`after` 两次相同）。

## 5. 证伪记录

三组注入，**全部在内存里 monkeypatch，仓库一个字节都没写**。判据按**网名**比，
不按用例名：同一张网在多份用例上各红一次是预期的，**连带**指别的网也红了。

| 组 | 注入的是什么 | 结果 |
|---|---|---|
| 前端（查看器） | 12 处真实前端 bug：0.1 也露内存曲线、两条曲线互相读错账本、容量标注取错域、SSD 画成有容量的第三条曲线、SSD 拆成两条泳道、两个方向挂同一箭头、柱子不写方向、写也画实线、单步那帧前端自己加时长、轴刻度混进泳道名列、0.2 的「开局就在 DRAM 里」判回 0.1 算法 | **12/12**：每一处都只让该红的那张网红 |
| 账本（`_TreeRecorder` 之外的 M1 语义） | 工作区不计账、运行时保留量不进峰值、两个域共用一个对齐 | **4/4**：对应的 4 条断言各自变红 |
| 搜索树记录（`_TreeRecorder`） | 换父漏改、换父改动作漏改、`on_expand` 漏记、被剪边当真实出边 | 见下 |

第三组的结论是**部分**的，逐条实测（跑在 6 份 fixture 上：`three_layer_chain-cap1024`、
`chain-cap160`、`fork-cap160`、`residual-cap160`、`m1-b-hybrid`、`m1-g-spill-dram96`）：

* **`on_expand` 漏记 → 抓得到**，6/6 红（`展开数：树上标了展开的节点 0 个 vs
  search expanded_states=…`）。这条网是结实的。
* **换父漏改 / 换父改动作漏改 → 抓不到**，而且**不是因为够不着**：`m1-b-hybrid`
  上「严格更省才换父」那条分支确实触发了 **2 次**（节点 33 与 40，共 88 个节点；
  全仓 1181 次登记里只有这 2 次）。漏掉的原因是 `_trace_check` 只沿**终局节点的
  父链**走（那一刻链上只有 9 个节点），33 与 40 都不在链上，于是父链长度、动作条数、
  节点数、展开数**四样一个都没变**。链外的损坏它看不见。
* **被剪边当真实出边 → 抓不到**：`_trace_check` 从不读 `children`，只从终局往上走。

所以「树记录器没写错」这句话，本轮的证据只覆盖到 `on_expand` 与父链本身；
**链外与 `children` 上的损坏没有网**。这是记录，不是待办：要补就得再加一条沿
`children` 走查的不变量，而 `children` 里本来就可能存在「曾走过的边、后来被更省的
路径换了父」这种合法情形，判据不能简单写成 `to.parent == this`——那一处要小心设计，
本轮没做。

## 6. 环境限制与尚未实测项

* **时长全部合成。** 见开头那段。桥接真硬件参数的 `costs_from_model_config` 在 0.2
  下一样要过三道门，随包的 config 全停在门 1；M1 没有改这件事，本记录里也没有任何
  一条数是从真机或真模型推出来的。
* **SSD 只建了平均有效读写速度。** 页、擦除块、队列深度、写放大、GC 都不在模型里
  （`M1_DESIGN.md` 的取舍），所以 `ssd_read_bytes` / `ssd_write_bytes` 是**字节数**，
  不是访存次数，也不能换算成真实盘上的一次操作。
* **证伪脚本是会话本地的，不在仓库里。** 上面三组注入都是在 `%TEMP%` 的一次性脚本里
  做的（前端那组改的是 `app.js` / `viewer_smoke.mjs` 的**临时副本**，靠 `--app`
  与脚本路径换进去）。它们**没有被检入**，所以从一次干净 clone 出发复现不了——
  所以第 5 节把每一处注入**它要抓的那个 bug** 逐条写了出来（不是只写「改了 12 处」），
  照着重做即可。这是本记录最弱的一环，明写在这里。
* **没有做过的事**：没有对 0.2 做墙钟/内存的规模测试（`m1-e-dram128` 展开 301 个状态，
  是 11 份里最大的）；没有验证查看器在 0.2 下的**导出快照**（`--export`）与在线状态
  逐字段一致——自检覆盖的是在线的 975 项（快照能出、233 KB、能换图，但没做逐字段比对）。
  这两条进入重复层计算图之前应补测。
* **由审阅补做的一项**：0.2 的搜索产物（G、H、E）连同旧 M0 场景，已交给 `model`
  子命令独立重放，事件、状态与主要统计量一致。这不是本文件跑的，记在这里是为了不让
  上面那句「没跑过 `model` 重放」被读成仍然成立。
* **上游没动**：`llama.cpp` 私有分支、旧 simulator、历史实验数据、`examples/` 里
  已有的 M0 文件都没改；推导出的 `Costs` 只在内存里，不写盘，`outputs/*/mapping.json`
  的指纹因此不受影响。


---

## 7. 收尾修正：搜索时钟（2026-09-28）

审阅指出 `WallClock` 用 `time.monotonic()`，本机粒度为 15.625 ms。**实测确认为真，而且比
「不准确」更严重：小于一个 tick 的时间预算根本不会触发。** 本机 `monotonic` 在 300 ms
里只出现过两种步进（16000 µs ×12、15000 µs ×7），`perf_counter` 出现 92 种、最常见
1 µs；`m1-e-dram128` 上设 1 ms 预算，实测跑满 **7.8～16.4 ms / 43～100 次展开**——那段时间里
`elapsed_s()` 一直读到 0.0。

`M1_DESIGN.md:225` 本来就写明「搜索的实际耗时和预算使用单调高分辨率壁钟，例如
`time.perf_counter()`」，所以这是实现与设计不符，不是取舍。改了两处：
`modeling/llm_infer_model/tensor/engine.py` 的 `WallClock` 与
`mapping/tensor_mapping/mapper.py` 的 `search` 计时起点。`perf_counter` 仍是单调时钟
（`QueryPerformanceCounter` 不受系统时钟调整影响），换掉不损失这条性质。

修前 / 修后（`m1-e-dram128`，无预算时 301 次展开）：

| 预算 | 修前实际耗时 | 修前展开 | 修后实际耗时 | 修后展开 |
|---|---:|---:|---:|---:|
| 0.5 ms | ~16 ms | 43～100 | **0.51～0.70 ms** | 3～4 |
| 1 ms | 7.8～16.4 ms | 43～100 | **1.05～1.08 ms** | 6 |
| 5 ms | 13.9～16.8 ms | ~76 | **5.01～5.14 ms** | 27～28 |
| 50 ms | 48～63.8 ms | 224 | **50.67～50.78 ms** | 260～267 |

超出量从此由「一次展开」（约 3%）而不是「一个 tick」（约 1500%）定。

网：`tests/test_mapper.py::test_the_real_clock_can_enforce_a_budget_below_one_tick`。
断言落在**停得下来**上（`expanded_states` 小于无预算那趟、状态不是 `optimal`、也不是
`infeasible`），不落在某个墙钟数字上——墙钟数字是关于机器的陈述，在忙的机器上也得成立。
证伪：把 `mapper.WallClock` 在内存里换回 `monotonic` 版本，这张网**当场变红**；换回
`perf_counter` 即绿。**没有改动任何既有的预期数字**，也没有删改既有用例。

顺带更正了三处**记录**（不是删掉旧结论，是在原处加日期更正，历史实测状态仍然可读）：
`mapping/M0_VERIFICATION.md` §3.2 与开头、`mapping/README.md`「命令行上的已知限制」与
开头、`MODELING_MAPPING_BRIEF.md` 的「验收与限制」；`viewer.py` 的 `FakeClock` 与
`viewer_payload.py` 的 `clock` 参数说明也改了——假时钟的理由不再是「真时钟做不到」，
而是 `M1_DESIGN.md:225` 那句「避免依赖某台机器恰好跑多快」。

`mapping` 侧因此 345 → **346** 项，`modeling` 侧仍 57 项。
