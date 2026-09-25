# mapping 基础设计：M0

日期：2026-09-25。状态：待实现。目标读者：负责实现和验收的开发者。

## 1. 目标、范围与分工

M0 回答：对于一个固定 GGML DAG，在一个 GPU 计算资源、一个 H2D 复制资源及有限显存下，怎样安排加载、计算与释放，使目标输出最早就绪？

参考 Timeloop 的职责划分，但不复刻其循环映射模型：

| 对象 | M0 定义 |
|---|---|
| workload | 不变的算子、张量、依赖、请求输出和初始数据位置 |
| architecture | DRAM/VRAM、可用容量、对齐、独占计算和复制资源 |
| costs | 固定的计算/复制持续时间和算子 workspace |
| mapspace | 允许的动作、可复制的张量，以及允许重叠的条件 |
| mapping | 一条具体动作序列；时间由 evaluator 推导 |
| evaluator / model | 验证并回放 mapping，生成时间、内存与事件记录 |
| mapper / search | 在同一状态转移系统中搜索 mapping |

只优化 makespan。峰值内存和搬运字节是统计量，首版不做多目标或字典序最优化。

支持：固定形状，FP32，连续且独立存储的张量，单次前向，MUL_MAT / RELU / ADD，单输出算子，确定的正整数耗时，不可抢占。

限制：所有计算在 GPU 模拟资源上执行；DRAM 视为充足、只读、始终有效的源副本存储。仅支持 H2D；输入可以初始驻留 VRAM，权重一般在 DRAM。GPU 中间结果没有 DRAM 备份。

暂不支持：CPU 数学计算、D2H、SSD、动态 shape、KV 更新、量化、view/alias/in-place、重计算、算子融合、多 GPU、并发 GPU kernel、共享带宽干扰、实际 allocator 碎片与真实 GPU 执行计划。发现未支持语义应明确拒绝，不静默近似。

## 2. 数据流和代码边界

```text
GGML 构图 → 导出逻辑 workload → 输入校验
                                  ↓
          architecture + costs + mapspace
                                  ↓
                  immutable spec + initial state
                                  ↓
         固定 mapping 回放 / mapper 枚举合法动作
                                  ↓
                    同一个 engine.transition
                                  ↓
            mapping + stats + events + state snapshots
```

建议 Python >=3.10，核心采用标准库、JSON 和 unittest；不为 M0 引入求解器、网络服务或 GPU 运行依赖。图前端为独立 C/C++ 可执行程序。

**模块归属（对照轮更新）**：张量级内核 `spec` / `engine` / `layer_costs` 现位于 `../modeling/llm_infer_model/tensor/`，本目录只保留策略侧（`mapper` 精确搜索、`policies` 窗口策略）与产物/入口。依赖方向单向：`tensor_mapping → llm_infer_model`。理由见 `ALIGNMENT_IMPLEMENTATION.md` §3——规则不该隶属其中任何一个策略。因此两个包要在同一次 pip 调用里一起装（`python -m pip install -e ./modeling -e ./mapping`）；`../modeling/` 仍然是本地包而不是第三方依赖，两边都只用标准库。

复用旧代码的事件队列、配置校验与数据记录经验；旧 `simulate_decode()` 的 next_layer、窗口驱逐、聚合 staging/KV/控制开销不进入新核心。第一版不强制抽取共享库，这一点在对照轮被改掉了：内核已抽到 `modeling` 侧共用，但旧目录本身（`simulator.py` 等）仍原样保留、未被改写。

## 3. GGML 前端

以本地 `llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml` 为候选依赖，记录实际 commit/工作树标识。通过 CMake 参数或明确环境配置提供源码路径，不写死个人绝对路径，不自动拉取最新版本。创建图可使用 `no_alloc=true`，不需要 CUDA 或加载 GGUF。

示例构造语义：

- chain：`h=mul_mat(W1,x); y=mul_mat(W2,h)`。
- residual：`h=mul_mat(W1,x); a=relu(h); b=mul_mat(W2,a); y=add(b,x)`。
- fork：`a=mul_mat(W1,x); b=mul_mat(W2,x); y=add(a,b)`。

利用 `ggml_build_forward_expand` 建图，通过图节点与 `src[]` 递归收集输入/权重叶子。不能只导出运算节点，也不能把节点数组顺序加成额外依赖。

只导出请求输出的祖先闭包。张量 ID 在一次导出中唯一，示例显式命名以稳定复现；不将指针地址作为持久化 ID。`op=NONE` 的叶子不生成 compute 动作。

保留 `ggml_type`、`ne`、`nb`、`op`、`op_params` 和输入对应关系。对于 ADD/RELU/MUL_MAT，首版仅接受已经声明支持的普通参数；非默认扩展语义明确拒绝。检查 F32、连续布局、无 `view_src`，并核查 API 构造中没有原地共享数据。M0 一个存储张量对应一个独立 allocation。

本地版本的 `ggml_relu()` 使用 `GGML_OP_UNARY`，具体子类型由 `ggml_get_unary_op()` 得到 `GGML_UNARY_OP_RELU`，不是一个独立的 `GGML_OP_RELU`。导出需同时保存原始操作与归一化语义，不能因原始op名称不同漏掉ReLU。`no_alloc=true` 时多个data指针均可为null，这不能作为存储别名的证据；依据view关系和已支持的构造语义检查。

以 GGML 实际 `ne/nb` 和 `ggml_nbytes()` 为准，不套用 NumPy 的维度顺序。例子使用 W 的 `ne=[4,4,1,1]`，向量 `ne=[4,1,1,1]`；矩阵 64 B、向量 16 B。补一个矩形 MUL_MAT 导出检查，防止方阵掩盖维度转置错误。

从后端分配/split 改写前的逻辑图提取依赖。现有 llama 图采集器依赖 llama 上下文并调用 split，只借鉴其图遍历/序列化代码，不能原样作为此入口。

## 4. 最小数据契约

采用 JSON，所有文件含 `schema_version: "0.1"`。M0 使用一个 scenario 文件组织独立命名空间，`workload_file` 引用独立 GGML 导出文件。相对路径一律相对于声明它的文件；与 shell 当前目录无关。

时间使用整数纳秒，字节使用整数 B，不接受浮点时间、NaN、负数或用 0 表示未知成本。开始/释放操作为零时间；每个 compute/copy 的 duration 必须 >0。字段缺失应报错，不能隐式给出零成本。

### 4.1 workload

最少包含：

| 字段 | 定义 |
|---|---|
| id | 图 ID |
| origin | `kind=ggml` 或 `synthetic_fixture`，GGML 版本/源代码标识和构造样例名 |
| tensors[] | id、name、role、dtype、ne、nb、storage_bytes、initial_locations |
| operations[] | id、semantic_op、ggml_op、unary_op、op_params、inputs[]、output |
| outputs[] | 必须最终驻留 VRAM 的张量 ID |

role 为 input / weight / intermediate / output，仅用于解释，不取代依赖和副本语义。位置只支持 `dram` / `vram`。只有叶子允许非空 initial_locations；可同时拥有 DRAM 和 VRAM 副本，分别表达。

dtype使用`GGML_TYPE_F32`；ne/nb均保存长度为4的数组，nb单位为字节。semantic_op为MUL_MAT / RELU / ADD；ggml_op保存原始枚举名，unary_op在非UNARY操作上为null。op_params保存固定版本的原始int32数组供溯源，同时按支持的操作检查其含义；不能把合法ReLU子类型参数的非零值当作未支持参数。

每个非叶子恰有一个生产者；叶子至少有一个初始位置。操作输入可以重复，如 add(x,x)，依赖/锁定按唯一张量计数。M0 的 operations 至少一个，且全部位于请求输出祖先闭包内；发现孤立节点、未知引用、循环、重复 ID、多个生产者或不合法 shape 明确报错。

`storage_bytes` 是数据占用，不含后端对齐；M0 的独立 allocation 按 architecture 对齐。实际后端 padding、workspace 与共享存储留待后续校准，不能宣称这是物理显存的精确预算。

### 4.2 scenario 示例

以下为完整的链式示例场景。工作负载由前述 GGML 导出程序产生；时间是合成值。

```json
{
  "schema_version": "0.1",
  "id": "chain-cap160",
  "workload_file": "chain.workload.json",
  "architecture": {
    "dram_capacity_bytes": null,
    "vram_capacity_bytes": 160,
    "runtime_reserved_bytes": 0,
    "allocation_alignment_bytes": 1,
    "gpu_compute_slots": 1,
    "h2d_copy_slots": 1
  },
  "costs": {
    "source": "synthetic_fixed",
    "compute": {
      "c1": {"duration_ns": 2000000, "workspace_bytes": 0},
      "c2": {"duration_ns": 2000000, "workspace_bytes": 0}
    },
    "h2d": {
      "W1": {"duration_ns": 3000000},
      "W2": {"duration_ns": 3000000}
    }
  },
  "mapspace": {
    "compute_device": "gpu",
    "copy_tensor_ids": ["W1", "W2"],
    "allow_copy_compute_overlap": true,
    "allow_eviction": true,
    "allow_recomputation": false
  },
  "mapper": {
    "algorithm": "uniform_cost",
    "max_expanded_states": 0,
    "wall_time_limit_s": 0
  }
}
```

固定资源槽数量为 1；其他数量、不支持的算法或 recomputation=true 应报 unsupported，不默默使用默认。`dram_capacity_bytes=null` 表示显式采用无限源存储假设，其他值在 M0 不支持。对齐必须为正整数，reserve 非负且不超过容量。

compute 表应覆盖所有操作；h2d 表覆盖 copy_tensor_ids。可复制对象必须为具有 DRAM 初始副本的叶子。其他副本的迁移不在动作空间中。

### 4.3 mapping 文件

mapping 记录 scenario ID、workload 与有效配置内容指纹，以及有序动作。最少动作形式如下：

```json
{
  "schema_version": "0.1",
  "scenario_id": "chain-cap160",
  "actions": [
    {"kind": "COPY_H2D", "tensor_id": "W1"},
    {"kind": "ADVANCE"},
    {"kind": "COMPUTE", "operation_id": "c1"},
    {"kind": "COPY_H2D", "tensor_id": "W2"},
    {"kind": "ADVANCE"},
    {"kind": "EVICT", "tensor_id": "W1"},
    {"kind": "EVICT", "tensor_id": "x"},
    {"kind": "ADVANCE"},
    {"kind": "COMPUTE", "operation_id": "c2"},
    {"kind": "ADVANCE"}
  ]
}
```

此段省略指纹以便阅读；实际 search 输出必须包含 SHA-256 指纹，model 应核对。规范化 JSON 使用稳定 key 顺序、固定 separators、UTF-8，对 workload 与 scenario 的有效语义内容分别计算；排除 origin 的时间戳和外部文件路径等非语义字段，保留所有影响合法性/耗时的参数。实现需在测试中固定规范化规则。

人工 mapping 可不带指纹，用于编写示例；model 仍完整校验并在结果标记 `fingerprint_verified=false`，不能伪造已核验身份。

mapping 不指定开始/结束时间；时间由动作转移推导。model 按给定动作逐条执行，不自动补预取、释放、推进或纠正非法动作。计划中可标注注释，但注释不影响执行。

## 5. 运行状态

不可变 spec 与每条分支的 State 分离。State 至少包含：

- 当前时间 t；操作状态 NOT_STARTED / RUNNING / DONE。
- 每个张量 GPU 副本状态 ABSENT / RESERVED_COPY / RESERVED_OUTPUT / READY。
- 正在进行的 compute/copy：任务 ID、剩余时间、输入读锁、输出/目标预留、workspace。
- 显存分配集合；初始驻留、输出、在途副本、workspace 均计费。

DRAM 副本在 M0 不变化，可放在 spec。GPU memory used 与读锁可以从状态推导；若缓存这些字段，每步断言其与分配集合/运行任务一致。不要出现两个互相独立的内存账本。

同一个张量的 GPU 副本最多一份，不允许对 READY 或 RESERVED 对象再次启动复制。权重被 EVICT 后仍可从 DRAM 重新加载；操作只允许执行一次。

`size_alloc(tensor)=ceil(storage_bytes/alignment)*alignment`，workspace 同样对齐。始终满足：

`used_vram = runtime_reserved + Σ tensor_allocations + Σ running_workspace ≤ capacity`

预留与就绪副本同样占容量。结果的 peak_vram_bytes 包含 runtime_reserved。H2D 传输字节按 storage_bytes 统计，不按 allocation padding 统计。

## 6. 动作的精确语义

### COPY_H2D(tensor)

前置：对象位于 copy_tensor_ids；DRAM 副本有效；GPU 副本 ABSENT；copy 资源空闲；存在至少一个 NOT_STARTED 消费者；目标 allocation 可放入显存。若禁用重叠，compute 资源也必须空闲。

效果：在 t 立即预留目标容量，置 RESERVED_COPY，占用 copy 资源；建立正持续时间任务。启动不推进时间。完成时置 READY 并释放 copy 资源。

只在当前时刻已知的状态上判定，不根据“下一步也许会释放空间”提前超额预留。没有未启动消费者时禁止多余复制，避免为已完成工作加载数据。

### COMPUTE(operation)

前置：操作 NOT_STARTED；全部输入 GPU 副本 READY；compute 资源空闲；输出尚未分配；输出 allocation 与 workspace 可以同时放入。禁用重叠时 copy 资源也必须空闲。

效果：预留输出与 workspace，锁定所有唯一输入，操作置 RUNNING，占用 compute 资源。不能先释放仍被读取的输入再给输出腾空间。

完成时：输出 READY，操作 DONE，解除读锁、释放 workspace 和 compute 资源。输入张量和权重保留，直到明确 EVICT；不偷偷自动驱逐。

### EVICT(tensor)

前置：allow_eviction=true；GPU 副本 READY；无运行中的读取；不是请求输出。RESERVED_COPY/RESERVED_OUTPUT 不可驱逐，也不允许取消运行任务。

- 若 DRAM 副本存在且对象允许再次复制：未启动消费者可以存在，后续按需重载。
- 若没有可恢复的 DRAM 源，或对象不在 copy_tensor_ids：必须所有消费者 DONE 才能释放。M0 不允许重计算或丢弃未来需要的唯一副本。

效果：立即释放 GPU allocation、置 ABSENT；模拟时间不变。只能驱逐已存在副本，不允许空操作形成零成本循环。

### ADVANCE

前置：至少一个任务正在运行。即使当前还有其他合法启动动作，也允许 ADVANCE，以保留主动等待分支；不能强制采用“能启动就必须启动”的贪心规则。

效果：Δ=min(所有运行任务剩余时间)>0；t 增加 Δ，全部剩余时间减 Δ；将同一时刻结束的全部任务作为一批完成，再生成后续合法动作。可采用半开执行区间 [start,end)，结束时释放资源。

禁止任意 sleep 时长、ADVANCE(0) 或无运行任务的 ADVANCE。成本恒定、无外部到达的 M0 将启动决策限制在事件边界与同刻的零时间动作序列；最优性声明限定在这个动作空间。

### 终止与无进展

成功：所有必需操作 DONE，全部请求输出在 GPU READY，没有运行任务。目标输出必须保留；成功后不再接受后续动作。

model 动作耗尽但未成功：incomplete_mapping。某状态没有合法后继且非成功：死路；它不能单独证明整个问题无解。mapper 只有穷尽所有可达状态后才能判 infeasible。

## 7. 精确搜索与最优性

采用 uniform-cost / Dijkstra 搜索。COPY、COMPUTE、EVICT 的边成本为 0；ADVANCE 的边成本为 Δ。优先队列按累计模拟时间排序，并以递增序号稳定打破平局。

初始化 → 取出最小代价状态 → 跳过过期条目 → 检查 goal → 枚举所有合法动作 → 调用 engine 转移 → 更新 best_cost、父节点与优先队列。

状态键必须包含：完成/运行操作、GPU 副本状态、各资源上任务 ID 与剩余时间、仍有效的分配和 workspace/读锁信息，或这些量的等价充分表示。以稳定 ID 排序构造键，不用 Python 对象地址。

在固定成本、无外部时变事件的 M0，绝对 t 是路径代价，不进入等价状态键；运行任务以剩余时间表示。历史 trace、累计搬运量、历史峰值不影响后续合法动作，作为路径统计保存，不进入键。

仅当新累计时间严格小于同键 best_cost 时更新。等成本同键保留稳定的首条路径，避免零成本重复。不能只按 DONE 集合或驻留集合去重。

峰值与字节不作为次级优化目标；等 makespan 计划可能有不同峰值/字节。以后增加多目标必须重新设计支配关系，不能沿用单个 best_cost 后声称所有指标最优。

第一版不加入未经证明的剪枝、beam search 或强制 ASAP 调度。统一成本搜索先返回被取出的第一个 goal，此时可证明在指定模型/动作空间内最优；并非真实硬件或任意调度空间的全局最优。

max_expanded_states=0、wall_time_limit_s=0 表示无限制；实际 wall time 用单调时钟，不能与模拟 t 混淆。预算在扩展状态前检查，已取出的 goal 可正常确认。预算不足时：有已发现可回放 goal 则 feasible/optimality_proven=false；无 goal 则 unknown/optimality_proven=false。不能判 infeasible。

## 8. 接口与产物

建议内部最小接口：load_and_validate、initial_state、legal_actions、transition、is_goal、state_key、evaluate_mapping、search。名称可调整，但语义不能分叉。

计划命令（实现后才能运行）：

```text
python -m tensor_mapping model --scenario examples/chain-cap160.json --mapping examples/chain.mapping.json --out outputs/chain-replay
python -m tensor_mapping search --scenario examples/chain-cap160.json --out outputs/chain-search
python -m unittest discover -s tests
```

model 与 search 都产出：

- `stats.json`：mode、status、reason、scenario/workload 指纹、成本来源、makespan_ns、peak_vram_bytes、h2d_bytes、动作数量、限制与假设。
- `events.json`：动作序号、动作类型、关联 ID、资源、开始/结束时间与字节数；同刻多个动作保留序号。
- `states.json`：初始及每次动作后的可核对状态摘要；避免将启动时间错误地当作任务完成时间。
- search 另外产出 `mapping.json`、expanded/visited state 数、wall time、termination_reason、optimality_proven。

status 区分：固定计划 valid / invalid_mapping / incomplete_mapping；搜索 optimal / feasible / infeasible / unknown；输入损坏或未支持语义为 invalid_input / unsupported。初始布局超过容量时，两种入口均返回固定初始条件下的infeasible，明确指出initial_capacity_exceeded；它不同于JSON/shape错误。

model 的 valid 不表示最优。失败输出包含动作索引、模拟时间、错误码和足够定位问题的资源/张量信息，例如 INPUT_NOT_READY、RESOURCE_BUSY、CAPACITY_EXCEEDED、TENSOR_IN_USE、LIVE_VALUE_LOSS。

成功结果可有文本摘要。首版不接前端；后续 adapter 将整数纳秒事件转换为已有查看器报告并标记 simulated，图来源与合成成本分别披露。

## 9. 实施顺序与验收门槛

| 子阶段 | 工作 | 交付门槛 |
|---|---|---|
| M0.1 | GGML 三种小图构造/导出、输入契约和校验 | 导出图可追溯；叶子、shape、字节、依赖正确 |
| M0.2 | State、四种动作、固定 mapping 回放 | ACCEPTANCE 的链式手写计划、容量和错误用例通过 |
| M0.3 | uniform-cost mapper、状态去重、父链回放 | 三种图的最优时间、无解与预算语义通过 |
| M0.4 | CLI、产物、可复现构建说明和交付清单 | 从干净输出目录复跑；映射独立回放一致 |

核心测试可使用与契约一致的固定 JSON fixture，但至少另有 GGML 编译→实际导出→模拟/搜索的集成验证。缺少 GGML 构建环境时不能将核心测试通过写成整体验收通过。

语义测试是必须交付；不为文件结构等机械细节写大量镜像测试。不追求一次搜索大图；保留展开状态数以观察规模增长。

后续 M1 加入真实算子/复制校准、CPU 计算与双向传输时，应版本化扩展规则，并重新检查状态等价、资源竞争和执行适配。M0 的整数固定成本模型不直接宣称可预测任意异构硬件。

## 10. 参考和已有资产

- Timeloop 本地分析及 mapper 文档：workload/architecture/mapping 与 evaluator/search 分离；搜索停止条件影响最优性。
- `../modeling/llm_infer_model/simulator.py`：事件、在途容量和 trace 的参考；不继承其固定层序与经验补偿。
- `../modeling/llm_infer_model/tensor/`：现在也是**本目录的核心依赖**（不再是「参考」）——`spec` / `engine` / `layer_costs` 里的状态转移与显存记账是全仓库唯一一份，`mapper` 与 `policies` 都导入它。
- `../modeling/llm_infer_model/model.py`：`LayerSpec.compute_seconds` / `transfer_seconds` 是成本桥接的求值入口；窗口、KV、控制开销等旧服务不在本轮范围内。
- `../../visualization-llama.cpp-tensor/collector/llama-export-graph-json.cpp`：张量与输入关系提取的参考；注意 scheduled 图与逻辑图差异。
- `../../llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml/include/ggml.h`：当前本地接口依据；实际构建必须核对所锁版本。

以上均为参考，不要求在首版复制或修改这些项目。
