# 从卸载论文到 IR：六项工作的快速对照

日期：2026-10-12。先看每篇的“一句话 → IR 片段 → 动作 → 缺口”。本页的原文描述是中文转述；可核对的英文短摘、章节和深入解析放在文末三份原文笔记中。

这里的“支持”指**某些论文机制已经有可执行表达**，不是完整论文复现。七份完整 `.strategy.json` 使用当前 `unified-strategy-0.2`，其中 FlexGen 有两种顺序。所有尺寸、成本与轨迹都是明确的合成小例，不能横向比较论文性能。

## 先用这张表选一篇

| 工作 | 原文主要想解决什么 | 本项目用什么表达 | 当前尚未表达的关键部分 |
| --- | --- | --- | --- |
| KTransformers | CPU 算路由专家，GPU 算共享专家和常用部分 | 固定任务设备 `placements` | 专用内核、量化、Expert Deferral |
| NEO | 把一部分请求的 decode Attention/KV 留在 CPU | 拆纯 Attention + 明确核心设备 | 完整双子批流水与动态组批 |
| Fiddler | 冷专家该搬权重给 GPU，还是把激活送给 CPU？ | 内置 `fiddler_paths` 规则 | 离线热门专家选择、原始报价近似 |
| HybriMoE | CPU、GPU、搬运怎样分工，缓存留谁？ | 内置三队列 + MRS 缓存 | 整层时间线计划、门控预测和收益预取 |
| FlexGen | 显存不够时，怎样按层/批复用权重并换入换出？ | 有限 block 遍历 + 存储位置 + 缓存/预取 | LP、比例切片、压缩与完整原算法流水 |
| ATSInfer | 细分张量分工，减少跨设备等待与临时空间浪费 | 分阶段设备 + CE/SM + 共享链路/槽 | 原始静态/在线 DP、任意逐张量绑定 |

## 五组字段怎么看

| 字段 | 可以直接读成 |
| --- | --- |
| `execution` | 打开的任务包里有哪些小步骤？ |
| `policy` | 谁来计算；看到当前情况后按什么规则决定？ |
| `cache` | GPU 权重缓存计多少空间，需要时先淘汰谁？ |
| `prefetch` | 要不要根据已知信息提前搬后层数据？ |
| `transport` | 这次搬运用哪种引擎，怎样重用空间？ |

模型、初始副本、性能价格和设备服务是另一类输入。比如“E0 已经在显存”由模型初始快照给出；写 `cache.rule` 并不会自动完成论文的离线热门专家选择。

以下 JSON 均为**从真实配置抽取的阅读片段**，省略了成本、尺寸及其他必填字段，不能单独运行。旁边的链接是完整可运行文件。解析路径统一为：

```text
IR 声明 → 按当前公开状态生成 Decision → Compiler 转成 COPY/COMPUTE/EVICT 等动作
       → Modeling 检查依赖、容量和资源，推进时间 → 下一次公开观察
```

策略看不到未来路由。论文里的自动规划器没有实现时，手工写入一个可执行计划，只能证明该计划能表示与执行。

## 1. KTransformers：先理解最简单的 CPU/GPU 分工

**原文描述。** SOSP 2025 论文把路由专家交给 CPU，共享专家和常用计算交给 GPU；另有专用内核及 Expert Deferral 优化。[原论文 §3–4](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)。

**IR 描述。** [完整 KTransformers 示例](../examples/paper-ir/ktransformers.strategy.json)：

```json
{
  "policy": {
    "placements": {
      "projection": "gpu",
      "attention": "gpu",
      "router": "gpu",
      "expert_ffn": "cpu",
      "shared_expert_ffn": "gpu",
      "combine": "gpu"
    },
    "experts": {"device_rule": "fixed", "selection_unit": "expert_region"}
  }
}
```

**人话解析。** router 公布选中的专家后，CPU 算这些路由专家，GPU 算共享专家；CPU 的输出要搬回 GPU 再合并。`fixed` 就是照表执行，不是调用 KTransformers 原系统。

**本例实际检查。** 三个被激活的路由专家在 CPU，共享专家及非专家任务在 GPU；没有把路由专家权重搬上 GPU。

**缺口。** AMX/量化/NUMA 和真实软件开销没有复现。Expert Deferral 会改变专家结果进入后续层的位置，需要改变计算图及验证模型质量；当前保持原依赖的 IR 不能用普通重排表示它。该例没有这项优化。

## 2. NEO：只把 Attention 中的一部分工作交给 CPU

**原文描述。** 一部分请求的 decode Attention 和 KV 在 CPU，另一些在 GPU，两组请求非对称流水，并由在线调度器组批。[原论文 §3.1–3.2](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf#page=4)。

**IR 描述。** [完整 NEO 示例](../examples/paper-ir/neo.strategy.json) 中的两个位置：

```json
{
  "variant": "pure_attention",
  "subbatches": [{
    "id": "selected-cpu-request",
    "request_ids": ["r_cpu"],
    "attention_device": "cpu"
  }]
}
```

这段在 `execution[0].spec`，选择一种已验证的拆包方式。设备表在 `policy`：

```json
{
  "placements": {
    "projection": "gpu",
    "attention_core": "execution_binding",
    "attention_output": "gpu",
    "dense_ffn": "gpu"
  }
}
```

**人话解析。** GPU 生成 Q/K/V → 当前 Q/K/V 送 CPU → CPU 读取历史 KV，做 QK/softmax/PV → context 送回 GPU → GPU 做输出投影和 FFN。历史 KV 留在 DRAM，权重留在 VRAM。`execution_binding` 表示使用上面明确选定的 CPU，而不是猜一个设备。

**缺口。** 当前 `.2` 示例只覆盖一个已选 CPU 请求的 decode 分支。旧 NEO 入口可组织多个独立 B=1 请求，但它们也不等于融合 batch。完整双子批流水、在线准入和组批尚未表示；不能称 NEO 完整复现。

## 3. Fiddler：遇到冷专家时，选哪条完整路线？

**原文描述。** 热专家预先驻留 GPU；router 后根据每个专家的输入规模，让冷专家在 CPU 算，或搬权重后在 GPU 算。原 Algorithm 1 近似忽略较小激活的往返费用。[原论文 §3.2–3.4](https://arxiv.org/html/2402.07033v3#S3)。

**IR 描述。** [完整 Fiddler 示例](../examples/paper-ir/fiddler.strategy.json)：

```json
{
  "policy": {
    "experts": {
      "device_rule": "fiddler_paths",
      "selection_unit": "expert_region"
    }
  },
  "cache": {
    "rule": "none", "unit": "expert_group",
    "budget_bytes": null, "alpha": null, "top_p": null
  },
  "prefetch": {"enabled": false}
}
```

**人话解析。** `fiddler_paths` 调用一个已有规则模块：专家在显存就优先 GPU；否则比較 CPU 和 GPU 完整路径。CPU 路线要付激活送来、计算、结果送回的费用；GPU 路线要付必要的权重搬运和计算费用。`expert_region` 表示按完整专家区域选择一次。

**原文与当前表达的差别。** 完整激活往返计费是本项目扩展，不能说与原阈值公式逐字一致。初始 E0 驻留由 [模型快照](../examples/paper-ir/moe-dram.invocation.json) 人工指定，离线热门专家选择尚未实现。例子关闭动态缓存和预取，避免把其他优化算成 Fiddler。

**本例实际检查。** E0 用已有权重在 GPU 算，E1/E2 搬权重后用 GPU；CPU 路线报价也记录在审计中。本例没有选中 CPU 路线，不代表 CPU 路线不可表达。

## 4. HybriMoE：三个工作队列，加一张缓存评分表

**原文描述。** GPU 优先大负载的已缓存专家；CPU 优先小负载的未缓存专家，空闲时可接缓存专家；搬运优先大负载的未缓存专家。MRS 用近期路由评分管理缓存；原文另外制定整层时间线计划，并预测后续层专家。[原论文 §IV-B–IV-D](https://arxiv.org/html/2504.05897v1#S4)。

**IR 描述。** [完整 HybriMoE 示例](../examples/paper-ir/hybrimoe.strategy.json)：

```json
{
  "policy": {
    "experts": {
      "device_rule": "hybrimoe_queues",
      "selection_unit": "expert_region"
    }
  },
  "cache": {
    "rule": "mrs", "unit": "expert_group",
    "budget_bytes": 176, "alpha": 0.5, "top_p": 4
  },
  "prefetch": {"enabled": false},
  "routing_scores_file": "../papers/router.scores.json"
}
```

**人话解析。** `hybrimoe_queues` 调用三组优先级；`mrs` 更新专家缓存评分；`alpha=0.5` 表示新旧评分各占一半；`top_p=4` 是专家个数，本例只有四个专家。完整分数由 Modeling 持有，router 完成才公开；不能用只知道的 top-k ID 猜全体分数。

**本例实际检查。** 有两次 MRS 更新与队列决策。因为当前贪心规则只对已经就绪的任务作决定，本例最后三个专家都在 CPU，未发生权重提升或缓存驱逐。

**缺口。** 三队列规则没有等同于原论文的整层时间线分配。未来三层门控预测与预取收益排序未实现，因此示例关闭预取。当前历史预取可以另作扩展，但不能叫原论文的 impact-driven prefetching。

## 5. FlexGen：同一层多算几批，复用同一份权重

**原文描述。** generation step → layer → GPU batch 的块遍历，在 GPU/CPU/磁盘之间存放并访问权重、激活和 KV，安排计算与 I/O 重叠；论文还优化比例和 batch 配置。[原论文 §4.2–4.3、Algorithm 1](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf#page=5)。

**IR 描述。** [层优先 IR](../examples/paper-ir/flexgen-layer-major.strategy.json) 的引用是：

```json
{
  "model": {
    "kind": "block",
    "file": "flexgen-layer-major.model.json",
    "override_model_policy": true
  }
}
```

引用的 [block 模型](../examples/paper-ir/flexgen-layer-major.model.json) 中 `order="layer_major"`；另有 [批优先对照](../examples/paper-ir/flexgen-batch-major.strategy.json)，引用 `order="batch_major"`。两个例子都使用同一组生成次数、三层、两批、价格、缓存和预取配置。

```text
层优先：L0/B0 → L0/B1 → L1/B0 → L1/B1 → L2/B0 → L2/B1
批优先：L0/B0 → L1/B0 → L2/B0 → L0/B1 → L1/B1 → L2/B1
```

**人话解析。** 层优先时，L0 权重装好以后连续供两批使用，再换 L1。权重从 SSD 读到 DRAM，再搬到 VRAM；KV 和激活有实际存储/交付动作。`override_model_policy` 明确让当前 IR 的缓存/释放规则接管旧固定搬运程序，计算遍历仍遵守模型声明。

**重要接缝。** 遍历 `order` 与张量 `home` 仍在模型输入，不是独立的 mapping 字段。现在能通过模型引用表达两个顺序，但尚不能说所有 FlexGen 策略维度都与不可变模型分离了。

**缺口。** 有限 block 为合成层任务，不是完整 LLM 算子前端；没有原 LP、比例切片、压缩或完整原算法流水。SSD 写回后 DRAM 副本仍保留，不能声称完整紧内存换入换出。

## 6. ATSInfer：工作拆细，搬运也分工

**原文描述。** 按张量安排 CPU/GPU，权重走 CE、激活走 SM，二者共享 PCIe；临时 GPU 空间按生命周期复用，计划由静态及在线 DP 产生/更新。[原论文 §4.2–4.4](https://arxiv.org/html/2607.10183v2#S4)。

**IR 描述。** [完整 ATSInfer 示例](../examples/paper-ir/atsinfer.strategy.json) 选择 staged 专家；设备表的一部分：

```json
{
  "policy": {
    "placements": {
      "expert_gate": "cpu",
      "expert_up": "gpu",
      "expert_swiglu": "gpu",
      "expert_down": "cpu"
    },
    "experts": {"device_rule": "fixed", "selection_unit": "expert_region"}
  }
}
```

搬运规则的一部分：

```json
{
  "transport": {
    "rules": [
      {"role": "weight", "stage": null, "src": "dram", "dst": "vram", "engine": "ce"},
      {"role": "intermediate", "stage": null, "src": "vram", "dst": "dram", "engine": "sm"}
    ],
    "slot_reuse": "release_unlocked_backed_copy"
  },
  "services_file": "atsinfer.services.json"
}
```

**人话解析。** 根据声明，CPU 算 gate/down，GPU 算 up/激活；边界数据真实搬运。CE 是专用搬运引擎，SM 搬运会使用 GPU 计算资源，二者不会免费各得一条全速 PCIe。[服务文件](../examples/paper-ir/atsinfer.services.json) 给八个跨层 up 权重定义一个共用槽，旧读者完成、旧副本释放后才能换 owner。

**本例实际检查。** 全专家初始在 DRAM；分阶段设备、权重 CE、激活 SM 和跨层槽释放都在动作里，无 SSD 读写。这个人为计划用于展示表达能力，不是原 DP 的输出，也不代表论文要求此固定拆法。

**缺口。** 当前按任务种类放置，同一种类跨层共享规则；任意单张量独立放置、原静态/在线 DP 和近期测量后的重新规划没有实现。当前 SM 使用保守 GPU 资源合同，启动开销和真实并发仍待校准。

## 从六篇对照里，应该怎样理解现在的 IR

你可以直接写出的已有选择包括：合法拆包方式、任务设备、权重缓存规则、因果预取和搬运规则。Compiler 能把这些选择翻译成明确动作，并由 Modeling 检查依赖、容量和资源。

有两种尚未通用化的内容：

1. **规则模块仍有论文机制名字。** `fiddler_paths`、`hybrimoe_queues` 是调用已实现的模块，内部公式/队列顺序并未展开成任意可编程表达式。换成全新规则通常仍需开发模块。
2. **部分策略信息仍嵌在模型输入。** 初始驻留、block 遍历及 home 没有完全独立到 mapping IR。统一入口完成了，所有策略维度的统一抽象仍需继续。

因此，这次可以逐项检验“论文机制怎样表示、动作是否按声明执行”，并据此找抽象缺口。它没有补全各论文的自动优化器，也没有证明真实加速。

当前可运行示例名单是以上六项机制。PowerInfer 等其他已调研工作尚无本次同等级的独立论文策略示例；LLMServingSim 是评价平台，本页不把它当成一个卸载策略来表示。

## 运行和检查

- [完整 IR 目录与运行命令](../examples/paper-ir/README.md)
- [可组合执行变换](EXECUTION_TRANSFORMS_GUIDE.md)
- `mapping/tests/test_paper_integration.py`、`test_paper_policy.py` 持续核对声明与动作。

当前文件保留论文机制与表达范围，不再依赖旧 research 或 validation 目录；合成小例不能用于论文之间的真实性能排名。
