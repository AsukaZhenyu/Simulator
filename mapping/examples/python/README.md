# 用 Python 模块描述一个策略

新增执行变换例子：`execution_transforms_demo.py` 在同一策略中组合 Attention 阶段、Dense/MoE FFN 分片与权重预载；`execution_phase_demo.py` 演示持续 Prefill→Decode 的设备放置与 KV 搬运。用法和边界见 [执行变换指南](../../docs/EXECUTION_TRANSFORMS_GUIDE.md)。下面的 `modular_*` 示例保留，用来理解最基础的模块放置。

这两个小例直接读取已有模型、成本和设备事件文件，用 `ModelPlan` 与 `Scheduling` 创建策略。它们不以旧的 `strategy.json` 为入口，也不重新实现模型数学结构。全部时间、带宽和容量均来自明确的**合成夹具**，不能解释为本机实测或加速结果。

## 运行

从仓库根目录运行。使用 Python 3.12，先使 modeling 与 mapping 两个源码包可导入。PowerShell 示例：

```powershell
$taskRoot = (Get-Location).Path
$env:PYTHONPATH = "$taskRoot\Simulator\mapping;$taskRoot\Simulator\modeling"

python Simulator/mapping/examples/python/modular_dense.py --out Simulator/mapping/outputs/python-dense
python Simulator/mapping/examples/python/modular_moe.py --variant opaque --out Simulator/mapping/outputs/python-moe-opaque
python Simulator/mapping/examples/python/modular_moe.py --variant swiglu_staged --out Simulator/mapping/outputs/python-moe-staged
```

也可以在自己的开发环境将两个包分别安装为 editable，然后运行同样的脚本：

```powershell
python -m pip install -e ./Simulator/modeling -e ./Simulator/mapping
```

`--out` 必须是新目录或空目录，已有文件会导致报错。脚本不覆盖结果目录。终端退出码 `0` 表示模拟有效，`2` 表示输入或编译失败，`3` 表示运行未达到有效完成状态。模拟失败时仍可检查已经写出的诊断产物。

## 先看 Dense

`modular_dense.py` 使用两层、Prefill 加一轮 Decode 的 `dense.global.json`。每层都有 Attention 与 FFN 两个策略模块。

```python
for layer_id, device in (("denseL0", "cpu"), ("denseL1", "gpu")):
    DecoderPlacement(device)(plan, name=layer_id, select=Select(layer_id=layer_id))
```

Attention 与 projection 都交给 GPU。第 0 层整个 FFN 交给 CPU，第 1 层整个 FFN 交给 GPU。选择同时作用于该层的 Prefill 与 Decode。这里演示**模块放置**，没有把同一个 FFN 切成 CPU/GPU 两片，因此不属于 TP。

模型原本的计算依赖、KV 与跨轮门控由 modeling 保留。`with plan.module(...)` 增加选择范围与来源路径，本身不增加执行屏障。全局 `Scheduling` 规定后备放置、LRU 缓存、下一层预取和 CE 搬运。32768 字节缓存预算是本合成小例的明确策略参数，不能拿它代表真实模型需求。

## 再看 MoE

`modular_moe.py` 提供两种已经有精确夹具成本的执行方式：

| 参数 | 模型输入 | 专家描述 |
|---|---|---|
| `opaque` | `moe.global.json` | 每个已选专家作为一个 `expert_ffn` 任务 |
| `swiglu_staged` | `moe-fragments.global.json` | 使用模型逐 scope 绑定的 execution 文件，展开为 gather、gate/up、SwiGLU、down、scatter 与 combine |

两种方式都把第 0 层专家模块放 CPU、第 1 层放 GPU，其余模块放 GPU。展开方式使用各阶段自己的成本，没有把 opaque 时间按比例拆开。它们是不同的合成执行合同，不能仅凭两次完成时间比较谁在真实硬件上更快。

全局缓存按 `expert_group` 管理，预算明确为 2048 字节。预取最多查看下一层，按**已公开历史频率**排序，每次源层调用最多准备两个候选。设备事件文件明确声明 CPU 暂不可用和额外内存占用及其恢复。共享一个 runtime 处理容量、驻留、版本、时钟与搬运竞争。

`RevealedExpertHistory` 是最小动态 Python 扩展示例：

```python
def decide(self, observation):
    for route in observation.route_history:
        # 仅统计现在已经揭示的路由，按层、请求、位置去重
        ...
    return DEFER
```

它保存已经激活过哪些专家的计数，供查看动态策略接口使用；实际放置仍由上面的固定计划完成。它不做预测，也不产生新的专家放置策略。`DEFER` 继续使用明确声明的后备调度，`None` 会停止运行。三种 override 标志都显式设为 `False`。以后需要产生 `Decision` 时，仍要通过现有编译与运行时检查，不能直接修改模拟器状态。Python 回调是可信代码，不是隔离沙箱。

路由 trace 属于 modeling 的输入，policy 只接收已经揭示的 `observation.route_history`。示例回调不读取模型文件和私有未来路由。回调源码自动纳入 fingerprint，两个示例还显式将自身与公共辅助脚本纳入来源记录。

## 对比两种门控模式

```powershell
python Simulator/mapping/examples/python/modular_dense.py --mode scope_barriers --out Simulator/mapping/outputs/python-dense-scope
python Simulator/mapping/examples/python/modular_dense.py --mode data_dependencies --out Simulator/mapping/outputs/python-dense-data
```

新建 Python 计划默认 `data_dependencies`。它放宽同一轮层间的整 scope 可见性屏障，继续检查真实数据依赖。跨轮 token 门控与 MoE 路由揭示条件仍然存在。`scope_barriers` 保持保守整 scope 门控。两种模式得到相同时间也正常，差异取决于图、规则与资源，不能承诺一定加速。

这还不是通用微批流水接口，模块间部分张量就绪、任意 TP、独立 P/D 服务实例都没有通过这两个例子实现。旧文档通过 `ModelPlan.from_strategy(...)` 导入时仍默认 `scope_barriers`，以保留旧行为。

## 查看什么

脚本调用 `CompiledPlan.write_result()`，输出中可重点查看：

- `plan.json`：模块路径、选择条件、固定绑定与后端合同。
- `summary.json`：是否有效、完成时间、来源 fingerprint 与能力边界。
- `policy_audit.json`：全局缓存、预取及动态 Python 状态。终端也打印 MoE 的 `revealed_route_count` 与 `expert_counts`。
- `decisions.json`、`events.json`、`states.json`：检查计算如何触发数据准备，以及实际占用和并发。
- `compiled_graph.json` 与 `persistent_state.json`：展开后的图和最终持续状态。

缺少成本、非法绑定、变化的来源文件都会明确失败。这些示例证明的是声明能经过共同 Compiler/Runtime 执行，不证明硬件预测准确或策略最优。
