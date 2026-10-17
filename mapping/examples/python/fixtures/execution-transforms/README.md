# 执行变换组合示例

入口：`../../execution_transforms_demo.py`。默认依次运行 Dense、MoE，结果各写入
一个子目录。这里所有阶段和传输价格均为**合成夹具**，不表示硬件实测或加速比。

从项目根目录运行：

```powershell
$taskRoot = (Get-Location).Path
$env:PYTHONPATH = "$taskRoot\Simulator\mapping;$taskRoot\Simulator\modeling"
python Simulator/mapping/examples/python/execution_transforms_demo.py --case both --out Simulator/mapping/outputs/execution-transforms
```

输出目录必须不存在或为空。`--case dense` / `--case moe` 可以单独运行。

## Dense：一个计划同时描述四种选择

1. Attention 拆成 QK、softmax、PV、output，保留真实的 QKV 和 KV 端口。
2. QK/softmax 在 CPU，PV/output 在 GPU。
3. FFN 中间维度 I=32，CPU 执行 `[0,8)`，GPU 执行 `[8,32)`，部分输出在 GPU 求和。
   Dense RMSNorm 与 residual 各一次。
4. 计算前将 GPU FFN 分片权重预载到显存，复制照常消耗时间、容量和搬运资源。

示例的关键代码是：

```python
plan.transform_file("dense-decode", attention_cost_file)
plan.transform("dense-decode", _ffn(ffn_cost_file, cpu_end=8, intermediate=32))
plan.place("attention_qk", "cpu")
plan.place("attention_softmax", "cpu")
plan.place("attention_pv", "gpu")
plan.prepare(DataPreparation(tensor_ids=gpu_ffn_weights,
                             destination="vram", trigger="startup"))
```

`_ffn` 中直接使用 Python 的 `FFNShard` 描述分片，阶段成本从独立文件读取。
改变分片位置时，新宽度必须有对应的精确成本；不能自动按比例换算。
改变分片数量还必须显式更新外层 `merge_input_count` 与对应合并成本合同；
仅替换 `shards` 而保留旧输入数量会直接拒绝。示例明确使用两片和 count=2。

## MoE：三种执行变换组合，专家复用同一 FFN 变换

先应用 selected rows 与独立 gate/up/down 矩阵表示，再展开 Attention，最后将
每个专家 I=8 分成 CPU `[0,2)` / GPU `[2,8)`。Attention 同样由 GPU 投影、CPU
QK/softmax、GPU PV/output 协作。gather、路由揭示、scatter、combine 继续由
modeling 管理；FFN 变换只处理各专家已经选中的紧凑行。专家 0 的 GPU 分片在启动时准备，
专家 3 的权重允许在资源空闲时尝试准备。这两个专家号都是固定声明，不读取
本次或未来路由来决定。

该 MoE 是新的、明确尺寸的合成工作负载：`W_proj` 是 192 B 的 F32 Q/K/V/O
完整权重包，H=4、2 个 query heads、1 个 KV head、head_dim=2；每个专家的
gate/up/down 合计 384 B。输入和 `state_in` 都同步声明这些实际字节数，并给出
192 B 的精确搬运成本。它结合两个已有夹具的物理声明，不能声称旧 32 B opaque
projection 自动具有 QKV，也不能将新合成成本称为原论文或真实模型复现。
完整权重包在 projection/output 所选设备上整体计费，没有免费矩阵别名。

## 文件来源与边界

- `dense.global.json` 直接引用已有完整 F32 hybrid Dense Decode 输入。
- `moe.global.json` 引用本目录 `moe.prefill.json`、`moe.layer.json`、`moe.costs.json`；
  它们基于已有明确三矩阵尺寸的 MoE 输入，新增显式 192 B QKV/O 包。
- `dense.attention-stages.json` 来自 Attention 分阶段验收夹具。
- `dense.ffn-costs.json` / `moe.ffn-costs.json` 来自共享 FFN 变换验收夹具。
  Dense 中与 Attention 已声明的同路线/字节数搬运键共用同一份价格，去掉重复条目；
  编译器遇到两个不同价格仍然报错，不静默覆盖。
- `moe.selected-rows.json` 来自已有 MoE 子权重验收夹具。
- `moe.attention-stages.json` 来自 Attention 的明确 QKV/O 包验收夹具；三 pass
  遇到同路线/字节数时复用前面已声明的价格，不覆盖为另一种数字。
- `environment.json` 显式声明空设备事件，不引入未说明的后台负载。

成本文件与 Python 源码均纳入编译来源和 fingerprint。该脚本不比较真实性能，
不提供独立 P/D 服务队列，也没有策略搜索算法。要检查组合是否忠实执行，重点看
输出的 `compiled_graph.json`、`events.json`、`decisions.json` 与
`persistent_state.json`：其中应保留每个新阶段、跨设备搬运和实际叶子权重驻留。
