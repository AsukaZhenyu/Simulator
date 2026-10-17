# 因果 MoE 与运行中设备波动示例

从 `Simulator` 目录执行，需要安装 modeling/mapping 两个包。示例复用 `../m2/moe-interface-synthetic-prefill-2.json`，成本和路由均为合成夹具。不要把下面的34/36 ms解释为真实硬件性能。

```powershell
python -m tensor_mapping.causal_cli mapping/examples/m2/moe-interface-synthetic-prefill-2.json --environment mapping/examples/causal/no-disturbance.environment.json --policy serial_cpu --max-decisions 100 --out mapping/outputs/causal-cpu
python -m tensor_mapping.causal_cli mapping/examples/m2/moe-interface-synthetic-prefill-2.json --environment mapping/examples/causal/cpu-blackout.environment.json --policy serial_cpu --max-decisions 100 --out mapping/outputs/causal-blackout
python -m tensor_mapping.causal_cli mapping/examples/m2/moe-interface-synthetic-prefill-2.json --environment mapping/examples/causal/memory-conflict.environment.json --policy serial_cpu --max-decisions 100 --out mapping/outputs/causal-conflict
```

| 环境 | 预期 |
| --- | --- |
| `no-disturbance` | CPU按需基线完成34 ms |
| `cpu-blackout` | 0.5 ms开始暂停CPU并新增128 B DRAM外占，2.5 ms恢复并释放外占；完成36 ms |
| `memory-conflict` | 0.5 ms新增4096 B外占，叠加112 B张量和16 B workspace超过4096 B容量；外部分配被拒，停止于0.5 ms |

基线依次完成 Projection 1 ms、Attention 1 ms、Router 2 ms、三个专家SSD读共9 ms、专家计算5+10+5 ms、Combine 1 ms，合计34 ms。暂停发生在首Projection中途：其有效计算仍1 ms，实际墙钟区间0–3 ms，总时长增加2 ms。内存冲突不提交4096 B外占，仍保留在途任务、输入锁与输出预留；不是模拟OS接受过量分配。

使用 `--policy serial_gpu --max-decisions 150` 可检查GPU按需基线的显式搬运和KV副本。该基线不按性能搜索设备、不重叠独立工作，也不自动释放缓存；本例有足够容量，不可当成完整卸载论文实现。

环境必须显式提供 `format_version="environment-trace-0.1"` 与 `events`。每条事件必须有非负整数 `t_ns`，并至少声明一项：`dram_external_bytes`、`vram_external_bytes`（非负绝对字节），`cpu_available`（布尔值）。字段省略表示保持原值，不接受null。事件按时间非降序排列；同刻不同字段原子更新，重复赋值同一字段拒绝。外占在已有runtime reserve之外相加；当前成本与workspace固定。CPU占用只用服务暂停窗口表达，未定义任意利用率百分比。

Policy看不到环境文件、未来事件时间、完整trace或未揭示专家组。所有专家权重的静态目录可见，提前猜测预取合法；错误猜测付搬运时间、流量与容量。Router完成后真实路由才公开；由于当前跨请求专家分组的限制，专家/Combine任务需等全部Router完成再公开。详见[共同执行接口](../../docs/UNIFIED_EXECUTION_RUNTIME.md)。

Python调用方式：

```python
from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.tensor.environment import EnvironmentEvent
from tensor_mapping.causal import SerialDemandPolicy, run_causal

session = CausalM2Session.from_file(
    "mapping/examples/m2/moe-interface-synthetic-prefill-2.json",
    environment_events=(EnvironmentEvent(500000, cpu_available=False),
                        EnvironmentEvent(2500000, cpu_available=True)),
)
run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=100)
assert run.evaluation.makespan_ns == 36000000
state_out = session.state_out()  # 只在有效完成后导出
```

自定义有状态策略的 `__call__(observation)` 返回 `Decision` 或None。编译器逐动作检查公开引用和精确价格；Modeling检查物理合法性。回调不得通过闭包读取私有session/trace；这是可信代码API边界，不是安全沙箱。已揭示路由可保留到下一调用作预测依据；不默认生成历史统计或预测器。

每次运行输出新目录（非空目录拒绝覆盖）：

| 产物 | 用途 |
| --- | --- |
| `summary.json` | 状态、时间/字节/峰值、预算、失败位置、实际解析输入SHA；明确无搜索/最优性/物理校准/论文复现承诺 |
| `decisions.json` | 策略当时所见观察、hash、Decision、Action与接受/拒绝证据 |
| `execution.json` | 实际提交动作及对应decision_index；拒绝动作也可追溯 |
| `events.json` | 已接受动作，任务实际墙钟起止与独立名义终点；在途任务实际终点为null |
| `service_intervals.json` | 有效资源服务区间；CPU暂停区间不计有效忙时 |
| `environments.json` | 已发生的环境事件、内存分解、分配尝试是否被接受 |
| `states.json` | 完整图调试状态；明确标记不作为策略输入 |
| `revealed_history.json` | 此调用已经公开的路由历史 |
| `state_out.json` | 仅有效完成时写入实际KV与权重副本/版本 |

退出码0表示有效完成，3表示运行失败/未完成，2表示输入/编译/文件错误。预算结束只说明本次未完成，不表示最优或无解。输出JSON使用LF。输入在解析或导出前被修改会拒绝，不将哈希绑定到另一份字节。

重新运行本页命令可生成结果；因果和环境规则由 `mapping/tests/test_causal_mapping.py`、`test_causal_cli.py` 持续回归。
