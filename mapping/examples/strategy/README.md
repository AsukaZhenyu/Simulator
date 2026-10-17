# 手写策略 → 编译 → 性能回放

首版将 mapping 分成 Strategy IR 和 Compiler；执行、时间、依赖和容量仍使用 modeling 的同一份规则。这里的示例都采用**合成成本**，用于验证策略表达，未运行硬件、搜索或宣称全模型性能准确。

从 `Simulator` 目录，使用已经安装 modeling/mapping 的 Python 环境：

```powershell
python -m tensor_mapping.strategy mapping/examples/strategy/prefetch.strategy.json --out mapping/outputs/ir-prefetch
python -m tensor_mapping.strategy mapping/examples/strategy/stateful-three-step.strategy.json --out mapping/outputs/ir-stateful
python -m tensor_mapping.strategy mapping/examples/strategy/m2-moe-prefill.strategy.json --out mapping/outputs/ir-moe
```

输出目录须不存在或为空。退出码 0 表示 valid；3 表示程序执行被拒；2 表示输入/编译/输出错误。失败回放保留执行前缀和原 IR 指令位置，不自动补齐等待、搬运或驱逐。

## 从一个简单方案看起

`prefetch.strategy.json` 的方案是：GPU 计算第一层，同时加载第二层权重；等两者完成后，释放第一层权重的 VRAM 副本，再计算第二层。

```json
[
  {"kind": "compute", "operation_id": "layer0"},
  {"kind": "transfer", "tensor_id": "W:L1", "src": "dram", "dst": "vram"},
  {"kind": "wait", "operations": ["layer0"], "copies": [{"tensor_id": "W:L1", "location": "vram"}]},
  {"kind": "release", "tensor_id": "W:L0", "location": "vram"},
  {"kind": "compute", "operation_id": "layer1"},
  {"kind": "wait", "operations": ["layer1"], "copies": []}
]
```

placement 明确指定两个操作都在 GPU。IR 的 region 可以把两层“打包”成一个展示分组；它**不增加屏障**。图中的实际 operation 决定最细执行粒度。想拆 Attention/FFN，需要先提供拆分后的真实端口、操作图和对应成本，首版不会自动拆分或分摊整体价格。

## 七个可运行例子

| 文件 | 验证内容 | 预期时间 | DRAM/VRAM 峰值 |
| --- | --- | ---: | ---: |
| `prefetch.strategy.json` | 跨层权重预取与显式释放；5 ms 计算和3 ms搬运重叠。 | 7 ms | 128/176 B |
| `ssd-prefetch.strategy.json` | SSD→VRAM 经 DRAM 两跳；2 ms SSD读与3 ms H2D在第一层计算期间完成。 | 7 ms | 128/176 B |
| `router-experts.strategy.json` | router 完成后才能加载专家；独立 CPU/GPU 分支并发。 | 6 ms | 128/128 B |
| `stateful-three-step-static.strategy.json` | 同一图、两层三步，显式 KV 版本依赖，保持全局时钟与副本。 | 24 ms | 256/176 B |
| `stateful-three-step.strategy.json` | 首步结束于8 ms，切换完整GPU成本表及内存容量/reserve；后两步各12 ms。 | 32 ms | 256/256 B |
| `m2-hybrid-prefill.strategy.json` | 现有 Dense M2宏接口，显式CPU/GPU/CPU；导出真实KV/hidden状态。 | 3,100 ns | 11,756/2,432 B |
| `m2-moe-prefill.strategy.json` | 现有 MoE M2接口，固定CPU；导出KV及实际READY专家副本。 | 34 ms | 496/0 B |

前三个通用图及三步 KV 图改编自已冻结的独立固定计划，新增文件不修改旧验收资产。三步图使用 ADD 表达依赖与状态生命周期，**不等于真实 Attention**。两个 M2 IR 是对明确给定 placement 的串行 baseline 转写，未枚举或选择设备。

动态案例完整声明新的成本，不用“GPU忙了所以乘二”的隐式估计。切换后 VRAM容量256 B、runtime reserve80 B，已有副本继续占据实际容量。容量降到143 B时边界立即拒绝；降到255 B时边界可通过，但第三步GPU计算峰值256 B被拒。运行中切换也拒绝，不自动等待。

## 看哪些输出

- `strategy.json`：规范化 IR；路径引用仍相对原输入目录。它是输入快照，不保证复制到输出目录即可独立加载。
- `compiled.json`：实际降低后的程序、region、placement、场景指纹及输入文件SHA256。
- `execution.json`：每个物理动作对应的 IR 指令序号和当前成本场景；SSD两跳及wait展开均可追溯。
- `events.json` / `states.json`：执行区间和动作后的容量/副本账；`action_profiles`标明实际场景。
- `environments.json`：边界设备状态切换时刻与新容量占用。动作快照之外的切换记录在这里。
- `summary.json`：时间、流量、峰值、拒绝原因和错误IR位置；明确 `search_performed=false` 与 `physical_calibration_claimed=false`。
- `state_out.json`：仅M2成功调用导出。Dense旧合同保存KV/hidden；MoE还保存权重版本与READY位置。

下一次M2调用必须显式采用上一份实际state_out的请求/块记录，并匹配position、版本和精确成本键。Dense旧state_out没有权重驻留，新入口目前不提供跨独立Dense调用的完整缓存续接；共享DAG内部可持续保留权重，旧 `m2_schedule_cli` 的boundary sidecar能力仍保留。测试已覆盖Dense和MoE Prefill→Decode，MoE Decode使用实际已加载专家副本，SSD读取为0。

**当前边界：**固定离线程序、预展开有限多层多步图、空闲边界设备状态变化。开放式生成、真实全LLM前端、在线规则、算子任意切分/融合、运行中的DVFS/共享带宽变化和真实混合性能校准仍需后续实现。

详细分工见[代码地图](../../../docs/CODE_LAYOUT.md)；固定程序的当前回归位于 `mapping/tests/test_strategy_compiler.py`。
