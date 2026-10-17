# 完整声明 Dense block 的测量与 M2 对接

本入口测量 CPU、B=1 的一个明确 decoder block 变体。模型和测量共用 `modeling/llm_infer_model/m2/dense_contract.py`，不从模型名称补参数。`m2-layer-0.3` 从 hidden activation 和 I32 绝对位置开始；旧 0.1 合成链、0.2 部分 block 保留原语义。

## 三个任务与一次组合测量

| stage | 计时包含的计算 | 显式输出 |
|---|---|---|
| projection | RMSNorm/scale、独立 Q/K/V projection、可选 gate projection 与 Q/K head norm、normal RoPE | F32 Q/K/V、可选 gate |
| attention | KV 追加、QKᵀ/scale/causal mask/softmax/PV、可选 sigmoid gate、O projection、第一条 residual | attention_hidden、新 KV 槽 |
| dense_ffn | RMSNorm/scale、gate/up、SiLU 与乘法、down、第二条 residual | hidden_out |
| block | 上述三个阶段的组合图，保留相同的物化输出 copy | 所有阶段输出 |

位置为 `history ... history+q-1`，RoPE 参数按 GGML F32 的实际值绑定。normal RoPE 是相邻元素配对；Neox/其他位置变换明确拒绝。权重矩阵逐个声明 F32/F16/Q4_K/Q6_K，norm gamma 固定 F32。hidden 宽度不要求等于 heads×head_dim。

独立 attention/FFN 在计时前，用相同 seed 执行上游组合图准备真实边界输入；每阶段仅分配该阶段的 resident 输入、输出和权重。计时图的 setup 一次、warmup 次数、额外上游准备次数分别保存，不能混称总 warmup。每个样本运行一次完成的 CPU 图，使用持久 GGML threadpool。

组合图用于核对三段成本之和，不导出为第四个可调度任务。导出要求同形状四个 stage 齐全：几何、dtype、线程、seed、native executable、操作清单和运行协议一致。全量端口 FNV1a64 字节摘要核对输入来源和对应输出；文件/二进制另使用 SHA256。FNV 摘要是运行一致性检查，不是独立数值正确性证明。

## 数值与内存边界

输入和权重由明确的确定性生成器产生。独立参考使用实际存储权重的解量化值、double 累加和已声明的 F32 算子边界，减少精确二进制测试数据掩盖错误的问题。小输出全量比较；大输出明确保存抽样覆盖范围、误差、容差及判据。K/V 新槽与全部历史内容始终全量校验，历史内容不得被覆盖。

测量记录 resident logical inventory、实际 resident allocator 容量、内部 gallocr 容量与 CPU cplan scratch。M2 收费边界为独立逻辑 resident 张量加运行中宏任务的 internal scratch；新 K/V 输出和 cache 追加槽分开收费。allocator padding、线程栈、host 参考数据和其他运行时开销不等于零，它们不在这个逻辑容量模型中。

联合图可能重用内部 allocator，不能把联合 workspace 直接当作三个可独立调度任务的 workspace。M2 用各自测得的 scratch，CPU 串行槽按运行任务计费。所有后续调用需要的历史 KV 都是持久交付目标，不能靠驱逐旧历史虚增可用容量。

## 构建和复用

在 Simulator 根目录运行；公共头、链接库和运行 DLL 必须来自匹配 GGML 构建。路径占位符须替换为实际目录；构建使用新目录，避免覆盖冻结实验的程序。

```powershell
cmake -S profiling/native -B "<new-CPU-build-directory>" -DGGML_SOURCE_DIR="<GGML source>" -DGGML_BUILD_DIR="<existing build>" -DMEASURE_ENABLE_CUDA=OFF
cmake --build "<new-CPU-build-directory>" --config Release --target ggml-dense-layer-probe
python profiling/run_dense_layer_measurements.py --suite profiling/cases/dense_layer_cpu_smoke.json --native-bin "<CPU-probe-directory>" --runtime-dir "<matching GGML DLL directory>" --threads 2 --warmup 5 --samples 20 --seed 17 --output profiling/results/my-dense-smoke
```

Linux 单配置构建需显式设置 `CMAKE_BUILD_TYPE=Release`，`--native-bin` 指向实际 executable 目录。共享库位置用 `--runtime-dir` 传入，不能假定系统自动找到。`--dry-run` 只输出计划，不测硬件；测量共享 `profiling/.measurement.lock`，输出目录必须为空，失败也保存 raw。

真实几何套件为 `cases/dense_layer_qwen35_geometry.json`；对应模型描述符为 `mapping/examples/m2/qwen35-geometry.declared-block.cpu.layer.json`。小尺寸对应 `dense-block.cpu.layer.json`。这两个描述符显式绑定 threads=2；改线程时须同步修改物理描述符并重新测量。

```powershell
python profiling/export_dense_layer.py --measurements profiling/results/my-dense-smoke/measurements.json --layer mapping/examples/m2/dense-block.cpu.layer.json --output profiling/results/my-dense-export
```

默认拒绝 CV>0.1 的数据，不删样本。明确使用 `--allow-unstable` 才能导出验证表，调用还必须声明 `allow_validation_only_costs=true`。`--phase prefill` 是显式选择，遗漏记录保留在报告里。未知 workspace 可以保留为未知，不能用于成本表或容量可行性判断。

`export_report.json` 保存阶段中位数之和、独立联合图中位数、带符号/绝对相对误差、各自 CV、端口核对、原始文件与 source 身份。未传 `--max-relative-error` 时只报告误差，不暗设验收阈值；显式传入才执行误差门。`--allow-composition-error` 的豁免也只能用于验证。

`validation_only=false` 仅表示没有质量豁免，不代表完整层已校准。当前报告始终 `calibration_claimed=false`。精确查表不插值，不跨 history 复用 FFN，不用孤立 matmul 替代任意宏任务。

安装 modeling 后可复用既有 `python -m llm_infer_model.m2 --invocation ... --output ...`，保存 workload/scenario、事件、容量、KV 状态和来源身份。回放不测量硬件、不搜索策略。

## 范围与后续

这是完整**已声明变体**，不是原始预训练模型的执行复现。Qwen 几何案例采用独立 Q/gate projection、F32 KV、显式 causal attention 和 normal RoPE；不代表原模型的交错 Q/gate、FlashAttention、F16 KV 或其位置变换实现。分阶段缓存、图启动和联合执行的差距必须通过报告观察。

本测量入口仅覆盖单层 CPU/B=1，不覆盖跨层、GPU/搬运重叠、ragged batch 或其他模型族。跨形状与组合误差需要独立验证，不能由功能通过推出校准通过。其他执行边界和重新采集步骤见[执行变换工具](execution_calibration/README.md)；当前完整性能质量门尚未通过，本机历史结果不随仓库分发。
