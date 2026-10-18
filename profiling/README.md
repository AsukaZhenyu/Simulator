# 可复用的 GGML 与 CUDA 测量工具

Performance calculation belongs to modeling. The primary integration contract is
[physical parameter calibration](parameter_calibration/README.md): profiling owns
raw measurements, medians, uncertainty and parameter evidence; its validation
calls `llm_infer_model.performance.evaluate` for every prediction. See
[ownership boundaries](../docs/PERFORMANCE_BOUNDARIES.md). Existing completed-action
measurements remain useful references; their timing scope must not be silently
changed into pure kernel throughput or bandwidth.

本目录采集真实执行成本，并验证这些成本能否预测组合路径。模型、精度、线程、端口、初始驻留和计时边界必须明确；程序不会把孤立矩阵乘改称完整 Attention/FFN，也不会把未知 GPU workspace 填成 0。

推荐从[执行变换校准工具](execution_calibration/README.md)开始。当前 FFN 分片、Attention 分段和显式权重重载已做数值与独立预测检查，但完整延迟质量门尚未通过；不是完整 Dense/MoE 或新 IR 全链的校准结果。测量结果由使用者在目标硬件上生成，不随库分发；核心库可使用显式提供的成本或标为 synthetic 的示例成本，不要求已有 profile。

## 选择工具

| 要做什么 | 入口与说明 |
|---|---|
| FFN 分片、Attention 分段、CPU/GPU 路径与权重重载校准 | [execution_calibration](execution_calibration/README.md) |
| compact FFN 阶段、Attention core/output、CE/SM 搬运与侧表回放 | [matched_mechanisms](matched_mechanisms/README.md) |
| 单设备驻留 Dense 的 profile/heldout 校准 | [strategy_calibration](strategy_calibration/README.md) |
| 独立 MUL_MAT、H2D/D2H microbench | `run_measurements.py`、`profile_contract.py`，用法见下文 |
| CPU Attention/FFN 部分子图 | [MACRO_PROFILING.md](MACRO_PROFILING.md) |
| 含 projection/norm/RoPE/residual 的 CPU Dense block | [DENSE_BLOCK_MEASUREMENT.md](DENSE_BLOCK_MEASUREMENT.md) |
| 单设备完整 Dense CUDA | [DENSE_CUDA_MEASUREMENT.md](DENSE_CUDA_MEASUREMENT.md) |
| 同 resident 的 CPU 分段/联合图配对 | [DENSE_ERROR_VALIDATION.md](DENSE_ERROR_VALIDATION.md) |
| CUDA 内存观察与显式容量预算 | [DENSE_CUDA_MEMORY.md](DENSE_CUDA_MEMORY.md) |
| CPU MoE Router/专家/Combine 子图 | [MOE_MEASUREMENT.md](MOE_MEASUREMENT.md) |
| selected-row MoE 的独立数值参考 | [MOE_ROWS_VERIFICATION.md](MOE_ROWS_VERIFICATION.md) |
| 路由导入、独立搬运、allocator/重载实验 | `moe_routes.py`、`moe_transfer.py`、`reconfiguration/run.py`；分别查看 `--help` |

这些工具对应不同执行边界，不能互换价格。数据接入规则见 [M2 物理合同](../modeling/llm_infer_model/m2/PHYSICAL_CONTRACT.md)与[执行校准合同](EXECUTION_CALIBRATION_CONTRACT.md)。

## 独立算子的构建和运行

Python 只使用标准库，需要 Python ≥3.10；native 使用 C++17 和 GGML 公共接口。需已有匹配的 GGML 源码、库和运行时；CUDA 测量另需 CUDA Toolkit。更换 GGML 版本后重新编译并验证接口。构建到新目录，避免覆盖已被实验 SHA 绑定的程序。

从 `Simulator/` 运行，将占位符替换为实际路径：

```powershell
cmake -S profiling/native -B "<new-build-directory>" -DGGML_SOURCE_DIR="<GGML source>" -DGGML_BUILD_DIR="<existing GGML build>" -DMEASURE_ENABLE_CUDA=ON
cmake --build "<new-build-directory>" --config Release

python profiling/run_measurements.py --dry-run --suite profiling/cases/dense_smoke.json --backends CPU,CUDA0 --kinds compute,transfer
python profiling/run_measurements.py --native-bin "<native executable directory>" --runtime-dir "<matching GGML runtime directory>" --runtime-dir "<CUDA runtime directory>" --suite profiling/cases/dense_smoke.json --backends CPU,CUDA0 --kinds compute,transfer --threads 16 --warmup 5 --samples 20 --output "<new-output-directory>"
```

CPU-only 构建用 `-DMEASURE_ENABLE_CUDA=OFF`，运行选 `--backends CPU --kinds compute`。Windows 多配置程序通常在构建目录的 `Release/`；Linux 单配置构建需指定 `-DCMAKE_BUILD_TYPE=Release`。运行库目录通过 `--runtime-dir` 显式传入，不混用同名不同版本的库。

compute 当前支持 CPU/CUDA0。`CUDA_VISIBLE_DEVICES` 可显式选择实际 GPU，环境记录保留变量与设备身份；`--device` 只改变 transfer 设备，两者联合运行必须一致。工具不关闭桌面应用，不调整功耗或频率。

输出目录必须为空。所有测量工具共用 `.measurement.lock`；不得移除仍由其他进程持有的锁。失败/超时保留 raw，`--keep-going` 可继续其他 case。

## 自定义形状与查询

```json
{
  "schema_version": "measurement_suite_v1",
  "compute_cases": [
    {"id": "projection", "op": "MUL_MAT", "m": 2048, "k": 2048, "n": 1, "dtype": "f32"}
  ],
  "transfer_cases": [
    {"id": "weight_h2d", "direction": "H2D", "host_memory": "pinned", "size_bytes": 16777216}
  ]
}
```

GGML 权重为 `[k,m]`、输入 `[k,n]`、输出 `[m,n]`；n 是已经打包的总 token 数，不再额外乘 batch。Q4_K/Q6_K 要求 k 能被 256 整除；权重实际字节由 GGML 返回，不能根据模型文件名猜 dtype。权重由 seed 生成，不加载预训练权重。

```powershell
python profiling/make_suite_from_graph.py --graph "<collector graph.json>" --graph-id g1 --layer-id 3 --tokens 1,32,128 --dtype-mode native --output "<new-suite.json>"
python profiling/query_profile.py --profile "<completed-measurement-directory>" --case-id "<exact-case-id>"
```

提取器只收集支持的二维权重矩阵，不能提取完整 Attention；不支持的 dtype/形状明确记录，不静默替代。查询输出整数纳秒、workspace、签名、统计与出处，不自动生成 M1/M2 成本。

## 结果与计时合同

每次运行保存 `manifest.json`、`plan.json`、`raw/`、`measurements.json` 和 `summary.csv`，包括源代码、程序、运行库、协议与环境身份。全部样本保留，CV 是波动观察，不自动等于校准通过。

- 计算：分配、初始化、量化和外部 H2D 在计时外；同一权重/输入驻留。墙钟包含图提交与完成同步，CUDA graphs 显式关闭。不是纯 kernel、冷访存或并发成本。
- 搬运：墙钟包含 enqueue、event 与等待；CUDA event 时间另存。pageable 可能包含 staging 或 stream idle，不能当作纯 PCIe 链路时间。
- 数值：独立 matmul 只检查声明的有限输出点，不能据此宣称全模型正确或量化质量合格。完整子图工具有各自的端口/KV 合同。
- CPU workspace：独立 matmul 来自 graph plan；其他子图按各自合同记录 gallocr/cplan。权重、端口和内部 scratch 不能重复计数。
- GPU 私有 workspace：未知保持 `null`。内部张量和、设备 free 差值及外部 reserve 不能冒充完整 scratch 上界。

签名区分 CPU `threads_requested/threads_applied` 与 CUDA 线程不适用。运行库 hash 只绑定提供的文件，不证明动态加载器最终解析了所有库；调用方声明的 backend revision 也不冒充已核验 commit。

接入 modeling 时必须匹配完整语义、形状、存储 dtype 与实际计算精度、设备、线程/提交方式、端口和物化范围。COMPUTE 已含本地访存；仅对真实跨域移动另外计 COPY。不能把孤立 matmul 相加称为已校准宏任务，也不能按比例分摊宏任务价格。并发与预取需要匹配的真实组合验证。

`export_m2_macros.py`、`export_dense_layer.py`、`export_dense_cuda.py` 对接不同版本的明确边界；默认拒绝不稳定数据。显式豁免只能生成诊断/validation-only 表，未知 workspace 仍不能参加容量执行。

## MoE 数据与独立验证

`moe_routes.py` 支持本地 llama.cpp 路由与 FreeToken trace 导入，保留原专家 ID、top-k、phase 和绝对位置。可选的公开输入、CC BY 4.0 出处和重建命令见[路由数据](data/public-routes/README.md)；它是路由输入，不是性能校准或原模型数值证据。原始测量结果、较大的本机路由 bundle 和构建记录保留在本机，不纳入库安装包。

`validate_moe_prediction.py` 使用独立 profile/holdout 比较完整声明的 CPU MoE FFN。`moe_transfer.py` 测 pinned/pageable H2D/D2H 和 OS 缓冲文件 I/O；没有清空文件缓存，不能把读速称为 SSD 冷读带宽，也不能凭单项速度推导重叠。

## 无硬件测试

```powershell
python -B -m unittest discover -s profiling/tests -v
python -B -m unittest discover -s profiling/execution_calibration/tests -v
```

专题工具另有自己的 `tests/`，见各目录 README 和[代码地图](../docs/CODE_LAYOUT.md)。fake probe 和合同测试不替代真实硬件、性能或容量验收。
