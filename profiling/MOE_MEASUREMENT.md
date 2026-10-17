# MoE FFN 实测工具

本工具测量明确定义的 **resident F32 MoE FFN 子图**：Router → 按 trace 分组的专家 SwiGLU → 按真实 token/slot 映射加权 Combine 与 residual。它与 M2 的完整 synthetic MoE layer 分开，不能把这些费用填进原 .6 整层投影/Attention 占位任务。

## 支持边界

- 只测显式 CPU、显式 threads、F32、固定 top-k、不丢 token。CUDA 参数会拒绝，不回退 CPU。
- 真实外部路由决定分组负载，保留原 expert ID、top-k 和 token 位置。示例是公开 trace layer 3 的 Prefill 首 3 token 和绝对位置 180 的 Decode 单 token，前者不冒充完整 180-token 请求。
- 输入和 Router/专家权重是可复现的随机张量。原 trace 模型是 NVFP4；这里测的是记录几何对应的 F32 声明实现，不复现原模型权重、量化 kernel 或全部模型输出。
- Router 执行真实 linear、softmax、argsort top-k discovery；分组仍使用外部 trace IDs。selected scores 来自随机 router logits 对这些 IDs 的概率并重新归一化，不冒充原模型分数。
- Attention、QKV、normalization、KV Cache、原模型 shared expert/hyper connection 不在本子图内。搬运由独立 transfer 工具测量，此处计时不包含 offload。

## 两种协议

v1 合同 `ggml_cpu_resident_moe_ffn_f32_v1` 对每个孤立任务连续热身与采样；这与轮换众多专家的联合图访问条件不同。它与下述 v2 合同不兼容，不能直接互换阶段价格。

当前 v2 合同 ggml_cpu_resident_moe_ffn_f32_v2，case 必须明确 sampling_mode=interleaved_joint_context。使用同一个 joint graph、resident buffers 和 allocator，通过公开 GGML graph_add_node API 引用原图节点分割任务。每轮按 **Router → 按 expert ID 升序的全部活跃专家 → Combine → Joint** 顺序计时。任务与联合图采用相同权重访问顺序和工作集，不再对单专家连续热缓存。签名和模型桥固定这一顺序；其他专家顺序需要重新测量。

所有物化端口按 FNV1a64 digest 与 Joint 对齐。独立标量参考检查覆盖 selected scores、top-k discovery、专家 SwiGLU 和最终 weighted residual。计时稳定性、数值正确性和预测误差各有证据，互不替代。

## 容量与 workspace

stage_ports(case, stage) 完整声明 native 外部端口/权重的 dtype、4D ne、bytes、category；joint 的 ports 是全部外部 resident inventory。expert 包含整份 hidden input 和 token row indices，gather 在图内；Combine 包含 row/slot indices 和所有物化的 expert outputs。

CPU workspace 来自实际 GGML gallocr 内部 buffer，加实际共同 CPU graph-plan work arena。它不包含已声明的外部端口/权重，也不包含进程 RSS、运行库、GGML metadata 或线程池管理内存。

v2 memory_accounting 中：

- resident_logical_bytes 是该阶段 declared ports 的总数。
- shared_resident_logical_bytes / shared_resident_allocated_bytes 是共同 joint resident buffer 的逻辑/实际分配。
- resident_allocated_bytes 也记录实际共同 buffer，不能减阶段 logical 来算 padding。共同 padding 用 shared_allocated − shared_logical，避免重复计算其他专家权重。
- internal_buffer_bytes 和 cpu_graph_plan_work_bytes 是共同 allocator/arena 的实际值；stage_cpu_graph_plan_required_bytes 是切片需求。workspace 不按专家数量重复相加。

这是声明实现的内存模型，不是整个进程峰值或原模型容量证明。

## 复用方法

准备已有共享库版 GGML/llama.cpp build（含公共头和 ggml-base/ggml-cpu 库），本工具不修改或重新构建其源码树。独立 C++17 CMake 目标位于 profiling/native/moe/CMakeLists.txt：

~~~text
cmake -S profiling/native/moe -B profiling/native/build-moe-reuse -DGGML_SOURCE_DIR=<ggml-source> -DGGML_BUILD_DIR=<existing-build>
cmake --build profiling/native/build-moe-reuse --config Release
~~~

Windows 若 CMake 无法发现 VS instance，可在 Developer Command Prompt 配置 MSVC 环境后使用 Ninja 或显式 cl。使用新构建目录和匹配库，避免覆盖已冻结的程序；将以下 executable 路径替换为该次构建产物。

Suite 使用 moe_measurement_suite_v1；case 明确 config（hidden/intermediate/experts/top_k/tokens）、routes、backend=CPU、threads、sampling_mode。可附加 phase、work_items、source。未知执行字段会拒绝。

在 Simulator 目录执行，参数显式给出，输出目录必须不存在或为空：

~~~text
python profiling/run_moe_measurements.py --suite profiling/cases/moe_public_ffn_cpu_shared.json --native-bin <new-ggml-moe-probe.exe> --runtime-dir <matching-ggml-runtime-directory> --output <new-output-directory> --warmup 16 --samples 31 --seed 123
~~~

添加 --dry-run 只输出计划，不执行硬件。输出含 plan、manifest、measurements、每 case 的 raw JSON/stderr。manifest 固定 suite/binary/runtime hashes；record 固定 signature、contract hash、raw hash、统计和质量标记。JSON 使用 LF。

load_measurements(directory) 校对 raw、签名、端口、统计和状态，拒绝改动后的证据；verify_live_sources=True 还核当前 suite、binary 和 runtime。跨机器复制的证据可以只校对 portable 内容。数字缺少 manifest/raw 证据时不能作为可靠测量导入。

## 独立验证

分别采集 profile 和新的独立 holdout 进程，只用 profile 阶段样本构造价格。joint 是组合预测目标，不能用来拟合自身成本。显式给出容量预算及验收阈值：

~~~text
python profiling/validate_moe_prediction.py --profile <profile-directory> --holdout <holdout-directory-1> --holdout <holdout-directory-2> --hardware <explicit-hardware-budget.json> --max-relative-error <declared-threshold> --output <new-comparison-directory>
~~~

阶段 CV>0.1 默认拒绝；显式 `--allow-unstable` 只产生诊断/validation-only 结果，不升级为性能承诺。新形状、线程数与专家顺序需要重新测量，不能用插值、workspace=0 或联合时间乘系数补价。

本工具的合同与预测回归在 `profiling/tests/test_moe_measurements.py`、`test_moe_prediction.py`。[执行变换工具](execution_calibration/README.md)另有 compact FFN 分片测量入口；它不含本工具的 Router/gather/Combine，不是该 MoE 子图的整层校准证据，当前完整性能质量门尚未通过。本机历史结果不随仓库分发。
