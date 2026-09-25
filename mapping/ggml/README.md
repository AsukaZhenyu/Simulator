# ggml/：M0 的 GGML 导出器

把一张真实的 GGML 计算图导出成 M0 的 workload 文档（`DESIGN.md` §3）。这是本仓库唯一的 C++ 部分，也是 `examples/ggml/*.workload.json` 里 `origin.kind = "ggml"` 的依据——`ACCEPTANCE.md:113` 把这个值保留给真实导出，手写 fixture 不得使用。

导出器只做前端，不含任何模拟或搜索逻辑；它不链接进 Python 包，缺它时核心测试照常运行。

## 构建

前置：CMake ≥ 3.16 与一个 C++17 编译器。本机实测工具链为 CMake 3.26.0 + Visual Studio 2022（MSVC 19.44）。**不要求 CUDA，也不需要 GGUF。**

源码路径通过 `-DGGML_SOURCE_DIR=` 指定（或设同名环境变量），期望形状为 `<ggml>/include/ggml.h` 与 `<ggml>/CMakeLists.txt`；不存在则 `FATAL_ERROR`，不自动拉取任何版本。

```bash
cmake -S mapping/ggml -B mapping/ggml/build -G "Visual Studio 17 2022" -A x64 \
      -DGGML_SOURCE_DIR="<指向 ggml 目录>" -DGGML_CUDA=OFF
cmake --build mapping/ggml/build --config Release --target ggml-export-workload
```

产物在 `mapping/ggml/build/bin/Release/ggml-export-workload.exe`（`build/` 已被根 `.gitignore` 覆盖，不入库）。

本机没有 `make`/`ninja`/`mingw32-make`，所以上面的构建入口绑定 VS 生成器；若在别处构建，用任意生成器都可以，测试按 `mapping/ggml/build/bin/{Release,RelWithDebInfo,}` 顺序找产物，也可用 `TENSOR_MAPPING_GGML_EXPORTER` 直接指定路径。

**为什么 CPU-only**：导出全程 `no_alloc = true`，从不分配张量数据、也从不初始化任何后端，CUDA 对导出没有作用（`DESIGN.md` §3 也只要求 F32 + `no_alloc`）；关掉它同样让构建快得多。实测从零配置 + 构建（含 ggml 本身）约 **31 s + 38 s**，产物与已入库路径的产物逐字节相同。

**回退路径**（**目前没有实现，跨树 `add_subdirectory` 在实测中可用，所以不需要它**）：若某环境下这条路径出问题，可把上游 ggml 当独立工程配置到 `mapping/ggml/build/ggml`，再给 `CMakeLists.txt` 加一个 `-DGGML_LIBRARY=<ggml.lib>` 分支直接链接。计划阶段预留了这条路线，但本轮没有走，因此 `CMakeLists.txt` 里**没有**这个开关——写在这里是为了不让下一个人重新推一遍，而不是说它已经能用。

## 已入库产物的出处

`mapping/examples/ggml/` 里的四个 `*.workload.json` 由上面的命令从下面这棵树导出并入库：

| 项 | 值 | 来源 |
|---|---|---|
| GGML 版本 | **0.9.11** | 运行时 `ggml_version()`；与 `<ggml>/CMakeLists.txt` 的 `GGML_VERSION_{MAJOR,MINOR,PATCH}` 一致 |
| commit | **unknown** | 运行时 `ggml_commit()`：该目录是 release 快照，**不是 git 检出**，没有 commit 可报 |
| 源码树 | `LLM-Infer/llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml` | 本机构建参数；**在 Simulator 仓库之外** |

版本与 commit 都取自运行时而非编译期常量：`GGML_VERSION`/`GGML_COMMIT` 是 `target_compile_definitions(... PRIVATE)`，消费方的编译单元看不到它们，只有 `ggml_version()`/`ggml_commit()` 能拿到。两者连同构建方式记录在产物顶层的 `comment` 里——`origin` 只允许 `kind`/`ggml_version`/`sample` 三个键（`spec.py` 的 `_reject_unknown_keys`）。

复现命令：

```bash
mapping/ggml/build/bin/Release/ggml-export-workload.exe \
    --graph all --out-dir mapping/examples/ggml
```

`tests/test_ggml.py::ExporterRerunTests` 会重跑这一步并要求与入库产物**逐字节相同**（文档用 `'\n'` 写出、不含随环境变化的字段）。仓库被克隆到别处时，只要 ggml 源码树相同，产物就相同；若换了 ggml 版本，两处都会变，测试会要求重新生成并提交。

## 用法

```text
ggml-export-workload --graph <chain|residual|fork|matvec|all>
                     (--out FILE | --out-dir DIR)
```

`--graph all` 展开为四张正图；`--out` 只接受单个 `--graph`，与 `--out-dir` 互斥。每次运行向 stderr 打一行 `ggml <version> commit <commit>`，可直接引用为验证报告的出处。

### 正图

| 图 | 语义 | 张量 / 算子 |
|---|---|---|
| `chain` | `h=W1*x; y=W2*h` | `W1 W2 x h y` / `c1 c2` |
| `residual` | `h=W1*x; a=relu(h); b=W2*a; y=b+x` | `+a b` / `+r add` |
| `fork` | `a=W1*x; b=W2*x; y=a+b` | `+a b` / `+add` |
| `matvec` | `y=W*x`，`W` 为 3×4 的矩形权重 | `W x y` / `c1` |

id 与 `mapping/examples/` 里同名 fixture **故意完全一致**，这样导出产物与 fixture 可以逐字段对拍，不需要任何重命名表；两者只应差 `origin`、`comment` 与张量的 `name`（fixture 用 `blk.0.attn_q.weight` 这类真实名字，导出只能用 id）。`tests/test_ggml.py` 断言这一点。

`matvec` 是矩形 MUL_MAT 的检查点：方阵权重掩盖 `ne`/`nb` 约定错误（交换 in/out 后字节数不变），3×4 权重配上 3 宽的激活则会让误读多要一个 16 B 的激活而不是 12 B。

### 负图（只用于验证拒绝路径）

`reject-f16` / `reject-op` / `reject-unary` / `reject-inplace` 能按名字传 `--graph`，但**不属于 `all`**：每个都只违反一条布局规则，必须非零退出、指名张量、且不留下任何文件。

| 图 | 违反的规则 | 实际报错 |
|---|---|---|
| `reject-f16` | 只支持 F32 | `unsupported dtype on 'W' (op=NONE): f16` |
| `reject-op` | 算子必须是 `MUL_MAT`/`RELU`/`ADD` 之一 | `unsupported op on 'y' (op=DUP): DUP` |
| `reject-unary` | unary 只支持 `RELU` | `unsupported unary op on 'y' (op=SILU): SILU` |
| `reject-inplace` | 无 `view_src`、无原地共享 | `unsupported view on 'y' (op=ADD): view_src is set` |

这些负图用 `ggml_silu` 而不是 `ggml_sqr`：在本版 ggml 里 `sqr` 是独立的 `GGML_OP_SQR`，根本走不到 unary 分支——它看起来在测这条规则，实际测的是另一条。`silu` 和 `relu` 一样经 `ggml_unary()`，也是 LLM 前馈层真正用的激活。

## 导出器做什么

**遍历**：先 `ggml_build_forward_expand` 走真实 API（证明图自洽，也是 `DESIGN.md` §3 指定的入口），然后**从请求的输出自己递归 `src[]`** 收集祖先闭包。不读 `struct ggml_cgraph` 的字段——它在 `ggml.h` 里只有不透明前置声明；这也顺带满足「不能把节点数组顺序加成额外依赖」：依赖只来自 `src[]`。`ggml_graph_n_nodes()` 作为交叉校验（公开访问器），只在 ggml 报出的节点数**少于**声明数时报错。

**分类**：图描述表为每个叶子显式声明 `role ∈ {weight, input}` 与初始位置，不靠形状猜——ggml 张量本身不说明一个权重在 DRAM 还是 VRAM。计算张量是请求输出则 `role=output`，否则 `intermediate` 且省略 `initial_locations`。

**语义**：`ggml_op` 存原始枚举（ReLU 即 `GGML_OP_UNARY`），`unary_op` 存 `GGML_UNARY_OP_RELU`，`semantic_op` 存归一化后的 `MUL_MAT`/`RELU`/`ADD`——原始与归一化同时落盘，ReLU 陷阱两头都留住。`op_params` 只存该算子语义相关的前缀（UNARY 取 `[0]`，正是 fixture 里的 `6`），避免把 64 字节填充倒进产物。

**拒绝**：非 F32、`view_src != NULL`、`nb` 不满足 `nb[0]==4 && nb[i]==nb[i-1]*ne[i-1]`、`ggml_nbytes` 与 `nb*ne` 不一致、零尺寸、算子不在三个语义内、叶子 `op != NONE`、声明与可达闭包不一致、arity 不匹配、声明顺序非拓扑——一律非零退出并指名张量。`view_src` 是「一张存储张量 = 一次独立分配」的判据：`no_alloc=true` 下所有 `data` 都是空指针，**不能**用它判断别名。

**JSON 落盘**：最小手写 writer，不引第三方库；键序与 fixture 一致（`schema_version, id, origin, comment, tensors, operations, outputs`），2 空格缩进 + 尾换行，`std::ios::binary` 写出以免 `'\n'` 被改写成 CRLF。每张图一个独立 `ggml_context`：一次导出不依赖另一次，共用内存池会让产物取决于写图顺序。

## 未做的事

- 只支持 `MUL_MAT` / `RELU` / `ADD` 与 F32 连续张量——未支持的语义**报错而非猜测**，所以真实模型图目前导不出来。
- 不做算子融合、不做后端切分、不消费已编译的 ggml 库（构建时从源码编）。
- 源码树携带不了 VCS 信息（不是 git 检出、mtime 统一），所以它是否被本地改动过**无法判定**；这一点按实记录，不作为「干净」的断言。
