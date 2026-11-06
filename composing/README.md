# composing/：组装层

这个包只回答一个问题：**表里的一条条价格，怎么加成一个请求的一步？**

它**不测量任何东西**。墙钟在 `profiling/`（llama-bench）；一个动作在某台设备上值
多少在成本表里（由 `profiling/` 导出，在这里通过
`modeling.llm_infer_model.m2.costs.parse_cost_table` 读取）。这里只有：

- 一条求和规则（`compose.py`）；
- 表里没有定价的量怎么被**声明**进来（`terms.py`）；
- 预测、比照与归因三份文档（`predict.py` / `compare.py` / `decompose.py`）；
- 一个点估计旁边必须有的**带**（`variability.py`）；
- 同一个预测的两次墙钟怎么并排读（`pairs.py`）；
- 从别的机器回来的结果怎么被接收（`result_check.py`）。

## 四个模块，四个问题

```text
mapping      该做什么         策略、IR、编译、决策
modeling     一个动作怎么执行  图展开、状态、容量、时钟
profiling    一个动作值多少    探针、campaign、成本表导出
composing    这些价格怎么加    求和规则、声明项、预测与归因
```

依赖方向是单向的：

- `composing` **可以**读 `modeling`（成本表的物理格式、契约签名）；
- `modeling` **不许** import `composing`；
- `composing` **不许** import `profiling`——一个需要测量装置的组装就不是模型。

形状词表（`pp256` / `tg128` / `tg512` / `tg1024`）定义在 `spec.py`，由
`profiling/run_dense_endtoend.py` 导入：墙钟和预测必须是同一个问题的两个数，
所以这个词表只能有一份。

## 三份文档，顺序是规矩

```text
predicted.json    表说一步值多少          ← 先落盘
measured.json     墙钟说了多少、在什么协议下   ← 后测
comparison.json   两者并排（只读上两份）      ← 最后算
```

`profiling/run_dense_endtoend.py measure` 读不到 `predicted.json` 就拒绝运行。
`compare` 不改写预测。R11：**改了表就必须重测**，旧墙钟只能当诊断。

## 带：点估计旁边必须有它

`comparison.json` 从 v3 起每一行都带 `observation`（这一行自己的重复之间散开多远），
文档顶层带 `session`（同一个进程里哪几行散得比 `DEFAULT_BAND` 还宽）。它**不修正任何数**：
误差仍然对点估计算。它只是让"这个数对不对"和"这台机器这一轮抖没抖"成为两列。

为什么必须分两列：在这台机器上用同一份探针二进制、同一条命令，2026-11-04 18:58 那
次是**单峰**（8000 个样本里只有 6 个慢的，中位 1092 µs），2026-11-05 同一条命令变成
**双峰**（约 1085 µs 与约 1270 µs，慢模占 61.6%，每几毫秒翻一次）。频率逐位没变
（4.99 vs 5.00 cycles/ns），换五个物理核形状相同，把负载钉在探针线程的 SMT 兄弟上
可以**逐位复现慢模**、钉在另一个 CCD 上则完全无效。所以它是**状态**，不是漂移；
落在带里的误差不是成本模型的证据，落在带外的误差也不能拿机器解释。

`variability.split_modes` 对少于 `MODE_MIN_SAMPLES`（200）个样本一律拒绝判双峰：
五行 llama-bench 重复只能说明它们相距多远，不能说明它们来自两个进程。双峰只在
探针那种几千样本的运行上读。

```bash
python -m composing band --run PROBE.json [--run ...] [--json PATH]
```

## 表在不在同一个带里

`band` 说的是**一次运行内部**的抖动；它答不了另一个问题：**这张表是在哪个带里定的价，
现在的钟在哪个带。** 2026-11-05 用同一个探针二进制、同一个 llama-bench 二进制、
逐项相同的回显协议重测，六个模型的 pp256 有五个快了 8.7–10.6%，只有那张在台阶之后
才定价的表没动；于是"旧带的表 × 新带的钟"会给出 −9…−15% 的误差，而同带是 −4…−9%。

查法是拿表自己保存的那条 **joint block 记录**（导出器只用它做组成参考、从不求和）
与刚才用同一探针测的同名记录相比：

```bash
python -m composing table-state --table DIR --sensor PROBE.json [--sensor ...] [--json PATH]
```

超过 `DEFAULT_TABLE_TOLERANCE`（5%）判 `different_band`。5% 落在两个已测的数之间：
同一会话内的组成漂移约 2.4 点（不该触发），台阶是 9–14%（必须触发）。**比值不是修正
系数**——探针动得比引擎少（实测 0.9224 对 0.9023），所以它只回答"这个比较成不成立"。

## 两轮：同一预测的两次墙钟

```bash
python -m composing pairs --a DIR --b DIR [--json PATH]
```

它回答一个很窄的问题：**同一个预测，两轮墙钟有没有落到同一个数上**。拒绝三件事——
两侧的 `prediction.sha256` 不同（那中间换了表，差值是模型变化和机器变化叠在一起，
事后分不开）、两侧的 K cache 类型不同（字节数不同）、任一侧文件内部混了两种 K 类型。
只在一轮里测过的行不配对，只是被点名。

它是**并列**不是修正：两轮各自保留自己的点估计与展开，`pairs` 只加一个比值、一个
"这一行两轮分不开吗"（差值不超过两轮自己较宽的那个 range），以及一条
`|比值−1| > DEFAULT_BAND` 的清单。跨轮的水平差本来就不能单点比（见上一节），
能读的是**配对差**——B2 的三对就是这么读的。

## 口径：斜率属于哪一次分组

`comparison.json` 顶层从 v3 起还带 `decode_grouping` 与 `warmup`，`validate_comparison`
**缺了分组就拒绝**——和 `session` 同理：没说，不等于没影响。

理由是 2026-11-05 的 B2：同样三行 decode、同一张表、交替三对，**逐形状**（每个深度一个
进程）的斜率是 6.525 / 3.525 / 4.037 µs/KVtok，**分组**（一个进程跑三个深度）是
8.481 / 7.173 / 6.940。判据（两口径差 < 2× 点内 CV）三对全在外，配对差三对同号
（−1.96 / −3.65 / −2.90 µs）。机制是**进程的第一条解码贵**且代价随深度递减：逐形状让
每个深度都付一次，分组只让第一个形状付、并把它剔掉（`warmup`）。事后把每个点的第一条
去掉再拟合，能把差解释掉 42–73%，剩下的**没有归因**。

所以 `decompose` 把分组原样带进 `decomposition.json` 顶层，`show` 在逐形状的运行上
直接印一行"这个斜率偏低"。`--comparison` 合并两份文件时，两份声明的分组不一致会拒绝：
把两种口径的行拟成一条线，报出来的斜率没有对应的协议。

## 声明项：三条规矩

一件表里没定价的量，只有在下面三条都成立时才会被加进去：

1. **不声明就不加。** 没有"从端到端误差拟合出来的系数"。`predict` 的
   `composition_terms: []` 是一句话：表之外一个量都没定价。
2. **每一项都指名测它的旋钮。** 没有可执行旋钮的量还是假设，不是项。
   `terms.KNOWN_TERMS` 是闭集，其余名字直接拒绝。
3. **没测就关着。** `enabled` 默认 `False`：声明一个量不改变任何数字。预测同时
   记录"声明了哪些"和"用了哪些"，读的人才能分清"没定价"和"定价为零"。

| 项 | 单位 | 阶段 | 旋钮 |
|---|---|---|---|
| `per_layer_marginal` | ns / 层 | prefill, decode | 探针 `--repeat-layers N`：循环 N 层的斜率（隔离 vs 在位） |
| `per_token_fixed` | ns / token | prefill | 固定深度扫 prompt token 数（`-p 128/256/512`） |
| `kv_append_per_token` | ns / token | decode | KV 元素类型（`-ctk/-ctv`），斜率对每 token 追加字节 |
| `stage_overlap` | 占块栈的比例 | prefill | `-ub` / `-t`：批内阶段并行后消失的那部分块时间 |

零声明项时求和逐位等于历史表达式（`block_sum_ns * layers + embedding + head`），
不经过浮点、不取整。`composing/tests/test_compose.py` 钉住这一条（`block_sum * layers
+ boundary` 必须与整数算法逐位相同，且没有 `terms_ns` 字段）。搬包时另用存档验过一次：
六个还留在本机的存档 `comparison.json` 用当前代码重算，`models` 逐位相同；只有
`created_at_utc` / `measured_at_utc` 两个时间戳不同。

## 命令

```bash
python -m composing predict   --output DIR --table id=DIR ... --models models.json --models-dir DIR
python -m composing compare   --output DIR
python -m composing decompose --e2e DIR [--json PATH]
python -m composing check     --result DIR [--json PATH]
python -m composing band      --run PROBE.json [--run ...] [--json PATH]
python -m composing table-state --table DIR --sensor PROBE.json [--json PATH]
```

测量不在这里：

```bash
python profiling/run_dense_endtoend.py measure --output DIR --llama-bench ... \
    --config pp256=5 ... [--models-dir DIR]
```

## 从别的机器回来的结果

仓库会推到 GitHub，其他设备拉下来测完推自己的分支。`check` 不重跑、不判断精度，
只回答"这份文件是不是它自称的那份"：预测的 sha256 对不对、cpu 掩码与线程数在
表头和每一行里是否一致、llama-bench 二进制的 sha256 是否是要的那一个。结果写成
`manifest`，和原始结果一起归档——没有环境自述的数字不是任何东西的证据。

**一份预测可以在任何设备上测。** `predicted.json` 里的 `gguf` 是**做预测那台机器**的
绝对路径，它是记录，不是输入：另一台机器用 `--models-dir DIR` 按**文件名**重新定位权重，
文档和它的 sha256 一个字都不改（改了就不再是预注册的那份预测）。这次实际读的是哪个目录，
记在 `measured.json` 的 `models_directory` 里，逐行命令里也有完整路径。成本表同样显式给：
`--table id=DIR`，不靠任何继承来的路径。

**表不用每次测量都重测。** 一张表的身份是 `(设备, 线程数, 实现路径, KV dtype, 模型)`；
这五项不变就是同一张表，一次测、多次用。准确度迭代改变的是组装层或墙钟，表是固定资产。
换机器或换上述任一项，才需要在那台机器上重新测（判据见
`profiling/multimodel/STANDARD.md` R3 一节的"一张表什么时候要重测"）。
