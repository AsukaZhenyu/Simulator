# composing/：组装层

这个包只回答一个问题：**表里的一条条价格，怎么加成一个请求的一步？**

它**不测量任何东西**。墙钟在 `profiling/`（llama-bench）；一个动作在某台设备上值
多少在成本表里（由 `profiling/` 导出，在这里通过
`modeling.llm_infer_model.m2.costs.parse_cost_table` 读取）。这里只有：

- 一条求和规则（`compose.py`）；
- 表里没有定价的量怎么被**声明**进来（`terms.py`）；
- 预测、比照与归因三份文档（`predict.py` / `compare.py` / `decompose.py`）；
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
