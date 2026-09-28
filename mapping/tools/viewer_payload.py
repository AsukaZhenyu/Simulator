"""查看器的纯数据装配层：把 scenario 变成一页能画出来的载荷。

这个模块**不做任何决策**。它只做三件事：把内核已有的数字搬进一个 JSON 友好的
形状、把搜索的**过程**记下来、把两处会「拒绝」的边界原样报出来。合法性、时间推进、
最优性一律还是问内核——和 ``artifacts.py`` 同一条纪律。

搜索树是怎么来的（``explore()``）
----------------------------------
以前它**在包外重走了一遍 Dijkstra**：``mapper.search()`` 只回一个 :class:`SearchResult`
（终局、数字、动作序列），不回搜索树，于是这里按 ``mapper.py`` 逐条对着抄了一份主循环，
再把两趟的结果对账。``M1_ACCEPTANCE.md`` §11 把这条路否掉了——「GUI 不再重走第二份
Dijkstra」。画出来的树只跟那份副本一样可信，而副本是会腐烂的。

现在树是**这一趟搜索的副产品**：:class:`tensor_mapping.mapper.SearchRecorder` 把搜索
自己做出的每个决定报出来，这里只负责记。内核那边没有任何分支读记录器，所以「挂不挂
记录器，搜索结果一样」是结构性的，不是靠两趟各自算对。

对账没有跟着走（``reconcile``）
-------------------------------
树不再是独立算出来的，所以旧的那种「两份实现互相比对」没有意义了。留下的是**树与它
自己那趟搜索的自洽**，四条都能被证伪：

1. 树上被展开的节点数 == ``result.expanded_states``；
2. 树上的节点总数 == ``result.visited_states``（图被截断时这条不成立，见下）；
3. **沿父链从终局节点走回起点，路上每个节点的动作拼起来 == ``result.actions``**
   （tuple 相等，不是长度相等）。这条是真正的网：它证明画出来的那棵树自己就能拼出
   报告出去的那份映射；
4. 终局节点存在 ⟺ 搜索报的是 ``optimal`` / ``feasible``；没有终局节点时
   ``result.actions`` 必须为空。

对不上就在屏上打红条并禁用搜索树屏——宁可说「我不知道」，不画一棵可能是错的树。

图被截断时（节点数超过 ``max_nodes``）节点不全，第 2、3 条**验不了**，于是
``reconcile.checked`` 是 ``False``：不是「一致」，是「没有结论」。截断本身另有
``truncated`` 徽章，屏上写「展示不完整」，两者不混。

节点与边的形状
--------------
``nodes`` 覆盖**每一个被接受过的状态**：起点、被展开过的、进过堆但没被展开的（前沿）。
后一类带 ``expand_order == -1``。只由被剪边到达的节点**不存在**——``search()`` 只在
接受一条边时才有状态可报，被支配的后继从来没被接受过，所以它连 ``State`` 都不是这棵树
的一部分。每个被展开的节点带一份
``children``——它在 ``legal_actions`` 里的**全部**合法动作，每个或者有出边、或者是一条
``cut``（被支配）。因此「从某个已展开的节点出发，所有合法动作都能点」在离线也成立，
不需要 Python 侧再算一次。**没被展开的节点没有 ``children``**（前沿就是前沿），屏上
要照实说，不能装成「这个状态没有合法动作」。

``t_ns`` 与 ``g_ns`` 恒等（``t`` 就是路径代价，这也是 ``State.key()`` 敢把 ``t`` 排除在外的
原因）。两个都发是为了让自检能独立验证这条不变式；只发一个就验不了了。

单位与边界
----------
本模块只处理内核的整数纳秒，不碰 ``modeling`` 的浮点秒——那条换算在
:mod:`llm_infer_model.tensor.layer_costs` 里，``bridge_table()`` 只调用它、不重算。
"""

from __future__ import annotations

from typing import Any

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY_H2D,
    ACTION_EVICT,
    Action,
    EvaluationResult,
    MappingError,
    State,
    WallClock,
    evaluate_mapping,
    initial_state,
    is_goal,
    legal_actions,
    transition,
    used_bytes,
    used_vram_bytes,
)
# ``ordered_candidates`` 要的是 ``legal_actions`` 那套次序的**全集**。私有名字跨包 import
# 是刻意的：契约说这个次序是确定性的来源，那它就只能有一份定义。改名会在 import 处炸。
from llm_infer_model.tensor.engine import (  # noqa: E402
    _m0_candidates,
    _m1_candidates,
)
from llm_infer_model.tensor.spec import (
    DEVICE_CPU,
    DEVICE_GPU,
    GOAL_LOCATIONS_M1,
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    RESOURCE_CPU_COMPUTE,
    RESOURCE_GPU_COMPUTE,
    RESOURCE_OF_DEVICE,
    RESOURCE_OF_ROUTE,
    RESOURCE_PCIE_COPY,
    RESOURCE_SSD_IO,
    ROLE_WEIGHT,
    ROUTES,
    ROUTE_ENDS,
    InitialCapacityExceeded,
    Scenario,
)
from tensor_mapping.artifacts import (
    MODE_SEARCH,
    action_to_dict,
    assumptions,
    events_document,
    limits_block,
    search_block,
    states_document,
)
from tensor_mapping.mapper import (
    STATUS_FEASIBLE,
    STATUS_OPTIMAL,
    SearchResult,
    search,
)

__all__ = [
    "DEFAULT_MAX_NODES",
    "VIEWER_SCHEMA",
    "explore",
    "graph_block",
    "payload",
]

# 查看器自己的载荷版本，**不是** ``spec.SCHEMA_VERSION``。两者会各自演化：产物格式是
# 冻结契约，载荷是这一页的私有形状。共用一个号会让「产物没变但载荷变了」看起来像
# 「产物变了」。
VIEWER_SCHEMA = 1

# 载荷里最多记多少个状态节点。防御性的：真到这一步说明图上了一个量级，此时用户要的是
# 「搜索给了什么结论」而不是「把几百 MB 的树画出来」。超了就**停止记录、把搜索跑完**，
# 终局照常正确，另挂一个「图已截断」徽章。
DEFAULT_MAX_NODES = 5000

# 被剪边只有一种原因，而且只有一个（``mapper.py`` 的更新条件）。写成常量是为了让
# 前端的字符串有一个来源，而不是散在各处的字面量。
CUT_DOMINATED = "dominated"


# ---------------------------------------------------------------------------
# 计算图
# ---------------------------------------------------------------------------


def _ranks(scenario: Scenario) -> dict[str, int]:
    """拓扑秩：张量的秩是「生产者算子的秩 + 1」，算子的秩是「最晚输入的秩 + 1」。

    这是**最长的那个**前驱路径长度（不是最短），所以同一秩里的节点之间不可能有边，
    分层布局因此不会把边画回头。图是 DAG（``spec._has_cycle`` 在校验时就拒了环），
    所以这里可以直接记忆化递归，不需要另写拓扑排序。
    """
    workload = scenario.workload
    ranks: dict[str, int] = {}

    def tensor_rank(tensor_id: str) -> int:
        cached = ranks.get(tensor_id)
        if cached is not None:
            return cached
        producer = workload.producer_of.get(tensor_id)
        # 叶张量（输入与权重）在最左边；只有它们没有生产者。
        value = 0 if producer is None else operation_rank(producer) + 1
        ranks[tensor_id] = value
        return value

    def operation_rank(operation_id: str) -> int:
        cached = ranks.get(operation_id)
        if cached is not None:
            return cached
        inputs = workload.operation_by_id[operation_id].inputs
        value = max((tensor_rank(t) for t in inputs), default=-1) + 1
        ranks[operation_id] = value
        return value

    for tensor in workload.tensors:
        tensor_rank(tensor.id)
    for operation in workload.operations:
        operation_rank(operation.id)
    return ranks


def movable_tensors(scenario: Scenario) -> set[str]:
    """哪些张量**有办法被搬**。

    0.1 是 ``mapspace.copy_tensor_ids`` 一个字面清单。0.2 没有这张清单：可搬性是
    「某条路线上允许搬它」的并集（``M1Mapspace.tensors_for``）。按域问，不猜字段名。
    """
    if not scenario.is_m1:
        return set(scenario.mapspace.copy_tensor_ids)
    movable: set[str] = set()
    for route in ROUTES:
        movable.update(scenario.mapspace.tensors_for(route))
    return movable


def m1_block(scenario: Scenario) -> dict[str, Any] | None:
    """0.2 才有的一块：域、资源、路线、设备、交付目标。0.1 返回 ``None``。

    返回 ``None`` 而不是空字典：0.1 的载荷因此**一个键都不多**，前端也能用一个
    ``payload.m1 == null`` 判断「这一屏要不要画 M1 那些面板」。

    这块里没有一个是这里算出来的：容量/预留/对齐走 ``architecture`` 的**按域访问器**
    （0.1 与 0.2 拼字段的差异就锁在那一处），可跑的设备与可搬的张量走 ``mapspace``，
    资源名字走 ``spec`` 的常量。前端因此不需要认识任何字段名就能画泳道。
    """
    if not scenario.is_m1:
        return None
    architecture = scenario.architecture
    mapspace = scenario.mapspace

    labels = {LOC_DRAM: "DRAM", LOC_VRAM: "VRAM", LOC_SSD: "SSD"}
    devices = {LOC_DRAM: DEVICE_CPU, LOC_VRAM: DEVICE_GPU, LOC_SSD: None}
    locations: list[dict[str, Any]] = []
    for domain in GOAL_LOCATIONS_M1:
        locations.append(
            {
                "id": domain,
                "label": labels[domain],
                # ``None`` 有两种来源，但屏上的意思是同一个：没有上限。
                # ``capacity_of`` 的文档里写着这两个来源（0.1 的 DRAM、以及 SSD）。
                "capacity_bytes": architecture.capacity_of(domain),
                "reserved_bytes": architecture.reserved_of(domain),
                "alignment_bytes": architecture.alignment_of(domain),
                "device": devices[domain],
                # 淘汰只在**有容量**的域上有意义：SSD 没有容量，所以没有可淘汰的。
                "can_evict": architecture.capacity_of(domain) is not None,
            }
        )

    # 四个资源，槽位从 ``limits`` 同一批字段取——与 ``limits_block`` 共用一份名字，
    # 屏上的「几条泳道」和产物里的「几个槽」因此不可能对不上。
    slot_fields = {
        RESOURCE_CPU_COMPUTE: "cpu_compute_slots",
        RESOURCE_GPU_COMPUTE: "gpu_compute_slots",
        RESOURCE_PCIE_COPY: "pcie_copy_slots",
        RESOURCE_SSD_IO: "ssd_io_slots",
    }
    resources = [
        {
            "id": resource,
            "slots": getattr(architecture, field),
            # 计算资源与搬运资源在屏上是两种东西（一个是算子，一个是副本）。
            "kind": "compute" if resource in RESOURCE_OF_DEVICE.values() else "transfer",
        }
        for resource, field in slot_fields.items()
    ]

    routes = [
        {
            "id": route,
            "src": ROUTE_ENDS[route][0],
            "dst": ROUTE_ENDS[route][1],
            "resource": RESOURCE_OF_ROUTE[route],
            "tensors": list(mapspace.tensors_for(route)),
        }
        for route in ROUTES
    ]

    return {
        "locations": locations,
        "resources": resources,
        "routes": routes,
        "devices": {
            operation_id: list(allowed)
            for operation_id, allowed in sorted(mapspace.allowed_devices.items())
        },
        # 交付目标：每个输出**必须**落在哪个域。屏上「输出是否已交付」的判据就是它。
        "targets": dict(scenario.goal.output_locations),
        "outputs": list(scenario.workload.outputs),
        "allow_cpu_gpu_overlap": mapspace.allow_cpu_gpu_overlap,
        "allow_copy_compute_overlap": mapspace.allow_copy_compute_overlap,
        "allow_eviction": mapspace.allow_eviction,
    }


def graph_block(scenario: Scenario) -> dict[str, Any]:
    """计算图本身：张量、算子、连线、布局秩。

    一切从 ``workload`` 推，**没有任何常量**。依赖关系用 ``producer_of`` /
    ``consumers_of``——内核的合法性检查（``engine._check_compute`` 的 INPUT_NOT_READY、
    ``_check_evict`` 的 LIVE_VALUE_LOSS）就是从这两张表推的，所以画出来的图不可能与
    模型不一致。前端只负责算坐标。
    """
    workload = scenario.workload
    ranks = _ranks(scenario)
    outputs = set(workload.outputs)
    copyable = movable_tensors(scenario)

    tensors = [
        {
            "id": tensor.id,
            "name": tensor.name,
            "role": tensor.role,
            "dtype": tensor.dtype,
            "storage_bytes": tensor.storage_bytes,
            # 占用是**对齐后**的；``storage_bytes`` 是搬运量。两者在 alignment > 1 时
            # 不同，混淆它们正是 ``ACCEPTANCE.md`` §5 点名的那类错误。
            "alloc_bytes": scenario.size_alloc(tensor.id),
            "initial_locations": list(tensor.initial_locations),
            "producer": workload.producer_of.get(tensor.id),
            "consumers": list(workload.consumers_of.get(tensor.id, ())),
            "is_leaf": tensor.id in workload.leaf_tensors,
            "is_output": tensor.id in outputs,
            "is_weight": tensor.role == ROLE_WEIGHT,
            "copyable": tensor.id in copyable,
            "rank": ranks[tensor.id],
        }
        for tensor in sorted(workload.tensors, key=lambda t: t.id)
    ]
    operations = [
        {
            "id": operation.id,
            "semantic_op": operation.semantic_op,
            "ggml_op": operation.ggml_op,
            "unary_op": operation.unary_op,
            "inputs": list(operation.inputs),
            "output": operation.output,
            # 0.1 的时长与工作区是**一个数**；0.2 的是「每台设备一个」。两个键都给，
            # 各自在另一个版本里是 ``None``。不共用一个「有时是数、有时是字典」的键：
            # 那种键要求每个读它的人都先知道版本，而读的人会忘。
            "compute_ns": (
                None
                if scenario.is_m1
                else scenario.costs.compute[operation.id].duration_ns
            ),
            "compute_ns_devices": (
                {
                    device: cost.duration_ns
                    for device, cost in sorted(
                        scenario.costs.compute.get(operation.id, {}).items()
                    )
                }
                if scenario.is_m1
                else None
            ),
            "workspace_bytes": (
                None
                if scenario.is_m1
                else scenario.costs.compute[operation.id].workspace_bytes
            ),
            "workspace_bytes_devices": (
                {
                    device: cost.workspace_bytes
                    for device, cost in sorted(
                        scenario.costs.compute.get(operation.id, {}).items()
                    )
                }
                if scenario.is_m1
                else None
            ),
            "output_alloc_bytes": scenario.size_alloc(operation.output),
            "rank": ranks[operation.id],
        }
        for operation in sorted(workload.operations, key=lambda o: o.id)
    ]
    # 连线在 Python 侧算好，前端只画。``from``/``to`` 是张量或算子的 id，靠两条表
    # （``tensor_by_id`` / ``operation_by_id``）区分，不需要单独的节点类型字段。
    links: list[dict[str, Any]] = []
    for operation in sorted(workload.operations, key=lambda o: o.id):
        for index, tensor_id in enumerate(operation.inputs):
            links.append(
                {
                    "from": tensor_id,
                    "to": operation.id,
                    "kind": "input",
                    "slot": index,
                    "weight": workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT,
                }
            )
        links.append({"from": operation.id, "to": operation.output, "kind": "output"})

    return {
        "workload_id": workload.id,
        "origin": {
            "kind": workload.origin.kind,
            "ggml_version": workload.origin.ggml_version,
            "sample": workload.origin.sample,
        },
        "outputs": list(workload.outputs),
        "tensors": tensors,
        "operations": operations,
        "links": links,
        # 张量画在偶数列、算子画在奇数列（所以算子的列号是 ``2*秩 - 1``）。秩越大越靠右；
        # 算子的输入张量秩 ≤ 秩-1、输出张量秩 = 秩+1，于是**每条边都从左指向右**，不会画
        # 回头。（注意：「同秩」的算子与张量里，算子在左——不是反过来。）
        "columns": {
            "tensors": {t["id"]: 2 * t["rank"] for t in tensors},
            "operations": {o["id"]: 2 * o["rank"] - 1 for o in operations},
        },
    }


# ---------------------------------------------------------------------------
# 搜索树：挂在搜索上的记录器
# ---------------------------------------------------------------------------


class _TreeRecorder:
    """``SearchRecorder`` 的实现：把搜索**自己报出来**的每一步记成一棵树。

    它只写、不读、不做判断：每个钩子返回 ``None``，记下来的东西没有任何一处回流到
    ``search()``。所以「挂不挂它，结果一样」不靠纪律，靠结构。

    两条记录规则与 ``mapper.py`` 逐字相同，不是这里发明的：

    * **只在严格更省时换父**。等值保持第一条路径——这正是零成本动作不会被按各种排列
      各走一遍的原因。钩子只在「这条边被接受了」时被叫到，所以这条规则天然成立。
    * **节点数超过 ``max_nodes`` 就停止建节点，但搜索照常跑完**。终局因此永远正确，
      另挂一个「图已截断」徽章。
    """

    def __init__(self, scenario: Scenario, max_nodes: int) -> None:
        self.scenario = scenario
        self.max_nodes = max_nodes
        # 动作的 JSON 拼法分 0.1/0.2 两套。树上的每条边都要写成**这个 scenario 的**
        # 那套：0.2 的动作缺了 ``device`` / ``src`` / ``dst`` 就不是这个动作了，
        # 而屏上「这一步在哪台机器上跑的」正是从这几个字段读的。
        self.m1 = scenario.is_m1
        self.nodes: list[dict[str, Any]] = []
        # 与 ``nodes`` 同序的 State，只为搜索结束后补算「每个已展开节点有哪些动作不合法」。
        # 放这里而不是塞进节点字典，是因为 State 不是 JSON，而且节点要精简。
        self.states: list[State] = []
        # 与 ``nodes`` 同序的 Action，只为沿父链重建动作序列（那份映射由搜索给出，
        # 这里是拿树自己拼一份来对账）。节点字典里存的是 JSON 形状，对不了 tuple。
        self.actions: list[Action | None] = []
        self.index_by_key: dict[tuple[Any, ...], int] = {}
        self.cut_counts: dict[str, int] = {}
        self.truncated = False
        self.start_id: int | None = None
        self.goal_id: int | None = None
        self.pending_id: int | None = None

    # -- 记录 ---------------------------------------------------------------

    def _action_document(self, action: Action | None) -> dict[str, Any] | None:
        """一个动作写成 JSON，按**这个 scenario 的**版本。

        0.2 的动作多带 ``device`` / ``src`` / ``dst`` / ``location``，那不是装饰：
        屏上「这一步在哪台机器上跑的、从哪搬到哪」只能从这里读。少调一次
        ``m1=True`` 的后果不是报错，是这几列安静地空掉。
        """
        if action is None:
            return None
        return action_to_dict(action, m1=self.m1)

    def _register(
        self,
        state: State,
        cost: int,
        parent_id: int | None,
        action: Action | None,
    ) -> int | None:
        """把一个状态登记成节点；已经登记过的只在**更省**时更新父与代价。

        返回节点 id，超出 ``max_nodes`` 时返回 ``None``（此时**不**建节点，但搜索
        照常往下跑）。
        """
        key = state.key()
        existing = self.index_by_key.get(key)
        if existing is not None:
            if cost < self.nodes[existing]["g_ns"]:
                node = self.nodes[existing]
                node["g_ns"] = cost
                node["t_ns"] = state.t
                node["parent"] = parent_id
                node["action"] = self._action_document(action)
                self.actions[existing] = action
            return existing
        if len(self.nodes) >= self.max_nodes:
            self.truncated = True
            return None
        node_id = len(self.nodes)
        self.index_by_key[key] = node_id
        node: dict[str, Any] = {
            "id": node_id,
            "t_ns": state.t,
            "g_ns": cost,
            "used_vram_bytes": used_vram_bytes(self.scenario, state),
            "op_status": dict(state.op_status),
            "copy_status": dict(state.copy_status),
            "running": [
                {
                    "resource": task.resource,
                    "target_id": task.target_id,
                    "remaining_ns": task.remaining_ns,
                }
                for task in state.running
            ],
            "is_goal": is_goal(self.scenario, state),
            "parent": parent_id,
            "action": self._action_document(action),
            "expand_order": -1,
            "role": "frontier",
            "depth": 0,
            "on_solution_path": False,
            "children": [],
        }
        if self.m1:
            # 三个账本分开报，不并成一个。``copy_status`` 在 0.1 与 0.2 里**都是
            # 显存那一本**（``engine.State`` 的文档写着），0.2 另加 DRAM 与 SSD 两本。
            # 屏上按域画格子时，缺哪本就会把「在 DRAM 里」读成「不在任何地方」。
            # 只问 DRAM：SSD 没有容量也没有用量（``used_bytes`` 的文档写着
            # 「向它要 SSD 的用量就是在要一个设计里没定义的数」），所以这一格
            # 不是漏了，是本来就没有。
            node["used_dram_bytes"] = used_bytes(self.scenario, state, LOC_DRAM)
            node["dram_status"] = dict(state.dram_status)
            node["ssd_status"] = dict(state.ssd_status)
        self.nodes.append(node)
        self.states.append(state)
        self.actions.append(action)
        return node_id

    def _child(
        self,
        parent_id: int | None,
        action: Action,
        to_id: int | None,
        cut: dict[str, Any] | None,
    ) -> None:
        """给父节点记一条出边（或一条被剪边）。父或子没建节点时什么都不记。"""
        if parent_id is None or to_id is None:
            return
        self.nodes[parent_id]["children"].append(
            {"action": self._action_document(action), "to": to_id, "cut": cut}
        )

    # -- SearchRecorder -----------------------------------------------------
    #
    # 钩子按 ``mapper.py`` 的调用顺序被叫到，所以 ``children`` 的先后与
    # ``legal_actions`` 的先后一致——每个父节点只被展开一次，追加即等价。

    def on_start(self, state: State, cost: int) -> None:
        self.start_id = self._register(state, cost, None, None)

    def on_expand(self, state: State, cost: int, order: int) -> None:
        node_id = self.index_by_key.get(state.key())
        if node_id is not None:
            self.nodes[node_id]["expand_order"] = order

    def on_edge(
        self, parent: State, action: Action, successor: State, cost: int
    ) -> None:
        parent_id = self.index_by_key.get(parent.key())
        successor_id = self._register(successor, cost, parent_id, action)
        self._child(parent_id, action, successor_id, None)

    def on_cut(
        self, parent: State, action: Action, successor: State, cost: int, incumbent: int
    ) -> None:
        # 被支配：已经有一条不更差的路径。**只记录，不展开**——这就是「被剪边」在屏上
        # 的全部含义。后继本身不会被登记（它从没被接受过），所以 to_id 可能是 None：
        # 那条边只连到一个不在树上的状态，屏上就不画。
        self.cut_counts[CUT_DOMINATED] = self.cut_counts.get(CUT_DOMINATED, 0) + 1
        self._child(
            self.index_by_key.get(parent.key()),
            action,
            self.index_by_key.get(successor.key()),
            {"reason": CUT_DOMINATED, "g_new_ns": cost, "g_existing_ns": incumbent},
        )

    def on_pending_goal(self, state: State, cost: int) -> None:
        # 只在「record 被刷新」时被叫到，所以最后一次就是这个搜索在预算切断时会
        # 报 ``feasible`` 的那个目标。这里不需要重推 ``_budget_cut`` 的取舍规则。
        self.pending_id = self.index_by_key.get(state.key())

    def on_goal_popped(self, state: State, cost: int) -> None:
        self.goal_id = self.index_by_key.get(state.key())


def _trace_check(
    block: dict[str, Any],
    result: SearchResult,
    *,
    path_actions: tuple[Action, ...] | None,
    complete: bool,
) -> dict[str, Any]:
    """把记下来的这棵树与**产生它的那趟搜索**比一遍。

    树不再是独立实现算出来的，所以这里比的不再是「两份 Dijkstra 一致」，而是「这棵树
    自己能不能拼出搜索报出去的东西」。四条，每条都能被证伪：

    * 树上**标了展开序号**的节点数 == ``expanded_states``；
    * 树上的节点总数 == ``visited_states``；
    * 沿父链拼出来的动作序列 == ``actions``（tuple 相等）；
    * 终局节点存在 ⟺ 报的是 ``optimal`` / ``feasible``。

    计数一律从**节点数据**上数，不从记录器自己的计数器上取——计数器是搜索调用了几次
    钩子的回声，与屏上真画出来什么无关，比它等于自证。

    ``complete`` 为 ``False``（图被截断）时后三条验不了（节点不全），于是 ``checked``
    也是 ``False``——「没有结论」，不是「一致」。
    """
    differences: list[str] = []

    verdict = block["verdict"]
    for field in ("status", "termination_reason", "optimality_proven"):
        if verdict[field] != getattr(result, field):
            differences.append(f"{field}: 树 {verdict[field]!r} vs search {getattr(result, field)!r}")

    if block["goal_id"] is not None:
        if result.status != STATUS_OPTIMAL:
            differences.append(
                f"树上有弹出的目标节点，但 search 报的是 {result.status!r}"
            )
    elif block["pending_id"] is not None:
        if result.status != STATUS_FEASIBLE:
            differences.append(
                f"树上有待定目标节点，但 search 报的是 {result.status!r}"
            )
    elif result.actions:
        differences.append("树上没有终局节点，但 search 给了一份非空映射")

    if not complete:
        # 截断：节点不全，下面几条不但验不了，比出来还会误导。
        return {"checked": False, "agrees": None, "differences": differences}

    expanded_marked = sum(1 for node in block["nodes"] if node["expand_order"] > 0)
    if expanded_marked != result.expanded_states:
        differences.append(
            f"展开数：树上标了展开的节点 {expanded_marked} 个 vs search expanded_states="
            f"{result.expanded_states}"
        )

    if len(block["nodes"]) != result.visited_states:
        differences.append(
            f"节点数：树上 {len(block['nodes'])} vs search visited_states="
            f"{result.visited_states}"
        )
    if path_actions is None:
        # 没有终局节点时 ``search`` 也不该有映射（上面那条已经这样要求了），两处都说
        # 「没有」就不算分叉——例如起点就装不下：可证的不可行，零个动作。
        if result.actions:
            differences.append("父链没能走回起点，拼不出映射")
    elif path_actions != result.actions:
        differences.append(
            f"动作序列不同：树上拼出 {len(path_actions)} 条 vs search {len(result.actions)} 条"
        )
    if block["solution"]["makespan_ns"] != result.makespan_ns:
        differences.append(
            f"makespan_ns: 树 {block['solution']['makespan_ns']} vs search {result.makespan_ns}"
        )
    return {"checked": True, "agrees": not differences, "differences": differences}


def explore(
    scenario: Scenario,
    *,
    max_expanded_states: int | None = None,
    wall_time_limit_s: float | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    clock: Any = None,
) -> tuple[dict[str, Any], SearchResult]:
    """跑一次 ``mapper.search()``（挂上记录器），把这一趟记成一棵树。

    返回 ``(树, 那次搜索的结果)``。**两个一起返回**，因为它们是同一件事的两面：树不是
    另算的，它就是这趟搜索的过程记录。分开拿会在调用方手里重新长出「再搜一次」的余地，
    而那正是 ``M1_ACCEPTANCE.md`` §11 要拆掉的东西。

    ``clock`` 只在自检里用：极小的 ``wall_time_limit_s`` 下跑到第几次展开才停取决于
    机器，注入一个假时钟才能确定性地走一遍 ``time_budget_exhausted``——``M1_DESIGN.md``
    :225 要的就是这个（预算测试不依赖某台机器恰好跑多快），与 ``mapper.search``
    的做法一致。真时钟本身现在是可用的（``WallClock`` 用 1 µs 粒度的 ``perf_counter``；
    2026-09-28 之前用粒度 15.625 ms 的 ``monotonic``，那时极小预算**永远不触发**）。
    """
    recorder = _TreeRecorder(scenario, max_nodes)
    result = search(
        scenario,
        max_expanded_states=max_expanded_states,
        wall_time_limit_s=wall_time_limit_s,
        clock=clock,
        recorder=recorder,
    )

    nodes = recorder.nodes
    verdict = {
        "status": result.status,
        "termination_reason": result.termination_reason,
        "optimality_proven": result.optimality_proven,
        "expanded_states": result.expanded_states,
        "visited_states": result.visited_states,
        "wall_time_s": result.wall_time_s,
    }

    # 终局节点：弹出的目标优先，其次是待定目标。与 ``mapper._budget_cut`` 同一套判断，
    # 但不重推它——刚才那条 ``_trace_check`` 会验这两者是不是一回事。
    terminal_id = recorder.goal_id if recorder.goal_id is not None else recorder.pending_id

    # 解路径：沿父链把节点标出来（前端的树据此高亮），顺带**用树自己的边拼一份映射**。
    # 这份拼出来的动作序列只用来与 ``result.actions`` 对账，不发给前端——发给前端的是
    # 搜索给的那份，屏上不能出现两个真相。
    path_actions: tuple[Action, ...] | None = None
    if terminal_id is not None:
        walked: list[int] = []
        node_id: int | None = terminal_id
        while node_id is not None:
            walked.append(node_id)
            if node_id == recorder.start_id:
                break
            node_id = nodes[node_id]["parent"]
        if node_id == recorder.start_id:
            walked.reverse()
            actions: list[Action] = []
            for step_id in walked:
                nodes[step_id]["on_solution_path"] = True
                step_action = recorder.actions[step_id]
                if step_action is not None:
                    actions.append(step_action)
            path_actions = tuple(actions)

    # 角色与深度：进过堆的（也就是登记过的）没被展开就是前沿。深度从父链算，父在
    # 更新时会换，所以放在最后。
    frontier: list[int] = []
    for node in nodes:
        if node["expand_order"] > 0:
            node["role"] = "start" if node["id"] == recorder.start_id else "expanded"
        else:
            node["role"] = "frontier"
            frontier.append(node["id"])
        parent_id = node["parent"]
        if parent_id is not None:
            node["depth"] = nodes[parent_id]["depth"] + 1
    frontier.sort()

    evaluation = result.evaluation
    solution = {
        "actions": [action_to_dict(action, m1=scenario.is_m1) for action in result.actions],
        "action_count": len(result.actions),
        "makespan_ns": evaluation.makespan_ns if evaluation else None,
        "peak_vram_bytes": evaluation.peak_vram_bytes if evaluation else None,
        "h2d_bytes": evaluation.h2d_bytes if evaluation else None,
        "evaluation_status": evaluation.status if evaluation else None,
        "evaluation_reason": evaluation.reason if evaluation else None,
        "evaluation": evaluation,
    }
    if scenario.is_m1:
        # 0.2 才有：另一个域的峰值与另外三个方向的字节。``h2d_bytes`` 留在上面不动，它在
        # 0.2 里仍是「dram_to_vram 搬了多少」那一个意思（``EvaluationResult`` 的文档写着），
        # 所以屏头「显存峰值 / H2D」两格在 0.1 与 0.2 上是同一句话。
        #
        # 缺评估结果时这四个键仍然**在**（值为 None）：它们与上面的键是同一屏的同一排
        # 格子，格子在不在不该随评估成败忽有忽无——那会让前端每次都要判「这个格子存在吗」，
        # 而它判错的后果是整排数字错位。
        solution["peak_dram_bytes"] = evaluation.peak_dram_bytes if evaluation else None
        solution["d2h_bytes"] = evaluation.d2h_bytes if evaluation else None
        solution["ssd_read_bytes"] = evaluation.ssd_read_bytes if evaluation else None
        solution["ssd_write_bytes"] = evaluation.ssd_write_bytes if evaluation else None

    block: dict[str, Any] = {
        "verdict": verdict,
        "start_id": recorder.start_id,
        "goal_id": recorder.goal_id,
        "pending_id": recorder.pending_id,
        "nodes": nodes,
        "frontier": frontier,
        "cut_counts": recorder.cut_counts,
        "budget": {
            "max_expanded_states": result.max_expanded_states,
            "wall_time_limit_s": result.wall_time_limit_s,
        },
        "truncated": recorder.truncated,
        "solution": solution,
        # 每个已展开节点的**非法**候选（合法的在 children 里，不存两遍）。去重后的理由
        # 表放在 candidates.messages。
        "candidates": _intern_candidates(scenario, nodes, recorder.states),
        "reconcile": None,
    }
    block["reconcile"] = _trace_check(
        block,
        result,
        path_actions=path_actions,
        complete=not recorder.truncated,
    )
    return block, result


# ---------------------------------------------------------------------------
# 载荷装配
# ---------------------------------------------------------------------------


def ordered_candidates(scenario: Scenario) -> list[Action]:
    """**全部**候选动作，固定顺序。

    顺序就是内核 ``legal_actions`` 用的那一份（``engine`` 里按版本分的
    ``_m0_candidates`` / ``_m1_candidates``）。``legal_actions`` 只给合法集，这一份是
    全集的顺序来源，两者必须用同一套次序，否则屏上「为什么这个不行」会指着一个邻居说。

    **不在这里重排，也不再抄一份。** 以前抄了一份 0.1 的，加 0.2 时没跟上，于是查看器
    一打开 0.2 场景就在 ``mapspace.copy_tensor_ids`` 上抛 ``AttributeError``（0.2 的
    mapspace 没有这个字段）。抄一份就是两个真相，而漂移是静默的——直接拿内核那一份，
    它改名时会在 import 处炸，那是个响亮的失败，比屏上悄悄换一个次序好。
    """
    candidates = _m1_candidates(scenario) if scenario.is_m1 else _m0_candidates(scenario)
    return list(candidates)


def illegal_candidates(scenario: Scenario, state: State) -> list[dict[str, Any]]:
    """**不合法**的候选动作，带内核自己给的理由。

    理由靠 ``transition`` 自己抛：:class:`MappingError` 带公开的 ``code`` / ``message``，
    所以这里读的是契约，不是内部实现。合法的那些**不在这里**——它们在节点的
    ``children`` 里，存两遍就是两个真相。
    """
    rows: list[dict[str, Any]] = []
    for action in ordered_candidates(scenario):
        try:
            transition(scenario, state, action)
        except MappingError as error:
            rows.append(
                {
                    "action": action_to_dict(action),
                    "code": error.code,
                    "message": error.message,
                }
            )
    return rows


def _intern_candidates(
    scenario: Scenario, nodes: list[dict[str, Any]], states: list[State]
) -> dict[str, Any]:
    """把每个已展开节点的非法候选收成一张**去重字符串表**。

    实测（``three_layer_chain-cap1024``）：2206 条非法候选的原文是 463 KB，但只有
    **45 条不同的理由**，去重后 64 KB。少了七倍，所以不搞按需取、不搞懒加载，
    直接烤进载荷——快照模式下没有 Python 侧可以问。
    """
    messages: list[list[str | None]] = []
    seen: dict[tuple[str | None, str], int] = {}
    rows: list[list[Any]] = []
    for node, state in zip(nodes, states):
        if node["expand_order"] <= 0:
            continue  # 没被展开过就没有「为什么不行」可言
        entries: list[list[Any]] = []
        for item in illegal_candidates(scenario, state):
            key = (item["code"], item["message"])
            index = seen.get(key)
            if index is None:
                index = len(messages)
                seen[key] = index
                messages.append([item["code"], item["message"]])
            action = item["action"]
            entries.append(
                [
                    action["kind"],
                    action.get("operation_id") or action.get("tensor_id"),
                    index,
                ]
            )
        rows.append([node["id"], entries])
    return {"messages": messages, "nodes": rows}


def _log_block(
    scenario: Scenario, evaluation: EvaluationResult | None, note: str | None
) -> dict[str, Any]:
    """事件与状态，**直接用 artifacts 的公开序列化器**。

    自己手写事件/状态字典等于给「查看器与磁盘上的 ``events.json`` 早晚对不上」埋雷。
    复用之后载荷就是产物再加坐标：``_state_to_dict`` 会补上 ``StateSnapshot`` 刻意没有
    的 ``action_index``，并把 ``t`` 改名成 ``t_ns``。
    """
    events = events_document(mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=note)
    states = states_document(mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=note)
    return {"events": events["events"], "states": states["states"]}


def payload(
    scenario: Scenario,
    *,
    scenario_path: str | None = None,
    mode: str = "live",
    max_expanded_states: int | None = None,
    wall_time_limit_s: float | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """把一个 scenario 装成整页载荷。

    ``mode`` 是 ``"live"``（本地服务）或 ``"snapshot"``（导出的自包含 HTML）。快照是
    活页面的**死拷贝**：没有 Python 侧可以对话，所以参数编辑与重跑在那一侧必须禁用，
    屏上也要说清楚原因，而不是给一个点了没反应的按钮。

    整页只搜**一次**：树与终局都来自 ``explore()`` 里那趟挂了记录器的 ``search()``。
    这里没有「传一个已经算好的结果进来」的口子——那个口子只会长成第二趟搜索
    （``M1_ACCEPTANCE.md`` §11）。
    """
    block, result = explore(
        scenario,
        max_expanded_states=max_expanded_states,
        wall_time_limit_s=wall_time_limit_s,
        max_nodes=max_nodes,
    )
    evaluation = block["solution"]["evaluation"]
    block["solution"].pop("evaluation", None)

    limits = limits_block(
        scenario,
        max_expanded_states=result.max_expanded_states,
        wall_time_limit_s=result.wall_time_limit_s,
    )
    return {
        "viewer_schema": VIEWER_SCHEMA,
        "generated_by": "mapping/tools/viewer.py",
        "mode": mode,
        "scenario_path": scenario_path,
        "scenario_id": scenario.id,
        "scenario_fingerprint": result.scenario_fingerprint,
        "workload_fingerprint": result.workload_fingerprint,
        "solution": {
            **block["solution"],
            "status": result.status,
            "termination_reason": result.termination_reason,
            "optimality_proven": result.optimality_proven,
            "search": search_block(result),
        },
        "explore": block,
        "graph": graph_block(scenario),
        # 0.2 才有的一块（0.1 是 ``None``，于是 0.1 的载荷一个键都不多）。
        "m1": m1_block(scenario),
        "log": _log_block(scenario, evaluation, None),
        "source": {"limits": limits, "assumptions": assumptions(scenario)},
        "params_echo": params_echo(scenario, result),
        "checks": [],
    }


# ---------------------------------------------------------------------------
# 成本桥：modeling 的逐层预测 → 内核的整数纳秒
# ---------------------------------------------------------------------------

# 示例图里每个算子的 FLOPs：4×4 的权重乘 4 维向量 = 16 次乘加 = 32 FLOP
# （``demo_window_vs_search.py`` 的注释与示例里写死的成本都是这么来的）。
FLOPS_PER_OP = 32.0

# 一组能复现示例那几行成本的硬件参数。带宽 32 kB/s 是刻意取小的：64 B / 32 kB/s 恰好
# 是 1 ms，加上 1 ms 的固定延迟就是示例里的 3 ms。
#
# **这份常量不 import demo**：让一个工具 import 另一个会破坏「包外一把手工具」的纪律，
# 也会把 demo 的数字变成查看器的承重件。重复一份是安全的，因为**它是被检查的**——
# ``--self-check`` 的 B 组断言这组参数能复现 fixture 声明的 ns。两份被检查的常量没问题，
# 两份没人查的才有问题。
REFERENCE_HARDWARE = {
    "gpu_effective_flops": 16_000.0,
    "h2d_bandwidth_bytes_per_s": 32_000.0,
    "h2d_latency_s": 1e-3,
}

# ``layer_costs._assert_no_extra_services`` 逐项点名的那些开关。``needle`` 是那条**原样
# 消息**里的字段路径片段——消息永远是权威：未来 guard 加了字段而这张表里没有，消息照常
# 渲染成散文，只是勾选框不全，是降级不是矛盾。
BRIDGE_TOGGLES: tuple[dict[str, Any], ...] = (
    {"block": "hardware", "field": "cpu_effective_flops", "off": None,
     "needle": "hardware.cpu_effective_flops", "label": "CPU 数学"},
    {"block": "hardware", "field": "global_bytes", "off": 0,
     "needle": "hardware.global_bytes", "label": "全局暂存 global_bytes"},
    {"block": "hardware", "field": "kv_bytes", "off": 0,
     "needle": "hardware.kv_bytes", "label": "KV 常驻 kv_bytes"},
    {"block": "hardware", "field": "workspace_bytes", "off": 0,
     "needle": "hardware.workspace_bytes", "label": "工作区 workspace_bytes"},
    {"block": "hardware", "field": "activation_transfer_bytes", "off": 0,
     "needle": "hardware.activation_transfer_bytes", "label": "激活搬运"},
    {"block": "hardware", "field": "host_staging_bandwidth_bytes_per_s", "off": None,
     "needle": "host_staging_bandwidth_bytes_per_s", "label": "host staging"},
    {"block": "hardware", "field": "state_transfer_bandwidth_bytes_per_s", "off": None,
     "needle": "state_transfer_bandwidth_bytes_per_s", "label": "状态往返搬运"},
    {"block": "hardware", "field": "kv_storage_cached_bandwidth_bytes_per_s", "off": None,
     "needle": "kv_storage_cached_bandwidth_bytes_per_s", "label": "KV 分层：缓存带宽"},
    {"block": "hardware", "field": "kv_storage_uncached_bandwidth_bytes_per_s", "off": None,
     "needle": "kv_storage_uncached_bandwidth_bytes_per_s", "label": "KV 分层：未缓存带宽"},
    {"block": "hardware", "field": "kv_storage_cache_bytes", "off": 0,
     "needle": "kv_storage_cache_bytes", "label": "KV 分层：缓存容量"},
    {"block": "state", "field": None, "off": None,
     "needle": "state（每层 KV", "label": "每层 KV / SSM 状态"},
    {"block": "policy", "field": "static_gpu_layers", "off": 0,
     "needle": "policy.static_gpu_layers", "label": "静态常驻层"},
    {"block": "policy", "field": "token_pipeline_overhead_s", "off": 0.0,
     "needle": "policy.token_pipeline_overhead_s", "label": "每 token 控制开销"},
    {"block": "policy", "field": "layer_scheduler_overhead_s", "off": 0.0,
     "needle": "policy.layer_scheduler_overhead_s", "label": "层调度开销"},
)

# 三类拒绝的补救方向**不一样**，屏上必须分开说。只讲「未迁移」那一类，用户关掉所有开关
# 之后撞上第二道门时会以为查看器坏了。
REFUSAL_REMEDY = {
    "unsupported": "补救：把这些服务**逐项关掉**（下面每一行都能点）。它们在本轮的动作空间里没有对应动作。",
    "invalid": "补救：**改数字或改图**。这一道门与迁移无关——它是「浮点秒 → 整数纳秒」的分辨率损失本身。",
    "value": "补救：**改参数**。这是 frozen dataclass 自己的不变式拒绝了手填的值。",
    # 第四类不是门：三道门都是 ``costs_from_model_config`` 在拒绝，这一类是**还没走到门口**
    # ——配置根本没读进来。所以措辞上不说「门 4」，也不给「换成参考硬件」（那也要先有个配置）。
    "config": "补救：**检查 ``--config`` 指的文件**。这是三道门之前的一步——配置没读进来，"
              "所以开关清单是空的，那不是坏了。",
    # 第五类同样不是门，也同样刻意不编号。三道门说的都是「有一份能对齐的成本表，但这一份
    # 过不去」；这一类是**没有可对齐的表**——0.2 的成本表与桥产的那份不同构。补救方向因此
    # 不是改参数、也不是关服务，而是「这一屏在 0.2 上没有要展示的东西」。
    "m1": "补救：**不用补救**。0.2 的时长本来就在 scenario 自带的成本表里，时间线、计算图、"
          "状态图照常画；这一屏只是没有「从硬件参数推一遍」这条对照。",
}

REFUSAL_CLASS_LABEL = {
    "unsupported": "Unsupported · 未迁移的服务",
    "invalid": "InvalidInput · 单位分辨率 / 图形状",
    "value": "ValueError · 参数不变式",
    "config": "配置未载入 · 路径或内容",
    "m1": "0.2 的两级成本表 · 成本桥只产 0.1 的一级形状",
}


def _sole_weight(workload: Any, operation_id: str) -> tuple[str | None, int]:
    """这个算子的唯一权重输入。返回 ``(权重 id | None, 权重份数)``。

    份数一起返回是因为「0 份」与「2 份」都要报出来，而它们都不是 ``None`` 能表达的。
    """
    operation = workload.operation_by_id[operation_id]
    weights = [
        tensor_id
        for tensor_id in operation.inputs
        if workload.tensor_by_id[tensor_id].role == ROLE_WEIGHT
    ]
    return (weights[0] if len(weights) == 1 else None), len(weights)


def derive_layers(scenario: Scenario, *, flops_per_op: float = FLOPS_PER_OP) -> tuple[
    dict[str, Any], list[dict[str, Any]]
]:
    """算子 → :class:`LayerSpec`，**weight_bytes 从 scenario 推**。

    ``demo_window_vs_search.py:81`` 写死了 ``WEIGHT_BYTES = 64``，但 ``matvec`` 的权重是
    **48 字节**——照抄常量会让五个示例里的一个静默算错。从张量的 ``storage_bytes`` 推，
    ``costs_for_chain`` 的字节一致性检查就此恒真，还消掉一整类伪拒绝。

    返回 ``(layers, problems)``。``problems`` 非空表示图的**形状**不在「一个计算节点恰好
    一份权重」这个范围内（``fork``/``residual`` 的 0 权重 ADD 算子）。这一条
    ``costs_for_chain`` 自己也会查（它的条件 2），但那样报出来的是「键集合缺少 ['add']」，
    指向的是配对而不是图形状。所以这里先查一遍，好把话说准。
    """
    from llm_infer_model.model import LayerSpec

    workload = scenario.workload
    layers: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    for index, operation_id in enumerate(sorted(workload.operation_by_id)):
        weight_id, weight_count = _sole_weight(workload, operation_id)
        if weight_id is None:
            problems.append(
                {
                    "operation_id": operation_id,
                    "weight_count": weight_count,
                    "inputs": list(workload.operation_by_id[operation_id].inputs),
                }
            )
            continue
        layers[operation_id] = LayerSpec(
            name=f"blk.{index}",
            weight_bytes=workload.tensor_by_id[weight_id].storage_bytes,
            flops=flops_per_op,
        )
    return layers, problems


def apply_toggle(config: Any, block: str, field: str | None, value: Any) -> Any:
    """把一个开关换成 off 值，两层 ``dataclasses.replace``。

    ``replace`` 会重跑 ``__post_init__``，所以手填的非法参数会当场 ``ValueError``——
    这不是要绕开的麻烦，它本身就是要显示的一类拒绝（门 3）。
    """
    from dataclasses import replace

    if block == "state":
        return replace(config, state=None)
    section = getattr(config, block)
    return replace(config, **{block: replace(section, **{field: value})})


def _toggle_rows(config: Any, message: str) -> list[dict[str, Any]]:
    """每一行开关当前是 on 还是 off。

    判据是**那条原样消息里有没有这个词**（消息是权威），另附上配置里**现在**的值——
    用户在屏上点掉一行之后，看得到值确实变了。两者不一致时两栏并排，一眼能看出。
    """
    rows: list[dict[str, Any]] = []
    for toggle in BRIDGE_TOGGLES:
        block, field = toggle["block"], toggle["field"]
        current = getattr(config, block) if field is None else getattr(getattr(config, block), field)
        rows.append(
            {
                "block": block,
                "field": field,
                "label": toggle["label"],
                "off_value": toggle["off"],
                "off_is_null": toggle["off"] is None,
                "enabled": toggle["needle"] in message,
                "value_now": None if current is None else (
                    current if isinstance(current, (int, float, str)) else str(current)
                ),
                "value_is_null": current is None,
            }
        )
    return rows


def _refusal(
    scenario: Scenario,
    config: Any,
    kind: str,
    *,
    code: str | None,
    message: str,
    stage: str,
    layers: dict[str, Any] | None = None,
    problems: list[dict[str, Any]] | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "ok": False,
        "refusal_class": kind,
        "refusal_label": REFUSAL_CLASS_LABEL[kind],
        "code": code,
        "message": message,
        "stage": stage,
        "remedy": REFUSAL_REMEDY[kind],
        # 「来源」那一栏的措辞。**只有 0.2 那一类给**：它要说的是「没有可对齐的表」，而
        # 三四道门那句「modeling 推不出来」套上去是反的（0.2 的时长推得出来，就在场景里）。
        # 缺省 ``None`` 时由 :func:`sourced_scenario` 按老办法拼。
        "source_label": source_label,
        "toggles": _toggle_rows(config, message) if config is not None else [],
        "structural_problems": problems or [],
        "layers": [
            {
                "operation_id": operation_id,
                "name": layer.name,
                "weight_bytes": layer.weight_bytes,
                "flops": layer.flops,
                "weight_tensor_id": _sole_weight(scenario.workload, operation_id)[0],
            }
            for operation_id, layer in sorted((layers or {}).items())
        ],
        "table": [],
        "h2d": [],
        "all_agree": None,
        # 见 ``bridge_table`` 的 docstring：装的是推导出来的 ``Costs``，由
        # ``sourced_scenario`` pop 掉。拒绝这一路上没有东西可装，但键要在，
        # 好让调用方无条件 pop。
        "_costs": None,
    }


def bridge_table(
    scenario: Scenario,
    *,
    config: Any,
    config_error: str | None = None,
    flops_per_op: float = FLOPS_PER_OP,
    reference: bool = False,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 modeling 的逐层预测换算成内核的整数纳秒动作，并把**被拒绝的那些项**也报出来。

    这是「时长来源」那一屏的全部数据。它只用公开 API：``costs_from_model_config``、
    ``Unsupported``/``InvalidInput``/``ValueError`` 三种异常、以及 scenario 自己的成本表。
    **永不** import 私有的 ``_assert_no_extra_services``——那条 guard 的消息就是给用户看的
    文本，读它、渲染它，但不复制它的判断。

    ``reference=True`` 时把硬件换成 :data:`REFERENCE_HARDWARE`（VRAM 仍取 scenario 的），
    仍然走 ``costs_from_model_config``，所以门 1 还在链路上——参考硬件不是绕过 guard 的
    后门，它只是换一组能让数字落到纳秒分辨率之上的参数。

    ``hardware`` 在生效硬件之上**逐字段**覆盖那三个数（FLOP/s、带宽、延迟）。这是屏上那组
    旋钮的入口，也是「联动」唯一看得见的地方：参考硬件那组恰好复现示例写死的值，不动它，
    时间线上的宽度一个像素都不会变。

    成功的返回值里带一个 ``_costs`` 键，装的是**推导出来的那个** :class:`Costs` 对象。
    它不是 JSON 能序列化的，所以 :func:`sourced_scenario` 会把它 pop 掉——留着就会在
    ``render_snapshot`` 的 ``json.dumps`` 上炸。之所以夹带而不是让调用方再推一次，是因为
    这里已经把层、硬件、异常路径都摆好了，重推一遍等于把同一条链路走两次，将来只会漂。
    """
    from llm_infer_model.tensor.layer_costs import costs_from_model_config
    from llm_infer_model.tensor.spec import InvalidInput, Unsupported

    if scenario.is_m1:
        # 0.2 的成本表**不同构**，不是「算不出来」：0.1 是 ``compute[op]`` / ``h2d[tensor]``
        # 两个一维表，0.2 是 ``compute[op][device]`` / ``transfer[route][tensor]`` 两个二维表
        # （``M1Costs`` 的文档写着为什么另起一个类型而不是把字段摊平）。而这道桥的下游
        # ``costs_for_chain`` 产的是 0.1 那一份，也没有 0.2 的对应物——把它硬塞进来就得在
        # 这里替 CPU/GPU 挑一个 flops、替四条路线挑一组带宽，那是**建模决策**，不是查看器
        # 该做的。所以这一屏在 0.2 上如实说「没有可推导的东西」，而时间线、计算图、状态图
        # 用的仍是 scenario 自带的 0.2 成本表（内核从它算出 ``log.events``），一屏都不缺。
        #
        # 这一支排在 ``--config`` 前面：形状不对是**还没走到门口**就看出来的事，跟配置读没读
        # 进来无关。与第四类一样刻意不编号——三道门都是 ``costs_from_model_config`` 在拒绝。
        return _refusal(
            scenario, config, "m1",
            code=None,
            message=(
                "这份 0.2 场景自带**逐（算子,设备）与逐（路线,张量）**的成本表，而成本桥"
                "（``costs_from_model_config`` / ``costs_for_chain``）产的是 0.1 那份一维形状："
                "算子和搬运各一个时长。两者不同构，桥没有可对齐的字段。"
                "要把 0.2 的时长也从硬件参数推出来，得先定 CPU/GPU 各自的有效算力与四条路线"
                "各自的带宽——那是建模决策，不在查看器里做。"
            ),
            stage="scenario.costs 的形状（M1Costs vs Costs）",
            source_label="scenario 声明值（0.2 的两级成本表，成本桥不产这一形状）",
        )

    if config is None and config_error is not None:
        # ``--config`` 指的文件没读进来（``main()`` 的选择是「打印失败、继续跑」，因为
        # scenario 那边还活着，图和时间线照常能看）。
        #
        # 不加这一支的后果实测过：``costs_from_model_config(None, …)`` 会在
        # ``config.hardware`` 上抛 AttributeError——不在下面三个 except 里，于是**整个
        # 查看器**（连和时长来源无关的屏一起）死在一条 traceback 上，而上面刚打印的那句
        # 「✗ 载入 config 失败」反倒成了噪音。``_refusal`` 本来就容得下 ``config=None``
        # （开关清单为空），缺的只是别往下走。
        #
        # 注意这一支只接「给了但读不出来」。**没给** ``--config`` 是另一回事：那不是失败，
        # 是没要，下面照样用参考硬件推——否则默认打开查看器时时长来源永远是空的。
        return _refusal(
            scenario, config, "config",
            code=None,
            message=config_error,
            stage="--config 指的文件",
        )
    if config is None:
        reference = True

    effective: Any = config
    try:
        # 层的构造也在 try 里：``LayerSpec(flops=0)`` 会当场抛 ValueError，那正是门 3
        # 可被触发的一条真实路径（屏上那个 flops_per_op 旋钮）。
        layers, problems = derive_layers(scenario, flops_per_op=flops_per_op)
        if problems:
            detail = "；".join(
                f"算子 {p['operation_id']} 有 {p['weight_count']} 份权重输入"
                for p in problems
            )
            return _refusal(
                scenario,
                effective,
                "unsupported",
                code=None,
                message=(
                    f"{detail}。成本桥接只支持「一个计算节点恰好一份权重」，不做任何平均摊派，"
                    "所以这张图（分叉 / 残差汇合）没有对应的层。通用搜索照常支持这类图——"
                    "受限的是成本桥，不是内核。"
                ),
                stage="graph ↔ layers 的配对",
                layers=layers,
                problems=problems,
            )
        effective = _effective_config(
            scenario, config, layers, reference=reference, hardware=hardware
        )
        derived = costs_from_model_config(effective, scenario.workload, layers)
    except Unsupported as error:  # 门 1/3：未迁移的服务
        return _refusal(
            scenario, effective, "unsupported",
            code=getattr(error, "code", None), message=str(error),
            stage="costs_from_model_config", layers=None,
        )
    except InvalidInput as error:  # 门 2/3：单位分辨率 / 图形状
        return _refusal(
            scenario, effective, "invalid",
            code=getattr(error, "code", None), message=str(error),
            stage="costs_from_model_config", layers=None,
        )
    except ValueError as error:  # 门 3/3：frozen dataclass 自己的不变式
        return _refusal(
            scenario, effective, "value",
            code=None, message=f"{error}（{type(error).__name__}）",
            stage="LayerSpec / dataclasses.replace", layers=None,
        )

    table = [
        {
            "operation_id": operation_id,
            "model_ns": derived.compute[operation_id].duration_ns,
            "declared_ns": scenario.costs.compute[operation_id].duration_ns,
            "agrees": derived.compute[operation_id].duration_ns
            == scenario.costs.compute[operation_id].duration_ns,
        }
        for operation_id in sorted(derived.compute)
    ]
    h2d = [
        {
            "tensor_id": tensor_id,
            # 字节数带上屏，是因为搬运时长那一行的算式是
            # ``延迟 + 字节 ÷ 带宽``——不给字节，那一行就只能看不能验。
            # 从张量自己推（**不是**抄常量）：``matvec`` 的权重是 48 字节，
            # 而 ``demo_window_vs_search.py:81`` 写死的 64 只对另外四个示例成立。
            "bytes": scenario.workload.tensor_by_id[tensor_id].storage_bytes,
            "model_ns": derived.h2d[tensor_id].duration_ns,
            "declared_ns": scenario.costs.h2d[tensor_id].duration_ns,
            "agrees": derived.h2d[tensor_id].duration_ns == scenario.costs.h2d[tensor_id].duration_ns,
        }
        for tensor_id in sorted(derived.h2d)
    ]
    return {
        "kind": "ok",
        "ok": True,
        "refusal_class": None,
        "refusal_label": None,
        "code": None,
        "message": None,
        "stage": "costs_from_model_config",
        "remedy": None,
        "toggles": _toggle_rows(effective, ""),
        "structural_problems": [],
        "layers": [
            {
                "operation_id": operation_id,
                "name": layer.name,
                "weight_bytes": layer.weight_bytes,
                "flops": layer.flops,
                "weight_tensor_id": _sole_weight(scenario.workload, operation_id)[0],
            }
            for operation_id, layer in sorted(layers.items())
        ],
        "table": table,
        "h2d": h2d,
        "all_agree": all(row["agrees"] for row in (*table, *h2d)),
        "hardware": {
            "gpu_effective_flops": effective.hardware.gpu_effective_flops,
            "h2d_bandwidth_bytes_per_s": effective.hardware.h2d_bandwidth_bytes_per_s,
            "h2d_latency_s": effective.hardware.h2d_latency_s,
            "vram_capacity_bytes": effective.hardware.vram_capacity_bytes,
        },
        "reference_used": reference,
        "_costs": derived,
    }


def _effective_config(
    scenario: Scenario,
    config: Any,
    layers: dict[str, Any],
    *,
    reference: bool,
    hardware: dict[str, Any] | None,
) -> Any:
    """这次求值实际用的 :class:`ModelConfig`。

    三条路，**分开**是关键：``config is None`` 不是「坏了」，是「没要」——现造一份除硬件外
    一切服务都关着的配置，时长来源才默认是 modeling。写成 ``replace(config, …)`` 会在
    ``None`` 上当场 ``TypeError``（实测踩到），而那个异常不在 ``bridge_table`` 的三个
    ``except`` 里，整页会死在一条 traceback 上。

    ``hardware`` 是**逐字段**覆盖，不是整份换掉：``HardwareSpec`` 还有十几个字段
    （``kv_bytes``、state 带宽…），屏上那三个旋钮只说得了三个；整份换会把其余的静默清零，
    而「清零」正好是门 1 的判据——等于用一次界面操作把拒绝清单伪造掉。
    """
    from dataclasses import replace

    from llm_infer_model.model import HardwareSpec, ModelConfig, PolicySpec

    override = {**REFERENCE_HARDWARE, **(hardware or {})}
    if config is None:
        # ``PolicySpec()`` 的默认值必须全都是「关」：guard 逐项点名的那几个字段一旦非零
        # 就会把这条默认路径也拒掉（参考硬件**不**豁免它们——实测：拿 toy 配置配参考硬件
        # 时仍然被 ``policy.static_gpu_layers`` 拒，因为换硬件换不掉策略）。
        # ``window_size`` 是唯一没有默认值的字段，而 ``costs_from_model_config`` **不读它**
        # （``layer_costs.py`` 的注释明写：窗口尺寸与逐层成本无关），所以填 1 就行。
        return ModelConfig(
            name=scenario.id,
            layers=tuple(layers.values()),
            hardware=HardwareSpec(
                vram_capacity_bytes=scenario.architecture.vram_capacity_bytes, **override
            ),
            policy=PolicySpec(window_size=1),
        )
    if reference:
        return replace(
            config,
            hardware=HardwareSpec(
                vram_capacity_bytes=scenario.architecture.vram_capacity_bytes, **override
            ),
        )
    if hardware:
        return replace(
            config,
            hardware=replace(
                config.hardware, **{key: float(value) for key, value in hardware.items()}
            ),
        )
    return config


def sourced_scenario(
    scenario: Scenario,
    bridge: dict[str, Any],
    *,
    edited: bool = False,
) -> tuple[Scenario, dict[str, Any]]:
    """按「时长来源」决定喂给内核的那一份 scenario。

    **这一步就是「联动」本身。** 内核算出来的 makespan、峰值、事件、以及时间线上每一根
    柱子的宽度，全部是从这份 ``Costs`` 推出来的；把它换成 :func:`bridge_table` 推出来的
    那一份，屏上的一切就跟着 modeling 走。``examples/*.json`` 一个字节都不动——换的只是
    内存里的这一份，所以 ``M0_VERIFICATION.md`` 里那些逐字节记录仍然对得上它自己记的
    scenario。

    两条**必须**退回声明值的路，都不是保守，是正确性：

    * ``edited``——用户在参数页手改过某个算子的 ``duration_ns``。那些改动手写在
      ``scenario.costs`` 上，而推导出来的 ``Costs`` 会把它整份换掉。不退回的话，「改了参数」
      和「改了没反应」在屏上长得一模一样，而表单还会照常回显用户填的数。
    * 推导失败（结构性 / 门 1 / 门 2 / 门 3 / 配置没载入）——没有东西可换。

    两条都带上原样原因，屏上要写出来：一个静默回退的来源面板，比一个空来源面板更坏。
    """
    from dataclasses import replace

    derived = bridge.pop("_costs", None)
    rows = [
        {"kind": "compute", "id": row["operation_id"], **row} for row in bridge.get("table", [])
    ] + [{"kind": "h2d", "id": row["tensor_id"], **row} for row in bridge.get("h2d", [])]
    disagreements = [
        {"kind": row["kind"], "id": row["id"],
         "model_ns": row["model_ns"], "declared_ns": row["declared_ns"]}
        for row in rows
        if not row["agrees"]
    ]

    block: dict[str, Any] = {
        "rows": rows,
        "disagreements": disagreements,
        "source": getattr(derived, "source", None) if derived is not None else None,
    }

    if edited:
        # 「手改过」在最前面：这一条与推导成功与否无关。推导失败时退回声明值本来就对，
        # 但退回的**理由**是「你改过」，不是「推不出来」——两者的补救方向完全相反。
        block.update(
            kind="declared", swapped=False, reason_class="edited",
            label="scenario 声明值（你在参数页手改过）",
            message="参数页里改过的算子时长写在 scenario 的成本表上，所以这一份用的是你的值，"
                    "不是 modeling 推的。想回到推导值，把那个框改回原数（或刷新页面）。",
            remedy="补救：把参数页里手改过的 ``duration_ns`` 改回去，来源会自动回到推导。",
        )
        return scenario, block

    if derived is None:
        # 「推不出来」这个说法只对三道门与「配置没载入」成立。0.2 那一类**不是**推不出来：
        # 0.2 的时长就在 scenario 自带的表里，只是那张表与桥产的不同构，没有可对齐的字段。
        # 措辞由 ``bridge`` 自己给（``source_label``），别在这里替它猜。
        block.update(
            kind="declared", swapped=False,
            reason_class=bridge.get("kind"),
            label=bridge.get("source_label")
            or f"scenario 声明值（modeling 推不出来：{bridge.get('refusal_label') or bridge.get('kind')}）",
            message=bridge.get("message"),
            remedy=bridge.get("remedy"),
        )
        return scenario, block

    block.update(
        kind="model", swapped=True, reason_class=None,
        label="modeling 推导（llm_infer_model.layer_costs）",
        message=None,
        remedy=None,
    )
    return replace(scenario, costs=derived), block


def params_echo(scenario: Scenario, result: SearchResult) -> dict[str, Any]:
    """实际生效的旋钮值，原样回显。

    屏上的表单是**从这里**生成的，不是前端写死的一份。否则改了一个旋钮、表单却还显示
    旧值，用户会以为没生效。

    两个版本的键各写各的，不共用一个「差不多」的形状：0.1 的 ``copy_tensor_ids`` 在
    0.2 里根本不存在（0.2 没有那张清单，可搬性按路线问），而 0.2 的三样东西（DRAM
    预算、两个域各自的预留、四个资源的槽位）在 0.1 里也没有。共用一个形状就得有一半
    的键在另一半里是假的。
    """
    common = {
        "vram_capacity_bytes": scenario.architecture.vram_capacity_bytes,
        "max_expanded_states": result.max_expanded_states,
        "wall_time_limit_s": result.wall_time_limit_s,
        "allow_copy_compute_overlap": scenario.mapspace.allow_copy_compute_overlap,
        "allow_eviction": scenario.mapspace.allow_eviction,
    }
    if scenario.is_m1:
        architecture = scenario.architecture
        return {
            **common,
            "dram_capacity_bytes": architecture.dram_capacity_bytes,
            "dram_runtime_reserved_bytes": architecture.dram_runtime_reserved_bytes,
            "vram_runtime_reserved_bytes": architecture.reserved_of(LOC_VRAM),
            "cpu_compute_slots": architecture.cpu_compute_slots,
            "gpu_compute_slots": architecture.gpu_compute_slots,
            "pcie_copy_slots": architecture.pcie_copy_slots,
            "ssd_io_slots": architecture.ssd_io_slots,
            # 0.2 的计算时长按（算子，设备）两级索引，所以这里也是两级——摊平成一个
            # ``op`` 键就得分不清哪台机器上的那个数。
            "compute_ns": {
                operation_id: {
                    device: cost.duration_ns for device, cost in sorted(by_device.items())
                }
                for operation_id, by_device in sorted(scenario.costs.compute.items())
            },
            "transfer_ns": {
                route: {
                    tensor_id: cost.duration_ns for tensor_id, cost in sorted(by_tensor.items())
                }
                for route, by_tensor in sorted(scenario.costs.transfer.items())
            },
        }
    return {
        **common,
        "runtime_reserved_bytes": scenario.architecture.runtime_reserved_bytes,
        "allocation_alignment_bytes": scenario.architecture.allocation_alignment_bytes,
        "copy_tensor_ids": list(scenario.mapspace.copy_tensor_ids),
        "compute_ns": {
            operation_id: cost.duration_ns
            for operation_id, cost in sorted(scenario.costs.compute.items())
        },
        "h2d_ns": {
            tensor_id: cost.duration_ns for tensor_id, cost in sorted(scenario.costs.h2d.items())
        },
    }
