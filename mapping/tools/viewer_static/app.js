/* Tensor Mapping 查看器 —— 全部前端逻辑。手写 SVG，不引任何框架。
 *
 * 三条纪律，改这个文件之前先读：
 *
 * 1. **一套前端两种模式，分叉只在本文件末尾一处。** 只在那里判断
 *    ``window.__BAKED__``：在就直接用（导出的快照），不在就 fetch（本地服务）。
 *    别处**永不 fetch**，所以导出快照只需要把 style.css 和 app.js 内联进去，
 *    前端零改动、不分叉、不需要模板。
 *
 * 2. **只有一个「当前步」游标，五块屏全从它派生，且每帧五块屏都重绘。** ``render()`` 是
 *    唯一重绘出口。加一块屏时接上 ``render()``，不要自己另起一套更新路径——那样迟早出现
 *    「图上是第 7 步、甘特是第 9 步」这种看起来像内核错了的假象。
 *    五块屏**全部常驻在同一页上**（栏上那排是跳转条，不是标签页）：原先用 ``pane.hidden``
 *    切显隐，而 ``.pane`` 是 ``display:flex``、样式表里又没有 ``[hidden]`` 规则，作者样式
 *    压过浏览器默认的 ``[hidden]{display:none}``——于是那个属性一直是**失效**的，屏幕上
 *    从来就是五块并排、其中四块空着（``render()`` 每帧只填一块）。「第一次打开感觉很鸡肋」
 *    说的就是这个，不是「切换标签太麻烦」。
 *
 * 3. **回放与单步必须视觉上分开。** 回放的游标走**烤好的计划**（log.states 的下标），
 *    单步的游标是一条**分支路径**（explore.nodes 里的 id 数组）。整数游标表达不了
 *    「已经离开计划分叉出去」这件事，所以单步用数组。混同这两种模式正是这类工具
 *    开始骗人的起点。
 *
 * 4. **图上的左右是拓扑层，上下是先后。** 这一条改过两次，最后是用户拍的板：先是
 *    「x = 执行顺序、y = 拓扑秩」，被他否掉（同一个算子的父张量被拆到两列、箭头一长一短，
 *    纵轴还按层各占一行，图又高又散）；现在 x = 拓扑层（张量贴到消费它的算子左边一列），
 *    y = 先后（先加载的在上、算子与它的输出同一行）。规则、例外与它掉出来的紧凑度都在
 *    ``graphLayout()`` 的注释里。排布基准仍取**解路径**而非当前游标——按游标实时重排会让
 *    拖动走带时整张图不停跳位。
 */
'use strict';

const S = {
  payload: null,
  pane: 'graph',       // 跳转条上高亮的那一屏（跟滚动位置走）
  mode: 'replay',      // 'replay' | 'step'
  cursor: 0,           // 回放：log.states 的下标
  path: [],            // 单步：explore 节点 id 的数组，起点 [start_id]
  playing: false,
  timer: null,
  compare: null,       // {before, after} 上一次参数重跑的结果
  bridgeMsg: null,     // 422 回来的门 3 拒绝
  treeDisabled: false, // 走查与 search() 对不上：不画一棵可能是错的树
};

const ACTION_LABEL = {
  COPY_H2D: 'COP',
  COMPUTE: 'CMP',
  EVICT: 'EV',
  ADVANCE: 'ADV',
};

/* ------------------------------------------------------------------ 小工具 */

function esc(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key in (attrs || {})) {
    if (attrs[key] != null) node.setAttribute(key, attrs[key]);
  }
  if (text != null) node.textContent = String(text);
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function byId(id) { return document.getElementById(id); }

/** 整数纳秒 → 人读的字符串。mapping 侧的**唯一**单位，全程不换算成秒。 */
function fmtNs(ns) {
  if (ns == null) return '—';
  const n = Number(ns);
  if (!isFinite(n)) return '—';
  if (n === 0) return '0 ns';
  if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 3) + ' ms';
  if (n >= 1e3) return (n / 1e3).toFixed(n % 1e3 === 0 ? 0 : 2) + ' µs';
  return n + ' ns';
}

function fmtBytes(b) {
  if (b == null) return '—';
  const n = Number(b);
  if (!isFinite(n)) return '—';
  if (n >= 1024 * 1024) return (n / 1048576).toFixed(2) + ' MiB';
  if (n >= 1024) return (n / 1024).toFixed(1) + ' KiB';
  return n + ' B';
}

function num(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString('en-US');
}

/** SVG 文字宽度的一个**偏大**估计，单位 px。
 *
 * 为什么需要它：状态时间线右边那一列要**紧跟在各自那根条后面**（用户的原话是「把字放到阶段的
 * 傍边」），条的位置随时刻变，所以右列不能再写死一个宽度——得知道这份载荷里最长的那条说明有
 * 多宽，才能把画布留够。而垫片里**没有排版引擎**（`getBBox()` 在 node 里不存在），宽度只能自己算。
 *
 * 宁可估大不可估小：估小了字会被 SVG **静默裁掉**（`overflow: hidden`，不报错、不留痕），
 * 正是这个仓库反复踩的那一类 bug。
 *
 * 字号 14px 来自 `style.css` 的 `body { font: 14px/1.5 … }` —— SVG 的 `<text>` 继承它，
 * 而 `style.css` 里**没有任何规则**改过 SVG 文字的字号（没有 `.axis`、没有 `svg text`）。
 * 真值参考：数字 ≈ 0.55em、空格 ≈ 0.28em、大写 ≈ 0.68em，所以非全角取 0.62em 是上界。 */
const TEXT_FONT_PX = 14;
function textWidth(text) {
  let em = 0;
  for (const ch of String(text)) em += ch.codePointAt(0) >= 0x2e80 ? 1 : 0.62;
  return em * TEXT_FONT_PX;
}

/* 动作字典 → 一行标签。target 是算子 id 或张量 id，两者不会同名（内核保证）。 */
function actionLabel(action) {
  if (!action) return '（起点）';
  const kind = action.kind;
  const target = action.operation_id || action.tensor_id || '';
  return ACTION_LABEL[kind] + (target ? ' ' + target : '');
}

function actionKey(action) {
  if (!action) return 'null';
  return action.kind + '|' + (action.operation_id || action.tensor_id || '');
}

/* --------------------------------------------------- 当前步：唯一的真相来源 */

function isReplay() { return S.mode === 'replay'; }

/** 当前步对应的状态快照（t_ns / used_vram_bytes / op_status / copy_status）。 */
function currentSnapshot() {
  const payload = S.payload;
  if (isReplay()) {
    const states = payload.log.states;
    if (!states.length) return null;
    return states[Math.min(S.cursor, states.length - 1)];
  }
  const node = currentNode();
  return node || null;
}

function currentNode() {
  const payload = S.payload;
  if (!S.path.length) return null;
  return payload.explore.nodes[S.path[S.path.length - 1]] || null;
}

/** 单步模式下的行走记录：每一步走了哪条边。 */
function stepHops() {
  const nodes = S.payload.explore.nodes;
  const hops = [];
  for (let i = 1; i < S.path.length; i++) {
    const prev = nodes[S.path[i - 1]];
    const next = nodes[S.path[i]];
    hops.push({ from: prev, to: next, action: next.action });
  }
  return hops;
}

function stepLength() {
  return isReplay()
    ? Math.max(0, S.payload.log.states.length - 1)
    : Math.max(0, S.path.length - 1);
}

/* ------------------------------------------------------------------ 顶栏 */

/** 横幅只有一条。写的时候带一个 ``tag``，清的时候只清自己那条——否则某个渲染器的
 *  「清除」会把另一个渲染器的错误顺手抹掉，于是失败变成静默的。 */
function setBanner(tag, text, cls) {
  const banner = byId('banner');
  banner.hidden = false;
  banner.className = 'banner' + (cls ? ' ' + cls : '');
  banner.textContent = text;
  banner.dataset.tag = tag;
}

function clearBanner(tag) {
  const banner = byId('banner');
  if (banner.dataset.tag !== tag) return;
  banner.hidden = true;
  banner.textContent = '';
  delete banner.dataset.tag;
}

function verdictTone(payload) {
  const status = payload.explore.verdict.status;
  // feasible / unknown **不是**证明。措辞必须让这一点无法被误读。
  if (status === 'optimal') return { cls: 'ok', text: '最优（已证明）' };
  if (status === 'infeasible') return { cls: 'bad', text: '无解（已证明）' };
  if (status === 'feasible') return { cls: 'warn', text: '可行，但未证明最优' };
  if (status === 'unknown') return { cls: 'warn', text: '未知（预算耗尽）' };
  return { cls: '', text: String(status || '—') };
}

function renderHeader() {
  const payload = S.payload;
  byId('scenario-id').textContent = payload.scenario_id;
  byId('scenario-path').textContent = payload.scenario_path || '（内存里构造的）';

  const mode = byId('mode-badge');
  const live = payload.mode === 'live';
  mode.className = 'badge ' + (live ? 'live' : 'snapshot');
  mode.textContent = live ? '本地服务' : '导出快照 · 只读';

  const verdict = verdictTone(payload);
  const reconcile = payload.explore.reconcile;
  // 标签一律用中文，并在 ``title`` 里挂上**载荷里的字段名**——屏上要能读，读产物时要能对上。
  // 「展开状态」这种写法最坑：它其实是个**计数**，看着却像一句状态说明。
  const rows = [
    { k: '总时长', t: 'solution.makespan_ns', v: fmtNs(payload.solution.makespan_ns) },
    { k: '峰值显存', t: 'solution.peak_vram_bytes', v: fmtBytes(payload.solution.peak_vram_bytes) },
    { k: 'H2D 搬运', t: 'solution.h2d_bytes', v: fmtBytes(payload.solution.h2d_bytes) },
    { k: '动作数', t: 'solution.action_count', v: num(payload.solution.action_count) },
    { k: '展开的状态数', t: 'explore.verdict.expanded_states', v: num(payload.explore.verdict.expanded_states) },
  ];

  const head = byId('headline');
  clear(head);
  const status = document.createElement('div');
  status.innerHTML = '<dt title="explore.verdict.status">搜索结果</dt><dd><span class="badge ' +
    verdict.cls + '">' + esc(verdict.text) + '</span></dd>';
  head.appendChild(status);

  for (const row of rows) {
    const div = document.createElement('div');
    div.innerHTML = '<dt title="' + esc(row.t) + '">' + esc(row.k) + '</dt><dd>' +
      esc(row.v) + '</dd>';
    head.appendChild(div);
  }

  // 对账：走查与 search() 逐字段一致才画树。不一致时红条 + **禁掉树那一屏**。
  // 禁用必须落在屏上，不能落在标签按钮上：标签栏现在是跳转条，按钮一禁用，用户连
  // 「为什么禁」都点不到——屏还在，里面写着原因，这才是能读的失败。
  // 形状是 {checked, agrees, differences}——``checked: false`` 表示压根没对（没传
  // SearchResult），那不是失败，别报成红的。
  const bad = reconcile && reconcile.checked && !reconcile.agrees;
  S.treeDisabled = !!bad;
  if (bad) {
    setBanner('reconcile', '走查与 mapper.search() 对不上：' +
      (reconcile.differences || []).join('；') +
      '　搜索树屏已禁用（不画一棵可能是错的树）。手动单步不受影响。');
  } else {
    clearBanner('reconcile');
  }

  renderScenarioPicker();
}

/** 顶栏那格「换计算图」。
 *
 *  名单由服务端给（``payload.scenarios``，按当前这份所在的目录扫出来），一次会话里不会变，
 *  所以选项**只建一次**：``render()`` 每帧都跑，每帧重建 ``<select>`` 会把正要展开的下拉
 *  关掉（和参数表单那个焦点问题同一类，只是这里连 ``activeElement`` 都保不住）。之后每帧
 *  只同步三件小事——选中值、禁用状态、要不要露出来。
 *
 *  选项里没有当前那一份时把选中值置空，**绝不退而显示第一个选项**：``<select>`` 的 value
 *  不在选项里时浏览器就是那么干的，于是顶栏会理直气壮地写着另一张图的名字。服务端已经把
 *  当前这份补进名单（``Viewer.catalog_block``），这里只是别把那句话变成假设。
 */
function renderScenarioPicker() {
  const wrap = byId('scenario-pick-wrap');
  const pick = byId('scenario-pick');
  const catalog = S.payload.scenarios || {};
  const items = catalog.items || [];
  const names = items.map(item => item.name);

  const shown = [];
  for (const option of pick.querySelectorAll('option')) shown.push(option.value);
  if (shown.join('\n') !== names.join('\n')) {
    clear(pick);
    for (const item of items) {
      const option = document.createElement('option');
      option.value = item.name;
      option.textContent = item.name;
      // ``title`` 挂的是这一份在哪：同一个名字在别的目录里是另一张图。
      option.title = item.path || '';
      pick.appendChild(option);
    }
  }

  pick.value = catalog.current != null && names.indexOf(catalog.current) >= 0 ? catalog.current : '';
  const live = S.payload.mode === 'live';
  pick.disabled = !live;
  pick.title = live
    ? '换一张计算图：旋钮不动——VRAM、预算、桥的开关都跟着走'
    : '导出的快照是只读的：换图要跟 Python 侧对话，这里改不了。换图请跑 python tools/viewer.py';
  // 只有一份可换时整格收起来：一个只有一个选项的下拉框是个陷阱。
  wrap.hidden = items.length < 2;
}

/* ------------------------------------------------------------- 计算图屏 */

const GRAPH = { colW: 196, rowH: 62, padX: 26, padY: 22, boxW: 168, boxH: 40, fillAlpha: 0.5 };

/* **两族色**：张量一律琥珀（与甘特图的 H2D 泳道同色），算子一律蓝（与 GPU 计算泳道同色），
 * 于是「一个张量被搬运」与「一个算子被计算」在图和时间线上是同一对颜色。族内用底色深浅与
 * 边线虚实表达状态，不再借用别的色系——原先两者都按状态取色（绿/灰/蓝混着来），形状相同、
 * 色系相同，只能靠读标签才知道哪个是哪个。 */
const TENSOR_TONE = {
  ABSENT: { fill: 'var(--surface)', stroke: 'var(--ns)', width: 1.2, dash: '3 3', fade: 0.45 },
  RESERVED_COPY: { fill: 'var(--ns-soft)', stroke: 'var(--ns)', width: 1.5, dash: '3 3', fade: 1 },
  READY: { fill: 'var(--ns-soft)', stroke: 'var(--ns)', width: 1.8, dash: null, fade: 1 },
};
const OP_TONE = {
  PENDING: { fill: 'var(--surface)', stroke: 'var(--accent)', width: 1.2, fade: 0.45 },
  RUNNING: { fill: 'var(--accent-soft)', stroke: 'var(--accent)', width: 2.4, fade: 1 },
  DONE: { fill: 'var(--accent-soft)', stroke: 'var(--accent)', width: 1.6, fade: 1 },
};

/* **方块填充半透明**（`GRAPH.fillAlpha`），好让被它压住的连线透出来。
 *
 * 边先画、方块后画，所以一条**跨列**的边（从很左边连到很右边，沿途经过中间那些列）会被
 * 中间那些方块盖住——实测五份图 38 条边里现在**只剩 1 条**被压住（`residual` 那条跨 7 列的
 * 残差边 `x → add`，它压住 `c2` 与 `b`）。被压住的那一段恰好是「它从哪儿来、跳过谁到哪儿去」
 * 最要紧的一段。
 *
 * （这个数是**按层重排之后重量的**：天顶之下只有一条横跨的边，因为父张量都贴到了算子左边一列、
 * 边基本只跨一格；被压的边 15 条掉到 1 条。再往前两次重量分别是「纵轴改按拓扑深度分层」
 * （10 → 15：方块在竖直方向摊开后更多边斜穿中间那几行）与 `executionSlots` 的起始列那次
 * （开局就绪的输入张量从最右边挪到最左边，11 → 10）。**改了布局就该重跑这条账**——
 * 引用旧数比不写数更坏。）
 *
 * 为什么半透明够用：**同一点最多只叠一个方块**（实测七份用例全部如此，每次改排布都重新量过，
 * 始终是 1 层——按层重排后那条唯一的被压边在 `(1006, 85)` 处也只压 1 层），
 * 所以不存在「叠了几层就彻底看不见」。而填充本身几乎不承担状态——
 * `--ns-soft` 对背景的对比只有 1.15:1，状态主要靠**描边**的虚实与粗细在表达
 * （描边不透明，`fill-opacity` 管不到它）。
 * 代价因此极不对称：填充从 1.15:1 降到 1.07:1（几乎看不出），被压住的边却从 3.07:1
 * 回到 1.73:1（救回一半多）。
 *
 * α 由上面这组数选出来，不是随手挑的：再低（0.35）边只多回到 1.46:1，填充却继续变淡。
 * 被压住的**灰色激活边**是这里最弱的一对（它和 `--accent-soft` 本来就只差 3.07:1），
 * 所以「能看见」的标准按它定。改 token 之前请先重跑这条账。 */

/** 计算图的排布：**横轴是拓扑层，纵轴是先后**。三条规则，逐条来自用户的话。
 *
 * 用户原话：「保证拓扑关系正确、张量加载、算子执行顺序从左到右两条基本要求下，保证拓扑图适当
 * 紧凑，线减少交叉，**一个算子的所有父张量应该是在 y 轴上排布，而不是在 x 轴上前后排布**，
 * 先加载的在左上，后加载的在右下，同时加载的就对齐，张量间可以有重叠，在左上的张量连接到算子
 * 的箭头也放在上面，避免交叉，**算子后的张量和算子在竖轴上对齐**」。
 *
 * 被否掉的那一版是「x 按执行顺序 + y 按拓扑深度分带」：x 取自「哪一步搬的它」，于是同一个算子
 * 的两个父张量一个在第 0 列、一个在第 2 列，两枚箭头长短不一；`residual` 里 `W2 → c2` 横跨四列
 * 斜穿一摞方块；深度分带又让每层各占一行，图被拉得很高（`residual` 602px 高只装 11 个方块）。
 * 这一版按他给的三条重排，五份用例都落到 **2~3 行**。
 *
 * 1. **列 = 拓扑层。** 算子用载荷给的秩列（`graph.columns`，张量在偶数列、算子在奇数列）；
 *    张量贴到**消费它的算子左边一列**，终值（没有消费者）贴到产出者右边一列。于是同一个算子的
 *    父张量自然落在**同一列**里——这正是「在 y 轴上排布，不要在 x 轴上前后排布」。
 *    全图唯一的例外是一个张量被**两个算子**吃：`residual` 的残差 `x` 同时喂 `c1` 和 `add`
 *    （秩 0 与秩 7），它只能贴在**最早**那个左边，另一条边跨列——分层画法的固有限制，除非把
 *    同一个张量画两遍。（`fork` 的 `x` 也喂两个算子，但 `c1`/`c2` 同层同列，所以那一条仍然成立。）
 * 2. **行 = 先后。** 同一列里按「谁先出现」自上而下：张量是**搬它那一步**（开局就在显存里的记
 *    −1，永远最上），算子是**算它那一步**。于是「先加载的在左上」，而且**同一算子的父张量从上到
 *    下就是加载顺序**——`fanOut` 也是按这个次序摊开的，上面来的箭头接在上面那枚锚点上。
 * 3. **算子与它产出的张量同行。** 算子的行 = 它所有父张量里**最靠下**的那一行（父都在更左的列，
 *    行已经定好），激活张量继承产出它的算子的行。于是「算子后的张量和算子在竖轴上对齐」，而且
 *    每条边都**不会往上画**（终点不高于起点——这条被单独钉成一张网）。
 *
 * 紧凑是从这三条里掉出来的，不是另外调的：列数从「每一步一列」压成「每层一列」。
 *
 * 没有解（`actions` 为空）时先后退回**拓扑秩**——那时没有「第几步」可言，只剩「层」这一个次序，
 * 而层与列本来就是同一件事；开局就在显存里的张量照样靠 `initial_locations` 认出来排在最上，
 * 所以被否掉的「inp_embd 被丢到最右边」那种错在这一版里**结构上不可能**：张量的列只由「谁吃它」
 * 决定，跟它在动作表里上没上过场无关。
 *
 * **`graph.columns` 是按 id 的列号表，不是一个基数。** 这里曾经按「每类一个数字基数」再乘秩，
 * 于是 ``{…} + rank*2`` 在 JS 里被拼成 ``"[object Object]0"``，坐标成了 NaN；而 **SVG 对无效
 * 属性是静默的**：``x="NaN"`` 被忽略（所有框挤到左边缘）、``d="…NaN…"`` 整条元素丢弃（一条边
 * 都画不出来），只留下一个看起来正常的空图。所以下面只做**查表 + 减一**，不做乘法。
 */
function graphLayout(graph, actions) {
  const columns = graph.columns || {};
  const opColumns = columns.operations || {};

  // 产出者与消费者都从 link 上读，不猜。`input` 边是「张量 → 算子」，`output` 边是「算子 → 张量」。
  const producedBy = {}, consumersOf = {};
  for (const link of graph.links) {
    if (link.kind === 'output') {
      if (link.from != null && link.to != null) producedBy[link.to] = link.from;
    } else if (link.from != null && link.to != null) {
      (consumersOf[link.from] = consumersOf[link.from] || []).push(link.to);
    }
  }

  // 先后：搬它那一步 / 算它那一步。开局就在显存里的记 −1（比第 0 步还早）。
  // 没有解时这两张表都是空的，全部落到 `1000 + 秩`，只剩「层」这一个次序。
  const copyStep = {}, computeStep = {};
  actions.forEach((action, index) => {
    if (action.kind === 'COPY_H2D' && action.tensor_id != null) {
      if (!(action.tensor_id in copyStep)) copyStep[action.tensor_id] = index;
    } else if (action.kind === 'COMPUTE' && action.operation_id != null) {
      if (!(action.operation_id in computeStep)) computeStep[action.operation_id] = index;
    }
  });
  const order = {};
  for (const op of graph.operations) {
    order[op.id] = computeStep[op.id] != null ? computeStep[op.id] : 1000 + op.rank;
  }
  for (const tensor of graph.tensors) {
    const atVram = (tensor.initial_locations || []).indexOf('vram') >= 0 && !tensor.is_weight;
    if (copyStep[tensor.id] != null) order[tensor.id] = copyStep[tensor.id];
    else if (atVram) order[tensor.id] = -1;
    else if (producedBy[tensor.id] != null && computeStep[producedBy[tensor.id]] != null) {
      order[tensor.id] = computeStep[producedBy[tensor.id]];
    } else order[tensor.id] = 1000 + tensor.rank;
  }

  // 列（x）。算子查表取秩列；张量贴到最早那个消费者左边一列；终值贴到产出者右边一列。
  const raw = {};
  for (const op of graph.operations) {
    raw[op.id] = opColumns[op.id] != null ? opColumns[op.id] : 2 * op.rank - 1;
  }
  for (const tensor of graph.tensors) {
    const consumers = consumersOf[tensor.id] || [];
    if (consumers.length) {
      let first = Infinity;
      for (const id of consumers) if (raw[id] != null) first = Math.min(first, raw[id]);
      raw[tensor.id] = isFinite(first) ? first - 1 : 0;
    } else if (producedBy[tensor.id] != null) {
      raw[tensor.id] = (raw[producedBy[tensor.id]] || 0) + 1;
    } else raw[tensor.id] = 0;
  }
  // 压实：秩列中间会留空档（`fork` 的终值 `y` 排在秩 4，而最后一个算子只到秩 3），
  // 把用到的列号重排成 0,1,2…，「适当紧凑」有一半是这一步。
  const usedColumns = [];
  for (const id in raw) if (usedColumns.indexOf(raw[id]) < 0) usedColumns.push(raw[id]);
  usedColumns.sort((a, b) => a - b);
  const col = {};
  for (const id in raw) col[id] = usedColumns.indexOf(raw[id]);

  // 行（y）：从左到右逐列定。父张量都在更左的列，所以轮到一列时它的输入行全部已知。
  const isOperation = {};
  for (const op of graph.operations) isOperation[op.id] = true;
  const byColumn = {};
  for (const entity of graph.tensors.concat(graph.operations)) {
    (byColumn[col[entity.id]] = byColumn[col[entity.id]] || []).push(entity);
  }
  const parentsOf = {};
  for (const link of graph.links) {
    if (link.kind !== 'output' && link.from != null && link.to != null) {
      (parentsOf[link.to] = parentsOf[link.to] || []).push(link.from);
    }
  }
  const row = {}, usedRows = {};
  for (let c = 0; c < usedColumns.length; c += 1) {
    const taken = usedRows[c] = {};
    // 从 `from` 往下找第一个空行；同列撞车就让位（同列的是同层、互不依赖，往下让不影响任何规则）。
    const take = from => {
      let r = from;
      while (taken[r]) r += 1;
      taken[r] = true;
      return r;
    };
    const entities = byColumn[c] || [];
    // 算子先定：行 = 父张量里最靠下那一行。
    const ops = entities.filter(e => isOperation[e.id]).sort((a, b) => order[a.id] - order[b.id]);
    for (const op of ops) {
      let r = 0;
      for (const id of parentsOf[op.id] || []) if (row[id] != null) r = Math.max(r, row[id]);
      row[op.id] = take(r);
    }
    // 激活张量：继承产出它的算子的行（「算子后的张量和算子在竖轴上对齐」）。
    // 走 `take` 而不是直接赋值：产出者在**更左的列**，所以这一步跨列对齐；万一这一行在同列
    // 已经被占（终值张量与同列的算子撞上——`fork` 的终值 `y` 与最后一个算子同列），
    // 直接赋值会让两个方块叠在一起（`方块互不遮盖` 那张网会抓到），所以往下让一格。
    for (const tensor of entities) {
      if (isOperation[tensor.id]) continue;
      const producer = producedBy[tensor.id];
      if (producer != null && row[producer] != null) row[tensor.id] = take(row[producer]);
    }
    // 输入/权重（没有产出者）以及产出行未知的：按先后从最上面往下找空行。
    const free = entities
      .filter(e => !isOperation[e.id] && row[e.id] == null)
      .sort((a, b) => order[a.id] - order[b.id]);
    for (const tensor of free) row[tensor.id] = take(0);
  }
  return { col: col, row: row };
}

/** 当前步的**算子真是 RUNNING 还是只是没做完**。op_status 只说 DONE/PENDING，
 *  正在算的那一个要从 ``running`` 队列里认。 */
function isRunning(snap, id) {
  return !!snap && (snap.running || []).some(task => task.target_id === id);
}

function renderGraph() {
  const payload = S.payload;
  const svg = byId('graph-svg');
  clear(svg);
  const graph = payload.graph;
  const snap = currentSnapshot();

  const actions = (payload.solution && payload.solution.actions) || [];
  const byExecution = actions.length > 0;
  // 排布在 :func:`graphLayout` 里一次算完（列 = 拓扑层，行 = 先后，见那里的长注释）。图宽高由它推，
  // 不再另外算 `maxSlot`：列号已经被压成 0,1,2…，最后一行方块的下沿就是图的下沿。
  const layout = graphLayout(graph, actions);
  const colOf = layout.col, rowOf = layout.row;
  const columnCount = Math.max(1, ...Object.values(colOf).map(c => c + 1));
  const rows = Math.max(1, ...Object.values(rowOf).map(row => row + 1));

  const width = GRAPH.padX * 2 + (columnCount - 1) * GRAPH.colW + GRAPH.boxW;
  const height = GRAPH.padY * 2 + (rows - 1) * GRAPH.rowH + GRAPH.boxH;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  // 三种箭头各自一个 marker：箭头要和它那条线同色。共用一个 marker 就得三选一。
  const defs = svgEl('defs');
  for (const [id, color] of [
    ['arrow-line', 'var(--ink-3)'],      // 激活输入
    ['arrow-accent', 'var(--accent)'],   // 算子的输出
    ['arrow-ns', 'var(--ns)'],           // 权重输入（要搬运的那些）
  ]) {
    const marker = svgEl('marker', {
      id: id, viewBox: '0 0 10 10', refX: 9, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse',
    });
    marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }));
    defs.appendChild(marker);
  }
  svg.appendChild(defs);

  const pos = {};
  for (const entity of graph.tensors.concat(graph.operations)) {
    pos[entity.id] = {
      x: GRAPH.padX + (colOf[entity.id] || 0) * GRAPH.colW,
      y: GRAPH.padY + rowOf[entity.id] * GRAPH.rowH,
    };
  }

  // 先画边，节点压在上面。边的颜色沿用**流动的东西**那一族的色：权重输入是张量（琥珀），
  // 激活输入是普通连线（灰），算子的输出跟着算子（蓝）。一条边因此自己就说明了它搬的是什么。
  // 一个节点的入边常常不止一条（`c1` 要 `W1` 和 `x`，`add` 要 `a` 和 `b`）。全都钉在左边
  // 中点的话，箭头会在**同一个点**上叠成一坨——用户报的就是这个。实测五份图各有 2~3 对
  // **完全重合**的锚点（`chain` 的 `614,42` 被 `W1→c1` 与 `x→c1` 同时打），`fork`/`residual`
  // 还有节点带两条出边、起点也重合。所以按条数把锚点**摊开**在边上：k 条落在 1/(k+1) … k/(k+1)。
  // 摊开的次序按**对面那一端的高度**排，这样摊完的线之间不交叉。
  //
  // 这张表按**边的对象本身**取值，所以必须是 `Map`。用 `{}` 的话键会被 `String()` 化：
  // 每一条边都是 `"[object Object]"`，六条边共用一格，后写的盖掉先写的——于是**所有边**
  // 都从最后一条边算出的那两个锚点出发（图上表现为一堆线在同一个高度集合上乱窜）。
  // 这不是假想的坑：摊开这一版就是这么做出来的，而「边数相等」「扫 NaN」两张网全绿。
  const anchors = new Map();
  /** `attr` 是**要摊开的那一端**（`to` = 终点摊在左边线上，`from` = 起点摊在右边线上），
   *  摊开的次序按另一端的高度排。两个方向各跑一次。 */
  const fanOut = attr => {
    const other = attr === 'to' ? 'from' : 'to';
    const buckets = new Map();
    for (const link of graph.links) {
      const a = pos[link.from], b = pos[link.to];
      if (!a || !b) continue;
      const key = link[attr];
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(link);
    }
    for (const [key, all] of buckets) {
      const list = all.slice().sort((p, q) => pos[p[other]].y - pos[q[other]].y);
      list.forEach((link, i) => {
        const anchor = anchors.get(link) || {};
        anchor[attr === 'to' ? 'y2' : 'y1'] =
          pos[key].y + GRAPH.boxH * (i + 1) / (list.length + 1);
        anchors.set(link, anchor);
      });
    }
  };
  fanOut('to');      // 入边：终点摊开在目标的左边线上
  fanOut('from');    // 出边：起点摊开在起点的右边线上

  const layer = svgEl('g');
  for (const link of graph.links) {
    const a = pos[link.from];
    const b = pos[link.to];
    if (!a || !b) continue;
    const anchor = anchors.get(link) || {};
    const x1 = a.x + GRAPH.boxW, y1 = anchor.y1 != null ? anchor.y1 : a.y + GRAPH.boxH / 2;
    const x2 = b.x, y2 = anchor.y2 != null ? anchor.y2 : b.y + GRAPH.boxH / 2;
    const mid = (x1 + x2) / 2;
    const out = link.kind === 'output';
    const tone = out ? 'accent' : (link.weight ? 'ns' : 'line');
    // 一律实线。虚线在搜索树屏已经是「被剪掉的边」的意思，这里再用一次会和那个含义打架。
    // 权重输入画粗一点：要搬运的就是它们，这整张图的排程就是为了它们。
    layer.appendChild(svgEl('path', {
      d: 'M ' + x1 + ' ' + y1 + ' C ' + mid + ' ' + y1 + ', ' + mid + ' ' + y2 + ', ' + x2 + ' ' + y2,
      fill: 'none',
      stroke: tone === 'accent' ? 'var(--accent)' : tone === 'ns' ? 'var(--ns)' : 'var(--ink-3)',
      'stroke-width': out || link.weight ? 1.9 : 1.3,
      'marker-end': 'url(#arrow-' + tone + ')',
    }));
  }
  svg.appendChild(layer);

  const copyStatus = (snap && snap.copy_status) || {};
  const opStatus = (snap && snap.op_status) || {};

  // **两族色**（常量在文件上方）：张量与算子各自一个色系，状态在族内用深浅与虚实表达。
  for (const tensor of graph.tensors) {
    const p = pos[tensor.id];
    const status = copyStatus[tensor.id] || (tensor.is_weight ? 'ABSENT' : 'READY');
    const tone = TENSOR_TONE[status] || TENSOR_TONE.ABSENT;
    const g = svgEl('g');
    const rect = svgEl('rect', {
      x: p.x, y: p.y, width: GRAPH.boxW, height: GRAPH.boxH, rx: 6,
      fill: tone.fill, 'fill-opacity': GRAPH.fillAlpha,
      stroke: tone.stroke, 'stroke-width': tone.width,
      'stroke-opacity': tone.fade,
    });
    if (tone.dash) rect.setAttribute('stroke-dasharray', tone.dash);
    g.appendChild(rect);
    g.appendChild(svgEl('text', { x: p.x + 10, y: p.y + 17, class: 'node-label' },
      tensor.name + (tensor.is_weight ? '' : ' ·act')));
    g.appendChild(svgEl('text', { x: p.x + 10, y: p.y + 31, class: 'node-sub' },
      fmtBytes(tensor.alloc_bytes) + ' · ' + status));
    const title = svgEl('title', null, tensor.id + '\n角色 ' + tensor.role +
      '\n存储 ' + fmtBytes(tensor.storage_bytes) + '，分配 ' + fmtBytes(tensor.alloc_bytes) +
      '\n初始位置 ' + (tensor.initial_locations || []).join(', '));
    g.appendChild(title);
    svg.appendChild(g);
  }

  for (const op of graph.operations) {
    const p = pos[op.id];
    const done = opStatus[op.id] === 'DONE' || opStatus[op.id] === 'COMPLETE';
    const running = !done && isRunning(snap, op.id);
    const status = done ? 'DONE' : running ? 'RUNNING' : (opStatus[op.id] || 'PENDING');
    const tone = OP_TONE[status] || OP_TONE.PENDING;
    const g = svgEl('g');
    g.appendChild(svgEl('rect', {
      x: p.x, y: p.y, width: GRAPH.boxW, height: GRAPH.boxH, rx: 6,
      fill: tone.fill, 'fill-opacity': GRAPH.fillAlpha,
      stroke: tone.stroke, 'stroke-width': tone.width,
      'stroke-opacity': tone.fade,
    }));
    g.appendChild(svgEl('text', { x: p.x + 10, y: p.y + 17, class: 'node-label' },
      (op.ggml_op || op.semantic_op || op.unary_op || '?') + ' · ' + op.id));
    g.appendChild(svgEl('text', { x: p.x + 10, y: p.y + 31, class: 'node-sub' },
      fmtNs(op.compute_ns) + ' · ' + status));
    const title = svgEl('title', null, op.id + '\n语义 ' + op.semantic_op +
      '\n输入 ' + (op.inputs || []).join(', ') + '\n输出 ' + op.output +
      '\n计算 ' + op.compute_ns + ' ns\n工作区 ' + fmtBytes(op.workspace_bytes));
    g.appendChild(title);
    svg.appendChild(g);
  }

  // 横纵两条轴都在右上角**说出来**（两条轴各一句），并标出当前分支是否已经离开解路径
  // （排布基准是解路径，所以离开时图**不会**跟着动——那是有意的，要说出来）。
  //
  // 这一版的两句话必须一起改：横轴已经不是「执行顺序」而是「拓扑层」，纵轴也不是「拓扑深度」
  // 而是「先后」。只改一句就会让屏幕主张一件图上没做的事——烟测里「说明说的列数 == 图上真正
  // 就绪的张量数」那张网当初就是为这种事立的，它随着「最左一列」的取消一起下掉了。
  const focusNode = isReplay() ? findNodeForState(S.cursor) : currentNode();
  const caption = '横轴 = 拓扑层（算子按层分列，张量贴在消费它的算子左边一列，所以同一算子的父张量上下叠着）'
    + '；纵轴 = 先后（先加载/先算的在上，算子与它产出的张量同一行，所以依赖边只往右下走）'
    + (byExecution ? '' : '　·　这一步没有解，先后退回收敛前的拓扑秩');
  // 这句话写在**屏头那段 `.hint` 里**，不画在 SVG 上。画在 SVG 上的话它不换行，
  // 而这一版的两句话合起来有 80 多个汉字——`matvec` 那张图只有 612px 宽，尾巴会被
  // 画布裁掉（SVG 默认 overflow: hidden），看起来就像「说明被砍了一半」。
  // 屏头是 HTML，自己会换行，顺带也少了一处「静态 HTML 与动态绘制各说一套」的隐患
  // （`index.html` 里那段 `.hint` 原先是写死的「横轴是执行顺序」，正好就是这么过期的）。
  byId('graph-hint').textContent = caption +
    (focusNode && focusNode.on_solution_path === false ? '　·　当前分支已离开解路径，图仍按解排布' : '');

  const legend = byId('graph-legend');
  clear(legend);
  legend.innerHTML =
    '<span><i style="background:var(--ns-soft);border:1.8px solid var(--ns)"></i>张量 · 已在显存</span>' +
    '<span><i style="background:var(--ns-soft);border:1.5px dashed var(--ns)"></i>张量 · 已排搬运</span>' +
    '<span><i style="background:var(--surface);border:1.2px dashed var(--ns);opacity:.55"></i>张量 · 不在显存</span>' +
    '<span><i style="background:var(--accent-soft);border:2.4px solid var(--accent)"></i>算子 · 正在算</span>' +
    '<span><i style="background:var(--accent-soft);border:1.6px solid var(--accent)"></i>算子 · 已完成</span>' +
    '<span><i style="background:var(--surface);border:1.2px solid var(--accent);opacity:.55"></i>算子 · 未开始</span>' +
    '<span><i class="edge weight"></i>权重输入（琥珀）</span>' +
    '<span><i class="edge act"></i>激活输入（灰）</span>' +
    '<span><i class="edge out"></i>算子的输出（蓝）</span>' +
    '<span>方块填充是<strong>半透明</strong>的：跨列的连线会被沿途的方块压住，透出来才看得出它打哪儿来、跳过谁</span>';
}

/* ------------------------------------------------------------- 时间线屏 */

/** 回放：直接用烤好的事件；单步：从路径的边现推。
 *
 * 单步事件为什么能推出来：动作 COPY_H2D / COMPUTE 在**起步的那一刻**就把任务
 * 放进 running，而只有 ADVANCE 动时钟，所以起点是该步的 t_ns、时长就是下一步
 * running 里那个任务的 remaining_ns（刚入队，还没被推进过）。EVICT 是瞬时的。
 */
function currentEvents() {
  const payload = S.payload;
  if (isReplay()) return payload.log.events;

  const nodes = payload.explore.nodes;
  const events = [];
  let previousRunning = {};
  for (let i = 0; i < S.path.length; i++) {
    const node = nodes[S.path[i]];
    const running = {};
    for (const task of (node.running || [])) running[task.resource + '|' + task.target_id] = task;

    if (i > 0) {
      const action = node.action;
      const prev = nodes[S.path[i - 1]];
      if (action) {
        if (action.kind === 'COMPUTE' && action.operation_id) {
          const task = running['gpu_compute|' + action.operation_id];
          const duration = task ? task.remaining_ns
            : (previousRunning['gpu_compute|' + action.operation_id] || {}).remaining_ns;
          events.push({
            kind: 'COMPUTE', resource: 'gpu_compute', target_id: action.operation_id,
            t_ns: prev.t_ns, start_ns: prev.t_ns,
            end_ns: prev.t_ns + (duration || 0),
          });
        } else if (action.kind === 'COPY_H2D' && action.tensor_id) {
          const task = running['h2d_copy|' + action.tensor_id];
          const duration = task ? task.remaining_ns
            : (previousRunning['h2d_copy|' + action.tensor_id] || {}).remaining_ns;
          events.push({
            kind: 'COPY_H2D', resource: 'h2d_copy', target_id: action.tensor_id,
            t_ns: prev.t_ns, start_ns: prev.t_ns,
            end_ns: prev.t_ns + (duration || 0),
          });
        } else if (action.kind === 'EVICT' && action.tensor_id) {
          events.push({
            kind: 'EVICT', resource: null, target_id: action.tensor_id,
            t_ns: prev.t_ns, start_ns: prev.t_ns, end_ns: prev.t_ns,
          });
        }
      }
    }
    previousRunning = running;
  }
  return events;
}

function renderTimeline() {
  const payload = S.payload;
  const hint = byId('timeline-hint');
  hint.textContent = isReplay()
    ? '正在重放搜索出的计划。'
    : '正在探索这条分支：这不是搜索出的计划，柱是从路径的边现推的。';

  const events = currentEvents();
  const states = S.payload.log.states;
  const snap = currentSnapshot();
  const nowNs = snap ? snap.t_ns : 0;
  const makespan = Math.max(1, payload.solution.makespan_ns || 1);

  // --- 甘特：两条泳道（搬运 / 计算），加上 EVICT / ADVANCE 的点标记
  const gantt = byId('gantt-svg');
  clear(gantt);
  const lanes = [
    { key: 'h2d_copy', label: 'H2D 搬运', color: 'var(--ns)' },
    { key: 'gpu_compute', label: 'GPU 计算', color: 'var(--accent)' },
  ];  const padL = 78, padR = 18, padT = 16, padB = 26, laneH = 30, barH = 15;
  const width = 900;
  const plotW = width - padL - padR;
  const height = padT + lanes.length * laneH + padB;
  gantt.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  gantt.setAttribute('width', width);
  gantt.setAttribute('height', height);

  const x = ns => padL + (ns / makespan) * plotW;

  lanes.forEach((lane, index) => {
    const y = padT + index * laneH;
    gantt.appendChild(svgEl('text', { x: padL - 9, y: y + barH, class: 'axis', 'text-anchor': 'end' }, lane.label));
    gantt.appendChild(svgEl('rect', {
      x: padL, y: y, width: plotW, height: barH + 4, fill: 'var(--surface-2)', rx: 3,
    }));
    for (const event of events) {
      if (event.resource !== lane.key) continue;
      const start = event.start_ns == null ? event.t_ns : event.start_ns;
      const end = event.end_ns == null ? event.t_ns : event.end_ns;
      const w = Math.max(1.5, x(end) - x(start));
      const walked = start < nowNs;
      // 柱子用**同族色描边 + 浅底**，不是实心块。实心同色柱挨在一起会糊成一根长条
      // （相邻两次搬运本来就是同色），而描边让每一根自己立得住。这样也正好和计算图
      // 用的是同一套写法：浅底 = 这块东西在场，族色边 = 它是张量还是算子。
      // 还没走到的段画成**空底虚线**，与计算图里 ABSENT 的写法一致。
      gantt.appendChild(svgEl('rect', {
        x: x(start), y: y, width: w, height: barH + 4, rx: 3,
        fill: walked ? (lane.key === 'h2d_copy' ? 'var(--ns-soft)' : 'var(--accent-soft)')
                     : 'var(--surface)',
        stroke: lane.color, 'stroke-width': 1.2,
        'stroke-dasharray': walked ? null : '3 2',
      }));
      if (w > 26) {
        gantt.appendChild(svgEl('text', {
          x: x(start) + w / 2, y: y + barH, class: 'axis',
          'text-anchor': 'middle', fill: 'var(--ink-2)',
        }, event.target_id || ''));
      }
    }
  });

  // EVICT / ADVANCE 没有 resource，是**点**不是条：画在最下面一条细带上。
  const markY = padT + lanes.length * laneH + 3;
  gantt.appendChild(svgEl('line', {
    x1: padL, y1: markY + 5, x2: padL + plotW, y2: markY + 5, class: 'grid-line',
  }));
  for (const event of events) {
    if (event.resource != null) continue;
    const cx = x(event.t_ns);
    const isEvict = event.kind === 'EVICT';
    gantt.appendChild(svgEl('circle', {
      cx: cx, cy: markY + 5, r: isEvict ? 3.4 : 2.4,
      fill: isEvict ? 'var(--bad)' : 'var(--ink-3)',
      opacity: event.t_ns >= nowNs ? 0.35 : 1,
    }));
  }
  gantt.appendChild(svgEl('text', { x: padL - 9, y: markY + 9, class: 'axis', 'text-anchor': 'end' },
    '释放/推进'));
  gantt.appendChild(svgEl('text', { x: padL, y: height - 8, class: 'axis' }, '0'));
  gantt.appendChild(svgEl('text', {
    x: padL + plotW, y: height - 8, class: 'axis', 'text-anchor': 'end',
  }, fmtNs(makespan)));
  gantt.appendChild(svgEl('line', {
    x1: x(nowNs), y1: padT - 4, x2: x(nowNs), y2: markY + 10,
    stroke: 'var(--ink)', 'stroke-width': 1.4,
  }));

  // --- 显存曲线：阶梯 + 容量虚线。容量上限**只在 limits**，states 里没有。
  const vram = byId('vram-svg');
  clear(vram);
  const limit = (payload.source.limits || {}).vram_capacity_bytes;
  const peak = Math.max(limit || 0, ...states.map(s => s.used_vram_bytes || 0), 1);
  // `vpadT` 是 26 而不是 14，只为「容量 xxx」这行字：容量线画在**顶上那格**（`peak` 取的就是
  // 容量上限，所以 `vy(limit)` 正好落在绘图区上沿），它的标注只能放在线上方——而中文在 14px 下
  // 从基线往上要占约 12.3px，`ly - 6` 的基线因此至少要离画布上边 12px 才不被裁。14 的时候基线在
  // y=9，**「容量」两个字的上半截被画布切掉了**（用户报的「被第一张图覆盖了」）。
  // SVG 裁掉就是裁掉，不报错——所以这条要靠几何留够，另有一条网盯着「没有一行字贴着画布上下边」。
  const vh = 170, vpadL = 78, vpadR = 18, vpadT = 26, vpadB = 22;
  const vw = width;
  const vplotW = vw - vpadL - vpadR;
  const vplotH = vh - vpadT - vpadB;
  vram.setAttribute('viewBox', '0 0 ' + vw + ' ' + vh);
  vram.setAttribute('width', vw);
  vram.setAttribute('height', vh);

  const vx = ns => vpadL + (ns / makespan) * vplotW;
  const vy = bytes => vpadT + vplotH - (bytes / peak) * vplotH;

  for (let i = 0; i <= 4; i++) {
    const value = (peak / 4) * i;
    const y = vy(value);
    vram.appendChild(svgEl('line', { x1: vpadL, y1: y, x2: vpadL + vplotW, y2: y, class: 'grid-line' }));
    vram.appendChild(svgEl('text', { x: vpadL - 9, y: y + 3.5, class: 'axis', 'text-anchor': 'end' },
      fmtBytes(value)));
  }

  // 超限带：沿用 plot_results.py 的视觉语言（容量线以上整片淡红）。
  if (limit != null) {
    const ly = vy(limit);
    if (ly > vpadT + 0.5) {
      vram.appendChild(svgEl('rect', {
        x: vpadL, y: vpadT, width: vplotW, height: Math.max(0, ly - vpadT),
        fill: 'var(--bad)', opacity: 0.09,
      }));
    }
    vram.appendChild(svgEl('line', {
      x1: vpadL, y1: ly, x2: vpadL + vplotW, y2: ly,
      stroke: 'var(--bad)', 'stroke-width': 1.3, 'stroke-dasharray': '5 3',
    }));
    vram.appendChild(svgEl('text', { x: vpadL + vplotW - 4, y: ly - 6, class: 'axis', 'text-anchor': 'end', fill: 'var(--bad)' },
      '容量 ' + fmtBytes(limit)));
  }

  const path = states.map(s => (vx(s.t_ns)) + ' ' + vy(s.used_vram_bytes || 0)).join(' L ');
  vram.appendChild(svgEl('path', {
    d: 'M ' + path, fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1.1, opacity: 0.5,
  }));
  // 已走过的段落实线加粗，未走到的淡出。
  const walked = states.filter(s => s.t_ns <= nowNs);
  if (walked.length) {
    vram.appendChild(svgEl('path', {
      d: 'M ' + walked.map(s => vx(s.t_ns) + ' ' + vy(s.used_vram_bytes || 0)).join(' L '),
      fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.2,
    }));
    const last = walked[walked.length - 1];
    vram.appendChild(svgEl('circle', {
      cx: vx(last.t_ns), cy: vy(last.used_vram_bytes || 0), r: 4.4,
      fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2,
    }));
  }
  vram.appendChild(svgEl('line', {
    x1: vx(nowNs), y1: vpadT - 4, x2: vx(nowNs), y2: vpadT + vplotH,
    stroke: 'var(--ink)', 'stroke-width': 1.4,
  }));

  renderStateTimeline(states, nowNs);
  renderRefused();
}

/** 「从这一步出发，哪些候选被引擎拒了、理由是什么」。
 *
 * 搜索树上的节点**全部都是走得到的**——不可行性不在节点上，只在「从某状态出发的某个候选」
 * 上。所以「展开到不可行为止，并说明原因」在数据上唯一成立的形式就是这一块：当前状态的
 * 合法动作列在走带上（可点），被拒的列在这里（原样理由，不改写）。
 *
 * 用 ``<details>`` 是因为它默认收起：理由经常有十几条，展开着会把时间线顶下去。
 */
function renderRefused() {
  const body = byId('refused-body');
  if (!body) return;
  clear(body);
  const node = isReplay() ? findNodeForState(S.cursor) : currentNode();
  if (!node) return;
  const illegal = illegalByNode(node.id);
  if (!illegal.length) return;

  const box = document.createElement('details');
  box.className = 'refused';
  const summary = document.createElement('summary');
  summary.textContent = '从这一步出发，' + illegal.length + ' 个候选被引擎拒绝';
  box.appendChild(summary);
  const list = document.createElement('div');
  list.className = 'refused-list';
  for (const item of illegal) {
    const row = document.createElement('div');
    row.innerHTML = '<span class="kind">' + esc(ACTION_LABEL[item.action.kind]) + '</span>' +
      '<span class="mono">' + esc(item.action.operation_id || item.action.tensor_id || '（全局）') +
      '</span><span class="why">' + esc(item.message) + '</span>';
    list.appendChild(row);
  }
  box.appendChild(list);
  body.appendChild(box);
}

function renderStateTimeline(states, nowNs) {
  const svg = byId('states-svg');
  clear(svg);
  const actions = (S.payload.solution && S.payload.solution.actions) || [];
  const padL = 78, rowH = 17, padT = 8, width = 900, GAP = 7;
  // **空的时候也要留出一行的高度。** 原先这里是 `padT + 0 * rowH + 6 = 14`，而下面那句人话画在
  // y=19——整句落在画布之外，被 `overflow: hidden` 静默裁掉，屏上就是个空盒子：「没有状态可画」
  // 这句话说了等于没说。是「时间线的每行字都离画布上下边够远」那条网把它抓出来的，
  // 它盯的是性质、不是这一句（写它的时候并不知道这里藏着一个 bug）。`Math.max(1, …)` 一句就够。
  const height = padT + Math.max(1, states.length) * rowH + 6;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  // **`states` 为空是正常输入，不是异常输入。** 搜索排不出计划（infeasible）时它就是空数组，
  // 于是 `states[states.length - 1]` 是 `undefined`，读 `.t_ns` 会抛。曾经没有这一句，
  // 那一抛把「时长来源」与「参数」两屏一起带空（见 `render()` 的注释：所有渲染器挤在一个
  // try 里，第一个抛的把后面的全带走）。空的时候画一句人话，别留一个空盒子。
  if (!states.length) {
    svg.appendChild(svgEl('text', { x: 8, y: padT + 11, class: 'axis' },
      '没有状态可画：这次搜索没有排出一条可执行的计划（无解），所以没有时间线。'));
    return;
  }
  // 每一行右边那列的字：**这一步在干什么 · 什么时候 · 用了多少**。
  // `action_index` 是「产生这个状态」的那个动作在 `solution.actions` 里的下标——states[0] 是
  // 起点，没有前驱，所以是 null（`actionLabel(null)` 给「（起点）」）。
  const labels = states.map(state => {
    const action = state.action_index == null ? null : actions[state.action_index];
    return actionLabel(action) + ' · ' + fmtNs(state.t_ns) + ' · ' + fmtBytes(state.used_vram_bytes || 0);
  });
  // 右列留多宽，按**这一份载荷里最长的那条**现算。原先写死 `padR = 168`，字右对齐钉在 x=892，
  // 于是每条说明离它自己那根条都有 30~650px 不等的空档——用户的原话是「把字放到阶段的傍边」。
  // 现在字跟在条后面走，右列宽度就得跟着内容走：`GAP + labelW + 10` 保证**最长**的那条也离
  // 画布右边还有 10px（条最远能到 `width - padR`，见下面 `w` 的算法）。代价是画布窄了一点，
  // 换来的是「说明」与「它说明的那个阶段」不再隔着半张图。
  const labelW = Math.max(...labels.map(textWidth));
  const padR = GAP + labelW + 10;
  const maxT = Math.max(1, states[states.length - 1].t_ns);
  const plotW = width - padL - padR;

  states.forEach((state, index) => {
    const y = padT + index * rowH;
    const walked = state.t_ns <= nowNs;
    const bx = padL + (state.t_ns / maxT) * plotW;
    svg.appendChild(svgEl('text', { x: padL - 9, y: y + 11, class: 'axis', 'text-anchor': 'end' },
      '#' + index));
    const nx = states[index + 1] ? states[index + 1].t_ns : maxT;
    const w = Math.max(1, ((nx - state.t_ns) / maxT) * plotW);
    const rect = svgEl('rect', {
      x: bx, y: y, width: w, height: rowH - 4, rx: 2,
      fill: walked ? 'var(--accent-soft)' : 'var(--surface)',
      stroke: 'var(--accent)', 'stroke-width': 1,
      'stroke-dasharray': walked ? null : '3 2',
    });
    svg.appendChild(rect);
    // 字**紧贴着这一行自己的那根条**。同一时刻的几个状态因此自动对齐成一列——字的位置本身
    // 就是时刻，与横轴说的是同一件事，不必再去对照刻度。
    svg.appendChild(svgEl('text', {
      x: bx + w + GAP, y: y + 11, class: 'axis',
    }, labels[index]));
    rect.appendChild(svgEl('title', null,
      '状态 #' + index + '（产生它的动作 ' + (state.action_index == null ? '—' : '#' + state.action_index) +
      '）\nt = ' + fmtNs(state.t_ns) + '\n已用 ' + fmtBytes(state.used_vram_bytes || 0) +
      '\n算子 ' + JSON.stringify(state.op_status || {}) + '\n张量 ' + JSON.stringify(state.copy_status || {})));
  });
}

/* ------------------------------------------------------------- 搜索树屏 */

function renderTree() {
  const payload = S.payload;
  const svg = byId('tree-svg');
  clear(svg);
  if (S.treeDisabled) {
    // 走查与 search() 对不上：**宁可空着也不画一棵可能是错的树**。屏留在这里（跳转条
    // 指得到它），内容换成原因——禁掉的必须是这块屏，不是那个按钮：按钮禁用之后，
    // 用户连「为什么禁」都点不到。
    svg.setAttribute('viewBox', '0 0 640 60');
    svg.setAttribute('width', 640);
    svg.setAttribute('height', 60);
    svg.appendChild(svgEl('text', { x: 12, y: 24, class: 'node-label' },
      '已禁用：走查与 mapper.search() 对不上，不画一棵可能是错的树。'));
    svg.appendChild(svgEl('text', { x: 12, y: 44, class: 'node-sub' },
      '差异列在顶部红条里。手动单步不受影响；这一屏恢复要等两边对上。'));
    clear(byId('tree-legend'));
    return;
  }
  const block = payload.explore;
  const nodes = block.nodes;
  if (!nodes.length) {
    svg.setAttribute('viewBox', '0 0 300 60');
    svg.appendChild(svgEl('text', { x: 10, y: 30, class: 'node-label' }, '起点就装不下，没有状态可画。'));
    return;
  }

  // 布局：x 用 g_ns（代价轴，这才是搜索真正排的东西），y 用深度错开同代价的节点。
  const maxG = Math.max(1, ...nodes.map(n => n.g_ns));
  const depthCount = {};
  const slot = {};
  for (const node of nodes) {
    const d = node.depth;
    slot[node.id] = depthCount[d] = (depthCount[d] || 0);
    depthCount[d] += 1;
  }
  const maxSlot = Math.max(1, ...Object.values(slot).map(v => v + 1));
  const padL = 44, padR = 130, padT = 22, padB = 30;
  const width = 940, height = 118 + 15;  // 单行高度：节点画小圆点，避免 216 个盒子糊成一团
  const plotW = width - padL - padR;
  const rowH = 8;
  const gridH = maxSlot * rowH;

  const h = padT + gridH + padB;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + h);
  svg.setAttribute('width', width);
  svg.setAttribute('height', h);

  const px = g => padL + (g / maxG) * plotW;
  const py = node => padT + (slot[node.id] + 0.5) * rowH;

  const onPath = new Set(block.nodes.filter(n => n.on_solution_path).map(n => n.id));
  const pathEdges = new Set();
  let cursor = block.goal_id != null ? block.goal_id : null;
  while (cursor != null) {
    const node = nodes[cursor];
    if (!node || node.parent == null) break;
    pathEdges.add(node.parent + '>' + node.id);
    cursor = node.parent;
  }

  svg.appendChild(svgEl('text', { x: padL, y: 13, class: 'axis' }, 'x = 累计代价 g（整数纳秒）'));
  for (let i = 0; i <= 4; i++) {
    const g = (maxG / 4) * i;
    svg.appendChild(svgEl('line', {
      x1: px(g), y1: padT - 4, x2: px(g), y2: padT + gridH, class: 'grid-line',
    }));
    svg.appendChild(svgEl('text', { x: px(g), y: h - 12, class: 'axis', 'text-anchor': 'middle' }, fmtNs(g)));
  }

  // 剪边：只画当前路径附近的，全画（379 条）会糊成一团。
  const focus = currentNode() || nodes[block.start_id || 0];
  const nearby = new Set([focus.id]);
  if (focus.parent != null) nearby.add(focus.parent);

  for (const node of nodes) {
    const x2 = px(node.g_ns), y2 = py(node);
    for (const child of (node.children || [])) {
      const target = nodes[child.to];
      if (!target) continue;
      const x1 = px(target.g_ns), y1 = py(target);
      const key = node.id + '>' + child.to;
      const isCut = child.cut != null;
      const isPath = pathEdges.has(key);
      if (isCut && !(nearby.has(node.id) && nearby.has(child.to))) continue;
      svg.appendChild(svgEl('line', isCut
        ? { x1: x2, y1: y2, x2: x2 + 14, y2: y1, class: 'cut-edge' }
        : {
            x1: x2, y1: y2, x2: x1, y2: y1,
            stroke: isPath ? 'var(--ok)' : 'var(--line-2)',
            'stroke-width': isPath ? 2 : 0.9, opacity: isPath ? 1 : 0.75,
          }));
    }
  }

  const roleColor = {
    start: 'var(--ink)',
    expanded: 'var(--accent)',
    frontier: 'var(--warn)',
    cut: 'var(--line-2)',
  };
  for (const node of nodes) {
    const isGoal = node.id === block.goal_id;
    const cx = px(node.g_ns), cy = py(node);
    svg.appendChild(svgEl('circle', {
      cx: cx, cy: cy, r: isGoal ? 5.5 : 3,
      fill: onPath.has(node.id) ? 'var(--ok)' : roleColor[node.role] || 'var(--ink-3)',
      stroke: node.id === focus.id ? 'var(--ink)' : 'none', 'stroke-width': 2,
    }));
    // 命中圈：可见圆点只有 r=3，根本点不中。透明的大圈专门接点击，并带 ``data-id``——
    // 「这个元素是给谁用的」写在元素自己身上，不靠序号、不靠闭包外的数组下标。
    // ``renderTree()`` 每帧 ``clear(svg)`` 重建，所以监听器不会累积，不需要解绑。
    const hit = svgEl('circle', { cx: cx, cy: cy, r: 7, class: 'tree-hit' });
    hit.dataset.id = String(node.id);
    hit.appendChild(svgEl('title', null,
      '节点 #' + node.id + '　g = ' + fmtNs(node.g_ns) + '　深度 ' + node.depth +
      '\n角色 ' + node.role + '　' + (onPath.has(node.id) ? '在解路径上' : '不在解路径上') +
      '\n已用显存 ' + fmtBytes(node.used_vram_bytes) +
      '\n点一下：计算图与时间线一起搬到这条路径上'));
    hit.addEventListener('click', () => jumpTo(node.id));
    svg.appendChild(hit);
  }

  const verdict = block.verdict;
  const info = [
    '展开 ' + verdict.expanded_states + ' · 见过 ' + verdict.visited_states +
      ' · 节点 ' + nodes.length,
    '剪边 ' + (block.cut_counts.dominated || 0) + '（被支配）',
    '预算 展开≤' + block.budget.max_expanded_states + ' · 时间≤' + block.budget.wall_time_limit_s + ' s',
  ];
  info.forEach((line, index) => {
    svg.appendChild(svgEl('text', { x: padL, y: padT + gridH + 16 + index * 13, class: 'node-sub' }, line));
  });

  const legend = byId('tree-legend');
  legend.innerHTML =
    '<span><i style="background:var(--ink)"></i>起点</span>' +
    '<span><i style="background:var(--accent)"></i>已展开</span>' +
    '<span><i style="background:var(--warn)"></i>前沿（入堆未展开）</span>' +
    '<span><i style="background:var(--line-2)"></i>仅被剪边到达</span>' +
    '<span><i style="background:var(--ok)"></i>解路径</span>' +
    '<span>实线 = 入堆 · 点线 = 被支配剪掉（仅画焦点附近）</span>' +
    '<span><strong>点任一节点</strong>：下面的计算图与时间线一起搬到那条路径上</span>';
}

/* --------------------------------------------------------- 时长来源屏 */

/** 这一屏先回答「这条时间线的时间是从哪来的」，再给「每一行是怎么算出来的」。
 *
 * 顺序不能反。上一版这里是个「拒绝博物馆」：只讲 modeling 推不出来的那些情况，于是时长
 * 来源默认走通之后，它反而成了**唯一**看不出「数字是推的还是抄的」的地方——而「mapping
 * 出策略、modeling 出时长」正是这个工具要展示的那件事。
 */
function renderBridge() {
  const body = byId('bridge-body');
  clear(body);
  const bridge = S.payload.bridge;
  const source = S.payload.cost_source;
  const live = S.payload.mode === 'live';

  if (S.bridgeMsg) {
    const div = document.createElement('div');
    div.className = 'banner';
    div.textContent = S.bridgeMsg;
    body.appendChild(div);
  }

  // --- 第一块：来源。**永远**先出现，四种结局各说各的。
  if (source) body.appendChild(sourceCard(source));

  // --- 第二块：逐行算式。只在推导成功时才有内容。
  if (!bridge) {
    body.appendChild(emptyNote('这份载荷里没有换算表（未指定 --config）。'));
    return;
  }

  if (!bridge.ok) {
    body.appendChild(refusalCard(bridge, live));
    return;
  }

  // --- 成功：换算表。行的实体在 bridge.layers（按 operation_id 与 table 对齐），
  //     table 里只有 ns 与 agrees——两者分开存是为了「层」和「换算结果」各自可测。
  const byOp = {};
  for (const layer of (bridge.layers || [])) byOp[layer.operation_id] = layer;

  const card = document.createElement('div');
  card.className = 'gate done';
  card.innerHTML =
    '<div class="gate-head"><span class="gate-num">ƒ</span>' +
    '<h3>每一行是怎么算出来的</h3>' +
    '<span class="badge ' + (bridge.reference_used ? 'warn' : 'live') + '">' +
    (bridge.reference_used ? '参考硬件' : '场景自带硬件') + '</span>' +
    '<span class="badge ' + (bridge.all_agree ? 'ok' : 'bad') + '">' +
    (bridge.all_agree ? '与声明值逐行一致' : '有行与声明值不一致') + '</span></div>';
  const inner = document.createElement('div');
  inner.className = 'gate-body';

  const alignNote = document.createElement('p');
  alignNote.className = 'note';
  alignNote.innerHTML = '<strong>这里对齐的是算式，不是标定精度。</strong>' +
    '链形示例的成本是刻意构造成能被这组参数精确复现的，所以「推导 ns == 声明 ns」证明的是' +
    '桥的算式对，不是模型准。真实标定配置的数字比这里大好几个数量级。';
  inner.appendChild(alignNote);

  const hw = bridge.hardware || {};
  const hwLine = document.createElement('p');
  hwLine.className = 'field';
  hwLine.style.fontSize = '11.5px';
  hwLine.style.color = 'var(--ink-3)';
  hwLine.textContent = '硬件：' + num(hw.gpu_effective_flops) + ' FLOP/s · ' +
    num(hw.h2d_bandwidth_bytes_per_s) + ' B/s · 延迟 ' + hw.h2d_latency_s + ' s · VRAM ' +
    fmtBytes(hw.vram_capacity_bytes);
  inner.appendChild(hwLine);

  const wrap = document.createElement('div');
  wrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.innerHTML =
    '<thead><tr><th>层</th><th>算子</th><th class="num">权重 B</th><th class="num">FLOP</th>' +
    '<th class="num">model ns</th><th class="num">declared ns</th><th>一致</th></tr></thead>';
  const tbody = document.createElement('tbody');
  for (const row of bridge.table) {
    const layer = byOp[row.operation_id] || {};
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td>' + esc(layer.name || '—') + '</td>' +
      '<td>' + esc(row.operation_id) + '</td>' +
      '<td class="num">' + num(layer.weight_bytes) + '</td>' +
      '<td class="num">' + num(layer.flops) + '</td>' +
      '<td class="num">' + num(row.model_ns) + '</td>' +
      '<td class="num">' + num(row.declared_ns) + '</td>' +
      '<td class="' + (row.agrees ? 'agree' : 'disagree') + '">' + (row.agrees ? '=' : '≠') + '</td>';
    // 算式挂在行上：光给数字，这一屏还是「两个数并排放」，看不出 modeling 到底做了什么。
    tr.title = '计算：' + num(layer.flops) + ' FLOP ÷ ' + num(hw.gpu_effective_flops) +
      ' FLOP/s = ' + (layer.flops / hw.gpu_effective_flops).toExponential(3) + ' s → ' +
      row.model_ns + ' ns';
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  inner.appendChild(wrap);

  if (bridge.h2d && bridge.h2d.length) {
    const h2dWrap = document.createElement('div');
    h2dWrap.className = 'table-wrap';
    const h2dTable = document.createElement('table');
    h2dTable.innerHTML =
      '<thead><tr><th>搬运</th><th class="num">字节</th><th class="num">model ns</th>' +
      '<th class="num">declared ns</th><th>一致</th></tr></thead>';
    const hbody = document.createElement('tbody');
    for (const row of bridge.h2d) {
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + esc(row.tensor_id) + '</td>' +
        '<td class="num">' + num(row.bytes) + '</td>' +
        '<td class="num">' + num(row.model_ns) + '</td>' +
        '<td class="num">' + num(row.declared_ns) + '</td>' +
        '<td class="' + (row.agrees ? 'agree' : 'disagree') + '">' + (row.agrees ? '=' : '≠') + '</td>';
      tr.title = '搬运：延迟 ' + hw.h2d_latency_s + ' s + ' + num(row.bytes) + ' B ÷ ' +
        num(hw.h2d_bandwidth_bytes_per_s) + ' B/s = ' + row.model_ns + ' ns';
      hbody.appendChild(tr);
    }
    h2dTable.appendChild(hbody);
    h2dWrap.appendChild(h2dTable);
    inner.appendChild(h2dWrap);
  }

  // 硬件三旋钮：**这是「联动」唯一能被手指碰到的地方**。改一个，上面的 ns、时间线的柱宽、
  // 顶栏的 makespan 一起变——所以它们必须和参数屏走同一条通路（POST /api/params 的
  // ``hardware`` 键），而不是只改这一屏的显示。
  if (live) {
    inner.appendChild(hardwareKnobs(hw));
  } else {
    inner.appendChild(emptyNote('导出的快照是只读的；改硬件参数需要 python tools/viewer.py。'));
  }

  card.appendChild(inner);
  body.appendChild(card);

  if (source && source.disagreements && source.disagreements.length) {
    body.appendChild(divergenceCard(source.disagreements));
  }
}

function emptyNote(text) {
  const p = document.createElement('p');
  p.className = 'empty';
  p.textContent = text;
  return p;
}

/** 来源卡片：这一份 scenario 用的是**推导值还是声明值**，以及为什么。
 *
 * 四种结局都必须说清理由。一个静默回退的来源面板比一个空面板更坏——用户会以为屏上的
 * 数字是 modeling 算的，而它其实是 ``examples/*.json`` 里手写的常数。
 */
function sourceCard(source) {
  const card = document.createElement('div');
  card.className = 'gate ' + (source.swapped ? 'done' : 'blocked');

  card.innerHTML =
    '<div class="gate-head"><span class="gate-num">' + (source.swapped ? '✓' : '!') + '</span>' +
    '<h3>' + esc(source.label || '时长来源') + '</h3>' +
    '<span class="badge ' + (source.swapped ? 'ok' : 'bad') + '">' +
    (source.swapped ? '推导值' : '声明值') + '</span>' +
    '<span class="badge">' + source.rows.length + ' 行</span>' +
    (source.reason_class ? '<span class="badge">' + esc(source.reason_class) + '</span>' : '') +
    '</div>';

  const gb = document.createElement('div');
  gb.className = 'gate-body';

  const lede = document.createElement('p');
  lede.className = 'note';
  lede.innerHTML = source.swapped
    ? '时间线上的每一根柱子都是 <code>llm_infer_model</code> 现算的：计算走 ' +
      '<code>FLOP ÷ gpu_effective_flops</code>，搬运走 <code>延迟 + 字节 ÷ 带宽</code>，' +
      '取整成整数纳秒后喂给内核。内核算出的 makespan、峰值、事件全部由此推出——' +
      '<strong>改参数，屏上就会动</strong>。'
    : '<strong>这一份用的不是推导值</strong>，是 scenario 里写死的常数。理由在下面。';
  gb.appendChild(lede);

  if (source.message) {
    const reason = document.createElement('div');
    reason.className = 'reason ' + esc(source.reason_class || '');
    reason.textContent = source.message;
    gb.appendChild(reason);
  }
  if (source.remedy) {
    const remedy = document.createElement('p');
    remedy.className = 'note';
    remedy.innerHTML = '<strong>补救：</strong>' + esc(source.remedy);
    gb.appendChild(remedy);
  }

  card.appendChild(gb);
  return card;
}

/** 推导值 ≠ 声明值。这是**唯一**一处「接上 modeling 之后冻结数字会变」的地方，
 *  所以要红着说，并且说清哪一个是哪个、以及为什么冻结记录仍然成立。 */
function divergenceCard(disagreements) {
  const card = document.createElement('div');
  card.className = 'gate blocked';
  card.innerHTML = '<div class="gate-head"><span class="gate-num">!</span>' +
    '<h3>推导值与声明值不一致（' + disagreements.length + ' 行）</h3></div>';
  const gb = document.createElement('div');
  gb.className = 'gate-body';
  const p = document.createElement('p');
  p.className = 'note';
  p.innerHTML = '这不是 bug，是<b>示例自己不自洽</b>：<code>examples/*.json</code> 里的常数是照着' +
    '一组理想硬件手写的，而搬运字节数得从张量自己推（不是常量）。' +
    '把两者对上之后，这个示例的 makespan 就不再是存档里记的那个数了。' +
    '<strong>冻结的验收记录读的是 JSON 文件本身</strong>，所以它们照旧成立；变的是这一屏，' +
    '以及时间线上跟着它走的那几根柱子。';
  gb.appendChild(p);
  const wrap = document.createElement('div');
  wrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.innerHTML = '<thead><tr><th>类</th><th>对象</th><th class="num">推导 ns</th>' +
    '<th class="num">声明 ns</th><th class="num">差</th></tr></thead>';
  const tbody = document.createElement('tbody');
  for (const row of disagreements) {
    tbody.innerHTML += '<tr><td>' + esc(row.kind) + '</td><td>' + esc(row.id) + '</td>' +
      '<td class="num">' + num(row.model_ns) + '</td>' +
      '<td class="num">' + num(row.declared_ns) + '</td>' +
      '<td class="num">' + num(row.model_ns - row.declared_ns) + '</td></tr>';
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  gb.appendChild(wrap);
  card.appendChild(gb);
  return card;
}

/** 三个硬件旋钮。改完立刻重跑：ns → 柱宽 → makespan 是一条链，不是三个独立的显示。 */
function hardwareKnobs(hardware) {
  const box = document.createElement('div');
  box.className = 'knobs';
  const fields = [
    ['gpu_effective_flops', '有效算力', 'FLOP/s'],
    ['h2d_bandwidth_bytes_per_s', 'H2D 带宽', 'B/s'],
    ['h2d_latency_s', 'H2D 固定延迟', 's'],
  ];
  const inputs = {};
  for (const [field, label, unit] of fields) {
    const div = document.createElement('div');
    div.className = 'knob';
    const id = 'hw-' + field;
    const lab = document.createElement('label');
    lab.htmlFor = id;
    lab.textContent = label + '（' + unit + '）';
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.id = id;
    input.value = hardware[field];
    const sub = document.createElement('span');
    sub.className = 'field';
    sub.textContent = 'hardware.' + field;
    div.append(lab, input, sub);
    box.appendChild(div);
    inputs[field] = input;
  }
  const row = document.createElement('div');
  row.className = 'run-row';
  const run = document.createElement('button');
  run.className = 'run';
  run.textContent = '改这组硬件重跑';
  run.addEventListener('click', () => {
    const params = { hardware: {} };
    for (const [field, input] of Object.entries(inputs)) params.hardware[field] = Number(input.value);
    submitParams(params);
  });
  row.appendChild(run);

  // 「换回场景硬件 / 换成参考硬件」原本在这一屏，现在并到这里：两个都是「这套数从哪来」
  // 的开关，分开摆会让人以为它们管不同的事。
  const back = document.createElement('button');
  back.className = 'run ghost';
  back.textContent = (S.payload.bridge && S.payload.bridge.reference_used)
    ? '换回场景硬件' : '换成参考硬件';
  back.addEventListener('click', () => submitParams({
    reference: !(S.payload.bridge && S.payload.bridge.reference_used),
  }));
  row.appendChild(back);
  box.appendChild(row);
  return box;
}

/** 三道门之一的拒绝卡片。三类拒绝必须**分列**：只讲迁移那一类，第二步就会看起来像坏了。 */
function refusalCard(bridge, live) {
  const kindLabel = {
    unsupported: '门 1/3 · Unsupported',
    invalid: '门 2/3 · InvalidInput',
    value: '门 3/3 · ValueError',
    // 第四类不编号：三道门都是 ``costs_from_model_config`` 在拒绝，这一类是连配置都没读进来。
    // 写成「门 0/3」会被读成「比门 1 更早的一道门」，而它根本不是那道 guard 的判定。
    config: '未到门前 · 配置没载入',
  };
  const gate = document.createElement('div');
  gate.className = 'gate blocked';
  gate.innerHTML =
    '<div class="gate-head"><span class="gate-num">!</span>' +
    '<h3>' + esc(kindLabel[bridge.kind] || bridge.kind) + '</h3>' +
    (bridge.refusal_label ? '<span class="badge">' + esc(bridge.refusal_label) + '</span>' : '') +
    '<span class="badge">' + esc(bridge.stage || '') + '</span></div>';
  const gb = document.createElement('div');
  gb.className = 'gate-body';

  const reason = document.createElement('div');
  reason.className = 'reason ' + bridge.kind;
  reason.textContent = bridge.message;
  gb.appendChild(reason);

  const remedy = document.createElement('p');
  remedy.className = 'note';
  remedy.innerHTML = '<strong>补救：</strong>' + esc(bridge.remedy || '');
  gb.appendChild(remedy);

  // 结构性拒绝（图形状）：列出那几个算子，并说清「受限的是成本桥，不是内核」。
  if (bridge.structural_problems && bridge.structural_problems.length) {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.innerHTML = '<thead><tr><th>算子</th><th class="num">权重输入份数</th>' +
      '<th>层</th></tr></thead>';
    const tbody = document.createElement('tbody');
    for (const problem of bridge.structural_problems) {
      tbody.innerHTML += '<tr><td>' + esc(problem.operation_id) + '</td>' +
        '<td class="num">' + num(problem.weight_count) + '</td>' +
        '<td>' + esc(problem.name || '—') + '</td></tr>';
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    gb.appendChild(wrap);
  }

  // 开关清单只在**门 1** 有意义：门 2/3 的消息里不含那些字段名，于是没有一行是
  // enabled，这时该给的是「换成参考硬件」而不是一张全灰的清单。
  const enabled = (bridge.toggles || []).filter(row => row.enabled);
  if (enabled.length) {
    const title = document.createElement('p');
    title.className = 'note';
    title.innerHTML = '这份配置启用了内核本轮未迁移的服务。' +
      '<strong>逐项关掉</strong>，看清单一行行消失——然后会撞上第二道门。';
    gb.appendChild(title);
    for (const row of bridge.toggles) {
      const button = document.createElement('button');
      button.className = 'toggle' + (row.enabled ? '' : ' done');
      button.disabled = !live || !row.enabled;
      const now = row.value_is_null ? 'null' : String(row.value_now);
      button.innerHTML = '<span class="box">' + (row.enabled ? '!' : '✓') + '</span>' +
        '<span><span class="label">' + esc(row.label) + '</span>' +
        '<br><span class="field">' + esc(row.block + '.' + (row.field || '')) +
        '　现在 ' + esc(now) + '</span></span>';
      if (live && row.enabled) {
        button.addEventListener('click', () => submitParams({
          bridge_toggle: { block: row.block, field: row.field, value: row.off_value },
        }));
      }
      gb.appendChild(button);
    }
  } else if (bridge.kind === 'invalid' && live) {
    const ref = document.createElement('button');
    ref.className = 'run';
    ref.textContent = '换成参考硬件';
    ref.addEventListener('click', () => submitParams({ reference: true }));
    gb.appendChild(ref);
  }
  // 只读说明和上面那几支**并列**，不是 else：门 1 的快照里那些开关也是灰的，光灰不说
  // 会被读成坏了。这一条必须在任何一支都能出现。
  if (!live) {
    const off = document.createElement('p');
    off.className = 'empty';
    off.textContent = '导出的快照是只读的；点开关或换硬件需要 python tools/viewer.py。';
    gb.appendChild(off);
  }

  gate.appendChild(gb);
  return gate;
}

/* ------------------------------------------------------------- 参数屏 */

function renderParams() {
  const body = byId('params-body');
  // **表单没焦点时才重建。** ``render()`` 现在每帧调全部五个渲染器，而这里读的是用户
  // 已经敲进去的 DOM 值——每帧重建会把「填了一半的参数」冲回上一次提交的值，看起来
  // 像输入框自己在吃字符。焦点还在里面就不动它；用户点走（或按 Tab 出去）之后自然重建。
  if (body.contains(document.activeElement)) return;
  clear(body);
  const payload = S.payload;
  const live = payload.mode === 'live';
  const echo = payload.params_echo || {};

  if (!live) {
    const p = document.createElement('p');
    p.className = 'banner warn';
    p.textContent = '导出的快照是活页面的死拷贝：没有 Python 侧可以对话，所以参数编辑与' +
      '重跑在这里被禁用。改参数请跑 python tools/viewer.py。';
    body.appendChild(p);
  }

  const form = document.createElement('div');
  form.className = 'knobs';

  function knob(label, field, input) {
    const div = document.createElement('div');
    div.className = 'knob';
    const id = 'k-' + field.replace(/[^a-z0-9]/gi, '-');
    const lab = document.createElement('label');
    lab.htmlFor = id;
    lab.textContent = label;
    const sub = document.createElement('span');
    sub.className = 'field';
    sub.textContent = field;
    input.id = id;
    input.disabled = !live;
    div.append(lab, input, sub);
    form.appendChild(div);
    return input;
  }

  /** 有些旋钮在 scenario 里是嵌套的（costs.compute.<op>.duration_ns），
   *  echo 给的是扁平的点分路径，这里只用于显示，回填走的仍是 edit_scenario 的键。 */
  const vram = document.createElement('input');
  vram.type = 'number'; vram.value = echo.vram_capacity_bytes;
  knob('VRAM 容量', 'architecture.vram_capacity_bytes', vram);

  const reserved = document.createElement('input');
  reserved.type = 'number'; reserved.value = echo.runtime_reserved_bytes;
  knob('运行时预留', 'architecture.runtime_reserved_bytes', reserved);

  const budget = document.createElement('input');
  budget.type = 'number'; budget.value = echo.max_expanded_states;
  knob('状态预算（0 = 不限）', 'mapper.max_expanded_states', budget);

  const wall = document.createElement('input');
  wall.type = 'number'; wall.step = 'any'; wall.value = echo.wall_time_limit_s;
  knob('时间预算（秒，0 = 不限）', 'mapper.wall_time_limit_s', wall);

  for (const [field, label] of [['allow_copy_compute_overlap', '允许搬运/计算重叠'],
                                ['allow_eviction', '允许驱逐']]) {
    const div = document.createElement('div');
    div.className = 'knob';
    const wrap = document.createElement('label');
    wrap.className = 'check';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = !!echo[field];
    box.disabled = !live;
    box.dataset.flag = field;
    wrap.append(box, document.createTextNode(label));
    const sub = document.createElement('span');
    sub.className = 'field';
    sub.textContent = 'mapspace.' + field;
    div.append(wrap, sub);
    form.appendChild(div);
  }

  for (const op of payload.graph.operations) {
    const input = document.createElement('input');
    input.type = 'number';
    input.value = op.compute_ns;
    input.dataset.op = op.id;
    knob('算子 ' + op.id + ' 时长', 'costs.compute.' + op.id + '.duration_ns', input);
  }

  body.appendChild(form);

  const row = document.createElement('div');
  row.className = 'run-row';
  const run = document.createElement('button');
  run.className = 'run';
  run.textContent = '重跑搜索并对比';
  run.disabled = !live;
  run.addEventListener('click', () => {
    const params = {
      vram_capacity_bytes: Number(vram.value),
      runtime_reserved_bytes: Number(reserved.value),
      max_expanded_states: Number(budget.value),
      wall_time_limit_s: Number(wall.value),
      allow_copy_compute_overlap: !!body.querySelector('[data-flag="allow_copy_compute_overlap"]').checked,
      allow_eviction: !!body.querySelector('[data-flag="allow_eviction"]').checked,
    };
    const compute = {};
    for (const input of body.querySelectorAll('[data-op]')) compute[input.dataset.op] = Number(input.value);
    if (Object.keys(compute).length) params.compute_ns = compute;
    submitParams(params);
  });
  row.appendChild(run);

  const reset = document.createElement('button');
  reset.className = 'run ghost';
  reset.textContent = '恢复原值';
  reset.disabled = !live;
  reset.addEventListener('click', () => submitParams({
    vram_capacity_bytes: echo.vram_capacity_bytes,
    runtime_reserved_bytes: echo.runtime_reserved_bytes,
    max_expanded_states: echo.max_expanded_states,
    wall_time_limit_s: echo.wall_time_limit_s,
    allow_copy_compute_overlap: echo.allow_copy_compute_overlap,
    allow_eviction: echo.allow_eviction,
    compute_ns: Object.fromEntries(payload.graph.operations.map(op => [op.id, op.compute_ns])),
  }));
  row.appendChild(reset);
  body.appendChild(row);

  if (S.compare) renderCompare(body);
}

function renderCompare(body) {
  const { before, after } = S.compare;
  // 这一格现在有两种来历：同一张图上改了旋钮，或者顶栏换了一张图。**必须分开说**——
  // 「重跑前后」下面摆着两张不同计算图的柱状对比，会被读成「换了图省了多少时间」，
  // 而那两个数字之间根本没有可比的大小关系（不同的图、不同的算子）。
  const switched = before.scenario_id !== after.scenario_id;
  const box = document.createElement('div');
  box.className = 'gate';
  box.innerHTML = '<div class="gate-head"><h3>' + (switched ? '换图前后' : '重跑前后') + '</h3>' +
    '<span class="badge">' + (switched
      ? esc(before.scenario_id) + ' → ' + esc(after.scenario_id) + '：两张不同的图，数字不可比大小'
      : '同一套代码路径') + '</span></div>';
  const inner = document.createElement('div');
  inner.className = 'gate-body';

  const metrics = [
    ['makespan', before.solution.makespan_ns, after.solution.makespan_ns, fmtNs],
    ['峰值显存', before.solution.peak_vram_bytes, after.solution.peak_vram_bytes, fmtBytes],
    ['搬运', before.solution.h2d_bytes, after.solution.h2d_bytes, fmtBytes],
    ['动作数', before.solution.action_count, after.solution.action_count, num],
    ['展开状态', before.explore.verdict.expanded_states, after.explore.verdict.expanded_states, num],
  ];
  const diff = document.createElement('div');
  diff.className = 'diff';
  for (const [label, b, a, fmt] of metrics) {
    const div = document.createElement('div');
    let delta = '';
    if (typeof b === 'number' && typeof a === 'number' && b !== a) {
      const dir = a > b ? 'up' : 'down';
      const pct = b === 0 ? '' : ' (' + ((a - b) / Math.abs(b) * 100).toFixed(1) + '%)';
      delta = '<span class="delta ' + dir + '">' + (a > b ? '+' : '') + fmt(a - b) + pct + '</span>';
    }
    div.innerHTML = '<div class="k">' + esc(label) + '</div>' +
      '<div class="v">' + esc(fmt(a)) + ' ' + delta + '</div>' +
      '<div class="b">原 ' + esc(fmt(b)) + '</div>';
    diff.appendChild(div);
  }
  inner.appendChild(diff);

  const bStat = before.explore.verdict.status;
  const aStat = after.explore.verdict.status;
  if (bStat !== aStat) {
    const p = document.createElement('p');
    p.className = 'note';
    p.innerHTML = '<strong>终局变了：</strong>' + esc(bStat) + ' → ' + esc(aStat) +
      '。预算耗尽给出的 unknown/feasible 不是「无解」。';
    inner.appendChild(p);
  }
  box.appendChild(inner);
  body.appendChild(box);
}

/* ------------------------------------------------------- 走带与动作面板 */

function renderTransport() {
  const payload = S.payload;
  const total = stepLength();
  const index = Math.min(isReplay() ? S.cursor : S.path.length - 1, total);
  const slider = byId('scrub');
  slider.max = String(total);
  slider.value = String(index);

  byId('step-label').textContent = '第 ' + index + ' / ' + total + ' 步';

  for (const button of document.querySelectorAll('.mode')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === S.mode));
  }
  byId('play').textContent = S.playing ? '❚❚' : '▶';

  renderActions();
}

function renderActions() {
  const area = byId('actions');
  clear(area);
  const payload = S.payload;
  const live = payload.mode === 'live';

  if (isReplay()) {
    // 回放：只读。列出当前步的合法动作，计划的那个高亮，其余灰掉并给一行理由。
    const index = Math.min(S.cursor, payload.log.states.length - 1);
    const node = findNodeForState(index);
    const hint = document.createElement('span');
    hint.className = 'empty';
    hint.textContent = node
      ? '回放模式下动作只读——这是搜索出的计划。'
      : '此步的状态不在搜索图里（不该发生）。';
    area.appendChild(hint);
    if (!node) return;

    const plannedAction = payload.solution.actions[index] || null;
    const plannedKey = actionKey(plannedAction);
    const illegal = illegalByNode(node.id);

    for (const child of (node.children || [])) {
      const button = document.createElement('button');
      button.className = 'act' + (actionKey(child.action) === plannedKey ? ' planned' : '');
      button.disabled = true;
      button.title = child.cut ? '被支配：已有更省的路径（g=' + child.cut.g_existing_ns + ' ns）'
        : '回放模式下不可执行；切到「单步探索」可以走';
      button.innerHTML = '<span class="kind">' + esc(ACTION_LABEL[child.action.kind]) + '</span>' +
        esc(child.action.operation_id || child.action.tensor_id || '');
      area.appendChild(button);
    }
    for (const item of illegal) {
      const button = document.createElement('button');
      button.className = 'act';
      button.disabled = true;
      button.title = item.message;
      button.innerHTML = '<span class="kind">' + esc(ACTION_LABEL[item.action.kind]) + '</span>' +
        esc(item.action.operation_id || item.action.tensor_id || '');
      area.appendChild(button);
    }
    return;
  }

  // 单步：从当前节点出发的合法动作可点，非法动作列出理由但不可点。
  const node = currentNode();
  if (!node) return;
  const illegal = illegalByNode(node.id);
  const legalIds = new Set((node.children || []).map(c => c.to));

  const label = document.createElement('span');
  label.className = 'empty';
  label.textContent = '点一个合法动作前进：';
  area.appendChild(label);

  for (const child of (node.children || [])) {
    const button = document.createElement('button');
    button.className = 'act' + (child.cut ? '' : '');
    const target = payload.explore.nodes[child.to];
    button.disabled = !!child.cut;
    button.title = child.cut
      ? '被支配：这条边通往的状态已有更省的路径（g=' + child.cut.g_existing_ns + ' ns）'
      : '前进到节点 #' + child.to + '（g=' + fmtNs(target ? target.g_ns : 0) + '）';
    button.innerHTML = '<span class="kind">' + esc(ACTION_LABEL[child.action.kind]) + '</span>' +
      esc(child.action.operation_id || child.action.tensor_id || '') +
      (child.cut ? ' <span class="kind">被支配</span>' : '');
    button.addEventListener('click', () => jumpTo(child.to));
    area.appendChild(button);
  }
  for (const item of illegal) {
    const button = document.createElement('button');
    button.className = 'act';
    button.disabled = true;
    button.title = item.message;
    button.innerHTML = '<span class="kind">' + esc(ACTION_LABEL[item.action.kind]) + '</span>' +
      esc(item.action.operation_id || item.action.tensor_id || '');
    area.appendChild(button);
  }
  if (!live) return;
}

/** 回放游标 → 搜索图里的节点。靠 state_key 之外的等价关系：t_ns + op/copy 状态。 */
function findNodeForState(index) {
  const payload = S.payload;
  const state = payload.log.states[index];
  if (!state) return null;
  const nodes = payload.explore.nodes;
  for (const node of nodes) {
    if (node.t_ns !== state.t_ns) continue;
    if (node.used_vram_bytes !== state.used_vram_bytes) continue;
    if (sameMap(node.op_status, state.op_status) && sameMap(node.copy_status, state.copy_status)) {
      return node;
    }
  }
  return null;
}

function sameMap(a, b) {
  const ka = Object.keys(a || {}), kb = Object.keys(b || {});
  if (ka.length !== kb.length) return false;
  for (const key of ka) if ((a || {})[key] !== (b || {})[key]) return false;
  return true;
}

function illegalByNode(nodeId) {
  const payload = S.payload;
  const table = payload.explore.candidates;
  if (!table) return [];
  const entry = (table.nodes || []).find(row => row[0] === nodeId);
  if (!entry) return [];
  return entry[1].map(([kind, target, index]) => {
    const [code, message] = table.messages[index];
    const action = kind === 'COMPUTE'
      ? { kind: kind, operation_id: target }
      : kind === 'ADVANCE' ? { kind: kind } : { kind: kind, tensor_id: target };
    return { action: action, code: code, message: message };
  });
}

/* ------------------------------------------------------------- 检查器 */

function renderInspector() {
  const panel = byId('inspector');
  clear(panel);
  const payload = S.payload;
  const snap = currentSnapshot();
  if (!snap) {
    panel.innerHTML = '<p class="empty">没有当前状态。</p>';
    return;
  }

  function section(title) {
    const h = document.createElement('h3');
    h.textContent = title;
    panel.appendChild(h);
  }

  section('当前步');
  const kv = document.createElement('dl');
  kv.className = 'kv';
  const rows = [
    ['时间 t', fmtNs(snap.t_ns)],
    ['累计代价 g', isReplay() ? fmtNs(snap.t_ns) : fmtNs(currentNode() ? currentNode().g_ns : null)],
    ['显存占用', fmtBytes(snap.used_vram_bytes)],
    ['模式', isReplay() ? '回放计划' : '单步探索'],
  ];
  for (const [k, v] of rows) {
    kv.innerHTML += '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>';
  }
  panel.appendChild(kv);

  section('张量位置');
  const chips = document.createElement('div');
  chips.className = 'chips';
  const graph = payload.graph;
  for (const tensor of graph.tensors) {
    const status = (snap.copy_status || {})[tensor.id] || (tensor.is_weight ? 'ABSENT' : 'READY');
    const cls = status === 'READY' ? 'ready' : status === 'RESERVED_COPY' ? 'reserved' : '';
    chips.innerHTML += '<span class="chip ' + cls + '">' + esc(tensor.name) + ' · ' +
      esc(status) + '</span>';
  }
  panel.appendChild(chips);

  section('算子状态');
  const ops = document.createElement('div');
  ops.className = 'chips';
  for (const op of graph.operations) {
    const status = (snap.op_status || {})[op.id] || 'PENDING';
    const running = (snap.running || []).some(t => t.target_id === op.id);
    const cls = running ? 'running' : (status === 'DONE' ? 'done' : '');
    ops.innerHTML += '<span class="chip ' + cls + '">' + esc(op.id) + ' · ' +
      esc(running ? 'RUNNING' : status) + '</span>';
  }
  panel.appendChild(ops);

  const verdict = payload.explore.verdict;
  section('搜索');
  const vk = document.createElement('dl');
  vk.className = 'kv';
  vk.innerHTML =
    '<dt>status</dt><dd>' + esc(verdict.status) + '</dd>' +
    '<dt>reason</dt><dd>' + esc(verdict.termination_reason) + '</dd>' +
    '<dt>已证最优</dt><dd>' + esc(String(verdict.optimality_proven)) + '</dd>' +
    '<dt>展开 / 见过</dt><dd>' + num(verdict.expanded_states) + ' / ' + num(verdict.visited_states) + '</dd>' +
    '<dt>剪边</dt><dd>' + num((payload.explore.cut_counts || {}).dominated || 0) + '</dd>';
  panel.appendChild(vk);
}

/* ------------------------------------------------------------------ 渲染 */

/** 唯一重绘出口：**五块屏每帧全画**。
 *
 * 上一版每帧只画一块（``if (S.tab === …)``），配合一个失效的 ``hidden``，结果是屏幕上
 * 五块并排、四块永远空着。五块屏共用一个游标，所以它们本来就该一起动——把哪一块跳过，
 * 换来的不是性能，是「这块坏了」的观感。
 */
/** 八个渲染器各自的名字，给横幅点名用（下面 :func:`render` 用）。 */
const RENDERERS = [
  ['页眉对账', renderHeader],
  ['走带', renderTransport],
  ['检查器', renderInspector],
  ['搜索树', renderTree],
  ['计算图', renderGraph],
  ['时间线', renderTimeline],
  ['时长来源', renderBridge],
  ['参数', renderParams],
];

function render() {
  if (!S.payload) return;
  // **一个渲染器一个 try**，不是八个共用一个。
  //
  // 共用的时候，第一个抛出去的会把**后面还没跑的**全部带走：容量调小 → 搜索无解 →
  // 状态时间线读空数组抛 `undefined.t_ns` → 「时长来源」与「参数」两屏跟着一起变空。
  // 屏上只有一句「渲染出错」，看的人只会以为整个查看器坏了，而真正坏的只有一屏。
  // 分开之后故障停在它真正发生的地方，横幅点名是哪几屏，其余照画。
  const failed = [];
  for (const [label, paint] of RENDERERS) {
    try {
      paint();
    } catch (error) {
      failed.push(label);
      if (window.console) console.error(error);
    }
  }
  if (failed.length) {
    setBanner('render', '渲染出错：' + failed.join('、') + ' 画不出来（这是查看器的 bug，不是内核的结论）');
  } else {
    clearBanner('render');
  }
}

/* ------------------------------------------------------------- 交互 */

/** 跳到某一屏。那排按钮是**跳转条**，不是标签页：五块屏都在页面上，所以它做的事只是
 *  ``scrollIntoView``，不切显隐。 */
function jumpToPane(name) {
  const pane = document.querySelector('.pane[data-pane="' + name + '"]');
  if (!pane) return;
  if (pane.scrollIntoView) pane.scrollIntoView({ behavior: 'smooth', block: 'start' });
  S.pane = name;
  for (const button of document.querySelectorAll('.tab')) {
    button.setAttribute('aria-current', String(button.dataset.tab === name));
  }
}

/** 高亮跟着**滚动位置**走。五块屏同时在场，所以「当前是哪一屏」只可能由视口回答；
 *  跟着「最后点了谁」走的话，滚回上面时高亮会撒谎。 */
function syncPaneHighlight() {
  const panes = Array.from(document.querySelectorAll('.pane'));
  if (!panes.length) return;
  let current = panes[0];
  for (const pane of panes) {
    if (pane.getBoundingClientRect().top <= 140) current = pane;
  }
  if (current.dataset.pane === S.pane) return;
  S.pane = current.dataset.pane;
  for (const button of document.querySelectorAll('.tab')) {
    button.setAttribute('aria-current', String(button.dataset.tab === S.pane));
  }
}

function setMode(mode) {
  if (S.mode === mode) return;
  S.mode = mode;
  stopPlay();
  if (mode === 'step') {
    // 进单步时把游标锚在当前回放位置上对应的节点，别跳回起点。
    const node = findNodeForState(S.cursor);
    const startId = S.payload.explore.start_id;
    S.path = node ? pathTo(node.id) : (startId != null ? [startId] : []);
    if (!S.path.length && startId != null) S.path = [startId];
  }
  render();
}

/** 沿父链取回一条从起点到 id 的路径。 */
function pathTo(id) {
  const nodes = S.payload.explore.nodes;
  const path = [];
  let cursor = id;
  let guard = 0;
  while (cursor != null && guard++ < nodes.length + 2) {
    path.unshift(cursor);
    const node = nodes[cursor];
    if (!node || node.parent == null) break;
    cursor = node.parent;
  }
  return path;
}

/** 跳到搜索树上的某个节点：把单步游标搬到那条路径上，图与时间线随之一起变。
 *
 * 这是「点树上任一节点 → 下面两块跟着走」的全部实现。做成**路径**而不是单点，是因为
 * 单步模式的游标本来就是一条分支路径（见文件头第 3 条）：只跳到节点而不认它的来路，
 * 时间线上会凭空多出一段没走过的执行。``pathTo()`` 沿父链取回整条路径。
 *
 * 顺带把模式切到单步：只有单步模式的游标是一维路径，回放的整数游标表达不了「跳到
 * 搜索树上的任意节点」这件事。
 */
function jumpTo(nodeId) {
  const nodes = S.payload.explore.nodes;
  if (!nodes[nodeId]) return;
  stopPlay();
  S.mode = 'step';
  S.path = pathTo(nodeId);
  render();
}

function stepBack() {
  if (isReplay()) {
    if (S.cursor > 0) S.cursor -= 1;
  } else if (S.path.length > 1) {
    S.path = S.path.slice(0, -1);
  }
  render();
}

function stepForward() {
  if (isReplay()) {
    const last = S.payload.log.states.length - 1;
    if (S.cursor < last) S.cursor += 1;
    else stopPlay();
  } else {
    const node = currentNode();
    if (!node) return;
    const next = (node.children || []).find(child => !child.cut);
    if (next) S.path = S.path.concat([next.to]);
    else stopPlay();
  }
  render();
}

function resetCursor() {
  if (isReplay()) S.cursor = 0;
  else {
    const startId = S.payload.explore.start_id;
    S.path = startId != null ? [startId] : [];
  }
  stopPlay();
  render();
}

function togglePlay() {
  if (S.playing) { stopPlay(); renderTransport(); return; }
  S.playing = true;
  S.timer = setInterval(() => {
    if (isReplay()) {
      if (S.cursor >= S.payload.log.states.length - 1) { stopPlay(); renderTransport(); return; }
      S.cursor += 1;
    } else {
      const node = currentNode();
      const next = node && (node.children || []).find(child => !child.cut);
      if (!next) { stopPlay(); renderTransport(); return; }
      S.path = S.path.concat([next.to]);
    }
    render();
  }, 620);
  renderTransport();
}

function stopPlay() {
  S.playing = false;
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
}

/* ------------------------------------------------- 与服务端对话（仅服务模式） */

function submitParams(params) {
  // 快照模式下所有入口都已经禁用了，所以走到这里说明有哪个入口漏了禁用。**不静默返回**：
  // 点了没反应是这类工具最容易被当成「坏了」的失败方式，宁可显式说出来。
  if (S.payload.mode !== 'live') {
    setBanner('snapshot', '导出的快照是只读的，没有 Python 侧可以对话；' +
      '改参数请跑 python tools/viewer.py。', 'warn');
    return;
  }
  S.bridgeMsg = null;
  fetch('api/params', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(response => {
    if (response.status === 422) {
      return response.json().then(body => {
        // 门 3：手填的参数被 frozen dataclass 自己的不变式拒绝。原样显示，别吞。
        S.bridgeMsg = '参数被拒绝（' + (body.error_class || 'SpecError') + '）：' + body.error;
        jumpToPane('source');
        render();
        return null;
      });
    }
    if (!response.ok) throw new Error('HTTP ' + response.status);
    return response.json();
  }).then(result => {
    if (!result) return;
    S.compare = result;
    S.payload = result.after;
    keepCursor();
    render();
  }).catch(error => {
    setBanner('rerun', '重跑失败：' + error.message);
  });
}

/** 换载荷后把游标夹回合法范围，别让它指向不存在的步。 */
function keepCursor() {
  const states = S.payload.log.states;
  if (S.cursor >= states.length) S.cursor = Math.max(0, states.length - 1);
  const startId = S.payload.explore.start_id;
  if (!S.path.length && startId != null) S.path = [startId];
  const max = S.payload.explore.nodes.length - 1;
  S.path = S.path.filter(id => id <= max);
  if (!S.path.length && startId != null) S.path = [startId];
}

/* ------------------------------------------------------------------ 接线 */

function wire() {
  for (const button of document.querySelectorAll('.tab')) {
    button.addEventListener('click', () => jumpToPane(button.dataset.tab));
  }
  for (const button of document.querySelectorAll('.mode')) {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  }
  byId('play').addEventListener('click', togglePlay);
  byId('back').addEventListener('click', () => { stopPlay(); stepBack(); });
  byId('fwd').addEventListener('click', () => { stopPlay(); stepForward(); });
  byId('reset').addEventListener('click', resetCursor);
  // 换图走的是和参数提交**同一条路**：一次 POST，回来的是一整对新载荷。失败时（名字不认、
  // 那份载不进来）服务端回 422，横幅照常显示原因；而选择器会被下一帧的
  // ``renderScenarioPicker()`` 拨回载荷里那一份——屏上是哪张图，选择器就写着哪张，
  // 不留「顶栏说换了、图没换」这种半截状态。
  byId('scenario-pick').addEventListener('change', event => {
    stopPlay();
    submitParams({ scenario: event.target.value });
  });
  // 走带滑块的取值是**第几步**，不是节点 id。上一版回放时按步号用、单步时又当成节点 id
  // 交给 ``pathTo()``——同一根滑块两种语义，拖到第 3 步会跳到「id 为 3 的那个节点」，
  // 而那是另一条路径上走到第 5 步的位置。单步下正确的动作是把路径**截回**第 n 步。
  byId('scrub').addEventListener('input', event => {
    stopPlay();
    const step = Number(event.target.value);
    if (isReplay()) S.cursor = step;
    else S.path = S.path.slice(0, Math.max(1, step + 1));
    render();
  });
  document.addEventListener('keydown', event => {
    if (event.target.tagName === 'INPUT') return;
    if (event.key === 'ArrowRight') { stopPlay(); stepForward(); }
    else if (event.key === 'ArrowLeft') { stopPlay(); stepBack(); }
    else if (event.key === ' ') { event.preventDefault(); togglePlay(); }
    else if (event.key === 'r') resetCursor();
  });
  // 跳转条的高亮跟着滚动走。用 rAF 合流：scroll 事件比帧密得多，每来一次就读一遍
  // 全部 pane 的 getBoundingClientRect 会强制同步布局，滚动会卡。
  let queued = false;
  window.addEventListener('scroll', () => {
    if (queued) return;
    queued = true;
    (window.requestAnimationFrame || (fn => setTimeout(fn, 16)))(() => {
      queued = false;
      syncPaneHighlight();
    });
  }, { passive: true });
}

function bootViewer(payload) {
  S.payload = payload;
  const startId = payload.explore.start_id;
  S.path = startId != null ? [startId] : [];
  S.cursor = 0;
  wire();
  render();
  syncPaneHighlight();
}

/** 致命错误：把整页换成一句人话。别让用户对着浏览器自己的报错猜。 */
function fatal(message) {
  // ``white-space: pre-line`` 是为了让消息里的换行真的断行（esc 会把 <br> 转义掉，
  // 所以不能靠标签）。
  document.body.innerHTML = '<div style="padding:24px;max-width:64ch;' +
    'font:14px/1.7 system-ui;color:#333;white-space:pre-line">' + esc(message) + '</div>';
}

if (window.__BAKED__) {
  bootViewer(window.__BAKED__);
} else if (location.protocol === 'file:') {
  // 直接用浏览器打开 viewer_static/index.html：它是**外壳**，数据要从服务端取，而
  // file:// 下那个 fetch 一定失败。这条路径以前只丢一句浏览器自己的「Failed to fetch」，
  // 用户没法从里面读出该干什么——所以要把它单独认出来，说清两条出路。
  fatal('这是直接打开的源文件（file://），外壳里没有数据。两种打开方式：\n\n' +
    '· 起服务：在 mapping/ 目录下跑 python tools/viewer.py —scenario examples/chain-cap160.json' +
    '（用 modeling/.venv 里那个 python），然后打开终端打印的地址；\n' +
    '· 或者导出一个自包含快照：python tools/viewer.py --export out.html，那个双击就能开。');
} else {
  fetch('api/payload').then(response => {
    if (!response.ok) throw new Error('服务端回了 HTTP ' + response.status);
    return response.json();
  }).then(bootViewer).catch(error => {
    fatal('取不到数据：' + error.message + '。\n\n' +
      '服务可能已经停了——回到跑 viewer.py 的那个终端看它还在不在；' +
      '不在就在 mapping/ 目录下重新跑一次 python tools/viewer.py。');
  });
}
