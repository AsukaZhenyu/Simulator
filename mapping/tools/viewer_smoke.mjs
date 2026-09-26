#!/usr/bin/env node
/* 查看器前端的**落地**烟测：在 node 里把 app.js 真跑一遍。
 *
 * 为什么需要它：`viewer.py --self-check` 能断言 index.html 里有五个标签、每块屏有标题，
 * 但它**看不见 JS 干的事**。上一版「计算图一条边都没有」正是这样漏掉的——`render()` 不抛
 * 异常、横幅是空的、元素也建出来了，只是每个坐标都是 `NaN`（`graph.columns` 被当成数字
 * 基数用，`{…} + rank*2` 走字符串拼接）。而 **SVG 对无效属性是静默的**：`<rect x="NaN">`
 * 的 `x` 被忽略（所有框挤到 x=0）、`<path d="…NaN…">` 整条元素丢弃（一条边都画不出来）。
 * 所以「render 没抛」+「边数 > 0」这种断言在 bug 版本里照样全绿。
 *
 * 这个文件用 ~300 行 DOM 垫片换到三件 grep 拿不到的东西：
 *
 * 1. **五个渲染器都真的跑了。** 每块屏都要有「父节点不是 pane 自己、也不在 `.pane-head`
 *    里」的元素——空 `<svg>` 和空 `#params-body` 一个都没有。这正是 v1「第一次打开只有
 *    计算图、另外四块是空盒子」那条缺陷的形状。
 * 2. **通用网：扫整棵文档树的每个属性与每段文字，值里含 `NaN` 就失败。** 不针对某一个
 *    坐标写断言——将来任何一处算错都会落在这里，不需要有人先想到去写那条断言。
 * 3. **计数卡相等，不卡大于零。** 边的条数必须**等于** `graph.links.length`，命中圈必须
 *    **等于** `explore.nodes.length`。
 *
 * 还有一条：**`render()` 里的 try/catch 会把渲染异常吞成横幅**。所以「横幅是空的」比
 * 「没抛异常」强得多，两件都要断言，否则被吞掉的 bug 看起来和成功一模一样。
 *
 * 这个脚本**不引入任何依赖**，也不是测试套件的一部分：它是 `tools/` 下的开发工具，
 * 由 `viewer.py --self-check` 在 `node` 可用时调用，不可用就跳过（打印一句跳过原因）。
 *
 * 用法：
 *   node tools/viewer_smoke.mjs --app viewer_static/app.js --html viewer_static/index.html \
 *        --payload <cases.json>
 *   node tools/viewer_smoke.mjs … --falsify     # 证伪：注入一个 NaN 坐标，要求那张网变红
 *
 * `cases.json` 的形状是 `{"cases": [{"name": "...", "payload": {...}}, …]}`，由
 * `viewer.py` 现装配现写临时文件（**不落盘到仓库里**）。
 *
 * 输出：一行一个 JSON 对象（stdout 只有这些，别的都走 stderr），最后一行是
 * `{"done": true, "total": N, "failed": M}`。退出码 0 表示全部通过。
 */
'use strict';

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import vm from 'node:vm';

/* ------------------------------------------------------------------ 常量 */

const HTML_NS = 'http://www.w3.org/1999/xhtml';
const SVG_NS = 'http://www.w3.org/2000/svg';

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr']);

/** 垫片里能读能写的「属性背书的」property：读 = 读属性，写 = 写属性。
 *  `value` / `checked` 不在这里——真 DOM 里它们是**独立**的 property，属性只是初值。 */
const ATTR_PROPS = ['id', 'className', 'title', 'type', 'name', 'step', 'min', 'max',
  'src', 'href', 'placeholder', 'for', 'role', 'lang'];

const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' };

/* ------------------------------------------------------------------ 解析 */

function decode(text) {
  return String(text).replace(/&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z]+);/g, (all, body) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X'
        ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : all;
    }
    return Object.prototype.hasOwnProperty.call(ENT, body) ? ENT[body] : all;
  });
}

function escapeText(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAttr(value) {
  return escapeText(value).replace(/"/g, '&quot;');
}

const TAG_RE = new RegExp(
  '<!--[\\s\\S]*?-->' +
  '|<!\\[CDATA\\[[\\s\\S]*?\\]\\]>' +
  '|<!doctype[^>]*>' +
  '|<\\/([a-zA-Z][\\w:-]*)\\s*>' +
  '|<([a-zA-Z][\\w:-]*)((?:"[^"]*"|\'[^\']*\'|[^>"\'])*?)(\\/?)>',
  'gi');

const ATTR_RE = /([^\s"'>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

function parseAttrs(text) {
  const out = [];
  ATTR_RE.lastIndex = 0;
  let m;
  while ((m = ATTR_RE.exec(text || ''))) {
    out.push([m[1], decode(m[2] != null ? m[2] : (m[3] != null ? m[3] : (m[4] != null ? m[4] : '')))]);
  }
  return out;
}

/* ------------------------------------------------------------------ 节点 */

function serialize(node) {
  if (node.nodeType === 3) return escapeText(node.nodeValue);
  if (node.nodeType === 11) return node.childNodes.map(serialize).join('');
  const name = node.tagName.toLowerCase();
  let out = '<' + name;
  for (const [key, value] of node._attrs) {
    out += value === '' ? ' ' + key : ' ' + key + '="' + escapeAttr(value) + '"';
  }
  out += '>';
  if (VOID.has(name)) return out;
  out += node.childNodes.map(serialize).join('');
  return out + '</' + name + '>';
}

class TextNode {
  constructor(data) {
    this.nodeType = 3;
    this.nodeName = '#text';
    this.nodeValue = String(data);
    this.parentNode = null;
  }

  get textContent() { return this.nodeValue; }

  set textContent(value) { this.nodeValue = String(value); }
}

function hasClass(node, name) {
  return String(node.className || '').split(/\s+/).indexOf(name) >= 0;
}

function kebab(key) {
  return String(key).replace(/[A-Z]/g, ch => '-' + ch.toLowerCase());
}

/** 子节点容器：`El` 与 `DocumentFragment`、`Document` 共用这一套。
 *  只实现 `app.js` 真的调到的那部分 DOM。 */
class Parent {
  constructor() {
    this.childNodes = [];
    this.parentNode = null;
  }

  appendChild(node) {
    if (node && node.nodeType === 11) {
      for (const child of node.childNodes.slice()) this.appendChild(child);
      node.childNodes.length = 0;
      return node;
    }
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  append(...nodes) {
    for (const node of nodes) this.appendChild(typeof node === 'string' ? new TextNode(node) : node);
  }

  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index >= 0) this.childNodes.splice(index, 1);
    node.parentNode = null;
    return node;
  }

  get firstChild() { return this.childNodes[0] || null; }

  get children() { return this.childNodes.filter(node => node.nodeType === 1); }

  get textContent() { return this.childNodes.map(node => node.textContent).join(''); }

  set textContent(value) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    if (value !== '' && value != null) this.appendChild(new TextNode(value));
  }

  get innerHTML() { return this.childNodes.map(serialize).join(''); }

  set innerHTML(html) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    for (const child of parseFragment(String(html)).childNodes.slice()) this.appendChild(child);
  }

  contains(node) {
    for (let walk = node; walk; walk = walk.parentNode) if (walk === this) return true;
    return false;
  }

  querySelectorAll(selector) {
    const match = parseSelector(selector);
    const out = [];
    const walk = node => {
      for (const child of node.childNodes) {
        if (child.nodeType !== 1) continue;
        if (match(child)) out.push(child);
        walk(child);
      }
    };
    walk(this);
    return out;
  }

  querySelector(selector) {
    const all = this.querySelectorAll(selector);
    return all.length ? all[0] : null;
  }
}

/** 选择器支持：`tag`、`.class`、`#id`、`[attr]`、`[attr="v"]` 的任意拼接。
 *  `app.js` 只用到这几种（`.pane[data-pane="tree"]`、`[data-op]`、`.tab`、`.mode`）。
 *  遇到不认识的写法**抛异常**，不静默返回空——静默的空集正好是这类垫片最会骗人的地方。 */
function parseSelector(selector) {
  const text = String(selector).trim();
  const head = /^([a-zA-Z][\w-]*)?((?:[.#][\w-]+|\[[^\]]*\])*)$/.exec(text);
  if (!head) throw new Error('垫片不认识的 CSS 选择器：' + selector);
  const tag = head[1] ? head[1].toLowerCase() : null;
  const parts = [];
  const re = /([.#])([\w-]+)|\[([^\]=]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]/g;
  let m;
  while ((m = re.exec(head[2]))) {
    if (m[1] === '.') parts.push({ kind: 'class', value: m[2] });
    else if (m[1] === '#') parts.push({ kind: 'id', value: m[2] });
    else {
      const value = [m[4], m[5], m[6]].find(item => item != null);
      parts.push({ kind: 'attr', name: m[3], value: value == null ? null : value });
    }
  }
  return node => {
    if (tag && node.tagName.toLowerCase() !== tag) return false;
    for (const part of parts) {
      if (part.kind === 'class') {
        if (!hasClass(node, part.value)) return false;
      } else if (part.kind === 'id') {
        if (node.getAttribute('id') !== part.value) return false;
      } else if (part.value == null) {
        if (!node.hasAttribute(part.name)) return false;
      } else if (node.getAttribute(part.name) !== part.value) return false;
    }
    return true;
  };
}

class El extends Parent {
  constructor(tag, namespace) {
    super();
    this.nodeType = 1;
    this.namespaceURI = namespace || HTML_NS;
    // SVG 是大小写敏感的，真 DOM 里 `createElementNS(SVG_NS,'text').tagName === 'text'`。
    this.tagName = this.namespaceURI === SVG_NS ? String(tag) : String(tag).toUpperCase();
    this.nodeName = this.tagName;
    this._attrs = new Map();
    this._listeners = new Map();
    this._value = null;
    this.scrolledIntoView = 0;
    this.style = {
      setProperty(key, value) { this[key] = String(value); },
      getPropertyValue(key) { return this[key] || ''; },
      removeProperty(key) { delete this[key]; },
    };
    const self = this;
    this.dataset = new Proxy({}, {
      get(_, key) {
        if (typeof key !== 'string') return undefined;
        return self._attrs.get('data-' + kebab(key));
      },
      set(_, key, value) { self._attrs.set('data-' + kebab(key), String(value)); return true; },
      deleteProperty(_, key) { self._attrs.delete('data-' + kebab(key)); return true; },
      has(_, key) { return self._attrs.has('data-' + kebab(key)); },
      ownKeys() { return [...self._attrs.keys()].filter(key => key.startsWith('data-')); },
      getOwnPropertyDescriptor(_, key) {
        const value = self._attrs.get('data-' + kebab(key));
        return value === undefined ? undefined : { enumerable: true, configurable: true, value };
      },
    });
  }

  getAttribute(name) {
    return this._attrs.has(name) ? this._attrs.get(name) : null;
  }

  setAttribute(name, value) { this._attrs.set(name, String(value)); }

  removeAttribute(name) { this._attrs.delete(name); }

  hasAttribute(name) { return this._attrs.has(name); }

  get hidden() { return this.hasAttribute('hidden'); }

  set hidden(value) {
    if (value) this.setAttribute('hidden', '');
    else this.removeAttribute('hidden');
  }

  get disabled() { return this.hasAttribute('disabled'); }

  set disabled(value) {
    if (value) this.setAttribute('disabled', '');
    else this.removeAttribute('disabled');
  }

  get checked() { return this.hasAttribute('checked'); }

  set checked(value) {
    if (value) this.setAttribute('checked', '');
    else this.removeAttribute('checked');
  }

  get value() {
    return this._value != null ? this._value : (this.getAttribute('value') || '');
  }

  set value(value) { this._value = String(value); }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type).push(handler);
  }

  removeEventListener(type, handler) {
    const list = this._listeners.get(type) || [];
    const index = list.indexOf(handler);
    if (index >= 0) list.splice(index, 1);
  }

  /** 垫片自己的派发入口（不是 DOM API）：烟测用它模拟一次点击/提交。 */
  fire(type, event) {
    const detail = Object.assign({ type, target: this, preventDefault() {} }, event || {});
    for (const handler of (this._listeners.get(type) || []).slice()) handler(detail);
  }

  scrollIntoView() { this.scrolledIntoView += 1; }

  /** 假布局：按文档序给一个单调的 top，单位随意放大到 1000。
   *  `syncPaneHighlight()` 挑「top ≤ 140 的最后一块屏」；这样只有第一块屏的 top 会
   *  落进那个窗口，结果确定，不依赖真实排版。 */
  getBoundingClientRect() {
    const order = this.ownerDocument ? this.ownerDocument.preOrderIndex(this) : 0;
    return { top: order * 1000, bottom: order * 1000, left: 0, right: 0, width: 0, height: 0 };
  }
}

for (const name of ATTR_PROPS) {
  Object.defineProperty(El.prototype, name, {
    configurable: true,
    get() { return this.getAttribute(name === 'className' ? 'class' : name) || ''; },
    set(value) { this.setAttribute(name === 'className' ? 'class' : name, value); },
  });
}

class DocumentFragment extends Parent {
  constructor() {
    super();
    this.nodeType = 11;
    this.nodeName = '#document-fragment';
  }
}

class Doc extends Parent {
  constructor() {
    super();
    this.nodeType = 9;
    this.nodeName = '#document';
    this.activeElement = null;
    this._listeners = new Map();
    this._order = null;
  }

  createElement(tag) { return new El(tag, HTML_NS); }

  createElementNS(namespace, tag) { return new El(tag, namespace); }

  createTextNode(data) { return new TextNode(data); }

  createDocumentFragment() { return new DocumentFragment(); }

  get documentElement() { return this.childNodes.filter(node => node.nodeType === 1)[0] || null; }

  get body() {
    const root = this.documentElement;
    if (!root) return null;
    if (root.tagName === 'BODY') return root;
    return root.querySelector('body');
  }

  getElementById(id) {
    const all = this.querySelectorAll('#' + id);
    return all.length ? all[0] : null;
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type).push(handler);
  }

  removeEventListener(type, handler) {
    const list = this._listeners.get(type) || [];
    const index = list.indexOf(handler);
    if (index >= 0) list.splice(index, 1);
  }

  fire(type, event) {
    const detail = Object.assign({ type, target: this, preventDefault() {} }, event || {});
    for (const handler of (this._listeners.get(type) || []).slice()) handler(detail);
  }

  /** 文档序下标，给假布局用。每次现算——渲染会重建子树，缓存会过期。 */
  preOrderIndex(target) {
    let index = 0;
    let found = -1;
    const walk = node => {
      if (found >= 0) return;
      if (node === target) { found = index; return; }
      for (const child of node.childNodes) {
        if (child.nodeType !== 1) continue;
        index += 1;
        walk(child);
      }
    };
    walk(this);
    return found < 0 ? 0 : found;
  }
}

const ownerDocuments = new WeakMap();

function parseFragment(html) {
  const fragment = new DocumentFragment();
  const stack = [fragment];
  let last = 0;
  let m;
  TAG_RE.lastIndex = 0;
  while ((m = TAG_RE.exec(html))) {
    if (m.index > last) {
      const text = decode(html.slice(last, m.index));
      if (text) stack[stack.length - 1].appendChild(new TextNode(text));
    }
    last = TAG_RE.lastIndex;
    const raw = m[0];
    if (raw.startsWith('<!--') || raw.startsWith('<![') || /^<!doctype/i.test(raw)) continue;
    if (m[1]) {
      const name = m[1].toUpperCase();
      for (let i = stack.length - 1; i > 0; i -= 1) {
        if (stack[i].tagName === name) { stack.length = i; break; }
      }
      continue;
    }
    const document = stack[stack.length - 1].ownerDocument || null;
    const el = document ? document.createElement(m[2]) : new El(m[2], HTML_NS);
    for (const [key, value] of parseAttrs(m[3] || '')) el.setAttribute(key, value);
    stack[stack.length - 1].appendChild(el);
    if (VOID.has(m[2].toLowerCase()) || m[4] === '/') continue;
    stack.push(el);
  }
  if (last < html.length) {
    const text = decode(html.slice(last));
    if (text) stack[stack.length - 1].appendChild(new TextNode(text));
  }
  return fragment;
}

function parseDocument(html) {
  const document = new Doc();
  for (const child of parseFragment(html).childNodes.slice()) document.appendChild(child);
  stampOwner(document, document);
  return document;
}

function stampOwner(node, document) {
  if (node.nodeType === 1) {
    ownerDocuments.set(node, document);
    Object.defineProperty(node, 'ownerDocument', { value: document, configurable: true });
  }
  for (const child of node.childNodes || []) stampOwner(child, document);
}

/* ------------------------------------------------------------------ 跑一遍 */

/** 在垫片里跑一次 `app.js`。返回 `{document, errors, thrown}`。
 *
 *  `options.fetch` 是**给「换计算图」那一条用的**：那一下要真的发一次 POST、真的拿回一份
 *  新载荷，才有「三块屏跟着换」可验。默认那支仍然是会炸的 fetch——垫片里没有服务端，
 *  「不小心走到 fetch」当场暴露比静默挂住强，这个网不能为了一个用例拆掉。 */
export function runApp(source, html, payload, options) {
  const opt = options || {};
  const document = parseDocument(html);
  const errors = [];
  const console_ = {
    log() {},
    warn() {},
    info() {},
    // `render()` 的 catch 会 `window.console.error(error)`。**必须收下**：被吞掉的渲染
    // 异常和成功在界面上长得一模一样（都是「横幅空着、元素建出来了」）。
    error(error) { errors.push('console.error：' + ((error && error.message) || error)); },
  };
  const window = {
    __BAKED__: payload,
    console: console_,
    requestAnimationFrame(fn) { fn(); return 0; },
    addEventListener() {},
    removeEventListener() {},
  };
  const sandbox = {
    document,
    console: console_,
    location: { protocol: 'http:', href: 'http://127.0.0.1/' },
    // 唯一的分叉点是 app.js 末尾：`__BAKED__` 在就直接用。所以这里**不会** fetch；
    // 留一个会炸的实现，是为了让「不小心走到 fetch」当场暴露，而不是静默挂住。
    fetch: opt.fetch || (() => Promise.reject(new Error('烟测里没有服务端（不该走到 fetch）'))),
    setTimeout,
    clearTimeout,
    queueMicrotask,
  };
  sandbox.window = window;
  // `window` 上的东西在浏览器里同时是全局的：`requestAnimationFrame` 是裸写的吗？
  // app.js 一律写 `window.requestAnimationFrame`，所以不用铺到顶层。
  const context = vm.createContext(sandbox);
  let thrown = null;
  try {
    vm.runInContext(source, context, { filename: 'app.js' });
  } catch (error) {
    thrown = error;
  }
  return { document, errors, thrown };
}

/* ------------------------------------------------------------------ 断言 */

export function collect(root) {
  const out = [];
  const walk = node => {
    if (node.nodeType === 1) out.push(node);
    for (const child of node.childNodes || []) walk(child);
  };
  walk(root);
  return out;
}

/** 扫整棵子树：任何属性值或文字里出现 `NaN` 就是失败。
 *  这是**通用网**——不针对某个坐标写断言，将来任何一处算错都会落在这里。 */
export function scanNaN(document) {
  const bad = [];
  const walk = node => {
    if (node.nodeType === 3) {
      if (/NaN/.test(node.nodeValue)) bad.push('<text> ' + JSON.stringify(node.nodeValue.slice(0, 60)));
      return;
    }
    if (node.nodeType === 1) {
      for (const [key, value] of node._attrs) {
        if (/NaN/.test(value)) bad.push(node.tagName.toLowerCase() + '[' + key + '=' + value + ']');
      }
    }
    for (const child of node.childNodes || []) walk(child);
  };
  walk(document);
  return bad;
}

/** 一块屏里「真的被画出来的东西」：父节点不是这块屏自己（排除空容器），
 *  也不在 `.pane-head` 里（标题永远在，不能拿它冒充内容）。 */
function paintedContent(pane) {
  const out = [];
  const walk = node => {
    for (const child of node.childNodes || []) {
      if (child.nodeType !== 1) continue;
      if (hasClass(child, 'pane-head')) continue;
      if (child.parentNode !== pane) out.push(child);
      walk(child);
    }
  };
  walk(pane);
  return out;
}

class Report {
  constructor() {
    this.rows = [];
  }

  check(label, ok, detail) {
    this.rows.push({ label, ok: !!ok, detail: detail == null ? '' : String(detail) });
    return !!ok;
  }

  /** **没跑**，不是**跑过了**。记一行但打上 ``skip``，调用方把它单列出来。
   *  让跳过冒充通过，正是这一轮反复在防的那件事（``check_js`` 里没有 node 时同理：
   *  返回一句话挂在总结行上，不记成通过项）。 */
  skip(label, detail) {
    this.rows.push({ label, ok: true, skip: true, detail: detail == null ? '' : String(detail) });
  }
}

function caseChecks(report, name, document, payload, errors, thrown) {
  const tag = name + ' · ';
  if (!report.check(tag + 'app.js 跑到底不抛异常', !thrown, thrown ? thrown.stack || String(thrown) : '')) {
    return;
  }

  // 横幅是 `render()` try/catch 的出口，也是字面意义上的「有话说」。
  const banner = document.getElementById('banner');
  report.check(tag + '横幅是空的（没有渲染异常、没有对账失败）',
    !!banner && banner.textContent === '' && !banner.dataset.tag,
    banner ? ('tag=' + (banner.dataset.tag || '无') + ' 文字=' + JSON.stringify(banner.textContent)) : '找不到 #banner');
  report.check(tag + '没有 console.error', errors.length === 0, errors.join('　'));

  // 五块屏：每块都要有画出来的东西。**空 `<svg>` 与空 `#params-body` 一个都过不了。**
  // 没有解的时候时间线那一屏**也**不是空的——`#vram-svg` 的刻度与容量上限线照画
  // （实测：0 个 state 时它仍有 14 个子节点）。所以这里照旧一律要求非空；
  // 曾经想「从载荷推出哪几屏该是空的」，那是**按推理改的、不是量出来的**，量出来是反的。
  const panes = document.querySelectorAll('.pane');
  const empty = panes.filter(pane => paintedContent(pane).length === 0).map(pane => pane.dataset.pane);
  report.check(tag + '五块屏都画出了内容（没有空盒子）', panes.length === 5 && empty.length === 0,
    '屏数 ' + panes.length + (empty.length ? '，空的是：' + empty.join('、') : ''));

  // 跳转条 ↔ 屏：一一对应，副标题非空，`aria-current` 恰好一个。
  const tabs = document.querySelectorAll('.tab');
  const tabNames = tabs.map(tab => tab.dataset.tab);
  const paneNames = panes.map(pane => pane.dataset.pane);
  const same = tabNames.length === paneNames.length &&
    tabNames.every(item => paneNames.indexOf(item) >= 0);
  report.check(tag + '跳转条与屏一一对应', same,
    'tab=' + tabNames.join(',') + ' pane=' + paneNames.join(','));
  report.check(tag + '每个标签都有一行「属于哪一侧」',
    tabs.length > 0 && tabs.every(tab => {
      const scope = tab.querySelector('.tab-scope');
      return scope && scope.textContent.trim().length > 0;
    }));
  const current = tabs.filter(tab => tab.getAttribute('aria-current') === 'true');
  report.check(tag + '恰好一个标签是当前屏', current.length === 1,
    current.map(tab => tab.dataset.tab).join(','));

  // ---- 顶栏「换计算图」。这一格由**服务端**给名单（`payload.scenarios`），前端只负责把它
  // 如实画出来——所以这里验的是「画出来的和载荷里的一致」，不是「有几个选项」。
  // 载荷里那份清单与目录里真实文件的对应关系由 `viewer.py` 的 C 组管。
  const catalog = payload.scenarios || {};
  const items = catalog.items || [];
  const wrap = document.getElementById('scenario-pick-wrap');
  const pick = document.getElementById('scenario-pick');
  if (report.check(tag + '顶栏有换计算图的选择器', !!wrap && !!pick,
    wrap ? '' : '找不到 #scenario-pick-wrap / #scenario-pick')) {
    const options = pick.querySelectorAll('option');
    const values = options.map(option => option.value);
    report.check(tag + '选择器的选项 == 载荷里的名单（顺序也一样）',
      values.join('|') === items.map(item => item.name).join('|'),
      'DOM=' + values.join('|') + ' 载荷=' + items.map(item => item.name).join('|'));
    // `<select>` 的 value 不在选项里时，浏览器会显示**第一个**选项——也就是另一张图的名字。
    // 所以「当前这一份在选项里」不是完整性检查，是「顶栏有没有撒谎」。载荷里压根没有名字
    // （内存里构造的那一份）时，正确的显示是**空着**，不是退而显示第一个选项。
    const wantValue = catalog.current == null ? '' : catalog.current;
    report.check(tag + '选择器显示的正是屏上这一份',
      pick.value === wantValue && (wantValue === '' || values.indexOf(wantValue) >= 0),
      'value=' + JSON.stringify(pick.value) + ' current=' + JSON.stringify(catalog.current));
    // 只有一份时没有可换的：那一格收起来，而不是给一个只有一个选项的下拉框。
    report.check(tag + '只有一份时选择器收起来',
      wrap.hidden === (items.length < 2),
      'hidden=' + wrap.hidden + ' n=' + items.length);
    // 快照是死拷贝：换图要发 POST，而那边没有 Python。禁用它，且**说清为什么**——
    // 点了没反应是这类工具最容易被当成「坏了」的失败方式。
    report.check(tag + '快照模式选择器禁用且带一句原因',
      pick.disabled === (payload.mode !== 'live') &&
      (payload.mode === 'live' || String(pick.title).length > 0),
      'disabled=' + pick.disabled + ' title=' + JSON.stringify(pick.title));
  }
  // 参数表的禁用状态同一条规则。原先七份用例全是 live，这一支一次都没被验过。
  const knobs = document.getElementById('params-body').querySelectorAll('input');
  report.check(tag + '参数表的输入框禁用状态 == 是不是快照',
    knobs.length > 0 && knobs.every(input => input.disabled === (payload.mode !== 'live')),
    knobs.filter(input => input.disabled !== (payload.mode !== 'live'))
      .map(input => input.id || input.tagName).join('、') + '（共 ' + knobs.length + ' 个）');

  // 通用网。
  const bad = scanNaN(document);
  report.check(tag + '整棵文档树里没有一个 NaN', bad.length === 0,
    bad.length ? bad.slice(0, 6).join('　') + (bad.length > 6 ? '　…共 ' + bad.length + ' 处' : '') : '');

  // 计数**卡相等**，不卡「大于零」——后者在有 bug 的版本里照样过。
  const graph = payload.graph;
  const graphSvg = document.getElementById('graph-svg');
  const svgPaths = graphSvg.querySelectorAll('path');
  const edges = svgPaths.filter(path => path.getAttribute('marker-end') != null);
  report.check(tag + '计算图的边数 == graph.links.length', edges.length === graph.links.length,
    edges.length + ' vs ' + graph.links.length);
  const boxes = graphSvg.querySelectorAll('rect');
  report.check(tag + '计算图的方块数 == 张量 + 算子',
    boxes.length === graph.tensors.length + graph.operations.length,
    boxes.length + ' vs ' + (graph.tensors.length + graph.operations.length));

  // **方块之间不许互相盖住。** 计数相等只说「都画了」，不说「都看得见」——两个方块落在
  // 同一个 x/y 上时，计数、边数、扫 NaN 全是绿的，只有人眼看得出来。用户就是这么发现它的。
  // 坐标一律**从渲染出来的属性读**，不按布局公式重算：重算出来的是「我以为画在哪」，
  // 不是「画出来的在哪」，这一轮已经吃过一次这种亏（扫 NaN 那条网的由来）。
  const nodes = graphSvg.children
    .filter(child => child.tagName === 'g' && child.querySelectorAll('rect').length > 0)
    .map(child => {
      const rect = child.querySelectorAll('rect')[0];
      const title = child.querySelectorAll('title')[0];
      return {
        id: title ? String(title.textContent).split('\n')[0] : '（无 title）',
        x: Number(rect.getAttribute('x')), y: Number(rect.getAttribute('y')),
        w: Number(rect.getAttribute('width')), h: Number(rect.getAttribute('height')),
        alpha: rect.getAttribute('fill-opacity'),
      };
    });
  report.check(tag + '计算图里每个方块都读到了坐标（title 是认领方块的唯一线索）',
    nodes.length === graph.tensors.length + graph.operations.length,
    nodes.length + ' vs ' + (graph.tensors.length + graph.operations.length));
  const overlaps = [];
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i], b = nodes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 0 && dy > 0) {
        overlaps.push({ a: a.id, b: b.id, text: a.id + ' × ' + b.id + '（重叠 ' + dx + '×' + dy + '）' });
      }
    }
  }
  // **还是按「一个都不许叠」验**，虽然用户明确说「张量间可以有重叠」。两句话不冲突：
  // 那是给排布的**许可**，不是要求，而这一版排布在同列里靠 `take` 逐行占位，结构上就到不了
  // 重叠——实测七份用例 0 对。所以这里验的是**当前拿到的更强性质**，哪天要拿重叠换紧凑
  // （用户允许），把断言换成下面这行即可，别把这条删掉：
  //     const hard = overlaps.filter(pair => opIds[pair.a] || opIds[pair.b]);
  // 算子方块跟谁叠都不行：算子是这一屏的主语，被压住的算子等于这一格没画。
  const opIds = {};
  for (const op of graph.operations) opIds[op.id] = true;
  const hard = overlaps.filter(pair => opIds[pair.a] || opIds[pair.b]);
  report.check(tag + '计算图的方块互不遮盖（张量之间虽然允许重叠，这一版一个都没有）',
    overlaps.length === 0,
    overlaps.length
      ? '重叠 ' + overlaps.length + ' 对（其中牵涉算子的 ' + hard.length + ' 对）：' +
        overlaps.slice(0, 3).map(pair => pair.text).join('　') +
        (overlaps.length > 3 ? '　…共 ' + overlaps.length + ' 对' : '')
      : '');

  // **每条依赖边都必须指向右边。** 横轴是拓扑层，父在左、子在右，所以「谁依赖谁」在图上就是
  // 「谁在谁左边」。这条性质与布局公式无关，把任何一条边画反都一定违反它——是**通用网**。
  //
  // 它是被真 bug 逼出来的：开局就在显存里的输入张量（`inp_embd`，`initial_locations: ["vram"]`
  // 所以**没有** `COPY_H2D`）曾经落进「没定位」的兜底分支被丢到**最右边**，
  // 于是 `x → c1` 这条边从右往左画，而计数卡相等、边数、扫 NaN 全是绿的。
  //
  // 改成按层排之后，那条具体的错**结构上不可能**再发生（张量的列只由「谁吃它」决定，与它在
  // 动作表里上没上过场无关），但这条网留着：它盯的是**性质**，不是那个已修的实现。
  const at = {};
  for (const node of nodes) if (at[node.id] == null) at[node.id] = node.x;
  const flat = [], reversed = [];
  for (const link of graph.links) {
    const a = at[link.from], b = at[link.to];
    if (a == null || b == null) continue;          // 认不出端点的边已经由上面那条计数网把关
    if (a === b) flat.push(link.from + ' → ' + link.to);
    else if (a > b) reversed.push(link.from + ' → ' + link.to);
  }
  report.check(tag + '计算图的每条依赖边都指向右边（横轴是拓扑层）',
    flat.length === 0 && reversed.length === 0,
    (reversed.length ? '画反了：' + reversed.join('、') : '') +
    (reversed.length && flat.length ? '　' : '') +
    (flat.length ? '两端同列：' + flat.join('、') : ''));

  // 边的**端点几何**（`M x1 y1 C … x2 y2`）从画出来的 `d` 里读，下面两条网都用它。
  // 认领方式是「起点贴在来源方块的右边缘、终点贴在目标方块的左边缘，**且两端都落在方块的
  // 竖直范围内**」，不是按序号对齐——序号对齐是**我以为的**对应关系，而这里吃的就是「以为」
  // 的亏（扫 NaN 那条网的由来）。
  //
  // 竖直范围这一条不是装饰：同一列不同行的两个方块**左边缘 x 相同**，只比 x 的话
  // `W1 → c1` 会把 `x → c1` 那条路径认成自己的（这张网第一版就是这么把自己报红的，
  // 报的「两条边起点重合」其实是它认错了边）。再加一条**一次性认领**：同一条路径不会被
  // 两条 link 抢走，这样「两条边完全同源同目标」时也各拿各的那条。
  const geo = edges
    .map(path => /^M ([+-]?[\d.]+) ([+-]?[\d.]+) C ([+-]?[\d.]+) ([+-]?[\d.]+), ([+-]?[\d.]+) ([+-]?[\d.]+), ([+-]?[\d.]+) ([+-]?[\d.]+)$/
      .exec(path.getAttribute('d')))
    .filter(Boolean)
    .map(m => m.slice(1).map(Number));
  const claimed = new Set();
  const geom = graph.links.map(link => {
    const from = nodes.filter(node => node.id === link.from)[0];
    const to = nodes.filter(node => node.id === link.to)[0];
    if (!from || !to) return null;
    for (const n of geo) {
      if (claimed.has(n)) continue;
      if (n[0] !== from.x + from.w || n[6] !== to.x) continue;
      if (n[1] < from.y || n[1] > from.y + from.h) continue;
      if (n[7] < to.y || n[7] > to.y + to.h) continue;
      claimed.add(n);
      return n;
    }
    return null;
  });
  // 认不出来的边**不能静默跳过**：跳过的边越多，下面两张网覆盖得越少，而它们会照样报绿。
  const unclaimed = graph.links.filter((link, i) => geom[i] == null);
  report.check(tag + '计算图的每条边都能从渲染结果里认领回来（认不出就验不了边）',
    unclaimed.length === 0,
    unclaimed.length
      ? '认不出：' + unclaimed.slice(0, 4).map(l => l.from + ' → ' + l.to).join('、')
      : '');

  // **每条依赖边都不许往上画。** 上面那条网管左右（横轴是拓扑层），这条管**上下**：
  // 纵轴是先后——同一列里先加载的在上，算子的行取它父张量里**最靠下**那一行，产出的张量
  // 与算子同行。三条合起来，任何一条依赖边的终点都不高于起点。
  //
  // 用户的读法是「拓扑图尽量从左上到右下」；这一版把它**收紧成一条能验的性质**：不但要
  // 朝右，还要不朝上。判据取**方块的上沿**（`to.y - from.y >= 0`），不取箭头端点的 y：
  // 一个方块上挂多条边时锚点要摊开（见 `fanOut`），摊开幅度在一个方块高（40px）以内，
  // 而两个行之间差 62px，所以**行级的违规必然 ≥ 22px**、摊开的抖动必然 < 20px，取方块上沿
  // 就干净地把两者分开了，也不用往网里塞一个魔法容差。
  //
  // 它是被真 bug 逼出来的：行号曾经是 `rowCursor[rank]++`，而**秩只是行内计数器、不带偏移**，
  // 于是每个秩的第一个张量都落在第 0 行，`y`（深度 4）和 `W1`（深度 0）挤在同一行，实测五份图
  // 里有 **15 条边在往上画**（`residual` 一份 5 条），而计数、边数、扫 NaN 全绿。
  //
  // 证伪这个坑的两种写法都会变红：把算子的行改成取父张量里**最靠上**那一行（`min`），
  // `residual` 的 `add` 会落到第 0 行而它的父在第 1 行；或者退回按秩计行。
  const upward = [];
  graph.links.forEach((link, i) => {
    const from = nodes.filter(node => node.id === link.from)[0];
    const to = nodes.filter(node => node.id === link.to)[0];
    if (!from || !to) return;                      // 认不出端点的边已由上面的认领网把关
    if (to.y < from.y) {
      upward.push(link.from + ' → ' + link.to + '（方块上沿 y ' + from.y + ' → ' + to.y + '）');
    }
  });
  report.check(tag + '计算图的每条依赖边都不往上画（纵轴是先后：算子在它的父张量之下）',
    upward.length === 0,
    upward.length
      ? '往上画了 ' + upward.length + ' 条：' + upward.slice(0, 4).join('、') +
        (upward.length > 4 ? '　…共 ' + upward.length + ' 条' : '')
      : '');

  // **一个算子的父张量都贴在它左边一列。** 用户的原话是「一个算子的所有父张量应该是在 y 轴上
  // 排布，而不是在 x 轴上前后排布」——在图上就是这一列。判据取**画出来的 x**：
  // 父张量的右沿 + 一格列宽 == 算子的左沿（`GRAPH.colW` 从画出来的坐标推，不写死 196）。
  //
  // 例外是**被多个算子共用**的父张量：`residual` 的残差 `x` 同时喂 `c1`（秩 1）和 `add`（秩 7），
  // 它只能贴在**最早**那个左边，另一条边跨列——分层画法的固有限制，除非把同一个张量画两遍。
  // 例外**显式计数并报出来**（这条本来想省略，省了就变成「静默放行」）。
  const parentLinks = graph.links.filter(link => link.kind !== 'output');
  const consumers = {};
  for (const link of parentLinks) consumers[link.from] = (consumers[link.from] || 0) + 1;
  const colW = (function () {
    const xs = [...new Set(nodes.map(node => node.x))].sort((a, b) => a - b);
    return xs.length > 1 ? xs[1] - xs[0] : null;
  })();
  const farParents = [], adrift = [];
  for (const link of parentLinks) {
    const from = nodes.filter(node => node.id === link.from)[0];
    const to = nodes.filter(node => node.id === link.to)[0];
    if (!from || !to || colW == null) continue;
    const shared = consumers[link.from] > 1;
    if (from.x + colW === to.x) continue;
    if (shared) farParents.push(link.from);
    else adrift.push(link.from + ' → ' + link.to + '（x ' + from.x + ' → ' + to.x + '，列宽 ' + colW + '）');
  }
  report.check(tag + '计算图的每个算子的父张量都贴在它左边一列（共用的除外）',
    adrift.length === 0,
    adrift.length
      ? adrift.slice(0, 4).join('　')
      // 共用父张量是**固有例外**（一张张量被两个算子吃，它只能贴在最早那个消费者左边），
      // 列出来是为了让人一眼看到本图有几处；一处都没有时说「没有」，别留一句半截话。
      : (farParents.length
        ? '（共用的父张量 ' + [...new Set(farParents)].join('、') + ' 贴的是最早那个消费者，属例外）'
        : '（没有共用父张量，全部贴合）'));

  // **算子与它产出的张量同一行。** 用户原话：「算子后的张量和算子在竖轴上对齐」。
  const sameRow = [];
  for (const link of graph.links) {
    if (link.kind !== 'output') continue;
    const from = nodes.filter(node => node.id === link.from)[0];
    const to = nodes.filter(node => node.id === link.to)[0];
    if (!from || !to) continue;
    if (to.y !== from.y) sameRow.push(link.from + ' → ' + link.to + '（y ' + from.y + ' vs ' + to.y + '）');
  }
  report.check(tag + '计算图的算子的输出张量与它同一行（第 3 条规则）',
    sameRow.length === 0,
    sameRow.length ? sameRow.slice(0, 4).join('、') : '');

  // **同一列里按先后自上而下。** 第 2 条规则：先加载的在左上。这是「先后」这条轴唯一的
  // 直接检验——上面那条只管「不往上」，把一列整个倒过来它也是绿的。
  // 「先后」从**载荷的动作表**现推（第 i 步出现的实体），不读画出来的东西：图上没有别的地方
  // 写着谁先谁后，能读的只有这个，而它正是排布该服从的东西。开局就在显存里的记 −1（最上）。
  const actions = (payload.solution && payload.solution.actions) || [];
  const when = {};
  actions.forEach((action, index) => {
    const target = action.kind === 'COPY_H2D' ? action.tensor_id
      : action.kind === 'COMPUTE' ? action.operation_id : null;
    if (target != null && when[target] == null) when[target] = index;
  });
  const producedBy = {};
  for (const link of graph.links) if (link.kind === 'output') producedBy[link.to] = link.from;
  const firstSeen = {};
  for (const entity of graph.tensors.concat(graph.operations)) {
    if (when[entity.id] != null) firstSeen[entity.id] = when[entity.id];
    else if (producedBy[entity.id] != null && when[producedBy[entity.id]] != null) {
      firstSeen[entity.id] = when[producedBy[entity.id]];
    } else if ((entity.initial_locations || []) && !entity.is_weight) firstSeen[entity.id] = -1;
    else firstSeen[entity.id] = 1000;
  }
  // 比的是**被加载进来的那些张量**（每张图的输入与权重），不比算子和激活：算子的行由
  // 「父张量里最靠下那一行」定、激活的行由产出它的算子定（第 3 条规则），它们不服从加载先后，
  // 服从的是结构。用户那句话说的也正是输入——「一个算子的所有父张量……先加载的在左上」。
  // 不筛掉的话，无解那两份用例会假红：那里没有「第几步」，一切退回秩序，而激活 `h` 的行
  // 是被它的产出算子顶下去的（`h` 的秩比 `W2` 小，却排在 `W2` 下面）。
  const loaded = node => !opIds[node.id] && producedBy[node.id] == null;
  const byX = {};
  for (const node of nodes) (byX[node.x] = byX[node.x] || []).push(node);
  const misordered = [];
  for (const x of Object.keys(byX)) {
    const column = byX[x].filter(loaded).sort((p, q) => p.y - q.y);
    for (let i = 1; i < column.length; i += 1) {
      const before = column[i - 1], after = column[i];
      if (firstSeen[before.id] > firstSeen[after.id]) {
        misordered.push('第 ' + x + ' 列的 ' + after.id + '（第 ' + firstSeen[after.id] + ' 步）排在 ' +
          before.id + '（第 ' + firstSeen[before.id] + ' 步）下面');
      }
    }
  }
  report.check(tag + '计算图同一列里的输入/权重按加载先后自上而下（先加载的在左上）',
    misordered.length === 0,
    misordered.slice(0, 4).join('、'));

  // **紧凑。** 用户的原话里有「保证拓扑图适当紧凑」，而「太难看了」那次正是**看见**了不紧凑
  // （`residual` 602px 高只装 11 个方块，方块占画布 5%）。这条把「好看」折成一个能验的数：
  // 方块面积之和 / 画布面积。阈值取一个远低于当前值、又远高于旧值的数——它不是调优目标，
  // 是**下限**（防止哪天又滑回按层各占一行那种稀疏排布）。
  const boxArea = (graph.tensors.length + graph.operations.length) * Number(boxes[0].getAttribute('width')) *
    Number(boxes[0].getAttribute('height'));
  const view = String(graphSvg.getAttribute('viewBox')).split(/\s+/).map(Number);
  const density = boxArea / (view[2] * view[3]);
  report.check(tag + '计算图的方块占画布面积 ≥ 18%（适当紧凑，别摊成稀疏阶梯）',
    density >= 0.18, '占 ' + (density * 100).toFixed(1) + '%　（画布 ' + view[2] + '×' + view[3] + '）');

  // **被边压住的方块必须在填充上半透明**，否则那条边就白画了。
  // 边先画、方块后画，所以跨列的边会被沿途的方块盖住——实测五份图 38 条边里只剩 1 条
  // （`residual` 那条跨 7 列的残差边 `x → add`；按执行顺序排 x 的那一版是 15 条）。用户报的就是这个：
  // 「拓扑图的线和方框会相互遮挡」。这里沿三次贝塞尔采样，算出**哪些方块压住了边**，
  // 然后要求这些方块的 `fill-opacity` < 1。几何同样是从渲染出来的属性读的。
  //
  // 断言的是**这条性质本身**，不是「某处写了 0.5」：改 α 不会误报，把填充改回不透明才报。
  const buried = new Set();
  for (const path of edges) {
    const d = /^M ([-\d.]+) ([-\d.]+) C ([-\d.]+) ([-\d.]+), ([-\d.]+) ([-\d.]+), ([-\d.]+) ([-\d.]+)$/
      .exec(path.getAttribute('d'));
    if (!d) continue;
    const n = d.slice(1).map(Number);
    const bez = (p0, p1, p2, p3, t) => {
      const u = 1 - t;
      return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
    };
    for (let i = 0; i < 120; i += 1) {
      const t = (i + 0.5) / 120;                 // 端点不算：那里本来就贴着方块边
      const px = bez(n[0], n[2], n[4], n[6], t), py = bez(n[1], n[3], n[5], n[7], t);
      for (const node of nodes) {
        if (px > node.x + 1 && px < node.x + node.w - 1 &&
            py > node.y + 1 && py < node.y + node.h - 1) buried.add(node.id);
      }
    }
  }
  const opaque = [...buried].filter(id => {
    const node = nodes.filter(item => item.id === id)[0];
    return !(Number(node.alpha) < 1);
  });
  report.check(tag + '压住了连线的方块都是半透明的（否则那条线白画）',
    opaque.length === 0,
    '被压住的方块：' + (buried.size ? [...buried].join('、') : '（没有）') +
    (opaque.length ? '　其中不透明的：' + opaque.join('、') : ''));

  // 「时长来源」那一屏的**唯一非空转断言**：有分歧就必须说出来。
  // `matvec` 就是这一条——推导 2.5 ms、声明 3 ms，于是 makespan 从存档里那个 5 ms 变成
  // 4.5 ms（`M0_VERIFICATION.md:138`）。不一致却不说，屏上就是一个和存档对不上的数字
  // 而没有任何解释；反过来，没有分歧却弹一张卡，是另一种骗人。
  const disagreements = (payload.cost_source || {}).disagreements || [];
  const sourceBox = document.querySelector('.pane[data-pane="source"]');
  const heads = collect(sourceBox).filter(el => el.tagName === 'H3' &&
    el.textContent.indexOf('推导值与声明值不一致') >= 0);
  report.check(tag + '来源面板：有分歧就明说、没分歧就不说',
    heads.length === (disagreements.length ? 1 : 0),
    '分歧 ' + disagreements.length + ' 行，卡片 ' + heads.length + ' 张');
  if (disagreements.length && heads.length === 1) {
    // `<h3>` → `.gate-head` → 卡片。行数卡相等，不卡「大于零」。
    const rows = collect(heads[0].parentNode.parentNode)
      .filter(el => el.tagName === 'TR').length - 1;
    report.check(tag + '分歧表的行数 == disagreements.length', rows === disagreements.length,
      rows + ' vs ' + disagreements.length);
  }

  // 垫片不实现后代选择器（`#tree-svg .tree-hit`）——**不实现就抛**，不静默给空集，
  // 所以这里改成先取 svg 再在它里面找。
  const treeSvg = document.getElementById('tree-svg');
  const hits = treeSvg.querySelectorAll('.tree-hit').filter(hit => hit.dataset.id != null);
  report.check(tag + '搜索树的命中圈数 == explore.nodes.length', hits.length === payload.explore.nodes.length,
    hits.length + ' vs ' + payload.explore.nodes.length);

  // 甘特与状态时间线也**卡相等**。`> 0` 在「画了一根柱子就停」的版本里照样过。
  // 首帧是回放模式，所以 `currentEvents()` 就是 `payload.log.events`——正是内核给的计划，
  // 不必在垫片里重算一遍走查（那会变成自己给自己出题）。
  const bars = document.getElementById('gantt-svg').querySelectorAll('rect')
    .filter(rect => rect.hasAttribute('stroke')).length;
  const laneEvents = payload.log.events
    .filter(event => event.resource === 'h2d_copy' || event.resource === 'gpu_compute').length;
  report.check(tag + '甘特柱数 == 有 resource 的事件数', bars === laneEvents,
    bars + ' vs ' + laneEvents);
  const rows = document.getElementById('states-svg').querySelectorAll('rect').length;
  report.check(tag + '状态时间线行数 == log.states.length', rows === payload.log.states.length,
    rows + ' vs ' + payload.log.states.length);

  // ---- 时间线那三张图的**文字**。两条性质，都是用户看出来的，都不是计数能抓的。
  //
  // ① **没有一行字贴着画布边。** 整幅 SVG 的 `overflow` 是 `hidden`，贴边的字被**静默裁掉**
  // ——不报错、不留痕、扫 NaN 也扫不到。用户报的「容量 xxx 被第一张图覆盖了」就是这个：
  // 容量线画在绘图区上沿（`peak` 取的就是容量上限，所以 `vy(limit)` 正好落在上沿），标注只能
  // 放在线上方，旧基线落在 y=9，而 14px 的中文从基线往上要占约 12.3px——「容量」两个字的上半截
  // 被切了，看上去就像被上面那张图压住。判据：基线离上边 ≥ 12px、离下边 ≥ 4px（下伸部约 0.25em）。
  // 字号 14px 来自 `style.css` 的 `body { font: 14px/1.5 … }`，SVG 的 `<text>` 继承它，
  // 而 `style.css` 里没有任何规则改过 SVG 文字的字号。
  const EDGE_TOP = 12, EDGE_BOTTOM = 4;
  const pinned = [];
  for (const id of ['gantt-svg', 'vram-svg', 'states-svg']) {
    const svg = document.getElementById(id);
    const h = Number(svg.getAttribute('height'));
    for (const text of collect(svg).filter(el => el.tagName === 'text')) {
      const y = Number(text.getAttribute('y'));
      if (!(y >= EDGE_TOP && y <= h - EDGE_BOTTOM)) {
        pinned.push('#' + id + ' 的「' + String(text.textContent).slice(0, 18) + '」y=' + y + '（画布高 ' + h + '）');
      }
    }
  }
  report.check(tag + '时间线的每行字都离画布上下边够远（贴边会被静默裁掉）',
    pinned.length === 0, pinned.slice(0, 4).join('　'));

  // ② **状态时间线每一行的说明都紧挨着它自己那根条。** 用户的原话是「把字放到阶段的傍边」。
  // 旧写法把说明右对齐钉死在画布右边（`x = 892`），于是每条说明离它要说明的那个阶段有
  // 30~650px 不等的空档——图上「谁说明谁」只能靠数行。现在字跟着条走，网就盯这一条：
  // 字的起点落在条右沿之后 1~12px 内。**不许卡「某处写了 7」**：间距是可调的，紧挨着才是性质。
  //
  // 配对方式：同一行（`y` 差 11，正是基线在行内的位置）且在该条**右边**的那条文字。
  // `#i` 那个行号在条的左边（`padL - 9`），不会被认成说明。
  const statesSvg = document.getElementById('states-svg');
  const stateTexts = collect(statesSvg).filter(el => el.tagName === 'text');
  const stateBars = collect(statesSvg).filter(el => el.tagName === 'rect');
  const stateDrift = [];
  for (const bar of stateBars) {
    const barX = Number(bar.getAttribute('x'));
    const barRight = barX + Number(bar.getAttribute('width'));
    const mine = stateTexts.filter(text => Number(text.getAttribute('y')) === Number(bar.getAttribute('y')) + 11 &&
      Number(text.getAttribute('x')) > barX);
    if (mine.length !== 1) {
      stateDrift.push('第 ' + barX + ' 处那根条旁边有 ' + mine.length + ' 条字（要恰好 1 条）');
      continue;
    }
    const gap = Number(mine[0].getAttribute('x')) - barRight;
    if (!(gap > 0 && gap <= 12)) {
      stateDrift.push('「' + String(mine[0].textContent).slice(0, 16) + '」离它那根条 ' + gap.toFixed(1) + 'px');
    }
  }
  report.check(tag + '状态时间线每行的说明都紧挨着它自己那根条（不是钉在画布右边）',
    stateDrift.length === 0,
    stateDrift.length ? stateDrift.slice(0, 4).join('　') : '条 ' + stateBars.length + ' 根，逐行都对得上');

  // ③ **状态时间线的字都在画布内。** 右列宽度现在是按内容现算的（`app.js` 的 `textWidth`），
  // 算窄了字就被裁。这里用**同一个偏大估计**再算一遍：估宽是上界，所以「放得下」是有余量的结论。
  // 两份估计各写一份是**有意的**——`app.js` 那份决定**留多宽**，这份决定**够不够**；
  // 合成一份就等于自己验自己。垫片里没有排版引擎（`getBBox()` 在 node 里不存在），也没有别的路。
  const estWidth = text => {
    let em = 0;
    for (const ch of String(text)) em += ch.codePointAt(0) >= 0x2e80 ? 1 : 0.62;
    return em * 14;
  };
  const stateW = Number(String(statesSvg.getAttribute('viewBox')).split(/\s+/)[2]);
  const spilled = [];
  for (const text of stateTexts) {
    const x = Number(text.getAttribute('x'));
    const w = estWidth(text.textContent);
    const left = text.getAttribute('text-anchor') === 'end' ? x - w : x;
    const right = text.getAttribute('text-anchor') === 'end' ? x : x + w;
    if (left < 0 || right > stateW - 2) {
      spilled.push('「' + String(text.textContent).slice(0, 16) + '」占 ' + left.toFixed(0) + '~' + right.toFixed(0) +
        '（画布 0~' + stateW + '）');
    }
  }
  report.check(tag + '状态时间线的字都在画布内（右列宽度按内容现算，算窄了会被裁）',
    spilled.length === 0, spilled.slice(0, 4).join('　'));

  const vram = document.getElementById('vram-svg');
  report.check(tag + '#vram-svg 非空', !!vram && vram.childNodes.length > 0,
    vram ? vram.childNodes.length + ' 个子节点' : '找不到元素');
  for (const id of ['bridge-body', 'params-body']) {
    const box = document.getElementById(id);
    report.check(tag + '#' + id + ' 非空', box.childNodes.length > 0,
      box.childNodes.length + ' 个子节点');
  }
  report.check(tag + '#actions 非空', document.getElementById('actions').childNodes.length > 0);
  report.check(tag + '#headline 非空', document.getElementById('headline').childNodes.length > 0);

  // 已经红了的用例不再往下走交互——再叠一层「点一下也没 NaN」只会盖住第一条真正的失败。
  if (errors.length || bad.length > 0) return;

  // ---- 交互：点搜索树上的终点，游标应当切到单步并落在那条路径上。
  // 无解的那两份载荷树上没有终点圈，这一整段不适用——**跳过，不静默算通过**。
  const goal = payload.explore.goal_id;
  if (goal == null) {
    report.skip(tag + '本用例没有终点（搜索无解），交互与焦点守护一项跳过');
    return;
  }
  const goalHit = hits.filter(hit => Number(hit.dataset.id) === goal)[0];
  if (!report.check(tag + '终点有一个可点的命中圈', !!goalHit, 'goal_id=' + goal)) return;
  const before = document.getElementById('step-label').textContent;
  goalHit.fire('click');
  const stepButton = document.querySelector('.mode[data-mode="step"]');
  report.check(tag + '点树上的节点 → 走带切到单步',
    stepButton.getAttribute('aria-pressed') === 'true',
    'aria-pressed=' + stepButton.getAttribute('aria-pressed'));
  const after = document.getElementById('step-label').textContent;
  const index = Number((/第\s*(\d+)\s*\//.exec(after) || [])[1]);
  report.check(tag + '点树上的节点 → 游标往前走（不是停在 0）',
    Number.isFinite(index) && index >= 1, before + ' → ' + after);
  report.check(tag + '点完仍然没有 NaN', scanNaN(document).length === 0);

  // ---- 焦点守卫：`renderParams()` 每帧都跑，正在填的表单不能被冲掉。
  const body = document.getElementById('params-body');
  const input = body.querySelector('[data-op]') || body.querySelector('[data-flag]');
  if (input) {
    document.activeElement = input;
    document.querySelector('.mode[data-mode="replay"]').fire('click');
    report.check(tag + '有焦点在参数表单里时，表单不被重建',
      document.getElementById('params-body').contains(input));
    document.activeElement = null;
  }

  // ---- 跳转条：点一下要 `scrollIntoView` 到对应的屏，并把 `aria-current` 挪过去。
  const sourceTab = document.querySelector('.tab[data-tab="source"]');
  const sourcePane = document.querySelector('.pane[data-pane="source"]');
  sourceTab.fire('click');
  report.check(tag + '点跳转条 → 对应屏 scrollIntoView',
    sourcePane.scrolledIntoView > 0,
    'scrolledIntoView=' + sourcePane.scrolledIntoView);
  report.check(tag + '点跳转条 → aria-current 挪过去',
    sourceTab.getAttribute('aria-current') === 'true' &&
    document.querySelector('.tab[data-tab="tree"]').getAttribute('aria-current') === 'false');
}

/* ------------------------------------------------- 换计算图：真发一次 POST */

/** 顶栏「换计算图」那一下的**真跑**：换掉整份载荷，再看三块屏有没有跟着换。
 *
 *  垫片里没有服务端，所以只有这一条配一个 fetch 桩；其余用例走的仍是那个会炸的 fetch
 *  （「不小心走到 fetch」要当场暴露，这张网不能为了一个用例拆掉）。桩里返回的是
 *  **服务端真装配出来的**那份新载荷（`viewer.py` 现 build 好，随用例递进来）——在桩里
 *  现编一份载荷就成了自己给自己出题：编得对不对没人管，而这里要验的恰恰是「服务端给的
 *  那份新载荷，前端接不接得住」。
 *
 *  它是**交互的正例**（不像通用网那样对每份载荷都成立），所以单独跑、断言单独列，
 *  而且只在带 `switch_to` 的那个用例上跑。 */
async function switchChecks(report, source, html, item) {
  const target = item.switch_to;
  if (!target) return;
  const tag = item.name + ' · 换计算图 · ';
  const calls = [];
  const fetchStub = (url, options) => {
    calls.push({ url: String(url), body: String((options || {}).body || '') });
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({ before: item.payload, after: target.payload }),
    });
  };
  const run = runApp(source, html, item.payload, { fetch: fetchStub });
  const document = run.document;
  if (!report.check(tag + 'app.js 跑到底不抛异常', !run.thrown,
      run.thrown ? run.thrown.stack || String(run.thrown) : '')) return;
  const pick = document.getElementById('scenario-pick');
  if (!report.check(tag + '选择器在文档里', !!pick)) return;

  pick.value = target.name;
  pick.fire('change');
  // 一次宏任务把整条 promise 链（`.then` → `render()`）走完，再断言。
  await new Promise(resolve => setTimeout(resolve, 0));

  let posted = null;
  try { posted = JSON.parse((calls[0] || {}).body); } catch (error) { posted = null; }
  report.check(tag + '换一下 = 一次 POST /api/params，body 里带 scenario',
    calls.length === 1 && calls[0].url === 'api/params' && !!posted && posted.scenario === target.name,
    'calls=' + calls.length + ' body=' + ((calls[0] || {}).body || '（没发）'));

  report.check(tag + '顶栏换成了新那一份',
    document.getElementById('scenario-id').textContent === target.payload.scenario_id,
    document.getElementById('scenario-id').textContent + ' vs ' + target.payload.scenario_id);

  const boxes = document.getElementById('graph-svg').querySelectorAll('rect').length;
  const want = target.payload.graph.tensors.length + target.payload.graph.operations.length;
  const was = item.payload.graph.tensors.length + item.payload.graph.operations.length;
  // 两张图的方块数本来就不同——否则下面那条断言可能是空转（换没换都相等）。
  report.check(tag + '两张图的方块数确实不同（下一条断言因此不是空转）', was !== want,
    was + ' vs ' + want);
  report.check(tag + '计算图重画成了新图的方块数', boxes === want, boxes + ' vs ' + want);
  report.check(tag + '选择器停在新那一份上（不是弹回旧的）',
    pick.value === target.payload.scenarios.current,
    pick.value + ' vs ' + target.payload.scenarios.current);

  const bad = scanNaN(document);
  report.check(tag + '换完没有 NaN、没有 console.error、横幅是空的',
    bad.length === 0 && run.errors.length === 0 &&
    document.getElementById('banner').textContent === '',
    bad.slice(0, 3).join('　') + ' ' + run.errors.join('　') + ' ' +
    JSON.stringify(document.getElementById('banner').textContent));
}

/* ------------------------------------------------------------------ 证伪 */

/** 往 `GRAPH.boxW` 里注入一个 NaN。**这是这张网唯一的存在理由**：如果没有它，
 *  「扫 NaN」这条断言可能从来没红过，谁也不知道它是否真的在扫。 */
function injectNaN(source) {
  const before = source;
  const after = source.replace('boxW: 168', 'boxW: 0 / 0');
  if (after === before) throw new Error('证伪注入点没找到：app.js 里已经没有 `boxW: 168` 了吗？');
  return after;
}

/* ------------------------------------------------------------------ 入口 */

export function parseArgv(argv) {
  const out = { falsify: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--falsify') { out.falsify = true; continue; }
    const key = argv[i].replace(/^--/, '');
    out[key] = argv[i + 1];
    i += 1;
  }
  return out;
}

function emit(row) {
  process.stdout.write(JSON.stringify(row) + '\n');
}

async function main() {
  const args = parseArgv(process.argv.slice(2));
  const source = readFileSync(args.app, 'utf8');
  const html = readFileSync(args.html, 'utf8');
  const cases = JSON.parse(readFileSync(args.payload, 'utf8')).cases;

  if (args.falsify) {
    const report = new Report();
    let caught = false;
    let note = '';
    try {
      const broken = injectNaN(source);
      const one = cases[0];
      const run = runApp(broken, html, one.payload);
      const bad = scanNaN(run.document);
      caught = bad.length > 0;
      note = bad.length ? bad.slice(0, 3).join('　') : '一个 NaN 都没扫到';
    } catch (error) {
      note = String(error.message || error);
    }
    report.check('证伪：注入一个 NaN 坐标后，「扫 NaN」这张网必须变红', caught, note);
    for (const row of report.rows) emit(row);
    emit({ done: true, total: report.rows.length, failed: report.rows.filter(row => !row.ok).length });
    process.exitCode = report.rows.every(row => row.ok) ? 0 : 1;
    return;
  }

  const report = new Report();
  for (const item of cases) {
    const run = runApp(source, html, item.payload);
    caseChecks(report, item.name, run.document, item.payload, run.errors, run.thrown);
  }
  // 换图那一条要等 promise 落地（`async`），所以放在通用用例之后单独跑：通用用例是**同步**
  // 的，混进去会让「哪一条在等」看不出来。
  for (const item of cases) {
    await switchChecks(report, source, html, item);
  }
  for (const row of report.rows) emit(row);
  // 跳过的不进 `total`——**「没跑」不能算成「跑过了」**。数量单独报出来，调用方才能把它
  // 挂在总结行上，而不是让一个少了的计数静默地看起来和没少一样。
  const billed = report.rows.filter(row => !row.skip);
  const failed = billed.filter(row => !row.ok).length;
  emit({ done: true, total: billed.length, skipped: report.rows.length - billed.length, failed });
  process.exitCode = failed ? 1 : 0;
}

// 被 `import` 时不要自己跑起来——垫片与断言工具因此可以被别的脚本借用。
// **借用的意义是：量任何东西都量的是「真渲染出来的那张图」**，而不是把布局公式抄一遍
// 再算。这一条吃过亏：自己重算出来的坐标是「我以为画在哪」。
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  // `main` 是 async（换图那条要等一次 POST 的 promise）。没有这个 catch，一次同步的
  // 异常会变成一句 unhandled rejection 而不是一行能读的错误。
  main().catch(error => {
    process.stderr.write(String((error && error.stack) || error) + '\n');
    process.exitCode = 1;
  });
}
