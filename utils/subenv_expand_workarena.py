# -*- coding: utf-8 -*-
"""
AXTree 一次性子环境扩展（更全面视野）
- 解析：仅依赖缩进与 [id]，不做角色分类
- 扩展：以目标节点为中心做“邻域（父/兄弟/子）BFS 扩展”，可控 hops 与预算
- 渲染：严格输出原始 axtree 文本风格（逐行、缩进、原 raw 行）
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Set
from collections import deque

# =========================================================
# 数据结构
# =========================================================

@dataclass(eq=False)
class AXNode:
    id: Optional[str]           # [151] 的 151；非 id 行则为 None
    role: Optional[str]         # menubar / link / StaticText / ...
    name: str                   # 引号中的 name；无则 ""
    attrs: Dict[str, str]       # 解析出的属性，如 {"orientation":"horizontal"}
    raw_line: str               # 原始行（完整保留）
    indent: int                 # 该行的缩进宽度（tab->spaces 后）
    parent: Optional["AXNode"] = None
    children: List["AXNode"] = field(default_factory=list)

    def __hash__(self) -> int:
        # 允许放入 set / 作为 dict key（基于对象身份）
        return id(self)

# =========================================================
# 解析（仅依赖缩进与 [id]）
# =========================================================

LINE_RE = re.compile(
    r"""
    ^(?P<indent>\s*)
    (?:
        \[(?P<id>[^\]]+)\]\s+(?P<role_id>\w+)\s*(?P<rest_id>.*)   # 有 [id] 的行（支持 a311 / a / 46 等）
        |
        (?P<role_no>\w+)\s*(?P<rest_no>.*)                        # 无 [id] 的行
    )
    \s*$
    """,
    re.VERBOSE,
)
NAME_RE = re.compile(r"'([^']*)'")
ATTR_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*'([^']*)'|([A-Za-z_]\w*)\s*=\s*([^,\s]+)")

def _indent_width(s: str, tabsize: int = 4) -> int:
    """将制表符换算为空格宽度（默认 4）并计算缩进列宽。"""
    col = 0
    for ch in s:
        if ch == '\t':
            col += tabsize - (col % tabsize)
        else:
            col += 1
    return col

def _parse_name(rest: str) -> Tuple[str, str]:
    """从 rest 中抽取第一个引号 name，返回 (name, 剩余串)"""
    if not rest:
        return "", ""
    m = NAME_RE.search(rest)
    if not m:
        return "", rest
    name = m.group(1)
    start, end = m.span()
    remaining = (rest[:start] + rest[end:]).strip().lstrip(",").strip()
    return name, remaining

def _parse_attrs(rest: str) -> Dict[str, str]:
    """从剩余串里解析 key='value' 或 key=value（到逗号分隔）"""
    attrs: Dict[str, str] = {}
    for m in ATTR_RE.finditer(rest or ""):
        if m.group(1) is not None:  # key='value'
            key, val = m.group(1), m.group(2)
        else:                        # key=value（无引号）
            key, val = m.group(3), m.group(4)
        attrs[key] = val
    return attrs

def parse_axtree(text: str, tabsize: int = 4) -> Tuple[AXNode, Dict[str, AXNode]]:
    """解析 AXTree 文本为树结构。返回：(root, id_map)。root 为统一虚拟根 __ROOT__。"""
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
    id_map: Dict[str, AXNode] = {}
    stack: List[AXNode] = []
    root: Optional[AXNode] = None

    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            indent = _indent_width(ln[:len(ln) - len(ln.lstrip())], tabsize)
            node = AXNode(id=None, role=None, name="", attrs={}, raw_line=ln, indent=indent)
        else:
            indent = _indent_width(m.group("indent"), tabsize)
            if m.group("id"):  # [id] 行
                nid = m.group("id")
                role = m.group("role_id")
                rest = (m.group("rest_id") or "").strip()
            else:              # 非 [id] 行
                nid = None
                role = m.group("role_no")
                rest = (m.group("rest_no") or "").strip()
            name, rem = _parse_name(rest)
            attrs = _parse_attrs(rem)
            node = AXNode(id=nid, role=role, name=name, attrs=attrs, raw_line=ln, indent=indent)
            if nid:
                id_map[nid] = node

        # 用缩进构建父子关系（经典栈算法）
        while stack and stack[-1].indent >= indent:
            stack.pop()

        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            if root is None:
                root = node
            else:
                if root.parent is None and root.role != "__ROOT__":
                    fake = AXNode(id=None, role="__ROOT__", name="", attrs={}, raw_line="", indent=-1)
                    fake.children.append(root); root.parent = fake
                    root = fake
                node.parent = root
                root.children.append(node)

        stack.append(node)

    # 统一返回虚拟根
    if root is None:
        root = AXNode(id=None, role="__ROOT__", name="", attrs={}, raw_line="", indent=-1)
    if root.role != "__ROOT__":
        fake = AXNode(id=None, role="__ROOT__", name="", attrs={}, raw_line="", indent=-1)
        root.parent = fake
        fake.children.append(root)
        root = fake

    merge_textual_children_general(root)
    return root, id_map

# =========================================================
# 实用函数
# =========================================================

def extract_ids(text: str) -> List[str]:
    """只从 AXTree 节点行首抽取 [id]，按出现顺序去重。"""
    ids, seen = [], set()
    for ln in text.splitlines():
        m = re.search(r"^\s*\[([^\]]+)\]\s+", ln)   # 只匹配行首节点
        if m:
            x = m.group(1)
            if x not in seen:
                seen.add(x)
                ids.append(x)
    return ids

def path_to_root_inclusive(n: AXNode) -> List[AXNode]:
    """包含自身的路径：n, parent, parent^2, ..., root"""
    p = [n]
    cur = n.parent
    while cur:
        p.append(cur)
        cur = cur.parent
    return p

def choose_lca(nodes: List[AXNode]) -> AXNode:
    """最近公共祖先（路径含自身）。若无交集则回退到首节点。"""
    paths = [path_to_root_inclusive(t) for t in nodes]
    common = set(paths[0])
    for p in paths[1:]:
        common &= set(p)
    if not common:
        return nodes[0]
    for n in paths[0]:
        if n in common:
            return n
    return nodes[0]

def ascend(n: AXNode, up_levels: int) -> AXNode:
    """向上提升若干层，直到根或达到 up_levels"""
    cur = n
    while up_levels > 0 and cur.parent:
        cur = cur.parent
        up_levels -= 1
    return cur

# =========================================================
# 邻域（ring / hop）扩展：更全面的可见范围
# =========================================================

def neighborhood_expand(
    targets: List[AXNode],
    hops: int = 3,
    max_nodes: int = 1200,
    max_children_per_node: int = 50,
) -> Set[AXNode]:
    """
    从 targets 出发，按“父/兄弟/子”为边做 BFS 扩展：
      - hop=0: 仅 targets
      - hop=1: + 父、兄弟、子（子节点限制）
      - hop=2: 基于 hop=1 的新边界再扩……
    受 max_nodes 总预算限制；每个节点向下扩的子节点数受 max_children_per_node 限制。
    """
    kept: Set[AXNode] = set()
    q = deque()

    # 初始化
    for t in targets:
        if len(kept) >= max_nodes: break
        kept.add(t)
        q.append((t, 0))

    def add_node(n: AXNode) -> bool:
        if n not in kept and len(kept) < max_nodes:
            kept.add(n)
            return True
        return False

    while q:
        node, h = q.popleft()
        if h >= hops:
            continue

        # 1) 父
        if node.parent and add_node(node.parent):
            q.append((node.parent, h + 1))

        # 2) 兄弟（同父所有兄弟纳入）
        if node.parent:
            for s in node.parent.children:
                if s is node:
                    continue
                if add_node(s):
                    q.append((s, h + 1))

        # 3) 子（限制每个节点的向下展开数量）
        if node.children:
            cnt = 0
            for c in node.children:
                if cnt >= max_children_per_node:
                    break
                if add_node(c):
                    q.append((c, h + 1))
                    cnt += 1

        if len(kept) >= max_nodes:
            break

    return kept

# =========================================================
# 裁剪 + 渲染
# =========================================================

def clone_pruned_from(root: AXNode, kept: Set[AXNode]) -> AXNode:
    """
    从 root 出发克隆一个“瘦身子树”：
      - 若节点在 kept 或者其下存在需要保留的子树，则保留该节点；
      - 否则剪掉。
    """
    def rec(n: AXNode) -> Optional[AXNode]:
        new_children = []
        for ch in n.children:
            sub = rec(ch)
            if sub:
                new_children.append(sub)
        if (n in kept) or new_children:
            cp = AXNode(
                id=n.id, role=n.role, name=n.name, attrs=n.attrs.copy(),
                raw_line=n.raw_line, indent=n.indent
            )
            for c in new_children:
                c.parent = cp
            cp.children = new_children
            return cp
        return None

    pr = rec(root)
    return pr or root  # 理论上不为空；兜底

def render_axtree_text(root: AXNode) -> str:
    """
    严格还原为原始 axtree 文本风格：逐行 + 缩进（统一 4 空格） + 原 raw_line
    """
    lines: List[str] = []
    def rec(n: AXNode, depth: int):
        indent = "    " * depth
        if n.raw_line:
            line = n.raw_line.lstrip(" \t")
            lines.append(indent + line)
        else:
            # 极少数构造节点无 raw_line，尽量补一行
            header = f"[{n.id}] {n.role}" if n.id else (n.role or "")
            name = f" '{n.name}'" if n.name else ""
            tail = ""
            if n.attrs:
                kv = ", ".join(f"{k}='{v}'" for k, v in n.attrs.items())
                tail = f", {kv}"
            lines.append(indent + (header + name + tail).strip())
        for ch in n.children:
            rec(ch, depth + 1)
    rec(root, 0)
    return "\n".join(lines)

# =========================================================
# 公开 API：一次扩展并输出原格式文本（更全面视野）
# =========================================================

MODE_PRESETS = {
    "focused":  dict(hops=2, max_nodes=400,  max_children_per_node=3, up_levels_for_root=1),
    "balanced": dict(hops=3, max_nodes=800,  max_children_per_node=30, up_levels_for_root=1),
    "wide":     dict(hops=4, max_nodes=1200, max_children_per_node=50, up_levels_for_root=1),
}

def expand_subenv_once_to_text_wide(
    original_text: str,
    subenv_text: str,
    mode: str = "wide",
    hops: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_children_per_node: Optional[int] = None,
    up_levels_for_root: Optional[int] = None,
) -> str:
    """
    - original_text: 原始 axtree 文本
    - subenv_text:   子环境文本（只需包含若干 [id] 即可）
    - mode:          "focused" | "balanced" | "wide"
    - 其它参数：覆盖对应模式的默认值
    """
    root, id_map = parse_axtree(original_text)

    preset = MODE_PRESETS.get(mode, MODE_PRESETS["wide"]).copy()
    if hops is not None: preset["hops"] = hops
    if max_nodes is not None: preset["max_nodes"] = max_nodes
    if max_children_per_node is not None: preset["max_children_per_node"] = max_children_per_node
    if up_levels_for_root is not None: preset["up_levels_for_root"] = up_levels_for_root

    tgt_ids = [i for i in extract_ids(subenv_text) if i in id_map]
    targets = [id_map[i] for i in tgt_ids]
    if not targets:
        return "# 子环境中未发现任何存在于原始 axtree 的 [id]。"

    effective_max_nodes = max(preset["max_nodes"], len(targets))

    kept = neighborhood_expand(
        targets,
        hops=preset["hops"],
        max_nodes=effective_max_nodes,
        max_children_per_node=preset["max_children_per_node"],
    )

    render_root = safe_choose_render_root(
        targets,
        kept,
        preset["up_levels_for_root"],
    )
    pruned = clone_pruned_from(render_root, kept)

    def _count_id_nodes(n: AXNode) -> int:
        cnt, stack = 0, [n]
        while stack:
            x = stack.pop()
            if x.id is not None:
                cnt += 1
            stack.extend(x.children)
        return cnt

    if _count_id_nodes(pruned) == 0:
        lca = choose_lca(targets)
        pruned2 = clone_pruned_from(lca, kept)
        if _count_id_nodes(pruned2) > 0:
            pruned = pruned2
    return render_axtree_text(pruned)
# ===== 覆盖度计算：以“带 id 的节点数目”为单位 =====
def _count_id_nodes(root: AXNode) -> int:
    cnt = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.id is not None:
            cnt += 1
        stack.extend(n.children)
    return cnt

def _unique_ids_in_text(text: str) -> set[str]:
    ids = set()
    for ln in text.splitlines():
        m = re.search(r"^\s*\[([^\]]+)\]\s+", ln)   # 只匹配行首节点
        if m:
            ids.add(m.group(1))
    return ids

def expand_subenv_once_to_text_wide_with_metrics(
    original_text: str,
    subenv_text: str,
    mode: str = "wide",
    hops: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_children_per_node: Optional[int] = None,
    up_levels_for_root: Optional[int] = None,
) -> tuple[str, dict]:
    """
    返回: (expanded_text, metrics)
      metrics = {
        "original_total_ids": int,             # 原环境唯一 id 总数
        "before_ids_in_original": int,         # 子环境中落在原环境里的唯一 id 数
        "after_ids_in_expanded": int,          # 扩展后子环境里的唯一 id 数
        "before_ratio": float,                 # before_ids_in_original / original_total_ids
        "after_ratio": float                   # after_ids_in_expanded  / original_total_ids
      }
    """
    # 1) 解析原环境
    root, id_map = parse_axtree(original_text)

    # 2) 统计原环境唯一 id 总数
    original_total_ids = len(id_map)

    # 3) 子环境中出现在原环境里的唯一 id（扩展前）
    sub_ids_for_count = _unique_ids_in_text(subenv_text)
    before_ids_in_original = sum(1 for _id in sub_ids_for_count if _id in id_map)

    tgt_ids = [i for i in extract_ids(subenv_text) if i in id_map]
    if not tgt_ids:
        metrics = {
            "original_total_ids": original_total_ids,
            "before_ids_in_original": 0,
            "after_ids_in_expanded": 0,
            "before_ratio": 0.0 if original_total_ids else 0.0,
            "after_ratio": 0.0 if original_total_ids else 0.0,
        }
        raise ValueError("子环境中未发现任何存在于原始 axtree 的 [id]。")

    preset = MODE_PRESETS.get(mode, MODE_PRESETS["wide"]).copy()
    if hops is not None: preset["hops"] = hops
    if max_nodes is not None: preset["max_nodes"] = max_nodes
    if max_children_per_node is not None: preset["max_children_per_node"] = max_children_per_node
    if up_levels_for_root is not None: preset["up_levels_for_root"] = up_levels_for_root

    targets = [id_map[i] for i in tgt_ids]
    effective_max_nodes = max(preset["max_nodes"], len(targets))

    kept = neighborhood_expand(
        targets,
        hops=preset["hops"],
        max_nodes=effective_max_nodes,
        max_children_per_node=preset["max_children_per_node"],
    )

    render_root = safe_choose_render_root(
        targets,
        kept,
        preset["up_levels_for_root"],
    )
    pruned = clone_pruned_from(render_root, kept)

    def _count_id_nodes(n: AXNode) -> int:
        cnt, stack = 0, [n]
        while stack:
            x = stack.pop()
            if x.id is not None:
                cnt += 1
            stack.extend(x.children)
        return cnt

    if _count_id_nodes(pruned) == 0:
        lca = choose_lca(targets)
        pruned2 = clone_pruned_from(lca, kept)
        if _count_id_nodes(pruned2) > 0:
            pruned = pruned2

    # 5) 渲染文本
    expanded_text = render_axtree_text(pruned)

    # 6) 扩展后子环境的唯一 id 数（直接数裁剪树里的 id）
    after_ids_in_expanded = _count_id_nodes(pruned)

    # 7) 比例
    denom = original_total_ids if original_total_ids > 0 else 1
    before_ratio = before_ids_in_original / denom
    after_ratio  = after_ids_in_expanded  / denom

    metrics = {
        "original_total_ids": original_total_ids,
        "before_ids_in_original": before_ids_in_original,
        "after_ids_in_expanded": after_ids_in_expanded,
        "before_ratio": before_ratio,
        "after_ratio": after_ratio,
    }
    return expanded_text, metrics

# 文字节点判定：StaticText 或 无 id 且行内带引号内容（单/双引号均可）
TEXTUAL_ROLES = {"StaticText"}

_QUOTED_RE = re.compile(r"""(['"])(.*?)\1""")  # 捕获 'xxx' 或 "xxx"

def is_textual_node(n: AXNode) -> bool:
    """
    仅把“没有 id 的纯文本节点”视为可折叠文本节点。
    这样可以避免把 RootWebArea / LabelText / paragraph 等结构或半结构节点误折叠。
    """
    return (n.id is None) and ((n.role or "") in TEXTUAL_ROLES)

def merge_textual_children_general(root: AXNode) -> None:
    """
    通用折叠：
        [xxx] anyrole 'optional'
            StaticText 'A'
            StaticText "B"
            (或无 id 行含引号)
    →  [xxx] anyrole 'optional' StaticText 'A' 'B'
    规则：仅当“所有”子节点均为文字节点时合并；合并后删除这些子节点。
    """
    def rec(n: AXNode):
        # 先递归处理子树
        for ch in n.children:
            rec(ch)

        if not n.children:
            return

        # 子节点全为文字节点？
        if all(is_textual_node(c) for c in n.children):
            # 依次抽取所有子节点中的引号文字内容
            texts: list[str] = []
            for c in n.children:
                # 一行可能有多个引号片段，全部顺序拼接
                for m in _QUOTED_RE.finditer(c.raw_line):
                    texts.append(m.group(2))
            if texts:
                # 用单引号统一输出（必要时你可改为保留原引号样式）
                tail = " ".join(f"'{t}'" for t in texts)
                # 若父行本身已有 StaticText 标识就直接追加引号串；否则按 " StaticText '...'" 形式追加
                if "StaticText" in n.raw_line:
                    n.raw_line = (n.raw_line.rstrip() + " " + tail).rstrip()
                else:
                    n.raw_line = (n.raw_line.rstrip() + " StaticText " + tail).rstrip()
                # 清空子节点
                n.children.clear()

    rec(root)
def is_ancestor(a: AXNode, b: AXNode) -> bool:
    """a 是否是 b 的祖先（含相等）。"""
    cur = b
    while cur:
        if cur is a:
            return True
        cur = cur.parent
    return False

def covers_all(root_candidate: AXNode, nodes: list[AXNode]) -> bool:
    """root_candidate 是否覆盖 nodes 中的所有节点。"""
    return all(is_ancestor(root_candidate, n) for n in nodes)

def safe_choose_render_root(targets: list[AXNode], kept: set[AXNode], up_levels_for_root: int) -> AXNode:
    """
    在 LCA 基础上上抬 up_levels_for_root 层，但确保选出的根能覆盖 kept；
    否则逐级上溯到能覆盖的祖先；再不行就回退到 LCA。
    优先选择“有 id 的节点”作为根（避免把纯文本节点当根）。
    """
    # 仅以 targets 计算 LCA，更稳
    lca = choose_lca(targets)

    # 先尝试把 LCA 上抬
    cand = ascend(lca, up_levels_for_root)

    # 如果 cand 不是 kept 的共同祖先，就沿着 cand 的父链往上找，直到覆盖为止
    cur = cand
    while cur and not covers_all(cur, list(kept)):
        cur = cur.parent
    if cur:
        cand = cur
    else:
        cand = lca  # 兜底：回退到 LCA

    # 避免把“纯文本节点”当根：若 cand 无 id 且只有文本且没有子，也回退到 LCA
    def looks_textual_leaf(n: AXNode) -> bool:
        return (n.id is None) and (len(n.children) == 0) and (("'" in (n.raw_line or "")) or (n.role in {"StaticText", "LabelText", "paragraph", "heading"}))
    if looks_textual_leaf(cand) and cand is not lca:
        cand = lca

    # 再次保证覆盖
    if not covers_all(cand, list(kept)):
        cand = lca

    return cand
# =========================================================
# 可选：最小示例
# =========================================================
if __name__ == "__main__":
    with open("origin.txt", "r", encoding="utf-8") as f:
        ORIG = f.read().strip()
    with open("subenv.txt", "r", encoding="utf-8") as f:
        SUB = f.read().strip()
    expanded_text, m = expand_subenv_once_to_text_wide_with_metrics(
        original_text=ORIG,
        subenv_text=SUB,
        mode="focused", # "focused"|"balanced"|"wide"
    )
    print(expanded_text)
    print(
        f"[覆盖度] 原环境ID总数={m['original_total_ids']}, "
        f"扩展前={m['before_ids_in_original']}({m['before_ratio']:.2%}), "
        f"扩展后={m['after_ids_in_expanded']}({m['after_ratio']:.2%})"
    )
