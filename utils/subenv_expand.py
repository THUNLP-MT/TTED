# -*- coding: utf-8 -*-


import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Set
from collections import deque


@dataclass(eq=False)
class AXNode:
    id: Optional[str]           
    role: Optional[str]        
    name: str                   
    attrs: Dict[str, str]       
    raw_line: str               
    indent: int                 
    parent: Optional["AXNode"] = None
    children: List["AXNode"] = field(default_factory=list)

    def __hash__(self) -> int:

        return id(self)



LINE_RE = re.compile(
    r"""
    ^(?P<indent>\s*)
    (?:
        \[(?P<id>\d+)\]\s+(?P<role_id>\w+)\s*(?P<rest_id>.*)   
        |
        (?P<role_no>\w+)\s*(?P<rest_no>.*)                    
    )
    \s*$
    """,
    re.VERBOSE,
)
NAME_RE = re.compile(r"'([^']*)'")
ATTR_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*'([^']*)'|([A-Za-z_]\w*)\s*=\s*([^,\s]+)")

def _indent_width(s: str, tabsize: int = 4) -> int:
    
    col = 0
    for ch in s:
        if ch == '\t':
            col += tabsize - (col % tabsize)
        else:
            col += 1
    return col

def _parse_name(rest: str) -> Tuple[str, str]:
    
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
    
    attrs: Dict[str, str] = {}
    for m in ATTR_RE.finditer(rest or ""):
        if m.group(1) is not None:  
            key, val = m.group(1), m.group(2)
        else:                        
            key, val = m.group(3), m.group(4)
        attrs[key] = val
    return attrs

def parse_axtree(text: str, tabsize: int = 4) -> Tuple[AXNode, Dict[str, AXNode]]:
    
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
            if m.group("id"): 
                nid = m.group("id")
                role = m.group("role_id")
                rest = (m.group("rest_id") or "").strip()
            else:             
                nid = None
                role = m.group("role_no")
                rest = (m.group("rest_no") or "").strip()
            name, rem = _parse_name(rest)
            attrs = _parse_attrs(rem)
            node = AXNode(id=nid, role=role, name=name, attrs=attrs, raw_line=ln, indent=indent)
            if nid:
                id_map[nid] = node

  
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

 
    if root is None:
        root = AXNode(id=None, role="__ROOT__", name="", attrs={}, raw_line="", indent=-1)
    if root.role != "__ROOT__":
        fake = AXNode(id=None, role="__ROOT__", name="", attrs={}, raw_line="", indent=-1)
        root.parent = fake
        fake.children.append(root)
        root = fake

    merge_textual_children_general(root)
    return root, id_map


def extract_ids(text: str) -> List[str]:
   
    ids, seen = [], set()
    for ln in text.splitlines():
        m = re.search(r"\[(\d+)\]", ln)
        if m:
            x = m.group(1)
            if x not in seen:
                seen.add(x); ids.append(x)
    return ids

def path_to_root_inclusive(n: AXNode) -> List[AXNode]:
    
    p = [n]
    cur = n.parent
    while cur:
        p.append(cur)
        cur = cur.parent
    return p

def choose_lca(nodes: List[AXNode]) -> AXNode:
 
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

    cur = n
    while up_levels > 0 and cur.parent:
        cur = cur.parent
        up_levels -= 1
    return cur



def neighborhood_expand(
    targets: List[AXNode],
    hops: int = 3,
    max_nodes: int = 1200,
    max_children_per_node: int = 50,
) -> Set[AXNode]:

    kept: Set[AXNode] = set()
    q = deque()


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


        if node.parent and add_node(node.parent):
            q.append((node.parent, h + 1))

        if node.parent:
            for s in node.parent.children:
                if s is node:
                    continue
                if add_node(s):
                    q.append((s, h + 1))


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



def clone_pruned_from(root: AXNode, kept: Set[AXNode]) -> AXNode:

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
    return pr or root 

def render_axtree_text(root: AXNode) -> str:

    lines: List[str] = []
    def rec(n: AXNode, depth: int):
        indent = "    " * depth
        if n.raw_line:
            line = n.raw_line.lstrip(" \t")
            lines.append(indent + line)
        else:
           
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

    root, id_map = parse_axtree(original_text)

    preset = MODE_PRESETS.get(mode, MODE_PRESETS["wide"]).copy()
    if hops is not None: preset["hops"] = hops
    if max_nodes is not None: preset["max_nodes"] = max_nodes
    if max_children_per_node is not None: preset["max_children_per_node"] = max_children_per_node
    if up_levels_for_root is not None: preset["up_levels_for_root"] = up_levels_for_root
    tgt_ids = [i for i in extract_ids(subenv_text) if i in id_map]
    targets = [id_map[i] for i in tgt_ids]
    if not targets:
        return "Empty axtree"
    effective_max_nodes = max(preset["max_nodes"], len(targets))

    # kept = neighborhood_expand(
    #     targets,
    #     hops=preset["hops"],
    #     max_nodes=preset["max_nodes"],
    #     max_children_per_node=preset["max_children_per_node"],
    # )
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
        m = re.search(r"\[(\d+)\]", ln)
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


    root, id_map = parse_axtree(original_text)

    original_total_ids = len(id_map)


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
        raise ValueError(f"Empty axtree")


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


    expanded_text = render_axtree_text(pruned)


    after_ids_in_expanded = _count_id_nodes(pruned)


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


TEXTUAL_ROLES = {"StaticText"}

_QUOTED_RE = re.compile(r"""(['"])(.*?)\1""")  # 捕获 'xxx' 或 "xxx"

def is_textual_node(n: AXNode) -> bool:

    return (n.id is None) and ((n.role or "") in TEXTUAL_ROLES)

def merge_textual_children_general(root: AXNode) -> None:

    def rec(n: AXNode):
       
        for ch in n.children:
            rec(ch)

        if not n.children:
            return

    
        if all(is_textual_node(c) for c in n.children):
     
            texts: list[str] = []
            for c in n.children:
                
                for m in _QUOTED_RE.finditer(c.raw_line):
                    texts.append(m.group(2))
            if texts:
           
                tail = " ".join(f"'{t}'" for t in texts)
             
                if "StaticText" in n.raw_line:
                    n.raw_line = (n.raw_line.rstrip() + " " + tail).rstrip()
                else:
                    n.raw_line = (n.raw_line.rstrip() + " StaticText " + tail).rstrip()
         
                n.children.clear()

    rec(root)
def is_ancestor(a: AXNode, b: AXNode) -> bool:
    
    cur = b
    while cur:
        if cur is a:
            return True
        cur = cur.parent
    return False

def covers_all(root_candidate: AXNode, nodes: list[AXNode]) -> bool:
    
    return all(is_ancestor(root_candidate, n) for n in nodes)

def safe_choose_render_root(targets: list[AXNode], kept: set[AXNode], up_levels_for_root: int) -> AXNode:

  
    lca = choose_lca(targets)

  
    cand = ascend(lca, up_levels_for_root)

   
    cur = cand
    while cur and not covers_all(cur, list(kept)):
        cur = cur.parent
    if cur:
        cand = cur
    else:
        cand = lca  

   
    def looks_textual_leaf(n: AXNode) -> bool:
        return (n.id is None) and (len(n.children) == 0) and (("'" in (n.raw_line or "")) or (n.role in {"StaticText", "LabelText", "paragraph", "heading"}))
    if looks_textual_leaf(cand) and cand is not lca:
        cand = lca

    
    if not covers_all(cand, list(kept)):
        cand = lca

    return cand

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

