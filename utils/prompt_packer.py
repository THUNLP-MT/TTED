
"""
prompt_packer_simple.py

Minimal prompt packer that only considers three inputs:
  - base_prompt: str   (your original prompt text: instructions, rules, examples, etc.)
  - history:      str | list[str]  (agent's step summaries, e.g., node.get_any_node_history output)
  - environment:  str

Policy (per user spec):
  - Only shrink "history" and "environment"; everything else is left as-is (inside base_prompt).
  - Compute token count of the fully assembled prompt.
  - If exceeds `max_tokens`:
      (i) keep only the last 3 steps in history;
      (ii) if still exceeds, truncate environment tokens to fit target (max_tokens - reserve).

Placeholders:
  - If base_prompt contains "{HISTORY}" and/or "{ENVIRONMENT}", they will be replaced accordingly.
  - Otherwise, "## History" and "## Environment" sections will be appended to base_prompt.

Tokenizer:
  - If model_name is provided, use HuggingFace AutoTokenizer for exact counting (e.g., Qwen/Qwen3-8B).
  - Else fallback to a heuristic tokenizer (~4 chars ≈ 1 token).

Return:
  - PackSimpleResult with the final prompt, token counts, and flags indicating truncations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils.history_tree import HistoryTree
# -------------------- Tokenizer --------------------

def _get_tokenizer(model_name: Optional[str] = None):
    if model_name:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            return lambda s: tok.encode(s, add_special_tokens=False), tok
        except Exception:
            pass
    def heuristic_encode(s: str):
        chunk = 4
        return [s[i:i+chunk] for i in range(0, len(s), chunk)]
    return heuristic_encode, None

def _count_tokens(text: str, encode: Callable[[str], List[Any]]) -> int:
    return 0 if not text else len(encode(text))

# -------------------- History helpers --------------------

def _history_to_lines(history: Any) -> List[str]:
    """Accepts list[str] or string with one step per line; returns list of non-empty lines (oldest->newest)."""
    if history is None:
        return []
    if isinstance(history, list):
        # assume each item is one summary line (already oldest->newest or vice versa; we won't reorder here)
        lines = []
        for item in history:
            s = str(item).strip()
            if s:
                lines.append(s)
        return lines
    if isinstance(history, str):
        # node.get_any_node_history format
        raw_lines = history.splitlines()
        lines = [ln.strip() for ln in raw_lines if ln.strip()]
        return lines
    return []

def _render_history(lines: List[str]) -> str:
    """Render numbered with most recent first."""
    if not lines:
        return ""
    out = []
    for i, item in enumerate(reversed(lines), 1):
        out.append(f"{i}. {item}")
    return "\n".join(out)

# -------------------- Assembly --------------------

def _assemble_prompt(base_prompt: str, history_block: str, env_block: str) -> str:
    has_hist = "{HISTORY}" in base_prompt
    has_env = "{ENVIRONMENT}" in base_prompt

    prompt = base_prompt
    if has_hist:
        prompt = prompt.replace("{HISTORY}", history_block)
    if has_env:
        prompt = prompt.replace("{ENVIRONMENT}", env_block)

    if not has_hist or not has_env:
        # append missing sections
        extra_sections = []
        if not has_hist and history_block:
            extra_sections.append("## History\n" + history_block)
        if not has_env and env_block:
            extra_sections.append("## Environment\n" + env_block)
        if extra_sections:
            prompt = prompt.rstrip() + "\n\n" + "\n\n".join(extra_sections) + "\n"
    return prompt

# -------------------- Public API --------------------

@dataclass
class PackResult:
    truncated_part: str
    total_tokens: int
    base_tokens: int
    history_tokens: int
    environment_tokens: int
    truncated_hist: bool
    truncated_env: bool
    target_tokens: int
    is_truncated: bool
    original_tokens: dict

def truncate_prompt(
    base_prompt: str,
    history: HistoryTree,
    environment: str,
    max_tokens: int = 40960,
    reserve: int = 1,
    model_name: Optional[str] = None,
    keep_last_n_history: int = 3
) -> PackResult:
    """
    Build the final prompt under a hard token budget by shrinking only 'history' then 'environment'.
    Target final tokens = max_tokens - reserve.
    """
    prompt = base_prompt
    encode, tok_obj = _get_tokenizer(model_name)

    # Prepare history block (full at first)
    if history and type(history) == str:
        hist_block_full = history
    elif history and type(history) == HistoryTree:
        hist_block_full = history.get_history()
    else:
        hist_block_full = ""
    env_block = environment or ""
    base_block = base_prompt

    # Token counts per section
    base_tokens = _count_tokens(base_block, encode)
    hist_tokens_full = _count_tokens(hist_block_full, encode)
    env_tokens_full = _count_tokens(env_block, encode)
    original_tokens = {
        "base": base_tokens,
        "history": hist_tokens_full,
        "environment": env_tokens_full
    }
    total = base_tokens
    target = max_tokens - reserve

    truncated_hist = False
    truncated_env = False

    # If within budget -> assemble and return
    if total <= max_tokens:
        prompt = base_prompt
        is_truncated = False
        # For transparency, we still show target.
        return PackResult(None, total, base_tokens, hist_tokens_full, env_tokens_full,
                                truncated_hist, truncated_env, target, is_truncated, original_tokens)

    # Step (i): shrink history to last N lines
    if history and type(history) == HistoryTree:
        hist_block_cut = history.get_n_history(keep_last_n_history)
        hist_tokens_cut = _count_tokens(hist_block_cut, encode)

        total_after_hist = total + hist_tokens_cut - hist_tokens_full
        truncated_hist = hist_tokens_cut < hist_tokens_full

        if total_after_hist <= max_tokens:
            is_truncated = True
            return PackResult({"history": hist_block_cut}, total_after_hist, base_tokens, hist_tokens_cut, env_tokens_full,
                                    truncated_hist, truncated_env, target, is_truncated, original_tokens)
    else:
        hist_tokens_cut = 0
        truncated_hist = False
        hist_block_cut = None
        total_after_hist = total
    # Step (ii): truncate environment to fit target
    # Budget left for env = target - base - history_cut
    prompt_beyond_env = total_after_hist - env_tokens_full
    budget_for_env = max(0, target - prompt_beyond_env)

    # Encode environment once, slice tokens
    encoded_env = encode(env_block)
    truncated_env_tokens = encoded_env[:budget_for_env]

    # Decode env back to string:
    if truncated_env_tokens and isinstance(truncated_env_tokens[0], str):
        env_block_cut = "".join(truncated_env_tokens)
    else:
        # Try to decode using HF tokenizer if we have tok_obj
        if tok_obj is not None:
            try:
                env_block_cut = tok_obj.decode(truncated_env_tokens, skip_special_tokens=True)
            except Exception:
                env_block_cut = ""
        else:
            env_block_cut = ""

    env_tokens_cut = _count_tokens(env_block_cut, encode)
    truncated_env = env_tokens_cut < env_tokens_full

    final_total = base_tokens - hist_tokens_full - env_tokens_full + hist_tokens_cut + env_tokens_cut
    # Assemble
    is_truncated = True
    return PackResult({"history": hist_block_cut, "env": env_block_cut}, final_total, base_tokens, hist_tokens_cut, env_tokens_cut,
                            truncated_hist, truncated_env, target, is_truncated, original_tokens)
