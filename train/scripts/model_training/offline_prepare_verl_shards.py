# offline_prepare_verl_shards.py
import os
import json
import math
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import httpx
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class PrepConfig:
    input_jsonl: str
    output_dir: str

    # tokenizer / model
    tokenizer_path: str
    # chat_template_path: str

    # vLLM OpenAI server
    vllm_base_url: str       
    vllm_model: str          
    temperature: float = 1.0 
    add_special_tokens: bool = True

    # logprobs settings
    prompt_logprobs_k: int = 1

    # lengths
    max_prompt_len: int = 16384
    max_response_len: int = 4096
    max_seq_len: int = 20480 

    # padding strategy
    prompt_pad_side: str = "left" 
    response_pad_side: str = "right"

    # performance
    shard_size: int = 2048
    request_batch_size: int = 32
    concurrency: int = 64
    timeout_s: float = 120.0


def _left_pad(ids: List[int], pad_id: int, length: int) -> List[int]:
    if len(ids) >= length:
        return ids[-length:]
    return [pad_id] * (length - len(ids)) + ids


def _right_pad(ids: List[int], pad_id: int, length: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def truncate_prompt_response(
    prompt_ids: List[int],
    response_ids: List[int],
    max_prompt_len: int,
    max_response_len: int,
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    # 1) truncate response first
    response_ids = response_ids[:max_response_len]

    # 2) truncate prompt by max_prompt_len
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    # 3) ensure total <= max_seq_len
    keep = max_seq_len - len(response_ids)
    if keep <= 0:
        # response too long: keep at least 1 token for prompt if possible
        # (or you can drop the sample)
        new_resp_len = max(1, max_seq_len - 1)
        response_ids = response_ids[:new_resp_len]
        keep = max_seq_len - len(response_ids)

    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:] if keep > 0 else []

    return prompt_ids, response_ids


class VLLMScorer:
    """
    Score a *given* token sequence using vLLM OpenAI-compatible /v1/completions.

    IMPORTANT:
    - We use: echo=True + max_tokens=0 + prompt_logprobs=K + return_token_ids=true
      This forces vLLM to return logprobs for *every token in the prompt* (teacher-forcing / prefill logprobs),
      without actually generating new tokens.
    - If you pass prompt as token IDs (list[int] / list[list[int]]), vLLM will not re-tokenize text,
      which guarantees alignment between your offline tokenization and returned logprobs.

    Note:
    - vLLM documentation notes prompt_logprobs is not compatible with prefix caching; you may need to start server with:
      --no-enable-prefix-caching
    """
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
        )

    async def aclose(self):
        await self.client.aclose()

    @staticmethod
    def _normalize_prompt_token_ids(token_ids: Any) -> List[int]:
        if token_ids is None:
            raise RuntimeError("prompt_token_ids missing in vLLM response")
        if not isinstance(token_ids, list):
            raise RuntimeError(f"prompt_token_ids has unexpected type: {type(token_ids)}")
        out: List[int] = []
        for x in token_ids:
            # vLLM may return ints or strings
            out.append(int(x))
        return out

    @staticmethod
    def _extract_logprob_from_entry(entry_value: Any) -> float:
        """
        vLLM prompt_logprobs entry value examples:
          - {"logprob": -2.3, "rank": 1, "decoded_token": "foo"}
          - -2.3   (some variants)
        """
        if isinstance(entry_value, (int, float, np.number)):
            return float(entry_value)
        if isinstance(entry_value, dict):
            if "logprob" in entry_value:
                return float(entry_value["logprob"])
        raise RuntimeError(f"Unsupported logprob entry value type: {type(entry_value)}")

    @classmethod
    def _lookup_token_logprob_in_topk_map(cls, token_id: int, lp_map: Dict[Any, Any]) -> float:
        """
        lp_map is a dict mapping token_id -> logprob info (or token string -> info).
        We try several key forms robustly.
        """
        # 1) exact match by int key
        if token_id in lp_map:
            return cls._extract_logprob_from_entry(lp_map[token_id])
        # 2) exact match by str key
        k = str(token_id)
        if k in lp_map:
            return cls._extract_logprob_from_entry(lp_map[k])
        # 3) some servers stringify keys; try int-cast on keys
        for kk, vv in lp_map.items():
            try:
                if int(kk) == token_id:
                    return cls._extract_logprob_from_entry(vv)
            except Exception:
                continue
        raise KeyError(f"Token id {token_id} missing in prompt_logprobs map keys(sample)={list(lp_map)[:5]}")

    async def score_batch_prompt_logprobs(
        self,
        prompts: Union[List[str], List[List[int]]],
        temperature: float,
        max_tokens: int,
        prompt_logprobs_k: int = 1,
        return_token_ids: bool = True,
        echo: bool = True,
        add_special_tokens: Optional[bool] = None,
    ) -> List[Tuple[List[int], List[Optional[float]]]]:
        """
        Return for each text:
          token_ids (prompt_token_ids)
          token_logprobs aligned to token_ids (may include None for first token)
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_logprobs": int(prompt_logprobs_k),
            "return_token_ids": return_token_ids,
            "echo": echo,
            "n": 1,
        }
        # Only set add_special_tokens when caller explicitly asks.
        # For token-id prompts, you almost always want add_special_tokens=False (avoid server-side mutation).
        if add_special_tokens is not None:
            payload["add_special_tokens"] = bool(add_special_tokens)

        r = await self.client.post("/v1/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        # vLLM returns one choice per prompt when n=1
        choices = data["choices"]
        if len(choices) != len(prompts):
            # Some servers return flattened choices; handle simplest case only
            raise RuntimeError(f"Unexpected choices size: {len(choices)} vs {len(prompts)}")

        out: List[Tuple[List[int], List[Optional[float]]]] = []
        for ch in choices:
            token_ids_raw = ch.get("prompt_token_ids")
            token_ids = self._normalize_prompt_token_ids(token_ids_raw)

            plp = ch.get("prompt_logprobs", None)
            if plp is None:
                # Some variants may put it under ch["logprobs"] with echo; but vLLM commonly exposes prompt_logprobs.
                raise RuntimeError(
                    "prompt_logprobs missing in vLLM response. "
                    "Make sure you set echo=True and prompt_logprobs, and consider starting vLLM with "
                    "--no-enable-prefix-caching if you enabled prefix caching."
                )
            if not isinstance(plp, list):
                raise RuntimeError(f"prompt_logprobs has unexpected type: {type(plp)}")

            if len(plp) > len(token_ids):
                plp = plp[: len(token_ids)]
            if len(plp) != len(token_ids):
                raise RuntimeError(f"prompt_logprobs length mismatch: {len(plp)} vs token_ids {len(token_ids)}")

            token_logprobs: List[Optional[float]] = []
            for i, (tid, lp_map) in enumerate(zip(token_ids, plp)):
                if lp_map is None:
                    # Typically only the first token has None (no previous context).
                    token_logprobs.append(None)
                    continue
                if not isinstance(lp_map, dict):
                    raise RuntimeError(f"prompt_logprobs[{i}] has unexpected type: {type(lp_map)}")
                try:
                    lp = self._lookup_token_logprob_in_topk_map(tid, lp_map)
                except KeyError as e:
                    raise RuntimeError(
                        f"Token id {tid} missing in prompt_logprobs map at position {i}. "
                        f"This should be rare; you can try increasing prompt_logprobs_k."
                    ) from e
                token_logprobs.append(lp)

            out.append((token_ids, token_logprobs))
        return out

    async def score_texts(
        self,
        texts: List[str],
        temperature: float,
        add_special_tokens: bool,
        prompt_logprobs_k: int = 1,
    ) -> List[Tuple[List[int], List[Optional[float]]]]:
        try:
            return await self.score_batch_prompt_logprobs(
                prompts=texts,
                temperature=temperature,
                max_tokens=0,
                prompt_logprobs_k=prompt_logprobs_k,
                return_token_ids=True,
                echo=True,
                add_special_tokens=add_special_tokens,
            )
        except Exception:
            return await self.score_batch_prompt_logprobs(
                prompts=texts,
                temperature=temperature,
                max_tokens=1,
                prompt_logprobs_k=prompt_logprobs_k,
                return_token_ids=True,
                echo=True,
                add_special_tokens=add_special_tokens,
            )

    async def score_token_ids(
        self,
        token_id_seqs: List[List[int]],
        temperature: float,
        prompt_logprobs_k: int = 1,
    ) -> List[Tuple[List[int], List[Optional[float]]]]:
        """
        Token-id mode (recommended):
        - prompt is list[list[int]]
        - add_special_tokens MUST be False to avoid server-side token mutation
        """
        try:
            return await self.score_batch_prompt_logprobs(
                prompts=token_id_seqs,
                temperature=temperature,
                max_tokens=0,
                prompt_logprobs_k=prompt_logprobs_k,
                return_token_ids=True,
                echo=True,
                add_special_tokens=False,
            )
        except Exception:
            return await self.score_batch_prompt_logprobs(
                prompts=token_id_seqs,
                temperature=temperature,
                max_tokens=1,
                prompt_logprobs_k=prompt_logprobs_k,
                return_token_ids=True,
                echo=True,
                add_special_tokens=False,
            )

def build_advantages(
    reward: Any,
    resp_len: int,
    max_response_len: int,
    length_norm: bool = True,
) -> np.ndarray:
    """
    reward:
      - float/int: scalar reward
      - list[float]: token-level reward, length==resp_len (best effort)
    """
    adv = np.zeros((max_response_len,), dtype=np.float32)
    if resp_len <= 0:
        return adv

    if isinstance(reward, (int, float, np.number)):
        r = float(reward)
        if length_norm:
            r = r / max(1, resp_len)
        adv[:resp_len] = r
        return adv

    if isinstance(reward, list):
        raise NotImplementedError("Token-level reward advantage not implemented yet.")

    return adv


def make_shard_buffers(cfg: PrepConfig, pad_id: int):
    maxP, maxR = cfg.max_prompt_len, cfg.max_response_len
    maxS = maxP + maxR

    tensors = {
        "prompts": [],
        "responses": [],
        "input_ids": [],
        "attention_mask": [],
        "response_mask": [],
        "position_ids": [],
        "rollout_log_probs": [],
        "token_level_scores": [],
        "advantages": [],
    }
    non_tensors = {
        "sample_id": [],
        "processing_times": [],
        "tool_calls_times": [],
        "param_version_start": [],
        "param_version_end": [],
    }
    return tensors, non_tensors


def finalize_and_save_shard(cfg: PrepConfig, shard_idx: int, tensors, non_tensors):
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"shard_{shard_idx:05d}.pt")

    # stack tensors
    stacked = {k: torch.stack(v, dim=0) for k, v in tensors.items()}

    # non-tensors -> numpy arrays
    nt = {k: np.array(v, dtype=object) for k, v in non_tensors.items()}

    torch.save(
        {
            "tensors": stacked,
            "non_tensors": nt,
            "meta_info": {
                "temperature": cfg.temperature,
                "offline": True,
            },
        },
        path,
    )
    return path


async def main(cfg: PrepConfig):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        # common fallback: use eos as pad
        tok.pad_token = tok.eos_token

    # with open(cfg.chat_template_path, "r", encoding="utf-8") as f:
    #     chat_template_jinja = f.read()

    pad_id = tok.pad_token_id

    scorer = VLLMScorer(cfg.vllm_base_url, cfg.vllm_model, timeout_s=cfg.timeout_s)

    tensors, non_tensors = make_shard_buffers(cfg, pad_id)
    shard_idx = 0
    shard_count = 0

    with open(cfg.input_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for i in tqdm(range(0, len(rows), cfg.request_batch_size), desc="Preparing"):
        batch_rows = rows[i : i + cfg.request_batch_size]

        # 1) Pre-tokenize & truncate (CPU), build token-id sequences for scoring
        prep: List[Dict[str, Any]] = []
        token_id_seqs: List[List[int]] = []
        for row in batch_rows:
            sample_id = row.get("sample_id") or row.get("id") or f"row_{np.random.randint(1<<30)}"
            
            assert len(row["messages"]) == 2, "Each data item must contain exactly 2 messages: user and assistant."

            prompt_ids = tok.apply_chat_template(
                [row["messages"][0]],
                add_generation_prompt=True,
            )
            response_ids = tok.apply_chat_template(
                row["messages"],
                add_generation_prompt=False,
            )
            response_ids = response_ids[len(prompt_ids):]  # only keep response 
            print("response lengths:", len(response_ids))

            prompt_ids, response_ids = truncate_prompt_response(
                prompt_ids, response_ids,
                cfg.max_prompt_len, cfg.max_response_len, cfg.max_seq_len
            )
            resp_len = len(response_ids)

            full_ids = prompt_ids + response_ids
            prep.append(
                {
                    "row": row,
                    "sample_id": sample_id,
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "resp_len": resp_len,
                    "full_ids": full_ids,
                }
            )
            token_id_seqs.append(full_ids)

        # 2) Score in *token-id mode* to guarantee alignment
        scored = await scorer.score_token_ids(
            token_id_seqs=token_id_seqs,
            temperature=cfg.temperature,
            prompt_logprobs_k=cfg.prompt_logprobs_k,
        )

        # 3) Build final tensor items in the original order
        results: List[Dict[str, Any]] = []
        for p, (ret_token_ids, ret_token_logprobs) in zip(prep, scored, strict=False):
            row = p["row"]
            sample_id = p["sample_id"]
            prompt_ids = p["prompt_ids"]
            response_ids = p["response_ids"]
            resp_len = p["resp_len"]
            full_ids = p["full_ids"]

            # Sanity check alignment. With token-id prompts, ret_token_ids should equal full_ids.
            # Some servers may prepend BOS if add_special_tokens mishandled; we handle "endswith" as fallback.
            offset = 0
            if ret_token_ids != full_ids:
                if len(ret_token_ids) >= len(full_ids) and ret_token_ids[-len(full_ids):] == full_ids:
                    offset = len(ret_token_ids) - len(full_ids)
                else:
                    raise RuntimeError(
                        "Returned prompt_token_ids do not match input token IDs. "
                        "Check tokenizer/model mismatch and ensure add_special_tokens=False for token-id prompts."
                    )

            start = offset + len(prompt_ids)
            end = start + resp_len
            resp_logprobs = ret_token_logprobs[start:end]
            if len(resp_logprobs) != resp_len:
                raise RuntimeError(f"Response logprobs length mismatch: {len(resp_logprobs)} vs resp_len {resp_len}")
            # None should not appear in response portion. If it does, abort (silent 0.0 is very dangerous).
            if any(v is None for v in resp_logprobs):
                raise RuntimeError("Found None in response token logprobs; this indicates server-side logprobs issue.")
            resp_lp = np.array([float(v) for v in resp_logprobs], dtype=np.float32)

            # pad prompt/response ids
            if cfg.prompt_pad_side == "left":
                p_pad = _left_pad(prompt_ids, pad_id, cfg.max_prompt_len)
            else:
                p_pad = _right_pad(prompt_ids, pad_id, cfg.max_prompt_len)

            if cfg.response_pad_side == "right":
                r_pad = _right_pad(response_ids, pad_id, cfg.max_response_len)
            else:
                r_pad = _left_pad(response_ids, pad_id, cfg.max_response_len)

            prompts_t = torch.tensor(p_pad, dtype=torch.long)
            responses_t = torch.tensor(r_pad, dtype=torch.long)

            # masks (pad positions are 0)
            prompt_mask = (prompts_t != pad_id).to(torch.long)
            response_mask = (responses_t != pad_id).to(torch.long)
            attention_mask = torch.cat([prompt_mask, response_mask], dim=0)
            position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

            # rollout_log_probs padded (response length only)
            rlp = np.zeros((cfg.max_response_len,), dtype=np.float32)
            rlp[:resp_len] = resp_lp[:resp_len]

            # advantages padded
            reward = row.get("reward", 0.0)
            token_level_scores = np.zeros((cfg.max_response_len,), dtype=np.float32)
            token_level_scores[resp_len-1] = reward  # simple scalar reward at the end
            adv = build_advantages(reward, resp_len, cfg.max_response_len, length_norm=False)

            results.append(
                {
                    "sample_id": sample_id,
                    "prompts": prompts_t,
                    "responses": responses_t,
                    "input_ids": torch.cat([prompts_t, responses_t], dim=0),
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "position_ids": position_ids,
                    "rollout_log_probs": torch.tensor(rlp, dtype=torch.float32),
                    "token_level_scores": torch.tensor(token_level_scores, dtype=torch.float32),
                    "advantages": torch.tensor(adv, dtype=torch.float32),
                }
            )

        for item in results:
            tensors["prompts"].append(item["prompts"])
            tensors["responses"].append(item["responses"])
            tensors["input_ids"].append(item["input_ids"])
            tensors["attention_mask"].append(item["attention_mask"])
            tensors["response_mask"].append(item["response_mask"])
            tensors["position_ids"].append(item["position_ids"])
            tensors["rollout_log_probs"].append(item["rollout_log_probs"])
            tensors["token_level_scores"].append(item["token_level_scores"])
            tensors["advantages"].append(item["advantages"])

            non_tensors["sample_id"].append(item["sample_id"])
            non_tensors["processing_times"].append(0.0)
            non_tensors["tool_calls_times"].append(0.0)
            non_tensors["param_version_start"].append(0)
            non_tensors["param_version_end"].append(0)

            shard_count += 1

            if shard_count >= cfg.shard_size:
                out = finalize_and_save_shard(cfg, shard_idx, tensors, non_tensors)
                print(f"[Saved] {out} with {shard_count} samples")
                shard_idx += 1
                shard_count = 0
                tensors, non_tensors = make_shard_buffers(cfg, pad_id)

    # flush last shard
    if shard_count > 0:
        out = finalize_and_save_shard(cfg, shard_idx, tensors, non_tensors)
        print(f"[Saved] {out} with {shard_count} samples")

    await scorer.aclose()


if __name__ == "__main__":
    # Example:
    # cfg = PrepConfig(
    #     input_jsonl="test.jsonl",
    #     output_dir="test_shards",
    #     tokenizer_path="Qwen3-8B",
    #     vllm_base_url="",
    #     vllm_model="",
    #     temperature=1.0,
    #     add_special_tokens=True,
    #     max_prompt_len=16384,
    #     max_response_len=6144,
    #     max_seq_len=22528,
    #     shard_size=1024,
    #     request_batch_size=16,
    #     concurrency=64,
    # )
    # asyncio.run(main(cfg))

    # cfg = PrepConfig(
    #     input_jsonl="train.jsonl",
    #     output_dir="train_shards",
    #     tokenizer_path="Qwen3-8B",
    #     vllm_base_url="",
    #     vllm_model="",
    #     temperature=1.0,
    #     add_special_tokens=True,
    #     max_prompt_len=16384,
    #     max_response_len=6144,
    #     max_seq_len=22528,
    #     shard_size=1024,
    #     request_batch_size=16,
    #     concurrency=64,
    # )
    # asyncio.run(main(cfg))
