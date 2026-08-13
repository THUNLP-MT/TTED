# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    processing_times: list[float]
    tool_calls: list[float]
    param_version: int
    param_version_start: list[int]
    param_version_end: list[int]
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None
    global_steps: Optional[int] = None
    param_version: Optional[int] = None


def _ensure_non_tensor_field(dp: DataProto, key: str, *, length: int, default, dtype, make_unique=False):
    """
    确保 dp.non_tensor_batch[key] 存在且 shape[0]==length
    不存在则创建；长度不匹配则重建（保守策略：用第一个值 broadcast 或重建 unique）
    """
    nt = dp.non_tensor_batch
    if key not in nt:
        if make_unique:
            nt[key] = np.array([f"{default}_{i}" for i in range(length)], dtype=dtype)
        else:
            nt[key] = np.array([default] * length, dtype=dtype)
        return

    val = nt[key]
    try:
        n = len(val)
    except Exception:
        nt[key] = np.array([val] * length, dtype=dtype)
        return

    if n == length:
        return

    if make_unique:
        nt[key] = np.array([f"{default}_{i}" for i in range(length)], dtype=dtype)
    else:
        first = val[0] if n > 0 else default
        nt[key] = np.array([first] * length, dtype=dtype)


def normalize_rollout_sample_inplace(
    rs: "RolloutSample",
    *,
    default_param_version: int = 0,
    fallback_rollout_status: Optional[dict[str, Any]] = None,
) -> "RolloutSample":
    """
    让 RolloutSample.full_batch 具备 assemble 所需要的字段，并保证 non_tensor_batch 长度一致。
    """
    if rs.rollout_status is None:
        rs.rollout_status = fallback_rollout_status or {}

    fb = rs.full_batch
    if not isinstance(fb, DataProto):
        if isinstance(fb, dict):
            fb = DataProto.from_single_dict(fb)
            rs.full_batch = fb
        else:
            raise TypeError(f"RolloutSample.full_batch must be DataProto or dict, got {type(fb)}")

    bsz = len(fb)

    _ensure_non_tensor_field(
        fb,
        "uid",
        length=bsz,
        default=f"uid_{rs.sample_id}",
        dtype=object,
        make_unique=True,
    )
    _ensure_non_tensor_field(fb, "processing_times", length=bsz, default=0.0, dtype=float)
    _ensure_non_tensor_field(fb, "tool_calls_times", length=bsz, default=0.0, dtype=float)
    _ensure_non_tensor_field(fb, "param_version_start", length=bsz, default=default_param_version, dtype=int)
    _ensure_non_tensor_field(fb, "param_version_end", length=bsz, default=default_param_version, dtype=int)
    _ensure_non_tensor_field(fb, "param_version", length=bsz, default=default_param_version, dtype=int)

    return rs


def coerce_to_rollout_sample(
    obj: Any,
    *,
    sample_id: str,
    epoch: int,
    default_param_version: int = 0,
    fallback_rollout_status: Optional[dict[str, Any]] = None,
) -> "RolloutSample":
    """
    兼容离线格式：
    - RolloutSample / DataProto / dict(batch_dict)
    """
    if isinstance(obj, RolloutSample):
        rs = obj
        rs.sample_id = getattr(rs, "sample_id", None) or sample_id
        rs.epoch = getattr(rs, "epoch", None) if getattr(rs, "epoch", None) is not None else epoch
        rs.param_version = getattr(rs, "param_version", default_param_version)
        if getattr(rs, "agent_loop_output_list", None) is None:
            rs.agent_loop_output_list = []
        if getattr(rs, "rollout_status", None) is None:
            rs.rollout_status = fallback_rollout_status or {}
        return normalize_rollout_sample_inplace(
            rs,
            default_param_version=default_param_version,
            fallback_rollout_status=fallback_rollout_status,
        )

    if isinstance(obj, DataProto):
        rs = RolloutSample(
            full_batch=obj,
            agent_loop_output_list=[],
            sample_id=sample_id,
            epoch=epoch,
            processing_times=[],
            tool_calls=[],
            param_version=default_param_version,
            param_version_start=[],
            param_version_end=[],
            rollout_status=fallback_rollout_status or {},
        )
        return normalize_rollout_sample_inplace(rs, default_param_version=default_param_version)

    if isinstance(obj, dict):
        dp = DataProto.from_single_dict(obj)
        rs = RolloutSample(
            full_batch=dp,
            agent_loop_output_list=[],
            sample_id=sample_id,
            epoch=epoch,
            processing_times=[],
            tool_calls=[],
            param_version=default_param_version,
            param_version_start=[],
            param_version_end=[],
            rollout_status=fallback_rollout_status or {},
        )
        return normalize_rollout_sample_inplace(rs, default_param_version=default_param_version)

    raise TypeError(f"Unsupported offline object type: {type(obj)}")


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    full_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    # Setting selected agent, that supports partial
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(
            ["async_partial_tool_agent"] * len(full_batch), dtype=object
        )
    else:
        full_batch.non_tensor_batch["agent_name"] = np.array(
            ["partial_single_turn_agent"] * len(full_batch), dtype=object
        )

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    processing_times = []
    tool_calls = []
    # 兼容：rollout_status 可能为空
    rollout_status = rollout_samples[0].rollout_status or {}
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        # 双保险：确保字段存在且长度正确
        normalize_rollout_sample_inplace(rs, default_param_version=getattr(rs, "param_version", 0))
        rollout_samples_batch.append(rs.full_batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    if "advantages" in final_batch.batch.keys():
        final_batch.batch["pre_compute_advantages"] = final_batch.batch["advantages"]
        final_batch.batch.pop("advantages")

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if "reward_baselines" not in final_batch.batch.keys():
        final_batch.batch["reward_baselines"] = torch.zeros_like(final_batch.batch["token_level_scores"][:, 0])

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    # normalize_rollout_sample_inplace 已经确保这些字段存在
    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics

    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["param_version_start"]
    param_version_end = final_batch.non_tensor_batch["param_version_end"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    param_versions = [rs.param_version for rs in rollout_samples]
    trajectorys_param_versions = final_batch.non_tensor_batch["param_version_end"]

    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            "trajectory_param_versions": trajectorys_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t}")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }
