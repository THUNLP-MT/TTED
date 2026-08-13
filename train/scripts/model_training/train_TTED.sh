#!/usr/bin/env bash
set -xeuo pipefail

# TODO: Fill in a project name before running.
project_name=''

# TODO: Fill in an experiment name before running.
exp_name=''

# Paths

# TODO: Set MODEL_PATH to the pretrained model directory.
MODEL_PATH=${MODEL_PATH:-""}

# TODO: Set CKPTS_DIR to the checkpoint output directory.
CKPTS_DIR=${CKPTS_DIR:-""}

# TODO: Set TRAIN_DIR to the training-shard directory.
TRAIN_DIR=${TRAIN_DIR:-""}

# TODO: Set TEST_DIR to the validation/test-shard directory.
TEST_DIR=${TEST_DIR:-""}

# Ray cluster resources
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Rollout backend
rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"}  # sglang or vllm

# set -u safety: always define this
export VLLM_USE_V1=1
return_raw_chat="True"

# ---------------------------
# Fixed training schedule
# ---------------------------
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-125}

TRAIN_STEP_BSZ=${TRAIN_STEP_BSZ:-96}

# required_samples = ppo_mini_batch_size * require_batches
require_batches=${REQUIRE_BATCHES:-1}
train_prompt_mini_bsz=${TRAIN_STEP_BSZ}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}

# 让 current_param_version == trainer update step
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}

# total_train_steps = total_rollout_steps / (required_samples * trigger_parameter_sync_step)
# => total_rollout_steps = NUM_TRAIN_STEPS * required_samples * trigger_parameter_sync_step
required_samples=$((train_prompt_mini_bsz * require_batches))
total_rollout_steps=$((NUM_TRAIN_STEPS * required_samples * trigger_parameter_sync_step))

train_max_samples=${TRAIN_MAX_SAMPLES:-$total_rollout_steps}

rollout_total_epochs=${ROLLOUT_TOTAL_EPOCHS:-1}

# ---------------------------
# Validation size control
# ---------------------------
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256} 
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16} 
VALIDATION_SHUFFLE=${VALIDATION_SHUFFLE:-False}

# Validation frequency (by param_version / step)
test_freq=${TEST_FREQ:-25}

# ---------------------------
# Original algorithm/training params (keep)
# ---------------------------
adv_estimator=reinforce_plus_plus
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.005
clip_ratio_high=0.01

max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 6))

temperature=1.0
top_p=0.95
top_k=20
val_top_p=0.95

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
ref_offload=True
actor_offload=False
gen_tp=1
sp_size=1
fsdp_size=-1

staleness_threshold=200
partial_rollout=False

# For fully async pipeline requirement
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=1

echo "[INFO] NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}"
echo "[INFO] required_samples=${required_samples} (ppo_mini_batch_size=${train_prompt_mini_bsz} * require_batches=${require_batches})"
echo "[INFO] trigger_parameter_sync_step=${trigger_parameter_sync_step}"
echo "[INFO] total_rollout_steps=${total_rollout_steps}"
echo "[INFO] VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}, validation_shuffle=${VALIDATION_SHUFFLE}"

# HYDRA_FULL_ERROR=1 \
python -m verl.experimental.fully_async_offline_policy.fully_async_main \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.train_batch_size=${train_prompt_bsz} \
  data.gen_batch_size=${gen_prompt_bsz} \
  data.return_raw_chat=${return_raw_chat} \
  data.train_max_samples=${train_max_samples} \
  data.val_max_samples=${VAL_MAX_SAMPLES} \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.validation_shuffle=${VALIDATION_SHUFFLE} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_liger=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
  actor_rollout_ref.rollout.name=${rollout_name} \
  actor_rollout_ref.rollout.mode=${rollout_mode} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.total_epochs="${rollout_total_epochs}" \
  trainer.val_before_train=True \
  trainer.save_freq=25 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=auto \
  trainer.nnodes="${NNODES_TRAIN}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  rollout.nnodes="${NNODES_ROLLOUT}" \
  rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
  rollout.total_rollout_steps="${total_rollout_steps}" \
  rollout.total_epochs="${rollout_total_epochs}" \
  rollout.test_freq="${test_freq}" \
  async_training.staleness_threshold="${staleness_threshold}" \
  async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
  async_training.require_batches="${require_batches}" \
  async_training.partial_rollout="${partial_rollout}" \
  async_training.use_rollout_log_probs=True \
  async_training.offline_rollout.enable=True \
  async_training.offline_rollout.train_data_dir="${TRAIN_DIR}" \
  async_training.offline_rollout.val_data_dir="${TEST_DIR}" \
  async_training.offline_rollout.file_glob="*.pt" \
  async_training.offline_rollout.total_epochs=1 \
  async_training.offline_use_reward_from_data=True \
  async_training.offline_validate_freq=25 \
  async_training.drop_samples_if_queue_full=False \
