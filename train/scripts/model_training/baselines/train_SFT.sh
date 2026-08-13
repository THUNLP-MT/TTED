set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: train_1.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=<path_to_train.parquet> \
    data.val_files=<path_to_test.parquet> \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=20480 \
    data.truncation=left \
    data.multiturn.enable=true \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.0 \
    optim.clip_grad=5.0 \
    model.partial_pretrain=<path_to_pretrained_model> \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=<project_name> \
    trainer.experiment_name=<experiment_name> \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@

