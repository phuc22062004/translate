#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running GRPO training (vi -> en)..."
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 -m viamr.training.grpo \
    --dataset1_path "data/train.jsonl" \
    --output_dir "outputs/Qwen-1.7B-GRPO-VI2EN" \
    --model_name "outputs/Qwen-1.7B-SFT-VI2EN" \
    --learning_rate 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --num_train_epochs 2 \
    --save_steps 200 \
    --log_on_each_node \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --use_lora 0 \
    --wandb_project "vi2en-translation" \
    --wandb_run_name "grpo-bleu" \
    2>&1 | tee Qwen-GRPO-VI2EN.log
