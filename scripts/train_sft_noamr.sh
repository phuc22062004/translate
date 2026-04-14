#!/bin/bash
set -e

# Run from the repo root so `viamr` is importable.
cd "$(dirname "$0")/.."

echo "Running SFT training (vi -> en, NO AMR baseline)..."
export CUDA_VISIBLE_DEVICES=0,1

python -m viamr.training.sft \
    --dataset1_path "data/train.jsonl" \
    --output_dir "outputs/Qwen-1.7B-SFT-VI2EN-NoAMR" \
    --model_name "Qwen/Qwen3-1.7B" \
    --learning_rate 2e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_input_length 1024 \
    --num_train_epochs 3 \
    --save_steps 500 \
    --eval_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --use_lora 0 \
    --use_amr 0 \
    2>&1 | tee Qwen-SFT-VI2EN-NoAMR.log
