import argparse

import torch
import wandb
from trl import GRPOConfig, GRPOTrainer

from ..dataset import get_data
from ..rewards import bleu_reward
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def main(args: argparse.Namespace) -> None:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="grpo")
    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        log_on_each_node=args.log_on_each_node,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.6,
        report_to="wandb",
        deepspeed=args.deepspeed_path,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[bleu_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for vi→en translation")
    add_common_args(parser)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--log_on_each_node", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vi2en-translation")
    parser.add_argument("--wandb_run_name", type=str, default="grpo-run")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
