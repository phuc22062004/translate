"""GRPO training entrypoint for Vietnamese→English translation (BLEU reward)."""
import argparse
import inspect
import os

import torch
from trl import GRPOConfig, GRPOTrainer

from ..dataset import get_data
from ..rewards import bleu_reward
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def _filter_kwargs(cls, kwargs: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def main(args: argparse.Namespace) -> None:
    if args.wandb_project and os.environ.get("WANDB_MODE") != "disabled":
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        report_to = "wandb"
    else:
        report_to = "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="grpo")
    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    grpo_config_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
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
        report_to=report_to,
    )
    if args.deepspeed_path:
        grpo_config_kwargs["deepspeed"] = args.deepspeed_path
    training_args = GRPOConfig(**_filter_kwargs(GRPOConfig, grpo_config_kwargs))

    trainer_kwargs = dict(
        model=model,
        reward_funcs=[bleu_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    grpo_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in grpo_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in grpo_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

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
