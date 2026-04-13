"""Supervised fine-tuning entrypoint for Vietnamese→English translation."""
import argparse
import inspect
import os

import torch
from trl import SFTConfig, SFTTrainer

from ..dataset import get_data
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only kwargs that the target class's __init__ actually accepts."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="sft")
    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    sft_config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_total_limit=2,
        report_to="none",
        completion_only_loss=True,
        deepspeed=args.deepspeed_path,
        max_length=args.max_input_length,
        max_seq_length=args.max_input_length,
    )
    training_args = SFTConfig(**_filter_kwargs(SFTConfig, sft_config_kwargs))

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
    )
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for vi→en translation")
    add_common_args(parser)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--eval_steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
