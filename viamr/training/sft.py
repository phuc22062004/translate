import argparse
import os

import torch
from trl import SFTConfig, SFTTrainer

from ..dataset import get_data
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="sft")
    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    training_args = SFTConfig(
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

        bf16=use_bf16,
        fp16=not use_bf16,

        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        report_to="none",

        max_length=args.max_input_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for vi→en translation")
    add_common_args(parser)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--eval_steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())