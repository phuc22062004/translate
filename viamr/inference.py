"""Inference pipeline for Vietnamese→English translation."""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import SYSTEM_PROMPT, build_user_prompt


class Translator:
    def __init__(self, model_name: str, lora_path: str | None = None, device: str = "cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    @torch.no_grad()
    def translate(self, amr: str, sentence: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(amr, sentence)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        trimmed = generated[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()


def _read_inputs(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                rows.append(json.loads(line))
            else:
                rows.append({"vi": line})
    return rows


def main(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    rows = _read_inputs(args.input_file)
    model = Translator(model_name=args.model_name, lora_path=args.lora_path)

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            src = row["vi"].replace("_", " ").strip()
            amr = row.get("input", "").strip()
            pred = model.translate(amr, src, max_new_tokens=args.max_new_tokens)

            record = {"id": idx, "vi": src, "amr": amr, "pred": pred}
            if "output" in row:
                record["gold"] = row["output"]
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"[{idx}] {src} -> {pred}")

    print(f"Save completed. Results saved to {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vi→en translation inference.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
