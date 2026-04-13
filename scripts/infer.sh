#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running vi->en translation inference..."
python3 -m viamr.inference \
    --input_file "data/tst2013.jsonl" \
    --output_file "results.jsonl" \
    --model_name "outputs/Qwen-1.7B-SFT-VI2EN" \
    --max_new_tokens 512
