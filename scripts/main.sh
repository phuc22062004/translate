#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Full vi->en pipeline: SFT -> inference -> BLEU ==="

bash scripts/train_sft.sh
bash scripts/infer.sh
bash scripts/get_score.sh
