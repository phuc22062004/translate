#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Computing corpus BLEU (vi -> en)..."
python3 -m viamr.scoring \
    --predict_file "results.jsonl"
