"""CLI to compute corpus BLEU between predicted and gold translations."""
import argparse
import json

from sacrebleu.metrics import BLEU


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(args: argparse.Namespace) -> None:
    preds_rows = _load_jsonl(args.predict_file)
    preds = [r["pred"].strip() for r in preds_rows]

    if args.gold_file:
        gold_rows = _load_jsonl(args.gold_file)
        refs = [r["output"].strip() for r in gold_rows]
    else:
        refs = [r["gold"].strip() for r in preds_rows if "gold" in r]

    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]
    print(f"Number of predictions: {len(preds)}, Number of references: {len(refs)}")

    bleu = BLEU()
    score = bleu.corpus_score(preds, [refs])
    print(f"Corpus BLEU: {score.score:.4f}")
    print(str(score))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute corpus BLEU for vi→en translation.")
    parser.add_argument("--predict_file", type=str, required=True,
                        help="JSONL with at least a `pred` field (and optional `gold`).")
    parser.add_argument("--gold_file", type=str, default=None,
                        help="Optional JSONL with `output` field. If omitted, `gold` is read from --predict_file.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
