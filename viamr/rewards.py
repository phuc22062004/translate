"""Reward function for vi→en translation GRPO training (sentence BLEU)."""
from sacrebleu.metrics import BLEU

_bleu = BLEU(effective_order=True)


def sentence_bleu(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    score = _bleu.sentence_score(hypothesis.strip(), [reference.strip()])
    return score.score / 100.0


def bleu_reward(prompts, completions, answers, **kwargs) -> list[float]:
    """GRPO reward: sentence-level BLEU between completion and gold English."""
    scores = []
    for completion, gold in zip(completions, answers):
        hyp = completion[0]["content"].strip()
        ref = gold.strip()
        score = sentence_bleu(hyp, ref)
        scores.append(score)
        print(f"BLEU: {score:.4f} | pred: {hyp[:80]} | gold: {ref[:80]}")
    return scores
