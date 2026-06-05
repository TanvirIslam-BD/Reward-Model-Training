"""Scoring and win-rate evaluation for a trained reward model."""

from __future__ import annotations

import torch

from .config import get_device


def score_text(model, tokenizer, text: str, max_length: int = 512,
               device: torch.device | None = None) -> float:
    """Return the scalar reward (logit) the model assigns to one text."""
    device = device or get_device()
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze().item()


def compare(model, tokenizer, text_a: str, text_b: str, **kw) -> tuple[str, float, float]:
    """Score two texts; return (winner_text, score_a, score_b)."""
    score_a = score_text(model, tokenizer, text_a, **kw)
    score_b = score_text(model, tokenizer, text_b, **kw)
    winner = text_a if score_a > score_b else text_b
    return winner, score_a, score_b


def win_rate(model, tokenizer, chosen: list[str], rejected: list[str],
             n: int | None = None, verbose: bool = False, **kw) -> float:
    """Fraction of pairs where the model scores `chosen` above `rejected`.

    This is the reward model's accuracy at agreeing with human preference.
    Evaluate on a HELD-OUT set for an honest number.
    """
    n = len(chosen) if n is None else min(n, len(chosen))
    correct = 0
    for i in range(n):
        winner, s_chosen, s_rejected = compare(
            model, tokenizer, chosen[i], rejected[i], **kw
        )
        if winner == chosen[i]:
            correct += 1
        if verbose:
            mark = "OK " if winner == chosen[i] else "X  "
            print(f"[{mark}] chosen={s_chosen:+.3f}  rejected={s_rejected:+.3f}")
    return correct / n if n else 0.0
