"""Evaluate a trained reward model: score a pair and/or compute win rate.

Examples:
    # Win rate on a held-out subset using a saved model directory
    python scripts/evaluate_reward.py --model-dir outputs/model_output --n 20

    # Compare two ad-hoc texts
    python scripts/evaluate_reward.py --model-dir outputs/model_output \
        --text-a "Human: hi\n\nAssistant: Hello! How can I help?" \
        --text-b "Human: hi\n\nAssistant: no."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reward_model.config import DataConfig, ModelConfig, get_device  # noqa: E402
from reward_model.data import prepare_data  # noqa: E402
from reward_model.evaluate import compare, win_rate  # noqa: E402
from reward_model.model import load_model_and_tokenizer, load_trained_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reward model")
    parser.add_argument("--model-dir", required=True,
                        help="Directory of a trained reward model.")
    parser.add_argument("--base-model", default=ModelConfig().model_name,
                        help="Base model name (for the tokenizer).")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--text-a", default=None)
    parser.add_argument("--text-b", default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Tokenizer comes from the base model; weights from the trained dir.
    _, tokenizer = load_model_and_tokenizer(ModelConfig(model_name=args.base_model), device)
    model = load_trained_model(args.model_dir, device=device)

    if args.text_a and args.text_b:
        winner, sa, sb = compare(model, tokenizer, args.text_a, args.text_b)
        print(f"score A: {sa:+.3f}\nscore B: {sb:+.3f}")
        print(f"WINNER: {'A' if winner == args.text_a else 'B'}")
        return

    # Otherwise compute win rate on a held-out subset of the dataset.
    _, test_pairs = prepare_data(tokenizer, DataConfig(), ModelConfig(model_name=args.base_model))
    acc = win_rate(model, tokenizer, test_pairs["chosen"], test_pairs["rejected"],
                   n=args.n, verbose=True)
    print(f"\nWin rate (test, n={args.n}): {acc:.2%}")


if __name__ == "__main__":
    main()
