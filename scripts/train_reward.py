"""Train a reward model end-to-end with LoRA + RewardTrainer.

Examples:
    python scripts/train_reward.py --tiny --epochs 1
    python scripts/train_reward.py --epochs 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reward_model.config import (  # noqa: E402
    DataConfig,
    LoraSettings,
    ModelConfig,
    TrainSettings,
    get_device,
)
from reward_model.data import prepare_data  # noqa: E402
from reward_model.evaluate import win_rate  # noqa: E402
from reward_model.model import load_model_and_tokenizer  # noqa: E402
from reward_model.train import build_trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reward model")
    parser.add_argument("--model", default=ModelConfig().model_name)
    parser.add_argument("--epochs", type=int, default=TrainSettings().num_train_epochs)
    parser.add_argument("--lora-r", type=int, default=LoraSettings().r)
    parser.add_argument("--tiny", action="store_true",
                        help="Use a tiny subset (CPU smoke test).")
    parser.add_argument("--save-dir", default=str(Path("outputs") / "model_output"))
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_cfg = ModelConfig(model_name=args.model)
    model, tokenizer = load_model_and_tokenizer(model_cfg, device)

    limit = 40 if args.tiny else None
    dataset, test_pairs = prepare_data(tokenizer, DataConfig(), model_cfg, limit=limit)
    print(f"Train: {len(dataset['train'])}  Test: {len(dataset['test'])}")

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        train_settings=TrainSettings(num_train_epochs=args.epochs),
        lora_settings=LoraSettings(r=args.lora_r),
    )

    print("Starting training...")
    trainer.train()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    print(f"Model saved to {save_dir}")

    # Honest evaluation on the held-out test split.
    print("\nEvaluating win rate on held-out test set...")
    acc = win_rate(model, tokenizer, test_pairs["chosen"], test_pairs["rejected"],
                   n=min(20, len(test_pairs["chosen"])), verbose=True)
    print(f"\nWin rate (test): {acc:.2%}")


if __name__ == "__main__":
    main()
