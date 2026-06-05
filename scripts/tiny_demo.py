"""End-to-end TINY reward-modeling demo (CPU-friendly).

  1. Load GPT-2 with a single-score head
  2. Show the UNTRAINED judge's win rate (should be ~50%, a coin flip)
  3. Train briefly with LoRA + RewardTrainer on a small subset (watch loss)
  4. Show the TRAINED judge's win rate on held-out pairs (should improve)

Run:
    python scripts/tiny_demo.py
"""

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

LIMIT = 60     # tiny subset of the dataset
EPOCHS = 1
N_EVAL = 15


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main() -> None:
    device = get_device()
    banner(f"Device: {device}  (CPU is expected for this demo)")

    model_cfg = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(model_cfg, device)

    dataset, test_pairs = prepare_data(tokenizer, DataConfig(), model_cfg, limit=LIMIT)
    print(f"Train: {len(dataset['train'])}  Test: {len(dataset['test'])}")

    banner("STEP 1/2 — UNTRAINED judge (before)")
    n = min(N_EVAL, len(test_pairs["chosen"]))
    before = win_rate(model, tokenizer, test_pairs["chosen"], test_pairs["rejected"], n=n)
    print(f">>> Win rate BEFORE: {before:.2%}  (≈50% = random)")

    banner(f"TRAINING — LoRA + RewardTrainer ({len(dataset['train'])} pairs, {EPOCHS} epoch)")
    print("Watch the 'loss' fall as the judge learns to prefer chosen answers...\n")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        train_settings=TrainSettings(num_train_epochs=EPOCHS, eval_strategy="no",
                                     logging_steps=1),
        lora_settings=LoraSettings(),
    )
    trainer.train()

    banner("STEP 2/2 — TRAINED judge (after)")
    after = win_rate(model, tokenizer, test_pairs["chosen"], test_pairs["rejected"],
                     n=n, verbose=True)

    banner("RESULT")
    print(f"Win rate BEFORE : {before:.2%}")
    print(f"Win rate AFTER  : {after:.2%}")
    losses = [round(l["loss"], 4) for l in trainer.state.log_history if "loss" in l]
    print(f"\nLoss history: {losses}")


if __name__ == "__main__":
    main()
