"""Dataset loading, formatting, filtering, tokenizing and splitting.

Pipeline (order matters so we can keep a clean held-out eval set):
    load -> format (Human/Assistant) -> filter by length -> split
         -> stash raw test strings -> tokenize -> return

Returns both the tokenized DatasetDict (for RewardTrainer) and the raw
(chosen, rejected) string pairs for the TEST split (for win-rate evaluation).
"""

from __future__ import annotations

from datasets import DatasetDict, load_dataset

from .config import DATASET_NAME, DataConfig, ModelConfig, format_pair


def add_combined_columns(example: dict) -> dict:
    """Add prompt_chosen / prompt_rejected dialogue-formatted columns."""
    example["prompt_chosen"] = format_pair(example["prompt"], example["chosen"])
    example["prompt_rejected"] = format_pair(example["prompt"], example["rejected"])
    return example


def _fits(tokenizer, text: str, max_length: int) -> bool:
    return len(tokenizer(text)["input_ids"]) <= max_length


def make_preprocess_fn(tokenizer, max_length: int):
    """Build the tokenization function producing the 4 RewardTrainer fields."""

    def preprocess_function(examples: dict) -> dict:
        tok_chosen = tokenizer(
            examples["prompt_chosen"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tok_rejected = tokenizer(
            examples["prompt_rejected"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids_chosen": tok_chosen["input_ids"],
            "attention_mask_chosen": tok_chosen["attention_mask"],
            "input_ids_rejected": tok_rejected["input_ids"],
            "attention_mask_rejected": tok_rejected["attention_mask"],
        }

    return preprocess_function


def prepare_data(
    tokenizer,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    limit: int | None = None,
):
    """Return (tokenized_dataset_dict, test_pairs).

    test_pairs is a dict {"chosen": [...str...], "rejected": [...str...]}
    holding the raw TEST-split strings for win-rate evaluation.
    """
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    max_length = model_cfg.max_length

    ds = load_dataset(DATASET_NAME)
    train = ds["train"]

    if limit is not None:
        train = train.select(range(min(limit, len(train))))

    # Format into Human/Assistant dialogue.
    train = train.map(add_combined_columns)

    # Filter: keep pairs where BOTH sides fit in max_length tokens.
    if data_cfg.filter_by_tokens:
        train = train.filter(
            lambda ex: _fits(tokenizer, ex["prompt_chosen"], max_length)
            and _fits(tokenizer, ex["prompt_rejected"], max_length)
        )

    # Split BEFORE tokenizing so the test strings stay intact for evaluation.
    split = train.train_test_split(test_size=data_cfg.test_size, seed=data_cfg.seed)

    # Stash raw test strings (used by the win-rate evaluator).
    test_pairs = {
        "chosen": list(split["test"]["prompt_chosen"]),
        "rejected": list(split["test"]["prompt_rejected"]),
    }

    # Tokenize both splits, dropping raw text columns.
    preprocess = make_preprocess_fn(tokenizer, max_length)
    remove_cols = [
        "prompt", "chosen", "rejected", "prompt_chosen", "prompt_rejected",
    ]
    tokenized = DatasetDict(
        train=split["train"].map(preprocess, batched=True, remove_columns=remove_cols),
        test=split["test"].map(preprocess, batched=True, remove_columns=remove_cols),
    )

    return tokenized, test_pairs
