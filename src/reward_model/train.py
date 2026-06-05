"""RewardTrainer setup for pairwise reward modeling."""

from __future__ import annotations

from transformers import TrainingArguments
from trl import RewardTrainer

from .config import LoraSettings, TrainSettings
from .model import build_lora_config


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    train_settings: TrainSettings | None = None,
    lora_settings: LoraSettings | None = None,
) -> RewardTrainer:
    """Configure a TRL RewardTrainer.

    RewardTrainer implements the Bradley-Terry pairwise loss:
        loss = -log( sigmoid( reward_chosen - reward_rejected ) )
    pushing the chosen response's score above the rejected one's.
    """
    train_settings = train_settings or TrainSettings()

    training_args = TrainingArguments(
        per_device_train_batch_size=train_settings.per_device_train_batch_size,
        num_train_epochs=train_settings.num_train_epochs,
        gradient_accumulation_steps=train_settings.gradient_accumulation_steps,
        learning_rate=train_settings.learning_rate,
        output_dir=train_settings.output_dir,
        logging_steps=train_settings.logging_steps,
        eval_strategy=train_settings.eval_strategy,
        eval_steps=train_settings.eval_steps,
        save_steps=train_settings.save_steps,
        save_total_limit=train_settings.save_total_limit,
        fp16=train_settings.fp16,
    )

    return RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(lora_settings),
    )
