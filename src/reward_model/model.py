"""Model + tokenizer setup and LoRA configuration for reward modeling."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from .config import LoraSettings, ModelConfig, get_device


def load_model_and_tokenizer(
    cfg: ModelConfig | None = None,
    device: torch.device | None = None,
):
    """Load GPT-2 as a sequence-classification model with a single-score head.

    num_labels=1 turns GPT-2's text head into a "scoreboard": text in -> one
    scalar reward out. GPT-2 has no pad token, so we reuse the EOS token.
    """
    cfg = cfg or ModelConfig()
    device = device or get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = GPT2ForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    model.to(device)
    return model, tokenizer


def build_lora_config(settings: LoraSettings | None = None) -> LoraConfig:
    """LoRA config for a sequence-classification (reward) task."""
    settings = settings or LoraSettings()
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.target_modules),
    )


def load_trained_model(model_dir: str, cfg: ModelConfig | None = None,
                       device: torch.device | None = None):
    """Load a previously trained reward model from a directory."""
    cfg = cfg or ModelConfig()
    device = device or get_device()
    model = GPT2ForSequenceClassification.from_pretrained(
        model_dir, num_labels=cfg.num_labels
    ).to(device)
    return model
