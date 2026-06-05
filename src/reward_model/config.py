"""Central configuration for the reward-modeling pipeline.

All tunable knobs live here. Values mirror the original lab defaults; device
is auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Hugging Face pairwise preference dataset used by the lab.
DATASET_NAME = "Dahoas/synthetic-instruct-gptj-pairwise"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    num_labels: int = 1          # single scalar score (the "scoreboard head")
    max_length: int = 1024       # GPT-2's max context length


@dataclass
class DataConfig:
    test_size: float = 0.2
    seed: int = 42
    # Keep only pairs where BOTH responses fit within max_length *tokens*.
    # (The original lab filtered on character length, which is inconsistent
    # with the token-based truncation — fixed here.)
    filter_by_tokens: bool = True


@dataclass
class LoraSettings:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # GPT-2 attention projection layers.
    target_modules: tuple[str, ...] = ("attn.c_attn", "attn.c_proj")


@dataclass
class TrainSettings:
    output_dir: str = str(OUTPUT_DIR / "model_output")
    per_device_train_batch_size: int = 3
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.41e-5
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    # fp16 only helps on GPU; auto-disabled on CPU.
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())


# Dialogue format applied to every prompt/response pair.
def format_pair(prompt: str, response: str) -> str:
    return f"\n\nHuman: {prompt}\n\nAssistant: {response}"
