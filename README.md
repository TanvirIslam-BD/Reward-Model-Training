# Reward Model Training (GPT-2 + LoRA + TRL RewardTrainer)

It trains a **reward model** — a "judge" that reads a (prompt, response) and
outputs a single **score** for how good the response is — by learning from pairs of
**chosen** (preferred) vs **rejected** responses.

This is the stage *after* instruction tuning in the RLHF pipeline:

```
1. SFT / Instruction tuning   → teach the model to follow instructions
2. Reward modeling  (THIS)    → train a judge from human preferences
3. RLHF / PPO  or  DPO        → use the judge to improve the chat model
```

## How it works

1. **Data** — `Dahoas/synthetic-instruct-gptj-pairwise`: each row has a `prompt`,
   a `chosen` response, and a `rejected` response.
2. **Format** — wrap each into `\n\nHuman: ... \n\nAssistant: ...`.
3. **Model** — `GPT2ForSequenceClassification(num_labels=1)`: GPT-2 with its text
   head replaced by a single-number **score head**.
4. **LoRA** — efficient fine-tuning of only the attention projections
   (`attn.c_attn`, `attn.c_proj`).
5. **Train** — TRL's `RewardTrainer` optimizes the **Bradley-Terry pairwise loss**
   `-log σ(reward_chosen − reward_rejected)`, pushing chosen scores above rejected.
6. **Evaluate** — **win rate**: the fraction of held-out pairs where the model
   scores the chosen response higher than the rejected one.

## Project layout

```
src/reward_model/
  config.py     # all settings (paths, model, LoRA, training)
  data.py       # load / format / filter / tokenize / split (+ keep raw test strings)
  model.py      # GPT-2 score head, LoRA config, load trained model
  train.py      # RewardTrainer wiring (Bradley-Terry loss)
  evaluate.py   # score_text, compare, win_rate
scripts/
  train_reward.py     # end-to-end training + held-out win rate
  evaluate_reward.py  # win rate, or compare two ad-hoc texts
  tiny_demo.py        # before/after win-rate demo on a tiny subset (CPU)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

# Install PyTorch first (pick ONE)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # CUDA

pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Tiny CPU smoke test: see win rate before vs. after a short training run
python scripts/tiny_demo.py

# Train (use a GPU for the full dataset!)
python scripts/train_reward.py --epochs 3

# Evaluate a saved model's win rate on held-out pairs
python scripts/evaluate_reward.py --model-dir outputs/model_output --n 20

# Compare two responses directly
python scripts/evaluate_reward.py --model-dir outputs/model_output \
    --text-a "\n\nHuman: hi\n\nAssistant: Hello! How can I help you today?" \
    --text-b "\n\nHuman: hi\n\nAssistant: no."
```

## What this version fixes vs. the original lab

The lab is a great teaching notebook but had a few rough edges. This project fixes them:

| Lab issue | Fix here |
|-----------|----------|
| Evaluated win rate on **training** data (leakage) | Evaluates on a **held-out test split** (`data.py` keeps raw test strings) |
| Length filter compared **characters** to a **token** limit | Filters by actual **token length** (`_fits`) |
| Inconsistent max_length (train 1024 / eval 512) | Centralized in `config.py` |
| Pairwise print bug (printed `text2` for both winner and loser) | Correct `compare` / `win_rate` logic |
| `trainer.train()` commented out; ships pretrained | Actually trains (`--tiny` for a fast CPU run) |

## Notes

- **CPU is slow**: GPT-2 + padding to 1024 tokens is heavy. Use `--tiny` /
  `tiny_demo.py`, which subset the data and shorten training.
- **fp16** auto-disabled on CPU (`TrainSettings.fp16`).
- **Pad token**: GPT-2 has none, so we set `pad_token = eos_token`.
- **Pinned versions** match the lab (`trl 0.11`, `transformers 4.43.4`,
  `peft 0.14.0`). Keep this env separate from the instruction-tuning project,
  which used older pins.


Dataset:
[Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise).
