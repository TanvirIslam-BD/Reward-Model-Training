"""Reward modeling with GPT-2, LoRA and TRL's RewardTrainer.

A modular, runnable adaptation of the IBM Skills Network "Reward Modeling" lab.
Trains a reward model (a "judge") that scores responses, learning from pairs of
(chosen, rejected) responses so it assigns higher scores to preferred answers.
Uses the Bradley-Terry pairwise loss under the hood via TRL's RewardTrainer.
"""

__version__ = "1.0.0"
