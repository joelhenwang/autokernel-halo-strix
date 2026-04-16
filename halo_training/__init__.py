"""
Halo Training Stack — composable training for AMD Strix Halo (gfx1151).

Usage:
    from halo_training import train
    train(model, dataset="babylm", time_budget_minutes=15, compile=True)
"""

from halo_training.trainer import train
from halo_training.optimizer import build_optimizer, build_scheduler
from halo_training.data import BabyLMDataset, build_dataloader
from halo_training.callbacks import (
    PhaseScheduler, MemoryMonitor, StateNormMonitor, PerParamGradMonitor,
)
from halo_training.metrics import compute_bpb, ThroughputTracker, TrainingLogger
from halo_training.model_utils import get_layer_iterator, count_parameters, estimate_memory
from halo_training.smoke import run_smoke_test
from halo_training.streaming import LayerStreamingTrainer
from halo_training.memory import MemoryBudget, suggest_mode
from halo_training.evaluate import evaluate_bpb, benchmark_inference
from halo_training.chat_template import ChatMLTokenizer, build_tokenizer, resize_embeddings
from halo_training.sft_data import SFTDataset
from halo_training.sft_loss import WeightedCrossEntropyLoss, build_sft_loss_fn

__all__ = [
    "train",
    "build_optimizer",
    "build_scheduler",
    "BabyLMDataset",
    "build_dataloader",
    "PhaseScheduler",
    "MemoryMonitor",
    "StateNormMonitor",
    "PerParamGradMonitor",
    "compute_bpb",
    "ThroughputTracker",
    "TrainingLogger",
    "get_layer_iterator",
    "count_parameters",
    "estimate_memory",
    "run_smoke_test",
    "LayerStreamingTrainer",
    "MemoryBudget",
    "suggest_mode",
    "evaluate_bpb",
    "benchmark_inference",
    "ChatMLTokenizer",
    "build_tokenizer",
    "resize_embeddings",
    "SFTDataset",
    "WeightedCrossEntropyLoss",
    "build_sft_loss_fn",
]
