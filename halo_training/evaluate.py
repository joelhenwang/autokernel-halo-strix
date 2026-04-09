"""Evaluation utilities: BPB validation, inference benchmarks, quick probes.

Targets from llm_engineer playbook:
- Primary: val_bpb = (CE_loss / ln2) / bytes_per_token
- Quick probes: HellaSwag 0-shot, ARC-Easy 0-shot (via lm-evaluation-harness)
- Inference: decode > 30 tok/s, prefill(512) < 200ms, VRAM < 6 GB
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from halo_training.data import BabyLMDataset, build_dataloader
from halo_training.metrics import compute_bpb


def evaluate_bpb(
    model: nn.Module,
    dataset_path: str = "datasets/babylm-strict-small",
    block_size: int = 1024,
    batch_size: int = 16,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Evaluate bits-per-byte on a dataset.

    Uses the last 10% of the dataset as validation split.
    """
    device = next(model.parameters()).device
    model.eval()

    ds = BabyLMDataset(root=dataset_path, block_size=block_size)

    # Clamp vocab
    model_vocab = _get_vocab_size(model)
    if model_vocab and ds.vocab_size > model_vocab:
        ds.tokens = ds.tokens.clamp(max=model_vocab - 1)

    # Use last 10% as validation
    n_val = max(1, len(ds) // 10)
    val_indices = list(range(len(ds) - n_val, len(ds)))
    val_ds = torch.utils.data.Subset(ds, val_indices)
    val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(input_ids)
                if isinstance(output, torch.Tensor):
                    logits = output
                elif hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output

                loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            n_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    bpb = compute_bpb(avg_loss)

    return {
        "val_loss": avg_loss,
        "val_bpb": bpb,
        "val_tokens": total_tokens,
        "val_batches": n_batches,
    }


def benchmark_inference(
    model: nn.Module,
    vocab_size: int = 32000,
    seq_len: int = 512,
    decode_tokens: int = 32,
    warmup_steps: int = 5,
    measure_steps: int = 20,
) -> Dict[str, float]:
    """Benchmark inference: prefill latency and decode throughput.

    Targets from playbook:
    - decode > 30 tok/s
    - prefill(512) < 200ms
    - VRAM < 6 GB
    """
    device = next(model.parameters()).device
    model.eval()

    # Prefill benchmark
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_steps):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model(input_ids)
        torch.cuda.synchronize()

        # Measure prefill
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model(input_ids)
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) / measure_steps * 1000

    # Decode benchmark (autoregressive, token by token)
    with torch.no_grad():
        prompt = torch.randint(0, vocab_size, (1, 16), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        current = prompt
        for _ in range(decode_tokens):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(current)
            if isinstance(logits, torch.Tensor):
                next_token = logits[:, -1:, :].argmax(dim=-1)
            else:
                next_token = logits.logits[:, -1:, :].argmax(dim=-1)
            current = torch.cat([current, next_token], dim=1)

        torch.cuda.synchronize()
        decode_time = time.perf_counter() - t0
        decode_tok_s = decode_tokens / decode_time

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    results = {
        "prefill_ms": prefill_ms,
        "prefill_seq_len": seq_len,
        "decode_tok_s": decode_tok_s,
        "decode_tokens": decode_tokens,
        "peak_memory_gb": peak_mem_gb,
    }

    # Check against targets
    results["prefill_pass"] = prefill_ms < 200
    results["decode_pass"] = decode_tok_s > 30
    results["memory_pass"] = peak_mem_gb < 6.0

    print(f"Prefill ({seq_len} tokens): {prefill_ms:.1f} ms {'PASS' if results['prefill_pass'] else 'FAIL'}")
    print(f"Decode: {decode_tok_s:.1f} tok/s {'PASS' if results['decode_pass'] else 'FAIL'}")
    print(f"Peak memory: {peak_mem_gb:.1f} GB {'PASS' if results['memory_pass'] else 'FAIL'}")

    return results


def _get_vocab_size(model: nn.Module) -> Optional[int]:
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return m.num_embeddings
    return None
