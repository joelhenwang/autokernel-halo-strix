"""Main training loop for Halo Training Stack (Mode A: direct training)."""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from halo_training.data import BabyLMDataset, build_dataloader
from halo_training.metrics import compute_bpb, ThroughputTracker, TrainingLogger
from halo_training.optimizer import build_optimizer, build_scheduler


def train(
    model: nn.Module,
    dataset: Union[str, Dataset] = "babylm",
    epochs: int = 1,
    time_budget_minutes: float = 45.0,
    batch_size: int = 64,
    block_size: int = 1024,
    accum_steps: int = 4,
    base_lr: float = 8e-4,
    param_groups: Optional[List[dict]] = None,
    loss_fn: Optional[Callable] = None,
    callbacks: Optional[List[Callable]] = None,
    compile: bool = False,
    optimize_kernels: bool = False,
    gradient_checkpointing: bool = False,
    max_grad_norm: float = 1.0,
    label_smoothing: float = 0.0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    log_interval: int = 10,
    num_workers: int = 4,
    mode: str = "auto",
    checkpoint_every: int = 2,
    use_muon: bool = False,
    use_bf16: bool = False,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a model using Mode A (direct) or Mode B (layer-streaming).

    Args:
        model: Any nn.Module.
        dataset: "babylm", path string, or a torch Dataset instance.
        epochs: Number of passes over the dataset.
        time_budget_minutes: Wall-clock limit in minutes (default 45).
        batch_size: Microbatch size (default 64, playbook recommends 64-128).
        block_size: Sequence length for tokenization.
        accum_steps: Gradient accumulation steps (effective batch = batch_size * accum_steps).
        base_lr: Base learning rate (8e-4 default from COOKBOOK.md).
        param_groups: Custom param groups (overrides COOKBOOK.md factory).
        loss_fn: Custom loss function(output, batch) -> loss. Default: cross-entropy.
        callbacks: List of callables(model, step) invoked each step.
        compile: Apply torch.compile to model (NOT optimizer).
        optimize_kernels: Apply autokernel.optimize() for HIP fusions.
        gradient_checkpointing: Enable gradient checkpointing to save memory.
        max_grad_norm: Gradient clipping max norm.
        label_smoothing: Label smoothing for cross-entropy loss.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_interval: Save checkpoint every N optimizer steps (default: log_interval * 10).
        log_interval: Log every N steps.
        num_workers: DataLoader workers.
        use_bf16: Use bfloat16 instead of float16 for mixed precision.
                  bf16 has same exponent range as fp32 — no GradScaler needed.
        resume_from: Path to checkpoint for continued pre-training (Approach B).
                     Loads model weights only — optimizer and LR schedule are fresh.

    Returns:
        Dict with training stats (final_loss, tok_s, steps, etc.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    callbacks = callbacks or []

    # --- Auto-detect mode ---
    if mode == "auto":
        from halo_training.memory import suggest_mode
        mode = suggest_mode(model, batch_size, block_size)
        print(f"Auto-detected training mode: {mode}")

    use_streaming = (mode == "B")
    streamer = None

    # --- Setup model ---
    model = model.to(device)

    # --- Continued pre-training: load weights from checkpoint (fresh optimizer) ---
    prev_tokens = 0
    if resume_from:
        print(f"Loading checkpoint for continued pre-training: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
            print("  Warning: loaded with strict=False (some keys may not match)")
        prev_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        prev_tokens = ckpt.get("total_tokens", 0) if isinstance(ckpt, dict) else 0
        print(f"  Resumed from step {prev_step} ({prev_tokens:,} prev tokens)")
        print(f"  Fresh optimizer + LR schedule (Approach B warm restart)")
        del ckpt, state_dict

    model.train()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    if optimize_kernels:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print("autokernel optimizations applied")
        except Exception as e:
            print(f"autokernel.optimize() failed ({e}), continuing without")

    if use_streaming:
        from halo_training.streaming import LayerStreamingTrainer
        streamer = LayerStreamingTrainer(
            model,
            checkpoint_every=checkpoint_every,
            compile_layers=compile,
        )
    elif compile:
        # Use mode="default" when autokernel patterns are active — CUDAGraphs in
        # "reduce-overhead" conflict with HIP kernel replacement modules.
        # Benchmarked: "default" vs "reduce-overhead" gives identical throughput
        # for SSM models (8258 vs 8278 tok/s) because chunked scan dominates.
        compile_mode = "default" if optimize_kernels else "reduce-overhead"
        print(f"Compiling model with torch.compile ({compile_mode})...")
        model = torch.compile(model, mode=compile_mode)

    # --- Setup data ---
    if isinstance(dataset, str):
        if dataset == "babylm":
            dataset = BabyLMDataset(block_size=block_size)
        else:
            dataset = BabyLMDataset(root=dataset, block_size=block_size)

    # Vocab size check: clamp token IDs to model's vocab size if needed
    model_vocab = _get_vocab_size(model)
    if model_vocab and hasattr(dataset, "vocab_size") and dataset.vocab_size > model_vocab:
        print(f"WARNING: tokenizer vocab ({dataset.vocab_size}) > model vocab ({model_vocab}), "
              f"clamping token IDs")
        dataset.tokens = dataset.tokens.clamp(max=model_vocab - 1)

    dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # --- Setup optimizer (NEVER compile this) ---
    optimizer = build_optimizer(model, base_lr=base_lr, use_muon=use_muon)

    total_steps = len(dataloader) * epochs // accum_steps
    scheduler = build_scheduler(optimizer, total_steps)

    # --- Setup loss ---
    chunked_ce = None
    if loss_fn is None:
        # Try chunked CE for memory-efficient LM head backward when using optimized kernels
        if optimize_kernels and hasattr(model, "lm_head"):
            try:
                from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
                chunked_ce = ChunkedLinearCrossEntropyLoss(chunk_size=1024)
                print("Using chunked linear cross-entropy (saves ~300MB, 25% faster LM head backward)")
            except Exception:
                pass

        ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        def default_loss_fn(output, batch):
            _, targets = batch
            targets = targets.to(device)
            if isinstance(output, torch.Tensor):
                logits = output
            elif hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output
            return ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss_fn = default_loss_fn

    # --- Setup mixed precision ---
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # bf16 doesn't need loss scaling (same exponent range as fp32)
    use_scaler = (device.type == "cuda") and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # --- Training loop ---
    model.train()
    global_step = 0
    total_tokens = 0
    running_loss = 0.0
    start_time = time.time()
    deadline = start_time + time_budget_minutes * 60 if time_budget_minutes else None
    best_loss = float("inf")

    stats = {
        "steps": 0,
        "tokens": 0,
        "final_loss": float("inf"),
        "tok_s": 0.0,
        "peak_memory_gb": 0.0,
    }

    # --- Setup metrics ---
    n_params = sum(p.numel() for p in model.parameters())
    throughput = ThroughputTracker(n_params)
    throughput.start()
    logger = TrainingLogger(
        log_file=os.path.join(checkpoint_dir, "train_log.jsonl") if checkpoint_dir else None
    )

    eff_batch = batch_size * accum_steps
    print(f"Training: {_count_params(model):.1f}M params, batch={batch_size}x{accum_steps}={eff_batch}, "
          f"block={block_size}, lr={base_lr}")
    print(f"Time budget: {time_budget_minutes:.0f} min | "
          f"Steps/epoch: {len(dataloader)}, optimizer steps: {total_steps}")

    try:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Check time budget
                if deadline and time.time() > deadline:
                    print(f"Time budget reached at step {global_step}")
                    break

                # Callbacks (phase scheduling, memory monitoring, etc.)
                for cb in callbacks:
                    cb(model, global_step)

                # Forward + backward with mixed precision
                input_ids, targets = batch
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if streamer is not None:
                        output = streamer.forward(input_ids)
                    else:
                        # Pass targets if model accepts them (adaptive head)
                        output = model(input_ids, targets=targets)

                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        # Model returned scalar loss (adaptive head with chunked CE)
                        loss = output / accum_steps
                    elif chunked_ce is not None and isinstance(output, torch.Tensor) and output.dim() >= 2:
                        # Chunked CE: compute loss directly from hidden/logits + lm_head weight
                        # If output has vocab-sized last dim, it's logits (use standard CE)
                        # If output has hidden-sized last dim, use chunked CE with lm_head
                        vocab_size = getattr(model, "vocab_size", None) or model.lm_head.weight.shape[0]
                        if output.shape[-1] != vocab_size and hasattr(model, "lm_head"):
                            # Output is hidden states — use chunked CE
                            loss = chunked_ce(
                                output.view(-1, output.shape[-1]),
                                model.lm_head.weight,
                                targets.view(-1),
                            ) / accum_steps
                        else:
                            loss = loss_fn(output, batch) / accum_steps
                    else:
                        loss = loss_fn(output, batch) / accum_steps

                scaler.scale(loss).backward()

                tokens_in_batch = input_ids.numel()
                total_tokens += tokens_in_batch
                throughput.update(tokens_in_batch)
                running_loss += loss.item() * accum_steps

                # Optimizer step (every accum_steps)
                if (batch_idx + 1) % accum_steps == 0:
                    # cudagraph step boundary — required for torch.compile(mode="reduce-overhead")
                    # with gradient accumulation. Without this, graph capture silently degrades.
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except (AttributeError, RuntimeError):
                        pass  # older PyTorch or no compile — safe to skip

                    # Unscale for gradient clipping
                    scaler.unscale_(optimizer)

                    # Compute pre-clip grad norm (for diagnostics — clipping alone masks instability)
                    pre_clip_grad = _grad_norm(model)

                    # Gradient clipping
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )

                    # Skip step if non-finite gradients
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                    else:
                        print(f"[step {global_step}] Non-finite grad norm, skipping")

                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    # Logging
                    if global_step % log_interval == 0:
                        elapsed = time.time() - start_time
                        tp = throughput.get_stats()
                        avg_loss = running_loss / log_interval
                        bpb = compute_bpb(avg_loss)
                        lr = scheduler.get_last_lr()[0]
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

                        print(
                            f"[step {global_step:>5d}] "
                            f"loss={avg_loss:.4f} "
                            f"bpb={bpb:.3f} "
                            f"lr={lr:.2e} "
                            f"grad={grad_norm:.2f} "
                            f"tok/s={tp['tok_s']:,.0f} "
                            f"mfu={tp['mfu']:.1%} "
                            f"mem={mem_gb:.1f}GB"
                        )

                        logger.log(
                            step=global_step, loss=avg_loss, bpb=bpb,
                            lr=lr, grad_norm=grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                            tok_s=tp["tok_s"], mfu=tp["mfu"], mem_gb=mem_gb,
                        )

                        running_loss = 0.0
                        best_loss = min(best_loss, avg_loss)

                    # Checkpoint
                    ckpt_every = checkpoint_interval or (log_interval * 10)
                    if checkpoint_dir and global_step % ckpt_every == 0:
                        _save_checkpoint(model, optimizer, global_step, checkpoint_dir, total_tokens)

            else:
                continue
            break  # Time budget reached

    except KeyboardInterrupt:
        print(f"\nInterrupted at step {global_step}")

    # Final stats
    elapsed = time.time() - start_time
    stats["steps"] = global_step
    stats["tokens"] = total_tokens
    stats["final_loss"] = best_loss
    stats["tok_s"] = total_tokens / elapsed if elapsed > 0 else 0
    stats["elapsed_s"] = elapsed
    if device.type == "cuda":
        stats["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nDone: {global_step} steps, {total_tokens:,} tokens in {elapsed:.0f}s "
          f"({stats['tok_s']:,.0f} tok/s), best loss={best_loss:.4f}")

    return stats


def _count_params(model: nn.Module) -> float:
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _get_vocab_size(model: nn.Module) -> Optional[int]:
    """Try to detect model's vocabulary size from embedding layer."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module.num_embeddings
    return None


def _grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm (pre-clip) for diagnostics."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total ** 0.5


def _save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    step: int, checkpoint_dir: str, total_tokens: int = 0,
):
    """Save model and optimizer state."""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_tokens": total_tokens,
    }, path)
    print(f"Checkpoint saved: {path}")
