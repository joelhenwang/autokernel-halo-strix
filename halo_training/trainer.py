"""Main training loop for Halo Training Stack (Mode A: direct training)."""

import os
import time

# Persist torch.compile + autotuning cache across runs (10+ min compile → instant on reuse)
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))
os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_CACHE", "1")
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from halo_training.data import BabyLMDataset, build_dataloader
from halo_training.metrics import compute_bpb, ThroughputTracker, TrainingLogger
from halo_training.optimizer import build_optimizer, build_scheduler, build_wsd_scheduler


def train(
    model: nn.Module,
    dataset: Union[str, Dataset] = "babylm",
    epochs: int = 1,
    time_budget_minutes: Optional[float] = None,
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
    use_lion: bool = False,
    lion_lr_ratio: float = 0.3,
    use_clion: bool = False,
    clion_nu: float = 1.0,
    use_bf16: bool = False,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    resume_from: Optional[str] = None,
    resize_vocab: Optional[int] = None,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
    scheduler_type: str = "cosine",
    z_loss_weight: float = 0.0,
    z_loss_fraction: float = 0.4,
    context_schedule: Optional[List] = None,
    tokenizer_path: Optional[str] = None,
    wd_start: float = 0.1,
    wd_end: float = 0.01,
    min_lr_ratio: float = 0.0,
    polar_ns: bool = False,
    chunked_ce: bool = False,
) -> Dict[str, Any]:
    """Train a model using Mode A (direct) or Mode B (layer-streaming).

    Args:
        model: Any nn.Module.
        dataset: "babylm", path string, or a torch Dataset instance.
        epochs: Number of passes over the dataset.
        time_budget_minutes: Optional wall-clock limit in minutes. None = epoch-driven (preferred).
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

    # --- Apply autokernel BEFORE checkpoint load ---
    # Fused QKV keys must exist before load_state_dict() (CLAUDE.md constraint).
    # Checkpoints saved with autokernel have w_qkv; without it, model has wq/wk/wv.
    if optimize_kernels:
        try:
            import autokernel
            model = autokernel.optimize(model, training=True)
            print("autokernel optimizations applied")
        except Exception as e:
            print(f"autokernel.optimize() failed ({e}), continuing without")

    # --- Continued pre-training: load weights from checkpoint (fresh optimizer) ---
    # If resize_vocab is set, detect whether checkpoint already has the resized vocab
    # and resize model accordingly before or after loading.
    prev_tokens = 0
    if resume_from:
        print(f"Loading checkpoint for continued pre-training: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Check if checkpoint has resized vocab (SFT checkpoint)
        if resize_vocab:
            from halo_training.chat_template import resize_embeddings
            ckpt_vocab = state_dict.get("tok_embeddings.weight", state_dict.get("output.weight"))
            if ckpt_vocab is not None and ckpt_vocab.shape[0] == resize_vocab:
                # Checkpoint already resized — resize model first, then load
                model = resize_embeddings(model, resize_vocab)
                print(f"  Resized embeddings to {resize_vocab} (matching checkpoint)")
                resize_vocab = None  # already done

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

    # --- Resize embeddings for SFT (after checkpoint load, before compile) ---
    if resize_vocab:
        from halo_training.chat_template import resize_embeddings
        model = resize_embeddings(model, resize_vocab)
        print(f"  Resized embeddings to vocab_size={resize_vocab}")

    model.train()

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    if use_streaming:
        from halo_training.streaming import LayerStreamingTrainer
        streamer = LayerStreamingTrainer(
            model,
            checkpoint_every=checkpoint_every,
            compile_layers=compile,
        )
    elif compile:
        # compile_mode can be overridden via env var TORCH_COMPILE_MODE.
        # Options: "default", "reduce-overhead" (CUDA graphs, lower memory),
        # "max-autotune" (slowest warmup, potentially fastest runtime).
        #
        # NOTE: reduce-overhead is NOT currently compatible with looped models
        # (Parcae/HALO) because per-layer CUDA graphs reuse buffers across
        # iterations, invalidating saved activations for backward. Falls back
        # to "default" for such models.
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
        if compile_mode == "reduce-overhead" and hasattr(model, "compile_zones"):
            print("WARNING: reduce-overhead is incompatible with looped models "
                  "(buffer reuse across Parcae iterations). Falling back to default.")
            compile_mode = "default"
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torchinductor")
        cache_exists = os.path.isdir(os.path.expanduser(cache_dir)) and len(os.listdir(os.path.expanduser(cache_dir))) > 0
        cache_status = "warm cache" if cache_exists else "cold — first compile will be slow"
        # For looped models (HALO family) compile per-layer via compile_zones.
        # This avoids re-compiling the shared layer body at each Parcae iteration
        # and lets Inductor fuse ops within a single layer cleanly.
        if hasattr(model, "compile_zones") and compile_mode == "default":
            print(f"Compiling model via compile_zones (per-layer, looped), cache: {cache_status}")
            model.compile_zones()
        else:
            print(f"Compiling model with torch.compile ({compile_mode}), cache: {cache_status}")
            model = torch.compile(model, mode=compile_mode)

    # --- Setup EMA (TRM paper: 7.5% generalization gain) ---
    ema_model = None
    if use_ema:
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model,
            multi_avg_fn=lambda avg_params, model_params, num: [
                a * ema_decay + m * (1 - ema_decay) for a, m in zip(avg_params, model_params)])
        print(f"EMA enabled (decay={ema_decay})")

    # --- Setup data ---
    dataset_root = None
    if isinstance(dataset, str):
        dataset_root = dataset
        initial_block_size = block_size
        if context_schedule:
            initial_block_size = context_schedule[0][1]
        if dataset == "babylm":
            dataset = BabyLMDataset(block_size=initial_block_size, tokenizer_path=tokenizer_path)
        else:
            dataset = BabyLMDataset(root=dataset, block_size=initial_block_size,
                                    tokenizer_path=tokenizer_path)

    # Vocab size check: clamp token IDs to model's vocab size if needed
    model_vocab = _get_vocab_size(model)
    if model_vocab and hasattr(dataset, "vocab_size") and dataset.vocab_size > model_vocab:
        print(f"WARNING: tokenizer vocab ({dataset.vocab_size}) > model vocab ({model_vocab}), "
              f"clamping token IDs")
        dataset.tokens = dataset.tokens.clamp(max=model_vocab - 1)

    dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # --- Setup optimizer (NEVER compile this) ---
    optimizer = build_optimizer(model, base_lr=base_lr, use_muon=use_muon,
                                use_lion=use_lion, lion_lr_ratio=lion_lr_ratio,
                                use_clion=use_clion, clion_nu=clion_nu,
                                polar_ns=polar_ns)

    total_steps = len(dataloader) * epochs // accum_steps
    if max_steps:
        total_steps = min(total_steps, max_steps)
    if scheduler_type == "wsd":
        scheduler = build_wsd_scheduler(
            optimizer, total_steps, warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio, wd_start=wd_start, wd_end=wd_end)
    else:
        scheduler = build_scheduler(optimizer, total_steps, warmup_steps=warmup_steps)

    # --- Setup loss ---
    chunked_ce_fn = None
    chunked_ce_handles_zloss = False
    if loss_fn is None:
        # Chunked CE: activated if user opts in via --chunked-ce AND model supports it
        # (model.use_chunked_ce=True → forward returns h_low instead of logits).
        if chunked_ce and hasattr(model, "use_chunked_ce") and model.use_chunked_ce:
            if os.environ.get("TORCH_COMPILE_MODE") == "reduce-overhead":
                print("WARNING: --chunked-ce + TORCH_COMPILE_MODE=reduce-overhead is not "
                      "currently supported (CUDA graph aliasing issue). "
                      "Falling back to standard CE.")
                model.use_chunked_ce = False
            else:
                try:
                    from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
                    softcap = float(getattr(model, "logit_softcap", 0.0) or 0.0)
                    # Pass z_loss through chunked CE so we avoid re-computing over logits
                    chunked_z_weight = float(z_loss_weight) if z_loss_weight > 0 else 0.0
                    chunked_ce_fn = ChunkedLinearCrossEntropyLoss(
                        chunk_size=512,
                        softcap=softcap,
                        ignore_index=-100,
                        label_smoothing=float(label_smoothing),
                        z_loss_weight=chunked_z_weight,
                    )
                    chunked_ce_handles_zloss = chunked_z_weight > 0
                    print(f"Using ChunkedLinearCrossEntropyLoss "
                          f"(softcap={softcap}, z_loss={chunked_z_weight}, "
                          f"label_smoothing={label_smoothing}, chunk=512)")
                except Exception as e:
                    print(f"Chunked CE disabled: {e}")
                    chunked_ce_fn = None

        ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)

        def default_loss_fn(output, batch):
            _, targets = batch
            targets = targets.to(device, non_blocking=True)
            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, dict):
                logits = output["logits"]
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler,
                                   init_scale=1024.0, backoff_factor=0.25)

    # --- Training loop ---
    model.train()
    global_step = 0
    total_tokens = 0
    running_loss = 0.0
    start_time = time.time()
    deadline = start_time + time_budget_minutes * 60 if (time_budget_minutes and not max_steps) else None
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
    budget_str = f"{time_budget_minutes:.0f} min" if time_budget_minutes else "none (epoch-driven)"
    print(f"Time budget: {budget_str} | "
          f"Epochs: {epochs} | Steps/epoch: {len(dataloader)}, optimizer steps: {total_steps}")

    current_block_size = context_schedule[0][1] if context_schedule else block_size

    try:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Check stopping criteria
                if deadline and time.time() > deadline:
                    print(f"Time budget reached at step {global_step}")
                    break
                if max_steps and global_step >= max_steps:
                    print(f"Max steps reached: {global_step}")
                    break

                # Callbacks (phase scheduling, memory monitoring, etc.)
                for cb in callbacks:
                    cb(model, global_step)

                # Forward + backward with mixed precision
                input_ids, targets = batch
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except (AttributeError, RuntimeError):
                    pass

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if streamer is not None:
                        output = streamer.forward(input_ids)
                    else:
                        # Pass targets if model accepts them (adaptive head)
                        output = model(input_ids, targets=targets)

                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        # Model returned scalar loss (adaptive head with chunked CE)
                        loss = output / accum_steps
                    elif chunked_ce_fn is not None and isinstance(output, torch.Tensor) and output.dim() >= 2:
                        # Chunked CE: compute loss directly from hidden/logits + lm_head weight
                        vocab_size = getattr(model, "vocab_size", None)
                        if vocab_size is None and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                            vocab_size = model.lm_head.weight.shape[0]
                        if vocab_size is not None and output.shape[-1] != vocab_size and hasattr(model, "lm_head"):
                            if hasattr(model.lm_head, "embed_table"):
                                weight = model.lm_head.embed_table.weight
                            else:
                                weight = model.lm_head.weight
                            loss = chunked_ce_fn(
                                output.view(-1, output.shape[-1]),
                                weight,
                                targets.view(-1),
                            ) / accum_steps
                        else:
                            loss = loss_fn(output, batch) / accum_steps
                    else:
                        loss = loss_fn(output, batch) / accum_steps

                # z_loss: skip if chunked CE already incorporated it
                if (not chunked_ce_handles_zloss) and z_loss_weight > 0 and global_step < total_steps * z_loss_fraction:
                    if isinstance(output, dict) and "logits" in output:
                        z_logits = output["logits"]
                    elif isinstance(output, torch.Tensor) and output.dim() >= 2:
                        z_logits = output
                    else:
                        z_logits = None
                    if z_logits is not None:
                        loss = loss + z_loss_weight * z_logits.float().pow(2).mean() / accum_steps

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
                    if ema_model is not None:
                        ema_model.update_parameters(model)
                    global_step += 1

                    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                    if hasattr(raw_model, "set_training_progress"):
                        raw_model.set_training_progress(global_step, total_steps)

                    if context_schedule and dataset_root:
                        for frac, new_bs in context_schedule:
                            boundary = int(frac * total_steps)
                            if global_step == boundary and new_bs != current_block_size:
                                current_block_size = new_bs
                                print(f"[Context Schedule] block_size -> {new_bs} at step {global_step}")
                                dataset = BabyLMDataset(
                                    root=dataset_root, block_size=new_bs,
                                    tokenizer_path=tokenizer_path)
                                dataloader = build_dataloader(
                                    dataset, batch_size=batch_size, num_workers=num_workers)
                                try:
                                    torch._dynamo.reset()
                                except Exception:
                                    pass
                                break

                    # Logging
                    if global_step % log_interval == 0:
                        elapsed = time.time() - start_time
                        tp = throughput.get_interval_stats()
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
                        _save_checkpoint(model, optimizer, global_step, checkpoint_dir, total_tokens, ema_model)

            else:
                continue
            break  # Time budget reached

    except KeyboardInterrupt:
        print(f"\nInterrupted at step {global_step}")

    # Always save final checkpoint
    if checkpoint_dir and global_step > 0:
        _save_checkpoint(model, optimizer, global_step, checkpoint_dir, total_tokens, ema_model)
        print(f"Final checkpoint saved at step {global_step}")

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
    ema_model=None,
):
    """Save model and optimizer state."""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_tokens": total_tokens,
    }
    if ema_model is not None:
        raw_ema = ema_model.module._orig_mod if hasattr(ema_model.module, "_orig_mod") else ema_model.module
        ckpt["ema_state_dict"] = raw_ema.state_dict()
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path}")
