"""
Compile × Kernel-Optimization Ablation (OdinHalo V=32768).

8 configurations tested:
  compile={off, zones}  ×  kernels={baseline, +HIP_CE, +HIP_CE+RoPE, +HIP_CE+RoPE+Chunked}

Each config runs 400 steps total; first 200 are warmup (not counted). The last
200 steps are timed for tok/s. This amortizes torch.compile's Inductor warmup
cost (which can burn 30-60 s on first run) and ensures steady-state measurement.

Peak memory captured during measurement window only (reset after warmup).
"""
import sys, time, gc, argparse
import torch

sys.path.insert(0, '.')

WARMUP_STEPS = 200
MEASURE_STEPS = 200
BATCH_SIZE = 4
BLOCK_SIZE = 256
device = 'cuda'


def load_batch_iterator():
    """Use BabyLM real data (matches prior ablation scripts)."""
    from halo_training.data import BabyLMDataset, build_dataloader
    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=BLOCK_SIZE,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    while True:
        for batch in dl:
            yield batch


# ---------------------------------------------------------------------------
# Kernel-opt toggles
# ---------------------------------------------------------------------------

import models.components.conv_blocks as cb
_ROPE_FUSION_ORIG = cb.HyPEShortConvBlock.forward

def _rope_fusion_off_forward(self, x, freqs_cis):
    """Undo RoPE fusion — apply gate RoPE + gate multiply as separate ops."""
    B, T, _ = x.shape
    normed = self.pre_norm(x)
    b, c, h_tilde = self.proj(normed).chunk(3, dim=-1)
    b = self._rope_on_gate(b, freqs_cis)
    y = b * h_tilde
    if cb._HAS_CAUSAL_CONV1D:
        z = cb.causal_conv1d_fn(
            y.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
    else:
        z = self.conv(y.transpose(1, 2))[:, :, :T].transpose(1, 2)
    conv_out = self.out_proj(c * z)
    ffn_out = self.ffn(self.ffn_norm(x + conv_out))
    return x + conv_out + ffn_out


def set_rope_fusion(on: bool):
    cb.HyPEShortConvBlock.forward = _ROPE_FUSION_ORIG if on else _rope_fusion_off_forward


# ---------------------------------------------------------------------------
# CE path builder
# ---------------------------------------------------------------------------

def build_ce_loss_fn(ce_kind: str):
    """Return a function loss_fn(model, output, targets) -> scalar loss."""
    if ce_kind == "pytorch":
        def f(model, logits, targets):
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return f
    elif ce_kind == "hip_tiny":
        import kernel as opt_ce
        def f(model, logits, targets):
            return opt_ce.ce_full(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                mode="tiny",
            )
        return f
    elif ce_kind == "chunked":
        from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
        loss_mod = ChunkedLinearCrossEntropyLoss(chunk_size=256, softcap=30.0)
        def f(model, h_low, targets):
            return loss_mod(
                h_low.view(-1, h_low.size(-1)),
                model.lm_head.embed_table.weight,
                targets.view(-1),
            )
        return f
    else:
        raise ValueError(ce_kind)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_config(name, build_model_fn, loss_fn, use_compile=False, use_friendly=False):
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    model = build_model_fn().to(device)
    model.train()

    if use_compile:
        if use_friendly and hasattr(model, "compile_zones_friendly"):
            model.compile_zones_friendly()
        elif hasattr(model, "compile_zones"):
            model.compile_zones()
        else:
            model = torch.compile(model, mode="default")

    # Use AdamW with a tame lr so fp16 training doesn't diverge during the 400-step window.
    # Throughput measurements are independent of lr; we just want clean final_loss readouts.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    batches = load_batch_iterator()

    # ---- Warmup ----
    warmup_start = time.time()
    for step in range(WARMUP_STEPS):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids)
            loss = loss_fn(model, output, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start

    # ---- Measurement ----
    torch.cuda.reset_peak_memory_stats()
    total_tokens = 0
    t0 = time.time()
    for step in range(MEASURE_STEPS):
        input_ids, targets = next(batches)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids)
            loss = loss_fn(model, output, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_tokens += input_ids.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    tok_s = total_tokens / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    final_loss = loss.item()
    print(f"  {name:55s} tok/s={tok_s:>7,.0f}  peak={peak_gb:>4.2f} GB  "
          f"warmup={warmup_time:>5.1f}s  final_loss={final_loss:.3f}")

    # Cleanup
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return tok_s, peak_gb, warmup_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Run only matching config(s)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compiled configs (faster test)")
    args = parser.parse_args()

    print(f"=== Compile × Kernel Ablation (OdinHalo V=32768) ===")
    print(f"Warmup: {WARMUP_STEPS} steps | Measure: {MEASURE_STEPS} steps | batch={BATCH_SIZE} block={BLOCK_SIZE}")
    print()

    from models.odin_halo import OdinHalo

    results = {}

    def run(label, ce_kind, rope_fusion, use_compile, use_chunked_ce=False, use_friendly=False):
        if args.only and args.only not in label:
            return
        set_rope_fusion(rope_fusion)
        loss_fn = build_ce_loss_fn(ce_kind)

        def build():
            if use_chunked_ce:
                return OdinHalo(use_chunked_ce=True)
            return OdinHalo()

        t, m, w = run_config(label, build, loss_fn, use_compile=use_compile, use_friendly=use_friendly)
        results[label] = {"tok_s": t, "peak_gb": m, "warmup_s": w}

    # ---- Eager configurations ----
    print("[eager — no torch.compile]")
    run("baseline (PyTorch CE, no RoPE fusion)", "pytorch", False, False)
    run("+HIP CE (tiny)",                        "hip_tiny", False, False)
    run("+HIP CE +RoPE fusion",                  "hip_tiny", True,  False)
    run("+HIP CE +RoPE +Chunked CE",             "chunked",  True,  False, use_chunked_ce=True)

    if not args.skip_compile:
        print()
        print("[compiled — compile_zones (HIP kernels, graph breaks allowed)]")
        run("compile + baseline",                "pytorch",  False, True)
        run("compile + HIP CE",                  "hip_tiny", False, True)
        run("compile + HIP CE + RoPE fusion",    "hip_tiny", True,  True)
        run("compile + HIP CE + RoPE + Chunked", "chunked",  True,  True, use_chunked_ce=True)

        print()
        print("[compile-friendly — native RoPE + manual conv (0 graph breaks)]")
        # Kernels cannot coexist with compile-friendly (it DISABLES HIP fused kernels
        # in the block). We combine with HIP CE at the loss stage (which is outside the
        # compiled layer and thus doesn't break the graph).
        run("compile-friendly (PyTorch CE)",     "pytorch",  False, True, use_friendly=True)
        run("compile-friendly + HIP CE",         "hip_tiny", False, True, use_friendly=True)
        run("compile-friendly + Chunked CE",     "chunked",  False, True, use_chunked_ce=True, use_friendly=True)

    # Reset
    set_rope_fusion(True)

    print()
    print("=" * 85)
    print("SUMMARY TABLE")
    print("=" * 85)
    if not results:
        print("No results.")
        return
    baseline = results.get("baseline (PyTorch CE, no RoPE fusion)", None)
    baseline_tok = baseline["tok_s"] if baseline else None
    print(f"{'Config':<55s} {'tok/s':>9} {'peak':>8} {'warm':>7} {'speedup':>8}")
    print("-" * 90)
    for label, r in results.items():
        sp = f"{r['tok_s']/baseline_tok:.3f}x" if baseline_tok else "  —"
        print(f"{label:<55s} {r['tok_s']:>9,.0f} {r['peak_gb']:>5.2f} GB  {r['warmup_s']:>5.1f}s {sp:>8}")

    # Also show compile-lift per kernel config
    print()
    print("=" * 85)
    print("COMPILE LIFT (tok/s compiled / tok/s eager)")
    print("=" * 85)
    pairs = [
        ("baseline (PyTorch CE, no RoPE fusion)", "compile + baseline"),
        ("baseline (PyTorch CE, no RoPE fusion)", "compile-friendly (PyTorch CE)"),
        ("+HIP CE (tiny)",                        "compile + HIP CE"),
        ("+HIP CE (tiny)",                        "compile-friendly + HIP CE"),
        ("+HIP CE +RoPE fusion",                  "compile + HIP CE + RoPE fusion"),
        ("+HIP CE +RoPE +Chunked CE",             "compile + HIP CE + RoPE + Chunked"),
        ("+HIP CE +RoPE +Chunked CE",             "compile-friendly + Chunked CE"),
    ]
    for eager_label, compiled_label in pairs:
        e = results.get(eager_label)
        c = results.get(compiled_label)
        if e and c:
            lift = c["tok_s"] / e["tok_s"]
            print(f"  {eager_label:<48s} -> {compiled_label:<36s}  lift={lift:.3f}x")


if __name__ == "__main__":
    main()
