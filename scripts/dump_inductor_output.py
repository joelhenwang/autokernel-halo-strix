"""WI6: Dump Inductor triton output code for OdinHalo under compile_zones().

Captures all FX graphs produced by torch.compile and dumps them to a log file
along with the generated triton kernels. Used to build the Inductor fusion catalog.

Usage:
  python scripts/dump_inductor_output.py --out docs/perf/inductor-triton-dump.log
"""
import os, sys, argparse, pathlib, io, contextlib
sys.path.insert(0, '.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="docs/perf/inductor-triton-dump.log")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    # Configure Inductor to dump output code to stdout/stderr BEFORE any torch import
    os.environ["TORCH_LOGS"] = "output_code,graph_code"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor-phase2-wi6"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"  # we want output_code via TORCH_LOGS, not full debug

    # Clean cache so we see fresh generation
    import shutil
    cache = pathlib.Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Prepare output file and redirect root logger handlers to it
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(out_path, "w", encoding="utf-8", errors="replace")

    import logging
    # TORCH_LOGS routes to the "torch._inductor.output_code" logger via torch._logging
    # Add a FileHandler to root so everything lands in our file
    fh = logging.FileHandler(out_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.DEBUG)

    # Ensure the inductor output_code logger is captured
    for name in ("torch._inductor.output_code", "torch._inductor.graph",
                 "torch._dynamo.output_graph", "torch._inductor.compile_fx"):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)

    import torch
    from models.odin_halo import OdinHalo
    from halo_training.data import BabyLMDataset, build_dataloader

    device = 'cuda'
    torch.manual_seed(42)

    model = OdinHalo().to(device)
    model.train()

    print(f"[dump_inductor] applying compile_zones()", file=sys.stderr)
    log_file.write(f"\n\n=== begin compile_zones() ===\n")
    log_file.flush()
    model.compile_zones()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    ds = BabyLMDataset(
        root='datasets/babylm-odin32k.bin',
        block_size=args.block_size,
        tokenizer_path='tokenizers/odin-32k/tokenizer.json',
    )
    dl = build_dataloader(ds, batch_size=args.batch_size, num_workers=0, shuffle=True)

    log_file.write(f"\n\n=== begin training steps (this triggers compile) ===\n")
    log_file.flush()

    # Run a few steps to trigger all compilations
    n_steps = 4  # two fwd/bwd with warm + cold cache behavior
    it = iter(dl)
    for step in range(n_steps):
        input_ids, targets = next(it)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        log_file.write(f"\n=== step {step+1} complete; loss={loss.item():.4f} ===\n")
        log_file.flush()
        print(f"[dump_inductor] step {step+1}/{n_steps} loss={loss.item():.4f}",
              file=sys.stderr)

    log_file.write(f"\n=== DONE ===\n")
    log_file.flush()
    log_file.close()

    # List inductor cache contents - python sources for each FxGraph are stored there
    print(f"\n[dump_inductor] log written: {out_path} ({out_path.stat().st_size} bytes)",
          file=sys.stderr)
    py_files = sorted(cache.rglob("*.py"))
    print(f"[dump_inductor] inductor cache: {cache} ({len(py_files)} .py files)",
          file=sys.stderr)

    # Concatenate all cache .py files into a second dump for easier parsing
    cache_dump = out_path.with_suffix(".cache.py")
    with open(cache_dump, "w", encoding="utf-8", errors="replace") as f:
        for py in py_files:
            f.write(f"\n\n################ FILE: {py.relative_to(cache)} ################\n\n")
            try:
                f.write(py.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                f.write(f"<read error: {e}>\n")
    print(f"[dump_inductor] cache concat written: {cache_dump} ({cache_dump.stat().st_size} bytes)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
