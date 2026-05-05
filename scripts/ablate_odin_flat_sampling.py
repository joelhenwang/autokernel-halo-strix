"""3-stage sampling-parameter ablation for OdinFlat.

Loads model + checkpoint once, runs a narrowing sweep over (temperature,
repetition_penalty), then top_p, then top_k. Reports distinct-2, self-PPL,
length, and a preview for each config across two prompts.

Usage:
    python scripts/ablate_odin_flat_sampling.py \
        --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPTS = [
    "The history of the Roman Empire",
    "In the field of physics,",
]

SAMPLES_PER_CONFIG = 3
MAX_TOKENS = 120
BASE_SEED = 42


def load_model(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


@torch.no_grad()
def generate(model, prompt_ids, *, max_tokens, temperature, top_k, top_p,
             repetition_penalty, eos_token, vocab_size, max_seq_len=2048):
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    for _ in range(max_tokens):
        context = ids[:, -max_seq_len:]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(context)
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            seen = set(ids[0].tolist())
            for tid in seen:
                if 0 <= tid < vocab_size:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        if temperature > 0:
            logits = logits / temperature

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[:, [-1]],
                                 torch.full_like(logits, float("-inf")), logits)

        if top_p and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == eos_token:
            break
    return ids


@torch.no_grad()
def compute_self_ppl(model, ids, prompt_len):
    """CE on generated portion under same softcap/fp16 path as generation."""
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(ids)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].float()
    shift_targets = ids[:, 1:]
    # Score only the generated span
    gen_logits = shift_logits[:, prompt_len - 1:, :]
    gen_targets = shift_targets[:, prompt_len - 1:]
    if gen_logits.shape[1] == 0:
        return float("inf")
    loss = F.cross_entropy(
        gen_logits.reshape(-1, gen_logits.size(-1)),
        gen_targets.reshape(-1),
        reduction="mean",
    )
    return math.exp(min(loss.item(), 20))  # Cap to prevent overflow


def distinct_2(token_ids):
    """Unique bigrams / total bigrams. Higher = less repetitive."""
    if len(token_ids) < 2:
        return 0.0
    bigrams = [(token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)]
    return len(set(bigrams)) / len(bigrams)


def eval_config(model, tok, bl_decoder, eos_id, vocab_size, cfg, prompt_ids_list,
                prompts):
    """Returns dict with per-config aggregate metrics + preview strings."""
    all_dist2 = []
    all_ppl = []
    all_lens = []
    previews = {}  # prompt_idx -> first sample's decoded text (first 80 chars)
    sample_no = 0
    device = next(model.parameters()).device

    for p_idx, prompt_ids in enumerate(prompt_ids_list):
        for s_idx in range(SAMPLES_PER_CONFIG):
            torch.manual_seed(BASE_SEED + sample_no)
            out = generate(
                model, prompt_ids,
                max_tokens=MAX_TOKENS,
                temperature=cfg["temp"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
                repetition_penalty=cfg["rep_pen"],
                eos_token=eos_id,
                vocab_size=vocab_size,
            )
            prompt_len = prompt_ids.shape[1]
            gen_ids = out[0, prompt_len:].tolist()
            all_dist2.append(distinct_2(gen_ids))
            all_lens.append(len(gen_ids))
            all_ppl.append(compute_self_ppl(model, out, prompt_len))
            if s_idx == 0:
                toks = [tok.id_to_token(i) for i in out[0].tolist()]
                text = bl_decoder.decode(toks)
                previews[p_idx] = text[:100].replace("\n", " ")
            sample_no += 1

    return {
        "dist2": sum(all_dist2) / len(all_dist2),
        "ppl": sum(all_ppl) / len(all_ppl),
        "len": sum(all_lens) / len(all_lens),
        "previews": previews,
    }


def pick_winner(results):
    """Highest dist2 among configs with ppl <= 1.5 * min(ppl)."""
    min_ppl = min(r["metrics"]["ppl"] for r in results)
    ppl_threshold = 1.5 * min_ppl
    eligible = [r for r in results if r["metrics"]["ppl"] <= ppl_threshold]
    if not eligible:
        eligible = results
    return max(eligible, key=lambda r: r["metrics"]["dist2"])


def print_table(title, configs, results, fixed_note):
    print(f"\n{'=' * 88}")
    print(f"{title}  ({fixed_note})")
    print("=" * 88)
    hdr = f"  {'cfg':>3}  {'temp':>5}  {'rep_pen':>7}  {'top_p':>5}  {'top_k':>5}  " \
          f"{'dist2':>6}  {'self-PPL':>8}  {'len':>4}  preview (P0 / P1)"
    print(hdr)
    print("-" * 88)
    for i, (cfg, r) in enumerate(zip(configs, results)):
        m = r["metrics"]
        prev = r["metrics"]["previews"]
        print(f"  {i+1:>3}  {cfg['temp']:>5.2f}  {cfg['rep_pen']:>7.2f}  "
              f"{cfg['top_p']:>5.2f}  {cfg['top_k']:>5}  "
              f"{m['dist2']:>6.3f}  {m['ppl']:>8.2f}  {m['len']:>4.0f}  "
              f"{prev[0][:70]!r}")
        print(f"       {'':>26}{'':>26}{prev[1][:70]!r}")


def run_stage(stage_name, stage_configs, fixed_note, model, tok, bl_decoder,
              eos_id, vocab_size, prompt_ids_list, prompts):
    results = []
    for cfg in stage_configs:
        t0 = time.time()
        metrics = eval_config(model, tok, bl_decoder, eos_id, vocab_size,
                              cfg, prompt_ids_list, prompts)
        elapsed = time.time() - t0
        results.append({"cfg": cfg, "metrics": metrics, "elapsed": elapsed})
        print(f"  ran cfg temp={cfg['temp']:.2f} rep_pen={cfg['rep_pen']:.2f} "
              f"top_p={cfg['top_p']:.2f} top_k={cfg['top_k']} "
              f"in {elapsed:.1f}s — dist2={metrics['dist2']:.3f} "
              f"ppl={metrics['ppl']:.1f}")
    # Attach previews into a nested dict for print_table
    for r in results:
        r["metrics"]["previews"] = r["metrics"]["previews"]
    print_table(stage_name, [r["cfg"] for r in results], results, fixed_note)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/odin_flat.py")
    parser.add_argument("--class-name", default="OdinFlat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", default="tokenizers/odin-32k/tokenizer.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Tokenizer ---
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tok = HFTokenizer.from_file(args.tokenizer_path)
    bl_decoder = ByteLevelDecoder()
    vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0
    print(f"Tokenizer: vocab={vocab_size}, eos={eos_id}")

    # --- Model ---
    model = load_model(args.model, args.class_name).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    print(f"  Loaded cleanly; step={ckpt.get('step', '?')}")
    model.half().eval()

    # --- Prompts ---
    prompt_ids_list = []
    for p in PROMPTS:
        enc = tok.encode(p)
        prompt_ids_list.append(torch.tensor([enc.ids], dtype=torch.long))
        print(f"  Prompt: {p!r} -> {len(enc.ids)} tokens")

    print(f"\nSamples per config: {SAMPLES_PER_CONFIG} x {len(PROMPTS)} prompts")
    print(f"Max tokens: {MAX_TOKENS}")

    # ========================================================================
    # Stage 1: temp x rep_pen
    # ========================================================================
    print("\n\n### STAGE 1: temperature x repetition_penalty")
    stage1_configs = []
    for temp in [0.6, 0.8, 1.0]:
        for rep_pen in [1.0, 1.15, 1.3]:
            stage1_configs.append({
                "temp": temp, "rep_pen": rep_pen,
                "top_k": 40, "top_p": 0.95,
            })
    stage1_results = run_stage(
        "Stage 1: temp x rep_pen", stage1_configs,
        "top_k=40, top_p=0.95",
        model, tok, bl_decoder, eos_id, vocab_size, prompt_ids_list, PROMPTS,
    )
    s1_winner = pick_winner(stage1_results)
    s1_cfg = s1_winner["cfg"]
    print(f"\n>> Stage 1 winner: temp={s1_cfg['temp']:.2f}, "
          f"rep_pen={s1_cfg['rep_pen']:.2f}  "
          f"(dist2={s1_winner['metrics']['dist2']:.3f}, "
          f"ppl={s1_winner['metrics']['ppl']:.1f})")

    # ========================================================================
    # Stage 2: top_p refinement
    # ========================================================================
    print("\n\n### STAGE 2: top_p refinement")
    stage2_configs = [
        {"temp": s1_cfg["temp"], "rep_pen": s1_cfg["rep_pen"],
         "top_k": 40, "top_p": top_p}
        for top_p in [0.85, 0.95, 1.0]
    ]
    stage2_results = run_stage(
        "Stage 2: top_p refinement", stage2_configs,
        f"temp={s1_cfg['temp']:.2f}, rep_pen={s1_cfg['rep_pen']:.2f}, top_k=40",
        model, tok, bl_decoder, eos_id, vocab_size, prompt_ids_list, PROMPTS,
    )
    s2_winner = pick_winner(stage2_results)
    s2_cfg = s2_winner["cfg"]
    print(f"\n>> Stage 2 winner: top_p={s2_cfg['top_p']:.2f}  "
          f"(dist2={s2_winner['metrics']['dist2']:.3f}, "
          f"ppl={s2_winner['metrics']['ppl']:.1f})")

    # ========================================================================
    # Stage 3: top_k verification
    # ========================================================================
    print("\n\n### STAGE 3: top_k verification")
    stage3_configs = [
        {"temp": s2_cfg["temp"], "rep_pen": s2_cfg["rep_pen"],
         "top_p": s2_cfg["top_p"], "top_k": top_k}
        for top_k in [0, 40, 100]
    ]
    stage3_results = run_stage(
        "Stage 3: top_k verification", stage3_configs,
        f"temp={s2_cfg['temp']:.2f}, rep_pen={s2_cfg['rep_pen']:.2f}, "
        f"top_p={s2_cfg['top_p']:.2f}",
        model, tok, bl_decoder, eos_id, vocab_size, prompt_ids_list, PROMPTS,
    )
    s3_winner = pick_winner(stage3_results)
    s3_cfg = s3_winner["cfg"]
    print(f"\n>> Stage 3 winner: top_k={s3_cfg['top_k']}  "
          f"(dist2={s3_winner['metrics']['dist2']:.3f}, "
          f"ppl={s3_winner['metrics']['ppl']:.1f})")

    # ========================================================================
    # Final: full-text samples for the winner
    # ========================================================================
    print(f"\n\n{'=' * 88}")
    print("FINAL WINNING CONFIG")
    print("=" * 88)
    print(f"  temp = {s3_cfg['temp']:.2f}")
    print(f"  rep_pen = {s3_cfg['rep_pen']:.2f}")
    print(f"  top_p = {s3_cfg['top_p']:.2f}")
    print(f"  top_k = {s3_cfg['top_k']}")
    print(f"  dist2 = {s3_winner['metrics']['dist2']:.3f}")
    print(f"  self-PPL = {s3_winner['metrics']['ppl']:.2f}")

    print(f"\n{'=' * 88}")
    print("FINAL SAMPLES (3 per prompt, seed=42/43/44)")
    print("=" * 88)
    for p_idx, (prompt, prompt_ids) in enumerate(zip(PROMPTS, prompt_ids_list)):
        print(f"\n[Prompt {p_idx+1}] {prompt!r}")
        print("-" * 60)
        for s_idx in range(SAMPLES_PER_CONFIG):
            torch.manual_seed(BASE_SEED + s_idx)
            out = generate(
                model, prompt_ids,
                max_tokens=MAX_TOKENS,
                temperature=s3_cfg["temp"],
                top_k=s3_cfg["top_k"],
                top_p=s3_cfg["top_p"],
                repetition_penalty=s3_cfg["rep_pen"],
                eos_token=eos_id,
                vocab_size=vocab_size,
            )
            toks = [tok.id_to_token(i) for i in out[0].tolist()]
            text = bl_decoder.decode(toks)
            print(f"\n  Sample {s_idx+1}:")
            print(f"  {text}")


if __name__ == "__main__":
    main()
