"""Text generation from OdinFlat checkpoint using the odin-32k HF tokenizer.

Usage:
    python scripts/sample_odin_flat.py \
        --checkpoint checkpoints/odin-flat-wikitext-ddp/step_1869.pt \
        --prompt "The history of" --max-tokens 100 --num-samples 3
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_model(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


@torch.no_grad()
def generate(model, prompt_ids, *, max_tokens=200, temperature=0.8,
             top_k=40, top_p=0.9, repetition_penalty=1.0,
             eos_token=0, max_seq_len=2048, vocab_size=32768):
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    model.eval()

    for step in range(max_tokens):
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


def main():
    parser = argparse.ArgumentParser(description="Generate text from OdinFlat checkpoint")
    parser.add_argument("--model", default="models/odin_flat.py")
    parser.add_argument("--class-name", default="OdinFlat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", default="tokenizers/odin-32k/tokenizer.json")
    parser.add_argument("--prompt", default="The history of the Roman Empire")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer (HuggingFace odin-32k)
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tok = HFTokenizer.from_file(args.tokenizer_path)
    bl_decoder = ByteLevelDecoder()
    vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0  # Fallback per AGENTS.md
    print(f"Tokenizer: {args.tokenizer_path} (vocab={vocab_size}, eos={eos_id})")

    # Load model
    model = load_model(args.model, args.class_name).to(device)
    print(f"Loaded {args.class_name}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip torch.compile's ._orig_mod. prefix (from per-layer compile_zones
    # wrappers during training). Fail loudly if any keys still don't match —
    # silent strict=False previously masked a random-weights load bug.
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch after _orig_mod stripping: "
            f"{len(missing)} missing, {len(unexpected)} unexpected.\n"
            f"  First missing:    {list(missing)[:3]}\n"
            f"  First unexpected: {list(unexpected)[:3]}"
        )
    print(f"  Loaded cleanly (strict); step={ckpt.get('step', '?')}")

    model.half()
    model.eval()

    # Tokenize prompt
    enc = tok.encode(args.prompt)
    prompt_ids = torch.tensor([enc.ids], dtype=torch.long)
    print(f"\nPrompt ({len(enc.ids)} tokens): {args.prompt!r}")
    print(f"Config: temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
          f"rep_pen={args.repetition_penalty} max_tokens={args.max_tokens}")
    print("=" * 70)

    for i in range(args.num_samples):
        t0 = time.time()
        output_ids = generate(
            model, prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token=eos_id, vocab_size=vocab_size,
        )
        n_new = output_ids.shape[1] - prompt_ids.shape[1]
        elapsed = time.time() - t0
        tokens = [tok.id_to_token(i) for i in output_ids[0].tolist()]
        text = bl_decoder.decode(tokens)

        print(f"\n--- Sample {i+1} ({n_new} new tokens in {elapsed:.2f}s, "
              f"{n_new/elapsed:.1f} tok/s) ---")
        print(text)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
