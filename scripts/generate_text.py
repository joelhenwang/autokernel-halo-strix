"""Quick text generation from a trained model checkpoint.

Usage:
    python scripts/generate_text.py \
        --model models/argus_prime.py --class-name ArgusPrime \
        --checkpoint checkpoints/argus_prime_gpt/step_12000.pt \
        --prompt "The quick brown fox" --max-tokens 200
"""

import argparse
import sys
import importlib.util

import torch
import torch.nn.functional as F
import tiktoken


def load_model(model_path, class_name):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls()


def generate(model, prompt_ids, max_tokens=200, temperature=0.8, top_k=40, top_p=0.9,
             repetition_penalty=1.0, frequency_penalty=0.0):
    """Autoregressive generation with top-k + top-p + repetition/frequency penalties."""
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    token_counts = torch.zeros(1, 50257, device=device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for _ in range(max_tokens):
            # Truncate to max_seq_len if needed
            context = ids[:, -1024:]  # use last 1024 tokens
            logits = model(context)
            logits = logits[:, -1, :]

            # Repetition penalty: penalize tokens already in the sequence
            if repetition_penalty != 1.0:
                generated = ids[0].tolist()
                seen = set(generated)
                for token_id in seen:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Frequency penalty: subtract penalty * count for each token
            if frequency_penalty > 0.0:
                logits -= frequency_penalty * token_counts

            logits /= temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)
            token_counts[0, next_token.item()] += 1

            # Stop on EOS
            if next_token.item() == 50256:
                break

    return ids


def main():
    parser = argparse.ArgumentParser(description="Generate text from checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Penalize repeated tokens (1.0=off, 1.1-1.5=typical)")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="Subtract penalty*count from logits (0.0=off, 0.5-2.0=typical)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    sys.path.insert(0, ".")
    model = load_model(args.model, args.class_name)
    model = model.to(device)

    # Apply autokernel (checkpoint was saved with autokernel-optimized model)
    try:
        import autokernel
        model = autokernel.optimize(model, training=False)
        print("Applied autokernel.optimize()")
    except Exception as e:
        print(f"autokernel not applied: {e}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        # Try strict first, fall back to non-strict
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("  Warning: loaded with strict=False (some keys may not match)")
        print(f"  Step: {ckpt.get('step', '?')}")
    else:
        model.load_state_dict(ckpt, strict=False)

    # Tokenize prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    print(f"\nPrompt: {args.prompt}")
    print(f"Config: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
          f"rep_pen={args.repetition_penalty}, freq_pen={args.frequency_penalty}")
    print("=" * 60)

    for i in range(args.num_samples):
        output_ids = generate(
            model, prompt_ids, args.max_tokens,
            args.temperature, args.top_k, args.top_p,
            args.repetition_penalty, args.frequency_penalty
        )
        text = enc.decode(output_ids[0].tolist())
        print(f"\n--- Sample {i+1} ---")
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
