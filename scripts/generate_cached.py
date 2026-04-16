#!/usr/bin/env python3
"""KV-cached text generation for JORMUNGANDR-HALO and ARGUS-PRIME.

Pre-allocates KV-cache and conv state for compile-friendly fixed-shape
inference. Each decode step processes only the new token.

Usage:
    python scripts/generate_cached.py \
        --model models/jormungandr_halo.py --class-name JormungandrHalo \
        --checkpoint checkpoints/step_5000.pt \
        --prompt "The history of" --max-tokens 200

    python scripts/generate_cached.py \
        --model models/argus_prime.py --class-name ArgusPrime \
        --checkpoint checkpoints/argus_prime_gpt/step_12000.pt \
        --prompt "Once upon a time"
"""

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ---------------------------------------------------------------------------
# Inference Cache
# ---------------------------------------------------------------------------

@dataclass
class InferenceCache:
    """Pre-allocated inference state for KV-cache + conv state."""

    # GQA KV-caches: {layer_name: (K, V)} where K/V are (B, n_kv, max_seq, hd)
    kv_caches: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

    # Conv states: {layer_name: (B, kernel_size-1, d_conv)} for GatedConv
    conv_states: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Current sequence position (how many tokens have been processed)
    seq_pos: int = 0


def create_cache(model, batch_size, max_seq_len, device, dtype):
    """Pre-allocate cache tensors for all cacheable layers."""
    cache = InferenceCache()

    for name, module in model.named_modules():
        # KV-cache for attention layers (standard or autokernel fused)
        is_attn = (hasattr(module, 'wq') and hasattr(module, 'wk')) or hasattr(module, 'w_qkv')
        if is_attn and hasattr(module, 'n_kv_heads') and hasattr(module, 'head_dim'):
            n_kv = module.n_kv_heads
            hd = module.head_dim
            k_cache = torch.zeros(batch_size, n_kv, max_seq_len, hd, device=device, dtype=dtype)
            v_cache = torch.zeros(batch_size, n_kv, max_seq_len, hd, device=device, dtype=dtype)
            cache.kv_caches[name] = [k_cache, v_cache]

        # Conv state for GatedConv layers
        if isinstance(module, nn.Module) and hasattr(module, 'conv') and hasattr(module, 'd_conv'):
            ks = module.kernel_size
            d_conv = module.d_conv
            state = torch.zeros(batch_size, ks - 1, d_conv, device=device, dtype=dtype)
            cache.conv_states[name] = state

    return cache


# ---------------------------------------------------------------------------
# Cached forward passes for individual components
# ---------------------------------------------------------------------------

def attention_cached(attn, x, freqs_cis, cache, cache_key, value_bias=None):
    """GQA attention with KV-cache. x is (B, T_new, D).

    Handles both standard Attention (separate wq/wk/wv) and autokernel's
    _FusedQKVAttentionReplacement (fused w_qkv + rotary_fn).
    """
    from models.argus import apply_rotary_emb

    B, T_new, _ = x.shape
    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads
    hd = attn.head_dim
    n_rep = attn.n_rep

    if hasattr(attn, 'w_qkv'):
        # Autokernel fused QKV path
        qkv = attn.w_qkv(x)
        q_size = attn.q_size
        k_size = attn.k_size
        q = qkv[:, :, :q_size].view(B, T_new, n_heads, hd)
        k = qkv[:, :, q_size:q_size + k_size].view(B, T_new, n_kv, hd)
        v = qkv[:, :, q_size + k_size:].view(B, T_new, n_kv, hd)
    else:
        # Standard separate Q/K/V path
        q = attn.wq(x).view(B, T_new, n_heads, hd)
        k = attn.wk(x).view(B, T_new, n_kv, hd)
        v = attn.wv(x).view(B, T_new, n_kv, hd)

    if value_bias is not None:
        v = v + value_bias.view(B, T_new, n_kv, hd)

    q, k = apply_rotary_emb(q, k, freqs_cis)
    q = q.transpose(1, 2)  # (B, n_heads, T_new, hd)
    k = k.transpose(1, 2)  # (B, n_kv, T_new, hd)
    v = v.transpose(1, 2)

    # QK-Norm if present
    if hasattr(attn, 'qk_norm') and attn.qk_norm:
        q = F.normalize(q, dim=-1) * attn.q_scale
        k = F.normalize(k, dim=-1) * attn.k_scale

    # Update KV-cache (store un-expanded K/V)
    k_cache, v_cache = cache.kv_caches[cache_key]
    pos = cache.seq_pos
    k_cache[:, :, pos:pos + T_new, :] = k
    v_cache[:, :, pos:pos + T_new, :] = v

    # Attend over full cached sequence
    k_full = k_cache[:, :, :pos + T_new, :]
    v_full = v_cache[:, :, :pos + T_new, :]

    if n_rep > 1:
        k_full = k_full.repeat_interleave(n_rep, dim=1)
        v_full = v_full.repeat_interleave(n_rep, dim=1)

    y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=(T_new > 1))
    return attn.wo(y.transpose(1, 2).contiguous().view(B, T_new, -1))


def gated_conv_cached(conv_module, x, cache, cache_key):
    """GatedConv with conv state cache. x is (B, T_new, D)."""
    B, T_new, _ = x.shape
    b, c, h_tilde = conv_module.proj(x).chunk(3, dim=-1)
    y = b * h_tilde  # (B, T_new, d_conv)

    if cache_key in cache.conv_states:
        state = cache.conv_states[cache_key]
        # Prepend cached state to y for convolution
        y_with_state = torch.cat([state, y], dim=1)  # (B, K-1+T_new, d_conv)

        # Run conv on the padded input
        if hasattr(conv_module, 'conv_weight'):
            # causal_conv1d path — manual conv for cached decode
            # For T_new=1: conv over (K-1+1) = K positions, take last output
            y_conv = y_with_state.transpose(1, 2)  # (B, d_conv, K-1+T_new)
            weight = conv_module.conv_weight  # (d_conv, K)
            bias = conv_module.conv_bias      # (d_conv,)
            # Depthwise conv1d: groups=d_conv
            z = F.conv1d(y_conv, weight.unsqueeze(1), bias, groups=y_conv.shape[1])
            z = z[:, :, -T_new:].transpose(1, 2)  # (B, T_new, d_conv)
        else:
            # nn.Conv1d path
            y_conv = y_with_state.transpose(1, 2)
            z = conv_module.conv(y_conv)[:, :, -T_new:].transpose(1, 2)

        # Update conv state: keep last K-1 values of y
        ks = state.shape[1] + 1  # kernel_size
        if y_with_state.shape[1] >= ks - 1:
            cache.conv_states[cache_key] = y_with_state[:, -(ks - 1):, :].detach()
    else:
        # No cache — standard forward
        if hasattr(conv_module, 'conv_weight'):
            from causal_conv1d import causal_conv1d_fn
            z = causal_conv1d_fn(
                y.transpose(1, 2), conv_module.conv_weight, conv_module.conv_bias
            ).transpose(1, 2)
        else:
            z = conv_module.conv(y.transpose(1, 2))[:, :, :T_new].transpose(1, 2)

    return c * z


def shortconv_block_cached(block, x, velocity, cache, name_prefix):
    """ShortConvBlock forward with conv state cache."""
    normed = block.pre_norm(x)
    conv_out = gated_conv_cached(block.conv, normed, cache, f"{name_prefix}.conv")
    mixer_out = block.out_proj(conv_out)

    beta = torch.sigmoid(block.log_beta)
    velocity = beta * velocity + mixer_out
    x = x + velocity

    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.float() * rms).to(x.dtype) * block.ffn_norm.weight
    x = x + block.ffn(normed)
    return x, velocity


def gqa_block_cached(block, x, velocity, freqs_cis, cache, name_prefix,
                     ttt_target=None, input_ids=None):
    """GQABlock / CodaGQABlock forward with KV-cache."""
    value_bias = None
    if hasattr(block, 'value_embedding') and block.value_embedding is not None and input_ids is not None:
        value_bias = block.value_embedding(input_ids)

    attn_out = attention_cached(
        block.attn, block.pre_norm(x), freqs_cis,
        cache, f"{name_prefix}.attn", value_bias=value_bias,
    )

    beta = torch.sigmoid(block.log_beta)
    velocity = beta * velocity + attn_out
    x = x + velocity

    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.float() * rms).to(x.dtype) * block.ffn_norm.weight

    ttt_mode = getattr(block, 'ttt_mode', 'none')
    if ttt_mode != "none" and ttt_target is not None:
        x = x + block.ffn(normed, ttt_target=ttt_target)
    else:
        x = x + block.ffn(normed)
    return x, velocity


# ---------------------------------------------------------------------------
# Cached forward for full JORMUNGANDR-HALO
# ---------------------------------------------------------------------------

def jormungandr_forward_cached(model, input_ids, cache):
    """Full cached forward pass for JORMUNGANDR-HALO."""
    B, T_new = input_ids.shape
    pos = cache.seq_pos

    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[pos:pos + T_new]

    # === PRELUDE ===
    velocity_768 = torch.zeros(B, T_new, model.d_prelude, device=h.device, dtype=h.dtype)
    h, velocity_768 = shortconv_block_cached(
        model.prelude_conv, h, velocity_768, cache, "prelude_conv"
    )
    h, velocity_768 = gqa_block_cached(
        model.prelude_gqa, h, velocity_768, freqs_cis, cache, "prelude_gqa"
    )
    input_embed = h

    # Cache proj_down(input_embed)
    input_embed_down = model.injection.proj_down(input_embed)

    # === FUSED ENTRY ===
    h_core = model.injection(model.injection.proj_down(h), input_embed_down)

    # === CORE LOOP ===
    velocity_512 = torch.zeros(B, T_new, model.d_core, device=h.device, dtype=h.dtype)
    n_iters = model.mean_recurrence

    for t in range(n_iters):
        h_core = model.injection(h_core, input_embed_down)
        for j, layer in enumerate(model.core_layers):
            h_core, velocity_512 = shortconv_block_cached(
                layer, h_core, velocity_512, cache, f"core_layers.{j}"
            )

    # === EXIT ===
    h = model.proj_up(h_core)

    # === CODA ===
    velocity_768 = torch.zeros(B, T_new, model.d_prelude, device=h.device, dtype=h.dtype)

    for i, layer in enumerate(model.coda_layers):
        if hasattr(layer, 'attn'):
            # GQA block
            ttt_mode = getattr(layer, 'ttt_mode', 'none')
            ttt_target = h if ttt_mode != "none" else None
            h, velocity_768 = gqa_block_cached(
                layer, h, velocity_768, freqs_cis, cache, f"coda_layers.{i}",
                ttt_target=ttt_target, input_ids=input_ids,
            )
        else:
            # ShortConv block
            h, velocity_768 = shortconv_block_cached(
                layer, h, velocity_768, cache, f"coda_layers.{i}"
            )

    logits = model.output(model.norm(h))
    cache.seq_pos += T_new
    return logits


# ---------------------------------------------------------------------------
# Cached forward for ARGUS-PRIME (simpler — no loop)
# ---------------------------------------------------------------------------

def argus_prime_forward_cached(model, input_ids, cache):
    """Full cached forward pass for ARGUS-PRIME."""
    B, T_new = input_ids.shape
    pos = cache.seq_pos

    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[pos:pos + T_new]

    velocity = torch.zeros(B, T_new, model.d_model, device=h.device, dtype=h.dtype)
    gqa_set = model.gqa_set

    context = None
    for i, layer in enumerate(model.layers):
        if model.use_film and model.film is not None and i == model.film_start:
            context = model.film.compute_context(h)

        if i in gqa_set:
            ttt_mode = getattr(layer, 'ttt_mode', 'none')
            ttt_target = h if ttt_mode != "none" else None
            h, velocity = gqa_block_cached(
                layer, h, velocity, freqs_cis, cache, f"layers.{i}",
                ttt_target=ttt_target, input_ids=input_ids,
            )
        else:
            h, velocity = shortconv_block_cached(
                layer, h, velocity, cache, f"layers.{i}"
            )

        if model.use_film and context is not None and i >= model.film_start:
            h = model.film.apply(h, context, i - model.film_start)

    logits = model.output(model.norm(h))
    cache.seq_pos += T_new
    return logits


# ---------------------------------------------------------------------------
# Auto-detect model type and dispatch
# ---------------------------------------------------------------------------

def cached_forward(model, input_ids, cache):
    """Dispatch to the right cached forward based on model type."""
    cls_name = model.__class__.__name__
    if 'Jormungandr' in cls_name:
        return jormungandr_forward_cached(model, input_ids, cache)
    else:
        return argus_prime_forward_cached(model, input_ids, cache)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, prompt_ids, max_tokens=200, temperature=0.8,
             top_k=40, top_p=0.9, max_seq_len=1024):
    """Autoregressive generation with KV-cache."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B = 1

    model.eval()

    # Create pre-allocated cache
    cache = create_cache(model, B, max_seq_len, device, torch.float16)

    # Prefill: process entire prompt at once
    prompt = prompt_ids.to(device)
    T_prompt = prompt.shape[1]

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = cached_forward(model, prompt, cache)

    generated = prompt[0].tolist()

    # Decode: one token at a time
    for step in range(max_tokens):
        next_logits = logits[0, -1, :] / temperature

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float('inf')

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            indices_to_remove = remove.scatter(0, sorted_idx, remove)
            next_logits[indices_to_remove] = -float('inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)

        if next_token == 50256 or cache.seq_pos >= max_seq_len:
            break

        # Feed single token through model with cache
        token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = cached_forward(model, token_input, cache)

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model(model_path, class_name):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def main():
    parser = argparse.ArgumentParser(description="KV-cached text generation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, ".")

    # Load model
    model = load_model(args.model, args.class_name)
    model = model.to(device)

    # Apply autokernel if available
    try:
        import autokernel
        model = autokernel.optimize(model, training=False)
        print("Applied autokernel.optimize()")
    except Exception:
        pass

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("  Warning: loaded with strict=False")
    if isinstance(ckpt, dict):
        print(f"  Step: {ckpt.get('step', '?')}")
    del ckpt, state_dict

    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    print(f"\nPrompt: {args.prompt}")
    print(f"Config: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_seq_len={args.max_seq_len}")
    print(f"KV-cache: pre-allocated to {args.max_seq_len} tokens")
    print("=" * 60)

    for i in range(args.num_samples):
        t0 = time.time()
        tokens = generate(
            model, prompt_ids, args.max_tokens,
            args.temperature, args.top_k, args.top_p, args.max_seq_len,
        )
        elapsed = time.time() - t0
        n_generated = len(tokens) - len(prompt_tokens)
        tok_s = n_generated / elapsed if elapsed > 0 else 0

        text = enc.decode(tokens)
        print(f"\n--- Sample {i+1} ({n_generated} tokens, {tok_s:.1f} tok/s) ---")
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
