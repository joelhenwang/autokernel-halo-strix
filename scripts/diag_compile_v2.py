"""Verify compile-friendly HyPEShortConvBlock has fewer/zero graph breaks."""
import sys, os, torch, traceback
sys.path.insert(0, '.')

import models.components.conv_blocks as cb
from models.odin_halo import OdinHalo
from models._components import precompute_freqs_cis
import torch._dynamo as dynamo

device = 'cuda'
torch.manual_seed(42)

model = OdinHalo(use_chunked_ce=False).to(device)
model.train()
freqs_cis = precompute_freqs_cis(model.head_dim, 256).to(device)
input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)
h = model.tok_embeddings(input_ids).half()

layer = model.shared_layers[0]
print(f"Layer type: {type(layer).__name__}")

# --- Baseline (with HIP fused kernel) ---
print("\n[1] Baseline (HIP fused_rope_gate_mul)")
layer._compile_friendly = False
dynamo.reset()
try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        exp = dynamo.explain(layer)(h, freqs_cis)
    print(f"  graph_count={exp.graph_count} graph_break_count={exp.graph_break_count} op_count={exp.op_count}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# --- Compile-friendly path ---
print("\n[2] Compile-friendly (native PyTorch RoPE+gate)")
layer._compile_friendly = True
dynamo.reset()
try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        exp = dynamo.explain(layer)(h, freqs_cis)
    print(f"  graph_count={exp.graph_count} graph_break_count={exp.graph_break_count} op_count={exp.op_count}")
    if exp.break_reasons:
        print("  remaining breaks:")
        for br in exp.break_reasons[:3]:
            print(f"    - {br.reason[:200]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

# --- Fullgraph test ---
print("\n[3] fullgraph=True with compile-friendly")
layer._compile_friendly = True
dynamo.reset()
try:
    full_layer = torch.compile(layer, mode="default", fullgraph=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = full_layer(h, freqs_cis)
    print(f"  PASS: compiled cleanly, output shape {out.shape}")
except Exception as e:
    print(f"  BROKE: {type(e).__name__}")
    err_str = str(e)
    print(err_str[:2000])
