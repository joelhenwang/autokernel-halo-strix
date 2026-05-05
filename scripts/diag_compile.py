"""Diagnose why torch.compile provides only ~8% lift on OdinHalo.

Usage:
  TORCH_LOGS=graph_breaks,recompiles python scripts/diag_compile.py
"""
import sys, os, torch, warnings, traceback
sys.path.insert(0, '.')

import models.components.conv_blocks as cb
from models.odin_halo import OdinHalo
from models._components import precompute_freqs_cis

device = 'cuda'
torch.manual_seed(42)

print("=" * 70)
print("DIAGNOSIS: torch.compile on OdinHalo")
print("=" * 70)

model = OdinHalo(use_chunked_ce=False).to(device)
model.train()

# Manually generate freqs_cis as the model expects
freqs_cis = precompute_freqs_cis(model.head_dim, 256).to(device)

# Compile one HyPEShortConvBlock layer to see where breaks occur
layer = model.shared_layers[0]
print(f"\nLayer 0 type: {type(layer).__name__}")

input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)
h = model.tok_embeddings(input_ids).half()

# --- Attempt 1: fullgraph=False (default) — allow breaks, see perf ---
print("\n--- Attempt 1: fullgraph=False (compile with graph breaks allowed) ---")
import torch._dynamo as dynamo
dynamo.reset()

compiled_layer = torch.compile(layer, mode="default", fullgraph=False)

try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = compiled_layer(h, freqs_cis)
    print(f"  PASS: output shape={out.shape if isinstance(out, torch.Tensor) else type(out)}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# --- Attempt 2: fullgraph=True — force no breaks, show error ---
print("\n--- Attempt 2: fullgraph=True (shows first break point) ---")
dynamo.reset()
full_layer = torch.compile(layer, mode="default", fullgraph=True)
try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = full_layer(h, freqs_cis)
    print("  PASS: no graph breaks!")
except Exception as e:
    print(f"  BROKE: {type(e).__name__}")
    err = str(e)
    # Print first 3000 chars of error (which shows break point)
    print(err[:3000])

# --- Attempt 3: Inspect HyPEShortConvBlock forward for complex tensor use ---
print("\n--- Inspect forward source for complex operators ---")
import inspect
src = inspect.getsource(cb.HyPEShortConvBlock.forward)
for i, line in enumerate(src.split('\n'), 1):
    if any(k in line for k in ['view_as_complex', 'view_as_real', 'torch.polar', 'freqs_cis']):
        print(f"  L{i}: {line.strip()}")

# --- Attempt 4: count graph breaks via explain ---
print("\n--- dynamo.explain() for shared layer ---")
dynamo.reset()
try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        explanation = dynamo.explain(layer)(h, freqs_cis)
    print(f"  graph_count: {explanation.graph_count}")
    print(f"  graph_break_count: {explanation.graph_break_count}")
    print(f"  op_count: {explanation.op_count}")
    if explanation.break_reasons:
        print(f"  break reasons (first 5):")
        for br in explanation.break_reasons[:5]:
            print(f"    - {br.reason[:200]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:500]}")
