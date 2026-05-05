"""Probe batch=32 feasibility + GQA block graph breaks."""
import sys, torch
sys.path.insert(0, '.')

import torch._dynamo as dynamo
from models.odin_halo import OdinHalo
from models._components import precompute_freqs_cis

device = 'cuda'
torch.manual_seed(42)

model = OdinHalo().to(device)
model.train()
freqs_cis = precompute_freqs_cis(model.head_dim, 256).to(device)

# Find GQA layer
from models.odin_halo import NoPEMoDAGQABlock
gqa = None
for layer in model.shared_layers:
    if isinstance(layer, NoPEMoDAGQABlock):
        gqa = layer
        break

print(f"GQA layer: {type(gqa).__name__}")

input_ids = torch.randint(0, model.vocab_size, (4, 256), device=device)
h = model.tok_embeddings(input_ids).half()

# GQA forward signature: (x, depth_kvs=[])
print("\n--- GQA block graph breaks ---")
dynamo.reset()
try:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        exp = dynamo.explain(gqa)(h, [])
    print(f"  graph_count={exp.graph_count} graph_break_count={exp.graph_break_count} op_count={exp.op_count}")
    if exp.break_reasons:
        for i, br in enumerate(exp.break_reasons[:5]):
            print(f"  break {i}: {br.reason[:200]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")
