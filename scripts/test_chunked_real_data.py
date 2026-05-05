"""Check chunked CE loss value matches PyTorch CE on real OdinHalo + BabyLM data."""
import sys, torch
sys.path.insert(0, '.')

from models.odin_halo import OdinHalo
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
from halo_training.data import BabyLMDataset, build_dataloader
import torch.nn.functional as F

device = 'cuda'
torch.manual_seed(42)

# Both models same seed → same init
torch.manual_seed(42)
model_base = OdinHalo(use_chunked_ce=False).to(device)
torch.manual_seed(42)
model_chunk = OdinHalo(use_chunked_ce=True).to(device)

model_base.train()
model_chunk.train()

chunked_loss_fn = ChunkedLinearCrossEntropyLoss(
    chunk_size=256, softcap=30.0, ignore_index=-100,
)

# Real BabyLM batch
ds = BabyLMDataset(root='datasets/babylm-odin32k.bin', block_size=256,
                   tokenizer_path='tokenizers/odin-32k/tokenizer.json')
dl = build_dataloader(ds, batch_size=4, num_workers=0, shuffle=False)
batch = next(iter(dl))
input_ids, targets = batch
input_ids = input_ids.to(device)
targets = targets.to(device)

print(f"input_ids: shape={tuple(input_ids.shape)} min={input_ids.min().item()} max={input_ids.max().item()}")
print(f"targets:   shape={tuple(targets.shape)}   min={targets.min().item()}   max={targets.max().item()}")

# Baseline: standard logits + F.cross_entropy (with softcap applied by model)
with torch.amp.autocast("cuda", dtype=torch.float16):
    logits = model_base(input_ids)
    print(f"baseline logits: shape={tuple(logits.shape)} dtype={logits.dtype}")
    loss_pt = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    print(f"  PyTorch CE loss: {loss_pt.item():.4f}")

# Chunked CE path
with torch.amp.autocast("cuda", dtype=torch.float16):
    h_low = model_chunk(input_ids)
    print(f"chunked h_low:   shape={tuple(h_low.shape)} dtype={h_low.dtype}")
    loss_ch = chunked_loss_fn(
        h_low.view(-1, h_low.size(-1)),
        model_chunk.lm_head.embed_table.weight,
        targets.view(-1),
    )
    print(f"  Chunked CE loss: {loss_ch.item():.4f}")

# PyTorch CE WITHOUT softcap (to check if softcap is the issue)
with torch.amp.autocast("cuda", dtype=torch.float16):
    # Manually construct logits without softcap
    h2 = model_chunk(input_ids)  # h_low
    logits_raw = F.linear(h2, model_chunk.lm_head.embed_table.weight)
    loss_pt_no = F.cross_entropy(logits_raw.view(-1, logits_raw.size(-1)), targets.view(-1))
    # And with softcap
    logits_sc = 30.0 * torch.tanh(logits_raw / 30.0)
    loss_pt_sc = F.cross_entropy(logits_sc.view(-1, logits_sc.size(-1)), targets.view(-1))
    print(f"  PyTorch no softcap:     {loss_pt_no.item():.4f}")
    print(f"  PyTorch with softcap=30: {loss_pt_sc.item():.4f}")
