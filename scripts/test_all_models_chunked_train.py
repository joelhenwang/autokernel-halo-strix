"""Test smoke_test + 5-step training with --chunked-ce for each halo variant."""
import sys, time, torch
sys.path.insert(0, '.')

variants = [
    ("models.vidar_halo", "VidarHalo"),
    ("models.vidar_halo", "VidarHaloMini"),
    ("models.fenrir_halo", "FenrirHalo"),
    ("models.tyr_halo", "TyrHalo"),
    ("models.baldr_halo", "BaldrHalo"),
    ("models.chimera_halo", "ChimeraHalo"),
    ("models.odin_halo", "OdinHalo"),
    ("models.odin_halo", "OdinHaloMini"),
]

device = 'cuda'
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss
chunked_loss = ChunkedLinearCrossEntropyLoss(chunk_size=512)

for module_path, class_name in variants:
    torch.manual_seed(42)
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        m = cls(use_chunked_ce=True).to(device)
        m.train()
        optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        V = getattr(m, 'vocab_size', None) or m.tok_embeddings.embed.weight.shape[0]
        input_ids = torch.randint(0, V, (4, 256), device=device)
        targets = torch.randint(0, V, (4, 256), device=device)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = m(input_ids)
            if isinstance(out, dict):
                out = out.get("h_low") or out["logits"]
            loss = chunked_loss(
                out.view(-1, out.size(-1)),
                m.lm_head.embed_table.weight,
                targets.view(-1),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        print(f"  {class_name:20s} loss={loss.item():.3f}  output_shape={tuple(out.shape)}")
    except Exception as e:
        print(f"  {class_name:20s} FAIL: {type(e).__name__}: {str(e)[:150]}")
    finally:
        try: del m, optimizer, scaler
        except: pass
        torch.cuda.empty_cache()
