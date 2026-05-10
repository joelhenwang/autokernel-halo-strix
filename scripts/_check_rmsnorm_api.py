"""Check what RMSNorm APIs are available in this PyTorch version."""
import torch
print("torch version:", torch.__version__)
print("has torch.nn.RMSNorm:", hasattr(torch.nn, "RMSNorm"))
print("has F.rms_norm:", hasattr(torch.nn.functional, "rms_norm"))

if hasattr(torch.nn, "RMSNorm"):
    m = torch.nn.RMSNorm(768, eps=1e-6).cuda().half()
    x = torch.randn(4, 512, 768, device="cuda", dtype=torch.float16)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        y = m(x)
    print("RMSNorm works:", y.shape, y.dtype, "norm:", y.float().norm().item())
