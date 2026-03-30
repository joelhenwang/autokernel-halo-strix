import torch
import sys
sys.path.insert(0, '.')
import reference

torch.manual_seed(42)
x = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)
half_dim = 32
cos = torch.randn(64, half_dim, device='cuda', dtype=torch.float16)
sin = torch.randn(64, half_dim, device='cuda', dtype=torch.float16)

expected = reference.rotary_embedding_ref(x, cos, sin)

import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
output = mod.kernel_fn(x, cos, sin)

diff = (output.float() - expected.float()).abs()
max_diff = diff.max().item()
print(f"max_diff: {max_diff}")

flat_idx = diff.view(-1).argmax().item()
total_head_dim = 64
row = flat_idx // total_head_dim
col = flat_idx % total_head_dim
print(f"max_diff at flattened row={row}, col={col}")
print(f"output val: {output.view(-1, total_head_dim)[row, col].item()}")
print(f"expected val: {expected.view(-1, total_head_dim)[row, col].item()}")

# Check first few elements
print("\nFirst row comparison (first 8 elements):")
print(f"output:   {output.view(-1, 64)[0, :8].tolist()}")
print(f"expected: {expected.view(-1, 64)[0, :8].tolist()}")
