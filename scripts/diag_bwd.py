import sys, time, torch
sys.path.insert(0, '.')
import kernel as opt_ce

device = 'cuda'
B, V = 4096, 32768
logits = torch.randn(B, V, device=device, dtype=torch.float16, requires_grad=True)
targets = torch.randint(0, V, (B,), device=device)

# Warmup
for _ in range(5):
    loss = opt_ce.kernel_fn(logits, targets)
    loss.backward()
    logits.grad.zero_()
    loss_t = torch.nn.functional.cross_entropy(logits, targets)
    loss_t.backward()
    logits.grad.zero_()
torch.cuda.synchronize()

N = 50

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = opt_ce.kernel_fn(logits, targets)
    loss.backward()
    logits.grad.zero_()
torch.cuda.synchronize()
opt_total = (time.perf_counter() - t0) / N * 1000

logits.requires_grad_(False)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = opt_ce.kernel_fn(logits, targets)
torch.cuda.synchronize()
opt_fwd = (time.perf_counter() - t0) / N * 1000
logits.requires_grad_(True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    logits.grad.zero_()
torch.cuda.synchronize()
torch_total = (time.perf_counter() - t0) / N * 1000

logits.requires_grad_(False)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = torch.nn.functional.cross_entropy(logits, targets)
torch.cuda.synchronize()
torch_fwd = (time.perf_counter() - t0) / N * 1000
logits.requires_grad_(True)

opt_bwd = opt_total - opt_fwd
torch_bwd = torch_total - torch_fwd

print(f'component       opt (ms)   pytorch (ms)   speedup')
print(f'forward         {opt_fwd:>8.2f}   {torch_fwd:>12.2f}   {torch_fwd/opt_fwd:>6.2f}x')
print(f'backward        {opt_bwd:>8.2f}   {torch_bwd:>12.2f}   {torch_bwd/opt_bwd:>6.2f}x')
print(f'fwd+bwd         {opt_total:>8.2f}   {torch_total:>12.2f}   {torch_total/opt_total:>6.2f}x')
