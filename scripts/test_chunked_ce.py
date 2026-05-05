"""Gradient parity test for ChunkedLinearCrossEntropyLoss vs naive linear+CE.

Verifies loss and gradients match reference PyTorch path at production shapes.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from kernels.hip.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyLoss


def reference_loss_and_grads(hidden, weight, targets, softcap=0.0, ignore_index=-100,
                              label_smoothing=0.0, z_loss_weight=0.0):
    """Reference: run linear in fp16 (match our implementation), CE in fp32 for stability."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)

    logits_fp16 = F.linear(h, w)                       # [N, V] fp16
    # Upcast to fp32 for CE (avoids fp16 overflow with label_smoothing).
    logits = logits_fp16.float()
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)

    loss = F.cross_entropy(
        logits, targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    if z_loss_weight > 0:
        valid = (targets != ignore_index).float()
        lse = torch.logsumexp(logits, dim=-1)
        n_valid = valid.sum().clamp(min=1.0)
        z_loss = ((lse * valid) ** 2).sum() / n_valid
        loss = loss + z_loss_weight * z_loss

    loss.backward()
    return loss.detach(), h.grad.detach(), w.grad.detach()


def chunked_loss_and_grads(hidden, weight, targets, chunk_size=256, **kwargs):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)

    loss_fn = ChunkedLinearCrossEntropyLoss(chunk_size=chunk_size, **kwargs)
    loss = loss_fn(h, w, targets)
    loss.backward()
    return loss.detach(), h.grad.detach(), w.grad.detach()


def rel_err(a, b, eps=1e-6):
    return ((a.float() - b.float()).abs() / (b.float().abs() + eps)).mean().item()


def test(name, N, D, V, *, softcap=0.0, ignore_index=-100, label_smoothing=0.0,
         z_loss_weight=0.0, chunk_size=256, ignore_frac=0.0, seed=42):
    torch.manual_seed(seed)
    device = 'cuda'

    hidden = torch.randn(N, D, device=device, dtype=torch.float16) * 0.1
    weight = torch.randn(V, D, device=device, dtype=torch.float16) * 0.02
    targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    if ignore_frac > 0:
        n_ignore = int(N * ignore_frac)
        perm = torch.randperm(N, device=device)
        targets[perm[:n_ignore]] = ignore_index

    ref_loss, ref_grad_h, ref_grad_w = reference_loss_and_grads(
        hidden, weight, targets,
        softcap=softcap, ignore_index=ignore_index,
        label_smoothing=label_smoothing, z_loss_weight=z_loss_weight,
    )
    opt_loss, opt_grad_h, opt_grad_w = chunked_loss_and_grads(
        hidden, weight, targets, chunk_size=chunk_size,
        softcap=softcap, ignore_index=ignore_index,
        label_smoothing=label_smoothing, z_loss_weight=z_loss_weight,
    )

    loss_err = (opt_loss.float() - ref_loss.float()).abs().item()
    loss_rel = loss_err / max(abs(ref_loss.item()), 1e-6)

    gh_err = (opt_grad_h - ref_grad_h).abs().max().item()
    gh_rel = rel_err(opt_grad_h, ref_grad_h)
    gw_err = (opt_grad_w - ref_grad_w).abs().max().item()
    gw_rel = rel_err(opt_grad_w, ref_grad_w)

    # Tolerances: fp16 matmul + fp16 softmax = looser than raw CE
    loss_tol = 1e-2
    grad_tol = 5e-2  # fp16 matmul accuracy

    passed = (loss_rel < loss_tol) and (gh_rel < grad_tol) and (gw_rel < grad_tol)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name:42s} loss_rel={loss_rel:.2e} "
          f"gh_max={gh_err:.2e} gh_rel={gh_rel:.2e} gw_max={gw_err:.2e} gw_rel={gw_rel:.2e}")
    return passed


def memory_test(N, D, V, chunk_size):
    """Verify peak memory reduction."""
    torch.manual_seed(42)
    device = 'cuda'
    hidden = torch.randn(N, D, device=device, dtype=torch.float16) * 0.1
    weight = torch.randn(V, D, device=device, dtype=torch.float16) * 0.02
    targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)

    # Naive
    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    logits = F.linear(h1, w1)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    naive_peak = torch.cuda.max_memory_allocated() / 1e6

    del h1, w1, logits, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Chunked
    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    loss_fn = ChunkedLinearCrossEntropyLoss(chunk_size=chunk_size)
    loss2 = loss_fn(h2, w2, targets)
    loss2.backward()
    chunked_peak = torch.cuda.max_memory_allocated() / 1e6

    print(f"  naive_peak={naive_peak:.1f} MB  chunked_peak={chunked_peak:.1f} MB  "
          f"saved={naive_peak-chunked_peak:.1f} MB  ({100*(naive_peak-chunked_peak)/naive_peak:.1f}%)")
    return chunked_peak < naive_peak


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    all_pass = True
    print("=" * 75)
    print("ChunkedLinearCrossEntropy GRADIENT PARITY")
    print("=" * 75)

    print("\n[small (N=256, D=128, V=1024)]")
    all_pass &= test("baseline", 256, 128, 1024)
    all_pass &= test("softcap=30", 256, 128, 1024, softcap=30.0)
    all_pass &= test("label_smoothing=0.1", 256, 128, 1024, label_smoothing=0.1)
    all_pass &= test("ignore 20%", 256, 128, 1024, ignore_frac=0.2)
    all_pass &= test("z_loss=1e-4", 256, 128, 1024, z_loss_weight=1e-4)
    all_pass &= test("all combined", 256, 128, 1024,
                     softcap=30.0, label_smoothing=0.1, ignore_frac=0.2, z_loss_weight=1e-4)

    print("\n[medium (N=512, D=256, V=4096), chunk=128]")
    all_pass &= test("baseline", 512, 256, 4096, chunk_size=128)
    all_pass &= test("softcap=30", 512, 256, 4096, chunk_size=128, softcap=30.0)

    print("\n[OdinHalo-like (N=1024, D=256, V=32768)]")
    all_pass &= test("baseline", 1024, 256, 32768, chunk_size=256)
    all_pass &= test("softcap=30", 1024, 256, 32768, chunk_size=256, softcap=30.0)
    all_pass &= test("z_loss=1e-4", 1024, 256, 32768, chunk_size=256, z_loss_weight=1e-4)

    print("\n[chunk size sweep (N=1024, D=256, V=32768)]")
    for cs in [128, 256, 512, 1024]:
        all_pass &= test(f"chunk_size={cs}", 1024, 256, 32768, chunk_size=cs)

    print("\n[Memory comparison (N=1024, D=256, V=32768, chunk=256)]")
    memory_test(1024, 256, 32768, 256)
    print("  [naive reference uses F.cross_entropy which internally allocates logits]")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 75)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
