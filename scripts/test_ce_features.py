"""Correctness tests for new CE kernel features: softcap, z_loss, ignore_index, label_smoothing.

Compares HIP kernel against PyTorch reference on each feature in isolation
and in combination. Also covers both fused and tiny modes.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from kernel import ce_full


def pytorch_ref(logits, targets, softcap=0.0, ignore_index=-100, label_smoothing=0.0):
    """Reference implementation in fp32."""
    logits32 = logits.float()
    if softcap > 0:
        logits32 = softcap * torch.tanh(logits32 / softcap)
    return F.cross_entropy(
        logits32, targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def pytorch_grad(logits, targets, softcap=0.0, ignore_index=-100, label_smoothing=0.0):
    """Get reference gradient wrt logits (fp32)."""
    x = logits.float().detach().requires_grad_(True)
    if softcap > 0:
        y = softcap * torch.tanh(x / softcap)
    else:
        y = x
    loss = F.cross_entropy(
        y, targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    loss.backward()
    return x.grad


def test_case(name, B, V, *, softcap=0.0, ignore_index=-100, label_smoothing=0.0,
              ignore_frac=0.0, seed=42, mode="fused"):
    torch.manual_seed(seed)
    device = 'cuda'
    logits_fp16 = torch.randn(B, V, device=device, dtype=torch.float16) * 2.0
    targets = torch.randint(0, V, (B,), device=device, dtype=torch.long)

    if ignore_frac > 0:
        n_ignore = int(B * ignore_frac)
        perm = torch.randperm(B, device=device)
        targets[perm[:n_ignore]] = ignore_index

    # Reference loss (fp32)
    ref_loss = pytorch_ref(logits_fp16, targets, softcap, ignore_index, label_smoothing)
    ref_grad = pytorch_grad(logits_fp16, targets, softcap, ignore_index, label_smoothing)

    # HIP kernel loss + grad
    x = logits_fp16.clone().detach().requires_grad_(True)
    loss = ce_full(x, targets, softcap=softcap, ignore_index=ignore_index,
                   label_smoothing=label_smoothing, mode=mode)
    loss.backward()
    hip_grad = x.grad.float()

    loss_err = (loss.float() - ref_loss).abs().item()
    loss_rel = loss_err / max(abs(ref_loss.item()), 1e-6)

    grad_err = (hip_grad - ref_grad).abs()
    grad_max = grad_err.max().item()
    grad_mean = grad_err.mean().item()

    # Tolerances: fp16 storage introduces ~1e-3 baseline error
    loss_tol = 5e-3
    grad_tol = 5e-3
    passed = (loss_rel < loss_tol) and (grad_max < grad_tol)

    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name:50s} loss_err={loss_err:.4e} (rel {loss_rel:.2e})  "
          f"grad_max={grad_max:.4e} grad_mean={grad_mean:.4e}")
    return passed


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    all_pass = True
    B, V = 256, 1024

    print("=" * 70)
    print(f"CE FEATURE CORRECTNESS TESTS  (B={B}, V={V})")
    print("=" * 70)

    print("\n[fused mode]")
    all_pass &= test_case("baseline (no features)", B, V, mode="fused")
    all_pass &= test_case("softcap=30", B, V, softcap=30.0, mode="fused")
    all_pass &= test_case("softcap=10 (aggressive)", B, V, softcap=10.0, mode="fused")
    all_pass &= test_case("label_smoothing=0.1", B, V, label_smoothing=0.1, mode="fused")
    all_pass &= test_case("label_smoothing=0.2", B, V, label_smoothing=0.2, mode="fused")
    all_pass &= test_case("ignore 20% of targets", B, V, ignore_frac=0.2, mode="fused")
    all_pass &= test_case("softcap+ls combined", B, V, softcap=30.0, label_smoothing=0.1, mode="fused")
    all_pass &= test_case("softcap+ls+ignore combined", B, V, softcap=30.0,
                          label_smoothing=0.1, ignore_frac=0.2, mode="fused")

    print("\n[tiny mode]")
    all_pass &= test_case("baseline (no features)", B, V, mode="tiny")
    all_pass &= test_case("softcap=30", B, V, softcap=30.0, mode="tiny")
    all_pass &= test_case("label_smoothing=0.1", B, V, label_smoothing=0.1, mode="tiny")
    all_pass &= test_case("ignore 20% of targets", B, V, ignore_frac=0.2, mode="tiny")
    all_pass &= test_case("softcap+ls+ignore combined", B, V, softcap=30.0,
                          label_smoothing=0.1, ignore_frac=0.2, mode="tiny")

    print("\n[larger vocab V=32768]")
    all_pass &= test_case("prod shape baseline", 1024, 32768, mode="fused")
    all_pass &= test_case("prod shape +softcap=30", 1024, 32768, softcap=30.0, mode="fused")

    print("\n[z_loss emission]")
    torch.manual_seed(42)
    logits = torch.randn(B, V, device='cuda', dtype=torch.float16) * 2.0
    targets = torch.randint(0, V, (B,), device='cuda', dtype=torch.long)
    loss, lse = ce_full(logits, targets, return_z=True)
    lse_ref = torch.logsumexp(logits.float(), dim=-1)
    lse_err = (lse - lse_ref).abs().max().item()
    z_ok = lse_err < 1e-2
    print(f"  {'PASS' if z_ok else 'FAIL'} z_loss LSE emission: max_err={lse_err:.4e}")
    all_pass &= z_ok

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
