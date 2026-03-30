"""Validate MI300 environment: PyTorch, Triton, GPU availability."""
import torch
print("TORCH_VERSION:", torch.__version__)
print("CUDA_AVAILABLE:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU_NAME:", torch.cuda.get_device_name(0))
    print("GPU_COUNT:", torch.cuda.device_count())
    x = torch.randn(2, 2, device="cuda")
    print("TENSOR_TEST: is_cuda =", x.is_cuda, "device =", x.device)
else:
    print("ERROR: No GPU available")

try:
    import triton
    print("TRITON_VERSION:", triton.__version__)
    t = triton.runtime.driver.active.get_current_target()
    print("TRITON_BACKEND:", t.backend)
    print("TRITON_ARCH:", t.arch)
except Exception as e:
    print("TRITON_ERROR:", e)
