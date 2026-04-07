"""Validate Strix Halo (gfx1151) environment: PyTorch, ROCm, HIP, GPU availability."""
import shutil
import torch

print("TORCH_VERSION:", torch.__version__)
print("CUDA_AVAILABLE:", torch.cuda.is_available())

# ROCm / HIP version
hip_version = getattr(torch.version, "hip", None)
print("HIP_VERSION:", hip_version or "NOT FOUND (expected ROCm 7.2+)")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU_NAME:", props.name)
    print("GPU_COUNT:", torch.cuda.device_count())
    gcn_arch = getattr(props, "gcnArchName", "")
    print("GCN_ARCH:", gcn_arch or "NOT FOUND")
    if gcn_arch:
        if "gfx1151" in gcn_arch:
            print("TARGET_MATCH: Yes (Strix Halo gfx1151)")
        else:
            print("TARGET_MATCH: No (expected gfx1151, got", gcn_arch + ")")
    print("CU_COUNT:", props.multi_processor_count)
    print("MEMORY_GB:", round(props.total_memory / (1024 ** 3), 1))

    # Basic tensor test
    x = torch.randn(2, 2, device="cuda")
    print("TENSOR_TEST: is_cuda =", x.is_cuda, "device =", x.device)
else:
    print("ERROR: No GPU available")

# hipcc compiler check
hipcc = shutil.which("hipcc")
print("HIPCC:", hipcc or "NOT FOUND (required for HIP C++ kernel compilation)")

# rocm-smi check
rocm_smi = shutil.which("rocm-smi")
print("ROCM_SMI:", rocm_smi or "NOT FOUND")
