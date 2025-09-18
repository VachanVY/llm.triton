import time

import triton
import torch
from torch import Tensor

from triton_layernorm import layernorm_forward
from torch_layernorm import LayerNorm

@torch.no_grad()
def benchmark(
    init_shape: tuple,  # (B, T=1, FEAT)
    T_max: int,      # get graph for performance with increasing T till T_max
    dtype=torch.float32,
    device="cuda",
):
    B, Tinit, FEAT = init_shape
    torch_ln_class = LayerNorm(FEAT, bias=True, eps=1e-5).to(device).to(dtype)
    wei = torch_ln_class.weight
    bias = torch_ln_class.bias

    results = {"T": [], "torch_ms": [], "triton_ms": [], "torchF_ms": []}

    # Sweep T from Tinit up to T_max
    for T in range(Tinit, T_max + 1):
        x = torch.randn(B, T, FEAT, dtype=dtype, device=device)*0.1

        # --- warmup ---
        for _ in range(10):
            _ = torch_ln_class(x)  # torch impl
            _ = layernorm_forward(x, wei, bias)        # triton impl

        # --- benchmark torch ---
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                y_torch = torch_ln_class(x)
            torch.cuda.synchronize()
            torch_ms = (time.time() - start) * 1e3 / 100

            # --- benchmark torch ---
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                y_torch = torch_ln_class(x, torch_version=True)
            torch.cuda.synchronize()
            torchF_ms = (time.time() - start) * 1e3 / 100

            # --- benchmark triton ---
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                y_triton = layernorm_forward(x, wei, bias)
            torch.cuda.synchronize()
            triton_ms = (time.time() - start) * 1e3 / 100

        # correctness check
        assert torch.allclose(y_torch, y_triton, atol=1e-5, rtol=1e-2), f"Mismatch at T={T}"

        # store results
        results["T"].append(T)
        results["torch_ms"].append(torch_ms)
        results["torchF_ms"].append(torchF_ms)
        results["triton_ms"].append(triton_ms)

        print(f"| T={T:<4d} | Torch: {torch_ms:.3f} ms | TorchF: {torchF_ms:.3f} ms | Triton: {triton_ms:.3f} ms |")

    return results



if __name__ == "__main__":
    B, FEAT = 32, 1024
    results = benchmark(init_shape=(B, 1, FEAT), T_max=512)

    import matplotlib.pyplot as plt
    plt.plot(results["T"], results["torch_ms"], label="Torch (custom)")
    plt.plot(results["T"], results["triton_ms"], label="Triton", color="red")
    plt.plot(results["T"], results["torchF_ms"], label="Torch (torch.nn.functional)", color="green", linestyle="--", alpha=0.5)
    plt.ylabel("Time per forward (ms)")
    plt.legend()
    plt.title("LayerNorm Benchmark (Torch vs Triton)")
    plt.grid(True)
    plt.savefig("triton_vs_torch_layernorm.png")
    plt.close()
