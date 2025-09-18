import time
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from triton_mlpblock import linear_forward

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def benchmark(
    B: int,
    T: int,
    FEAT_IN: int,
    FEAT_OUT: int,
    dtype=torch.float32,
    device="cuda",
    activation_type="gelu"
):
    x = torch.randn(B, T, FEAT_IN, dtype=dtype, device=device) * 0.1
    
    # Triton implementation
    triton_weight = torch.randn(FEAT_IN, FEAT_OUT, dtype=dtype, device=device) * 0.1
    triton_bias = torch.randn(FEAT_OUT, dtype=dtype, device=device) * 0.1

    # Torch implementation
    torch_weight = triton_weight.clone()
    torch_bias = triton_bias.clone()

    # --- warmup ---
    for _ in range(10):
        _ = linear_forward(x, triton_weight, triton_bias, activation_type=activation_type)#dotprod_precision="ieee")
        _ = torch.nn.functional.gelu(torch.nn.functional.linear(x, torch_weight.T, torch_bias))

    # --- benchmark triton ---
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = linear_forward(x, triton_weight, triton_bias, activation_type=activation_type)
        torch.cuda.synchronize()
        triton_ms = (time.time() - start) * 1e3 / 100

        # --- benchmark torch ---
        torch.cuda.synchronize()
        _torch_weight = torch_weight.T
        start = time.time()
        for _ in range(100):
            y_torch = torch.nn.functional.gelu(torch.nn.functional.linear(x, _torch_weight, torch_bias))
        torch.cuda.synchronize()
        torch_ms = (time.time() - start) * 1e3 / 100

    # correctness check
    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), "Mismatch"

    return triton_ms, torch_ms

if __name__ == "__main__":
    B, FEAT_IN, FEAT_OUT = 32, 1024, 4096
    
    results = {"T": [], "torch_ms": [], "triton_ms": []}
    T_max = 512

    for T in list(range(1, T_max + 1, 16)) + [T_max]:
        triton_ms, torch_ms = benchmark(B, T, FEAT_IN, FEAT_OUT)
        results["T"].append(T)
        results["triton_ms"].append(triton_ms)
        results["torch_ms"].append(torch_ms)
        print(f"| T={T:<4d} | Torch: {torch_ms:.3f} ms | Triton: {triton_ms:.3f} ms |")

    plt.plot(results["T"], results["torch_ms"], label="Torch")
    plt.plot(results["T"], results["triton_ms"], label="Triton (Fused)")
    plt.xlabel("T (sequence length)")
    plt.ylabel("Time per forward (ms)")
    plt.legend()
    plt.title("Activation Fused Linear: Benchmark (Torch vs Triton)")
    plt.grid(True)
    plt.savefig("triton_vs_torch_mlp.png")
    plt.close()
