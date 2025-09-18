import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from triton_attn import flash_attn_forward

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def benchmark(B, num_heads, T, dim):
    from torch_attn import custom_scaled_dot_product_attention
    # Generate input tensors
    q = torch.randn(B, num_heads, T, dim, device="cuda").contiguous()
    k = torch.randn(B, num_heads, T, dim, device="cuda").contiguous()
    v = torch.randn(B, num_heads, T, dim, device="cuda").contiguous()

    maxlen = 768
    assert T <= maxlen, f"T={T} > maxlen={maxlen}"
    _attn_mask = torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen)
    attn_mask = torch.where(_attn_mask[:,:,:T,:T] == 0, float('-inf'), 0.0).cuda()

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        _ = flash_attn_forward(q, k, v, attn_mask=attn_mask)
        _ = custom_scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # Benchmark PyTorch
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_torch = custom_scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        torch.cuda.synchronize()
        torch_ms = (time.time() - start) * 1e3 / 100

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # internally uses float16 ig; so time difference may be larger than my triton impl
            y_torch0 = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        torch.cuda.synchronize()
        torchF_ms = (time.time() - start) * 1e3 / 100

        max_diff = (abs_diff:=(y_torch - y_torch0).abs()).max()
        atol, rtol = 1e-5, 1e-5
        if torch.backends.cuda.matmul.allow_tf32:
            atol, rtol = 1e-2, 1e-2  # More relaxed for TF32
        assert torch.allclose(y_torch, y_torch0, atol=atol, rtol=rtol), f"max diff: {max_diff.item():e} | mean diff: {abs_diff.mean().item():e}"

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_triton = flash_attn_forward(q, k, v, attn_mask, DOTPROD_PRECISION="tf32")
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) * 1e3 / 100

    # Check correctness
    max_diff = (abs_diff:=(y_torch0 - y_triton).abs()).max()
    assert torch.allclose(y_torch0, y_triton, atol=1e-2, rtol=0.0), f"max diff: {max_diff.item()} | mean diff: {abs_diff.mean().item()}"

    return torchF_ms, torch_ms, triton_ms

if __name__ == "__main__":
    B, num_heads, dim = 32, 96, 128
    results = {"T": [], "torchF_ms": [], "triton_ms": [], "torch_ms": []}

    # Sweep sequence lengths
    for T in list(range(1, 513, 16)) + [512]:
        torchF_ms, torch_ms, triton_ms = benchmark(B, num_heads, T, dim)
        results["T"].append(T)
        results["torchF_ms"].append(torchF_ms)
        results["torch_ms"].append(torch_ms)
        results["triton_ms"].append(triton_ms)
        print(f"| T={T:<4d} | Torch (custom): {torch_ms:.3f} ms | Torch (Flash): {torchF_ms:.3f} ms | Triton: {triton_ms:.3f} ms |")

    # Plot results
    plt.plot(results["T"], results["torchF_ms"], label="PyTorch Flash")
    plt.plot(results["T"], results["torch_ms"], label="PyTorch Custom SDPA")
    plt.plot(results["T"], results["triton_ms"], label="Triton Flash Attn", color="red")
    plt.xlabel("Sequence Length (T)")
    plt.ylabel("Time per forward (ms)")
    plt.legend()
    plt.title("Flash Attention Benchmark")
    plt.grid(True)
    plt.savefig("triton_vs_torch_flash_attn.png")
    plt.close()
