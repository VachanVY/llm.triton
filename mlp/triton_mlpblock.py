import triton.language as tl
import triton
from torch import Tensor
import torch

@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 16, 'BLOCK_T': 16, 'BLOCK_FEAT_IN': 32, 'BLOCK_FEAT_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_T': 16, 'BLOCK_FEAT_IN': 32, 'BLOCK_FEAT_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_T': 32, 'BLOCK_FEAT_IN': 32, 'BLOCK_FEAT_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_T': 16, 'BLOCK_FEAT_IN': 64, 'BLOCK_FEAT_OUT': 32}, num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_T': 16, 'BLOCK_FEAT_IN': 32, 'BLOCK_FEAT_OUT': 64}, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_T': 32, 'BLOCK_FEAT_IN': 64, 'BLOCK_FEAT_OUT': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_T': 64, 'BLOCK_FEAT_IN': 32, 'BLOCK_FEAT_OUT': 32}, num_warps=8, num_stages=3),
    ],
    key=['FEAT_IN', 'FEAT_OUT'],  # Only autotune when model architecture changes, not B/T
)
@triton.jit
def _fused_linear_forward_kernel(
    x_ptr,      # (B, T, FEAT_IN)
    out_ptr,    # (B, T, FEAT_OUT)
    wei_ptr,    # (1, FEAT_IN, FEAT_OUT)
    bias_ptr,   # (FEAT_OUT,)
    B:tl.constexpr, T:tl.constexpr, 
    FEAT_IN:tl.constexpr, FEAT_OUT:tl.constexpr,
    BLOCK_B:tl.constexpr, BLOCK_T:tl.constexpr, 
    BLOCK_FEAT_IN:tl.constexpr, BLOCK_FEAT_OUT:tl.constexpr,
    activation_type:tl.constexpr # gelu or none
):
    Bid = tl.program_id(2)
    T_id = tl.program_id(1)
    FOUT_id = tl.program_id(0)

    Brange = (tl.arange(0, BLOCK_B) + Bid * BLOCK_B)[:, None, None]                       # (BLOCK_B, 1, 1)
    Trange = (tl.arange(0, BLOCK_T) + T_id * BLOCK_T)[None, :, None]                      # (1, BLOCK_T, 1)
    Fout_range = (tl.arange(0, BLOCK_FEAT_OUT) + FOUT_id * BLOCK_FEAT_OUT)[None, None, :] # (1, 1, BLOCK_FEAT_OUT)

    accumulator = tl.zeros((BLOCK_B, BLOCK_T, BLOCK_FEAT_OUT), tl.float32)
    for strd in tl.range(0, FEAT_IN, BLOCK_FEAT_IN):
        Fin_range = tl.arange(0, BLOCK_FEAT_IN) + strd # (BLOCK_FEAT_IN,)
        x_range = (
            Brange * T * FEAT_IN + 
            Trange * FEAT_IN + 
            Fin_range[None, None, :]
        ) # (BLOCK_B, BLOCK_T, BLOCK_FEAT_IN)
        xmask = (Brange < B) & (Trange < T) & (Fin_range[None, None, :] < FEAT_IN)
        x = tl.load(x_ptr + x_range, mask=xmask, other=0.0) # (BLOCK_B, BLOCK_T, BLOCK_FEAT_IN)

        wei_range = (
            Fin_range[None, :, None] * FEAT_OUT +
            Fout_range
        ) # (1, BLOCK_FEAT_IN, BLOCK_FEAT_OUT)
        wei_mask = (Fin_range[None, :, None] < FEAT_IN) & (Fout_range < FEAT_OUT)
        wei = tl.load(wei_ptr + wei_range, mask=wei_mask, other=0.0)
        wei = tl.broadcast_to(wei, (BLOCK_B, BLOCK_FEAT_IN, BLOCK_FEAT_OUT)) # IS THIS NEEDED? DOES TRITON HANDLE `wei` WITHOUT THIS?

        accumulator += tl.dot(x, wei)

    bias_range = Fout_range # (BLOCK_FEAT_OUT,)
    bias_mask = (Fout_range < FEAT_OUT)
    bias = tl.load(
        bias_ptr + bias_range,
        bias_mask,
        other=0.0
    )
    accumulator += bias
    if activation_type == "gelu":
        accumulator = gelu(accumulator)

    out_range = (
        Brange * T * FEAT_OUT + 
        Trange * FEAT_OUT +
        Fout_range
    ) # (BLOCK_B, BLOCK_T, BLOCK_FEAT_OUT)
    out_mask = (Brange < B) & (Trange < T) & (Fout_range < FEAT_OUT)
    tl.store(out_ptr + out_range, accumulator, mask=out_mask)



def linear_forward(
    x:Tensor,       # (B, T, FEAT_IN)
    weight:Tensor,  # (FEAT_IN, FEAT_OUT)
    bias:Tensor,    # (FEAT_OUT,)
    activation_type:str="none"
) -> Tensor:
    assert activation_type in ["gelu", "none"]
    B, T, FEAT_IN = x.size()
    FEAT_OUT = weight.size(1)

    weight = weight.contiguous().view(1, FEAT_IN, FEAT_OUT)

    out = torch.empty((B, T, FEAT_OUT), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(FEAT_OUT, META['BLOCK_FEAT_OUT']), 
        triton.cdiv(T, META['BLOCK_T']), 
        triton.cdiv(B, META['BLOCK_B'])
    )
    _fused_linear_forward_kernel[grid](
        x, out, weight, bias,
        B, T, FEAT_IN, FEAT_OUT,
        activation_type=activation_type
    )
    return out

if __name__ == "__main__":
    for B in [2, 15, 32, 127]:
        for T in [32, 64, 125]:
            for FEAT_IN in [128, 512, 1024]:
                for FEAT_OUT in [510, 1024]:
                    x = torch.rand((B, T, FEAT_IN), device="cuda", dtype=torch.float32)*0.01
                    weight = torch.rand((FEAT_IN, FEAT_OUT), device="cuda", dtype=torch.float32)*0.01
                    bias = torch.rand((FEAT_OUT,), device="cuda", dtype=torch.float32)*0.01
                    out_triton = linear_forward(x, weight, bias, activation_type="gelu")

                    out_torch = torch.nn.functional.gelu(torch.nn.functional.linear(x, weight.T, bias))
                    print(f"B={B}, T={T}, FEAT_IN={FEAT_IN}, FEAT_OUT={FEAT_OUT}", end=" ")
                    print(True if not torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3, equal_nan=True) else False)
