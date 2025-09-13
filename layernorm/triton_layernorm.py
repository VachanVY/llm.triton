import triton
import triton.language as tl
import torch
from torch import Tensor

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_FEAT': 32},  num_warps=4),
        triton.Config({'BLOCK_FEAT': 64},  num_warps=4),
        triton.Config({'BLOCK_FEAT': 128}, num_warps=4),
        triton.Config({'BLOCK_FEAT': 256}, num_warps=4),
        triton.Config({'BLOCK_FEAT': 512}, num_warps=8),
    ],
    key=['FEAT'],  # a list of argument names whose change in value will trigger the evaluation of all provided configs.
)
@triton.jit
def _layernorm_forward_kernel(
    x_ptr,   # (M, FEAT)
    out_ptr, # (M, FEAT)
    wei_ptr, # (FEAT,)
    bias_ptr, # (FEAT,)
    FEAT: tl.constexpr, # number of columns
    BLOCK_FEAT: tl.constexpr, # number of columns per block # Tile Size
    eps: tl.constexpr,
):
    row_id = tl.program_id(0)

    x_ptr   = x_ptr   + FEAT * row_id
    out_ptr = out_ptr + FEAT * row_id

    # find `mean` in tiles of `BLOCK_FEAT` along column dimension
    temp_mu = tl.zeros([BLOCK_FEAT], tl.float32)
    for strd in tl.range(0, FEAT, BLOCK_FEAT):
        # [0:B-1, B:2B-1, 2B:3B-1, ...]
        cols = tl.arange(0, BLOCK_FEAT) + strd
        x_rc = tl.load(x_ptr + cols, mask=cols < FEAT, other=0.0) # (BLOCK_FEAT,)
        temp_mu += x_rc
    mu = tl.sum(temp_mu)/FEAT # mean of a row

    # find `var` in tiles of `BLOCK_FEAT` along column dimension
    temp_var = tl.zeros_like(temp_mu)
    for strd in tl.range(0, FEAT, BLOCK_FEAT):
        cols = tl.arange(0, BLOCK_FEAT) + strd
        x_rc = tl.load(x_ptr + cols, mask=cols < FEAT, other=0.0) # (BLOCK_FEAT,)
        x_rc_sq = (x_rc - mu) * (x_rc - mu)
        temp_var += tl.where(cols < FEAT, x_rc_sq, 0.0)
    rsigma = tl.rsqrt(tl.sum(temp_var)/FEAT + eps) # standard deviation of a row

    # norma using `mean` and `var` | renorma using `wei` and `bias`
    for strd in tl.range(0, FEAT, BLOCK_FEAT):
        cols = tl.arange(0, BLOCK_FEAT) + strd
        mask = cols < FEAT

        x_rc = tl.load(x_ptr + cols, mask, other=0.0)

        wei_c = tl.load(wei_ptr + cols, mask, other=0.0)
        bias_c = tl.load(bias_ptr + cols, mask, other=0.0)

        x_norma = (x_rc - mu) * rsigma
        x_renorma = x_norma * wei_c + bias_c

        tl.store(out_ptr + cols, x_renorma, mask)


def layernorm_forward(x:Tensor, weight:Tensor, bias:Tensor, eps:float=1e-5): # (B, T, d_model)
    B, T, FEAT = x.size()
    x = x.contiguous().view(-1, FEAT)
    
    out = torch.empty_like(x)

    BLOCK_BT = 1
    BLOCK_FEAT = triton.next_power_of_2(FEAT) # 2 ^ BLOCK_FEAT >= FEAT such that BLOCK_FEAT is the smallest power satisfying this condition

    grid = (triton.cdiv(B*T, BLOCK_BT),)
    _layernorm_forward_kernel[grid](
        x, out, weight, bias, FEAT, eps=eps
    )
    out = out.view(B, T, FEAT)
    return out


if __name__ == "__main__":
    from torch_layernorm import LayerNorm

    for B in [2, 15, 32, 127]:
        for T in [2048, 1025]:
            for FEAT in [256, 512, 1035]:
                torch.manual_seed(B*T*FEAT)
                print(f"test case: B={B}, T={T}, FEAT={FEAT}", end=" ")
                x = torch.randn(B, T, FEAT).cuda()

                with torch.no_grad():
                    torch_out = (torch_ln_class:=LayerNorm(ndim=FEAT, bias=True, eps=1e-5).cuda())(x, torch_version=False)
                triton_out = layernorm_forward(x, torch_ln_class.weight, torch_ln_class.bias, eps=1e-5)
                
                torch.testing.assert_close(torch_out, triton_out)
                print("test passed!")