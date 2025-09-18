import math

from triton import language as tl
import triton
from torch import Tensor
import torch

@triton.jit
def exp(x):
    """why use tl.exp2 not tl.exp: https://github.com/triton-lang/triton/issues/2893#issuecomment-1909910123"""
    return tl.exp2(1.4426950408889634 * x)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BR': 16, 'BLOCK_BC': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_BR': 16, 'BLOCK_BC': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_BR': 32, 'BLOCK_BC': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_BR': 64, 'BLOCK_BC': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_BR': 64, 'BLOCK_BC': 64}, num_stages=3, num_warps=8),
    ],
    key=['dim'],  # dimensions for tuning
)
@triton.jit
def _fused_flash_attention_forward_kernel(
    q_ptr:     tl.tensor, # (B, num_heads, T, dim)
    k_ptr:tl.tensor, # (B, num_heads, T, dim).T = (B, num_heads, dim, T)
    v_ptr:     tl.tensor, # (B, num_heads, T, dim)
    mask_ptr:  tl.tensor, # (T, T) # including a separate mask bcause i can pass any kind of mask now; tldr: flexibility
    out_ptr:   tl.tensor, # (B, num_heads, T, dim)
    # ------------------------------------ STRIDE STUFF ------------------------------------------------ #
    qB_stride0:tl.constexpr, qNH_stride1:tl.constexpr, qT_stride2:tl.constexpr, qDIM_stride3:tl.constexpr,
    kB_stride0:tl.constexpr, kNH_stride1:tl.constexpr, kT_stride2:tl.constexpr, kDIM_stride3:tl.constexpr,
    vB_stride0:tl.constexpr, vNH_stride1:tl.constexpr, vT_stride2:tl.constexpr, vDIM_stride3:tl.constexpr,
    mT_stride0:tl.constexpr, mT_stride1: tl.constexpr,
    oB_stride0:tl.constexpr, oNH_stride1:tl.constexpr, oT_stride2:tl.constexpr, oDIM_stride3:tl.constexpr,
    # ------------------------------------ STRIDE STUFF ------------------------------------------------ #
    T:int, dim:tl.constexpr,
    # ------------------ BLOCK STUFF ---------------------- #
    BLOCK_BR:tl.constexpr, # BLOCK SIZE ALONG `T` for Q
    BLOCK_BC:tl.constexpr, # BLOCK SIZE ALONG `T` for K and V
    # ------------------ BLOCK STUFF ---------------------- #
    sm_scale:tl.constexpr,
    DOTPROD_PRECISION:tl.constexpr # "tf32" or "ieee"
):
    Bid = tl.program_id(0)
    NHid = tl.program_id(1)
    # first for loop in Psedo Code Algo in paper # we will not write the for loop, we will parallelize it; so...
    Q_tile_id = tl.program_id(2) # q tile id

    # get Q,K,V tile Pointer
    q_ptr = q_ptr + (Bid * qB_stride0 + NHid * qNH_stride1)   # Q[Bid, NHid, :, :]
    qo_Trange = tl.arange(0, BLOCK_BR) + BLOCK_BR * Q_tile_id # (BLOCK_BR,)
    dimrange = tl.arange(0, dim)
    qo_range = (qo_Trange[:, None] * qT_stride2 + dimrange[None, :] * qDIM_stride3) # (BLOCK_BR, dim)
    qo_mask = (qo_Trange[:, None] < T) & (dimrange[None, :] < dim)                  # (BLOCK_BR, dim)
    q_blc = tl.load(q_ptr + qo_range, mask=qo_mask, other=0.0)                      # (BLOCK_BR, dim)

    k_ptr = k_ptr + (Bid * kB_stride0 + NHid * kNH_stride1) # K[Bid, NHid, :, :]
    v_ptr = v_ptr + (Bid * vB_stride0 + NHid * vNH_stride1) # V[Bid, NHid, :, :]

    # init (new max, max), (new norma, norma)
    prev_max_blc = tl.full([BLOCK_BR], value=float("-inf"), dtype=tl.float32)
    prev_norma_blc = tl.zeros_like(prev_max_blc)

    # init out_blc
    out_blc = tl.zeros([BLOCK_BR, dim], dtype=tl.float32) # (BLOCK_BR, dim)

    # for loop across `TC` (number of blocks along `T` for K and V) with block size `BLOCK_BC`
    for kv_blc_num in tl.range(0, tl.cdiv(T, BLOCK_BC)): # btw we can't parallelize this... obviously
        kv_Trange = tl.arange(0, BLOCK_BC) + BLOCK_BC * kv_blc_num # (BLOCK_BC,)

        # load mask block
        attn_mask_range = qo_Trange[:, None] * mT_stride0 + kv_Trange[None, :] * mT_stride1            # (BLOCK_BR, BLOCK_BC)
        attn_mask_mask = (qo_Trange[:, None] < T) & (kv_Trange[None, :] < T) # (BLOCK_BR, BLOCK_BC)
        mask_blc = tl.load(mask_ptr + attn_mask_range, mask=attn_mask_mask, other=float("-inf"))  # (BLOCK_BR, BLOCK_BC)

        # load k, v
        krange = kv_Trange[:, None] * kT_stride2 + dimrange[None, :] * kDIM_stride3 # (BLOCK_BC, dim)
        kmask = (kv_Trange[:, None] < T) & (dimrange[None, :] < dim)   # (BLOCK_BC, dim)
        k_blc = tl.load(k_ptr + krange, mask=kmask, other=0.0) # (BLOCK_BC, dim)
        k_trans_blc = tl.trans(k_blc, (1, 0))         # (BLOCK_BC, dim).T = (dim, BLOCK_BC)

        vrange = kv_Trange[:, None] * vT_stride2 + dimrange[None, :] * vDIM_stride3 # (BLOCK_BC, dim)
        vmask = (kv_Trange[:, None] < T) & (dimrange[None, :] < dim)   # (BLOCK_BC, dim)
        v_blc = tl.load(v_ptr + vrange, mask=vmask, other=0.0) # (BLOCK_BC, dim)

        # dot prod
        S_blc = tl.dot(q_blc, k_trans_blc, input_precision=DOTPROD_PRECISION) * sm_scale # (BLOCK_BR, BLOCK_BC)
        S_blc += mask_blc # (BLOCK_BR, BLOCK_BC)

        # handle maxes and normas
        rowmax = tl.max(S_blc, axis=1, keep_dims=False)  # (BLOCK_BR,)
        curr_max_blc = tl.maximum(prev_max_blc, rowmax) # (BLOCK_BR,)
        nonorm_softmax = exp(S_blc - curr_max_blc[:, None]) # (BLOCK_BR, BLOCK_BC) # P in paper
        correction_factor = exp(prev_max_blc - curr_max_blc) # (BLOCK_BR,)
        curr_norma_blc = correction_factor * prev_norma_blc + tl.sum(nonorm_softmax, axis=1) # (BLOCK_BR,)
        out_blc = (
            correction_factor[:, None] * out_blc +              # (BLOCK_BR, 1) * (BLOCK_BR, dim) = (BLOCK_BR, dim)
            tl.dot(nonorm_softmax, v_blc, input_precision=DOTPROD_PRECISION) # (BLOCK_BR, BLOCK_BC) @ (BLOCK_BC, dim) = (BLOCK_BR, dim)
        )

        # assign curr to prev for next iteration
        prev_max_blc = curr_max_blc
        prev_norma_blc = curr_norma_blc

    out_blc = out_blc / prev_norma_blc[:, None] # (BLOCK_BR, dim)

    # store computed stuff to out pointer
    out_ptr = out_ptr + (Bid * oB_stride0 + NHid * oNH_stride1)
    out_range = qo_Trange[:, None] * oT_stride2 + dimrange[None, :] * oDIM_stride3 # (BLOCK_BR, dim)
    tl.store(out_ptr + out_range, out_blc, mask=qo_mask)


def flash_attn_forward(
    q:Tensor, # (B, num_heads, T, dim)
    k:Tensor, # (B, num_heads, T, dim)
    v:Tensor, # (B, num_heads, T, dim)
    attn_mask:Tensor, # (1, 1, T, T)
    **kwargs
):
    B, num_heads, T, dim = q.shape
    attn_mask = attn_mask[0, 0] # (T, T)

    # q, k, v = (ts.contiguous() for ts in (q, k, v))
    grid = lambda meta: (
        B,
        num_heads,
        triton.cdiv(T, meta['BLOCK_BR']),
    )

    out = torch.empty_like(q) # (B, num_heads, T, dim)
    _fused_flash_attention_forward_kernel[grid](
        q, k, v, attn_mask, out, 
        *q.stride(), *k.stride(), *v.stride(),
        *attn_mask.stride(), *out.stride(), 
        T, dim, sm_scale=(1/(dim**0.5)),
        DOTPROD_PRECISION=kwargs.get("DOTPROD_PRECISION", "ieee")
    )
    return out

if __name__ == "__main__":
    import sys
    try: DOTPROD_PRECISION=sys.argv[1] # "tf32" or "ieee"
    except: DOTPROD_PRECISION="ieee"
    assert DOTPROD_PRECISION in ["tf32", "ieee"], f"{DOTPROD_PRECISION=}"
    if DOTPROD_PRECISION=="tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for T in [1, 2, 3, 4, 5, 8, 16, 32, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024]:
        SHAPE = (B, num_heads, T, dim) = 16, 8, T, 64
        q, k, v = (torch.randn(SHAPE, device="cuda") for _ in range(3))
        maxlen = T
        _attn_mask = torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen)
        attn_mask = torch.where(_attn_mask[:,:,:T,:T] == 0, float('-inf'), 0.0).cuda()
        # attn_mask = torch.ones((1, 1, T, T), device="cuda") # no mask

        with torch.no_grad():
            torch_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False
            )
            triton_out = flash_attn_forward(q, k, v, attn_mask, DOTPROD_PRECISION=DOTPROD_PRECISION)

        max_diff = (abs_diff:=(torch_out - triton_out).abs()).max()
        rtol = 0.0 if DOTPROD_PRECISION=="tf32" else 1e-5
        atol = 1e-2 if DOTPROD_PRECISION=="tf32" else 1e-5
        print(f"| {T=:} | Max diff: {max_diff.item():e} | Mean diff: {abs_diff.mean().item():e} |", torch.allclose(torch_out, triton_out, atol=atol, rtol=rtol))
        torch.testing.assert_close(torch_out, triton_out, atol=atol, rtol=rtol)

    print("Success!")