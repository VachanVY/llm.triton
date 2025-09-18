import math
import torch.nn.functional as F

from torch import nn, Tensor
import torch

def custom_scaled_dot_product_attention(q, k, v, attn_mask=None):
    B, num_heads, T, dim = q.shape

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
    if attn_mask is not None:
        att = att + attn_mask
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, num_heads, T, T) x (B, num_heads, T, dim) -> (B, num_heads, T, dim)
    return y

class CausalSelfAttention(nn.Module):
    def __init__(self, config, flash_attn:bool=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, 'scaled_dot_product_attention') and flash_attn
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x:Tensor):
        B, T, FEAT = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q:Tensor; k:Tensor; v:Tensor
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, FEAT // self.n_head).transpose(1, 2) # (B, num_heads, T, dim)
        q = q.view(B, T, self.n_head, FEAT // self.n_head).transpose(1, 2) # (B, num_heads, T, dim)
        v = v.view(B, T, self.n_head, FEAT // self.n_head).transpose(1, 2) # (B, num_heads, T, dim)

        # causal self-attention; Self-attend: (B, num_heads, T, dim) x (B, num_heads, dim, T) -> (B, num_heads, T, T)
        mask = torch.where(self.bias[:,:,:T,:T] == 0, float('-inf'), 0.0) if not self.flash else None
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0, is_causal=True)
        else:
            # manual implementation of attention
            # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # att = F.softmax(att, dim=-1)
            # y = att @ v # (B, num_heads, T, T) x (B, num_heads, T, dim) -> (B, num_heads, T, dim)
            y = custom_scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, FEAT) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
    

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # test the custom attention implementation
    B, num_heads, T, dim = 16, 8, 128, 64
    q, k, v = (torch.randn(B, num_heads, T, dim, device="cuda") for _ in range(3))
    maxlen = T
    _attn_mask = torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen)
    attn_mask = torch.where(_attn_mask[:,:,:T,:T] == 0, float('-inf'), 0.0).cuda()

    with torch.no_grad():
        torch_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False
        )
        custom_out = custom_scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    max_diff = (abs_diff:=(torch_out - custom_out).abs()).max()
    print(f"max diff: {max_diff.item():e} | mean diff: {abs_diff.mean().item():e}")
    
    # Use appropriate tolerances for TF32 vs FP32
    # TF32 has reduced precision (10-bit mantissa vs 23-bit for FP32)
    atol, rtol = 1e-5, 1e-5
    if torch.backends.cuda.matmul.allow_tf32: 
        atol, rtol = 1e-2, 1e-2  # More relaxed for TF32
    
    assert torch.allclose(torch_out, custom_out, atol=atol, rtol=rtol)
    print("Success!")