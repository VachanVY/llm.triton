from torch import nn, Tensor
import torch.nn.functional as F
import torch

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias, eps:float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input:Tensor, torch_version=False): # (B, T, d_model)
        if torch_version:
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)

        mean = input.mean(dim=-1, keepdim=True) # (B, T, 1)
        var = input.var(dim=-1, keepdim=True, unbiased=False) # (B, T, 1)
        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
    

if __name__ == "__main__":
    B, T, d_model = 2, 3, 4
    x = torch.randn(B, T, d_model)
    layernorm = LayerNorm(d_model, bias=True)
    y1 = layernorm(x, torch_version=False)
    y2 = layernorm(x, torch_version=True)
    assert torch.allclose(y1, y2)