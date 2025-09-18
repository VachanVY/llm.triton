from torch import Tensor
import torch # For basic tensor ops ONLY...

from layernorm import layernorm_forward
from flash_attn import flash_attn_forward
from mlp import linear_forward


class Embedding:
    def __init__(self, embeddings:Tensor): # (vocab_size, d_model)
        self.embeddings = embeddings
    def __call__(self, x:Tensor): # (B, T)
        return self.embeddings[x] # (B, T, d_model)

class MLP:
    def __init__(self, weights1:tuple[Tensor, Tensor], weights2:tuple[Tensor, Tensor]):
        self.weights1 = weights1
        self.weights2 = weights2

    def __call__(self, x:Tensor):
        # TODO(VachanVY): Can we fuse both these into one kernel? do...
        x = linear_forward(
            x, self.weights1[0], self.weights1[1], activation_type="gelu"
        )
        x = linear_forward(
            x, self.weights2[0], self.weights2[1], activation_type="none"
        )
        return x
    

class Block:
    def __init__(
        self, 
        ln1_weights:tuple[Tensor, Tensor],
        ln2_weights:tuple[Tensor, Tensor],
        mlp_weights1:tuple[Tensor, Tensor], 
        mlp_weights2:tuple[Tensor, Tensor],
    ):
        self.ln1_weights = ln1_weights
        self.ln2_weights = ln2_weights
        self.mlp = MLP(weights1=mlp_weights1, weights2=mlp_weights2)

    def __call__(self, x:Tensor):
        y1 = layernorm_forward(x, weight=self.ln1_weights[0], bias=self.ln1_weights[1])
        x = x + flash_attn_forward(y1, ...)

        y2 = layernorm_forward(x, weight=self.ln2_weights[0], bias=self.ln2_weights[1])
        x = x + self.mlp(y2)
        return x
    

class GPT:
    """
    ```python
    gpt_weights = {
        "embedding_weights": embeddings,
        "positional_embeddings": pos_embeddings,
        "block_1": {
                "layernorm_1": (ln_weight, ln_bias),
                "attn": {
                    ...
                    }
                "layernorm_2": (ln_weight, ln_bias),
                "mlp_block": {
                    "weights1": (mlp_fc_weight, mlp_fc_bias),
                    "weights2": (mlp_proj_weight, mlp_proj_bias)
                }

            }
        ...
        "block_n": { ... }
        "final_layernorm": (ln_weight, ln_bias)
        "linear_head": (lm_head_weight, lm_head_bias=None)
    }
    ```
    """
    def __init__(self, weights_path:str, device:str="cuda"):
        # get weights
        gpt_weights = torch.load(weights_path, map_location=device)
        self.embedding_weights = gpt_weights["embedding_weights"] # (vocab_size, d_model)
        self.positional_embeddings = gpt_weights["positional_embeddings"] # (block_size, d_model)
        self.blocks = []
        for block_key in sorted([k for k in gpt_weights.keys() if k.startswith("block_")]):
            block_weights = gpt_weights[block_key]
            block = Block(
                ln1_weights=block_weights["layernorm_1"],
                ln2_weights=block_weights["layernorm_2"],
                mlp_weights1=(
                    block_weights["mlp_block"]["weights1"][0],
                    block_weights["mlp_block"]["weights1"][1]
                ),
                mlp_weights2=(
                    block_weights["mlp_block"]["weights2"][0],
                    block_weights["mlp_block"]["weights2"][1]
                )
            )
            self.blocks.append(block)
        self.final_layernorm_weights = gpt_weights["final_layernorm"]
        self.lm_head_weight, self.lm_head_bias = gpt_weights["linear_head"] # (weight, None) # (d_model, vocab_size)
        assert self.lm_head_bias is None
        assert self.embedding_weights.shape == self.lm_head_weight.shape, \
            f"{self.embedding_weights.shape=} {self.lm_head_weight.shape=}"
        torch.testing.assert_close(self.embedding_weights, self.lm_head_weight)

        # define layers
        self.embedding_layer = Embedding(self.embedding_weights)
        self.positional_embedding_layer = Embedding(self.positional_embeddings)

    def __call__(self, x:Tensor): # (B, T)
        B, T = x.shape
        x = self.embedding_layer(x)
        x = x + self.positional_embedding_layer(torch.arange(0, T, device=x.device)).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = layernorm_forward(x, weight=self.final_layernorm_weights[0], bias=self.final_layernorm_weights[1])
        x = linear_forward(x[:, [-1], :], self.lm_head_weight, self.lm_head_bias, activation_type="none")
        return x


if __name__ == "__main__":
    model = GPT("gpt2_weights.pth")
