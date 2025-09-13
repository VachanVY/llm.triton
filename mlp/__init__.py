# torch imports
from .torch_mlpblock import MLP as TorchMLPBlock, TorchLinear

# custom triton imports
from .triton_mlpblock import linear_forward