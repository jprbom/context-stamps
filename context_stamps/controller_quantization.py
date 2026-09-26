"""Optional split-precision conversion for the controller's mixed-scale input.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Embedding projection is quantized; six retrieval features stay in FP32.
No new weights are fitted and no executable checkpoint is serialized.
"""

import copy
import warnings

import torch
from torch import nn

from .contractive_controller import ContractiveRanker


class SplitInputProjection(nn.Module):
    """Algebraically W[x; f] + b = W_x x + W_f f + b before quantization."""

    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.in_features <= 6:
            raise ValueError("expected the original input projection")
        if linear.weight.device.type != "cpu" or linear.weight.dtype != torch.float32:
            raise ValueError("convert a CPU FP32 model")
        self.embedding_dim = linear.in_features - 6
        self.embedding = nn.Linear(self.embedding_dim, linear.out_features, bias=False)
        self.retrieval = nn.Linear(6, linear.out_features, bias=True)
        with torch.no_grad():
            self.embedding.weight.copy_(linear.weight[:, :-6])
            self.retrieval.weight.copy_(linear.weight[:, -6:])
            self.retrieval.bias.copy_(linear.bias)

    def forward(self, inputs):
        return self.embedding(inputs[..., :-6]) + self.retrieval(inputs[..., -6:])


def split_precision_controller(model, *, quantize=True):
    """Copy a trained CPU model; quantize only the embedding input matrix.

    Set the host's supported torch quantized engine before calling. This helper
    does not change global thread/backend settings. FP32 feature and output
    paths reduce mixed-range distortion; accuracy and speed still need testing.
    """
    if not isinstance(model, ContractiveRanker) or type(quantize) is not bool:
        raise ValueError("a ContractiveRanker and boolean quantize flag are required")
    converted = copy.deepcopy(model).eval()
    converted.input = SplitInputProjection(converted.input)
    if quantize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            converted.input = torch.ao.quantization.quantize_dynamic(
                converted.input, {"embedding": torch.ao.quantization.default_dynamic_qconfig}, dtype=torch.qint8,
            )
    return converted
