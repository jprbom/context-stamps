"""Optional, bounded residual evidence ranker. It does not grant access to evidence.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Import this module explicitly after installing the controller extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentEvidenceRanker(nn.Module):
    """Permutation-equivariant query/candidate attention with shared recurrent weights.

    Frozen embeddings remain external to the 256-bit routing stamp. The residual
    starts at zero, preserving the supplied hybrid score before training. Recurrence
    is bounded inference, not self-modification. Inputs must already be authorized.
    """

    def __init__(self, embedding_dim=384, width=64, heads=4, steps=2, attention=True):
        super().__init__()
        if (type(embedding_dim) is not int or not 8 <= embedding_dim <= 4096
                or type(width) is not int or not 16 <= width <= 256
                or type(heads) is not int or heads < 1 or width % heads
                or type(steps) is not int or not 1 <= steps <= 4
                or type(attention) is not bool):
            raise ValueError("invalid bounded architecture")
        self.config = dict(embedding_dim=embedding_dim, width=width, heads=heads,
                           steps=steps, attention=attention)
        self.features = nn.Linear(embedding_dim * 2 + 6, width)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3) if attention else None
        self.mix = nn.Linear(width, width) if attention else None
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, queries, documents, features, mask, *, steps=None):
        # Shape/bounds checks here; score_candidates also checks untrusted numerics.
        batch, count, dim = documents.shape
        if (not 1 <= batch <= 256 or not 1 <= count <= 256
                or dim != self.config["embedding_dim"] or queries.shape != (batch, dim)
                or features.shape != (batch, count, 6) or mask.shape != (batch, count)
                or mask.dtype != torch.bool):
            raise ValueError("unaligned or oversized candidate tensors")
        steps = self.config["steps"] if steps is None else steps
        if type(steps) is not int or not 1 <= steps <= 4:
            raise ValueError("one to four recurrent steps required")
        q = F.normalize(queries.float(), dim=-1).unsqueeze(1)
        d = F.normalize(documents.float(), dim=-1)
        state = F.gelu(self.features(torch.cat((q * d, (q - d).abs(), features), dim=-1)))
        heads = self.config["heads"]
        for _ in range(steps):
            if self.qkv is not None:
                qkv = self.qkv(self.norm1(state)).reshape(batch, count, 3, heads, -1)
                aq, ak, av = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                update = F.scaled_dot_product_attention(aq, ak, av, attn_mask=mask[:, None, None, :])
                state = state + self.mix(update.transpose(1, 2).reshape(batch, count, -1))
            state = state + self.ff(self.norm2(state))
        # Bounded correction limits extrapolation; it is not a quality guarantee.
        residual = 2 * torch.tanh(self.output(state).squeeze(-1))
        return (features[..., 2] + residual).masked_fill(~mask, -1e4)

    @torch.inference_mode()
    def score_candidates(self, queries, documents, features, mask, *, steps=None):
        if any(not torch.isfinite(value).all().item() for value in (queries, documents, features)):
            raise ValueError("candidate values must be finite")
        if mask.ndim != 2 or not mask.any(dim=1).all().item():
            raise ValueError("each query needs an eligible candidate")
        return self(queries, documents, features, mask, steps=steps)


def export_ranker(model, path, *, int8=False):
    """Safe tensor storage; per-output-channel symmetric int8 for matrix weights.

    Loading reconstructs FP32 weights. This reduces checkpoint/delivery size, not
    runtime memory or compute. This explicit distinction prevents an int8 speed claim.
    """
    from safetensors.torch import save_file

    tensors = {}
    for key, value in model.state_dict().items():
        value = value.detach().cpu().float().contiguous()
        if int8 and value.ndim == 2:
            scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
            tensors[key + ".int8"] = (value / scale).round().clamp(-127, 127).to(torch.int8)
            tensors[key + ".scale"] = scale
        else:
            tensors[key] = value
    save_file(tensors, str(path), metadata={"format": "context-ranker-v1",
              "config": json.dumps(model.config, sort_keys=True),
              "storage": "int8-per-row" if int8 else "float32"})


def load_ranker(path):
    """Load bounded safetensors, without pickle or remote code execution."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("bounded regular checkpoint required")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "context-ranker-v1":
            raise ValueError("unsupported checkpoint")
        model = RecurrentEvidenceRanker(**json.loads(metadata["config"]))
        expected = model.state_dict()
        state, consumed = {}, set()
        keys = set(handle.keys())
        for key, template in expected.items():
            if key in keys:
                value = handle.get_tensor(key)
                consumed.add(key)
                if value.dtype != torch.float32:
                    raise ValueError("expected float32 parameter")
            else:
                quant = handle.get_tensor(key + ".int8")
                scale = handle.get_tensor(key + ".scale")
                if (quant.dtype != torch.int8 or quant.shape != template.shape
                        or quant.ndim != 2 or scale.shape != (quant.shape[0], 1)
                        or scale.dtype != torch.float32 or not (scale > 0).all()):
                    raise ValueError("invalid quantized parameter")
                consumed.update((key + ".int8", key + ".scale"))
                value = quant.float() * scale
            if value.shape != template.shape or not torch.isfinite(value).all():
                raise ValueError("invalid checkpoint parameter")
            state[key] = value
        if consumed != keys:
            raise ValueError("unexpected checkpoint parameters")
    model.load_state_dict(state, strict=True)
    return model.eval()
