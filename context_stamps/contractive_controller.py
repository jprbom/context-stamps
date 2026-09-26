"""Optional contractive evidence diffusion and metric-aware ranking loss.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Convergence is a numerical property, not a correctness or authorization guarantee.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ContractiveRanker(nn.Module):
    def __init__(self, embedding_dim=384, width=64, recurrent=True, steps=4):
        super().__init__()
        if (type(embedding_dim) is not int or not 8 <= embedding_dim <= 4096
                or type(width) is not int or not 8 <= width <= 128
                or type(recurrent) is not bool or steps not in (2, 4, 8) or type(steps) is not int):
            raise ValueError("invalid bounded architecture")
        self.config = dict(embedding_dim=embedding_dim, width=width, recurrent=recurrent, steps=steps)
        self.input = nn.Linear(embedding_dim * 4 + 6, width)
        self.query = nn.Linear(width, 16, bias=False) if recurrent else None
        self.key = nn.Linear(width, 16, bias=False) if recurrent else None
        self.diagonal = nn.Parameter(torch.zeros(width)) if recurrent else None
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initial_state(self, queries, documents, features, mask):
        if documents.ndim != 3:
            raise ValueError("candidate tensor must have three dimensions")
        batch, count, dim = documents.shape
        if (not 1 <= batch <= 256 or not 1 <= count <= 256 or dim != self.config["embedding_dim"]
                or queries.shape != (batch, dim) or features.shape != (batch, count, 6)
                or mask.shape != (batch, count) or mask.dtype != torch.bool):
            raise ValueError("unaligned bounded tensors")
        q = F.normalize(queries.float(), dim=-1).unsqueeze(1).expand(-1, count, -1)
        d = F.normalize(documents.float(), dim=-1)
        combined = torch.cat((q, d, q * d, (q - d).abs(), features), dim=-1)
        # Also isolate masked outliers from dynamic-int8 activation calibration.
        combined = combined.masked_fill(~mask[..., None], 0)
        u = torch.tanh(self.input(combined))
        if not self.config["recurrent"]:
            return u, None
        logits = (self.query(u) @ self.key(u).transpose(-1, -2)).float() / 4
        logits = logits.masked_fill(~mask[:, None, :], -1e4)
        # P is fixed throughout a recurrence; normalization excludes padded keys.
        p = torch.softmax(logits, dim=-1) * mask[:, None, :]
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        return u, p

    def advance(self, u, p, state):
        """Contraction <=0.5 in max-row Euclidean norm for row-stochastic P.

        Convex row averaging is nonexpansive, |tanh(diagonal)| <= 1, and tanh
        is 1-Lipschitz. No state-dependent attention or unbounded residual sum.
        """
        return .5 * u + .5 * torch.tanh(p @ (state * torch.tanh(self.diagonal)))

    def forward(self, queries, documents, features, mask, *, steps=None):
        steps = self.config["steps"] if steps is None else steps
        if type(steps) is not int or not 1 <= steps <= 32:
            raise ValueError("invalid recurrence count")
        u, p = self.initial_state(queries, documents, features, mask)
        state = u
        if p is not None:
            for _ in range(steps):
                state = self.advance(u, p, state)
        correction = torch.tanh(self.output(state).squeeze(-1))
        return (features[..., 2] + correction).masked_fill(~mask, -1e4)

    @torch.inference_mode()
    def score_candidates(self, queries, documents, features, mask, *, steps=None):
        if steps is not None and (type(steps) is not int or steps not in (2, 4, 8)):
            raise ValueError("only trained depths 2/4/8 are supported for serving")
        if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(1).all().item():
            raise ValueError("each query requires eligible candidates")
        if any(not torch.isfinite(x).all().item() for x in (queries, documents, features)):
            raise ValueError("non-finite candidate input")
        scores = self(queries, documents, features, mask, steps=steps)
        if not torch.isfinite(scores).all().item():
            raise ValueError("numeric overflow")
        return scores.masked_fill(~mask, -torch.inf)


def ndcg_pair_loss(scores, relevance, mask):
    """LambdaRank-inspired linear-gain delta-nDCG@10 weighted logistic loss.

    Not a claim to reproduce every LambdaLoss variant. Unjudged/zero-relevance
    candidates remain uncertain negatives; this loss does not create judgments.
    """
    if scores.shape != relevance.shape or scores.shape != mask.shape:
        raise ValueError("aligned scores, relevance and mask required")
    with torch.no_grad():
        order = scores.argsort(dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order).scatter(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
        discount = 1 / torch.log2(ranks.float() + 2)
        discount = discount * (ranks < 10)
        ideal = relevance.masked_fill(~mask, 0).sort(dim=1, descending=True).values[:, :10]
        denom = (ideal / torch.log2(torch.arange(ideal.shape[1], device=scores.device) + 2)).sum(1).clamp_min(1e-8)
        gain = relevance[:, :, None] - relevance[:, None, :]
        valid = (gain > 0) & mask[:, :, None] & mask[:, None, :]
        weight = gain.abs() * (discount[:, :, None] - discount[:, None, :]).abs() / denom[:, None, None]
        weight = weight * valid
    penalty = F.softplus(-(scores[:, :, None] - scores[:, None, :]))
    return ((weight * penalty).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1e-8)).mean()


def centered_distillation(scores, teacher, mask):
    """Centered score MSE, equivalent to a scaled all-pair margin objective.

    Teacher scores are normalized per list to avoid learning arbitrary offsets.
    This is distillation of predictions, not additional human relevance labels.
    """
    count = mask.sum(1, keepdim=True).clamp_min(1)
    centered_s = scores - (scores * mask).sum(1, keepdim=True) / count
    centered_t = teacher - (teacher * mask).sum(1, keepdim=True) / count
    variance = (centered_t.square() * mask).sum(1, keepdim=True) / count
    centered_t = centered_t / variance.sqrt().clamp_min(1)
    return (((centered_s - centered_t).square() * mask).sum(1) / count[:, 0]).mean()


def save_contractive(model, path):
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}, str(path),
              metadata={"format": "context-contractive-v2", "config": json.dumps(model.config, sort_keys=True)})


def load_contractive(path):
    from safetensors import safe_open
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("bounded regular checkpoint required")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "context-contractive-v2":
            raise ValueError("unsupported checkpoint")
        model = ContractiveRanker(**json.loads(metadata["config"]))
        templates = model.state_dict()
        if set(handle.keys()) != set(templates):
            raise ValueError("checkpoint parameter mismatch")
        state = {}
        for key, expected in templates.items():
            value = handle.get_tensor(key)
            if value.shape != expected.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("invalid parameter")
            state[key] = value
    model.load_state_dict(state)
    return model.eval()
