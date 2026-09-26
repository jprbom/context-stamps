"""Complete offline inference example using the prepared, public-data-only cache."""

import argparse
from pathlib import Path

import numpy as np
import torch

from context_stamps.neural_controller import load_ranker

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
data = args.work / "controller-v1-data"
model = load_ranker(args.work / "controller-v1-models/selected-int8.safetensors")
arrays = np.load(data / "scifact-test.npz", allow_pickle=False)
documents = np.load(data / "scifact-D.npy", allow_pickle=False)
scores = model.score_candidates(
    torch.from_numpy(arrays["Q"][:1]),
    torch.from_numpy(documents[arrays["candidates"][:1]]),
    torch.from_numpy(arrays["F"][:1]),
    torch.from_numpy(arrays["mask"][:1]),
)
indexes = scores[0].argsort(descending=True)[:10].tolist()
print("Experimental ranked corpus rows:", arrays["candidates"][0, indexes].tolist())
print("Calibration did not qualify this learned model; the default retriever is unchanged.")
