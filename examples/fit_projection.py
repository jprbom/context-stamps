"""Reproducible fitting example on synthetic vectors; not a semantic benchmark."""

import argparse

import numpy as np

from context_stamps import stamp_vector
from context_stamps.learning import fit_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="family.json")
    args = parser.parse_args()
    vectors = np.random.default_rng(7).normal(size=(256, 32))
    family = fit_family(vectors, encoder="synthetic-normal-v1", bits=16)
    family.save(args.out)
    print(stamp_vector(vectors[0], family))
    print(f"Saved {args.out}; re-use only with embeddings from the same encoder.")


if __name__ == "__main__":
    main()
