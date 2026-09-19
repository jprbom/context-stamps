"""Optional NumPy fitting of centered and ITQ projections on caller-owned embeddings."""

from stamps import Family


def fit_family(
    vectors, *, encoder: str, bits: int = 128, method: str = "itq", seed: int = 20260919, iterations: int = 50
) -> Family:
    import numpy as np

    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("provide a finite matrix with at least two training rows")
    if method not in {"centered", "itq"} or iterations < 1:
        raise ValueError("method must be centered/itq and iterations positive")
    dim = x.shape[1]
    mean = x.mean(axis=0)
    # Validate before expensive decomposition.
    Family(encoder, dim, bits, seed)
    if method == "centered":
        return Family(encoder, dim, bits, seed, "centered-v1", tuple(mean))
    if bits > min(dim, len(x) - 1):
        raise ValueError("ITQ bits cannot exceed dimension or training row count minus one")
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:bits].T
    projected = centered @ basis
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((bits, bits)))
    for _ in range(iterations):
        binary = np.where(projected @ rotation >= 0, 1.0, -1.0)
        u, _, vt_r = np.linalg.svd(projected.T @ binary)
        rotation = u @ vt_r
    planes = (basis @ rotation).T
    return Family(encoder, dim, bits, seed, "itq-v1", tuple(mean), tuple(tuple(row) for row in planes))
