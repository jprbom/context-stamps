"""Quality-preserving vector refinement; install the optional learn extra."""

import numpy as np

from context_stamps import ResidualIndex

rng = np.random.default_rng(42)
vectors = rng.normal(size=(1000, 64)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
index = ResidualIndex(vectors, encoder_id="example-encoder-v1")
# Replace this callback with an authenticated, immutable vector store in an app.
result = index.search(vectors[17], encoder_id="example-encoder-v1",
                      eligible=np.arange(len(vectors)), fetch=lambda rows: vectors[rows], limit=5)
print(result)
assert result["rows"][0] == 17
print("Index arrays:", index.index_bytes, "bytes; full backing vectors:", vectors.nbytes, "bytes")
