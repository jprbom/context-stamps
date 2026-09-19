"""Optional neural encoder. Model loading is explicit and never happens in core imports."""

import re


class SentenceTransformerEncoder:
    def __init__(self, model: str, *, revision: str, device: str = "cpu") -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character model commit SHA")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.identity = (
            f"sentence-transformers:{model}@{revision}:normalize=true:"
            f"max_seq={self.model.max_seq_length}:float32-v1"
        )

    def encode(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True).tolist()
