"""Optional semantic + lexical views; downloads a pinned public encoder if absent.

Install with: python -m pip install -e '.[semantic]'
This is an integration example, not a retrieval benchmark.
"""

from context_stamps import Family, HashingEncoder, SphericalStamp
from context_stamps.encoders import SentenceTransformerEncoder

semantic = SentenceTransformerEncoder(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    device="cpu",
)
lexical = HashingEncoder(64)
families = {
    "meaning": Family(semantic.identity, semantic.dim, 128, 17),
    "wording": Family(lexical.identity, lexical.dim, 128, 41),
}


def stamp(text):
    return SphericalStamp.encode({"meaning": semantic.encode(text), "wording": lexical.encode(text)}, families)


query = stamp("The request failed because the credentials were invalid.")
candidate = stamp("The API rejected authentication for the request.")
print({"total_bits": query.bits, "per_view_agreement": query.compare(candidate)})
print("Different views use different encoders; neither score proves exact identity or correctness.")
