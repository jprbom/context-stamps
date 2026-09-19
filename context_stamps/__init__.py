"""Context Stamps, by Prashant Jagtap."""

from stamps import Family, HashingEncoder, Stamp, content_digest, hamming, similarity, stamp_vector

from .memory import ContextMemory, Packet
from .selection import LinearSelector, rank_candidates, select_evidence
from .sources import explain_versions, observe_files

__version__ = "0.2.0"
__all__ = [
    "LinearSelector",
    "rank_candidates",
    "select_evidence",
    "observe_files",
    "explain_versions",
    "Family",
    "HashingEncoder",
    "Stamp",
    "content_digest",
    "hamming",
    "similarity",
    "stamp_vector",
    "ContextMemory",
    "Packet",
]
