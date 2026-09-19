"""Context Stamps, by Prashant Jagtap."""

from stamps import Family, HashingEncoder, Stamp, content_digest, hamming, similarity, stamp_vector

from .memory import ContextMemory, Packet
from .requirements import Claim, Requirement, rank_candidates_safe, select_structured
from .selection import LinearSelector, rank_candidates, select_evidence
from .sources import explain_versions, observe_files

__version__ = "0.3.0"
__all__ = [
    "Claim",
    "Requirement",
    "rank_candidates_safe",
    "select_structured",
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
