"""Context Stamps, by Prashant Jagtap."""

from stamps import Family, HashingEncoder, Stamp, content_digest, hamming, similarity, stamp_vector

from .activation import StampSchema, activate
from .facet_model import FacetModel
from .facets import FacetQuery
from .guarded_activation import activate_constrained
from .memory import ContextMemory, Packet
from .packed_index import PackedStampIndex
from .pretrained import load_experimental_model
from .requirements import Claim, Requirement, rank_candidates_safe, select_structured
from .residual_index import ResidualIndex
from .routing import ProgressiveRouter, RoutingPolicy, fit_routing_policy
from .selection import LinearSelector, rank_candidates, select_evidence
from .session import ContextSession
from .sources import explain_versions, observe_files
from .spherical import SphericalStamp
from .stamp256 import Stamp256Codec
from .workflow import ContextGraph, ContextNode

__version__ = "0.3.3"
__all__ = [
    "FacetQuery",
    "ContextSession",
    "ProgressiveRouter",
    "RoutingPolicy",
    "fit_routing_policy",
    "ResidualIndex",
    "Stamp256Codec",
    "SphericalStamp",
    "PackedStampIndex",
    "load_experimental_model",
    "StampSchema",
    "FacetModel",
    "activate",
    "activate_constrained",
    "ContextGraph",
    "ContextNode",
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
