"""Explicit opt-in loading for the small procedural research scorers."""

from importlib.resources import files

from .facet_model import FacetModel


def load_experimental_model(*, seed=17):
    """Load bundled monotone pairwise weights, not a language or multimodal model.

    Requires the exact four-view lexical family schema used in pairwise-v1.
    Scores are ranking utilities, not calibrated activation probabilities.
    Do not use them as an authorization, identity or safety classifier.
    """
    if type(seed) is not int or seed not in (17, 41, 83):
        raise ValueError("available projection seeds are 17, 41 and 83")
    text = files("context_stamps").joinpath("models", f"pairwise-{seed}.json").read_text(encoding="utf-8")
    return FacetModel.from_json(text)
