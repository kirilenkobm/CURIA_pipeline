"""Island matchers: per-model scoring for the island-alignment step.

The RNA-FM window-MMD scorer (``rnafm``) and the RiNALMo embed-once
cosine-dotplot + nucleotide Smith-Waterman scorer (``rinalmo``) live in
separate modules. The ``island_alignment`` orchestrator picks one via
``get_matcher`` and both emit the same :class:`MatchResult`, so the shared
assignment / collinearity / output layer stays model-agnostic.
"""

from modules.pipeline.matchers.base import Chain, MatchResult, get_matcher

__all__ = ["Chain", "MatchResult", "get_matcher"]
