"""Shared contract for island matchers.

A matcher turns a per-model island *representation* into a per-pair
:class:`MatchResult`. The orchestrator (``island_alignment.py``) consumes only
this result, so the assignment / collinearity / output layer never sees
model-specific details.

Matcher interface (duck-typed):
    representation: str
        What ``prepare_island`` returns; documentation only.
    async prepare_island(seq, gpu, job_id, island_id, config) -> repr
        Fetch the island's embedding representation via the GPU client.
    gene_precompute(ref_reprs, q_reprs, valid_pairs, config) -> ctx
        Optional per-gene shared state (e.g. MMD gamma + per-island self-kernels);
        may return None.
    score_pair(ri, qi, ref_reprs, q_reprs, ctx, config) -> MatchResult
        Score reference island ri vs query island qi (runs in a thread pool).
        Takes indices + the repr lists so a matcher can reuse per-island ctx.

Direction convention (critical): ``score`` is higher-is-better and is used
ONLY for the ``> 0`` gate; ``dist`` is lower-is-better and drives the ascending
sort, the ``max_match_dist`` ceiling, and the reported ``diag_mmd``. Both
matchers must respect this so the shared layer needs no per-model branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chain:
    """One aligned band, in island-relative NUCLEOTIDE coordinates."""
    ref_from: int
    ref_to: int
    q_from: int
    q_to: int
    dist: float          # per-chain quality, LOWER = better (-> chains_json "mmd")


@dataclass
class MatchResult:
    score: float                       # higher = better; only the > 0 gate uses it
    dist: float                        # lower = better; sort + max_match_dist + diag_mmd
    eff_nt: int                        # effective aligned nucleotides
    chains: List[Chain] = field(default_factory=list)


# A pair that should be discarded by the shared filter.
EMPTY_MATCH = MatchResult(0.0, float("inf"), 0, [])


def get_matcher(model_name):
    """Return the island matcher for ``model_name``, keyed on its embed strategy.

    ``embed_once`` -> RiNALMo (dotplot + nt-SW); ``windowed`` -> RNA-FM
    (window-MMD + SW). No dedicated registry key is needed.
    """
    from modules.model_registry import get_embed_strategy
    strategy = get_embed_strategy(model_name)
    if strategy == "embed_once":
        from modules.pipeline.matchers.rinalmo import RinalmoMatcher
        return RinalmoMatcher()
    from modules.pipeline.matchers.rnafm import RnafmMatcher
    return RnafmMatcher()
