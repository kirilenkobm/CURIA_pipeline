#!/usr/bin/env python3
"""
On-the-fly chromosome-name aliasing for query 2bit access.

Chain files and the 2bit assembly they are read against sometimes disagree only
cosmetically: an accession version suffix (chain ``VIYN01000001`` vs 2bit
``VIYN01000001.1``) or a ``chr`` prefix. Names that originate from the chain
flow unchanged through TOGA, liftover, and island coordinates; the mismatch only
bites at the single point where a chain-derived name is used to fetch sequence
from the query 2bit.

``AliasedTwoBitAccessor`` wraps ``pyrion.TwoBitAccessor`` and resolves such names
at ``fetch`` time — no chain rewrite, no extra files. It is a strict no-op when
the requested name already exists in the 2bit, and falls through to the real
accessor (which raises as before) for names that are genuinely absent.
"""

from typing import Dict, List

from pyrion import TwoBitAccessor


class AliasedTwoBitAccessor:
    """TwoBitAccessor that tolerates version-suffix / ``chr``-prefix name
    differences between a chain and its query 2bit."""

    def __init__(self, path: str):
        self._acc = TwoBitAccessor(path)
        self._real = set(self._acc.chrom_names())
        # version-stripped base -> real name(s), for unambiguous "add version"
        self._by_base: Dict[str, List[str]] = {}
        for n in self._real:
            base = n.rsplit(".", 1)[0]
            if base != n:
                self._by_base.setdefault(base, []).append(n)
        self._cache: Dict[str, str] = {}

    def _resolve(self, chrom: str) -> str:
        if chrom in self._real:
            return chrom
        hit = self._cache.get(chrom)
        if hit is not None:
            return hit
        r = chrom  # fall through -> real accessor raises as before
        if chrom in self._by_base and len(self._by_base[chrom]) == 1:
            r = self._by_base[chrom][0]                     # 'X'    -> 'X.1'
        else:
            base = chrom.rsplit(".", 1)[0]
            if base != chrom and base in self._real:
                r = base                                    # 'X.1'  -> 'X'
            elif f"chr{chrom}" in self._real:
                r = f"chr{chrom}"                           # '1'    -> 'chr1'
            elif chrom.startswith("chr") and chrom[3:] in self._real:
                r = chrom[3:]                               # 'chr1' -> '1'
        self._cache[chrom] = r
        return r

    def fetch(self, chrom, start=None, end=None, *args, **kwargs):
        return self._acc.fetch(self._resolve(chrom), start, end, *args, **kwargs)

    def chrom_sizes(self):
        """Sizes keyed by the real 2bit names AND their version/chr-normalized
        aliases, so callers that look up sizes by chain-style names (e.g. the
        coordinate-clamp in query_islands_scanner) resolve correctly. Real keys
        are never overwritten; ambiguous version bases are skipped."""
        sizes = dict(self._acc.chrom_sizes())
        extra: Dict[str, int] = {}
        for n, s in sizes.items():
            base = n.rsplit(".", 1)[0]
            if base != n and base not in sizes and len(self._by_base.get(base, [])) == 1:
                extra.setdefault(base, s)
            if n.startswith("chr"):
                if n[3:] not in sizes:
                    extra.setdefault(n[3:], s)
            elif f"chr{n}" not in sizes:
                extra.setdefault(f"chr{n}", s)
        sizes.update(extra)
        return sizes

    def close(self):
        self._acc.close()

    def __getattr__(self, name):
        # Delegate everything else (chrom_names, chrom_sizes, list_chromosomes,
        # fetch_interval, validate_interval, ...) to the wrapped accessor.
        # Guard internal names to avoid recursion before _acc is assigned.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._acc, name)
