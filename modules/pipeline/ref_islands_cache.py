"""Persistent cache for reference-transcript islands.

Reference island scanning is species-INDEPENDENT: a given reference transcript's
exonic sequence + the model + the scan parameters fully determine its islands,
regardless of which query genome is being aligned. So when many species are run
against the same reference, the same reference transcripts are re-embedded and
re-scanned every time --- the dominant repeated GPU cost.

This cache stores each transcript's scan result (islands, or the fact that it has
none) keyed by (transcript_id, params_signature, blocks_signature):
  * params_signature  = model + window/stride/smooth/threshold  -> changing scan
    params or the model invalidates the entry (recompute).
  * blocks_signature   = chrom/strand/exon-blocks  -> changing the reference
    annotation for that transcript invalidates the entry (recompute).

"No islands" is a valid, cached result (islands=[]).

Concurrency: WAL + busy_timeout make concurrent reads safe and serialize writes;
INSERT OR REPLACE is idempotent, so two runs computing the same transcript before
either writes is harmless (identical result). Still, the simplest guidance is
"one running lane -> one cache DB".
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Dict, List, Tuple


def _sig(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def params_signature(model: str, window_size: int, stride: int,
                     smooth_window: int, prob_threshold: float) -> str:
    return _sig("v1", model, window_size, stride, smooth_window, prob_threshold)


def blocks_signature(chrom: str, strand: int, exon_blocks: List[Tuple[int, int]]) -> str:
    return _sig(chrom, strand, tuple((int(a), int(b)) for a, b in exon_blocks))


class RefIslandCache:
    """SQLite-backed store of per-transcript reference-island results."""

    def __init__(self, db_path: str, params_sig: str):
        self.params_sig = params_sig
        self.conn = sqlite3.connect(db_path, timeout=120)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=120000")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS ref_island_cache (
                transcript_id     TEXT NOT NULL,
                params_sig        TEXT NOT NULL,
                blocks_sig        TEXT NOT NULL,
                total_length      INTEGER,
                sum_exons_length  INTEGER,
                islands_json      TEXT,
                PRIMARY KEY (transcript_id, params_sig)
            )"""
        )
        self.conn.commit()

    def lookup(self, blocks_by_tid: Dict[str, str]) -> Dict[str, dict]:
        """Return {tid: payload} for transcripts whose cached row matches BOTH the
        current params_sig and the transcript's blocks_sig. payload =
        {total_length, sum_exons_length, islands}."""
        if not blocks_by_tid:
            return {}
        hits: Dict[str, dict] = {}
        cur = self.conn.execute(
            "SELECT transcript_id, blocks_sig, total_length, sum_exons_length, islands_json "
            "FROM ref_island_cache WHERE params_sig = ?", (self.params_sig,))
        for tid, bsig, tl, sel, isl in cur:
            want = blocks_by_tid.get(tid)
            if want is not None and want == bsig:
                hits[tid] = {"total_length": tl, "sum_exons_length": sel,
                             "islands": json.loads(isl)}
        return hits

    def store(self, entries: Dict[str, dict]) -> int:
        """entries: {tid: {"blocks_sig": str, "payload": {total_length,
        sum_exons_length, islands}}}. Idempotent upsert. Returns count stored."""
        rows = [
            (tid, self.params_sig, e["blocks_sig"],
             e["payload"]["total_length"], e["payload"]["sum_exons_length"],
             json.dumps(e["payload"]["islands"]))
            for tid, e in entries.items()
        ]
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO ref_island_cache "
                "(transcript_id, params_sig, blocks_sig, total_length, sum_exons_length, islands_json) "
                "VALUES (?, ?, ?, ?, ?, ?)", rows)
            self.conn.commit()
        return len(rows)

    def close(self) -> None:
        self.conn.close()


__all__ = ["RefIslandCache", "params_signature", "blocks_signature"]
