#!/usr/bin/env python3
"""Conventional nucleotide sequence-similarity metrics (no GPU, no external
alignment libs -- parasail/biopython/skbio are not installed in this env).

Every metric here is a *sequence-only* comparison of two nucleotide strings and is
used as a baseline against the RiNALMo embedding score. Definitions are frozen and
documented in analysis/embedding_vs_sequence/README.md.

Encoding: A=0 C=1 G=2 U/T=3 N=4 (N never matches, is excluded from k-mer/dinuc/GC).
"""
from __future__ import annotations

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
_COMP = str.maketrans("ACGTUN", "TGCAAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def enc(s: str) -> np.ndarray:
    return np.array([_MAP.get(c, 4) for c in s.upper()], dtype=np.int64)


# ---------------------------------------------------------------------------
# Levenshtein identity  (1 - edit/max(len));  same as compute_fig4_data._edit_ident
# ---------------------------------------------------------------------------
@njit(cache=True)
def _edit_ident(a, b):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = np.arange(lb + 1)
    cur = np.zeros(lb + 1, dtype=np.int64)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d = prev[j] + 1
            l = cur[j - 1] + 1
            diag = prev[j - 1] + cost
            m = d if d < l else l
            m = m if m < diag else diag
            cur[j] = m
        prev, cur = cur, prev
    return 1.0 - prev[lb] / max(la, lb)


# ---------------------------------------------------------------------------
# affine-gap local Smith-Waterman (Gotoh) with traceback -> (score, aln_len, n_ident)
#   match=+2, mismatch=-1, gap_open=5, gap_extend=1  (open cost includes 1st step)
#   N (code 4) never scores a match.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _sw_affine(a, b, match, mismatch, gap_open, gap_extend):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0, 0, 0
    NEG = -1.0e18
    H = np.zeros((la + 1, lb + 1))
    E = np.full((la + 1, lb + 1), NEG)   # gap in a (consume b / horizontal)
    F = np.full((la + 1, lb + 1), NEG)   # gap in b (consume a / vertical)
    hp = np.zeros((la + 1, lb + 1), dtype=np.int8)  # 0 stop,1 diag,2 fromE,3 fromF
    ep = np.zeros((la + 1, lb + 1), dtype=np.int8)  # 0 open,1 extend
    fp = np.zeros((la + 1, lb + 1), dtype=np.int8)
    best = 0.0
    bi = 0
    bj = 0
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            # E: gap in a
            e_open = H[i, j - 1] - gap_open
            e_ext = E[i, j - 1] - gap_extend
            if e_ext > e_open:
                E[i, j] = e_ext
                ep[i, j] = 1
            else:
                E[i, j] = e_open
                ep[i, j] = 0
            # F: gap in b
            f_open = H[i - 1, j] - gap_open
            f_ext = F[i - 1, j] - gap_extend
            if f_ext > f_open:
                F[i, j] = f_ext
                fp[i, j] = 1
            else:
                F[i, j] = f_open
                fp[i, j] = 0
            # diagonal
            s = match if (ai == b[j - 1] and ai != 4) else mismatch
            diag = H[i - 1, j - 1] + s
            # H = max(0, diag, E, F)
            m = 0.0
            ptr = 0
            if diag > m:
                m = diag
                ptr = 1
            if E[i, j] > m:
                m = E[i, j]
                ptr = 2
            if F[i, j] > m:
                m = F[i, j]
                ptr = 3
            H[i, j] = m
            hp[i, j] = ptr
            if m > best:
                best = m
                bi = i
                bj = j
    # traceback
    aln_len = 0
    n_ident = 0
    i, j = bi, bj
    state = 0  # 0=H,1=E,2=F
    while True:
        if state == 0:
            p = hp[i, j]
            if p == 0:
                break
            if p == 1:
                aln_len += 1
                if a[i - 1] == b[j - 1] and a[i - 1] != 4:
                    n_ident += 1
                i -= 1
                j -= 1
            elif p == 2:
                state = 1
            else:
                state = 2
        elif state == 1:  # E, consume b
            aln_len += 1
            if ep[i, j] == 0:
                j -= 1
                state = 0
            else:
                j -= 1
                state = 1
        else:  # F, consume a
            aln_len += 1
            if fp[i, j] == 0:
                i -= 1
                state = 0
            else:
                i -= 1
                state = 2
    return best, aln_len, n_ident


def sw_metrics(a_enc, b_enc, match=2.0, mismatch=1.0, gap_open=5.0, gap_extend=1.0):
    """Return (sw_raw, sw_norm, sw_aligned_ident).

    sw_norm    = sw_raw / (match * min(len_a, len_b))  in ~[0,1]
    sw_aligned_ident = n_ident / aln_len (identity over the SW-aligned region)
    """
    raw, aln_len, n_ident = _sw_affine(a_enc, b_enc, match, -abs(mismatch),
                                       abs(gap_open), abs(gap_extend))
    denom = match * min(len(a_enc), len(b_enc))
    sw_norm = raw / denom if denom > 0 else 0.0
    ident = n_ident / aln_len if aln_len > 0 else 0.0
    return float(raw), float(sw_norm), float(ident)


# ---------------------------------------------------------------------------
# k-mer cosine similarity (over 4^k count vectors; k-mers containing N are dropped)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _kmer_counts(a, k, size):
    v = np.zeros(size, dtype=np.float64)
    n = len(a)
    if n < k:
        return v
    for i in range(n - k + 1):
        idx = 0
        ok = True
        for t in range(k):
            c = a[i + t]
            if c == 4:
                ok = False
                break
            idx = idx * 4 + c
        if ok:
            v[idx] += 1.0
    return v


def kmer_cosine(a_enc, b_enc, k):
    size = 4 ** k
    va = _kmer_counts(a_enc, k, size)
    vb = _kmer_counts(b_enc, k, size)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ---------------------------------------------------------------------------
# dinucleotide-composition distance (euclidean between 16-dim freq vectors)
# ---------------------------------------------------------------------------
def dinuc_freq(a_enc):
    v = _kmer_counts(a_enc, 2, 16)
    s = v.sum()
    return v / s if s > 0 else v


def dinuc_dist(a_enc, b_enc):
    return float(np.linalg.norm(dinuc_freq(a_enc) - dinuc_freq(b_enc)))


# ---------------------------------------------------------------------------
# GC content (ignoring N)
# ---------------------------------------------------------------------------
def gc_content(a_enc):
    valid = a_enc[a_enc != 4]
    if len(valid) == 0:
        return 0.0
    return float(np.isin(valid, (1, 2)).sum() / len(valid))


# ---------------------------------------------------------------------------
# full metric bundle for one pair of strings
# ---------------------------------------------------------------------------
def all_metrics(ref_seq: str, qry_seq: str, best_orientation: bool = False) -> dict:
    """Compute every sequence metric for (ref, query).

    best_orientation=True also tries revcomp(query) and keeps the orientation that
    maximises the nucleotide-SW score (mirrors the embedding _dist min-over-orient).
    """
    ra = enc(ref_seq)
    qa = enc(qry_seq)
    cands = [qa]
    if best_orientation:
        cands.append(enc(revcomp(qry_seq)))

    best = None
    for qc in cands:
        raw, swn, swi = sw_metrics(ra, qc)
        if best is None or raw > best["sw_raw"]:
            best = dict(
                sw_raw=raw, sw_norm=swn, sw_aligned_ident=swi,
                ident_levenshtein=float(_edit_ident(ra, qc)),
                kmer3_cos=kmer_cosine(ra, qc, 3),
                kmer4_cos=kmer_cosine(ra, qc, 4),
                dinuc_dist=dinuc_dist(ra, qc),
                _q=qc,
            )
    q_used = best.pop("_q")
    best.update(
        len_ref=len(ra),
        len_query=len(qa),
        len_ratio=min(len(ra), len(qa)) / max(len(ra), len(qa)) if max(len(ra), len(qa)) else 0.0,
        gc_ref=gc_content(ra),
        gc_query=gc_content(q_used),
        gc_diff=abs(gc_content(ra) - gc_content(q_used)),
        has_N=bool((ra == 4).any() or (qa == 4).any()),
    )
    return best


# warm up numba
_ = _edit_ident(enc("ACGU"), enc("ACGU"))
_ = _sw_affine(enc("ACGU"), enc("ACGU"), 2.0, -1.0, 5.0, 1.0)
_ = _kmer_counts(enc("ACGUACGU"), 3, 64)
