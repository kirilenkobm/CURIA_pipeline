#!/usr/bin/env python3
"""Single-locus island-matching instrument.

For one gene + query species, reports EXACTLY where its island does/doesn't
become a matched core, reusing the real pipeline components (no reimplemented
scoring):
  * finder  = logreg on find-PCA(k64) of mean-pooled windows (w=72, stride=16,
    smooth=5, prob>=0.25) + real _get_islands merge  -> query islands
  * matcher = real RiNALMo _dotplot_sw (sw_tau_cos=0.5, sw_gap=0.3) -> score,
    ranges, mean_cos; dist=1/(1+score); eff_nt; ceiling max_match_dist=0.5;
    min_match_eff_nt=40.

Emits: reference-island length, projected query-interval length, #query islands,
best query-island length, and per candidate raw SW score / aligned nt / eff_nt /
mean_cos / distance / ceiling / accept + rejection reason, top-3 before filtering.

    .venv/bin/python analysis/debug_locus.py --gene RMRP --species mm39
    .venv/bin/python analysis/debug_locus.py --gene RMRP --gene RPPH1 --species mm39
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for sub in ("pipeline", "RiNALMo"):
    sys.path.insert(0, str(REPO / "modules" / sub))
sys.path.insert(0, str(REPO))

from modules.model_registry import load_model, get_pca_path, get_finding_pca_path, get_logreg_path  # noqa
from modules.pipeline.matchers.rinalmo import _dotplot_sw                                            # noqa
from modules.pipeline.reference_islands_scanner import _get_islands                                  # noqa
from modules.utils.signal_processing import smooth_signal                                            # noqa
from modules.logreg_signal_noise.apply_logreg import load_logreg_model, score_embeddings             # noqa
from short_ncrna import _extract_sequence                                                            # noqa
from pyrion import TwoBitAccessor                                                                     # noqa

# --- deployed params (model_registry island_scan + island_alignment defaults) ---
WIN, STRIDE, SMOOTH, PROB_THR = 72, 16, 5, 0.25
SW_TAU_COS, SW_GAP = 0.5, 0.3
MAX_MATCH_DIST, MIN_EFF_NT = 0.5, 40

SYMBOL_OVERRIDES = {"RMRP": "ENSG00000277027", "RPPH1": "ENSG00000277209",
                    "XACT": "ENSG00000241743", "JPX": "ENSG00000225470",
                    "MALAT1": "ENSG00000251562", "NEAT1": "ENSG00000245532"}


def _pca(path):
    d = np.load(path)
    return d["mean"].astype(np.float32), d["components"].astype(np.float32)  # (1280,), (k,1280)


class Embedder:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"# device: {self.device}; loading RiNALMo ...", flush=True)
        self.model, self.tok, self.ext = load_model("rinalmo", self.device)
        self.fmean, self.fcomp = _pca(get_finding_pca_path("rinalmo"))   # k64 (finding)
        self.mmean, self.mcomp = _pca(get_pca_path("rinalmo"))           # k16 (matching)
        self.logreg = load_logreg_model(str(get_logreg_path("rinalmo")))

    def _per_token(self, seq):
        with torch.no_grad():
            r = self.ext(self.model, self.tok([seq]))[0].float().cpu().numpy()
        return r[1:1 + len(seq)]                     # drop BOS, keep L tokens (as executor)

    def match_repr(self, seq):
        return (self._per_token(seq) - self.mmean) @ self.mcomp.T       # (L,16)

    def find_prob(self, windows):
        vecs = np.stack([self._per_token(w).mean(axis=0) for w in windows])   # (N,1280)
        proj = (vecs - self.fmean) @ self.fcomp.T                             # (N,64)
        return score_embeddings(proj, model=self.logreg)[0]


def find_query_islands(emb, region_seq):
    L = len(region_seq)
    if L < WIN:
        return []
    pos = list(range(0, L - WIN + 1, STRIDE))
    wins = [region_seq[i:i + WIN] for i in pos]
    probs = smooth_signal(emb.find_prob(wins), SMOOTH)
    mask = probs >= PROB_THR
    isl = _get_islands(mask, np.array(pos), WIN)
    for i in isl:
        i["max_prob"] = float(np.max(probs[i["indices"]]))
        i["end"] = min(i["end"], L)
    return isl


def score(emb, ref_seq, q_seq):
    a, b = emb.match_repr(ref_seq), emb.match_repr(q_seq)
    sc, r0, r1, q0, q1, mcos = _dotplot_sw(a, b, SW_TAU_COS, SW_GAP)
    eff = (int(r1 - r0) + int(q1 - q0)) // 2
    dist = 1.0 / (1.0 + sc) if sc > 0 else float("inf")
    if sc <= 0:
        reason = "SW_SCORE<=0"
    elif eff < MIN_EFF_NT:
        reason = f"EFF_NT<{MIN_EFF_NT} (={eff})"
    elif dist > MAX_MATCH_DIST:
        reason = f"DIST>{MAX_MATCH_DIST} (={dist:.3f})"
    else:
        reason = "ACCEPT"
    return dict(sw=sc, ref_aln=int(r1 - r0), q_aln=int(q1 - q0), eff_nt=eff,
                mean_cos=mcos, dist=dist, verdict=reason)


def resolve(g):
    return g if g.startswith("ENSG") else SYMBOL_OVERRIDES.get(g, g)


def run(emb, gene, species, results_dir, ref_2bit, q_2bit):
    ens = resolve(gene)
    D = results_dir / f"hg38_vs_{species}"
    ref_acc, q_acc = TwoBitAccessor(str(ref_2bit)), TwoBitAccessor(str(q_2bit))

    # reference transcript chrom/strand + island(s)
    bed = [l.split("\t") for l in open(D / "reference_union_transcripts.bed")]
    row = next((r for r in bed if ens in r[3]), None)
    refj = json.load(open(D / "preprocessed_reference_data.json"))
    tid = next((k for k in refj if ens in k), None)
    print(f"\n===== {gene} ({ens})  vs  {species} =====")
    if row is None or tid is None or not refj[tid].get("islands"):
        print(f"  no reference transcript/island (row={row is not None}, islands="
              f"{refj.get(tid, {}).get('islands') if tid else None})")
        return
    chrom, strand = row[0], (1 if row[5].strip() == "+" else -1)
    nblocks = int(row[9])
    islands = refj[tid]["islands"]

    # query region (union_to_query -> clusters)
    u2q = json.load(open(D / "mappings" / "union_to_query.json"))
    clusters = json.load(open(D / "mappings" / "query_regions_clusters.json"))
    mids = u2q.get(tid, [])
    if not mids:
        print("  REJECT @ liftover: transcript not in union_to_query (no projected region)")
        return

    for isl in islands:
        rlen = isl["end"] - isl["start"]
        if nblocks != 1:
            print(f"  [ref island {isl['start']}-{isl['end']} len={rlen}] multi-exon "
                  f"transcript ({nblocks} exons) — harness handles single-exon only; skipping")
            continue
        ref_seq = _extract_sequence(ref_acc, chrom, isl["start"], isl["end"], strand)
        print(f"  ref island: {chrom}:{isl['start']}-{isl['end']} len={rlen}")
        for mid in mids:
            mr = clusters[mid]["merged_region"]
            qs, qe = int(mr["start"]), int(mr["end"])
            qseq = _extract_sequence(q_acc, mr["chrom"], qs, qe, int(mr["strand"]))
            print(f"  query region {mid}: {mr['chrom']}:{qs}-{qe} len={qe - qs} strand={mr['strand']}")
            qisl = find_query_islands(emb, qseq)
            print(f"    query islands detected: {len(qisl)}"
                  + (f"  lens={[i['end'] - i['start'] for i in qisl]}"
                     f"  max_prob={[round(i['max_prob'], 3) for i in qisl]}" if qisl else ""))
            if not qisl:
                print("    >>> REJECT @ QUERY_FINDER: no query island detected in region")
                continue
            cands = []
            for qi in qisl:
                qsub = qseq[qi["start"]:qi["end"]]
                m = score(emb, ref_seq, qsub)
                m["q_island_len"] = len(qsub)
                cands.append(m)
            cands.sort(key=lambda c: -c["sw"])
            print(f"    ceiling: dist<={MAX_MATCH_DIST}, eff_nt>={MIN_EFF_NT}. Top candidates:")
            for c in cands[:3]:
                print(f"      qlen={c['q_island_len']:4d} SW={c['sw']:7.2f} "
                      f"ref_aln={c['ref_aln']:3d} q_aln={c['q_aln']:3d} eff_nt={c['eff_nt']:3d} "
                      f"mean_cos={c['mean_cos']:.3f} dist={c['dist']:.3f} -> {c['verdict']}")
            acc = [c for c in cands if c["verdict"] == "ACCEPT"]
            print(f"    >>> {'MATCH' if acc else 'REJECT'}: "
                  + ("accepted" if acc else f"best-candidate reason = {cands[0]['verdict']}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", action="append", required=True, help="symbol or ENSG (repeatable)")
    ap.add_argument("--species", default="mm39")
    ap.add_argument("--results-dir", type=Path, default=REPO / "preprint_results")
    ap.add_argument("--ref-2bit", type=Path, default=REPO / "input_data/2bit/hg38.2bit")
    ap.add_argument("--query-2bit", type=Path, default=None)
    args = ap.parse_args()
    q2 = args.query_2bit or (REPO / f"input_data/2bit/{args.species}.2bit")
    emb = Embedder()
    for g in args.gene:
        run(emb, g, args.species, args.results_dir, args.ref_2bit, q2)


if __name__ == "__main__":
    main()
