#!/usr/bin/env python3
"""Generate paper figures into paper/figures/*.pdf.

Each figure is one function registered in FIGURES. Add a function, decorate it
with @figure("fig1_overview"), and it will be written to <outdir>/fig1_overview.pdf.

Figures you build by hand in Affinity can simply be dropped into paper/figures/
and left out of this script -- this is only for the programmatic ones.

Usage:
    python analysis/make_figures.py --outdir paper/figures
    python analysis/make_figures.py --outdir paper/figures --only fig3_results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

FIGURES: dict[str, Callable[[Path], None]] = {}


def figure(name: str):
    """Register a figure builder under `name` (-> <outdir>/<name>.pdf)."""

    def deco(fn: Callable[[plt.Figure], None]):
        def build(outdir: Path) -> None:
            fig = plt.figure()
            fn(fig)
            out = outdir / f"{name}.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

        FIGURES[name] = build
        return fn

    return deco


# --------------------------------------------------------------------------
# Figure builders. These are placeholders; wire them to real data as results
# from the 10-mammal RiNALMo run land.
# --------------------------------------------------------------------------


@figure("fig3_results")
def _fig3(fig: plt.Figure) -> None:
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "fig3_results\n(placeholder)", ha="center", va="center")
    ax.set_axis_off()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    p.add_argument("--only", nargs="*", help="only build these figure names")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure: {name} (have: {', '.join(FIGURES)})")
        FIGURES[name](args.outdir)


if __name__ == "__main__":
    main()
