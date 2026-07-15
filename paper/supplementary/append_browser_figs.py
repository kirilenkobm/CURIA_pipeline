#!/usr/bin/env python3
"""Append the two UCSC-browser-view supplementary figures (S8 MALAT1, S9 XIST)
to the existing ViennaRNA supplementary PDF and write the combined document as
`supplementary.pdf`.

The existing PDF (2 intro pages + figures S1-S7 + Table S1) is produced by
analysis/vienna_structure_examples.ipynb (matplotlib PdfPages). This script does
NOT re-render those pages: it renders the two new figure pages here and
concatenates them after the existing pages with qpdf, so S1-S7/Table S1 remain
byte-identical and their numbering/order is untouched.

Run from anywhere:  python paper/supplementary/append_browser_figs.py
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent           # paper/supplementary
REPO = HERE.parents[1]
SCRATCH = REPO / "analysis" / "scratch"

EXISTING = HERE / "vienna_structure_supplementary.pdf"
OUT = HERE / "supplementary.pdf"

# A4 portrait, matching the supplement's text pages (8.27 x 11.69 in).
PAGE_W, PAGE_H = 8.27, 11.69

# Figure-fraction layout: image sits in a box above a bottom caption band.
IMG_BOX = dict(x0=0.07, x1=0.93, y0=0.215, y1=0.965)   # available image rectangle
CAP_TOP_Y = 0.185                                       # caption anchored below the box
CAP_LEFT_X = 0.07
CAP_FONTSIZE = 8.0
CAP_WRAP = 108                                          # chars/line ~ full text width @ 8pt

# New figures, in order. Captions edited only for plaintext consistency with the
# existing supplement (which spells "Smith-Waterman" with single hyphens and uses
# "embedding-SW"): the manuscript "embedding--Smith--Waterman" becomes
# "embedding-Smith-Waterman". Content is otherwise as supplied.
FIGS = [
    dict(
        num=8,
        image=SCRATCH / "MALAT1_screenshots.png",
        caption=(
            "Figure S8. Representative locus-level recovery of MALAT1 across mammalian query "
            "genomes. Stacked UCSC Genome Browser views show accepted CURIA query islands for "
            "mouse, rat, cat, cow, armadillo, and opossum. Darker blue indicates lower "
            "embedding-Smith-Waterman distance. The same localized terminal reference cores are "
            "recurrently recovered across the species panel, including genomes in which a native "
            "MALAT1 annotation is absent or incomplete. Gene annotations and cross-species "
            "transcript tracks are shown where available. These views are intended as qualitative "
            "locus-level examples and do not by themselves establish conserved molecular function."
        ),
    ),
    dict(
        num=9,
        image=SCRATCH / "XIST_screenshots.png",
        caption=(
            "Figure S9. Representative locus-level recovery of XIST across placental mammalian "
            "query genomes. Stacked UCSC Genome Browser views show accepted CURIA query islands "
            "for mouse, rat, cat, cow, and armadillo. Darker blue indicates lower "
            "embedding-Smith-Waterman distance. Several recurrent reference cores cluster within a "
            "localized portion of the XIST locus across species, whereas other internal regions "
            "are recovered less consistently. Gene annotations and cross-species transcript tracks "
            "are shown where available. No correspondence to a specific named XIST repeat or "
            "functional domain is asserted here."
        ),
    ),
]


def _fitted_axes_rect(img_w, img_h):
    """Largest rectangle (figure fraction) fitting the image in IMG_BOX, aspect preserved."""
    box_w_in = (IMG_BOX["x1"] - IMG_BOX["x0"]) * PAGE_W
    box_h_in = (IMG_BOX["y1"] - IMG_BOX["y0"]) * PAGE_H
    aspect = img_h / img_w                       # height / width
    w_in = box_w_in
    h_in = w_in * aspect
    if h_in > box_h_in:                          # tall image -> height-bound
        h_in = box_h_in
        w_in = h_in / aspect
    ax_w = w_in / PAGE_W
    ax_h = h_in / PAGE_H
    ax_x = 0.5 - ax_w / 2                         # centre horizontally
    box_cy = (IMG_BOX["y0"] + IMG_BOX["y1"]) / 2
    ax_y = box_cy - ax_h / 2                      # centre vertically in the box
    return [ax_x, ax_y, ax_w, ax_h], (w_in, h_in)


def render(pdf, fig_spec):
    img = ImageOps.exif_transpose(Image.open(fig_spec["image"]))   # keep intended orientation
    rect, (w_in, h_in) = _fitted_axes_rect(*img.size)
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes(rect)
    ax.imshow(img, interpolation="none")         # no resampling of the source pixels
    ax.axis("off")
    caption = textwrap.fill(fig_spec["caption"], width=CAP_WRAP)
    fig.text(CAP_LEFT_X, CAP_TOP_Y, caption, fontsize=CAP_FONTSIZE, va="top", ha="left",
             linespacing=1.35)
    pdf.savefig(fig, dpi=300)                     # ~native for these 300-dpi screenshots
    plt.close(fig)
    disp_dpi = img.size[0] / w_in
    print(f"  S{fig_spec['num']}: {fig_spec['image'].name} {img.size[0]}x{img.size[1]}px "
          f"-> {w_in:.2f}x{h_in:.2f} in  (~{disp_dpi:.0f} dpi displayed)")


def main():
    if not EXISTING.exists():
        sys.exit(f"missing existing supplement: {EXISTING}")
    for f in FIGS:
        if not f["image"].exists():
            sys.exit(f"missing image: {f['image']}")

    new_pages = HERE / "_new_browser_figs.pdf"
    print("rendering new figure pages:")
    with PdfPages(new_pages) as pdf:
        for f in FIGS:
            render(pdf, f)

    print(f"merging {EXISTING.name} + 2 new pages -> {OUT.name}")
    tmp_out = HERE / "_merged.pdf"
    subprocess.run(["qpdf", "--empty", "--pages", str(EXISTING), str(new_pages), "--",
                    str(tmp_out)], check=True)
    tmp_out.replace(OUT)
    new_pages.unlink()

    n = subprocess.run(["qpdf", "--show-npages", str(OUT)],
                       capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["qpdf", "--check", str(OUT)], check=True,
                   stdout=subprocess.DEVNULL)
    print(f"wrote {OUT.relative_to(REPO)} ({n} pages; was 10, +2 new figures S8/S9)")


if __name__ == "__main__":
    main()
