#!/usr/bin/env python3
"""Graphical abstract (PGF backend: all text typeset by LaTeX).

Top row: the master QCAR table, the baseline ANN workflow and its headline
R value, and the question the study asks. Middle row: one data-driven mini
panel per audit strategy S1-S6. Bottom: the conclusion.
"""
from __future__ import annotations

import json
import subprocess

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import FIGURES_DIR, RESULTS_DIR, TARGETS, ensure_directories, load_qcar  # noqa: E402

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif", "font.size": 7, "axes.labelsize": 6, "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5, "axes.linewidth": 0.5, "lines.linewidth": 0.9,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
BLACK, ORANGE, SKY, GREEN, BLUE, VERM, PURPLE = "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"
GREY, LIGHT = "#7f7f7f", "#f2f2f2"
TCOL = dict(zip(TARGETS, [BLUE, VERM, GREEN, PURPLE]))


def box(fig, x, y, w, h, face=LIGHT, edge=GREY, lw=0.6):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=face, edgecolor=edge, lw=lw, zorder=-10))


def arrow(fig, x0, y0, x1, y1, color=GREY):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=9, color=color, lw=1.0))


def mini(ax, title, finding):
    ax.set_title(title, fontsize=6.5, pad=2, loc="left", fontweight="bold")
    ax.text(1.12, 1.0, finding, transform=ax.transAxes, ha="left", va="top", fontsize=6.0, linespacing=1.25)


def main() -> None:
    ensure_directories()
    data = load_qcar()
    ext = json.loads((RESULTS_DIR / "extended_summary.json").read_text())
    fig = plt.figure(figsize=(6.3, 5.3))
    K = 4.35 / 5.3
    Y = lambda v: 1 - (1 - v) * K  # noqa: E731

    # ------------------------------------------------------------ top row
    box(fig, 0.005, Y(0.66), 0.30, 0.325 * K)
    fig.text(0.02, Y(0.965), r"\textbf{Master QCAR table}", va="top", fontsize=7.5)
    fig.text(0.02, Y(0.925), r"667 constructed rows: 9 elements (wt.\%)" + "\n" + r"$\rightarrow$ TS, PS, SS, HV", va="top", fontsize=6)
    ax = fig.add_axes([0.05, Y(0.705), 0.23, 0.14 * K])
    for t in TARGETS:
        v = data[t].to_numpy(float)
        ax.plot(data.row_id, (v - v.min()) / (v.max() - v.min()), color=TCOL[t], lw=0.8)
    ax.axvline(498.5, color=GREY, ls="--", lw=0.5)
    ax.set_xticks([0, 499, 666]); ax.set_yticks([0, 1])
    ax.set_xlabel("row index", labelpad=0.5); ax.set_ylabel("property", labelpad=0.5)
    ax.text(250, 1.02, "train", fontsize=5.5, ha="center", color=GREY)
    ax.text(583, 1.02, "test", fontsize=5.5, ha="center", color=GREY)

    box(fig, 0.345, Y(0.66), 0.30, 0.325 * K)
    fig.text(0.36, Y(0.965), r"\textbf{Baseline ANN workflow}", va="top", fontsize=7.5)
    ax = fig.add_axes([0.355, Y(0.70), 0.14, 0.215 * K]); ax.set_axis_off(); ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.1, 1.1)
    layers = [np.linspace(0, 1, 9), np.linspace(0.05, 0.95, 9), [0.5]]
    for k in range(2):
        for a in layers[k]:
            for b in layers[k + 1]:
                ax.plot([k, k + 1], [a, b], color=GREY, lw=0.15, alpha=0.6)
    for k, ys in enumerate(layers):
        for y in ys:
            ax.add_patch(Circle((k, y), 0.045, color=[BLUE, ORANGE, GREEN][k], zorder=3))
    ax.text(0, -0.1, "9", ha="center", va="top", fontsize=5.5); ax.text(1, -0.1, "9", ha="center", va="top", fontsize=5.5)
    ax.text(2, 0.38, "1", ha="center", va="top", fontsize=5.5)
    fig.text(0.505, Y(0.895), "random 70:15:15" + "\n" + "division of all rows", va="top", fontsize=6)
    fig.text(0.505, Y(0.79), r"$R \approx 0.9999$", va="top", fontsize=11, color=BLUE)
    fig.text(0.505, Y(0.735), "on the random" + "\n" + "test subset", va="top", fontsize=6)

    box(fig, 0.685, Y(0.66), 0.31, 0.325 * K, face="#fff4e5", edge=ORANGE)
    fig.text(0.70, Y(0.965), r"\textbf{The question}", va="top", fontsize=7.5)
    fig.text(0.70, Y(0.915), "Does $R\\approx0.9999$ show that the" + "\n" + "model can predict the properties" + "\n" +
             "of an \\emph{unseen} alloy?", va="top", fontsize=6.5)
    fig.text(0.70, Y(0.79), r"\textbf{Audit suite:} six strategies" + "\n" + "that test the data, the split, the" + "\n" +
             "validation, the leakage, the" + "\n" + "interpretation, and the uncertainty", va="top", fontsize=6)
    arrow(fig, 0.308, Y(0.82), 0.342, Y(0.82)); arrow(fig, 0.648, Y(0.82), 0.682, Y(0.82))
    arrow(fig, 0.84, Y(0.655), 0.84, Y(0.615), color=ORANGE)

    # ------------------------------------------------------------ audit suite
    box(fig, 0.005, 0.115, 0.99, 0.565, face="white", edge=ORANGE, lw=0.8)
    fig.text(0.015, 0.672, r"\textbf{Audit suite and key findings}", va="top", fontsize=7.5, color=ORANGE)
    cols_x = [0.068, 0.382, 0.692]
    rows_y = [0.415, 0.155]
    pos = [(cols_x[k % 3], rows_y[k // 3]) for k in range(6)]
    w, h = 0.115, 0.185

    # S1 physics
    pm = pd.read_csv(RESULTS_DIR / "physics_master.csv")
    ax = fig.add_axes([pos[0][0], pos[0][1], w, h])
    ax.plot(pm.row_id, pm.r_ps, color=BLUE)
    ax.axhline(1, color=VERM, ls="--", lw=0.6)
    ax.fill_between(pm.row_id, 1, pm.r_ps, where=pm.r_ps > 1, color=VERM, alpha=0.25, lw=0)
    ax.set_xticks([0, 666]); ax.set_xlabel("row index", labelpad=0.5); ax.set_ylabel(r"$R_{p0.2}/R_m$", labelpad=0.5)
    mini(ax, "S1 Integrity", f"proof stress $>$\ntensile strength in\n{ext['s1']['master_rows_ps_gt_ts']}/667 rows;\n"
                             "source strengths\ncopy a handbook\ntemper sequence")

    # S2 geometry
    nn = pd.read_csv(RESULTS_DIR / "nn_distances_extended.csv")
    nn = nn[nn.scaling == "z-standardized"]
    ax = fig.add_axes([pos[1][0], pos[1][1], w, h])
    for p, c in (("Random holdout", BLUE), ("Forward holdout", BLACK)):
        v = np.sort(nn[nn.protocol == p].distance.to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), color=c, where="post")
    ax.set_xscale("log"); ax.set_xticks([1e-2, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("NN distance", labelpad=0.5); ax.set_ylabel("ECDF", labelpad=0.5)
    ax.text(0.012, 0.55, "random", fontsize=5.5, color=BLUE); ax.text(0.5, 0.25, "forward", fontsize=5.5)
    ratio = ext["s2"]["z-standardized"]["forward_to_random_median_ratio"]
    mini(ax, "S2 Geometry", f"forward test rows lie\n{ratio:.0f}$\\times$ farther from\ntraining rows than\nrandom holdouts")

    # S3 validation regimes
    b = pd.read_csv(RESULTS_DIR / "model_benchmark_extended.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    ax = fig.add_axes([pos[2][0], pos[2][1], w, h])
    prots = ["Repeated shuffled 5-fold", "Contiguous 5-fold", "Original forward holdout"]
    cols = [BLUE, ORANGE, VERM]
    et = [b[(b.model == "Extra Trees") & (b.protocol == p)].nmae.median() for p in prots]
    an = [rep[rep.protocol == p].nmae.median() for p in prots]
    xx = np.arange(3)
    base = 5e-4
    ax.bar(xx - 0.2, np.array(et) - base, 0.38, color=cols, bottom=base)
    ax.bar(xx + 0.2, np.array(an) - base, 0.38, bottom=base, color=cols, alpha=0.45, hatch="////", edgecolor=cols, lw=0.3)
    ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(["shuf.", "cont.", "fwd."])
    ax.set_ylabel("nMAE", labelpad=0.5); ax.set_ylim(5e-4, 1)
    ax.set_ylim(5e-4, 8)
    mini(ax, "S3 Validation", "$R^2>0.999$ (shuffled)\nbecomes $R^2<0$\n(forward); ANN $R$ holds\nonly for random splits\n\\textit{solid: Extra Trees}\n\\textit{hatched: ANN replica}")

    # S4 negative controls
    c = pd.read_csv(RESULTS_DIR / "negative_controls.csv")
    ax = fig.add_axes([pos[3][0], pos[3][1], w, h])
    vals = [c[(c.control == k) & (c.model == "Extra Trees")].r2.median()
            for k in ("Master table", "Comonotone surrogate", "Response randomization")]
    ax.bar([0, 1, 2], vals, color=[BLACK, GREEN, VERM], width=0.6)
    ax.axhline(0, color=BLACK, lw=0.4)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["real", "sorted", "$y$-scr."], fontsize=5); ax.set_ylabel(r"shuffled $R^2$", labelpad=0.5)
    ax.set_ylim(-0.5, 1.1)
    mini(ax, "S4 Leakage", "sorted surrogates with\nno composition link\nmatch the real table;\nrow order alone wins")

    # S5 attribution
    st = pd.read_csv(RESULTS_DIR / "importance_stability.csv")
    ax = fig.add_axes([pos[4][0], pos[4][1], w, h])
    for k, (p, col) in enumerate((("Shuffled 5-fold", BLUE), ("Contiguous 5-fold", ORANGE))):
        s = st[st.protocol == p].set_index("target").loc[TARGETS].mean_pairwise_rank_spearman
        ax.bar(np.arange(4) + (k - 0.5) * 0.38, s, 0.38, color=col)
    ax.axhline(0, color=BLACK, lw=0.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(["TS", "PS", "SS", "HV"], fontsize=5); ax.set_ylim(-0.2, 1.35)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"rank $\rho$", labelpad=0.5)
    ax.text(0.02, 0.99, "shuffled", transform=ax.transAxes, fontsize=4.8, color=BLUE, va="top")
    ax.text(0.02, 0.89, "contiguous", transform=ax.transAxes, fontsize=4.8, color=ORANGE, va="top")
    mini(ax, "S5 Attribution", "Cu, Mg, Mn, Zn are\nperfect proxies;\nrankings collapse\nin contiguous folds")

    # S6 conformal
    cf = pd.read_csv(RESULTS_DIR / "conformal_extended.csv")
    ax = fig.add_axes([pos[5][0], pos[5][1], w, h])
    models = ["Extra Trees", "Ridge", "MLP"]
    for k, (regime, col) in enumerate((("Random exchangeable", BLUE), ("Forward shift", VERM))):
        v = [cf[(cf.model == m) & (cf.regime == regime)].coverage.mean() for m in models]
        ax.bar(np.arange(3) + (k - 0.5) * 0.38, v, 0.38, color=col)
    ax.axhline(0.9, color=BLACK, ls="--", lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["ET", "Ridge", "MLP"], fontsize=5); ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 0.9])
    ax.text(0.02, 0.99, "random", transform=ax.transAxes, fontsize=4.8, color=BLUE, va="top")
    ax.text(0.98, 0.99, "forward", transform=ax.transAxes, fontsize=4.8, color=VERM, ha="right", va="top")
    ax.set_ylabel("coverage", labelpad=0.5)
    mini(ax, "S6 Uncertainty", "90\\% intervals hold\nfor random splits but\ncover 0--22\\% of\nforward rows")

    # ------------------------------------------------------------ conclusion
    box(fig, 0.005, 0.005, 0.99, 0.09, face="#e8f1fa", edge=BLUE, lw=0.8)
    fig.text(0.5, 0.05, r"\textbf{Conclusion:} the near-perfect accuracy reflects row order and interpolation, not metallurgy." + "\n" +
             "Rebuild the data with provenance and grouped validation before interpretation or inverse alloy design.",
             ha="center", va="center", fontsize=7)
    arrow(fig, 0.5, 0.113, 0.5, 0.098, color=BLUE)

    for a in fig.axes:
        a.patch.set_alpha(0.0)
    name = "fig_graphical_abstract"
    fig.savefig(FIGURES_DIR / f"{name}.pgf")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(FIGURES_DIR / f"{name}.pdf"),
                    str(FIGURES_DIR / name)], check=False)
    print("wrote", name)


if __name__ == "__main__":
    main()
