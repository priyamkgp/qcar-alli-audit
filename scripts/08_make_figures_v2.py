#!/usr/bin/env python3
"""Publication figures for Section 3.

All text in the figures (axis labels, tick labels, legends, panel letters) is
typeset by LaTeX through Matplotlib's PGF backend. Each figure is written as
  figures/<name>.pgf  -- \\input into the manuscript (document fonts), and
  figures/<name>.pdf  -- compiled by pdflatex (Computer Modern), for \\includegraphics.
"""
from __future__ import annotations

import subprocess
import sys

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (DATA_DIR, FEATURES, FIGURES_DIR, PREDICTION_COLUMNS, RESULTS_DIR,  # noqa: E402
                    TARGETS, ensure_directories, load_qcar)

WIDTH = 6.3  # inches; single-column text width
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe--Ito palette
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#7f7f7f"
PROTOCOL_COLOR = {"Repeated shuffled 5-fold": BLUE, "Contiguous 5-fold": ORANGE, "Original forward holdout": VERM}
PROTOCOL_LABEL = {"Repeated shuffled 5-fold": "Shuffled 5-fold", "Contiguous 5-fold": "Contiguous 5-fold",
                  "Original forward holdout": "Forward holdout"}
TARGET_LABEL = {TARGETS[0]: "Tensile strength", TARGETS[1]: "Proof stress",
                TARGETS[2]: "Shear strength", TARGETS[3]: "Vickers hardness"}
TARGET_UNIT = {TARGETS[0]: "MPa", TARGETS[1]: "MPa", TARGETS[2]: "MPa", TARGETS[3]: "HV"}
TARGET_COLOR = {TARGETS[0]: BLUE, TARGETS[1]: VERM, TARGETS[2]: GREEN, TARGETS[3]: PURPLE}
BOUNDARY = 498.5


def save(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.pgf")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(FIGURES_DIR / f"{name}.pdf"),
                    str(FIGURES_DIR / name)], check=False)
    print("wrote", name, flush=True)


def panel(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, rf"\textbf{{({letter})}}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def minmax(v):
    v = np.asarray(v, float)
    return (v - v.min()) / (v.max() - v.min())


def boundary(ax, label: bool = False):
    ax.axvline(BOUNDARY, color=GREY, lw=0.7, ls="--", zorder=0)
    if label:
        ax.text(BOUNDARY - 6, 0.02, "train/test\nboundary", fontsize=6, color=GREY, va="bottom", ha="right")


# --------------------------------------------------------------------------- S1
def fig_sequences(data):
    resets = pd.read_csv(RESULTS_DIR / "segment_resets.csv")
    order = pd.read_csv(RESULTS_DIR / "order_correlations.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.3))
    ax = axes[0, 0]
    for t in TARGETS:
        ax.plot(data.row_id, minmax(data[t]), color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    boundary(ax)
    ax.set_xlabel(r"Row index $i$"); ax.set_ylabel("Min--max normalized property")
    ax.legend(loc="lower right", ncol=1); panel(ax, "a")

    ax = axes[0, 1]
    for f, c in zip(["Al", "Cu", "Mg", "Mn", "Zn"], [BLACK, BLUE, GREEN, ORANGE, PURPLE]):
        ax.plot(data.row_id, minmax(data[f]), color=c, label=f)
    boundary(ax)
    ax.set_xlabel(r"Row index $i$"); ax.set_ylabel("Min--max normalized mass fraction")
    ax.legend(loc="center left", ncol=1); panel(ax, "b")

    ax = axes[1, 0]
    ax.axvspan(343, 586.5, color=GREY, alpha=0.12, lw=0)
    ax.text(465, 0.93, r"Cd $=0$" + "\n(rows 343--586)", ha="center", fontsize=6, color=GREY)
    for f, c in zip(["Li", "Zr", "Si", "Cd"], [VERM, SKY, ORANGE, BLACK]):
        ax.plot(data.row_id, minmax(data[f]), color=c, label=f)
        rr = resets[resets.variable == f].row_id
        ax.plot(rr, [-0.06] * len(rr), marker="^", ls="none", ms=4, color=c, clip_on=False)
    boundary(ax)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel(r"Row index $i$"); ax.set_ylabel("Min--max normalized mass fraction")
    ax.legend(loc="upper left", ncol=1, handlelength=1.2); panel(ax, "c")

    ax = axes[1, 1]
    order = order.sort_values("spearman_with_row_id")
    names = {t: TARGET_LABEL[t] for t in TARGETS}
    labels = [names.get(v, v) for v in order.variable]
    colors = [BLUE if v in TARGETS else (BLACK if abs(r) >= 0.98 else ORANGE)
              for v, r in zip(order.variable, order.spearman_with_row_id)]
    ax.barh(range(len(order)), order.spearman_with_row_id, color=colors, height=0.7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
    ax.axvline(0, color=BLACK, lw=0.5); ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(r"Spearman $\rho$ with row index")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=5, label="Property"),
                       Line2D([], [], color=BLACK, lw=5, label=r"Solute, $|\rho|\geq0.98$"),
                       Line2D([], [], color=ORANGE, lw=5, label=r"Solute, $|\rho|<0.98$")],
              loc="upper left", fontsize=6)
    panel(ax, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "fig_s1_sequences")


def fig_physics(data):
    pm = pd.read_csv(RESULTS_DIR / "physics_master.csv")
    ps = pd.read_csv(RESULTS_DIR / "physics_source.csv")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9), gridspec_kw={"height_ratios": [1.3, 1]})
    specs = [("r_ps", r"$r_{\mathrm{PS}}=R_{p0.2}/R_m$", 1.0, "admissibility limit"),
             ("r_ss", r"$r_{\mathrm{SS}}=\tau_u/R_m$", 1 / np.sqrt(3), r"von Mises $1/\sqrt{3}$"),
             ("kappa", r"$\kappa=3R_m/(g_n\,\mathrm{HV})$", 1.0, "Tabor consistency")]
    rng = np.random.default_rng(0)
    for k, (col, lab, ref, reflab) in enumerate(specs):
        ax = axes[0, k]
        ax.plot(pm.row_id, pm[col], color=BLUE, lw=0.9)
        ax.axhline(ref, color=VERM, lw=0.8, ls="--")
        if col == "r_ps":
            ax.fill_between(pm.row_id, 1, pm[col], where=pm[col] > 1, color=VERM, alpha=0.18, lw=0)
        boundary(ax)
        ax.set_xlabel(r"Row index $i$"); ax.set_ylabel(lab)
        ax.text(0.98, ref, reflab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color=VERM)
        panel(ax, "abc"[k])
        ax = axes[1, k]
        for j, (vals, c, name) in enumerate([(pm[col], BLUE, "Master\n($n=667$)"), (ps[col], ORANGE, "Source\n($n=27$)")]):
            ax.boxplot([vals], positions=[j], widths=0.5, vert=False, showfliers=False,
                       medianprops={"color": BLACK}, boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
            ax.scatter(vals, j + rng.uniform(-0.18, 0.18, len(vals)), s=2 if j == 0 else 7, color=c,
                       alpha=0.35 if j == 0 else 0.9, lw=0, rasterized=(j == 0))
        ax.axvline(ref, color=VERM, lw=0.8, ls="--")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Master\n($n=667$)", "Source\n($n=27$)"])
        ax.set_xlabel(lab)
        panel(ax, "def"[k])
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "fig_s1_physics")


def fig_provenance(data):
    link = pd.read_csv(RESULTS_DIR / "tensile_record_linkage.csv")
    hb = pd.read_csv(DATA_DIR / "handbook_temper_sheet.csv")
    support = pd.read_csv(RESULTS_DIR / "support_ranges.csv")
    anchors = pd.read_csv(RESULTS_DIR / "source_anchor_distances.csv")
    contam = pd.read_csv(RESULTS_DIR / "four_property_contamination.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))

    ax = axes[0, 0]
    ax.axvspan(51.5, 93.5, color=GREY, alpha=0.10, lw=0)
    for start in (0, 52):
        sub = hb.iloc[: min(52, 94 - start)]
        ax.plot(sub.position + start, sub.mpa, color=GREY, lw=0.6, zorder=1)
    ax.scatter(link.record, link.tensile_strength, s=9, facecolor="none", edgecolor=BLUE, lw=0.7, zorder=2,
               label="Source, record order")
    own = link[link.own_designation_aligned]
    ax.scatter(own.record, own.tensile_strength, s=16, color=VERM, zorder=3, label="Own designation")
    ax.plot([], [], color=GREY, lw=0.6, label="Handbook temper sequence")
    ax.text(80, 655, "sequence restarts\nafter 52 records", ha="center", va="top", fontsize=6, color=GREY)
    ax.set_ylim(60, 700)
    ax.set_xlabel("Source record"); ax.set_ylabel("Tensile strength (MPa)")
    ax.legend(loc="upper left", fontsize=6); panel(ax, "a")

    ax = axes[0, 1]
    elems = list(support.element)
    for k, r in support.reset_index(drop=True).iterrows():
        lo, span = r.source_min, (r.source_max - r.source_min)
        f = lambda v: (v - lo) / span
        y = len(elems) - 1 - k
        ax.plot([f(r.train_min), f(r.train_max)], [y + 0.14] * 2, color=BLUE, lw=3.4, solid_capstyle="butt")
        ax.plot([f(r.test_min), f(r.test_max)], [y - 0.14] * 2, color=VERM, lw=3.4, solid_capstyle="butt")
    ax.set_yticks(range(len(elems))); ax.set_yticklabels(elems[::-1])
    ax.axvspan(0, 1, color=GREY, alpha=0.15, lw=0)
    ax.axvline(0, color=BLACK, lw=0.4); ax.axvline(1, color=BLACK, lw=0.4)
    ax.set_xlim(-0.25, 1.05)
    ax.set_xlabel("Position within source range")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.3, label="Source range ($n=94$)"),
                       Line2D([], [], color=BLUE, lw=3, label="Master, rows 0--498"),
                       Line2D([], [], color=VERM, lw=3, label="Master, rows 499--666")],
              loc="lower left", fontsize=5.5, ncol=3, bbox_to_anchor=(-0.02, 1.0), handlelength=1.0, columnspacing=0.6)
    panel(ax, "b", x=-0.12, y=1.02)

    ax = axes[1, 0]
    d = np.sort(anchors.maximum_absolute_component_difference_wt_pct.to_numpy())
    ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=BLUE)
    for thr, lab in ((0.001, "0.001"), (0.05, "0.05")):
        ax.axvline(thr, color=VERM, ls="--", lw=0.7)
        ax.text(thr * 1.15, 0.9, lab + r"\,wt.\%", fontsize=6, color=VERM)
    ax.set_xscale("log"); ax.set_xlim(5e-4, 20)
    ax.set_xlabel(r"$d_\infty$ to nearest master row (wt.\%)")
    ax.set_ylabel("Cumulative fraction of source records")
    panel(ax, "c")

    ax = axes[1, 1]
    props = [("tensile_strength", "TS"), ("proof_stress", "PS"), ("shear_strength", "SS"), ("hardness", "HV")]
    bad = contam[contam.equals_prediction].reset_index(drop=True)
    w = 0.22
    for a, (_, r) in enumerate(bad.iterrows()):
        for p, (col, lab) in enumerate(props):
            x0 = p + (a - 1) * w
            ax.scatter(x0, r[f"orig_{col}"], marker="o", facecolor="none", edgecolor=BLACK, s=18, lw=0.7, zorder=3)
            ax.scatter(x0, r[f"pred_{col}"], marker="x", color=VERM, s=18, lw=0.8, zorder=3)
            ax.scatter(x0, r[col], marker=".", color=BLUE, s=22, zorder=4)
            ax.plot([x0, x0], [r[f"orig_{col}"], r[f"pred_{col}"]], color=GREY, lw=0.5, zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([p[1] for p in props])
    ax.set_ylabel("Value (MPa; HV for hardness)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="none", mec=BLACK, label="Recorded original"),
                       Line2D([], [], marker="x", ls="none", color=VERM, label="ANN prediction"),
                       Line2D([], [], marker=".", ls="none", color=BLUE, ms=8, label="Stored in four-property table")],
              loc="upper right", fontsize=6, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(10, 175)
    panel(ax, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "fig_s1_provenance")


# --------------------------------------------------------------------------- S2
def fig_pca(data):
    p = pd.read_csv(RESULTS_DIR / "pca_scores_both.csv")
    ev = pd.read_csv(RESULTS_DIR / "pca_explained_both.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    for k, (pre, name, col) in enumerate([("z", "z-standardized", "z_ratio"), ("r", "source-range scaled", "range_ratio")]):
        ax = axes[k]
        test = p.split == "testing"
        sc = ax.scatter(p[f"{pre}_PC1"][~test], p[f"{pre}_PC2"][~test], c=p.row_id[~test], cmap="viridis",
                        vmin=0, vmax=666, s=5, lw=0, rasterized=True)
        ax.scatter(p[f"{pre}_PC1"][test], p[f"{pre}_PC2"][test], c=p.row_id[test], cmap="viridis",
                   vmin=0, vmax=666, s=9, marker="D", edgecolor=BLACK, lw=0.3, rasterized=True)
        e1, e2 = ev[col].iloc[0] * 100, ev[col].iloc[1] * 100
        ax.set_xlabel(rf"PC1 ({e1:.1f}\%)"); ax.set_ylabel(rf"PC2 ({e2:.1f}\%)")
        ax.set_title(name, fontsize=7)
        panel(ax, "ab"[k])
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"Row index $i$")
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="none", color=GREY, ms=3, label="Rows 0--498"),
                            Line2D([], [], marker="D", ls="none", mfc=GREY, mec=BLACK, ms=3, label="Rows 499--666")],
                   loc="best", fontsize=6)
    save(fig, "fig_s2_pca")


def fig_nn(data):
    nn = pd.read_csv(RESULTS_DIR / "nn_distances_extended.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    block_colors = ["#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]
    for k, scaling in enumerate(["z-standardized", "source-range"]):
        ax = axes[k]
        s = nn[nn.scaling == scaling]
        def ecdf(v, **kw):
            v = np.sort(v); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kw)
        ecdf(s[s.protocol == "Random holdout"].distance, color=BLUE, lw=1.3, label="Random holdouts (100)")
        for b in range(5):
            ecdf(s[(s.protocol == "Contiguous block") & (s.repeat == b)].distance, color=block_colors[b], lw=0.8,
                 label=f"Contiguous block {b + 1}")
        ecdf(s[s.protocol == "Forward holdout"].distance, color=BLACK, lw=1.5, label="Forward holdout")
        ax.set_xscale("log")
        ax.set_xlabel("Nearest-neighbour distance" + (" (z units)" if k == 0 else " (source-range units)"))
        ax.set_ylabel("Cumulative fraction of held-out rows")
        ax.set_title(scaling, fontsize=7)
        panel(ax, "ab"[k])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.5)
    save(fig, "fig_s2_nn")


# --------------------------------------------------------------------------- S3
def fig_benchmark(data):
    b = pd.read_csv(RESULTS_DIR / "model_benchmark_extended.csv")
    fwd = pd.read_csv(RESULTS_DIR / "forward_bootstrap.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    models = ["Dummy mean", "Ridge", "PLS (3 components)", "RBF-SVR", "Extra Trees", "Histogram GB", "MLP",
              "Baseline ANN replica", "Baseline ANN (stored)", "Order-only trend"]
    labels = ["Dummy mean", "Ridge", "PLS", "RBF-SVR", "Extra Trees", "Hist.\\ GB", "MLP",
              "Baseline ANN replica", "Baseline ANN (stored)", "Order-only trend"]
    marker = {"Repeated shuffled 5-fold": "o", "Contiguous 5-fold": "s", "Original forward holdout": "D"}
    offs = {"Repeated shuffled 5-fold": 0.22, "Contiguous 5-fold": 0.0, "Original forward holdout": -0.22}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), sharey=True)
    for k, t in enumerate(TARGETS):
        ax = axes.flat[k]
        rng_t = float(data[t].max() - data[t].min())
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            for prot in marker:
                if m in b.model.unique() and prot != "Original forward holdout":
                    v = b[(b.model == m) & (b.protocol == prot) & (b.target == t)].nmae.iloc[0]
                    ax.scatter(v, y + offs[prot], marker=marker[prot], color=PROTOCOL_COLOR[prot], s=14, zorder=3)
                elif m == "Baseline ANN replica":
                    r = rep[(rep.protocol == prot) & (rep.target == t)].nmae
                    ax.plot([r.min(), r.max()], [y + offs[prot]] * 2, color=PROTOCOL_COLOR[prot], lw=0.8)
                    ax.scatter(r.median(), y + offs[prot], marker=marker[prot], color=PROTOCOL_COLOR[prot], s=14, zorder=3)
                elif prot == "Original forward holdout":
                    f = fwd[(fwd.model == m) & (fwd.target == t)]
                    if len(f):
                        f = f.iloc[0]
                        ax.plot([f.mae_lo / rng_t, f.mae_hi / rng_t], [y + offs[prot]] * 2, color=VERM, lw=0.8)
                        ax.scatter(f.nmae, y + offs[prot], marker="D", color=VERM, s=14, zorder=3)
        ax.axhspan(0.5, 2.5, color=GREY, alpha=0.10, lw=0)
        ax.set_xscale("log"); ax.set_xlim(3e-4, 5)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(labels[::-1])
        ax.set_xlabel(r"Range-normalized MAE, $\mathrm{nMAE}$")
        ax.set_title(TARGET_LABEL[t], fontsize=7)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        panel(ax, "abcd"[k])
    handles = [Line2D([], [], marker=marker[p], ls="none", color=PROTOCOL_COLOR[p], label=PROTOCOL_LABEL[p]) for p in marker]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=1.0)
    save(fig, "fig_s3_benchmark")


def fig_ann(data):
    native = pd.read_csv(RESULTS_DIR / "ann_replica_native.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    stored = pd.read_csv(RESULTS_DIR / "baseline_ann_stored_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6))
    groups = ["Native", "Shuffled", "Contiguous", "Forward"]
    ticklab = ["Native", "Shuf.", "Cont.", "Fwd."]
    prot = {"Shuffled": "Repeated shuffled 5-fold", "Contiguous": "Contiguous 5-fold", "Forward": "Original forward holdout"}
    rng = np.random.default_rng(1)
    for k, (metric, ylabel) in enumerate([("R", r"$1-R$"), ("nmae", r"$\mathrm{nMAE}$")]):
        ax = axes[k]
        for g_i, g in enumerate(groups):
            for t_i, t in enumerate(TARGETS):
                if g == "Native":
                    v = 1 - native[native.target == t].R_test if metric == "R" else \
                        native[native.target == t].mae_test / float(data[t].max() - data[t].min())
                else:
                    r = rep[(rep.protocol == prot[g]) & (rep.target == t)]
                    v = 1 - r.pooled_R if metric == "R" else r.nmae
                x = g_i + (t_i - 1.5) * 0.17
                ax.scatter(x + rng.uniform(-0.04, 0.04, len(v)), v, s=4, color=TARGET_COLOR[t], alpha=0.7, lw=0)
                ax.plot([x - 0.07, x + 0.07], [np.median(v)] * 2, color=BLACK, lw=0.9)
        if metric == "R":
            ax.axhspan(1 - 0.99998, 1 - 0.99990, color=GREEN, alpha=0.25, lw=0)
            ax.text(3.45, 1.5e-5, "band: $R$ reported\nby the toolbox", fontsize=6, color=GREEN, va="bottom", ha="right")
        ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(ticklab, rotation=0)
        ax.set_ylabel(ylabel); ax.set_xlabel("Protocol (replica)")
        panel(ax, "ab"[k])
    ax = axes[2]
    for t_i, t in enumerate(TARGETS):
        r = rep[(rep.protocol == "Original forward holdout") & (rep.target == t)]
        rng_t = float(data[t].max() - data[t].min())
        ax.scatter(t_i + rng.uniform(-0.15, 0.15, len(r)), r.pooled_mae / rng_t, s=5, color=TARGET_COLOR[t], alpha=0.7, lw=0)
        s = stored[(stored.split == "testing") & (stored.target == t)].nmae.iloc[0]
        ax.scatter(t_i, s, marker="*", s=60, color=BLACK, zorder=4)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(["TS", "PS", "SS", "HV"])
    ax.set_ylabel(r"Forward-holdout $\mathrm{nMAE}$")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=GREY, ms=3, label="Replica, 30 initializations"),
                       Line2D([], [], marker="*", ls="none", color=BLACK, ms=7, label="Stored baseline predictions")],
              loc="upper center", fontsize=6, ncol=1)
    ax.set_ylim(top=8)
    panel(ax, "c")
    handles = [Line2D([], [], marker="o", ls="none", color=TARGET_COLOR[t], ms=3, label=TARGET_LABEL[t]) for t in TARGETS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    save(fig, "fig_s3_ann")


def fig_forward(data):
    preds = pd.read_csv(RESULTS_DIR / "model_predictions.csv")
    preds = preds[preds.protocol == "Original forward holdout"]
    order = pd.read_csv(RESULTS_DIR / "order_trend_predictions.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_predictions.csv")
    rep = rep[rep.protocol == "Original forward holdout"]
    test = data[data.supplied_split == "testing"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True)
    for k, t in enumerate(TARGETS):
        ax = axes.flat[k]
        q = rep[rep.target == t].pivot_table(index="row_id", columns="init", values="predicted")
        lo, med, hi = q.quantile(0.25, axis=1), q.median(axis=1), q.quantile(0.75, axis=1)
        ax.fill_between(q.index, lo, hi, color=GREY, alpha=0.25, lw=0, label="Replica, interquartile range")
        ax.plot(q.index, med, color=GREY, lw=0.8, label="Replica, median")
        for m, c in (("Extra Trees", GREEN), ("MLP", PURPLE)):
            s = preds[(preds.model == m) & (preds.target == t)].sort_values("row_id")
            ax.plot(s.row_id, s.predicted, color=c, lw=0.9, label=m)
        o = order[order.target == t].sort_values("row_id")
        ax.plot(o.row_id, o.predicted, color=ORANGE, lw=0.9, ls="--", label="Order-only trend")
        ax.plot(test.row_id, test[PREDICTION_COLUMNS[t]], color=BLUE, lw=1.0, label="Baseline ANN (stored)")
        ax.plot(test.row_id, test[t], color=BLACK, lw=1.6, label="Target")
        trmax = data[data.supplied_split == "training"][t].max()
        ax.axhline(trmax, color=GREY, lw=0.6, ls=":")
        yl = ax.get_ylim()
        for row, lab in ((587, "Cd"), (609, "Si"), (643, "Li"), (662, "Zr")):
            ax.axvline(row - 0.5, color=VERM, lw=0.5, ls="-", alpha=0.6)
        lo_lim, hi_lim = 0.55 * test[t].min(), 1.15 * test[t].max()
        ax.set_ylim(lo_lim, hi_lim)
        for row, lab in ((587, "Cd"), (609, "Si"), (643, "Li"), (662, "Zr")):
            ax.text(row - 0.5, hi_lim, lab, fontsize=6, color=VERM, ha="center", va="bottom")
        ax.set_ylabel(f"{TARGET_LABEL[t]} ({TARGET_UNIT[t]})")
        if k >= 2:
            ax.set_xlabel(r"Row index $i$ (forward testing set)")
        panel(ax, "abcd"[k], y=1.05)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.4, w_pad=1.2)
    save(fig, "fig_s3_forward")


def fig_blocks_distance(data):
    folds = pd.read_csv(RESULTS_DIR / "model_benchmark_fold_metrics.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_predictions.csv")
    ed = pd.read_csv(RESULTS_DIR / "error_distance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    ax = axes[0]
    cont = folds[folds.protocol == "Contiguous 5-fold"]
    models = [("Ridge", SKY), ("Extra Trees", GREEN), ("MLP", PURPLE)]
    for m_i, (m, c) in enumerate(models):
        for t_i, t in enumerate(TARGETS):
            s = cont[(cont.model == m) & (cont.target == t)].sort_values("fold")
            ax.plot(s.fold + 1 + (m_i - 1) * 0.2, s.r2, marker="os^"[m_i], ls="none", ms=3, color=c,
                    alpha=0.85, label=m if t_i == 0 else None)
    r = rep[(rep.protocol == "Contiguous 5-fold") & (rep.init == 0)]
    from sklearn.metrics import r2_score
    for t_i, t in enumerate(TARGETS):
        vals = [r2_score(g.actual, g.predicted) for _, g in r[r.target == t].groupby("fold")]
        ax.plot(np.arange(1, 6) + 0.3, vals, marker="D", ls="none", ms=3, color=BLACK, alpha=0.8,
                label="Baseline ANN replica" if t_i == 0 else None)
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["0--133", "134--267", "268--400", "401--533", "534--666"], fontsize=6)
    ax.set_xlabel("Held-out contiguous block (rows)"); ax.set_ylabel(r"Fold-level $R^2_k$ (symmetric log)")
    ax.legend(loc="lower left", fontsize=6)
    panel(ax, "a")
    ax = axes[1]
    colors = {"Ridge": SKY, "Extra Trees": GREEN, "MLP": PURPLE, "Baseline ANN replica": BLACK}
    sample = ed.sample(n=min(len(ed), 12000), random_state=0)
    ax.scatter(sample.distance, sample.norm_error, s=1, color=GREY, alpha=0.15, lw=0, rasterized=True)
    bins = np.logspace(np.log10(ed.distance[ed.distance > 0].min()), np.log10(ed.distance.max()), 14)
    for m, c in colors.items():
        g = ed[ed.model == m]
        idx = np.digitize(g.distance, bins)
        med = g.groupby(idx).norm_error.median()
        cen = [np.sqrt(bins[i - 1] * bins[min(i, len(bins) - 1)]) for i in med.index if 0 < i < len(bins)]
        vals = [med[i] for i in med.index if 0 < i < len(bins)]
        ax.plot(cen, vals, color=c, marker="o", ms=2.5, lw=0.9, label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Nearest-neighbour distance (z units)")
    ax.set_ylabel("Absolute error / property range")
    ax.legend(loc="upper left", fontsize=6)
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_s3_blocks_distance")


# --------------------------------------------------------------------------- S4
def fig_controls(data):
    c = pd.read_csv(RESULTS_DIR / "negative_controls.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    style = {"Master table": (BLACK, "*", 45), "Comonotone surrogate": (GREEN, "o", 6), "Response randomization": (VERM, "o", 6)}
    rng = np.random.default_rng(2)
    xt = []
    for k, (ax, transform) in enumerate([(axes[0], "linear"), (axes[1], "log")]):
        pos = 0
        xt = []
        for t in TARGETS:
            for m in ("Ridge", "Extra Trees"):
                for ctrl, (col, mk, sz) in style.items():
                    if transform == "log" and ctrl == "Response randomization":
                        continue
                    v = c[(c.control == ctrl) & (c.model == m) & (c.target == t)].r2.to_numpy()
                    v = v if transform == "linear" else 1 - v
                    jitter = 0 if ctrl == "Master table" else rng.uniform(-0.12, 0.12, len(v))
                    ax.scatter(pos + jitter, v, color=col, marker=mk, s=sz, lw=0, alpha=0.9, zorder=3 if mk == "*" else 2)
                xt.append((pos, ("R" if m == "Ridge" else "ET")))
                pos += 1
            pos += 0.6
        ax.set_xticks([p for p, _ in xt]); ax.set_xticklabels([l for _, l in xt], fontsize=6)
        for i, t in enumerate(TARGETS):
            ax.text(i * 2.6 + 0.5, -0.16, ["TS", "PS", "SS", "HV"][i], transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        if transform == "log":
            ax.set_yscale("log"); ax.set_ylabel(r"$1-R^2_{\mathrm{pool}}$ (shuffled 5-fold)")
        else:
            ax.axhline(0, color=BLACK, lw=0.5); ax.set_ylabel(r"$R^2_{\mathrm{pool}}$ (shuffled 5-fold)")
        panel(ax, "ab"[k])
    axes[0].legend(handles=[Line2D([], [], marker="*", ls="none", color=BLACK, ms=7, label="Master table"),
                            Line2D([], [], marker="o", ls="none", color=GREEN, ms=3, label="Comonotone surrogates (20)"),
                            Line2D([], [], marker="o", ls="none", color=VERM, ms=3, label="Response randomization (20)")],
                   loc="center right", fontsize=6)
    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "fig_s4_controls")


# --------------------------------------------------------------------------- S5
def fig_attribution(data):
    corr = pd.read_csv(RESULTS_DIR / "input_spearman.csv", index_col=0)
    imp = pd.read_csv(RESULTS_DIR / "permutation_importance_summary.csv")
    stab = pd.read_csv(RESULTS_DIR / "importance_stability.csv")
    jac = pd.read_csv(RESULTS_DIR / "importance_jaccard.csv")
    gi = pd.read_csv(RESULTS_DIR / "grouped_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0))
    ax = axes[0, 0]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(9)); ax.set_xticklabels(corr.columns, fontsize=6)
    ax.set_yticks(range(9)); ax.set_yticklabels(corr.index, fontsize=6)
    for i in range(9):
        for j in range(9):
            v = corr.values[i, j]
            txt = "1" if abs(v) > 0.9995 and v > 0 else f"{v:.2f}".replace("0.", ".")
            ax.text(j, i, txt, ha="center", va="center", fontsize=4.3,
                    color="white" if abs(corr.values[i, j]) > 0.7 else BLACK)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"Spearman $\rho$")
    panel(ax, "a")

    ax = axes[0, 1]
    cols = []
    mat = []
    for t in TARGETS:
        for p in ("Shuffled 5-fold", "Contiguous 5-fold"):
            s = imp[(imp.target == t) & (imp.protocol == p)].set_index("feature").loc[FEATURES, "mean_importance"].clip(lower=0)
            mat.append((s / s.sum()).to_numpy())
            cols.append(f"{['TS', 'PS', 'SS', 'HV'][TARGETS.index(t)]}\n{'S' if p.startswith('Sh') else 'C'}")
    mat = np.array(mat).T
    im2 = ax.imshow(mat, cmap="viridis", vmin=0, vmax=mat.max(), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.5,
                    color="white" if mat[i, j] < 0.6 * mat.max() else BLACK)
    ax.set_xticks(range(8)); ax.set_xticklabels(cols, fontsize=6)
    ax.set_yticks(range(9)); ax.set_yticklabels(FEATURES, fontsize=6)
    for x in (1.5, 3.5, 5.5):
        ax.axvline(x, color="white", lw=1.0)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Share of total importance")
    panel(ax, "b")

    ax = axes[1, 0]
    agg = gi.groupby(["target", "protocol", "group"]).mae_increase.mean().reset_index()
    w = 0.2
    for t_i, t in enumerate(TARGETS):
        for j, (p, g) in enumerate([("Shuffled 5-fold", "Monotone group"), ("Shuffled 5-fold", "Reset group"),
                                    ("Contiguous 5-fold", "Monotone group"), ("Contiguous 5-fold", "Reset group")]):
            v = agg[(agg.target == t) & (agg.protocol == p) & (agg.group == g)].mae_increase.iloc[0]
            color = BLUE if p.startswith("Sh") else ORANGE
            ax.bar(t_i + (j - 1.5) * w, max(v, 0.011) - 0.01, w, bottom=0.01, color=color, alpha=1.0 if g.startswith("Mono") else 0.45,
                   edgecolor=color, hatch=None if g.startswith("Mono") else "////", lw=0.4)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(["TS", "PS", "SS", "HV"])
    ax.set_ylabel("MAE increase on joint permutation")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Shuffled, monotone group"),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.45, hatch="////", label="Shuffled, reset group"),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Contiguous, monotone group"),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.45, hatch="////", label="Contiguous, reset group")],
              loc="upper center", ncol=2, fontsize=5.5, bbox_to_anchor=(0.56, 1.0), columnspacing=0.6, handlelength=1.4)
    ax.set_ylim(1e-2, 3e4); ax.set_yticks([1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    panel(ax, "c")

    ax = axes[1, 1]
    for p_i, p in enumerate(("Shuffled 5-fold", "Contiguous 5-fold")):
        col = BLUE if p_i == 0 else ORANGE
        s = stab[stab.protocol == p].set_index("target").loc[TARGETS]
        j = jac[jac.protocol == p].set_index("target").loc[TARGETS]
        x = np.arange(4) + (p_i - 0.5) * 0.3
        ax.scatter(x, s.mean_pairwise_rank_spearman, color=col, marker="o", s=16, label=f"{p}: rank $\\rho$")
        ax.scatter(x, j.mean_top3_jaccard, color=col, marker="^", s=16, facecolor="none", label=f"{p}: top-3 Jaccard")
        for xi, n_f in zip(x, s.finite_pair_count):
            if n_f < 10:
                ax.text(xi, s.mean_pairwise_rank_spearman.iloc[list(x).index(xi)] - 0.12, f"{n_f}/10 pairs", fontsize=5.5,
                        ha="center", color=col)
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["TS", "PS", "SS", "HV"]); ax.set_ylim(-0.3, 1.45)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Fold-to-fold agreement")
    ax.legend(loc="upper center", fontsize=5.5, ncol=2, columnspacing=0.8)
    panel(ax, "d")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig_s5_attribution")


# --------------------------------------------------------------------------- S6
def fig_conformal(data):
    c = pd.read_csv(RESULTS_DIR / "conformal_extended.csv")
    rows = pd.read_csv(RESULTS_DIR / "conformal_rows.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.8))
    models = ["Extra Trees", "Ridge", "MLP", "Order-only trend"]
    labels = ["Extra Trees", "Ridge", "MLP", "Order-only"]
    for k, (metric, ylabel) in enumerate([("coverage", "Empirical coverage"), ("norm_width", "Interval width / property range")]):
        ax = axes[0, k]
        for m_i, m in enumerate(models):
            for t_i, t in enumerate(TARGETS):
                for regime, mk in (("Random exchangeable", "o"), ("Forward shift", "D")):
                    s = c[(c.model == m) & (c.target == t) & (c.regime == regime)]
                    if not len(s):
                        continue
                    x = m_i + (t_i - 1.5) * 0.12 + (0.0 if regime.startswith("R") else 0.02)
                    v = s[metric]
                    if len(v) > 1:
                        ax.plot([x, x], [v.quantile(0.1), v.quantile(0.9)], color=TARGET_COLOR[t], lw=0.7)
                    ax.scatter(x, v.mean(), marker=mk, color=TARGET_COLOR[t], s=14,
                               facecolor=TARGET_COLOR[t] if regime.startswith("R") else "none", lw=0.8)
        if metric == "coverage":
            ax.axhline(0.9, color=BLACK, ls="--", lw=0.7); ax.set_ylim(-0.03, 1.03)
        else:
            ax.set_yscale("log")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylabel(ylabel)
        panel(ax, "ab"[k])
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", ls="none", color=GREY, label="Random exchangeable"),
                               Line2D([], [], marker="D", ls="none", mfc="none", mec=GREY, label="Forward shift")] +
                              [Line2D([], [], marker="s", ls="none", color=TARGET_COLOR[t], ms=4, label=TARGET_LABEL[t]) for t in TARGETS],
                      loc="center right", fontsize=5.5, bbox_to_anchor=(1.0, 0.45))
    ax = axes[1, 0]
    sub = rows[rows.model.isin(["Extra Trees", "Ridge", "MLP"])]
    bins = np.logspace(np.log10(max(sub.distance.min(), 1e-4)), np.log10(sub.distance.max() * 1.01), 7)
    for m, col in (("Extra Trees", GREEN), ("Ridge", SKY), ("MLP", PURPLE)):
        g = sub[sub.model == m]
        idx = np.clip(np.digitize(g.distance, bins), 1, len(bins) - 1)
        out = []
        for i in range(1, len(bins)):
            gi = g[idx == i]
            if len(gi) >= 20:
                out.append((np.sqrt(bins[i - 1] * bins[i]), gi.covered.mean(), len(gi)))
        out = np.array(out)
        ax.plot(out[:, 0], out[:, 1], marker="o", ms=3, color=col, label=m)
    ax.axhline(0.9, color=BLACK, ls="--", lw=0.7)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Nearest-neighbour distance to proper-training rows (z units)")
    ax.set_ylabel("Empirical coverage")
    ax.legend(loc="lower left", fontsize=6)
    panel(ax, "c")
    ax = axes[1, 1]
    t = TARGETS[0]
    test = data[data.row_id >= 499]
    for m, col in (("Extra Trees", GREEN), ("Ridge", SKY), ("Order-only trend", ORANGE)):
        g = rows[(rows.model == m) & (rows.target == t) & (rows.regime == "Forward shift")].sort_values("row_id")
        ax.fill_between(g.row_id, g.prediction - g.q, g.prediction + g.q, color=col, alpha=0.25, lw=0)
        ax.plot(g.row_id, g.prediction, color=col, lw=0.8, label=f"{m if m != 'Order-only trend' else 'Order-only'}")
    ax.plot(test.row_id, test[t], color=BLACK, lw=1.4, label="Target")
    trmax = data[data.row_id < 400][t].max()
    ax.axhline(trmax, color=GREY, ls=":", lw=0.7)
    ax.text(662, trmax + 2, "proper-training maximum", fontsize=6, color=GREY, ha="right")
    ax.set_xlabel(r"Row index $i$ (forward test set)"); ax.set_ylabel("Tensile strength (MPa)")
    ax.legend(loc="upper left", fontsize=6)
    panel(ax, "d")
    fig.tight_layout(h_pad=1.4, w_pad=1.5)
    save(fig, "fig_s6_conformal")


# --------------------------------------------------------------------------- Appendix
def fig_preliminary(data):
    src = pd.read_csv(DATA_DIR / "source_lookup_compositions.csv")
    pre = pd.read_csv(DATA_DIR / "preliminary_network_scaled.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    ax = axes[0]
    ax.scatter(src.tensile_strength, src.prelim_pred_50k, s=9, color=BLUE, lw=0, label=r"$5\times10^4$ iterations")
    ax.scatter(src.tensile_strength, src.prelim_pred_100k, s=9, facecolor="none", edgecolor=VERM, lw=0.6, label=r"$10^5$ iterations")
    lim = [60, 500]
    ax.plot(lim, lim, color=GREY, lw=0.7, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Recorded tensile strength (MPa)"); ax.set_ylabel("Predicted tensile strength (MPa)")
    ax.legend(loc="upper left", fontsize=6); panel(ax, "a")
    ax = axes[1]
    ax.scatter(pre.scaled_target, pre.scaled_prediction, s=9, color=GREEN, lw=0)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--"); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Scaled recorded tensile strength"); ax.set_ylabel("Scaled prediction")
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_a1_preliminary")


FIGURES = {
    "sequences": fig_sequences, "physics": fig_physics, "provenance": fig_provenance,
    "pca": fig_pca, "nn": fig_nn, "benchmark": fig_benchmark, "ann": fig_ann, "forward": fig_forward,
    "blocks": fig_blocks_distance, "controls": fig_controls, "attribution": fig_attribution,
    "conformal": fig_conformal, "preliminary": fig_preliminary,
}


def main() -> None:
    ensure_directories()
    data = load_qcar()
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n](data)


if __name__ == "__main__":
    main()
