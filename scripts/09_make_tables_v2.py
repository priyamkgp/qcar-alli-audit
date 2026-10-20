#!/usr/bin/env python3
"""Generate the LaTeX tables of Section 3 from the results files."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_DIR, RESULTS_DIR, TABLES_DIR, TARGETS, ensure_directories, load_qcar

SHORT = dict(zip(TARGETS, ["TS", "PS", "SS", "HV"]))
PROT = {"Repeated shuffled 5-fold": "Shuffled 5-fold", "Contiguous 5-fold": "Contiguous 5-fold",
        "Original forward holdout": "Forward holdout"}


def num(v, d=3):
    if pd.isna(v):
        return "--"
    s = f"{v:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def write(name: str, lines: list[str]) -> None:
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def table_audit(data):
    s1 = json.loads((RESULTS_DIR / "extended_summary.json").read_text())["s1"]
    audit = json.loads((RESULTS_DIR / "audit_summary.json").read_text())
    pm = pd.read_csv(RESULTS_DIR / "physics_master.csv")
    ps = pd.read_csv(RESULTS_DIR / "physics_source.csv")
    resets = pd.read_csv(RESULTS_DIR / "segment_resets.csv")
    reset_txt = "; ".join(f"{v} {', '.join(str(r) for r in g.row_id)}" for v, g in resets.groupby("variable", sort=False))
    rows = [
        ("Records", "667", "94 (27 with four properties)"),
        ("Maximum closure error (wt.\\%)", "$2.0\\times10^{-4}$", "$<10^{-13}$"),
        ("Unique composition vectors", "667/667", "93/94"),
        ("Properties strictly increasing with row index", "4/4", "--"),
        ("Minimum pairwise Spearman $\\rho$ between properties", f"{audit['minimum_pairwise_target_spearman']:.3f}", "--"),
        ("Solute resets (variable, row)", reset_txt, "--"),
        ("$r_{\\mathrm{PS}} > 1$ (proof stress above tensile strength)", f"{s1['master_rows_ps_gt_ts']}/667", f"{s1['source_records_ps_gt_ts']}/27"),
        ("$r_{\\mathrm{SS}}$, median [range]", f"{pm.r_ss.median():.2f} [{pm.r_ss.min():.2f}, {pm.r_ss.max():.2f}]",
         f"{ps.r_ss.median():.2f} [{ps.r_ss.min():.2f}, {ps.r_ss.max():.2f}]"),
        ("$\\kappa$, median [range]", f"{pm.kappa.median():.2f} [{pm.kappa.min():.2f}, {pm.kappa.max():.2f}]",
         f"{ps.kappa.median():.2f} [{ps.kappa.min():.2f}, {ps.kappa.max():.2f}]"),
        ("Source compositions matched within 0.001 / 0.05 wt.\\%", "--",
         f"{audit['source_exact_matches_max_abs_lt_0_001_wt_pct']} / {audit['source_close_matches_max_abs_lt_0_05_wt_pct']}"),
        ("Tensile strengths equal to aligned handbook entry", "--", f"{s1['linkage_positional_matches']}/94"),
        ("\\quad aligned with own designation / valid for own designation", "--",
         f"{s1['linkage_own_designation_aligned']}/94 / {s1['linkage_value_valid_for_own_designation']}/94"),
        ("\\quad integer-ksi values / distinct values / longest repeated run", "--",
         f"{s1['linkage_integer_ksi']}/94 / {s1['tensile_distinct_values']} / {s1['longest_repeated_run']}"),
        ("Four-property records equal to ANN predictions", "--", "3/26 (1100, 1420, 1421)"),
        ("Cd: maximum (wt.\\%) / non-zero entries", f"{s1['cd_master_max']:.4f} / 423", "0.25 / 1 (2020)"),
    ]
    lines = ["\\begin{tabular}{@{}p{0.47\\linewidth}p{0.24\\linewidth}p{0.22\\linewidth}@{}}", "\\toprule",
             "Diagnostic & Master table & Source compilation \\\\", "\\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_audit", lines)


def table_shift():
    sh = pd.read_csv(RESULTS_DIR / "split_shift.csv")
    names = {TARGETS[0]: "Tensile strength", TARGETS[1]: "Proof stress", TARGETS[2]: "Shear strength", TARGETS[3]: "Vickers hardness"}
    lines = ["\\begin{tabular}{@{}lrrrrrrl@{}}", "\\toprule",
             "Variable & \\multicolumn{2}{c}{Rows 0--498} & \\multicolumn{2}{c}{Rows 499--666} & $D$ & SMD & Overlap \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & min & max & min & max & & & \\\\", "\\midrule"]
    for _, r in sh.iterrows():
        d = 2 if r.variable in names else 4
        lines.append(f"{names.get(r.variable, r.variable)} & {num(r.train_min, d)} & {num(r.train_max, d)} & {num(r.test_min, d)} & "
                     f"{num(r.test_max, d)} & {r.ks_statistic:.3f} & {num(r.standardized_mean_difference, 2)} & "
                     f"{'yes' if r.range_overlap else 'no'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_shift", lines)

    s2 = json.loads((RESULTS_DIR / "extended_summary.json").read_text())["s2"]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{z-standardized} & \\multicolumn{2}{c}{Source-range scaled} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}", "Held-out configuration & Median & Mean & Median & Mean \\\\", "\\midrule"]
    for p in ("Random holdout", "Contiguous block", "Forward holdout"):
        z, r = s2["z-standardized"][p], s2["source-range"][p]
        lab = {"Random holdout": "Random holdouts (100 $\\times$ 168 rows)", "Contiguous block": "Contiguous blocks (5)",
               "Forward holdout": "Forward holdout (rows 499--666)"}[p]
        lines.append(f"{lab} & {z['median']:.4f} & {z['mean']:.4f} & {r['median']:.4f} & {r['mean']:.4f} \\\\")
    lines.append("\\midrule")
    lines.append(f"Forward/random ratio & {s2['z-standardized']['forward_to_random_median_ratio']:.0f} & "
                 f"{s2['z-standardized']['forward_to_random_mean_ratio']:.0f} & {s2['source-range']['forward_to_random_median_ratio']:.0f} & "
                 f"{s2['source-range']['forward_to_random_mean_ratio']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_nn", lines)


def table_benchmark():
    b = pd.read_csv(RESULTS_DIR / "model_benchmark_extended.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
             " & & \\multicolumn{4}{c}{Pooled $R^2$} & \\multicolumn{4}{c}{nMAE (\\%)} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}",
             "Model & Protocol & TS & PS & SS & HV & TS & PS & SS & HV \\\\", "\\midrule"]
    for m, lab in (("Ridge", "Ridge"), ("Extra Trees", "Extra Trees"), ("Histogram GB", "Hist.\\ GB"), ("MLP", "MLP"),
                   ("replica", "Baseline ANN replica$^{a}$")):
        for k, p in enumerate(PROT):
            if m == "replica":
                g = rep[rep.protocol == p].groupby("target")[["pooled_r2", "nmae"]].median()
            else:
                g = b[(b.model == m) & (b.protocol == p)].set_index("target")[["pooled_r2", "nmae"]]
            r2 = [num(g.loc[t, "pooled_r2"], 4 if p.startswith("Rep") else 2) for t in TARGETS]
            nm = [num(100 * g.loc[t, "nmae"], 2) for t in TARGETS]
            lines.append(f"{lab if k == 0 else ''} & {PROT[p]} & " + " & ".join(r2 + nm) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines[-1] = lines[-1]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_benchmark", lines)


def table_forward():
    f = pd.read_csv(RESULTS_DIR / "forward_bootstrap.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    rep = rep[rep.protocol == "Original forward holdout"]
    order = ["Dummy mean", "Ridge", "PLS (3 components)", "RBF-SVR", "Extra Trees", "Histogram GB", "MLP",
             "Baseline ANN replica", "Baseline ANN (stored)", "Order-only trend"]
    lab = {"PLS (3 components)": "PLS", "Histogram GB": "Hist.\\ GB", "Baseline ANN replica": "Baseline ANN replica$^{a}$",
           "Baseline ANN (stored)": "Baseline ANN (stored)", "Order-only trend": "Order-only trend$^{b}$"}
    lines = ["\\begin{tabular}{@{}lllll@{}}", "\\toprule",
             "Model & Tensile strength & Proof stress & Shear strength & Vickers hardness \\\\", "\\midrule"]
    for m in order:
        cells = []
        for t in TARGETS:
            if m == "Baseline ANN replica":
                g = rep[rep.target == t]
                q = g.pooled_mae.quantile([0.25, 0.5, 0.75]).to_numpy()
                cells.append(f"{q[1]:.1f} [{q[0]:.1f}, {q[2]:.1f}]; {num(g.pooled_r2.median(), 1)}")
            else:
                r = f[(f.model == m) & (f.target == t)].iloc[0]
                cells.append(f"{r.mae:.1f} [{r.mae_lo:.1f}, {r.mae_hi:.1f}]; {num(r.r2, 2 if abs(r.r2) < 10 else 1)}")
        lines.append(f"{lab.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_forward", lines)


def table_ann(data):
    native = pd.read_csv(RESULTS_DIR / "ann_replica_native.csv")
    rep = pd.read_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv")
    stored = pd.read_csv(RESULTS_DIR / "baseline_ann_stored_metrics.csv")
    reported_test_R = {TARGETS[0]: "0.99995", TARGETS[1]: "0.99992", TARGETS[2]: "0.99990", TARGETS[3]: "0.99996"}
    rows = []
    rows.append(("Toolbox-reported $R$, random test subset (100 rows)", [reported_test_R[t] for t in TARGETS]))
    rows.append(("Replica, native protocol: test $R$, median [range]",
                 [f"{native[native.target == t].R_test.median():.5f} [{native[native.target == t].R_test.min():.5f}, {native[native.target == t].R_test.max():.5f}]" for t in TARGETS]))
    for split, lab in (("training", "rows 0--498"), ("testing", "rows 499--666")):
        s = stored[stored.split == split].set_index("target")
        rows.append((f"Stored predictions, {lab}: $R$ / $R^2$", [f"{s.loc[t, 'R']:.3f} / {s.loc[t, 'r2']:.3f}" for t in TARGETS]))
        rows.append((f"Stored predictions, {lab}: MAE / relative RMSE (\\%)", [f"{s.loc[t, 'mae']:.2f} / {100 * s.loc[t, 'relative_rmse']:.1f}" for t in TARGETS]))
    s = stored[stored.split == "testing"].set_index("target")
    rows.append(("Stored test predictions above training maximum", [f"{int(round(168 * s.loc[t, 'fraction_above_train_max']))}/168" for t in TARGETS]))
    for p in ("Repeated shuffled 5-fold", "Contiguous 5-fold", "Original forward holdout"):
        g = rep[rep.protocol == p]
        rows.append((f"Replica, {PROT[p].lower()}: pooled $R^2$, median", [num(g[g.target == t].pooled_r2.median(), 4 if p.startswith("Rep") else 2) for t in TARGETS]))
    g = rep[rep.protocol == "Original forward holdout"]
    rows.append(("Replica, forward holdout: MAE, median [range]",
                 [f"{g[g.target == t].pooled_mae.median():.1f} [{g[g.target == t].pooled_mae.min():.1f}, {g[g.target == t].pooled_mae.max():.1f}]" for t in TARGETS]))
    rows.append(("Replicas with forward MAE $\\le$ stored MAE",
                 [f"{int((g[g.target == t].pooled_mae <= s.loc[t, 'mae']).sum())}/30" for t in TARGETS]))
    lines = ["\\begin{tabular}{@{}p{0.34\\linewidth}llll@{}}", "\\toprule",
             "Evidence & Tensile strength & Proof stress & Shear strength & Vickers hardness \\\\", "\\midrule"]
    lines += [f"{a} & " + " & ".join(c) + " \\\\" for a, c in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_ann", lines)


def table_order_controls():
    o = pd.read_csv(RESULTS_DIR / "order_trend_metrics.csv").set_index("target")
    c = pd.read_csv(RESULTS_DIR / "negative_controls.csv")
    lines = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "Diagnostic & Quantity & TS & PS & SS & HV \\\\", "\\midrule"]
    lines.append("Order-only trend & selected window $w$ & " + " & ".join(str(int(o.loc[t, "selected_window"])) for t in TARGETS) + " \\\\")
    lines.append(" & inner-validation MAE & " + " & ".join(f"{o.loc[t, 'inner_validation_mae']:.2f}" for t in TARGETS) + " \\\\")
    lines.append(" & forward $R^2$ & " + " & ".join(f"{o.loc[t, 'test_r2']:.3f}" for t in TARGETS) + " \\\\")
    lines.append(" & forward MAE & " + " & ".join(f"{o.loc[t, 'test_mae']:.2f}" for t in TARGETS) + " \\\\")
    lines.append("\\midrule")
    for ctrl in ("Master table", "Comonotone surrogate", "Response randomization"):
        for m in ("Ridge", "Extra Trees"):
            g = c[(c.control == ctrl) & (c.model == m)]
            if ctrl == "Master table":
                cells = [f"{g[g.target == t].r2.iloc[0]:.4f}" for t in TARGETS]
            else:
                cells = [f"{num(g[g.target == t].r2.median(), 4 if ctrl.startswith('Com') else 3)}" for t in TARGETS]
            lab = {"Master table": "Master table", "Comonotone surrogate": "Comonotone surrogates (20)",
                   "Response randomization": "Response randomization (20)"}[ctrl]
            lines.append(f"{lab if m == 'Ridge' else ''} & {m}: {'' if ctrl == 'Master table' else 'median '}$R^2_{{\\mathrm{{pool}}}}$ & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_controls", lines)


def table_attribution():
    st = pd.read_csv(RESULTS_DIR / "importance_stability.csv")
    j = pd.read_csv(RESULTS_DIR / "importance_jaccard.csv")
    g = pd.read_csv(RESULTS_DIR / "grouped_importance.csv").groupby(["target", "protocol", "group"]).mae_increase.mean().unstack()
    lines = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
             "Property & Protocol & Rank $\\rho$ (pairs) & Top-3 $J$ & Total imp. & Cont./shuf. & Monotone share \\\\", "\\midrule"]
    for t in TARGETS:
        for p in ("Shuffled 5-fold", "Contiguous 5-fold"):
            s = st[(st.target == t) & (st.protocol == p)].iloc[0]
            jj = j[(j.target == t) & (j.protocol == p)].mean_top3_jaccard.iloc[0]
            gm, gr = g.loc[(t, p), "Monotone group"], g.loc[(t, p), "Reset group"]
            lines.append(f"{SHORT[t] if p.startswith('Sh') else ''} & {p} & {num(s.mean_pairwise_rank_spearman)} ({int(s.finite_pair_count)}) & "
                         f"{jj:.2f} & {s.total_mean_importance:.2f} & "
                         f"{(100 * s.contiguous_to_shuffled_total_importance_ratio):.1f}\\% & {100 * gm / (gm + gr):.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_attribution", lines)


def table_conformal():
    c = pd.read_csv(RESULTS_DIR / "conformal_extended.csv")
    lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
             "Regime & Base model & \\multicolumn{4}{c}{Coverage} & Width/range & Reachable \\\\",
             "\\cmidrule(lr){3-6}", " & & TS & PS & SS & HV & (mean) & (ET) \\\\", "\\midrule"]
    for regime in ("Random exchangeable", "Forward shift"):
        for m in ("Extra Trees", "Ridge", "MLP", "Order-only trend"):
            g = c[(c.regime == regime) & (c.model == m)]
            if not len(g):
                continue
            cov = [f"{g[g.target == t].coverage.mean():.3f}" for t in TARGETS]
            reach = f"{g.reachable_fraction.mean():.3f}" if m == "Extra Trees" else "--"
            lines.append(f"{regime if m == 'Extra Trees' else ''} & {m} & " + " & ".join(cov) +
                         f" & {100 * g.norm_width.mean():.1f}\\% & {reach} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_res_conformal", lines)


def main() -> None:
    ensure_directories()
    data = load_qcar()
    table_audit(data)
    table_shift()
    table_benchmark()
    table_forward()
    table_ann(data)
    table_order_controls()
    table_attribution()
    table_conformal()


if __name__ == "__main__":
    main()
