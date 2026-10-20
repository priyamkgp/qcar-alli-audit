#!/usr/bin/env python3
"""Extended audit analyses (Section 2 v2): S1 physics and provenance, S2 scale
sensitivity and contiguous-block distances, S3 metrics with block bootstrap and
the error--distance relation, S4 negative controls, S5 grouped importance and
tie-robust stability, S6 additional conformal base models, reachability and
distance-resolved coverage."""
from __future__ import annotations

import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, pairwise_distances, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (DATA_DIR, FEATURES, PREDICTION_COLUMNS, RANDOM_SEED, RESULTS_DIR, TARGETS,
                    ensure_directories, load_qcar, regression_models)

warnings.filterwarnings("ignore")
SOLUTES = [f for f in FEATURES if f != "Al"]


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def nn_distance(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return pairwise_distances(x_test, x_train).min(axis=1)


# ---------------------------------------------------------------- S1 ------
def s1_integrity(data, source, four, prediction_table, handbook, out):
    ts, ps = data[TARGETS[0]], data[TARGETS[1]]
    ss, hv = data[TARGETS[2]], data[TARGETS[3]]
    physics = pd.DataFrame({
        "row_id": data["row_id"], "split": data["supplied_split"],
        "r_ps": ps / ts, "r_ss": ss / ts, "kappa": 3 * ts / (9.80665 * hv)})
    physics.to_csv(RESULTS_DIR / "physics_master.csv", index=False)

    # Four-property source table, as recorded and with recorded originals
    # substituted where the stored values are ANN predictions.
    merged = four.merge(prediction_table, on="alloy", how="left")
    props = ["tensile_strength", "proof_stress", "shear_strength", "hardness"]
    tol = 1e-3
    merged["equals_prediction"] = np.all(
        [np.isclose(merged[p], merged[f"pred_{p}"], atol=tol) for p in props], axis=0)
    merged["equals_original"] = np.all(
        [np.isclose(merged[p], merged[f"orig_{p}"], atol=tol) for p in props], axis=0)
    merged.to_csv(RESULTS_DIR / "four_property_contamination.csv", index=False)
    corrected = merged.copy()
    for p in props:
        corrected[p] = np.where(merged["equals_prediction"], merged[f"orig_{p}"], merged[p])
    src_phys = pd.DataFrame({
        "alloy": corrected["alloy"],
        "r_ps": corrected["proof_stress"] / corrected["tensile_strength"],
        "r_ss": corrected["shear_strength"] / corrected["tensile_strength"],
        "kappa": 3 * corrected["tensile_strength"] / (9.80665 * corrected["hardness"])})
    src_phys.to_csv(RESULTS_DIR / "physics_source.csv", index=False)

    # Segment resets.
    resets = []
    for f in SOLUTES:
        d = np.diff(data[f].to_numpy(float))
        for i in np.flatnonzero(d < 0):
            resets.append({"variable": f, "row_id": int(i + 1),
                           "value_before": float(data[f].iloc[i]), "value_after": float(data[f].iloc[i + 1])})
    pd.DataFrame(resets).to_csv(RESULTS_DIR / "segment_resets.csv", index=False)

    # Record linkage of source tensile strength to the handbook temper sheet.
    hb = handbook.sort_values("position").reset_index(drop=True)
    link = source[["record", "alloy", "tensile_strength"]].copy()
    link["aligned_position"] = link["record"] % len(hb)
    link["handbook_alloy"] = hb.loc[link["aligned_position"], "alloy"].to_numpy()
    link["handbook_temper"] = hb.loc[link["aligned_position"], "temper"].to_numpy()
    link["handbook_mpa"] = hb.loc[link["aligned_position"], "mpa"].to_numpy()
    link["positional_match"] = np.isclose(link["tensile_strength"], link["handbook_mpa"])
    link["own_designation_aligned"] = link["alloy"] == link["handbook_alloy"]
    valid = {a: set(g["mpa"]) for a, g in hb.groupby("alloy")}
    link["value_valid_for_own_designation"] = [
        (a in valid) and (v in valid[a]) for a, v in zip(link["alloy"], link["tensile_strength"])]
    link["integer_ksi"] = np.isclose(6.894757 * np.round(link["tensile_strength"] / 6.894757),
                                     link["tensile_strength"], atol=0.05)
    link.to_csv(RESULTS_DIR / "tensile_record_linkage.csv", index=False)

    v = source["tensile_strength"].to_numpy(float)
    runs = []
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            L = 0
            while b + L < len(v) and v[a + L] == v[b + L]:
                L += 1
            if L >= 6 and (a == 0 or v[a - 1] != v[b - 1]):
                runs.append({"start_a": a, "start_b": b, "length": L,
                             "alloy_a": int(source["alloy"].iloc[a]), "alloy_b": int(source["alloy"].iloc[b])})
    pd.DataFrame(runs).sort_values("length", ascending=False).to_csv(RESULTS_DIR / "tensile_repeated_runs.csv", index=False)

    # Support and range.
    tr, te = data[data.supplied_split == "training"], data[data.supplied_split == "testing"]
    support = []
    for f in FEATURES:
        support.append({"element": f, "source_min": source[f].min(), "source_max": source[f].max(),
                        "source_nonzero": int((source[f] > 0).sum()),
                        "train_min": tr[f].min(), "train_max": tr[f].max(),
                        "test_min": te[f].min(), "test_max": te[f].max(),
                        "master_nonzero": int((data[f] > 0).sum())})
    pd.DataFrame(support).to_csv(RESULTS_DIR / "support_ranges.csv", index=False)

    summary = {
        "master_rows_ps_gt_ts": int((physics.r_ps > 1).sum()),
        "master_first_row_ps_gt_ts": int(physics.row_id[physics.r_ps > 1].min()),
        "master_r_ps_range": [float(physics.r_ps.min()), float(physics.r_ps.max())],
        "master_r_ss_median": float(physics.r_ss.median()),
        "master_r_ss_range": [float(physics.r_ss.min()), float(physics.r_ss.max())],
        "master_kappa_range": [float(physics.kappa.min()), float(physics.kappa.max())],
        "master_kappa_median": float(physics.kappa.median()),
        "source_records_ps_gt_ts": int((src_phys.r_ps > 1).sum()),
        "source_four_property_records": int(len(src_phys)),
        "source_r_ss_median": float(src_phys.r_ss.median()),
        "source_kappa_range": [float(src_phys.kappa.min()), float(src_phys.kappa.max())],
        "contaminated_records": merged.loc[merged.equals_prediction, "alloy"].astype(int).tolist(),
        "records_equal_original": int(merged.equals_original.sum()),
        "records_in_prediction_table": int(merged["pred_tensile_strength"].notna().sum()),
        "linkage_positional_matches": int(link.positional_match.sum()),
        "linkage_records": int(len(link)),
        "linkage_own_designation_aligned": int(link.own_designation_aligned.sum()),
        "linkage_value_valid_for_own_designation": int(link.value_valid_for_own_designation.sum()),
        "linkage_integer_ksi": int(link.integer_ksi.sum()),
        "tensile_distinct_values": int(source["tensile_strength"].nunique()),
        "longest_repeated_run": int(max(r["length"] for r in runs)) if runs else 0,
        "resets": resets,
        "cd_master_max": float(data["Cd"].max()),
        "cd_source_nonzero": source.loc[source.Cd > 0, ["alloy", "Cd"]].values.tolist(),
    }
    out["s1"] = summary


# ---------------------------------------------------------------- S2 ------
def s2_shift(data, source, out):
    x = data[FEATURES].to_numpy(float)
    n = len(data)
    ranges = (source[FEATURES].max() - source[FEATURES].min()).to_numpy(float)
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    fwd_tr, fwd_te = np.arange(499), np.arange(499, n)
    random_sets = []
    for rep in range(100):
        te = rng.choice(n, 168, replace=False)
        random_sets.append((np.setdiff1d(np.arange(n), te), te))
    blocks = list(KFold(n_splits=5, shuffle=False).split(x))
    for scaling in ("z-standardized", "source-range"):
        def transform(tr, te):
            if scaling == "z-standardized":
                sc = StandardScaler().fit(x[tr]); return sc.transform(x[tr]), sc.transform(x[te])
            return x[tr] / ranges, x[te] / ranges
        a, b = transform(fwd_tr, fwd_te)
        for d in nn_distance(a, b):
            rows.append({"scaling": scaling, "protocol": "Forward holdout", "repeat": 0, "distance": d})
        for rep, (tr, te) in enumerate(random_sets):
            a, b = transform(tr, te)
            for d in nn_distance(a, b):
                rows.append({"scaling": scaling, "protocol": "Random holdout", "repeat": rep, "distance": d})
        for k, (tr, te) in enumerate(blocks):
            a, b = transform(tr, te)
            for d in nn_distance(a, b):
                rows.append({"scaling": scaling, "protocol": "Contiguous block", "repeat": k, "distance": d})
    nn = pd.DataFrame(rows)
    nn.to_csv(RESULTS_DIR / "nn_distances_extended.csv", index=False)
    summ = nn.groupby(["scaling", "protocol"]).distance.agg(["mean", "median"]).reset_index()
    summ.to_csv(RESULTS_DIR / "nn_distance_summary.csv", index=False)
    block_med = nn[nn.protocol == "Contiguous block"].groupby(["scaling", "repeat"]).distance.median().reset_index()
    block_med.to_csv(RESULTS_DIR / "nn_block_medians.csv", index=False)

    z = StandardScaler().fit_transform(x)
    pz = PCA().fit(z)
    xr = (x - x.mean(axis=0)) / ranges
    pr = PCA().fit(xr)
    sz, sr = pz.transform(z), pr.transform(xr)
    pd.DataFrame({"row_id": data.row_id, "split": data.supplied_split,
                  "z_PC1": sz[:, 0], "z_PC2": sz[:, 1], "r_PC1": sr[:, 0], "r_PC2": sr[:, 1]}).to_csv(
        RESULTS_DIR / "pca_scores_both.csv", index=False)
    pd.DataFrame({"component": np.arange(1, 10), "z_ratio": pz.explained_variance_ratio_,
                  "range_ratio": pr.explained_variance_ratio_}).to_csv(RESULTS_DIR / "pca_explained_both.csv", index=False)
    pd.DataFrame(pr.components_[:2], columns=FEATURES, index=["PC1", "PC2"]).to_csv(RESULTS_DIR / "pca_range_loadings.csv")
    pd.DataFrame(pz.components_[:2], columns=FEATURES, index=["PC1", "PC2"]).to_csv(RESULTS_DIR / "pca_z_loadings.csv")
    s = {}
    for scaling in ("z-standardized", "source-range"):
        g = summ[summ.scaling == scaling].set_index("protocol")
        s[scaling] = {p: {"mean": float(g.loc[p, "mean"]), "median": float(g.loc[p, "median"])} for p in g.index}
        s[scaling]["forward_to_random_median_ratio"] = float(g.loc["Forward holdout", "median"] / g.loc["Random holdout", "median"])
        s[scaling]["forward_to_random_mean_ratio"] = float(g.loc["Forward holdout", "mean"] / g.loc["Random holdout", "mean"])
        bm = block_med[block_med.scaling == scaling].distance.to_numpy()
        s[scaling]["block_medians"] = bm.tolist()
    s["pc1_row_spearman_z"] = float(spearmanr(sz[:, 0], data.row_id).statistic)
    s["pc1_row_spearman_range"] = float(spearmanr(sr[:, 0], data.row_id).statistic)
    s["range_pca_first_two"] = float(pr.explained_variance_ratio_[:2].sum())
    s["range_pca_first"] = float(pr.explained_variance_ratio_[0])
    out["s2"] = s
    return random_sets


# ---------------------------------------------------------------- S3 ------
def block_bootstrap(y, p, n_boot=2000, block=10, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yy, pp = y[idx], p[idx]
        stats.append((r2_score(yy, pp), mean_absolute_error(yy, pp), pearson(yy, pp)))
    stats = np.array(stats)
    return np.percentile(stats, [2.5, 97.5], axis=0)


def s3_metrics(data, out):
    preds = pd.read_csv(RESULTS_DIR / "model_predictions.csv")
    bench = pd.read_csv(RESULTS_DIR / "model_benchmark_summary.csv")
    order = pd.read_csv(RESULTS_DIR / "order_trend_predictions.csv")
    ranges = {t: float(data[t].max() - data[t].min()) for t in TARGETS}
    # Pooled R and nMAE for every model/protocol.
    rows = []
    for (protocol, target, model), g in preds.groupby(["protocol", "target", "model"]):
        pooled = g.groupby("row_id").agg(actual=("actual", "first"), predicted=("predicted", "mean"))
        rows.append({"protocol": protocol, "target": target, "model": model,
                     "pooled_R": pearson(pooled.actual, pooled.predicted) if pooled.predicted.std() > 0 else np.nan})
    rtab = pd.DataFrame(rows)
    bench = bench.merge(rtab, on=["protocol", "target", "model"], how="left")
    bench["nmae"] = bench.apply(lambda r: r["pooled_mae"] / ranges[r["target"]], axis=1)
    bench.to_csv(RESULTS_DIR / "model_benchmark_extended.csv", index=False)

    # Forward holdout with block-bootstrap intervals.
    test = data[data.supplied_split == "testing"].sort_values("row_id")
    fwd_rows = []
    fwd = preds[preds.protocol == "Original forward holdout"]
    candidates = [(m, fwd[fwd.model == m]) for m in fwd.model.unique()]
    for target in TARGETS:
        y = test[target].to_numpy(float)
        sets = []
        for m, g in candidates:
            gg = g[g.target == target].sort_values("row_id")
            sets.append((m, gg.predicted.to_numpy(float)))
        sets.append(("Baseline ANN (stored)", test[PREDICTION_COLUMNS[target]].to_numpy(float)))
        oo = order[order.target == target].sort_values("row_id")
        sets.append(("Order-only trend", oo.predicted.to_numpy(float)))
        for m, p in sets:
            ci = block_bootstrap(y, p)
            fwd_rows.append({"target": target, "model": m, "r2": float(r2_score(y, p)),
                             "mae": float(mean_absolute_error(y, p)), "R": pearson(y, p) if np.std(p) > 0 else np.nan,
                             "nmae": float(mean_absolute_error(y, p) / ranges[target]),
                             "r2_lo": ci[0, 0], "r2_hi": ci[1, 0], "mae_lo": ci[0, 1], "mae_hi": ci[1, 1],
                             "R_lo": ci[0, 2], "R_hi": ci[1, 2]})
    pd.DataFrame(fwd_rows).to_csv(RESULTS_DIR / "forward_bootstrap.csv", index=False)

    # Stored baseline ANN evidence.
    ann_rows = []
    tr = data[data.supplied_split == "training"]
    for split, sub in (("training", tr), ("testing", test)):
        for t in TARGETS:
            y, p = sub[t].to_numpy(float), sub[PREDICTION_COLUMNS[t]].to_numpy(float)
            rel = np.sqrt(np.mean(((y - p) / y) ** 2))
            ann_rows.append({"split": split, "target": t, "R": pearson(y, p), "r2": float(r2_score(y, p)),
                             "mae": float(mean_absolute_error(y, p)), "relative_rmse": float(rel),
                             "nmae": float(mean_absolute_error(y, p) / ranges[t]),
                             "fraction_above_train_max": float(np.mean(p > tr[t].max())) if split == "testing" else np.nan})
    pd.DataFrame(ann_rows).to_csv(RESULTS_DIR / "baseline_ann_stored_metrics.csv", index=False)


def s3_preliminary(source, prelim, out):
    y = source["tensile_strength"].to_numpy(float)
    res = {}
    for col in ("prelim_pred_50k", "prelim_pred_100k"):
        p = source[col].to_numpy(float)
        res[col] = {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
                    "spearman": float(spearmanr(y, p).statistic), "pred_min": float(p.min()), "pred_max": float(p.max()),
                    "pred_sd": float(p.std(ddof=1)), "obs_sd": float(y.std(ddof=1))}
    a, b = prelim.scaled_target.to_numpy(float), prelim.scaled_prediction.to_numpy(float)
    res["sheet3"] = {"n": int(len(a)), "R": pearson(a, b), "r2": float(r2_score(a, b)),
                     "pred_sd": float(b.std(ddof=1)), "target_sd": float(a.std(ddof=1))}
    out["preliminary"] = res


def s3_error_distance(data):
    """Pair every held-out prediction with its nearest-neighbour distance."""
    x = data[FEATURES].to_numpy(float)
    preds = pd.read_csv(RESULTS_DIR / "model_predictions.csv")
    preds = preds[preds.model.isin(["Extra Trees", "Ridge", "MLP"])]
    rep = RESULTS_DIR / "ann_replica_predictions.csv"
    frames = [preds]
    if rep.exists():
        r = pd.read_csv(rep)
        r = r[r.init == 0].drop(columns="init")
        r["model"] = "Baseline ANN replica"
        frames.append(r)
    allp = pd.concat(frames, ignore_index=True)
    n = len(data)
    rows = []
    for (protocol, fold), g in allp.groupby(["protocol", "fold"]):
        te = np.sort(g.row_id.unique())
        tr = np.setdiff1d(np.arange(n), te)
        sc = StandardScaler().fit(x[tr])
        d = nn_distance(sc.transform(x[tr]), sc.transform(x[te]))
        dist = dict(zip(te, d))
        gg = g.copy()
        gg["distance"] = gg.row_id.map(dist)
        rows.append(gg)
    ed = pd.concat(rows, ignore_index=True)
    ed["abs_error"] = (ed.actual - ed.predicted).abs()
    ranges = {t: float(data[t].max() - data[t].min()) for t in TARGETS}
    ed["norm_error"] = ed.apply(lambda r: r.abs_error / ranges[r.target], axis=1)
    ed[["protocol", "target", "model", "fold", "row_id", "distance", "abs_error", "norm_error"]].to_csv(
        RESULTS_DIR / "error_distance.csv", index=False)
    corr = []
    for (model, target), g in ed.groupby(["model", "target"]):
        corr.append({"model": model, "target": target, "spearman_all_protocols": float(spearmanr(g.distance, g.abs_error).statistic)})
    pd.DataFrame(corr).to_csv(RESULTS_DIR / "error_distance_correlation.csv", index=False)


# ---------------------------------------------------------------- S4 ------
def pooled_cv(model, x, y, splits):
    pred = np.zeros_like(y)
    for tr, te in splits:
        est = clone(model).fit(x[tr], y[tr])
        pred[te] = np.ravel(est.predict(x[te]))
    return float(r2_score(y, pred)), float(mean_absolute_error(y, pred))


def s4_controls(data, out):
    x = data[FEATURES].to_numpy(float)
    n = len(data)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(x))
    models = {k: v for k, v in regression_models(RANDOM_SEED).items() if k in ("Ridge", "Extra Trees")}
    rng = np.random.default_rng(RANDOM_SEED + 404)
    rows = []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        for name, model in models.items():
            r2, mae = pooled_cv(model, x, y, splits)
            rows.append({"control": "Master table", "replicate": 0, "model": name, "target": target, "r2": r2, "mae": mae})
        for b in range(20):
            yp = rng.permutation(y)
            for name, model in models.items():
                r2, mae = pooled_cv(model, x, yp, splits)
                rows.append({"control": "Response randomization", "replicate": b, "model": name, "target": target, "r2": r2, "mae": mae})
        print(f"y-scrambling done: {target}", flush=True)
    for b in range(20):
        xs = np.column_stack([np.sort(rng.choice(data[f].to_numpy(float), n, replace=True)) for f in SOLUTES])
        al = 100.0 - xs.sum(axis=1)
        xsur = np.column_stack([al, xs])  # column order follows FEATURES: Al first
        for target in TARGETS:
            ys = np.sort(rng.choice(data[target].to_numpy(float), n, replace=True))
            for name, model in models.items():
                r2, mae = pooled_cv(model, xsur, ys, splits)
                rows.append({"control": "Comonotone surrogate", "replicate": b, "model": name, "target": target, "r2": r2, "mae": mae})
        print(f"surrogate {b} done", flush=True)
    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(RESULTS_DIR / "negative_controls.csv", index=False)
    out["s4"] = ctrl.groupby(["control", "model", "target"]).r2.agg(["median", "min", "max"]).reset_index().to_dict("records")


# ---------------------------------------------------------------- S5 ------
def s5_attribution(data, out):
    x = data[FEATURES].to_numpy(float)
    order = pd.read_csv(RESULTS_DIR / "order_correlations.csv").set_index("variable")
    mono = [f for f in FEATURES if abs(order.loc[f, "spearman_with_row_id"]) >= 0.98]
    reset = [f for f in FEATURES if f not in mono]
    groups = {"Monotone group": mono, "Reset group": reset}
    protocols = {"Shuffled 5-fold": KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
                 "Contiguous 5-fold": KFold(n_splits=5, shuffle=False)}
    rows = []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        for protocol, cv in protocols.items():
            for fold, (tr, te) in enumerate(cv.split(x)):
                model = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=2, max_features=1.0,
                                            random_state=RANDOM_SEED + fold, n_jobs=1).fit(x[tr], y[tr])
                base = mean_absolute_error(y[te], model.predict(x[te]))
                rng = np.random.default_rng(RANDOM_SEED + 200 + fold)
                for gname, cols in groups.items():
                    idx = [FEATURES.index(c) for c in cols]
                    inc = []
                    for _ in range(5):
                        xp = x[te].copy()
                        perm = rng.permutation(len(te))
                        xp[:, idx] = xp[perm][:, idx]
                        inc.append(mean_absolute_error(y[te], model.predict(xp)) - base)
                    rows.append({"target": target, "protocol": protocol, "fold": fold, "group": gname,
                                 "mae_increase": float(np.mean(inc)), "baseline_mae": float(base)})
    gi = pd.DataFrame(rows)
    gi.to_csv(RESULTS_DIR / "grouped_importance.csv", index=False)

    folds = pd.read_csv(RESULTS_DIR / "permutation_importance_folds.csv")
    jac = []
    for (target, protocol), g in folds.groupby(["target", "protocol"]):
        tops = []
        for fold, gg in g.groupby("fold"):
            s = gg.set_index("feature")["mae_increase_when_permuted"]
            third = s.sort_values(ascending=False).iloc[2]
            tops.append(set(s[s >= third - 1e-12].index))
        js = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
        jac.append({"target": target, "protocol": protocol, "mean_top3_jaccard": float(np.mean(js))})
    pd.DataFrame(jac).to_csv(RESULTS_DIR / "importance_jaccard.csv", index=False)
    data[FEATURES].corr(method="spearman").to_csv(RESULTS_DIR / "input_spearman.csv")
    out["s5"] = {"monotone_group": mono, "reset_group": reset}


# ---------------------------------------------------------------- S6 ------
def conformal_q(res, alpha=0.10):
    n = len(res)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(res, level, method="higher"))


def order_only_fit_predict(y, fit_end, predict_index, inner_val=(300, 400)):
    idx = np.arange(len(y)).reshape(-1, 1).astype(float)
    windows = [25, 50, 75, 100, 150, 200, 250, 300]
    v0, v1 = inner_val
    best = None
    for w in windows:
        fit = np.arange(max(0, v0 - w), v0)
        m = LinearRegression().fit(idx[fit], y[fit])
        mae = mean_absolute_error(y[v0:v1], m.predict(idx[v0:v1]))
        if best is None or mae < best[0]:
            best = (mae, w)
    w = best[1]
    fit = np.arange(fit_end - w, fit_end)
    m = LinearRegression().fit(idx[fit], y[fit])
    return m.predict(idx[predict_index]), w


def s6_conformal(data, out):
    x = data[FEATURES].to_numpy(float)
    n = len(data)
    base_models = regression_models(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)  # reproduces the split sequence of script 04
    row_records, summary = [], []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        rngv = float(y.max() - y.min())
        for repeat in range(20):
            order = rng.permutation(n)
            tr, ca, te = order[:466], order[466:566], order[566:]
            sc = StandardScaler().fit(x[tr])
            dist = nn_distance(sc.transform(x[tr]), sc.transform(x[te]))
            models = {
                "Extra Trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, max_features=1.0,
                                                   random_state=RANDOM_SEED + repeat, n_jobs=1),
                "Ridge": base_models["Ridge"], "MLP": base_models["MLP"]}
            for name, model in models.items():
                est = clone(model).fit(x[tr], y[tr])
                q = conformal_q(np.abs(y[ca] - np.ravel(est.predict(x[ca]))))
                p = np.ravel(est.predict(x[te]))
                cov = (y[te] >= p - q) & (y[te] <= p + q)
                reach = (y[te] >= y[tr].min() - q) & (y[te] <= y[tr].max() + q)
                summary.append({"target": target, "regime": "Random exchangeable", "model": name, "repeat": repeat,
                                "coverage": float(cov.mean()), "width": 2 * q, "norm_width": 2 * q / rngv,
                                "mae": float(mean_absolute_error(y[te], p)), "reachable_fraction": float(reach.mean())})
                for r, d, c in zip(te, dist, cov):
                    row_records.append({"target": target, "regime": "Random exchangeable", "model": name,
                                        "repeat": repeat, "row_id": int(r), "distance": float(d), "covered": bool(c)})
        # forward regime
        tr, ca, te = np.arange(400), np.arange(400, 499), np.arange(499, n)
        sc = StandardScaler().fit(x[tr])
        dist = nn_distance(sc.transform(x[tr]), sc.transform(x[te]))
        models = {
            "Extra Trees": ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=1.0,
                                               random_state=RANDOM_SEED, n_jobs=1),
            "Ridge": base_models["Ridge"], "MLP": base_models["MLP"]}
        fwd_pred = {}
        for name, model in models.items():
            est = clone(model).fit(x[tr], y[tr])
            fwd_pred[name] = (np.ravel(est.predict(x[ca])), np.ravel(est.predict(x[te])))
        pc, _ = order_only_fit_predict(y, 400, ca)
        pt, w = order_only_fit_predict(y, 400, te)
        fwd_pred["Order-only trend"] = (pc, pt)
        for name, (pc, pt) in fwd_pred.items():
            q = conformal_q(np.abs(y[ca] - pc))
            cov = (y[te] >= pt - q) & (y[te] <= pt + q)
            reach = (y[te] >= y[tr].min() - q) & (y[te] <= y[tr].max() + q)
            summary.append({"target": target, "regime": "Forward shift", "model": name, "repeat": 0,
                            "coverage": float(cov.mean()), "width": 2 * q, "norm_width": 2 * q / rngv,
                            "mae": float(mean_absolute_error(y[te], pt)), "reachable_fraction": float(reach.mean()),
                            "train_max": float(y[tr].max()), "q": q})
            for r, d, c, pp in zip(te, dist, cov, pt):
                row_records.append({"target": target, "regime": "Forward shift", "model": name, "repeat": 0,
                                    "row_id": int(r), "distance": float(d), "covered": bool(c),
                                    "prediction": float(pp), "q": q})
        print(f"conformal done: {target}", flush=True)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "conformal_extended.csv", index=False)
    pd.DataFrame(row_records).to_csv(RESULTS_DIR / "conformal_rows.csv", index=False)


def main() -> None:
    import sys
    ensure_directories()
    data = load_qcar()
    source = pd.read_csv(DATA_DIR / "source_lookup_compositions.csv")
    four = pd.read_csv(DATA_DIR / "source_lookup_four_properties.csv")
    ptab = pd.read_csv(DATA_DIR / "source_prediction_table.csv")
    hb = pd.read_csv(DATA_DIR / "handbook_temper_sheet.csv")
    prelim = pd.read_csv(DATA_DIR / "preliminary_network_scaled.csv")
    stages = sys.argv[1:] or ["s1", "s2", "s3", "s4", "s5", "s6", "ed"]
    path = RESULTS_DIR / "extended_summary.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if "s1" in stages: s1_integrity(data, source, four, ptab, hb, out)
    if "s2" in stages: s2_shift(data, source, out)
    if "s3" in stages:
        s3_metrics(data, out); s3_preliminary(source, prelim, out)
    if "s4" in stages: s4_controls(data, out)
    if "s5" in stages: s5_attribution(data, out)
    if "s6" in stages: s6_conformal(data, out)
    if "ed" in stages: s3_error_distance(data)
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("extended audit stages complete:", stages)


if __name__ == "__main__":
    main()
