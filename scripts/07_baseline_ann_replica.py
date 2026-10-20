#!/usr/bin/env python3
"""Evaluate the baseline ANN replica under its native and the audit protocols."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

from ann_replica import LMNetwork, MinMaxScaling, fit_predict_replica
from common import FEATURES, PREDICTION_COLUMNS, RANDOM_SEED, RESULTS_DIR, TARGETS, ensure_directories, load_qcar

N_NATIVE = 30
N_FORWARD = 30
N_CV_INITS = 10


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def native_protocol(x, data) -> pd.DataFrame:
    rows = []
    n = len(data)
    for target in TARGETS:
        y = data[target].to_numpy(float)
        scaler = MinMaxScaling().fit(x, y)  # toolbox: processing configured on all rows
        xs, ys = scaler.x(x), scaler.y(y)
        for rep in range(N_NATIVE):
            rng = np.random.default_rng(RANDOM_SEED + 1000 + rep)
            order = rng.permutation(n)
            tr, va, te = order[:467], order[467:567], order[567:]
            net = LMNetwork(seed=RANDOM_SEED + rep).fit(xs[tr], ys[tr], xs[va], ys[va])
            pred = scaler.y_inverse(net.predict(xs))
            record = {"target": target, "replicate": rep, "epochs": net.epochs_, "best_epoch": net.best_epoch_}
            for name, idx in (("train", tr), ("validation", va), ("test", te), ("all", np.arange(n))):
                record[f"R_{name}"] = pearson(y[idx], pred[idx])
                record[f"r2_{name}"] = float(r2_score(y[idx], pred[idx]))
                record[f"mae_{name}"] = float(mean_absolute_error(y[idx], pred[idx]))
            rows.append(record)
        print(f"native protocol done: {target}", flush=True)
    return pd.DataFrame(rows)


def audit_protocols(x, data) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(data)
    protocols = {
        "Repeated shuffled 5-fold": (list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED).split(x)), N_CV_INITS),
        "Contiguous 5-fold": (list(KFold(n_splits=5, shuffle=False).split(x)), N_CV_INITS),
        "Original forward holdout": ([(np.arange(499), np.arange(499, n))], N_FORWARD),
    }
    metric_rows, prediction_rows = [], []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        y_range = float(y.max() - y.min())
        for protocol, (splits, n_inits) in protocols.items():
            for init in range(n_inits):
                by_row = defaultdict(list)
                for fold, (tr, te) in enumerate(splits):
                    seed = RANDOM_SEED + 10_000 * (init + 1) + fold
                    pred, epochs = fit_predict_replica(x, y, tr, te, seed=seed)
                    for row, value in zip(te, pred):
                        by_row[int(row)].append(float(value))
                        prediction_rows.append(
                            {"protocol": protocol, "target": target, "init": init, "fold": fold,
                             "row_id": int(row), "actual": float(y[row]), "predicted": float(value)}
                        )
                rows = np.array(sorted(by_row))
                pooled = np.array([np.mean(by_row[r]) for r in rows])
                actual = y[rows]
                metric_rows.append(
                    {"protocol": protocol, "target": target, "init": init,
                     "pooled_r2": float(r2_score(actual, pooled)),
                     "pooled_mae": float(mean_absolute_error(actual, pooled)),
                     "pooled_R": pearson(actual, pooled),
                     "nmae": float(mean_absolute_error(actual, pooled) / y_range)}
                )
            print(f"{protocol} done: {target}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def main() -> None:
    ensure_directories()
    data = load_qcar()
    x = data[FEATURES].to_numpy(float)

    native = native_protocol(x, data)
    native.to_csv(RESULTS_DIR / "ann_replica_native.csv", index=False)

    metrics, predictions = audit_protocols(x, data)
    metrics.to_csv(RESULTS_DIR / "ann_replica_protocol_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "ann_replica_predictions.csv", index=False)

    # Locate the stored baseline predictions within the forward replica distribution.
    test = data[data["supplied_split"] == "testing"]
    summary = {}
    for target in TARGETS:
        stored_mae = float(mean_absolute_error(test[target], test[PREDICTION_COLUMNS[target]]))
        forward = metrics[(metrics["protocol"] == "Original forward holdout") & (metrics["target"] == target)]
        summary[target] = {
            "stored_forward_mae": stored_mae,
            "replica_forward_mae_median": float(forward["pooled_mae"].median()),
            "replica_forward_mae_min": float(forward["pooled_mae"].min()),
            "replica_forward_mae_max": float(forward["pooled_mae"].max()),
            "fraction_of_replicas_at_or_below_stored": float((forward["pooled_mae"] <= stored_mae).mean()),
            "native_test_R_median": float(native.loc[native["target"] == target, "R_test"].median()),
        }
    (RESULTS_DIR / "ann_replica_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
