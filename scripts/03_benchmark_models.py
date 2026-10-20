#!/usr/bin/env python3
"""Benchmark composition-property models under leakage-sensitive validation."""
from __future__ import annotations

import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

from common import (
    FEATURES,
    PREDICTION_COLUMNS,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGETS,
    ensure_directories,
    load_qcar,
    regression_models,
    rmse,
)

warnings.filterwarnings("ignore")


def evaluate_protocol(
    protocol_name: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    x: np.ndarray,
    data: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict]]:
    models = regression_models(RANDOM_SEED)
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for target in TARGETS:
        y = data[target].to_numpy(float)
        for model_name, model in models.items():
            predictions_by_row: dict[int, list[float]] = defaultdict(list)
            target_fold_rows = []
            for fold, (train_index, test_index) in enumerate(splits):
                estimator = clone(model)
                estimator.fit(x[train_index], y[train_index])
                predicted = np.ravel(estimator.predict(x[test_index]))
                fold_result = {
                    "protocol": protocol_name,
                    "target": target,
                    "model": model_name,
                    "fold": fold,
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "r2": float(r2_score(y[test_index], predicted)),
                    "mae": float(mean_absolute_error(y[test_index], predicted)),
                    "rmse": rmse(y[test_index], predicted),
                }
                fold_rows.append(fold_result)
                target_fold_rows.append(fold_result)
                for row_id, actual, value in zip(test_index, y[test_index], predicted):
                    predictions_by_row[int(row_id)].append(float(value))
                    prediction_rows.append(
                        {
                            "protocol": protocol_name,
                            "target": target,
                            "model": model_name,
                            "fold": fold,
                            "row_id": int(row_id),
                            "actual": float(actual),
                            "predicted": float(value),
                        }
                    )

            evaluated_rows = np.array(sorted(predictions_by_row), dtype=int)
            pooled_prediction = np.array(
                [np.mean(predictions_by_row[row_id]) for row_id in evaluated_rows]
            )
            pooled_actual = y[evaluated_rows]
            target_fold_frame = pd.DataFrame(target_fold_rows)
            summary_rows.append(
                {
                    "protocol": protocol_name,
                    "target": target,
                    "model": model_name,
                    "n_evaluated": int(len(evaluated_rows)),
                    "pooled_r2": float(r2_score(pooled_actual, pooled_prediction)),
                    "pooled_mae": float(
                        mean_absolute_error(pooled_actual, pooled_prediction)
                    ),
                    "pooled_rmse": rmse(pooled_actual, pooled_prediction),
                    "median_fold_r2": float(target_fold_frame["r2"].median()),
                    "mean_fold_r2": float(target_fold_frame["r2"].mean()),
                    "sd_fold_r2": float(target_fold_frame["r2"].std(ddof=0)),
                    "median_fold_mae": float(target_fold_frame["mae"].median()),
                    "mean_fold_mae": float(target_fold_frame["mae"].mean()),
                }
            )
    return summary_rows, fold_rows, prediction_rows


def supplied_ann_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("training", "testing"):
        subset = data[data["supplied_split"] == split]
        for target in TARGETS:
            prediction_column = PREDICTION_COLUMNS[target]
            actual = subset[target].to_numpy(float)
            predicted = subset[prediction_column].to_numpy(float)
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "model": "Supplied MATLAB ANN",
                    "n": int(len(subset)),
                    "r2": float(r2_score(actual, predicted)),
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "rmse": rmse(actual, predicted),
                }
            )
    return pd.DataFrame(rows)


def order_trend_diagnostic(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a deliberately non-physical sample-order trend.

    The trailing-window length is selected using only an internal forward
    validation tail inside the supplied training table. This model is a
    diagnostic for ordinal encoding; it is not a deployable materials model.
    """
    order = data["row_id"].to_numpy(float).reshape(-1, 1)
    supplied_train = np.flatnonzero(data["supplied_split"].eq("training"))
    supplied_test = np.flatnonzero(data["supplied_split"].eq("testing"))
    inner_train = supplied_train[:399]
    inner_validation = supplied_train[399:499]
    candidate_windows = [25, 50, 75, 100, 150, 200, 250, 300, 399]

    metric_rows = []
    prediction_rows = []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        candidates = []
        for window in candidate_windows:
            fit_index = inner_train[-min(window, len(inner_train)) :]
            estimator = LinearRegression().fit(order[fit_index], y[fit_index])
            validation_prediction = estimator.predict(order[inner_validation])
            candidates.append(
                {
                    "window": int(window),
                    "validation_mae": float(
                        mean_absolute_error(y[inner_validation], validation_prediction)
                    ),
                    "validation_r2": float(
                        r2_score(y[inner_validation], validation_prediction)
                    ),
                }
            )
        selected = min(candidates, key=lambda item: item["validation_mae"])
        window = selected["window"]
        fit_index = supplied_train[-window:]
        estimator = LinearRegression().fit(order[fit_index], y[fit_index])
        predicted = estimator.predict(order[supplied_test])
        metric_rows.append(
            {
                "target": target,
                "model": "Order-only local linear trend",
                "selected_window": int(window),
                "inner_validation_mae": selected["validation_mae"],
                "inner_validation_r2": selected["validation_r2"],
                "test_r2": float(r2_score(y[supplied_test], predicted)),
                "test_mae": float(mean_absolute_error(y[supplied_test], predicted)),
                "test_rmse": rmse(y[supplied_test], predicted),
            }
        )
        for row_id, actual, value in zip(supplied_test, y[supplied_test], predicted):
            prediction_rows.append(
                {
                    "target": target,
                    "row_id": int(row_id),
                    "actual": float(actual),
                    "predicted": float(value),
                    "selected_window": int(window),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def main() -> None:
    ensure_directories()
    data = load_qcar()
    x = data[FEATURES].to_numpy(float)

    protocols = {
        "Repeated shuffled 5-fold": list(
            RepeatedKFold(
                n_splits=5, n_repeats=5, random_state=RANDOM_SEED
            ).split(x)
        ),
        "Contiguous 5-fold": list(KFold(n_splits=5, shuffle=False).split(x)),
        "Original forward holdout": [
            (
                np.flatnonzero(data["supplied_split"].eq("training")),
                np.flatnonzero(data["supplied_split"].eq("testing")),
            )
        ],
    }

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for protocol_name, splits in protocols.items():
        protocol_summary, protocol_folds, protocol_predictions = evaluate_protocol(
            protocol_name, splits, x, data
        )
        summary_rows.extend(protocol_summary)
        fold_rows.extend(protocol_folds)
        prediction_rows.extend(protocol_predictions)
        print(f"Completed {protocol_name}")

    benchmark = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    ann = supplied_ann_metrics(data)
    order_metrics, order_predictions = order_trend_diagnostic(data)

    benchmark.to_csv(RESULTS_DIR / "model_benchmark_summary.csv", index=False)
    folds.to_csv(RESULTS_DIR / "model_benchmark_fold_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "model_predictions.csv", index=False)
    ann.to_csv(RESULTS_DIR / "supplied_ann_metrics.csv", index=False)
    order_metrics.to_csv(RESULTS_DIR / "order_trend_metrics.csv", index=False)
    order_predictions.to_csv(
        RESULTS_DIR / "order_trend_predictions.csv", index=False
    )

    random_extra = benchmark[
        (benchmark["protocol"] == "Repeated shuffled 5-fold")
        & (benchmark["model"] == "Extra Trees")
    ].set_index("target")
    forward_extra = benchmark[
        (benchmark["protocol"] == "Original forward holdout")
        & (benchmark["model"] == "Extra Trees")
    ].set_index("target")
    ann_train = ann[ann["split"] == "training"].set_index("target")
    ann_test = ann[ann["split"] == "testing"].set_index("target")

    compact = {
        "extra_trees_forward_to_random_mae_ratio": {
            target: float(
                forward_extra.loc[target, "pooled_mae"]
                / random_extra.loc[target, "pooled_mae"]
            )
            for target in TARGETS
        },
        "supplied_ann_test_to_train_mae_ratio": {
            target: float(
                ann_test.loc[target, "mae"] / ann_train.loc[target, "mae"]
            )
            for target in TARGETS
        },
        "order_trend_test_r2": {
            row["target"]: float(row["test_r2"])
            for _, row in order_metrics.iterrows()
        },
    }
    (RESULTS_DIR / "benchmark_key_findings.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
