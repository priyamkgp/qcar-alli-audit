#!/usr/bin/env python3
"""Assess feature-attribution stability and uncertainty under distribution shift."""
from __future__ import annotations

import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from common import (
    FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGETS,
    ensure_directories,
    load_qcar,
)

warnings.filterwarnings("ignore")


def finite_pairwise_rank_stability(rank_vectors: list[np.ndarray]) -> dict:
    correlations = []
    for first, second in combinations(rank_vectors, 2):
        value = spearmanr(first, second).statistic
        if np.isfinite(value):
            correlations.append(float(value))
    if not correlations:
        return {
            "mean_pairwise_rank_spearman": float("nan"),
            "median_pairwise_rank_spearman": float("nan"),
            "minimum_pairwise_rank_spearman": float("nan"),
            "maximum_pairwise_rank_spearman": float("nan"),
            "finite_pair_count": 0,
        }
    return {
        "mean_pairwise_rank_spearman": float(np.mean(correlations)),
        "median_pairwise_rank_spearman": float(np.median(correlations)),
        "minimum_pairwise_rank_spearman": float(np.min(correlations)),
        "maximum_pairwise_rank_spearman": float(np.max(correlations)),
        "finite_pair_count": int(len(correlations)),
    }


def importance_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = data[FEATURES].to_numpy(float)
    protocols = {
        "Shuffled 5-fold": KFold(
            n_splits=5, shuffle=True, random_state=RANDOM_SEED
        ),
        "Contiguous 5-fold": KFold(n_splits=5, shuffle=False),
    }
    rows = []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        for protocol, cross_validator in protocols.items():
            for fold, (train_index, test_index) in enumerate(
                cross_validator.split(x)
            ):
                model = ExtraTreesRegressor(
                    n_estimators=80,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=RANDOM_SEED + fold,
                    n_jobs=1,
                ).fit(x[train_index], y[train_index])
                importance = permutation_importance(
                    model,
                    x[test_index],
                    y[test_index],
                    scoring="neg_mean_absolute_error",
                    n_repeats=5,
                    random_state=RANDOM_SEED + 100 + fold,
                    n_jobs=1,
                )
                for feature, mean_value, repeat_sd in zip(
                    FEATURES,
                    importance.importances_mean,
                    importance.importances_std,
                ):
                    rows.append(
                        {
                            "target": target,
                            "protocol": protocol,
                            "fold": fold,
                            "feature": feature,
                            "mae_increase_when_permuted": float(mean_value),
                            "repeat_sd": float(repeat_sd),
                        }
                    )
    fold_frame = pd.DataFrame(rows)
    summary = (
        fold_frame.groupby(["target", "protocol", "feature"], as_index=False)
        .agg(
            mean_importance=("mae_increase_when_permuted", "mean"),
            fold_sd=("mae_increase_when_permuted", "std"),
        )
        .fillna(0.0)
    )

    stability_rows = []
    for target in TARGETS:
        for protocol in protocols:
            vectors = []
            protocol_rows = fold_frame[
                (fold_frame["target"] == target)
                & (fold_frame["protocol"] == protocol)
            ]
            for fold in sorted(protocol_rows["fold"].unique()):
                values = (
                    protocol_rows[protocol_rows["fold"] == fold]
                    .set_index("feature")
                    .loc[FEATURES, "mae_increase_when_permuted"]
                )
                vectors.append(
                    values.rank(ascending=False, method="average").to_numpy(float)
                )
            stability = finite_pairwise_rank_stability(vectors)
            total_importance = float(
                summary[
                    (summary["target"] == target)
                    & (summary["protocol"] == protocol)
                ]["mean_importance"].sum()
            )
            stability_rows.append(
                {
                    "target": target,
                    "protocol": protocol,
                    "total_mean_importance": total_importance,
                    **stability,
                }
            )
    stability_frame = pd.DataFrame(stability_rows)

    # Magnitude collapse from interleaved to contiguous evaluation.
    pivot = stability_frame.pivot(
        index="target", columns="protocol", values="total_mean_importance"
    )
    magnitude_rows = []
    for target in TARGETS:
        shuffled = float(pivot.loc[target, "Shuffled 5-fold"])
        contiguous = float(pivot.loc[target, "Contiguous 5-fold"])
        magnitude_rows.append(
            {
                "target": target,
                "contiguous_to_shuffled_total_importance_ratio": contiguous
                / shuffled
                if shuffled != 0
                else float("nan"),
            }
        )
    magnitude_frame = pd.DataFrame(magnitude_rows)
    stability_frame = stability_frame.merge(magnitude_frame, on="target", how="left")
    return fold_frame, summary, stability_frame


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, level, method="higher"))


def conformal_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = data[FEATURES].to_numpy(float)
    rng = np.random.default_rng(RANDOM_SEED)
    alpha = 0.10
    trial_rows = []

    # Exchangeable random splits: 70% train, 15% calibration, 15% test.
    for target in TARGETS:
        y = data[target].to_numpy(float)
        target_range = float(y.max() - y.min())
        for repeat in range(20):
            indices = rng.permutation(len(data))
            n_train = int(0.70 * len(data))
            n_calibration = int(0.15 * len(data))
            train_index = indices[:n_train]
            calibration_index = indices[n_train : n_train + n_calibration]
            test_index = indices[n_train + n_calibration :]

            model = ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=1.0,
                random_state=RANDOM_SEED + repeat,
                n_jobs=1,
            ).fit(x[train_index], y[train_index])
            calibration_prediction = model.predict(x[calibration_index])
            radius = conformal_quantile(
                np.abs(y[calibration_index] - calibration_prediction), alpha
            )
            prediction = model.predict(x[test_index])
            coverage = np.mean(
                (y[test_index] >= prediction - radius)
                & (y[test_index] <= prediction + radius)
            )
            trial_rows.append(
                {
                    "target": target,
                    "protocol": "Random exchangeable split",
                    "repeat": repeat,
                    "nominal_coverage": 1 - alpha,
                    "observed_coverage": float(coverage),
                    "interval_width": float(2 * radius),
                    "normalized_interval_width": float(2 * radius / target_range),
                    "test_mae": float(
                        mean_absolute_error(y[test_index], prediction)
                    ),
                    "n_train": int(len(train_index)),
                    "n_calibration": int(len(calibration_index)),
                    "n_test": int(len(test_index)),
                }
            )

        print(f"Completed random conformal trials for {target}", flush=True)

        # Forward shift: first 400 train, next 99 calibrate, final 168 test.
        train_index = np.arange(0, 400)
        calibration_index = np.arange(400, 499)
        test_index = np.arange(499, len(data))
        model = ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ).fit(x[train_index], y[train_index])
        calibration_prediction = model.predict(x[calibration_index])
        radius = conformal_quantile(
            np.abs(y[calibration_index] - calibration_prediction), alpha
        )
        prediction = model.predict(x[test_index])
        coverage = np.mean(
            (y[test_index] >= prediction - radius)
            & (y[test_index] <= prediction + radius)
        )
        trial_rows.append(
            {
                "target": target,
                "protocol": "Forward-shift split",
                "repeat": 0,
                "nominal_coverage": 1 - alpha,
                "observed_coverage": float(coverage),
                "interval_width": float(2 * radius),
                "normalized_interval_width": float(2 * radius / target_range),
                "test_mae": float(mean_absolute_error(y[test_index], prediction)),
                "n_train": int(len(train_index)),
                "n_calibration": int(len(calibration_index)),
                "n_test": int(len(test_index)),
            }
        )

    trials = pd.DataFrame(trial_rows)
    summary_rows = []
    for (target, protocol), group in trials.groupby(["target", "protocol"]):
        summary_rows.append(
            {
                "target": target,
                "protocol": protocol,
                "nominal_coverage": float(group["nominal_coverage"].iloc[0]),
                "observed_coverage_mean": float(group["observed_coverage"].mean()),
                "observed_coverage_median": float(
                    group["observed_coverage"].median()
                ),
                "observed_coverage_p10": float(
                    group["observed_coverage"].quantile(0.10)
                ),
                "observed_coverage_p90": float(
                    group["observed_coverage"].quantile(0.90)
                ),
                "interval_width_mean": float(group["interval_width"].mean()),
                "normalized_interval_width_mean": float(
                    group["normalized_interval_width"].mean()
                ),
                "test_mae_mean": float(group["test_mae"].mean()),
                "trials": int(len(group)),
            }
        )
    return trials, pd.DataFrame(summary_rows)


def main() -> None:
    ensure_directories()
    data = load_qcar()

    importance_folds, importance_summary, importance_stability = importance_analysis(
        data
    )
    importance_folds.to_csv(
        RESULTS_DIR / "permutation_importance_folds.csv", index=False
    )
    importance_summary.to_csv(
        RESULTS_DIR / "permutation_importance_summary.csv", index=False
    )
    importance_stability.to_csv(
        RESULTS_DIR / "importance_stability.csv", index=False
    )

    conformal_trials, conformal_summary = conformal_analysis(data)
    conformal_trials.to_csv(RESULTS_DIR / "conformal_trials.csv", index=False)
    conformal_summary.to_csv(RESULTS_DIR / "conformal_summary.csv", index=False)

    key = {
        "importance_rank_stability": {
            f"{row['target']} | {row['protocol']}": float(
                row["mean_pairwise_rank_spearman"]
            )
            for _, row in importance_stability.iterrows()
        },
        "forward_conformal_coverage": {
            row["target"]: float(row["observed_coverage_mean"])
            for _, row in conformal_summary[
                conformal_summary["protocol"] == "Forward-shift split"
            ].iterrows()
        },
        "random_conformal_coverage": {
            row["target"]: float(row["observed_coverage_mean"])
            for _, row in conformal_summary[
                conformal_summary["protocol"] == "Random exchangeable split"
            ].iterrows()
        },
    }
    (RESULTS_DIR / "interpretability_uncertainty_key_findings.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )
    print(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()
