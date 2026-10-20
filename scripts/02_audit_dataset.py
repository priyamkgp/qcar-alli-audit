#!/usr/bin/env python3
"""Audit QCAR dataset geometry, ordering, split shift and source-table proximity."""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from common import (
    DATA_DIR,
    FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGETS,
    ensure_directories,
    load_qcar,
)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def main() -> None:
    ensure_directories()
    data = load_qcar()
    n = len(data)
    row_index = np.arange(n, dtype=float)

    # 1. Compositional closure and uniqueness.
    closure_error = (data[FEATURES].sum(axis=1) - 100.0).abs()
    unique_compositions = int(data[FEATURES].drop_duplicates().shape[0])
    unique_targets = {
        target: int(data[target].nunique(dropna=False)) for target in TARGETS
    }

    # 2. Ordering diagnostics.
    order_rows = []
    for variable in FEATURES + TARGETS:
        values = data[variable].to_numpy(float)
        differences = np.diff(values)
        order_rows.append(
            {
                "variable": variable,
                "spearman_with_row_id": safe_spearman(row_index, values),
                "strict_increases": int((differences > 0).sum()),
                "strict_decreases": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
                "nondecreasing_fraction": float((differences >= 0).mean()),
                "nonincreasing_fraction": float((differences <= 0).mean()),
            }
        )
    order_frame = pd.DataFrame(order_rows)
    order_frame.to_csv(RESULTS_DIR / "order_correlations.csv", index=False)

    target_spearman = data[TARGETS].corr(method="spearman")
    target_spearman.to_csv(RESULTS_DIR / "target_spearman.csv")

    # 3. Principal-component geometry.
    standardized = StandardScaler().fit_transform(data[FEATURES])
    pca = PCA().fit(standardized)
    pca_frame = pd.DataFrame(
        {
            "component": np.arange(1, len(FEATURES) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    pca_frame.to_csv(RESULTS_DIR / "pca_explained_variance.csv", index=False)
    scores = pca.transform(standardized)
    pd.DataFrame(
        {
            "row_id": data["row_id"],
            "supplied_split": data["supplied_split"],
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2],
        }
    ).to_csv(RESULTS_DIR / "pca_scores.csv", index=False)

    # 4. Distribution shift in the supplied 499/168 split.
    supplied_train = data[data["supplied_split"] == "training"]
    supplied_test = data[data["supplied_split"] == "testing"]
    split_rows = []
    for variable in FEATURES + TARGETS:
        train_values = supplied_train[variable].to_numpy(float)
        test_values = supplied_test[variable].to_numpy(float)
        pooled_sd = np.sqrt(
            (
                (len(train_values) - 1) * np.var(train_values, ddof=1)
                + (len(test_values) - 1) * np.var(test_values, ddof=1)
            )
            / (len(train_values) + len(test_values) - 2)
        )
        split_rows.append(
            {
                "variable": variable,
                "train_min": float(train_values.min()),
                "train_max": float(train_values.max()),
                "test_min": float(test_values.min()),
                "test_max": float(test_values.max()),
                "ks_statistic": float(ks_2samp(train_values, test_values).statistic),
                "standardized_mean_difference": float(
                    (test_values.mean() - train_values.mean()) / pooled_sd
                )
                if pooled_sd > 0
                else float("nan"),
                "range_overlap": bool(
                    max(train_values.min(), test_values.min())
                    <= min(train_values.max(), test_values.max())
                ),
            }
        )
    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(RESULTS_DIR / "split_shift.csv", index=False)

    # Nearest-neighbour distance of the forward test tail versus random holdouts.
    scaler = StandardScaler().fit(supplied_train[FEATURES])
    x_train = scaler.transform(supplied_train[FEATURES])
    x_test = scaler.transform(supplied_test[FEATURES])
    forward_nn = pairwise_distances(x_test, x_train).min(axis=1)

    rng = np.random.default_rng(RANDOM_SEED)
    random_rows = []
    holdout_size = len(supplied_test)
    all_indices = np.arange(n)
    for repeat in range(100):
        test_indices = rng.choice(n, holdout_size, replace=False)
        train_indices = np.setdiff1d(all_indices, test_indices)
        repeat_scaler = StandardScaler().fit(data.iloc[train_indices][FEATURES])
        repeat_train = repeat_scaler.transform(data.iloc[train_indices][FEATURES])
        repeat_test = repeat_scaler.transform(data.iloc[test_indices][FEATURES])
        distances = pairwise_distances(repeat_test, repeat_train).min(axis=1)
        for distance in distances:
            random_rows.append(
                {
                    "protocol": "random holdout",
                    "repeat": repeat,
                    "distance": float(distance),
                }
            )
    nn_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "protocol": "supplied forward tail",
                    "repeat": 0,
                    "distance": forward_nn,
                }
            ),
            pd.DataFrame(random_rows),
        ],
        ignore_index=True,
    )
    nn_frame.to_csv(RESULTS_DIR / "nearest_neighbor_shift.csv", index=False)

    # 5. Proximity to the complete source lookup compositions.
    source_path = DATA_DIR / "source_lookup_compositions.csv"
    source = pd.read_csv(source_path)
    for feature in FEATURES:
        source[feature] = pd.to_numeric(source[feature], errors="coerce")
    source = source.dropna(subset=FEATURES).copy()

    feature_scale = data[FEATURES].std(ddof=0).replace(0, 1.0)
    standardized_distance = pairwise_distances(
        source[FEATURES] / feature_scale,
        data[FEATURES] / feature_scale,
    )
    chebyshev_distance = pairwise_distances(
        source[FEATURES], data[FEATURES], metric="chebyshev"
    )
    nearest_index = standardized_distance.argmin(axis=1)
    source_distances = pd.DataFrame(
        {
            "alloy_series": source["alloy"] if "alloy" in source else pd.Series(range(len(source))),
            "nearest_qcar_row": nearest_index,
            "standardized_euclidean_distance": standardized_distance.min(axis=1),
            "maximum_absolute_component_difference_wt_pct": chebyshev_distance.min(
                axis=1
            ),
        }
    )
    source_distances.to_csv(
        RESULTS_DIR / "source_anchor_distances.csv", index=False
    )

    exact_at_001 = int(
        (source_distances["maximum_absolute_component_difference_wt_pct"] < 0.001).sum()
    )
    close_at_005 = int(
        (source_distances["maximum_absolute_component_difference_wt_pct"] < 0.05).sum()
    )

    summary = {
        "rows": int(n),
        "features": int(len(FEATURES)),
        "targets": int(len(TARGETS)),
        "supplied_training_rows": int(len(supplied_train)),
        "supplied_testing_rows": int(len(supplied_test)),
        "mean_absolute_closure_error_wt_pct": float(closure_error.mean()),
        "maximum_absolute_closure_error_wt_pct": float(closure_error.max()),
        "unique_compositions": unique_compositions,
        "unique_targets": unique_targets,
        "all_targets_strictly_increasing": bool(
            all(
                (np.diff(data[target].to_numpy(float)) > 0).all()
                for target in TARGETS
            )
        ),
        "minimum_pairwise_target_spearman": float(
            target_spearman.where(~np.eye(len(TARGETS), dtype=bool)).min().min()
        ),
        "pca_first_component_variance": float(pca.explained_variance_ratio_[0]),
        "pca_first_two_components_variance": float(
            pca.explained_variance_ratio_[:2].sum()
        ),
        "pca_first_four_components_variance": float(
            pca.explained_variance_ratio_[:4].sum()
        ),
        "variables_with_ks_equal_one": split_frame.loc[
            np.isclose(split_frame["ks_statistic"], 1.0), "variable"
        ].tolist(),
        "forward_test_nn_distance_mean": float(forward_nn.mean()),
        "random_holdout_nn_distance_mean": float(
            nn_frame.loc[nn_frame["protocol"] == "random holdout", "distance"].mean()
        ),
        "forward_to_random_nn_distance_ratio": float(
            forward_nn.mean()
            / nn_frame.loc[
                nn_frame["protocol"] == "random holdout", "distance"
            ].mean()
        ),
        "complete_source_lookup_compositions": int(len(source)),
        "source_exact_matches_max_abs_lt_0_001_wt_pct": exact_at_001,
        "source_close_matches_max_abs_lt_0_05_wt_pct": close_at_005,
        "source_distance_median_standardized": float(
            source_distances["standardized_euclidean_distance"].median()
        ),
    }
    (RESULTS_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
