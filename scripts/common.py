"""Shared configuration and utilities for the QCAR reassessment study."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

FEATURES = ["Al", "Cu", "Li", "Mg", "Mn", "Zr", "Si", "Zn", "Cd"]
TARGETS = [
    "Tensile Strength",
    "Proof Stress (mpa)",
    "Shear Strength (mpa)",
    "Hardness Vickers",
]
TARGET_SHORT = {
    "Tensile Strength": "Tensile strength",
    "Proof Stress (mpa)": "Proof stress",
    "Shear Strength (mpa)": "Shear strength",
    "Hardness Vickers": "Vickers hardness",
}
PREDICTION_COLUMNS = {
    "Tensile Strength": "Predict value of tensile Strength",
    "Proof Stress (mpa)": "predict value of Proof Stress",
    "Shear Strength (mpa)": "Predict valueof Shear Atrength",
    "Hardness Vickers": "Predict value of Hardness Vickers",
}
RANDOM_SEED = 20260806


def ensure_directories() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_qcar() -> pd.DataFrame:
    path = DATA_DIR / "qcar_667.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run scripts/01_extract_data.py first."
        )
    return pd.read_csv(path)


def regression_models(seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Return fixed, reproducible benchmark models.

    Hyperparameters are intentionally modest and fixed. The purpose is a
    validation-sensitivity benchmark rather than an exhaustive tuning contest.
    """
    return {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "PLS (3 components)": PLSRegression(
            n_components=3, scale=True, max_iter=1000
        ),
        "RBF-SVR": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(C=10.0, gamma="scale", epsilon=0.03),
            ),
            transformer=StandardScaler(),
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        ),
        "Histogram GB": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        ),
        "MLP": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(12,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1e-3,
                    max_iter=300,
                    tol=1e-5,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        ),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
