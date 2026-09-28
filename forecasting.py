"""
AQ forecasting: predicts pollutant spikes N days ahead.

- Compares several regressors under proper time-series cross-validation
  (chronological, never shuffled — avoids future-into-past leakage).
- Picks the best-performing model automatically and reports its CV score.
- Falls back gracefully if optional libraries (xgboost/lightgbm) aren't
  installed.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_features(df: pd.DataFrame, target_col: str = "no2_mol_m2", lags: list[int] = [1, 2, 3, 7]) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    df[f"{target_col}_roll3"] = df[target_col].rolling(3).mean()
    df[f"{target_col}_roll7"] = df[target_col].rolling(7).mean()
    df[f"{target_col}_std7"] = df[target_col].rolling(7).std()

    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    return df.dropna().reset_index(drop=True)


def _candidate_models() -> dict:
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    }
    if HAS_XGB:
        models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    if HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbosity=-1)
    return models


def cross_validate_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X) - 1))
    results = []

    for name, model in _candidate_models().items():
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            fold_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
        results.append({
            "model": name,
            "mean_mae": np.mean(fold_scores),
            "std_mae": np.std(fold_scores),
            "n_folds": len(fold_scores),
        })

    return pd.DataFrame(results).sort_values("mean_mae").reset_index(drop=True)


def train_forecaster(df: pd.DataFrame, target_col: str = "no2_mol_m2"):
    """Returns: (model, leaderboard_df, feature_cols)"""
    feature_df = build_features(df, target_col)
    feature_cols = [c for c in feature_df.columns if c not in ("date", target_col)]

    X = feature_df[feature_cols]
    y = feature_df[target_col]

    leaderboard = cross_validate_models(X, y)
    best_name = leaderboard.iloc[0]["model"]
    best_model = _candidate_models()[best_name]
    best_model.fit(X, y)

    return best_model, leaderboard, feature_cols


def save_model(model, path: str = "data/aq_forecaster.joblib"):
    joblib.dump(model, path)


def load_model(path: str = "data/aq_forecaster.joblib"):
    return joblib.load(path)


def predict_next(model, latest_features: pd.DataFrame) -> np.ndarray:
    return model.predict(latest_features)


if __name__ == "__main__":
    df = pd.read_csv("data/satellite_timeseries.csv")
    model, leaderboard, feature_cols = train_forecaster(df)

    print("Model leaderboard (chronological CV, lower MAE is better):")
    print(leaderboard.to_string(index=False))
    print(f"\nSelected: {leaderboard.iloc[0]['model']}")

    save_model(model)
    leaderboard.to_csv("data/model_leaderboard.csv", index=False)

    # Write the federated model-export manifest — see docs/federated_model_exchange.md
    from model_export import export_model_manifest
    export_model_manifest(
        model_name=leaderboard.iloc[0]["model"],
        feature_cols=feature_cols,
        mae=leaderboard.iloc[0]["mean_mae"],
        train_rows=len(df),
    )
