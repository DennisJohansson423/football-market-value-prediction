"""Train and evaluate models on each feature stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

_RANDOM_STATE = 42

# Feature columns to drop before training — identifiers and the raw target
_DROP_COLS = {"api_player_id", "season", "tm_player_id", "market_value_in_eur", "log_market_value"}


@dataclass
class ModelResult:
    model_name: str
    stage: str
    position: str          # "global" or "DEF" / "MID" / "FWD"
    n_samples: int
    # Log-space metrics
    rmse_log: float
    mae_log: float
    r2_log: float
    spearman_log: float
    # EUR metrics (after exp(pred) - 1)
    rmse_eur: float
    mae_eur: float


def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)
    spearman_log = spearmanr(y_true_log, y_pred_log).statistic

    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_log)
    rmse_eur = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)

    return dict(rmse_log=rmse_log, mae_log=mae_log, r2_log=r2_log,
                spearman_log=spearman_log, rmse_eur=rmse_eur, mae_eur=mae_eur)


def _make_models() -> dict[str, any]:
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=_RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=_RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=_RANDOM_STATE, verbosity=0,
        ),
    }


def _cv_predict(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Return out-of-fold predictions via k-fold CV."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all seasons except the latest; test on the latest."""
    test_season = df["season"].max()
    return df[df["season"] < test_season].copy(), df[df["season"] == test_season].copy()


def train_global(df: pd.DataFrame, stage: str) -> list[ModelResult]:
    """Train one model on all positions combined."""
    feature_cols = [c for c in df.columns if c not in _DROP_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["log_market_value"].values

    results = []
    for name, model in _make_models().items():
        if df["season"].nunique() >= 2:
            train, test = _temporal_split(df)
            X_tr = train[feature_cols].fillna(0).values
            y_tr = train["log_market_value"].values
            X_te = test[feature_cols].fillna(0).values
            y_te = test["log_market_value"].values
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            m = _metrics(y_te, y_pred)
            n = len(y_te)
        else:
            y_pred = _cv_predict(model, X, y)
            m = _metrics(y, y_pred)
            n = len(y)

        res = ModelResult(model_name=name, stage=stage, position="global",
                          n_samples=n, **m)
        results.append(res)
        logger.info("  [global] %s  R²=%.3f  RMSE_log=%.3f  MAE_eur=€%.0fM",
                    name, res.r2_log, res.rmse_log, res.mae_eur / 1e6)
    return results


def train_position_aware(df: pd.DataFrame, stage: str) -> list[ModelResult]:
    """Train separate models per position group (DEF / MID / FWD)."""
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    if not pos_cols:
        logger.warning("No position dummy columns found; skipping position-aware training.")
        return []

    # Recover position_group from dummies
    df = df.copy()
    df["position_group"] = df[pos_cols].idxmax(axis=1).str.replace("pos_", "")

    results = []
    for pos in ["DEF", "MID", "FWD"]:
        subset = df[df["position_group"] == pos].copy()
        if len(subset) < 30:
            logger.warning("Position %s: only %d samples — skipping.", pos, len(subset))
            continue

        feature_cols = [c for c in subset.columns if c not in _DROP_COLS | {"position_group"}]
        X = subset[feature_cols].fillna(0).values
        y = subset["log_market_value"].values

        for name, model in _make_models().items():
            if subset["season"].nunique() >= 2:
                train, test = _temporal_split(subset)
                X_tr = train[feature_cols].fillna(0).values
                y_tr = train["log_market_value"].values
                X_te = test[feature_cols].fillna(0).values
                y_te = test["log_market_value"].values
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                m = _metrics(y_te, y_pred)
                n = len(y_te)
            else:
                y_pred = _cv_predict(model, X, y)
                m = _metrics(y, y_pred)
                n = len(y)

            res = ModelResult(model_name=name, stage=stage, position=pos,
                              n_samples=n, **m)
            results.append(res)
            logger.info("  [%s] %s  R²=%.3f  RMSE_log=%.3f  MAE_eur=€%.0fM",
                        pos, name, res.r2_log, res.rmse_log, res.mae_eur / 1e6)
    return results


def results_to_df(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "stage": r.stage, "position": r.position, "model": r.model_name,
            "n": r.n_samples, "R²": round(r.r2_log, 3),
            "RMSE_log": round(r.rmse_log, 3), "MAE_log": round(r.mae_log, 3),
            "Spearman_ρ": round(r.spearman_log, 3),
            "MAE_eur_M": round(r.mae_eur / 1e6, 1),
        }
        for r in results
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.features import build_stage1, build_stage2
    from src.ingest import fetch_api_football, load_transfermarkt
    from src.join import build_joined_dataset

    tm = load_transfermarkt()
    api = fetch_api_football(seasons=[2022])
    joined = build_joined_dataset(api, tm)

    all_results = []
    for stage_name, builder in [("stage1", build_stage1), ("stage2", build_stage2)]:
        logger.info("=== %s ===", stage_name)
        df = builder(joined)
        all_results += train_global(df, stage_name)
        all_results += train_position_aware(df, stage_name)

    table = results_to_df(all_results)
    print("\n" + table.to_string(index=False))
