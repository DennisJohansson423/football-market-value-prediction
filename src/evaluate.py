"""SHAP feature importance and transfer-fee benchmarking."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

from src.models import _DROP_COLS, _RANDOM_STATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


def shap_importance(
    df: pd.DataFrame,
    position: str = "global",
    top_n: int = 15,
) -> tuple[plt.Figure, np.ndarray, list[str]]:
    """Train RF on earlier seasons, compute SHAP on the latest season.

    position: 'global' | 'DEF' | 'MID' | 'FWD'
    Returns (figure, shap_values, feature_names).
    """
    data = df.copy()

    if position != "global":
        pos_cols = [c for c in data.columns if c.startswith("pos_")]
        data["_pos"] = data[pos_cols].idxmax(axis=1).str.replace("pos_", "", regex=False)
        data = data[data["_pos"] == position].copy()

    feature_cols = [c for c in data.columns if c not in _DROP_COLS and not c.startswith("_")]
    test_season = data["season"].max()
    train = data[data["season"] < test_season]
    test = data[data["season"] == test_season]

    if len(train) < 10 or len(test) < 5:
        raise ValueError(f"Not enough data for position={position}")

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=_RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(train[feature_cols].fillna(0).values, train["log_market_value"].values)

    X_test = test[feature_cols].fillna(0).values
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [feature_cols[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features[::-1], top_vals[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|  (impact on log market value)")
    ax.set_title(f"Top {top_n} features by SHAP importance — {position}")
    plt.tight_layout()

    logger.info("SHAP done for position=%s, test_season=%d, n=%d", position, test_season, len(test))
    return fig, shap_values, feature_cols


# ---------------------------------------------------------------------------
# Transfer-fee benchmarking
# ---------------------------------------------------------------------------


def _parse_season_year(s: str) -> int | None:
    """'24/25' → 2024, '9/10' → 2009, etc."""
    try:
        left = str(s).split("/")[0].strip()
        n = int(left)
        return (2000 + n) if n < 100 else n
    except (ValueError, IndexError):
        return None


def build_transfer_benchmark(
    transfers: pd.DataFrame,
    predictions: pd.DataFrame,
    target_season_year: int | None = None,
) -> pd.DataFrame:
    """Compare model predictions vs TM values against actual paid transfer fees.

    transfers: tm.transfers (PL-filtered)
    predictions: DataFrame with columns [tm_player_id, predicted_eur, market_value_in_eur]
                 — one row per player (most recent season predictions)
    target_season_year: e.g. 2024 for the 2024/25 transfer window (summer 2024).
                        Defaults to the latest available season with paid transfers.

    Returns a DataFrame with one row per matched transfer, sorted by fee descending.
    """
    fees = transfers.copy()
    fees = fees.rename(columns={"player_id": "tm_player_id"})
    fees["transfer_fee"] = pd.to_numeric(fees["transfer_fee"], errors="coerce").fillna(0)
    fees["market_value_in_eur"] = pd.to_numeric(fees["market_value_in_eur"], errors="coerce").fillna(0)
    fees["transfer_date"] = pd.to_datetime(fees["transfer_date"], errors="coerce")
    fees["season_year"] = fees["transfer_season"].apply(_parse_season_year)

    # Only keep transfers with an actual paid fee
    fees = fees[fees["transfer_fee"] > 0].copy()

    if target_season_year is None:
        # Pick the most recent season that has any overlap with our predictions
        available = sorted(fees["season_year"].dropna().unique(), reverse=True)
        pred_ids = set(predictions["tm_player_id"].dropna())
        for yr in available:
            candidate = fees[fees["season_year"] == yr]
            if candidate["tm_player_id"].isin(pred_ids).any():
                target_season_year = yr
                break

    if target_season_year is None:
        logger.warning("No overlapping transfers found.")
        return pd.DataFrame()

    window = fees[fees["season_year"] == target_season_year].copy()
    logger.info(
        "Transfer window %d/%02d: %d paid transfers, checking against %d predicted players",
        target_season_year, (target_season_year + 1) % 100,
        len(window), len(predictions),
    )

    merged = window.merge(
        predictions[["tm_player_id", "predicted_eur", "market_value_in_eur"]].rename(
            columns={"market_value_in_eur": "tm_snap_value"}
        ),
        on="tm_player_id",
        how="inner",
    )

    if merged.empty:
        logger.warning("No matched transfers for season %d.", target_season_year)
        return pd.DataFrame()

    merged["actual_fee_M"] = merged["transfer_fee"] / 1e6
    merged["tm_value_M"] = merged["market_value_in_eur"] / 1e6
    merged["model_value_M"] = merged["predicted_eur"] / 1e6
    merged["error_model_M"] = (merged["model_value_M"] - merged["actual_fee_M"]).abs()
    merged["error_tm_M"] = (merged["tm_value_M"] - merged["actual_fee_M"]).abs()

    result = merged[[
        "player_name", "from_club_name", "to_club_name",
        "actual_fee_M", "tm_value_M", "model_value_M",
        "error_model_M", "error_tm_M",
    ]].sort_values("actual_fee_M", ascending=False).reset_index(drop=True)

    mae_model = result["error_model_M"].mean()
    mae_tm = result["error_tm_M"].mean()
    winner = "MODEL" if mae_model < mae_tm else "Transfermarkt"
    logger.info(
        "Benchmark: %d matched transfers | MAE model=€%.1fM | MAE TM=€%.1fM | winner=%s",
        len(result), mae_model, mae_tm, winner,
    )
    return result
