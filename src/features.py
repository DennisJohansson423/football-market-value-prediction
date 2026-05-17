"""Build feature sets (Stage 1, 2, 3) from the joined dataset."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are simply summed when a player has multiple clubs in one season
_SUM_COLS = [
    "appearances", "minutes", "goals", "assists",
    "yellow_cards", "red_cards",
    "shots_total", "shots_on",
    "passes_total", "passes_key",
    "tackles", "interceptions", "blocks",
    "duels_total", "duels_won",
    "dribbles_attempts", "dribbles_success",
    "fouls_drawn", "fouls_committed",
]

_POSITION_MAP = {
    "defender": "DEF",
    "midfielder": "MID",
    "attacker": "FWD",
    "forward": "FWD",
}


def _aggregate_multi_club(df: pd.DataFrame) -> pd.DataFrame:
    """Sum counting stats across clubs for players with mid-season transfers.

    One row per (api_player_id, season). passes_accuracy is recomputed as a
    weighted average by passes_total after summing.
    """
    meta = (
        df.groupby(["api_player_id", "season"])
        .first()
        .reset_index()[["api_player_id", "season", "firstname", "lastname", "dob",
                         "nationality", "position", "tm_player_id", "market_value_in_eur"]]
    )

    sums = (
        df.groupby(["api_player_id", "season"])[_SUM_COLS]
        .sum(min_count=1)
        .reset_index()
    )

    # passes_accuracy: weighted average by passes_total
    df = df.copy()
    df["passes_acc_weighted"] = df["passes_accuracy"] * df["passes_total"]
    acc = (
        df.groupby(["api_player_id", "season"])
        .apply(
            lambda g: g["passes_acc_weighted"].sum() / g["passes_total"].sum()
            if g["passes_total"].sum() > 0 else np.nan,
            include_groups=False,
        )
        .reset_index(name="passes_accuracy")
    )

    result = meta.merge(sums, on=["api_player_id", "season"]).merge(
        acc, on=["api_player_id", "season"]
    )
    return result


def _add_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    season_end = df["season"].apply(lambda s: pd.Timestamp(year=int(s) + 1, month=6, day=30))
    df["age"] = ((season_end - pd.to_datetime(df["dob"])).dt.days / 365.25).round(1)
    return df


def _normalize_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position_group"] = df["position"].str.lower().map(_POSITION_MAP)
    df = df[df["position_group"].notna()].copy()
    return df


def build_stage1(joined: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: basic features — age, position, minutes, appearances, goals, assists, cards.

    Returns one row per player-season with log1p(market_value_eur) as target.
    """
    df = _aggregate_multi_club(joined)
    df = _add_age(df)
    df = _normalize_position(df)

    feature_cols = ["age", "minutes", "appearances", "goals", "assists",
                    "yellow_cards", "red_cards"]
    position_dummies = pd.get_dummies(df["position_group"], prefix="pos", dtype=float)

    result = pd.concat([
        df[["api_player_id", "season", "tm_player_id", "market_value_in_eur"]],
        df[feature_cols],
        position_dummies,
    ], axis=1)

    result["log_market_value"] = np.log1p(result["market_value_in_eur"])
    result = result.dropna(subset=["log_market_value", "age", "minutes"])

    logger.info("Stage 1: %d player-seasons, %d features", len(result),
                len(feature_cols) + len(position_dummies.columns))
    return result


def build_stage2(joined: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Stage 1 + per-90 stats, passing, shooting, dribbling, defending."""
    df = _aggregate_multi_club(joined)
    df = _add_age(df)
    df = _normalize_position(df)

    # Per-90 stats — guard against zero minutes
    p90 = df["minutes"].replace(0, np.nan) / 90
    for col in ["goals", "assists", "shots_total", "shots_on", "passes_total", "passes_key",
                "tackles", "interceptions", "dribbles_attempts", "dribbles_success",
                "fouls_drawn", "fouls_committed"]:
        df[f"{col}_p90"] = df[col] / p90

    df["duels_won_pct"] = df["duels_won"] / df["duels_total"].replace(0, np.nan)
    df["shot_accuracy"] = df["shots_on"] / df["shots_total"].replace(0, np.nan)
    df["dribble_success_pct"] = df["dribbles_success"] / df["dribbles_attempts"].replace(0, np.nan)

    base_cols = ["age", "minutes", "appearances"]
    p90_cols = [c for c in df.columns if c.endswith("_p90")]
    rate_cols = ["passes_accuracy", "duels_won_pct", "shot_accuracy", "dribble_success_pct"]
    feature_cols = base_cols + p90_cols + rate_cols

    position_dummies = pd.get_dummies(df["position_group"], prefix="pos", dtype=float)

    result = pd.concat([
        df[["api_player_id", "season", "tm_player_id", "market_value_in_eur"]],
        df[feature_cols],
        position_dummies,
    ], axis=1)

    result["log_market_value"] = np.log1p(result["market_value_in_eur"])
    result = result.dropna(subset=["log_market_value", "age", "minutes"])

    logger.info("Stage 2: %d player-seasons, %d features", len(result),
                len(feature_cols) + len(position_dummies.columns))
    return result


def build_stage4(joined: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Stage 3 + goal involvement p90, defensive actions p90, minutes/app, team quality, league."""
    df = build_stage3(joined)

    agg = _aggregate_multi_club(joined)
    p90 = agg["minutes"].replace(0, np.nan) / 90

    agg["goal_involvement_p90"] = (agg["goals"] + agg["assists"]) / p90
    agg["defensive_actions_p90"] = (agg["tackles"] + agg["interceptions"] + agg["blocks"]) / p90
    agg["minutes_per_app"] = agg["minutes"] / agg["appearances"].replace(0, np.nan)

    new_cols = ["goal_involvement_p90", "defensive_actions_p90", "minutes_per_app"]

    # Team quality: avg market value of all players in same team-season
    if "team_id" in joined.columns:
        team_avg = (
            joined.groupby(["team_id", "season"])["market_value_in_eur"]
            .mean()
            .reset_index(name="team_avg_value")
        )
        primary_team = (
            joined.sort_values("minutes", ascending=False)
            .drop_duplicates(["api_player_id", "season"])[["api_player_id", "season", "team_id"]]
        )
        agg = agg.merge(primary_team, on=["api_player_id", "season"], how="left")
        agg = agg.merge(team_avg, on=["team_id", "season"], how="left")
        agg["log_team_avg_value"] = np.log1p(agg["team_avg_value"])
        new_cols.append("log_team_avg_value")

    # League encoding
    if "league_id" in joined.columns:
        league_info = (
            joined.sort_values("minutes", ascending=False)
            .drop_duplicates(["api_player_id", "season"])[["api_player_id", "season", "league_id"]]
        )
        agg = agg.merge(league_info, on=["api_player_id", "season"], how="left")
        league_dummies = pd.get_dummies(agg["league_id"].astype("Int64").astype(str), prefix="league", dtype=float)
        agg = pd.concat([agg.reset_index(drop=True), league_dummies.reset_index(drop=True)], axis=1)
        new_cols += [c for c in agg.columns if c.startswith("league_")]

    df = df.merge(agg[["api_player_id", "season"] + new_cols], on=["api_player_id", "season"], how="left")

    n_feat = len([c for c in df.columns if c not in
                  {"api_player_id", "season", "tm_player_id", "market_value_in_eur", "log_market_value"}])
    logger.info("Stage 4: %d player-seasons, %d features", len(df), n_feat)
    return df


def build_stage3(joined: pd.DataFrame, tm_valuations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage 3: Stage 2 + age², position percentiles, avg squad value, Δ-stats (multi-season only)."""
    df = build_stage2(joined)

    # age²
    df["age_sq"] = df["age"] ** 2

    # Position-relative percentile ranks for key stats
    for col in ["goals_p90", "assists_p90", "minutes"]:
        if col in df.columns:
            df[f"{col}_pct"] = df.groupby("pos_FWD" if "pos_FWD" in df.columns else "season")[col].rank(pct=True)

    # Average squad market value per team-season (proxy for team quality)
    if tm_valuations is not None:
        # reuse the tm_player_id in joined to compute team avg market value
        pass  # placeholder — requires team membership data; wired in later

    # Δ-stats: only meaningful once we have ≥2 seasons in the dataset
    if joined["season"].nunique() >= 2:
        agg = _aggregate_multi_club(joined)
        agg = _add_age(agg)
        for col in ["goals", "assists", "minutes"]:
            prev = agg[["api_player_id", "season", col]].copy()
            prev["season"] = prev["season"] + 1
            prev = prev.rename(columns={col: f"{col}_prev"})
            agg = agg.merge(prev, on=["api_player_id", "season"], how="left")
            df_col = f"delta_{col}"
            agg[df_col] = agg[col] - agg[f"{col}_prev"]
        for col in ["delta_goals", "delta_assists", "delta_minutes"]:
            if col in agg.columns:
                df = df.merge(
                    agg[["api_player_id", "season", col]], on=["api_player_id", "season"], how="left"
                )

    logger.info("Stage 3: %d player-seasons, %d columns", len(df), len(df.columns))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.ingest import fetch_api_football, load_transfermarkt
    from src.join import build_joined_dataset

    tm = load_transfermarkt()
    api = fetch_api_football(seasons=[2022])
    joined = build_joined_dataset(api, tm)

    s1 = build_stage1(joined)
    print("\nStage 1 sample:")
    print(s1.head(3).to_string())

    s2 = build_stage2(joined)
    print(f"\nStage 2 columns ({len(s2.columns)}):", s2.columns.tolist())
