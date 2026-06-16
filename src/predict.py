"""Interactive player market value predictor.

Run from repo root:
    python -m src.predict
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# A gap of >20% from TM is treated as under/overvalued.
_THRESHOLD = 0.20


def _verdict(predicted_eur: float, actual_eur: float) -> str:
    if actual_eur <= 0:
        return "unknown"
    ratio = (predicted_eur - actual_eur) / actual_eur
    if ratio > _THRESHOLD:
        return f"UNDERVALUED  (model sees +{ratio:.0%} upside)"
    elif ratio < -_THRESHOLD:
        return f"OVERVALUED   (model sees {ratio:.0%} downside)"
    else:
        return f"FAIR VALUE   (within {ratio:+.0%} of TM)"


def main() -> None:
    logger.info("Loading data (cached after first run)...")
    from src.ingest import fetch_api_football, load_transfermarkt

    tm = load_transfermarkt()
    api_players = fetch_api_football()

    logger.info("Building joined dataset...")
    from src.join import build_joined_dataset

    joined = build_joined_dataset(api_players, tm)

    logger.info("Building features...")
    from src.features import build_stage1

    s1 = build_stage1(joined)

    from src.models import _DROP_COLS

    feature_cols = [c for c in s1.columns if c not in _DROP_COLS]

    logger.info("Training Random Forest on 2022/23 + 2023/24, predicting 2024/25...")
    train = s1[s1["season"] < 2024]
    test = s1[s1["season"] == 2024].copy()

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    rf.fit(train[feature_cols].fillna(0).values, train["log_market_value"].values)

    test["predicted_eur"] = np.expm1(rf.predict(test[feature_cols].fillna(0).values))
    pred_df = test

    names = joined[["api_player_id", "season", "firstname", "lastname"]].drop_duplicates(
        ["api_player_id", "season"]
    )
    pred_df = pred_df.merge(names, on=["api_player_id", "season"], how="left")
    pred_df["name"] = pred_df["firstname"].fillna("") + " " + pred_df["lastname"].fillna("")
    pred_df["name_lower"] = pred_df["name"].str.lower()

    print("\n--- Player value predictor ---")
    print("Type part of a player name to look up. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Search player: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        matches = pred_df[pred_df["name_lower"].str.contains(query.lower(), na=False)]

        if matches.empty:
            print(f"  No player found matching '{query}'\n")
            continue

        for _, row in matches.sort_values("season", ascending=False).iterrows():
            season_label = f"{int(row['season'])}/{str(int(row['season']) + 1)[-2:]}"
            actual_m = row["market_value_in_eur"] / 1e6
            predicted_m = row["predicted_eur"] / 1e6
            verdict = _verdict(row["predicted_eur"], row["market_value_in_eur"])
            print(
                f"  {row['name']:<30}  {season_label}  "
                f"TM: €{actual_m:>6.1f}M   Model: €{predicted_m:>6.1f}M   {verdict}"
            )
        print()


if __name__ == "__main__":
    main()
