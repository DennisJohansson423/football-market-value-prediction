"""Match API-Football players to Transfermarkt players and produce a joined dataset."""

from __future__ import annotations

import logging
import re

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 80  # minimum token_sort_ratio score to accept a fuzzy match


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Lowercase, strip accents, remove non-alpha, collapse whitespace."""
    name = unidecode(str(name)).lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ---------------------------------------------------------------------------
# Flatten API-Football records into a DataFrame
# ---------------------------------------------------------------------------


def api_players_to_df(api_players: list[dict]) -> pd.DataFrame:
    """Flatten raw API-Football records into a tidy DataFrame.

    One row per player-season. Multi-club seasons (multiple statistics[] entries)
    are kept as-is here; summing happens in features.py.
    """
    rows = []
    for rec in api_players:
        p = rec["player"]
        for stat in rec["statistics"]:
            rows.append(
                {
                    "api_player_id": p["id"],
                    "season": rec["_season"],
                    "firstname": p.get("firstname") or "",
                    "lastname": p.get("lastname") or "",
                    "dob": p["birth"]["date"],
                    "nationality": p.get("nationality") or "",
                    "position": stat["games"].get("position") or "",
                    "team_name": stat["team"]["name"],
                    "team_id": stat["team"]["id"],
                    # games
                    "appearances": stat["games"].get("appearences"),
                    "minutes": stat["games"].get("minutes"),
                    # goals / assists
                    "goals": stat["goals"].get("total"),
                    "assists": stat["goals"].get("assists"),
                    # cards
                    "yellow_cards": stat["cards"].get("yellow"),
                    "red_cards": stat["cards"].get("red"),
                    # shots
                    "shots_total": stat["shots"].get("total"),
                    "shots_on": stat["shots"].get("on"),
                    # passes
                    "passes_total": stat["passes"].get("total"),
                    "passes_key": stat["passes"].get("key"),
                    "passes_accuracy": stat["passes"].get("accuracy"),
                    # tackles / interceptions
                    "tackles": stat["tackles"].get("total"),
                    "interceptions": stat["tackles"].get("interceptions"),
                    "blocks": stat["tackles"].get("blocks"),
                    # duels
                    "duels_total": stat["duels"].get("total"),
                    "duels_won": stat["duels"].get("won"),
                    # dribbles
                    "dribbles_attempts": stat["dribbles"].get("attempts"),
                    "dribbles_success": stat["dribbles"].get("success"),
                    # fouls
                    "fouls_drawn": stat["fouls"].get("drawn"),
                    "fouls_committed": stat["fouls"].get("committed"),
                    # rating / penalty
                    "rating": float(stat["games"]["rating"]) if stat["games"].get("rating") is not None else None,
                    "penalty_goals": stat["penalty"].get("scored"),
                    "league_id": rec.get("_league_id"),
                }
            )
    df = pd.DataFrame(rows)
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["full_name_norm"] = (df["firstname"] + " " + df["lastname"]).map(_normalize)
    df["last_name_norm"] = df["lastname"].map(_normalize)
    return df


# ---------------------------------------------------------------------------
# Build crosswalk: api_player_id → tm_player_id
# ---------------------------------------------------------------------------


def build_crosswalk(api_players: list[dict], tm_players: pd.DataFrame) -> pd.DataFrame:
    """Return a crosswalk DataFrame mapping api_player_id → tm_player_id.

    Columns: api_player_id, tm_player_id (NaN if unmatched), match_method.
    match_method: 'exact_full' | 'exact_last' | 'fuzzy' | 'unmatched'
    """
    api_df = api_players_to_df(api_players)

    # One row per unique API player (DOB is stable across seasons)
    api_unique = (
        api_df.drop_duplicates("api_player_id")[
            ["api_player_id", "full_name_norm", "last_name_norm", "dob"]
        ]
        .copy()
        .reset_index(drop=True)
    )
    api_unique["dob_str"] = api_unique["dob"].dt.strftime("%Y-%m-%d")

    # Build TM lookup tables
    tm = tm_players[["player_id", "name", "last_name", "date_of_birth"]].copy()
    tm = tm.rename(columns={"player_id": "tm_player_id", "date_of_birth": "dob"})
    tm["full_name_norm"] = tm["name"].map(_normalize)
    tm["last_name_norm"] = tm["last_name"].map(_normalize)
    tm["dob_str"] = pd.to_datetime(tm["dob"], errors="coerce").dt.strftime("%Y-%m-%d")

    full_key: dict[tuple, int] = {
        (r.full_name_norm, r.dob_str): r.tm_player_id for r in tm.itertuples()
    }
    last_key: dict[tuple, int] = {}
    for r in tm.itertuples():
        k = (r.last_name_norm, r.dob_str)
        if k not in last_key:
            last_key[k] = r.tm_player_id

    # Fuzzy: group TM candidates by DOB for efficiency
    tm_by_dob: dict[str, list[str]] = tm.groupby("dob_str")["full_name_norm"].apply(list).to_dict()
    tm_id_by_dob_name: dict[tuple, int] = {
        (r.dob_str, r.full_name_norm): r.tm_player_id for r in tm.itertuples()
    }

    results = []
    for row in api_unique.itertuples():
        # 1. Exact: full name + DOB
        tm_id = full_key.get((row.full_name_norm, row.dob_str))
        if tm_id is not None:
            results.append((row.api_player_id, tm_id, "exact_full"))
            continue

        # 2. Exact: last name + DOB (handles abbreviated firstnames)
        tm_id = last_key.get((row.last_name_norm, row.dob_str))
        if tm_id is not None:
            results.append((row.api_player_id, tm_id, "exact_last"))
            continue

        # 3. Fuzzy: token_sort_ratio on full name, restricted to same DOB
        candidates = tm_by_dob.get(row.dob_str, [])
        if candidates:
            best, score, _ = process.extractOne(
                row.full_name_norm, candidates, scorer=fuzz.token_set_ratio
            )
            if score >= _FUZZY_THRESHOLD:
                tm_id = tm_id_by_dob_name[(row.dob_str, best)]
                results.append((row.api_player_id, tm_id, "fuzzy"))
                continue

        results.append((row.api_player_id, None, "unmatched"))

    crosswalk = pd.DataFrame(results, columns=["api_player_id", "tm_player_id", "match_method"])

    total = len(crosswalk)
    matched = crosswalk["tm_player_id"].notna().sum()
    counts = crosswalk["match_method"].value_counts()
    logger.info(
        "Match rate: %d/%d (%.1f%%)",
        matched,
        total,
        100 * matched / total if total else 0,
    )
    logger.info(
        "  exact_full=%d  exact_last=%d  fuzzy=%d  unmatched=%d",
        counts.get("exact_full", 0),
        counts.get("exact_last", 0),
        counts.get("fuzzy", 0),
        counts.get("unmatched", 0),
    )
    return crosswalk


# ---------------------------------------------------------------------------
# Produce the final joined dataset
# ---------------------------------------------------------------------------


def build_joined_dataset(api_players: list[dict], tm: "TransfermarktData") -> pd.DataFrame:  # noqa: F821
    """Join API-Football stats with Transfermarkt market values.

    Returns one row per player-season with performance stats and
    the market value snapped to the end of that season (July).
    Excludes goalkeepers. Drops rows without a market value.
    """
    api_df = api_players_to_df(api_players)
    crosswalk = build_crosswalk(api_players, tm.players)

    # Attach tm_player_id to each api row
    api_df = api_df.merge(crosswalk[["api_player_id", "tm_player_id"]], on="api_player_id", how="left")

    # Exclude goalkeepers
    api_df = api_df[api_df["position"].str.lower() != "goalkeeper"]

    # Snap market value: for season N (e.g. 2022 = 2022/23), use the valuation
    # closest to the end of the season — July 1 of year N+1.
    vals = tm.valuations[["player_id", "date", "market_value_in_eur"]].rename(
        columns={"player_id": "tm_player_id"}
    )
    vals = vals.dropna(subset=["tm_player_id", "date", "market_value_in_eur"])
    vals["date"] = pd.to_datetime(vals["date"])

    # Build unique (api_player_id, season, tm_player_id) combos and snap nearest valuation.
    player_seasons = (
        api_df[api_df["tm_player_id"].notna()][["api_player_id", "season", "tm_player_id"]]
        .drop_duplicates(["api_player_id", "season"])
        .copy()
    )
    player_seasons["target_date"] = player_seasons["season"].apply(
        lambda s: pd.Timestamp(year=int(s) + 1, month=7, day=1)
    )

    vals_sub = vals[vals["tm_player_id"].isin(player_seasons["tm_player_id"].unique())].copy()
    merged = player_seasons.merge(vals_sub, on="tm_player_id", how="left")
    merged["delta"] = (merged["date"] - merged["target_date"]).abs()
    # Drop rows with no valuation, then keep closest per player-season
    merged = merged.dropna(subset=["delta"])
    best_idx = merged.groupby(["api_player_id", "season"])["delta"].idxmin()
    snapped = merged.loc[best_idx, ["api_player_id", "season", "market_value_in_eur"]]

    joined = api_df.merge(snapped, on=["api_player_id", "season"], how="left")
    joined = joined.dropna(subset=["market_value_in_eur"])

    logger.info(
        "Joined dataset: %d player-season-club rows, %d unique player-seasons",
        len(joined),
        joined.drop_duplicates(["api_player_id", "season"]).shape[0],
    )
    return joined


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.ingest import fetch_api_football, load_transfermarkt

    tm = load_transfermarkt()
    api_players = fetch_api_football()

    crosswalk = build_crosswalk(api_players, tm.players)
    print(crosswalk["match_method"].value_counts())
    print("\nSample unmatched API players:")
    api_df = api_players_to_df(api_players)
    unmatched_ids = crosswalk[crosswalk["match_method"] == "unmatched"]["api_player_id"]
    sample = (
        api_df[api_df["api_player_id"].isin(unmatched_ids)][
            ["api_player_id", "firstname", "lastname", "dob"]
        ]
        .drop_duplicates("api_player_id")
        .head(20)
    )
    print(sample.to_string(index=False))
