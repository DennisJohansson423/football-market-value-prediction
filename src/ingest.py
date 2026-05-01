"""Load raw data from API-Football (cached) and Transfermarkt CSVs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.api_football_client import ApiFootballClient

logger = logging.getLogger(__name__)

_TM_DIR = Path(__file__).parent.parent / "data" / "raw" / "transfermarkt"
_PL_COMPETITION_ID = "GB1"
_SEASONS = [2022, 2023, 2024]


@dataclass
class TransfermarktData:
    players: pd.DataFrame
    valuations: pd.DataFrame
    clubs: pd.DataFrame
    transfers: pd.DataFrame


def load_transfermarkt(tm_dir: Path | None = None) -> TransfermarktData:
    """Load and PL-filter the Transfermarkt CSVs."""
    d = tm_dir or _TM_DIR

    clubs = pd.read_csv(d / "clubs.csv")
    pl_clubs = clubs[clubs["domestic_competition_id"] == _PL_COMPETITION_ID].copy()
    pl_club_ids = set(pl_clubs["club_id"])
    logger.info("PL clubs: %d", len(pl_clubs))

    valuations = pd.read_csv(d / "player_valuations.csv", parse_dates=["date"])
    pl_valuations = valuations[
        valuations["player_club_domestic_competition_id"] == _PL_COMPETITION_ID
    ].copy()
    logger.info("PL valuations rows: %d", len(pl_valuations))

    # Use valuation history to find all players who ever played in the PL,
    # not just those currently at a PL club.
    pl_player_ids = set(pl_valuations["player_id"])
    players = pd.read_csv(d / "players.csv", parse_dates=["date_of_birth"])
    pl_players = players[players["player_id"].isin(pl_player_ids)].copy()
    logger.info("PL players: %d", len(pl_players))

    transfers = pd.read_csv(d / "transfers.csv", parse_dates=["transfer_date"])
    pl_transfers = transfers[
        transfers["from_club_id"].isin(pl_club_ids) | transfers["to_club_id"].isin(pl_club_ids)
    ].copy()
    logger.info("PL transfers: %d", len(pl_transfers))

    return TransfermarktData(
        players=pl_players,
        valuations=pl_valuations,
        clubs=pl_clubs,
        transfers=pl_transfers,
    )


def fetch_api_football(seasons: list[int] | None = None) -> list[dict]:
    """Fetch all player stats from API-Football for the given seasons (cached after first run).

    Each returned dict is the raw API response item with an added '_season' key.
    Shape: {"player": {...}, "statistics": [...], "_season": 2022}
    """
    seasons = seasons or _SEASONS
    client = ApiFootballClient()
    all_players: list[dict] = []
    for season in seasons:
        players = client.get_all_players(season)
        for p in players:
            p["_season"] = season
        all_players.extend(players)
        logger.info("Season %d: %d player-records", season, len(players))
    logger.info("Total API-Football records: %d across %d seasons", len(all_players), len(seasons))
    return all_players


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("=== Transfermarkt ===")
    tm = load_transfermarkt()
    logger.info(
        "clubs=%d  players=%d  valuations=%d  transfers=%d",
        len(tm.clubs),
        len(tm.players),
        len(tm.valuations),
        len(tm.transfers),
    )

    logger.info("=== API-Football ===")
    api_players = fetch_api_football()
    logger.info("Done. %d total player-season records.", len(api_players))
