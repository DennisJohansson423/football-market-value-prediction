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

# Maps a human-readable name → (API-Football league_id, Transfermarkt competition_id)
LEAGUE_CONFIG: dict[str, tuple[int, str]] = {
    "PL":         (39,  "GB1"),  # Premier League
    "LaLiga":     (140, "ES1"),  # La Liga
    "Bundesliga": (78,  "L1"),   # Bundesliga
    "SerieA":     (135, "IT1"),  # Serie A
    "Ligue1":     (61,  "FR1"),  # Ligue 1
}


@dataclass
class TransfermarktData:
    players: pd.DataFrame
    valuations: pd.DataFrame
    clubs: pd.DataFrame
    transfers: pd.DataFrame


def load_transfermarkt(
    tm_dir: Path | None = None,
    competition_ids: list[str] | None = None,
) -> TransfermarktData:
    """Load and filter the Transfermarkt CSVs to the given competitions.

    competition_ids: list of Transfermarkt competition IDs, e.g. ["GB1", "ES1", "L1"].
                     Defaults to Premier League only ["GB1"].
    """
    d = tm_dir or _TM_DIR
    comp_ids = competition_ids or [_PL_COMPETITION_ID]

    clubs = pd.read_csv(d / "clubs.csv")
    target_clubs = clubs[clubs["domestic_competition_id"].isin(comp_ids)].copy()
    target_club_ids = set(target_clubs["club_id"])
    logger.info("Clubs (%s): %d", "+".join(comp_ids), len(target_clubs))

    valuations = pd.read_csv(d / "player_valuations.csv", parse_dates=["date"])
    target_valuations = valuations[
        valuations["player_club_domestic_competition_id"].isin(comp_ids)
    ].copy()
    logger.info("Valuations rows: %d", len(target_valuations))

    # Use valuation history to find all players who ever played in any target league.
    target_player_ids = set(target_valuations["player_id"])
    players = pd.read_csv(d / "players.csv", parse_dates=["date_of_birth"])
    target_players = players[players["player_id"].isin(target_player_ids)].copy()
    logger.info("Players: %d", len(target_players))

    transfers = pd.read_csv(d / "transfers.csv", parse_dates=["transfer_date"])
    target_transfers = transfers[
        transfers["from_club_id"].isin(target_club_ids) | transfers["to_club_id"].isin(target_club_ids)
    ].copy()
    logger.info("Transfers: %d", len(target_transfers))

    return TransfermarktData(
        players=target_players,
        valuations=target_valuations,
        clubs=target_clubs,
        transfers=target_transfers,
    )


def fetch_api_football(
    seasons: list[int] | None = None,
    league_ids: list[int] | None = None,
) -> list[dict]:
    """Fetch all player stats from API-Football (cached after first run).

    seasons:    list of season start years, e.g. [2022, 2023, 2024].
    league_ids: list of API-Football league IDs, e.g. [39, 140, 78, 135].
                Defaults to Premier League only [39].

    Each returned dict has '_season' and '_league_id' keys added.
    Shape: {"player": {...}, "statistics": [...], "_season": 2022, "_league_id": 39}
    """
    seasons = seasons or _SEASONS
    league_ids = league_ids or [39]
    client = ApiFootballClient()
    all_players: list[dict] = []
    for season in seasons:
        for league_id in league_ids:
            players = client.get_all_players(season, league_id=league_id)
            for p in players:
                p["_season"] = season
            all_players.extend(players)
            logger.info("Season %d league %d: %d player-records", season, league_id, len(players))
    logger.info("Total API-Football records: %d across %d seasons × %d leagues",
                len(all_players), len(seasons), len(league_ids))
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
