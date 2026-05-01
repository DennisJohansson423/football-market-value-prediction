"""HTTP client for API-Football v3 with disk-based caching."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://v3.football.api-sports.io"
_PL_LEAGUE_ID = 39
_REQUEST_DELAY = 2.0  # seconds between live requests; free tier rate-limits at ~30 req/min
_RETRY_WAIT = 60.0   # seconds to wait after a 429 before retrying

# All PL team IDs across the 2022/23, 2023/24 and 2024/25 seasons.
PL_TEAM_IDS = [
    33,    # Manchester United
    34,    # Newcastle
    35,    # Bournemouth
    36,    # Fulham
    39,    # Wolves
    40,    # Liverpool
    41,    # Southampton
    42,    # Arsenal
    45,    # Everton
    46,    # Leicester
    47,    # Tottenham
    48,    # West Ham
    49,    # Chelsea
    50,    # Manchester City
    51,    # Brighton
    52,    # Crystal Palace
    55,    # Brentford
    57,    # Ipswich
    62,    # Sheffield Utd
    65,    # Nottingham Forest
    66,    # Aston Villa
    1359,  # Luton
]


class ApiFootballClient:
    """Thin wrapper around API-Football v3 with transparent disk caching."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        api_key = os.getenv("API_FOOTBALL_KEY")
        if not api_key:
            raise EnvironmentError("API_FOOTBALL_KEY not set in environment / .env")
        self._headers = {"x-apisports-key": api_key}
        self._cache_dir = cache_dir or Path(__file__).parent.parent / "data" / "raw" / "api_football"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_status(self) -> dict[str, Any]:
        """Return API quota info. Free endpoint — does not count against the daily limit."""
        return self._get("status", {})

    def get_players_by_team(self, team_id: int, season: int, page: int = 1) -> dict[str, Any]:
        """Fetch one page of player stats for a specific team and season."""
        return self._get("players", {"team": team_id, "season": season, "page": page})

    def get_all_players_for_team(self, team_id: int, season: int) -> list[dict[str, Any]]:
        """Fetch all pages for a team/season and return a flat list of player dicts."""
        first = self.get_players_by_team(team_id, season, page=1)
        total_pages: int = min(first.get("paging", {}).get("total", 1), 3)  # free tier cap
        players: list[dict[str, Any]] = list(first.get("response", []))
        for page in range(2, total_pages + 1):
            time.sleep(_REQUEST_DELAY)
            data = self.get_players_by_team(team_id, season, page=page)
            players.extend(data.get("response", []))
        return players

    def get_all_players(self, season: int, team_ids: list[int] | None = None) -> list[dict[str, Any]]:
        """Fetch all PL players for *season* by querying each team individually.

        Querying by team avoids the free-tier page-3 cap on the league endpoint.
        """
        teams = team_ids or PL_TEAM_IDS
        all_players: list[dict[str, Any]] = []
        for i, team_id in enumerate(teams):
            players = self.get_all_players_for_team(team_id, season)
            # Keep only PL stats — a player can appear in multiple competitions
            pl_players = [p for p in players if _has_pl_stats(p)]
            all_players.extend(pl_players)
            logger.info(
                "  [%d/%d] team %d — %d players (PL), %d total so far",
                i + 1,
                len(teams),
                team_id,
                len(pl_players),
                len(all_players),
            )
            if i < len(teams) - 1:
                time.sleep(_REQUEST_DELAY)
        return all_players

    # ------------------------------------------------------------------

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        slug = endpoint.replace("/", "_")
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        name = f"{slug}_{param_str}.json" if param_str else f"{slug}.json"
        return self._cache_dir / name

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_file = self._cache_path(endpoint, params)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            # Don't serve cached error responses — delete and re-fetch.
            if cached.get("errors"):
                logger.warning("Stale error response in cache, deleting: %s", cache_file.name)
                cache_file.unlink()
            else:
                logger.debug("cache hit  %s", cache_file.name)
                return cached

        url = f"{_BASE_URL}/{endpoint}"
        logger.info("fetching   %s  params=%s", url, params)
        for attempt in range(3):
            resp = requests.get(url, headers=self._headers, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("429 rate limit — waiting %.0fs (attempt %d/3)", _RETRY_WAIT, attempt + 1)
                time.sleep(_RETRY_WAIT)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        if data.get("errors"):
            raise RuntimeError(f"API error for {params}: {data['errors']}")

        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("cached     %s", cache_file.name)
        return data


def _has_pl_stats(player_rec: dict[str, Any]) -> bool:
    """Return True if the player has at least one statistics entry for the PL."""
    return any(s["league"]["id"] == _PL_LEAGUE_ID for s in player_rec.get("statistics", []))
