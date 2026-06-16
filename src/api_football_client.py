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
_REQUEST_DELAY = 7.0  # seconds between live requests; free tier hard-limits at 10 req/min
_RETRY_WAIT = 60.0   # seconds to wait after a 429 before retrying

# Hardcoded PL team IDs — used as fallback to avoid an extra API call for the default league.
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
        self._cache_dir = (
            cache_dir or Path(__file__).parent.parent / "data" / "raw" / "api_football"
        )
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
            data = self.get_players_by_team(team_id, season, page=page)
            players.extend(data.get("response", []))
        return players

    def get_all_players(
        self, season: int, team_ids: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all PL players for *season* by querying each team individually.

        Querying by team avoids the free-tier page-3 cap on the league endpoint.
    def get_teams_for_league(self, league_id: int, season: int) -> list[int]:
        """Return all team IDs for a league/season (result is cached)."""
        data = self._get("teams", {"league": league_id, "season": season})
        return [t["team"]["id"] for t in data.get("response", [])]

    def get_all_players(
        self,
        season: int,
        league_id: int = _PL_LEAGUE_ID,
        team_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all players for *season* in *league_id* by querying each team.

        team_ids: override the auto-fetched list (useful for testing or PL which has a hardcoded list).
        Each returned record gets '_league_id' set and its statistics filtered to the target league only.
        """
        if team_ids is None:
            # Use hardcoded list for PL to save one API call; fetch dynamically for other leagues.
            team_ids = PL_TEAM_IDS if league_id == _PL_LEAGUE_ID else self.get_teams_for_league(league_id, season)

        all_players: list[dict[str, Any]] = []
        for i, team_id in enumerate(team_ids):
            players = self.get_all_players_for_team(team_id, season)
            league_players = [p for p in players if _has_league_stats(p, league_id)]
            for p in league_players:
                # Trim statistics to this league only so downstream aggregation is correct.
                p["statistics"] = [s for s in p["statistics"] if s["league"]["id"] == league_id]
                p["_league_id"] = league_id
            all_players.extend(league_players)
            logger.info(
                "  [%d/%d] team %d — %d players (league %d), %d total so far",
                i + 1,
                len(team_ids),
                team_id,
                len(league_players),
                league_id,
                len(all_players),
            )
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
        time.sleep(_REQUEST_DELAY)
        for attempt in range(3):
            resp = requests.get(url, headers=self._headers, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning(
                    "429 rate limit — waiting %.0fs (attempt %d/3)", _RETRY_WAIT, attempt + 1
                )
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


def _has_league_stats(player_rec: dict[str, Any], league_id: int) -> bool:
    """Return True if the player has at least one statistics entry for the given league."""
    return any(s["league"]["id"] == league_id for s in player_rec.get("statistics", []))
