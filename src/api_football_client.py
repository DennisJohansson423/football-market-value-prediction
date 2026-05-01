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
_REQUEST_DELAY = 0.5  # seconds between live requests to avoid hammering the API


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

    def get_players_page(self, season: int, page: int) -> dict[str, Any]:
        """Fetch one page of PL player stats. season=2022 means the 2022/23 season."""
        return self._get("players", {"league": _PL_LEAGUE_ID, "season": season, "page": page})

    def get_all_players(self, season: int) -> list[dict[str, Any]]:
        """Fetch every page for *season* and return a flat list of player response dicts."""
        first = self.get_players_page(season, page=1)
        total_pages: int = first.get("paging", {}).get("total", 1)
        logger.info("Season %d: %d total pages", season, total_pages)

        players: list[dict[str, Any]] = list(first.get("response", []))
        for page in range(2, total_pages + 1):
            time.sleep(_REQUEST_DELAY)
            data = self.get_players_page(season, page=page)
            players.extend(data.get("response", []))
            logger.info("  page %d/%d — %d players so far", page, total_pages, len(players))

        return players

    # ------------------------------------------------------------------

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        slug = endpoint.replace("/", "_")
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        name = f"{slug}_{param_str}.json" if param_str else f"{slug}.json"
        return self._cache_dir / name

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_file = self._cache_path(endpoint, params)
        if cache_file.exists():
            logger.debug("cache hit  %s", cache_file.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{_BASE_URL}/{endpoint}"
        logger.info("fetching   %s  params=%s", url, params)
        resp = requests.get(url, headers=self._headers, params=params, timeout=30)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("cached     %s", cache_file.name)
        return data
