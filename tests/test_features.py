"""Tests for src/features.py — _aggregate_multi_club."""

import pandas as pd
import pytest

from src.features import _SUM_COLS, _aggregate_multi_club


def _player_row(**overrides):
    """Return a dict with all columns needed by _aggregate_multi_club, allowing overrides."""
    base = {
        "api_player_id": 1,
        "season": 2022,
        "firstname": "John",
        "lastname": "Doe",
        "dob": "1995-01-01",
        "nationality": "English",
        "position": "Attacker",
        "tm_player_id": 101,
        "market_value_in_eur": 5_000_000.0,
        # counting stats
        "appearances": 10,
        "minutes": 900,
        "goals": 5,
        "assists": 3,
        "yellow_cards": 1,
        "red_cards": 0,
        "shots_total": 20,
        "shots_on": 8,
        "passes_total": 200,
        "passes_key": 10,
        "tackles": 5,
        "interceptions": 3,
        "blocks": 1,
        "duels_total": 30,
        "duels_won": 15,
        "dribbles_attempts": 10,
        "dribbles_success": 6,
        "fouls_drawn": 4,
        "fouls_committed": 2,
        # weighted column
        "passes_accuracy": 80.0,
    }
    base.update(overrides)
    return base


class TestAggregateMultiClub:
    def test_single_club_passthrough(self):
        """A player at one club is returned unchanged for counting stats."""
        df = pd.DataFrame([_player_row()])
        result = _aggregate_multi_club(df)
        assert len(result) == 1
        assert result.iloc[0]["goals"] == 5
        assert result.iloc[0]["minutes"] == 900
        assert result.iloc[0]["appearances"] == 10

    def test_two_clubs_sums_counting_stats(self):
        """A mid-season transfer player has counting stats summed across clubs."""
        row1 = _player_row(goals=5, assists=3, minutes=900, passes_total=200,
                           passes_accuracy=80.0)
        row2 = _player_row(goals=3, assists=1, minutes=450, passes_total=100,
                           passes_accuracy=70.0)
        df = pd.DataFrame([row1, row2])
        result = _aggregate_multi_club(df)
        assert len(result) == 1
        assert result.iloc[0]["goals"] == 8
        assert result.iloc[0]["assists"] == 4
        assert result.iloc[0]["minutes"] == 1350

    def test_passes_accuracy_weighted_average(self):
        """passes_accuracy is a weighted mean by passes_total, not a simple mean."""
        # Club A: 200 passes at 80% → 160 accurate passes
        # Club B: 100 passes at 70% → 70 accurate passes
        # Weighted avg = (160 + 70) / (200 + 100) = 230/300 ≈ 76.67%
        row1 = _player_row(passes_total=200, passes_accuracy=80.0)
        row2 = _player_row(passes_total=100, passes_accuracy=70.0)
        df = pd.DataFrame([row1, row2])
        result = _aggregate_multi_club(df)
        expected = (200 * 80.0 + 100 * 70.0) / (200 + 100)
        assert result.iloc[0]["passes_accuracy"] == pytest.approx(expected)

    def test_passes_accuracy_nan_when_zero_passes(self):
        """Guard: if passes_total is 0, weighted avg should be NaN, not a division error."""
        row = _player_row(passes_total=0, passes_accuracy=0.0)
        df = pd.DataFrame([row])
        result = _aggregate_multi_club(df)
        assert pd.isna(result.iloc[0]["passes_accuracy"])

    def test_two_players_two_seasons(self):
        """Multiple players and seasons produce the correct number of output rows."""
        rows = [
            _player_row(api_player_id=1, season=2022),
            _player_row(api_player_id=1, season=2023),
            _player_row(api_player_id=2, season=2022),
        ]
        df = pd.DataFrame(rows)
        result = _aggregate_multi_club(df)
        assert len(result) == 3

    def test_output_columns_include_all_sum_cols(self):
        df = pd.DataFrame([_player_row()])
        result = _aggregate_multi_club(df)
        for col in _SUM_COLS:
            assert col in result.columns, f"Missing sum column: {col}"

    def test_meta_columns_preserved(self):
        df = pd.DataFrame([_player_row(firstname="Alice", lastname="Smith", tm_player_id=999)])
        result = _aggregate_multi_club(df)
        assert result.iloc[0]["firstname"] == "Alice"
        assert result.iloc[0]["lastname"] == "Smith"
        assert result.iloc[0]["tm_player_id"] == 999

    def test_yellow_card_sum(self):
        """Cards are summed, not averaged — common mistake to guard against."""
        row1 = _player_row(yellow_cards=2, passes_total=200, passes_accuracy=80.0)
        row2 = _player_row(yellow_cards=1, passes_total=100, passes_accuracy=80.0)
        df = pd.DataFrame([row1, row2])
        result = _aggregate_multi_club(df)
        assert result.iloc[0]["yellow_cards"] == 3

    def test_result_has_one_row_per_player_season(self):
        """Three rows for the same player-season collapse to one."""
        rows = [_player_row(goals=2), _player_row(goals=3), _player_row(goals=1)]
        df = pd.DataFrame(rows)
        result = _aggregate_multi_club(df)
        assert len(result) == 1
        assert result.iloc[0]["goals"] == 6
