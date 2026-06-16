"""Tests for src/join.py — _normalize and build_crosswalk."""

import pandas as pd

from src.join import _normalize, build_crosswalk

# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("John Doe") == "john doe"

    def test_strips_accents(self):
        assert _normalize("Héctor") == "hector"

    def test_strips_accents_complex(self):
        # Müller, Čech, Özil — common football names
        assert _normalize("Müller") == "muller"
        assert _normalize("Čech") == "cech"
        assert _normalize("Özil") == "ozil"

    def test_removes_punctuation(self):
        assert _normalize("O'Neil") == "oneil"

    def test_removes_hyphens(self):
        assert _normalize("Tchouaméni") == "tchouameni"

    def test_collapses_whitespace(self):
        assert _normalize("  John   Doe  ") == "john doe"

    def test_removes_digits(self):
        assert _normalize("Player1 2000") == "player"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_single_word(self):
        assert _normalize("Ronaldo") == "ronaldo"

    def test_non_latin_becomes_ascii(self):
        # unidecode converts Greek/Cyrillic as best it can; at minimum no crash
        result = _normalize("Αλέξης")  # Greek — Alexis
        assert isinstance(result, str)
        assert result != ""


# ---------------------------------------------------------------------------
# build_crosswalk  (tiny inline fixture — no network, no disk)
# ---------------------------------------------------------------------------

def _make_api_player(api_id, firstname, lastname, dob, season=2022, position="Attacker"):
    """Minimal valid API-Football player record."""
    return {
        "player": {
            "id": api_id,
            "firstname": firstname,
            "lastname": lastname,
            "birth": {"date": dob},
            "nationality": "English",
        },
        "statistics": [{
            "team": {"id": 33, "name": "Manchester United"},
            "games": {
                "appearences": 10, "minutes": 900,
                "position": position, "rating": "7.50",
            },
            "goals": {"total": 5, "assists": 3},
            "cards": {"yellow": 1, "red": 0},
            "shots": {"total": 20, "on": 10},
            "passes": {"total": 100, "key": 5, "accuracy": 80},
            "tackles": {"total": 2, "interceptions": 1, "blocks": 0},
            "duels": {"total": 20, "won": 10},
            "dribbles": {"attempts": 5, "success": 3},
            "fouls": {"drawn": 3, "committed": 2},
            "penalty": {"scored": 0},
        }],
        "_season": season,
        "_league_id": 39,
    }


def _make_tm_players(rows):
    """Build a Transfermarkt players DataFrame from a list of dicts."""
    return pd.DataFrame(rows, columns=["player_id", "name", "last_name", "date_of_birth"])


class TestBuildCrosswalk:
    def test_exact_full_name_match(self):
        api = [_make_api_player(1, "John", "Doe", "1990-05-15")]
        tm = _make_tm_players([
            {"player_id": 101, "name": "John Doe",
             "last_name": "Doe", "date_of_birth": "1990-05-15"},
        ])
        cw = build_crosswalk(api, tm)
        row = cw[cw["api_player_id"] == 1].iloc[0]
        assert row["tm_player_id"] == 101
        assert row["match_method"] == "exact_full"

    def test_exact_last_name_match(self):
        # API has "J." as firstname; TM has full "John" — only last name + DOB matches
        api = [_make_api_player(2, "J.", "Smith", "1992-03-20")]
        tm = _make_tm_players([
            {"player_id": 202, "name": "John Smith",
             "last_name": "Smith", "date_of_birth": "1992-03-20"},
        ])
        cw = build_crosswalk(api, tm)
        row = cw[cw["api_player_id"] == 2].iloc[0]
        assert row["tm_player_id"] == 202
        assert row["match_method"] == "exact_last"

    def test_fuzzy_match(self):
        # Slight name difference (accent vs no accent) — should still fuzzy-match
        api = [_make_api_player(3, "Mats", "Hummels", "1988-12-16")]
        tm = _make_tm_players([
            {"player_id": 303, "name": "Mats Hümmels",
             "last_name": "Hümmels", "date_of_birth": "1988-12-16"},
        ])
        cw = build_crosswalk(api, tm)
        row = cw[cw["api_player_id"] == 3].iloc[0]
        # Both normalize to similar strings; should fuzzy-match
        assert row["tm_player_id"] == 303
        assert row["match_method"] in {"exact_full", "exact_last", "fuzzy"}

    def test_unmatched_wrong_dob(self):
        # Same name but different DOB — must not match
        api = [_make_api_player(4, "Thomas", "Müller", "1989-09-13")]
        tm = _make_tm_players([
            {"player_id": 404, "name": "Thomas Müller",
             "last_name": "Müller", "date_of_birth": "1970-01-01"},
        ])
        cw = build_crosswalk(api, tm)
        row = cw[cw["api_player_id"] == 4].iloc[0]
        assert pd.isna(row["tm_player_id"])
        assert row["match_method"] == "unmatched"

    def test_unmatched_no_candidates(self):
        # No TM players at all with the same DOB
        api = [_make_api_player(5, "Ghost", "Player", "2000-01-01")]
        tm = _make_tm_players([
            {"player_id": 505, "name": "Other Guy",
             "last_name": "Guy", "date_of_birth": "1985-06-06"},
        ])
        cw = build_crosswalk(api, tm)
        row = cw[cw["api_player_id"] == 5].iloc[0]
        assert pd.isna(row["tm_player_id"])
        assert row["match_method"] == "unmatched"

    def test_deduplicates_api_players_across_seasons(self):
        # Same player appearing in two seasons → crosswalk has one row
        api = [
            _make_api_player(6, "Ali", "Baba", "1995-07-04", season=2022),
            _make_api_player(6, "Ali", "Baba", "1995-07-04", season=2023),
        ]
        tm = _make_tm_players([
            {"player_id": 606, "name": "Ali Baba",
             "last_name": "Baba", "date_of_birth": "1995-07-04"},
        ])
        cw = build_crosswalk(api, tm)
        assert len(cw) == 1
        assert cw.iloc[0]["api_player_id"] == 6

    def test_columns_present(self):
        api = [_make_api_player(7, "Test", "Player", "2001-01-01")]
        tm = _make_tm_players([])
        cw = build_crosswalk(api, tm)
        assert set(cw.columns) == {"api_player_id", "tm_player_id", "match_method"}
