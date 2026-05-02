import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collector import DataCollector
from src.features import FeatureEngineer


class MockCollector(DataCollector):
    """Mock collector that returns fixed data without API calls."""

    def __init__(self):
        super().__init__()
        self.api_key = "mock"

    def get_team_stats(self, team_id, league_id, season):
        return {
            "goals": {"for": {"average": {"total": 1.5}}, "against": {"average": {"total": 1.2}}},
            "home": {"wins": 8, "played": 14},
        }

    def get_recent_form(self, team_id, league_id, season, n=5):
        return {"points": 9, "goals_scored": 6, "goals_conceded": 4}

    def get_head2head(self, team1_id, team2_id, limit=5):
        return [
            {"winner": "home", "home_goals": 2, "away_goals": 1},
            {"winner": "draw", "home_goals": 1, "away_goals": 1},
            {"winner": "away", "home_goals": 0, "away_goals": 2},
        ]

    def get_injuries(self, team_id, season):
        return [{"player": {"status": "out"}} for _ in range(2)]

    def get_weather(self, city_name):
        return {"temp": 18, "humidity": 65, "wind": 3, "rain": 0}


@pytest.fixture
def sample_fixtures():
    return [
        {
            "fixture": {"id": 1001, "venue": {"city": "London"}},
            "teams": {"home": {"id": 50}, "away": {"id": 60}},
            "league": {"id": 39, "name": "Premier League"},
        },
        {
            "fixture": {"id": 1002, "venue": {"city": "Manchester"}},
            "teams": {"home": {"id": 51}, "away": {"id": 61}},
            "league": {"id": 39, "name": "Premier League"},
        },
    ]


def test_build_features_returns_dataframe(sample_fixtures):
    mock = MockCollector()
    engineer = FeatureEngineer(mock)
    df = engineer.build_features(sample_fixtures, league_id=39, season=2026)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_fixtures)
    # Check a few expected columns
    expected_cols = [
        "fixture_id", "home_team", "away_team", "home_goals_avg", "away_goals_avg",
        "home_points_last5", "away_points_last5", "h2h_home_wins", "h2h_draws",
        "home_injuries", "temperature", "rain", "home_advantage", "is_derby"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_build_features_handles_missing_data():
    """If API returns None, features should have safe defaults (no crash)."""
    class EmptyMock(MockCollector):
        def get_team_stats(self, *args, **kwargs):
            return None
        def get_recent_form(self, *args, **kwargs):
            return {"points": 0, "goals_scored": 0, "goals_conceded": 0}

    mock = EmptyMock()
    engineer = FeatureEngineer(mock)
    fixtures = [{
        "fixture": {"id": 999, "venue": {"city": "Unknown"}},
        "teams": {"home": {"id": 999}, "away": {"id": 888}},
        "league": {"id": 39, "name": "Test"}
    }]
    df = engineer.build_features(fixtures, league_id=39, season=2026)
    # Should not raise exception, and all numeric columns should be filled
    assert df["home_goals_avg"].iloc[0] != 0  # default value exists
    assert df["home_points_last5"].iloc[0] == 0


def test_add_target_calculates_correctly():
    df = pd.DataFrame({
        "fixture_id": [1, 2],
        "actual_home_goals": [3, 1],
        "actual_away_goals": [1, 2],
    })
    actual_results = {1: (3, 1), 2: (1, 2)}
    engineer = FeatureEngineer(MockCollector())  # doesn't need real collector
    df = engineer.add_target(df, actual_results)
    assert df["target"].iloc[0] == 0  # home win
    assert df["target"].iloc[1] == 2  # away win
    assert df["btts"].iloc[0] == 1
    assert df["btts"].iloc[1] == 1
