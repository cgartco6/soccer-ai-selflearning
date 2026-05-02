import sys
import os
import pytest
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collector import DataCollector
from src.features import FeatureEngineer

class MockCollector(DataCollector):
    def __init__(self):
        super().__init__()
        self.api_key = "mock"

    def get_team_stats(self, team_id, league_id, season):
        return {"goals": {"for": {"average": {"total": 1.5}}, "against": {"average": {"total": 1.2}}},
                "home": {"wins": 8, "played": 14}}

    def get_recent_form(self, team_id, league_id, season, n=5):
        return {"points": 9, "goals_scored": 6, "goals_conceded": 4}

    def get_head2head(self, team1_id, team2_id, limit=5):
        return [{"winner": "home", "home_goals": 2, "away_goals": 1}]

    def get_injuries(self, team_id, season):
        return [{"player": {"status": "out"}}]

    def get_weather(self, city):
        return {"temp": 18, "rain": 0}

    def get_player_form(self, team_id, season, limit=3):
        return [{"rating": 7.5, "goals": 2}]

    def get_coach_form(self, team_id, season):
        return {"win_rate_last5": 0.6}

    def get_transfers_impact(self, team_id, season):
        return 0.2

    def get_pitch_factor(self, stadium_name):
        return 1.0

    def get_referee_bias(self, referee_id):
        return 0.55

def test_build_features():
    mock = MockCollector()
    engineer = FeatureEngineer(mock)
    fixtures = [{
        "fixture": {"id": 1001, "venue": {"name": "Stadium", "city": "London"}, "referee": {"id": 99}},
        "teams": {"home": {"id": 50}, "away": {"id": 60}},
        "league": {"id": 39, "name": "Premier League"}
    }]
    df = engineer.build_features(fixtures, 39, 2026)
    assert not df.empty
    assert "home_player_rating" in df.columns
