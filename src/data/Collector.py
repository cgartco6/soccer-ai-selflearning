import pandas as pd
import requests
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_football_key = os.getenv('API_FOOTBALL_KEY')
        self.pulsescore_key = os.getenv('PULSESCORE_API_KEY')
        # Base URL for API-FOOTBALL (replace with actual endpoint)
        self.base_url = 'https://v3.football.api-sports.io/'
        self.conn = sqlite3.connect('data/soccer.db')
        
    def _make_request(self, endpoint, params=None):
        headers = {'x-apisports-key': self.api_football_key}
        try:
            response = requests.get(f'{self.base_url}{endpoint}', headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Data fetch error for {endpoint}: {e}")
            return None

    def get_fixtures(self, league_id, season):
        """Get fixtures for a specific league and season."""
        # Example for Premier League (league_id=39)
        params = {'league': league_id, 'season': season}
        data = self._make_request('fixtures', params)
        if data and data['response']:
            return pd.DataFrame(data['response'])
        return pd.DataFrame()

    def get_team_stats(self, team_id, league_id, season):
        """Get detailed team statistics."""
        params = {'team': team_id, 'league': league_id, 'season': season}
        data = self._make_request('teams/statistics', params)
        return data['response'] if data else None
    
    def get_player_data(self, team_id, season):
        """Get player names, positions, and basic stats."""
        params = {'team': team_id, 'season': season}
        data = self._make_request('players', params)
        if data and data['response']:
            return pd.DataFrame(data['response'])
        return pd.DataFrame()
    
    def get_injuries(self, team_id, season):
        """Get team injuries and suspensions."""
        params = {'team': team_id, 'season': season}
        data = self._make_request('injuries', params)
        return data['response'] if data else []
    
    def get_h2h(self, team1_id, team2_id):
        """Get head-to-head history."""
        params = {'h2h': f'{team1_id}-{team2_id}'}
        data = self._make_request('fixtures/headtohead', params)
        if data and data['response']:
            return data['response']
        return []
    
    def fetch_odds(self, fixture_id):
        """Fetch odds for a specific fixture."""
        params = {'fixture': fixture_id}
        data = self._make_request('odds', params)
        if data and data['response']:
            return data['response']
        return []
    
    def get_weather(self, match_id):
        """Fetch weather data for match location (if available via API-Football)."""
        # Note: Weather data requires a specific endpoint; this is a placeholder.
        print(f"Weather data for fixture {match_id} would be fetched here.")
        return {}
