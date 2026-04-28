import pandas as pd
import numpy as np

class FeatureEngineering:
    """Create all features including form, injuries, weather, ref bias etc."""
    
    def __init__(self, data_collector):
        self.dc = data_collector
    
    def create_match_features(self, fixtures_df):
        features = []
        for _, fixture in fixtures_df.iterrows():
            f = {}
            home_team_id = fixture['teams']['home']['id']
            away_team_id = fixture['teams']['away']['id']
            
            f['home_team_id'] = home_team_id
            f['away_team_id'] = away_team_id
            f['competition_id'] = fixture['league']['id']
            
            # Team form and goal stats
            home_stats = self.dc.get_team_stats(home_team_id, fixture['league']['id'], fixture['league']['season'])
            away_stats = self.dc.get_team_stats(away_team_id, fixture['league']['id'], fixture['league']['season'])
            
            if home_stats and away_stats:
                f['home_goals_scored_avg'] = float(home_stats['goals']['for']['average']['total'])
                f['home_goals_conceded_avg'] = float(home_stats['goals']['against']['average']['total'])
                f['away_goals_scored_avg'] = float(away_stats['goals']['for']['average']['total'])
                f['away_goals_conceded_avg'] = float(away_stats['goals']['against']['average']['total'])
                
                # Form features (last 5 matches)
                home_form = self.get_recent_form(home_team_id, fixture['league']['id'], fixture['league']['season'], 5)
                away_form = self.get_recent_form(away_team_id, fixture['league']['id'], fixture['league']['season'], 5)
                for key, value in home_form.items():
                    f[f'home_{key}'] = value
                for key, value in away_form.items():
                    f[f'away_{key}'] = value
                
                # Injuries and suspensions
                home_injuries = self.dc.get_injuries(home_team_id, fixture['league']['season'])
                away_injuries = self.dc.get_injuries(away_team_id, fixture['league']['season'])
                f['home_injury_count'] = len(home_injuries) if home_injuries else 0
                f['away_injury_count'] = len(away_injuries) if away_injuries else 0
                
                # Head to head
                h2h = self.dc.get_h2h(home_team_id, away_team_id)
                f['h2h_home_wins'] = len([m for m in h2h if m['teams']['home']['winner']])
                f['h2h_away_wins'] = len([m for m in h2h if m['teams']['away']['winner']])
                f['h2h_draws'] = len([m for m in h2h if m['teams']['home']['winner'] is None])
                f['h2h_btts'] = len([m for m in h2h if m['goals']['home'] > 0 and m['goals']['away'] > 0])
            
            # Referee Bias (if available)
            if fixture.get('fixture', {}).get('referee'):
                ref_stats = self.get_referee_stats(fixture['fixture']['referee'])
                f['ref_home_wins'] = ref_stats['home_wins']
                f['ref_away_wins'] = ref_stats['away_wins']
            
            # Weather conditions (placeholder)
            weather = self.dc.get_weather(fixture['fixture']['id'])
            f['temperature'] = weather.get('temperature', 20)
            f['humidity'] = weather.get('humidity', 50)
            f['rain'] = weather.get('rain', 0)
            
            features.append(f)
        return pd.DataFrame(features)
    
    def get_recent_form(self, team_id, league_id, season, n_matches):
        """Calculate recent form metrics."""
        fixtures = self.dc.get_fixtures(league_id, season)
        team_fixtures = fixtures[fixtures['teams'].apply(
            lambda x: x['home']['id'] == team_id or x['away']['id'] == team_id
        )].head(n_matches)
        # Placeholder logic for points, goals etc.
        points = 0
        return {'points_last5': points, 'goals_scored_last5': 0, 'goals_conceded_last5': 0}
    
    def get_referee_stats(self, referee_name):
        """Get referee historical statistics. Placeholder for actual data fetching."""
        return {'home_wins': 0, 'away_wins': 0}
