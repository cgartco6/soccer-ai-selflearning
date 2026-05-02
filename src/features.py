import pandas as pd
import numpy as np
from .collector import DataCollector

class FeatureEngineer:
    def __init__(self, collector: DataCollector):
        self.collector = collector

    def build_features(self, fixtures, league_id, season):
        rows = []
        for f in fixtures:
            home = f["teams"]["home"]["id"]
            away = f["teams"]["away"]["id"]
            home_stats = self.collector.get_team_stats(home, league_id, season)
            away_stats = self.collector.get_team_stats(away, league_id, season)

            row = {
                "fixture_id": f["fixture"]["id"],
                "home_team": home,
                "away_team": away,
                "league_id": league_id,
            }

            # Goals averages
            if home_stats:
                row["home_goals_avg"] = float(home_stats["goals"]["for"]["average"]["total"] or 1.2)
                row["home_conceded_avg"] = float(home_stats["goals"]["against"]["average"]["total"] or 1.3)
            else:
                row["home_goals_avg"] = 1.2
                row["home_conceded_avg"] = 1.3
            if away_stats:
                row["away_goals_avg"] = float(away_stats["goals"]["for"]["average"]["total"] or 1.1)
                row["away_conceded_avg"] = float(away_stats["goals"]["against"]["average"]["total"] or 1.4)
            else:
                row["away_goals_avg"] = 1.1
                row["away_conceded_avg"] = 1.4

            # Recent form
            home_form = self.collector.get_recent_form(home, league_id, season, 5)
            away_form = self.collector.get_recent_form(away, league_id, season, 5)
            row["home_points_last5"] = home_form["points"]
            row["away_points_last5"] = away_form["points"]
            row["home_goals_last5"] = home_form["goals_scored"]
            row["away_goals_last5"] = away_form["goals_scored"]

            # H2H
            h2h = self.collector.get_head2head(home, away, 5)
            row["h2h_home_wins"] = sum(1 for m in h2h if m["winner"] == "home")
            row["h2h_draws"] = sum(1 for m in h2h if m["winner"] == "draw")
            row["h2h_btts_rate"] = sum(1 for m in h2h if m["home_goals"] > 0 and m["away_goals"] > 0) / max(1, len(h2h))

            # Injuries
            home_inj = self.collector.get_injuries(home, season)
            away_inj = self.collector.get_injuries(away, season)
            row["home_injuries"] = len([i for i in home_inj if i["player"]["status"] == "out"])
            row["away_injuries"] = len([i for i in away_inj if i["player"]["status"] == "out"])

            # Weather
            city = f["fixture"].get("venue", {}).get("city", "London")
            weather = self.collector.get_weather(city)
            row["temperature"] = weather.get("temp", 15)
            row["rain"] = weather.get("rain", 0)

            # Home advantage
            if home_stats:
                row["home_advantage"] = home_stats["home"]["wins"] / max(1, home_stats["home"]["played"])
            else:
                row["home_advantage"] = 0.5

            # Derby flag
            league_name = f["league"]["name"].lower()
            row["is_derby"] = 1 if any(derby in league_name for derby in ["derby", "clasico", "rival"]) else 0

            # Player form
            home_players = self.collector.get_player_form(home, season, 3)
            away_players = self.collector.get_player_form(away, season, 3)
            row["home_player_rating"] = sum(p["rating"] for p in home_players) / max(1, len(home_players))
            row["away_player_rating"] = sum(p["rating"] for p in away_players) / max(1, len(away_players))
            row["home_player_goals"] = sum(p["goals"] for p in home_players)
            row["away_player_goals"] = sum(p["goals"] for p in away_players)

            # Coach form
            home_coach = self.collector.get_coach_form(home, season)
            away_coach = self.collector.get_coach_form(away, season)
            row["home_coach_winrate"] = home_coach.get("win_rate_last5", 0.5)
            row["away_coach_winrate"] = away_coach.get("win_rate_last5", 0.5)

            # Transfers impact
            row["home_transfers_impact"] = self.collector.get_transfers_impact(home, season)
            row["away_transfers_impact"] = self.collector.get_transfers_impact(away, season)

            # Pitch factor
            stadium_name = f.get("fixture", {}).get("venue", {}).get("name", "")
            row["pitch_factor"] = self.collector.get_pitch_factor(stadium_name)

            # Referee bias
            referee_id = f.get("fixture", {}).get("referee", {}).get("id")
            if referee_id:
                row["ref_home_bias"] = self.collector.get_referee_bias(referee_id)
            else:
                row["ref_home_bias"] = 0.5

            rows.append(row)

        df = pd.DataFrame(rows)
        return df.fillna(0)
