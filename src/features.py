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

            # basic stats
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

            # Recent form (last 5 matches)
            home_form = self.collector.get_recent_form(home, league_id, season, 5)
            away_form = self.collector.get_recent_form(away, league_id, season, 5)
            row["home_points_last5"] = home_form["points"]
            row["away_points_last5"] = away_form["points"]
            row["home_goals_last5"] = home_form["goals_scored"]
            row["away_goals_last5"] = away_form["goals_scored"]
            row["home_conceded_last5"] = home_form["goals_conceded"]
            row["away_conceded_last5"] = away_form["goals_conceded"]

            # H2H (last 5)
            h2h = self.collector.get_head2head(home, away, 5)
            home_h2h_wins = sum(1 for m in h2h if m["winner"] == "home")
            away_h2h_wins = sum(1 for m in h2h if m["winner"] == "away")
            draws_h2h = sum(1 for m in h2h if m["winner"] == "draw")
            btts_h2h = sum(1 for m in h2h if m["home_goals"] > 0 and m["away_goals"] > 0)
            row["h2h_home_wins"] = home_h2h_wins
            row["h2h_away_wins"] = away_h2h_wins
            row["h2h_draws"] = draws_h2h
            row["h2h_btts_rate"] = btts_h2h / max(1, len(h2h))

            # Injuries count (key players)
            home_inj = self.collector.get_injuries(home, season)
            away_inj = self.collector.get_injuries(away, season)
            row["home_injuries"] = len([i for i in home_inj if i["player"]["status"] == "out"])
            row["away_injuries"] = len([i for i in away_inj if i["player"]["status"] == "out"])

            # Weather (placeholder – use stadium city)
            city = f.get("fixture", {}).get("venue", {}).get("city", "London")
            weather = self.collector.get_weather(city)
            row["temperature"] = weather.get("temp", 15)
            row["humidity"] = weather.get("humidity", 60)
            row["wind"] = weather.get("wind", 5)
            row["rain"] = weather.get("rain", 0)

            # Home advantage (simple ratio of home wins last season)
            row["home_advantage"] = home_stats["home"]["wins"] / max(1, home_stats["home"]["played"]) if home_stats else 0.5

            # Match importance (derby / knockout)
            is_derby = any(derby in f["league"]["name"].lower() for derby in ["derby", "clasico"])
            row["is_derby"] = 1 if is_derby else 0
            # (Add more: cup knockout, relegation, etc. – can be extended)

            rows.append(row)

        df = pd.DataFrame(rows)
        # Fill any missing numeric values
        df = df.fillna(0)
        return df

    def add_target(self, df, actual_results):
        """Merge actual results (to be used after matches)."""
        df = df.copy()
        df["actual_home_goals"] = df["fixture_id"].map(actual_results).apply(lambda x: x[0] if x else None)
        df["actual_away_goals"] = df["fixture_id"].map(actual_results).apply(lambda x: x[1] if x else None)
        df["target"] = df.apply(lambda r: 0 if r["actual_home_goals"] > r["actual_away_goals"] 
                                         else 1 if r["actual_home_goals"] == r["actual_away_goals"] 
                                         else 2, axis=1)
        df["btts"] = ((df["actual_home_goals"] > 0) & (df["actual_away_goals"] > 0)).astype(int)
        return df
