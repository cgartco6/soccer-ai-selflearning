import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {"x-apisports-key": self.api_key}
        self.weather_key = os.getenv("OPENWEATHER_KEY")

    def _request(self, endpoint, params):
        for attempt in range(3):
            try:
                resp = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("response"):
                        return data["response"]
                    else:
                        return []
                elif resp.status_code == 429:
                    time.sleep(60)  # rate limit
                else:
                    print(f"API error {resp.status_code}: {resp.text}")
                    return []
            except Exception as e:
                print(f"Request failed: {e}, attempt {attempt+1}")
                time.sleep(2)
        return []

    def get_todays_fixtures(self, league_ids, season):
        """Fetch all fixtures for today across given leagues."""
        today = datetime.now().strftime("%Y-%m-%d")
        all_fixtures = []
        for lid in league_ids:
            params = {"league": lid, "season": season, "date": today}
            fixtures = self._request("fixtures", params)
            for f in fixtures:
                if f["fixture"]["status"]["long"] in ["Not Started", "Time to be defined"]:
                    all_fixtures.append(f)
        return all_fixtures

    def get_team_stats(self, team_id, league_id, season):
        """Get season stats: goals avg, form, etc."""
        params = {"team": team_id, "league": league_id, "season": season}
        stats = self._request("teams/statistics", params)
        return stats[0] if stats else None

    def get_recent_form(self, team_id, league_id, season, n=5):
        """Return points, goals scored/conceded in last n league matches."""
        params = {"team": team_id, "league": league_id, "season": season, "last": n}
        fixtures = self._request("fixtures", params)
        if not fixtures:
            return {"points": 0, "goals_scored": 0, "goals_conceded": 0}
        points = 0
        goals_for = 0
        goals_against = 0
        for f in fixtures:
            if f["teams"]["home"]["id"] == team_id:
                gf = f["goals"]["home"]
                ga = f["goals"]["away"]
            else:
                gf = f["goals"]["away"]
                ga = f["goals"]["home"]
            if gf is None or ga is None:
                continue
            goals_for += gf
            goals_against += ga
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        return {"points": points, "goals_scored": goals_for, "goals_conceded": goals_against}

    def get_head2head(self, team1_id, team2_id, limit=5):
        params = {"h2h": f"{team1_id}-{team2_id}", "last": limit}
        matches = self._request("fixtures/headtohead", params)
        if not matches:
            return []
        h2h_data = []
        for m in matches:
            h2h_data.append({
                "date": m["fixture"]["date"],
                "home_goals": m["goals"]["home"],
                "away_goals": m["goals"]["away"],
                "winner": m["teams"]["home"]["winner"] if m["teams"]["home"]["winner"] is not None else "draw"
            })
        return h2h_data

    def get_injuries(self, team_id, season):
        params = {"team": team_id, "season": season}
        injuries = self._request("injuries", params)
        return injuries if injuries else []

    def get_weather(self, city_name):
        if not self.weather_key:
            return {}
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={self.weather_key}&units=metric"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"],
                "rain": 1 if "rain" in data else 0
            }
        except:
            return {}
