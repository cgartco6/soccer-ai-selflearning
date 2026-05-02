import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import json

load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {"x-apisports-key": self.api_key}
        self.weather_key = os.getenv("OPENWEATHER_KEY")
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.cache = {}

    def _request(self, endpoint, params, cache_ttl=3600):
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['time'] < cache_ttl:
            return self.cache[cache_key]['data']
        for attempt in range(3):
            try:
                resp = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json().get("response", [])
                    self.cache[cache_key] = {'data': data, 'time': time.time()}
                    return data
                elif resp.status_code == 429:
                    time.sleep(60)
                else:
                    print(f"API error {resp.status_code} for {endpoint}")
                    return []
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(2)
        return []

    def get_todays_fixtures(self, league_ids, season):
        today = datetime.now().strftime("%Y-%m-%d")
        all_fixtures = []
        for lid in league_ids:
            params = {"league": lid, "season": season, "date": today}
            fixtures = self._request("fixtures", params)
            for f in fixtures:
                if f["fixture"]["status"]["long"] in ["Not Started", "Time to be defined"]:
                    all_fixtures.append(f)
        return all_fixtures

    def get_finished_fixtures(self, league_ids, season, date=None):
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        finished = []
        for lid in league_ids:
            params = {"league": lid, "season": season, "date": date}
            fixtures = self._request("fixtures", params)
            for f in fixtures:
                if f["fixture"]["status"]["short"] == "FT":
                    finished.append(f)
        return finished

    def get_team_stats(self, team_id, league_id, season):
        params = {"team": team_id, "league": league_id, "season": season}
        stats = self._request("teams/statistics", params)
        return stats[0] if stats else None

    def get_recent_form(self, team_id, league_id, season, n=5):
        params = {"team": team_id, "league": league_id, "season": season, "last": n}
        fixtures = self._request("fixtures", params)
        if not fixtures:
            return {"points": 0, "goals_scored": 0, "goals_conceded": 0}
        points = 0
        gf = 0
        ga = 0
        for f in fixtures:
            if f["goals"]["home"] is None or f["goals"]["away"] is None:
                continue
            if f["teams"]["home"]["id"] == team_id:
                gf += f["goals"]["home"]
                ga += f["goals"]["away"]
                if f["goals"]["home"] > f["goals"]["away"]:
                    points += 3
                elif f["goals"]["home"] == f["goals"]["away"]:
                    points += 1
            else:
                gf += f["goals"]["away"]
                ga += f["goals"]["home"]
                if f["goals"]["away"] > f["goals"]["home"]:
                    points += 3
                elif f["goals"]["away"] == f["goals"]["home"]:
                    points += 1
        return {"points": points, "goals_scored": gf, "goals_conceded": ga}

    def get_head2head(self, team1_id, team2_id, limit=5):
        params = {"h2h": f"{team1_id}-{team2_id}", "last": limit}
        matches = self._request("fixtures/headtohead", params)
        if not matches:
            return []
        h2h = []
        for m in matches:
            h2h.append({
                "winner": "home" if m["teams"]["home"]["winner"] else "away" if m["teams"]["away"]["winner"] else "draw",
                "home_goals": m["goals"]["home"],
                "away_goals": m["goals"]["away"]
            })
        return h2h

    def get_injuries(self, team_id, season):
        params = {"team": team_id, "season": season}
        injuries = self._request("injuries", params)
        return injuries if injuries else []

    def get_weather(self, city):
        if not self.weather_key:
            return {}
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_key}&units=metric"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            return {"temp": data["main"]["temp"], "humidity": data["main"]["humidity"], "wind": data["wind"]["speed"], "rain": 1 if "rain" in data else 0}
        except:
            return {}

    def get_odds(self, fixture_id):
        if not self.odds_api_key:
            return {}
        # Simplified: use The Odds API; requires mapping. For now return dummy.
        return {"home": 2.5, "draw": 3.2, "away": 2.8, "btts_yes": 1.8}

    def get_player_form(self, team_id, season, limit=3):
        params = {"team": team_id, "season": season}
        players = self._request("players", params)
        if not players:
            return []
        sorted_players = sorted(players, key=lambda x: float(x.get("statistics", [{}])[0].get("games", {}).get("rating", 0) or 0), reverse=True)
        top = []
        for p in sorted_players[:limit]:
            stats = p.get("statistics", [{}])[0]
            top.append({
                "name": p["player"]["name"],
                "rating": float(stats.get("games", {}).get("rating", 0) or 0),
                "goals": stats.get("goals", {}).get("total", 0),
                "assists": stats.get("goals", {}).get("assists", 0)
            })
        return top

    def get_coach_form(self, team_id, season):
        params = {"team": team_id, "season": season}
        coaches = self._request("coachs", params)
        if not coaches:
            return {"name": "unknown", "win_rate_last5": 0.5}
        coach = coaches[0]
        return {"name": coach.get("name", "unknown"), "win_rate_last5": 0.5}

    def get_transfers_impact(self, team_id, season):
        params = {"team": team_id, "season": season}
        transfers = self._request("transfers", params)
        if not transfers:
            return 0.0
        impact = 0
        for t in transfers:
            if t.get("type") == "arrival":
                impact += 0.1
            else:
                impact -= 0.1
        return impact

    def get_pitch_factor(self, stadium_name):
        small = ["Old Trafford", "Emirates", "Camp Nou"]
        large = ["Wembley", "Bernabeu"]
        if stadium_name in small:
            return 0.9
        elif stadium_name in large:
            return 1.1
        return 1.0

    def get_referee_bias(self, referee_id):
        params = {"referee": referee_id}
        fixtures = self._request("fixtures", params)
        if not fixtures:
            return 0.5
        home_wins = 0
        total = 0
        for f in fixtures:
            if f["fixture"]["status"]["short"] == "FT" and f["goals"]["home"] is not None:
                total += 1
                if f["teams"]["home"]["winner"]:
                    home_wins += 1
        return home_wins / total if total > 0 else 0.5
