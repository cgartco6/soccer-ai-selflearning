import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
import requests
from dotenv import load_dotenv
from .collector import DataCollector
from .features import FeatureEngineer
from .model import EnsemblePredictor
from .value_finder import ValueFinder
from .feedback import FeedbackLoop
from .ai_optimizer import AIOptimizer

load_dotenv()

class Pipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.collector = DataCollector()
        self.fe = FeatureEngineer(self.collector)
        self.feedback = FeedbackLoop()
        self.value_finder = ValueFinder(self.config)
        self.model = EnsemblePredictor(self.config)
        self.ai_optimizer = AIOptimizer()
        os.makedirs("models", exist_ok=True)

    def daily_predict(self):
        league_ids = [l["id"] for l in self.config["leagues"]]
        season = self.config["season"]
        fixtures = self.collector.get_todays_fixtures(league_ids, season)
        if not fixtures:
            print("No fixtures today.")
            return

        all_dfs = []
        for lid in league_ids:
            lid_fixtures = [f for f in fixtures if f["league"]["id"] == lid]
            if lid_fixtures:
                df = self.fe.build_features(lid_fixtures, lid, season)
                all_dfs.append(df)
        if not all_dfs:
            return
        X = pd.concat(all_dfs, ignore_index=True)

        # Store features for later training
        for _, row in X.iterrows():
            self.feedback.store_features(row["fixture_id"], row.to_dict())

        # Load model if exists
        model_path = "models/ensemble.pkl"
        if os.path.exists(model_path):
            self.model.ensemble = joblib.load(model_path)
        else:
            print("No trained model yet. Run weekly_retrain after collecting results.")
            return

        # Predict
        X_num = X.select_dtypes(include=[np.number]).drop(columns=["fixture_id"], errors="ignore")
        proba = self.model.predict_proba(X_num)
        X["home_prob"] = proba[:, 0]
        X["draw_prob"] = proba[:, 1]
        X["away_prob"] = proba[:, 2]
        X["btts_prob"] = self.model.predict_btts(X_num)

        # Get odds (simplified – in production use real odds API)
        odds_dict = {}
        for fid in X["fixture_id"]:
            odds_dict[fid] = self.collector.get_odds(fid)

        # Find value bets
        bets = self.value_finder.find_bets(X[["fixture_id", "home_prob", "draw_prob", "away_prob", "btts_prob"]], odds_dict)
        if bets:
            self.feedback.store_predictions(bets)
            print(f"Found {len(bets)} value bets.")
            self._send_telegram(f"🎯 *Value Bets Today*\n" + "\n".join([f"{b['fixture_id']} {b['market']} @ {b['odds']:.2f} (edge {b['edge']:.1%})" for b in bets[:5]]))
        else:
            print("No value bets found.")

    def update_results(self, date=None):
        league_ids = [l["id"] for l in self.config["leagues"]]
        season = self.config["season"]
        fixtures = self.collector.get_finished_fixtures(league_ids, season, date)
        for f in fixtures:
            fid = f["fixture"]["id"]
            home_g = f["goals"]["home"]
            away_g = f["goals"]["away"]
            if home_g is None or away_g is None:
                continue
            self.feedback.update_results(fid, home_g, away_g)
        print(f"Updated results for {len(fixtures)} matches.")

    def weekly_retrain(self):
        # Load historical features and results
        df = pd.read_sql_query("""
            SELECT f.*, r.home_goals, r.away_goals
            FROM features f
            JOIN fixture_results r ON f.fixture_id = r.fixture_id
        """, self.feedback.conn)
        if df.empty:
            print("No historical data yet. Run update_results a few times first.")
            return

        df["target"] = df.apply(lambda x: 0 if x["home_goals"] > x["away_goals"] else 1 if x["home_goals"] == x["away_goals"] else 2, axis=1)
        exclude = ["fixture_id", "home_goals", "away_goals", "target"]
        X = df[[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]].fillna(0)
        y = df["target"]
        print(f"Training on {len(X)} matches...")
        self.model.train(X, y)
        joblib.dump(self.model.ensemble, "models/ensemble.pkl")

        # AI weight tuning
        self.ai_optimizer.suggest_weight_adjustment([1,1,1])
        print("Model retrained and AI optimizer updated.")

    def _send_telegram(self, message):
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception as e:
            print(f"Telegram send failed: {e}")
