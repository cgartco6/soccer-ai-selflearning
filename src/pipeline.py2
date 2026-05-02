import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .collector import DataCollector
from .features import FeatureEngineer
from .model import EnsemblePredictor
from .value_finder import ValueFinder
from .feedback import FeedbackLoop
import joblib
import os

class Pipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.collector = DataCollector()
        self.fe = FeatureEngineer(self.collector)
        self.model = EnsemblePredictor(self.config)
        self.feedback = FeedbackLoop()
        self.value_finder = ValueFinder(self.config)

    def daily_predict(self):
        """Fetch fixtures, build features, predict today's matches."""
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
        X = X.fillna(0)

        # Predict
        if not os.path.exists("models/ensemble.pkl"):
            print("Model not found. Skipping predictions.")
            return
        proba = self.model.predict_proba(X.drop(columns=["fixture_id"], errors="ignore"))
        X["home_prob"] = proba[:, 0]
        X["draw_prob"] = proba[:, 1]
        X["away_prob"] = proba[:, 2]
        # BTTS prediction using Poisson approximation
        X["btts_prob"] = (X["home_goals_avg"] * X["away_goals_avg"]) / (1 + X["home_goals_avg"] * X["away_goals_avg"])

        # Fetch odds (simplified – in reality you'd call a free odds API or scrape)
        odds_dict = {}  # fixture_id -> {home, draw, away, btts_yes}
        # Placeholder: you can implement odds fetching using The Odds API free tier
        # For now we skip odds and don't place bets.
        # If you have odds, call value_finder.find_bets(...)

        # Store predictions for later feedback
        for idx, row in X.iterrows():
            for market in ["home_win", "draw", "btts_yes"]:
                prob = row["home_prob"] if market == "home_win" else row["draw_prob"] if market == "draw" else row["btts_prob"]
                # We'll store without odds (just for tracking)
                self.feedback.store_predictions([{
                    "fixture_id": row["fixture_id"],
                    "market": market,
                    "model_prob": prob,
                    "odds": 2.0,  # dummy
                    "edge": 0.0,
                    "stake": 0.0
                }])
        print(f"Predictions stored for {len(X)} fixtures.")

    def update_with_results(self):
        """After matches, fetch results and store in feedback."""
        # This would be called the next day.
        # For brevity, we assume you have a method to get results by fixture_id.
        # We'll implement a simple version that queries API.
        pass

    def weekly_retrain(self):
        """Retrain model using all historical data with feedback."""
        # Collect all previous features and targets
        # We need a historical dataset. For simplicity, we maintain a CSV.
        # In production, store features and results in DB.
        hist_path = "data/historical_features.csv"
        if not os.path.exists(hist_path):
            print("No historical data yet.")
            return
        data = pd.read_csv(hist_path)
        if "target" not in data.columns:
            print("No target column in historical data.")
            return
        X = data.drop(columns=["target", "fixture_id", "actual_home_goals", "actual_away_goals"], errors="ignore")
        y = data["target"]
        # Optional: sample weights based on recency
        self.model.train(X, y)
        print("Weekly retrain complete.")
