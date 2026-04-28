#!/usr/bin/env python
"""
Backtest Ensemble Model on Historical Fixtures
Usage: python scripts/backtest.py --start 2025-08-01 --end 2026-05-01 --league 39
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collector import DataCollector
from src.features import FeatureEngineer
from src.model import EnsemblePredictor
from src.feedback import FeedbackLoop
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def fetch_historical_fixtures(collector, league_id, season, start_date, end_date):
    """Fetch fixtures between two dates using pagination (simple version)."""
    # Because the free API may have limited history, we'll try to get by date ranges.
    # In practice, you can loop over days.
    fixtures = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        params = {"league": league_id, "season": season, "date": date_str}
        day_fixtures = collector._request("fixtures", params)
        if day_fixtures:
            # Keep only finished matches
            finished = [f for f in day_fixtures if f["fixture"]["status"]["short"] == "FT"]
            fixtures.extend(finished)
        current += timedelta(days=1)
    return fixtures


def backtest(league_id, season, start_date, end_date):
    collector = DataCollector()
    engineer = FeatureEngineer(collector)
    model = EnsemblePredictor({"value_bet": {}})  # minimal config

    # Fetch historical fixtures
    print(f"Fetching fixtures for league {league_id} from {start_date} to {end_date} ...")
    fixtures = fetch_historical_fixtures(collector, league_id, season, start_date, end_date)
    if not fixtures:
        print("No finished fixtures found.")
        return

    # Build features for each fixture
    feature_list = []
    for f in fixtures:
        # Build features expects a list; we process one at a time
        single = [f]
        df = engineer.build_features(single, league_id, season)
        if not df.empty:
            # Add actual results
            home_goals = f["goals"]["home"]
            away_goals = f["goals"]["away"]
            if home_goals is None or away_goals is None:
                continue
            df["actual_home_goals"] = home_goals
            df["actual_away_goals"] = away_goals
            df["target"] = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
            df["btts_actual"] = 1 if (home_goals > 0 and away_goals > 0) else 0
            feature_list.append(df)
    if not feature_list:
        print("No valid matches with complete data.")
        return

    X_df = pd.concat(feature_list, ignore_index=True)
    # Keep a copy of fixture IDs and actual results
    meta = X_df[["fixture_id", "actual_home_goals", "actual_away_goals", "target", "btts_actual"]].copy()
    # Select numerical features for model
    exclude = ["fixture_id", "home_team", "away_team", "league_id", "actual_home_goals", "actual_away_goals", "target", "btts_actual"]
    feature_cols = [c for c in X_df.columns if c not in exclude and X_df[c].dtype in [float, int]]
    X = X_df[feature_cols].fillna(0)
    y = meta["target"]

    # Train on all but last 30 days? For backtest we do time‑based split
    # Sort by date (need to get date from fixtures). We'll assume X is already chronological.
    # Simple: use first 70% as train, last 30% as test.
    split_idx = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    btts_train = meta["btts_actual"].iloc[:split_idx]
    btts_test = meta["btts_actual"].iloc[split_idx:]

    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches")
    # Train ensemble
    model.train(X_train, y_train)
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # BTTS using a separate simple model (or we could use the same ensemble)
    # For simplicity, use the same features for a BTTS classifier – but we can reuse the Poisson method
    from sklearn.linear_model import LogisticRegression
    btts_model = LogisticRegression()
    btts_model.fit(X_train, btts_train)
    btts_pred = btts_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print("\n=== 1X2 Prediction Performance ===")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== BTTS Prediction Performance ===")
    btts_acc = accuracy_score(btts_test, btts_pred)
    print(f"Accuracy: {btts_acc:.3f}")
    print(f"Precision: {precision_score(btts_test, btts_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(btts_test, btts_pred, zero_division=0):.3f}")

    # Store results for future reference
    results_df = pd.DataFrame({
        "fixture_id": meta["fixture_id"].iloc[split_idx:].values,
        "target_actual": y_test.values,
        "target_pred": y_pred,
        "btts_actual": btts_test.values,
        "btts_pred": btts_pred
    })
    os.makedirs("data/backtests", exist_ok=True)
    results_df.to_csv("data/backtests/latest_backtest.csv", index=False)
    print("\nDetailed results saved to data/backtests/latest_backtest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=int, default=39, help="API‑Football league ID (39=EPL)")
    parser.add_argument("--season", type=int, default=2026, help="Season year")
    parser.add_argument("--start", type=str, default="2025-08-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-04-28", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    backtest(args.league, args.season, args.start, args.end)
