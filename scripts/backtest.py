#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.pipeline import Pipeline

def backtest():
    print("Backtest: loading historical data...")
    pipe = Pipeline()
    # Load features and results
    df = pd.read_sql_query("""
        SELECT f.*, r.home_goals, r.away_goals
        FROM features f
        JOIN fixture_results r ON f.fixture_id = r.fixture_id
    """, pipe.feedback.conn)
    if df.empty:
        print("No historical data yet.")
        return
    df["target"] = df.apply(lambda x: 0 if x["home_goals"] > x["away_goals"] else 1 if x["home_goals"] == x["away_goals"] else 2, axis=1)
    exclude = ["fixture_id", "home_goals", "away_goals", "target"]
    X = df[[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]].fillna(0)
    y = df["target"]
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    pipe.model.train(X_train, y_train)
    preds = pipe.model.ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Backtest accuracy: {acc:.3f}")

if __name__ == "__main__":
    backtest()
