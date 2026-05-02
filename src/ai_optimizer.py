import numpy as np
import pandas as pd
import sqlite3

class AIOptimizer:
    def __init__(self, db_path="data/betting.db"):
        self.conn = sqlite3.connect(db_path)

    def compute_model_accuracy_per_market(self):
        df = pd.read_sql_query("""
            SELECT market, actual_outcome
            FROM predictions
            WHERE actual_outcome IS NOT NULL
        """, self.conn)
        if df.empty:
            return {}
        acc = {}
        for market in ["home_win", "draw", "btts_yes"]:
            sub = df[df["market"] == market]
            if len(sub) > 0:
                acc[market] = sub["actual_outcome"].mean()
        return acc

    def suggest_weight_adjustment(self, current_weights):
        acc = self.compute_model_accuracy_per_market()
        if not acc:
            return current_weights
        # Map market to model index: home_win, draw, away_win (we approximate)
        # For simplicity, we only adjust draw weight
        draw_acc = acc.get("draw", 0.33)
        new_weights = [current_weights[0], draw_acc, current_weights[2]]
        total = sum(new_weights)
        if total > 0:
            new_weights = [w / total for w in new_weights]
        print(f"AI optimizer new weights: {new_weights}")
        return new_weights
