import numpy as np
import pandas as pd
import sqlite3

class AIOptimizer:
    def __init__(self, db_path="data/betting.db"):
        self.conn = sqlite3.connect(db_path)

    def compute_model_accuracy_per_market(self):
        """Return dict with accuracy per market (home, draw, btts)."""
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
        """Use reinforcement learning (simplified) to adjust ensemble weights."""
        acc = self.compute_model_accuracy_per_market()
        # If draw accuracy is low, reduce weight for draw prediction, etc.
        # Simple rule: weight = accuracy / max_accuracy
        max_acc = max(acc.values()) if acc else 1.0
        new_weights = [acc.get("home_win", 0.33), acc.get("draw", 0.33), acc.get("away_win", 0.33)]
        new_weights = [w / max_acc for w in new_weights]
        total = sum(new_weights)
        return [w / total for w in new_weights]
