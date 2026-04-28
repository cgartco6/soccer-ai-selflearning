import sqlite3
import pandas as pd
import joblib
from datetime import datetime, timedelta

class FeedbackLoop:
    def __init__(self, db_path="data/betting.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                fixture_id INTEGER,
                market TEXT,
                predicted_prob REAL,
                odds REAL,
                edge REAL,
                stake REAL,
                prediction_time TEXT,
                actual_outcome INTEGER,
                profit REAL,
                PRIMARY KEY (fixture_id, market)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fixture_results (
                fixture_id INTEGER PRIMARY KEY,
                home_goals INTEGER,
                away_goals INTEGER,
                result TEXT,
                btts INTEGER
            )
        """)

    def store_predictions(self, bets, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        cur = self.conn.cursor()
        for bet in bets:
            cur.execute("""
                INSERT OR REPLACE INTO predictions
                (fixture_id, market, predicted_prob, odds, edge, stake, prediction_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (bet["fixture_id"], bet["market"], bet["model_prob"], bet["odds"], bet["edge"], bet["stake"], timestamp))
        self.conn.commit()

    def update_results(self, fixture_id, home_goals, away_goals):
        """Store actual result and calculate profit for all related bets."""
        result = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        btts = 1 if home_goals > 0 and away_goals > 0 else 0
        self.conn.execute("""
            INSERT OR REPLACE INTO fixture_results (fixture_id, home_goals, away_goals, result, btts)
            VALUES (?, ?, ?, ?, ?)
        """, (fixture_id, home_goals, away_goals, result, btts))

        # Update predictions with actual outcome and profit
        cur = self.conn.cursor()
        cur.execute("SELECT market, predicted_prob, odds, stake FROM predictions WHERE fixture_id = ?", (fixture_id,))
        for row in cur.fetchall():
            market, prob, odds, stake = row
            if market == "home_win":
                won = 1 if result == "home" else 0
            elif market == "draw":
                won = 1 if result == "draw" else 0
            elif market == "btts_yes":
                won = 1 if btts == 1 else 0
            else:
                continue
            profit = (stake * (odds - 1) if won else -stake) if stake else 0
            self.conn.execute("""
                UPDATE predictions SET actual_outcome = ?, profit = ? 
                WHERE fixture_id = ? AND market = ?
            """, (won, profit, fixture_id, market))
        self.conn.commit()

    def get_training_data(self, lookback_days=30):
        """Returns X, y for all fixtures that have results and predictions."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        query = """
            SELECT p.fixture_id, p.market, p.predicted_prob, p.actual_outcome, fr.home_goals, fr.away_goals
            FROM predictions p
            JOIN fixture_results fr ON p.fixture_id = fr.fixture_id
            WHERE p.prediction_time > ? AND p.actual_outcome IS NOT NULL
        """
        df = pd.read_sql(query, self.conn, params=(cutoff,))
        if df.empty:
            return None, None
        # Here you would join with original feature set to get X, y.
        # For simplicity, we return aggregated model performance per model.
        # In practice, you need to store features alongside predictions.
        return df

    def get_model_performance(self):
        """Compute per-market accuracy and per-model contribution (simplified)."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT market, COUNT(*) as total, SUM(actual_outcome) as correct
            FROM predictions WHERE actual_outcome IS NOT NULL
            GROUP BY market
        """)
        acc = {row[0]: row[2]/row[1] for row in cur.fetchall()}
        return acc
