import sqlite3
import pandas as pd
from datetime import datetime

class FeedbackLoop:
    def __init__(self, db_path="data/betting.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
                PRIMARY KEY (fixture_id, market, prediction_time)
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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                fixture_id INTEGER PRIMARY KEY,
                home_goals_avg REAL,
                away_goals_avg REAL,
                home_points_last5 REAL,
                away_points_last5 REAL,
                h2h_home_wins INTEGER,
                h2h_draws INTEGER,
                home_injuries INTEGER,
                away_injuries INTEGER,
                temperature REAL,
                rain INTEGER,
                home_advantage REAL,
                is_derby INTEGER,
                home_player_rating REAL,
                away_player_rating REAL,
                home_coach_winrate REAL,
                away_coach_winrate REAL,
                home_transfers_impact REAL,
                away_transfers_impact REAL,
                pitch_factor REAL,
                ref_home_bias REAL
            )
        """)
        self.conn.commit()

    def store_predictions(self, bets):
        cur = self.conn.cursor()
        now = datetime.now().isoformat()
        for bet in bets:
            cur.execute("""
                INSERT OR REPLACE INTO predictions
                (fixture_id, market, predicted_prob, odds, edge, stake, prediction_time)
                VALUES (?,?,?,?,?,?,?)
            """, (bet["fixture_id"], bet["market"], bet["model_prob"], bet["odds"], bet["edge"], bet["stake"], now))
        self.conn.commit()

    def update_results(self, fixture_id, home_goals, away_goals):
        result = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        btts = 1 if (home_goals > 0 and away_goals > 0) else 0
        self.conn.execute("""
            INSERT OR REPLACE INTO fixture_results (fixture_id, home_goals, away_goals, result, btts)
            VALUES (?,?,?,?,?)
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

    def store_features(self, fixture_id, feature_row):
        self.conn.execute("""
            INSERT OR REPLACE INTO features (
                fixture_id, home_goals_avg, away_goals_avg, home_points_last5, away_points_last5,
                h2h_home_wins, h2h_draws, home_injuries, away_injuries, temperature, rain,
                home_advantage, is_derby, home_player_rating, away_player_rating,
                home_coach_winrate, away_coach_winrate, home_transfers_impact, away_transfers_impact,
                pitch_factor, ref_home_bias
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            fixture_id,
            feature_row.get("home_goals_avg", 1.2),
            feature_row.get("away_goals_avg", 1.1),
            feature_row.get("home_points_last5", 0),
            feature_row.get("away_points_last5", 0),
            feature_row.get("h2h_home_wins", 0),
            feature_row.get("h2h_draws", 0),
            feature_row.get("home_injuries", 0),
            feature_row.get("away_injuries", 0),
            feature_row.get("temperature", 15),
            feature_row.get("rain", 0),
            feature_row.get("home_advantage", 0.5),
            feature_row.get("is_derby", 0),
            feature_row.get("home_player_rating", 0),
            feature_row.get("away_player_rating", 0),
            feature_row.get("home_coach_winrate", 0.5),
            feature_row.get("away_coach_winrate", 0.5),
            feature_row.get("home_transfers_impact", 0),
            feature_row.get("away_transfers_impact", 0),
            feature_row.get("pitch_factor", 1.0),
            feature_row.get("ref_home_bias", 0.5)
        ))
        self.conn.commit()
