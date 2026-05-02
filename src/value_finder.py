import numpy as np

class ValueFinder:
    def __init__(self, config):
        self.min_edge = config.get("value_bet", {}).get("min_edge", 0.05)
        self.kelly_frac = config.get("value_bet", {}).get("kelly_fraction", 0.25)

    def expected_value(self, model_prob, odds):
        return model_prob - (1 / odds)

    def kelly_stake(self, model_prob, odds, max_stake=0.1):
        b = odds - 1
        p = model_prob
        q = 1 - p
        if b <= 0:
            return 0.0
        k = (p * b - q) / b
        return max(0.0, min(k * self.kelly_frac, max_stake))

    def find_bets(self, predictions_df, odds_dict):
        bets = []
        for _, row in predictions_df.iterrows():
            fid = row["fixture_id"]
            if fid not in odds_dict:
                continue
            odds = odds_dict[fid]
            # Home win
            ev = self.expected_value(row["home_prob"], odds["home"])
            if ev > self.min_edge:
                bets.append({
                    "fixture_id": fid,
                    "market": "home_win",
                    "model_prob": row["home_prob"],
                    "odds": odds["home"],
                    "edge": ev,
                    "stake": self.kelly_stake(row["home_prob"], odds["home"])
                })
            # Draw
            ev = self.expected_value(row["draw_prob"], odds["draw"])
            if ev > self.min_edge:
                bets.append({
                    "fixture_id": fid,
                    "market": "draw",
                    "model_prob": row["draw_prob"],
                    "odds": odds["draw"],
                    "edge": ev,
                    "stake": self.kelly_stake(row["draw_prob"], odds["draw"])
                })
            # BTTS Yes
            if "btts_yes" in odds:
                ev = self.expected_value(row["btts_prob"], odds["btts_yes"])
                if ev > self.min_edge:
                    bets.append({
                        "fixture_id": fid,
                        "market": "btts_yes",
                        "model_prob": row["btts_prob"],
                        "odds": odds["btts_yes"],
                        "edge": ev,
                        "stake": self.kelly_stake(row["btts_prob"], odds["btts_yes"])
                    })
        return sorted(bets, key=lambda x: x["edge"], reverse=True)
