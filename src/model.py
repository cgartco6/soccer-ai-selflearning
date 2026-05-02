import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

class EnsemblePredictor:
    def __init__(self, config):
        self.models = {
            "xgb": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, use_label_encoder=False),
            "lgb": LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1),
            "rf": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        }
        self.ensemble = None

    def train(self, X, y):
        for name, m in self.models.items():
            m.fit(X, y)
        self.ensemble = VotingClassifier(
            estimators=[("xgb", self.models["xgb"]), ("lgb", self.models["lgb"]), ("rf", self.models["rf"])],
            voting="soft"
        )
        self.ensemble.fit(X, y)
        acc = accuracy_score(y, self.ensemble.predict(X))
        print(f"Model trained. Training accuracy: {acc:.3f}")

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def predict_btts(self, X):
        home_goals = X["home_goals_avg"].values if "home_goals_avg" in X else np.ones(len(X))
        away_goals = X["away_goals_avg"].values if "away_goals_avg" in X else np.ones(len(X))
        return (home_goals * away_goals) / (1 + home_goals * away_goals)
