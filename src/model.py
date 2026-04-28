import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

class EnsemblePredictor:
    def __init__(self, config):
        self.config = config
        self.models = {
            "xgb": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, use_label_encoder=False),
            "lgb": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1),
            "rf": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        }
        self.weights = [1.0, 1.0, 1.0]  # will be updated by feedback
        self.ensemble = None

    def train(self, X, y, sample_weight=None):
        for name, model in self.models.items():
            model.fit(X, y, sample_weight=sample_weight)
        # voting classifier for probability averaging
        from sklearn.ensemble import VotingClassifier
        self.ensemble = VotingClassifier(
            estimators=[("xgb", self.models["xgb"]), ("lgb", self.models["lgb"]), ("rf", self.models["rf"])],
            voting="soft",
            weights=self.weights
        )
        self.ensemble.fit(X, y)
        joblib.dump(self.ensemble, "models/ensemble.pkl")
        print(f"Model trained. Accuracy: {accuracy_score(y, self.ensemble.predict(X)):.3f}")

    def predict_proba(self, X):
        if self.ensemble is None:
            self.ensemble = joblib.load("models/ensemble.pkl")
        return self.ensemble.predict_proba(X)

    def update_weights(self, recent_performance):
        """
        recent_performance: list of dict with keys "model_name", "accuracy"
        Adjust weights proportionally to accuracy.
        """
        acc_dict = {p["model_name"]: p["accuracy"] for p in recent_performance}
        new_weights = []
        for name in ["xgb", "lgb", "rf"]:
            new_weights.append(acc_dict.get(name, 0.5))
        total = sum(new_weights)
        if total > 0:
            self.weights = [w / total for w in new_weights]
            # rebuild ensemble with new weights
            self.ensemble = VotingClassifier(
                estimators=[("xgb", self.models["xgb"]), ("lgb", self.models["lgb"]), ("rf", self.models["rf"])],
                voting="soft",
                weights=self.weights
            )
            # refit on the same data? We'll do it during weekly retrain.
            print(f"Weights updated: {self.weights}")
