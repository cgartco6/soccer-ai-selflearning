import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'randomforest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        }
        self.ensemble = None
        
    def train(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
        self.ensemble = VotingClassifier(
            estimators=[('xgb', self.models['xgboost']),
                       ('lgb', self.models['lightgbm']),
                       ('rf', self.models['randomforest'])],
            voting='soft'
        )
        self.ensemble.fit(X, y)
        joblib.dump(self.ensemble, 'models/ensemble_model.joblib')
        print("Model trained and saved.")
    
    def predict_proba(self, X):
        proba = self.ensemble.predict_proba(X)
        return {'home_win': proba[:, 0], 'draw': proba[:, 1], 'away_win': proba[:, 2]}
    
    def predict_btts(self, X):
        """Predict Both Teams to Score probability using Poisson distribution."""
        home_goals = X['home_goals_scored_avg'].values if 'home_goals_scored_avg' in X else np.ones(len(X))
        away_goals = X['away_goals_scored_avg'].values if 'away_goals_scored_avg' in X else np.ones(len(X))
        return (home_goals * away_goals) / (1 + home_goals * away_goals)
