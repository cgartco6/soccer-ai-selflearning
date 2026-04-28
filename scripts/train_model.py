import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.EnsemblePredictor import EnsemblePredictor

def load_historical_data():
    """Load historical match data."""
    # Replace with actual data loading logic
    return pd.DataFrame({
        'home_goals_scored_avg': np.random.randn(1000),
        'away_goals_scored_avg': np.random.randn(1000),
        'home_goals_conceded_avg': np.random.randn(1000),
        'away_goals_conceded_avg': np.random.randn(1000)
    })

if __name__ == "__main__":
    data = load_historical_data()
    X = data.drop(columns=['target'], errors='ignore')
    y = np.random.choice([0,1,2], size=len(data))  # Placeholder target
    
    predictor = EnsemblePredictor()
    predictor.train(X, y)
