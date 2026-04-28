import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Collector import DataCollector
from features.FeatureEngineering import FeatureEngineering
from models.EnsemblePredictor import EnsemblePredictor
from betting.ValueFinder import ValueFinder
from learn.FeedbackLoop import FeedbackLoop
from learn.ReinforcementOptimizer import GRPOptimizer

def run_full_pipeline():
    print("[1/6] Collecting data...")
    collector = DataCollector()
    # Example: EPL (league_id=39) for season 2026
    fixtures = collector.get_fixtures(39, 2026)
    
    print("[2/6] Engineering features...")
    fe = FeatureEngineering(collector)
    features = fe.create_match_features(fixtures)
    
    print("[3/6] Loading and training model...")
    import joblib
    predictor = EnsemblePredictor()
    
    if os.path.exists('models/ensemble_model.joblib'):
        predictor.ensemble = joblib.load('models/ensemble_model.joblib')
    else:
        # Dummy training data
        predictor.train(features, pd.Series([0,1,2]* (len(features)//3)))
        joblib.dump(predictor.ensemble, 'models/ensemble_model.joblib')
    
    print("[4/6] Generating predictions...")
    predictions = predictor.predict_proba(features)
    btts_probs = predictor.predict_btts(features)
    
    print("[5/6] Finding value bets...")
    odds_data = []
    for i, fixture in fixtures.iterrows():
        odds = collector.fetch_odds(fixture['fixture']['id'])
        match_odds = {'id': i, 'match_name': f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}", 'odds': {'home_win': 2.5, 'draw': 3.2, 'away_win': 2.8, 'btts_yes': 1.8}}
        odds_data.append(match_odds)
    
    model_probs = {i: {'home_win': predictions['home_win'][i], 'draw': predictions['draw'][i], 'away_win': predictions['away_win'][i], 'btts': btts_probs[i]} for i in range(len(fixtures))}
    value_finder = ValueFinder()
    bets = value_finder.find_value_bets(model_probs, odds_data)
    
    print("[6/6] Initializing learning systems...")
    feedback = FeedbackLoop()
    for i, fixture in fixtures.iterrows():
        feedback.record_prediction(fixture['fixture']['id'], 'draw', predictions['draw'][i])
    
    optimizer = GRPOptimizer(predictor, value_finder)
    print("Pipeline complete.")

if __name__ == "__main__":
    run_full_pipeline()
