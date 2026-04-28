import numpy as np

class ValueFinder:
    def __init__(self, kelly_fraction=0.25):
        self.kelly_fraction = kelly_fraction
    
    def find_value_bets(self, model_probs, odds_data):
        value_bets = []
        for match in odds_data:
            # Home win
            edge = model_probs[match['id']]['home_win'] - (1 / match['odds']['home_win'])
            if edge > 0.05:
                value_bets.append({'match': match['match_name'], 'market': 'Home Win', 'edge': edge})
            
            # Draw
            edge = model_probs[match['id']]['draw'] - (1 / match['odds']['draw'])
            if edge > 0.05:
                value_bets.append({'match': match['match_name'], 'market': 'Draw', 'edge': edge})
            
            # BTTS
            edge = model_probs[match['id']]['btts'] - (1 / match['odds']['btts_yes'])
            if edge > 0.05:
                value_bets.append({'match': match['match_name'], 'market': 'BTTS Yes', 'edge': edge})
        return sorted(value_bets, key=lambda x: x['edge'], reverse=True)
    
    def kelly_stake(self, edge, odds):
        b = odds - 1
        p = (1/odds) + edge
        kelly = (p * b - (1-p)) / b
        return max(0, min(kelly * self.kelly_fraction, 0.1))
