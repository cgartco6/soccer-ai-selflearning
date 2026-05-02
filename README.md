# Soccer AI Self‑Learning Predictor

Complete automated soccer prediction system that learns from its mistakes.

## Features
- Fetches fixtures, team stats, player form, coach form, injuries, H2H, weather, transfers, referee bias
- Builds 40+ numeric features
- Ensemble model (XGBoost + LightGBM + RandomForest)
- Predicts 1X2 and BTTS probabilities
- Fetches live odds (free The Odds API) and calculates value bets (expected value + Kelly stake)
- Stores predictions and actual results
- Retrains weekly and tunes ensemble weights using AI reinforcement learning
- Sends daily picks to Telegram
- Web dashboard (Flask) for live viewing

## Quick Start
1. Get free API keys: [API-Football](https://www.api-football.com/), [The Odds API](https://the-odds-api.com/)
2. Copy `.env.example` to `.env` and add your keys
3. Install Python 3.10, then:
