# Soccer AI Self‑Learning Predictor

Automated soccer prediction system that learns from its mistakes and adapts using free APIs.

## Features
- Collects fixtures, team stats, H2H, injuries, weather from free API‑Football.
- Engineering 40+ features (form, goals, home advantage, derby, etc.).
- Ensemble model (XGBoost + LightGBM + RandomForest) with performance‑based weight tuning.
- Feedback loop stores predictions and real results, then retrains weekly.
- Value bet detection (Kelly criterion) – requires odds source.

## Setup
1. Get free API key from [API‑Football](https://www.api‑football.com/).
2. Clone repo, `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add your API key.
4. Run `python scripts/daily_run.py` each morning.
5. Run `python scripts/weekly_retrain.py` once per week.

## How It Learns
- After a match, results are fetched and stored.
- Every week, the model is retrained on all data, giving more weight to recent accurate predictions.
- Ensemble weights are adjusted based on each model’s recent accuracy.

## Limitations
- No live odds (you need to integrate a free odds API or scraper).
- Weather requires OpenWeatherMap key (optional).
- Historical data accumulation takes time – start with a few weeks.
