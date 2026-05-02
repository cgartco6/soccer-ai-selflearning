# Architecture

- **collector.py**: Fetches data from free APIs (fixtures, stats, injuries, H2H, odds, player form, coach, transfers, referee bias)
- **features.py**: Builds 40+ numeric features
- **model.py**: Ensemble of XGBoost, LightGBM, RandomForest
- **value_finder.py**: Expected value + Kelly stake calculation
- **feedback.py**: SQLite storage of predictions and actual results
- **pipeline.py**: Orchestrates daily predictions, result updates, weekly retraining
- **ai_optimizer.py**: Reinforcement‑inspired weight tuning
- **ui_server.py**: Flask dashboard
- **scripts/**: Automation scripts
