
**`docs/architecture.md`**  
```markdown
# System Architecture

- **Data Collector**: Fetches fixtures, stats, injuries, H2H from API-Football.
- **Feature Engineering**: Builds 40+ numeric features (form, goals, injuries, weather, derby flag).
- **Ensemble Model**: XGBoost + LightGBM + RandomForest with soft voting.
- **Feedback Loop**: Stores predictions and actual results in SQLite; weekly retraining improves accuracy.
- **Value Finder**: Kelly Criterion + edge detection (requires odds input; you can add a free odds API).

### How It Learns
- Every prediction is saved with its confidence.
- After matches finish, results are stored (via a separate script or manual entry).
- Weekly retrain uses all historical data, optionally weighted by recency.
- The model adapts to new patterns (e.g., injuries, manager changes, league trends).
