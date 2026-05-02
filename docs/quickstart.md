# Quick Start Guide

1. Install Python 3.10
2. `pip install -r requirements.txt`
3. Get API key from api-football.com
4. Create `.env` with `API_FOOTBALL_KEY=yourkey`
5. Run `python scripts/daily_run.py`
6. After matches: `python scripts/update_results.py`
7. Weekly: `python scripts/weekly_retrain.py`
