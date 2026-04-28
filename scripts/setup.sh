#!/bin/bash
# Setup script for Soccer AI Betting Predictor

echo "Setting up Soccer AI Betting Predictor..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
playwright install-deps

# Setup directories
mkdir -p data/raw data/processed models logs

# Copy environment example
cp .env.example .env
echo "Please edit .env file to add your API keys."

# Initialize database
python -c "from src.learn.FeedbackLoop import FeedbackLoop; FeedbackLoop()"

echo "Setup complete! Run 'python src/pipeline/run_pipeline.py' to start."

# Test stealth scraper
echo "Testing stealth scraper..."
python -c "from src.data.Collector import DataCollector; d=DataCollector(); print('Stealth scrape test:', d.stealth_scrape_odds('https://www.hollywoodbets.net/sports/football')[:5])"
