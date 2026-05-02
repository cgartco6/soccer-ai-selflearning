import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import requests
import sqlite3
import pandas as pd

load_dotenv()

def send_daily_picks():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials missing")
        return
    conn = sqlite3.connect("data/betting.db")
    # Get latest predictions
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 10", conn)
    if df.empty:
        message = "No predictions for today."
    else:
        message = "*Today's AI Picks*\n"
        for _, row in df.iterrows():
            message += f"🔹 Match {row['fixture_id']}: {row['market']} @ prob {row['predicted_prob']:.1%}\n"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

if __name__ == "__main__":
    send_daily_picks()
