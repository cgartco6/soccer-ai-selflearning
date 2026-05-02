#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        print("Message sent")
    except Exception as e:
        print(f"Failed: {e}")

def send_daily_picks():
    conn = sqlite3.connect("data/betting.db")
    df = pd.read_sql_query("""
        SELECT fixture_id, market, predicted_prob, odds, edge, prediction_time
        FROM predictions
        ORDER BY prediction_time DESC
        LIMIT 10
    """, conn)
    if df.empty:
        send_telegram("No predictions for today.")
        return
    message = "*Today's AI Picks*\n"
    for _, row in df.iterrows():
        message += f"🔹 {row['fixture_id']} | {row['market']} | prob {row['predicted_prob']:.1%} | odds {row['odds']:.2f} | edge {row['edge']:.1%}\n"
    send_telegram(message)

if __name__ == "__main__":
    send_daily_picks()
