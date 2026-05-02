from flask import Flask, render_template_string, jsonify
import sqlite3
import pandas as pd

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Soccer AI Predictions</title></head>
<body>
<h1>Live Predictions</h1>
<table border="1">
    <tr><th>Fixture</th><th>Market</th><th>Prob</th><th>Odds</th><th>Edge</th></tr>
    {% for row in data %}
    <tr>
        <td>{{ row.fixture_id }}</td>
        <td>{{ row.market }}</td>
        <td>{{ "%.1f"|format(row.predicted_prob*100) }}%</td>
        <td>{{ row.odds }}</td>
        <td>{{ "%.1f"|format(row.edge*100) }}%</td>
    </tr>
    {% endfor %}
</table>
</body>
</html>
"""

@app.route('/')
def index():
    conn = sqlite3.connect("data/betting.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 50", conn)
    data = df.to_dict(orient="records")
    return render_template_string(HTML_TEMPLATE, data=data)

@app.route('/api/predictions')
def api():
    conn = sqlite3.connect("data/betting.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 50", conn)
    return jsonify(df.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
