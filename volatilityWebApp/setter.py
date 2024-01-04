import requests

res = requests.post("http://127.0.0.1:5000/data", json={"tickers": ["AAPL", "TSLA"]})

print(res.json())