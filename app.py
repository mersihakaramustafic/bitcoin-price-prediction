import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def fetch_bitcoin_data(start_date, end_date):
    # Convert dates to Unix timestamps
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Fetch data from CoinGecko API
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Extract price data
        prices = data.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        print("Error fetching data:", response.status_code, response.text)
        return None

# Fetch data for a specific range
start_date = "2024-01-01"
end_date = "2024-12-01"
bitcoin_data = fetch_bitcoin_data(start_date, end_date)

# Save to CSV and visualize
if bitcoin_data is not None:
    bitcoin_data.to_csv("bitcoin_prices.csv", index=False)
    print(bitcoin_data.head())

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(bitcoin_data["timestamp"], bitcoin_data["price"], label="Bitcoin Price (USD)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Over Time")
    plt.legend()
    plt.grid()
    plt.show()
