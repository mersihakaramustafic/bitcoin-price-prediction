import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def fetch_bitcoin_data_one_year():

    # Calculate the UNIX timestamps for the past year
    end_ts = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())

    # CoinGecko API URL for Bitcoin market data
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",  # Convert to USD
        "from": start_ts,  # Start timestamp
        "to": end_ts,  # End timestamp
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

def prepare_features(data):

    # Fatures (X) and target (y)    
    data["day"] = (data["timestamp"] - data["timestamp"].min()).dt.days
    X = data[["day"]]
    y = data["price"]
    return X, y

def main():

    # Fetch data for the past year
    bitcoin_data = fetch_bitcoin_data_one_year()

    if bitcoin_data is not None:
        # Save to CSV
        bitcoin_data.to_csv("bitcoin_last_year_prices.csv", index=False)
        print("Data saved to 'bitcoin_last_year_prices.csv'")
        print(bitcoin_data.head())

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(bitcoin_data["timestamp"], bitcoin_data["price"], label="Bitcoin Price (USD)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Bitcoin Price Over the Last Year")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
