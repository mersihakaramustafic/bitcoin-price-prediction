import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import constants as c

def fetch_bitcoin_data_one_year():

    # Calculate the UNIX timestamps for the past year from the current date
    end_ts = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())

    # CoinGecko API URL for Bitcoin market data
    url = c.coingecko_url
    params = {
        "vs_currency": c.currency,  # Convert to USD
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

def predict_prices(X, y, future_days=30):

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on test data: {mae:.2f}")

    # Predict future prices
    current_day = X["day"].max() + 1
    future_days = np.arange(current_day, current_day + future_days).reshape(-1, 1)
    future_prices = model.predict(future_days)

    return future_days, future_prices

def main():

    # Fetch data for the past year
    bitcoin_data = fetch_bitcoin_data_one_year()

    if bitcoin_data is not None:
        X, y = prepare_features(bitcoin_data)
        future_days, future_prices = predict_prices(X, y, future_days=30)

        # Generate future dates starting from today
        start_date = datetime.datetime.now()
        future_dates = [start_date + datetime.timedelta(days=i) for i in range(30)]

        # Visualize predictions
        plt.figure(figsize=(12, 6))
        plt.plot(bitcoin_data["timestamp"], bitcoin_data["price"], label="Historical Prices")
        plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Bitcoin Price Prediction for the Next 30 Days")
        plt.legend()
        plt.grid()
        plt.show()

        # Display predicted values
        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price (USD)": future_prices
        })
        print(prediction_df)

if __name__ == "__main__":
    main()
