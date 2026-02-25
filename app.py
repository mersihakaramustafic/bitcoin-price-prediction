import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import constants as c
from statsmodels.tsa.arima.model import ARIMA


def fetch_bitcoin_data_one_year():
    end_ts = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())

    url = c.coingecko_url
    params = {
        "vs_currency": c.currency,
        "from": start_ts,
        "to": end_ts,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        print("Error fetching data:", response.status_code, response.text)
        return None


def prepare_features(data):
    df = data.reset_index()
    df["day"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    X = df[["day"]]
    y = df["price"]
    return X, y


def predict_prices_by_linear_regression(X, y, future_days=30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (Linear Regression): {mae:.2f}")

    current_day = X["day"].max() + 1
    future_X = np.arange(current_day, current_day + future_days).reshape(-1, 1)
    future_prices = model.predict(future_X)

    future_dates = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(future_days)]
    return future_dates, future_prices


def predict_prices_by_arima(data, order=(5, 1, 0), forecast_steps=30):
    print("Training ARIMA model...")
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()

    future_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, forecast_steps + 1)]
    forecast_df.index = future_dates

    return forecast_df


def visualize_linear_regression(bitcoin_data, future_dates, future_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(bitcoin_data.index, bitcoin_data["price"], label="Historical Prices")
    plt.plot(future_dates, future_prices, label="Predicted Prices (Linear Regression)", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Prediction - Linear Regression (Next 30 Days)")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_arima(bitcoin_data, forecast_df):
    plt.figure(figsize=(12, 6))
    plt.plot(bitcoin_data.index, bitcoin_data["price"], label="Historical Prices")
    plt.plot(forecast_df.index, forecast_df["mean"], label="Forecasted Prices (ARIMA)", linestyle="--")
    plt.fill_between(
        forecast_df.index,
        forecast_df["mean_ci_lower"],
        forecast_df["mean_ci_upper"],
        color="gray", alpha=0.3, label="Confidence Interval"
    )
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Prediction - ARIMA (Next 30 Days)")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    bitcoin_data = fetch_bitcoin_data_one_year()

    if bitcoin_data is None:
        return

    # ---- Linear Regression ----
    X, y = prepare_features(bitcoin_data)
    future_dates_lr, future_prices_lr = predict_prices_by_linear_regression(X, y, future_days=30)
    visualize_linear_regression(bitcoin_data, future_dates_lr, future_prices_lr)

    prediction_df = pd.DataFrame({
        "Date": future_dates_lr,
        "Predicted Price (USD) by Linear Regression": future_prices_lr
    })
    print(prediction_df)

    # ---- ARIMA ----
    forecast_df = predict_prices_by_arima(bitcoin_data["price"], order=(5, 1, 0))
    visualize_arima(bitcoin_data, forecast_df)
    print(forecast_df[["mean", "mean_ci_lower", "mean_ci_upper"]])


if __name__ == "__main__":
    main()
