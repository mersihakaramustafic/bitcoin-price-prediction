# Bitcoin Price Prediction

A Python project that fetches historical Bitcoin price data and predicts future prices using two machine learning models: **Linear Regression** and **ARIMA**.

## Overview

The project pulls the last 365 days of Bitcoin price data from the CoinGecko API and uses it to forecast prices for the next 30 days. Both models produce visualizations of historical vs. predicted prices.

## Models

- **Linear Regression** — fits a trend line to historical data using day index as the feature, then extrapolates future prices.
- **ARIMA (5, 1, 0)** — a time series model that captures autocorrelation in price data and produces forecasts with confidence intervals.

## Project Structure

```
bitcoin-price-prediction/
├── app.py          # Main application logic (data fetching, modeling, visualization)
├── constants.py    # API URL and currency configuration
├── README.md
└── .gitignore
```

## Requirements

- Python 3.x
- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `statsmodels`

Install dependencies:

```bash
pip install requests pandas numpy matplotlib scikit-learn statsmodels
```

## Usage

```bash
python app.py
```

This will:
1. Fetch the last year of Bitcoin prices (USD) from CoinGecko
2. Train and evaluate both models
3. Print the Mean Absolute Error for Linear Regression
4. Display prediction charts for both models
5. Print the 30-day forecast tables

## Data Source

[CoinGecko API](https://www.coingecko.com/en/api) — `market_chart/range` endpoint, no API key required for basic usage.
