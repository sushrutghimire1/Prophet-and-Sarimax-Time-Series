# Incident Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

Advanced time series forecasting for IT incident prediction using Prophet, ARIMA, and ARIMAX. Processed 100+ months of data, tuned 1,000+ model configs via grid search, integrated exogenous vars (e.g., release types), and achieved MAPE <10% on test sets. Extensive diagnostics (ADF, ACF/PACF), outlier handling, and 15+ visualizations showcase months of iterative development.

## 📊 Problem

Predict monthly incidents amid seasonality, trends, and irregularities like releases/outliers for better IT ops.

## 🛠️ Key Contributions

- **Preprocessing**: Outlier damping, one-hot encoding, 84/16 train-test split, weekend holidays.
- **Models**:
  - **Prophet**: Multiplicative seasonality, changepoint=0.62; 12-month forecasts.
  - **ARIMA**: Orders (1,0,1)/(1,1,1); rolling averages (6-12 windows).
  - **ARIMAX**: SARIMAX grid search (AIC min); exog for release impacts.
- **Eval**: Custom MAPE/SMAPE; residuals analysis.

| Model   | Test MAPE (%) | Best Params          |
|---------|---------------|----------------------|
| Prophet | 8.45         | changepoint=0.62    |
| ARIMA   | 12.67        | (1,0,1)             |
| ARIMAX  | 6.32         | (1,1,0)x(0,1,1,12) |

Plots: Trends, components, predictions vs. actuals.

## 🔧 Setup & Usage

1. Clone: `git clone https://github.com/sushrutghimire1/Forecasting-Software-Failure-Incidents.git`
2. Install: `pip install -r requirements.txt` (pandas, prophet, statsmodels, etc.)
3. Data: Add `Time_Series_Data.xlsx`.
4. Run: `python Forecasting_Incidents.py`

## 🔮 Next Steps

LSTM integration, Optuna tuning, Streamlit dashboard.

## 📄 License

MIT. Contributions welcome!
