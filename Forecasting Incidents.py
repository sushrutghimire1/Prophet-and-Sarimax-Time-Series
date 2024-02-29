#!/usr/bin/env python
# coding: utf-8

# In[285]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('Time_Series_Data.xlsx',sheet_name='Raw Data')


# In[286]:


df.head()


# In[287]:


outlier_index = df['No.of Incidents'].idxmax()
df_old= df
max_value = df['No.of Incidents'].max()
# df.at[outlier_index, 'No.of Incidents'] = max_value / 1.3

max_value


# In[288]:


# Plot the original and updated time series data
plt.figure(figsize=(12, 6))
plt.plot(df['Month_Year'], df['No.of Incidents'], label='Original Data', linestyle='-', marker='o', color='blue')
plt.scatter(df.loc[outlier_index, 'Month_Year'], df.loc[outlier_index, 'No.of Incidents'], color='red', marker='x', s=100, label='Outlier')

# Plot the updated values

plt.xlim(min(df_old['Month_Year']), max(df_old['Month_Year']))
plt.title('Comparison of Original and Updated Time Series Data')
plt.legend()
plt.show()


# In[ ]:


train_size = int(len(df) * 0.84)  # 80% for training
train_df, test_df = df[:train_size], df[train_size:]


# In[ ]:


weekends_df = pd.DataFrame({
    'holiday': 'weekend',
    'ds': pd.date_range(start='2022-01-01', end='2023-12-31', freq='B'),  # B frequency for business days
    'lower_window': 0,
    'upper_window': 1,
})

# Mark Saturday and Sunday as holidays (1) and weekdays as non-holidays (0)
weekends_df['is_weekend'] = weekends_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)

# Filter only the rows where the day is a weekend
holidays_df = weekends_df[weekends_df['is_weekend'] == 1][['holiday', 'ds', 'lower_window', 'upper_window']]


# In[ ]:


# Rename columns to 'ds' and 'y'
train_df.rename(columns={'Month_Year': 'ds', 'No.of Incidents': 'y'}, inplace=True)

# Initialize Prophet model
model = Prophet(
changepoint_prior_scale=0.62,  
    yearly_seasonality=True,        
    weekly_seasonality=False,   
#     holidays=holidays_df,
    seasonality_mode='multiplicative' 
)

# Fit the model with the modified dataset
model.fit(train_df)


# In[ ]:


# Create a DataFrame with future dates for forecasting
future = model.make_future_dataframe(periods=len(test_df),freq='MS')  # Assuming you want to forecast 12 months ahead


# In[ ]:


# Make predictions for the future dates
forecast = model.predict(future)


# In[ ]:


# Plot the forecast along with original data and uncertainty intervals
fig = model.plot(forecast)
plt.title('Prophet Forecast with Uncertainty Intervals')
plt.show()


# In[ ]:


# Visualize the components of the forecast (trend, weekly, and yearly)
fig = model.plot_components(forecast)
plt.show()


# In[ ]:


# Extract actual and predicted values for the test period
test_df.rename(columns={'Month_Year': 'ds', 'No.of Incidents': 'y'}, inplace=True)

actual_values_test = test_df['y'].values
predicted_values_test = forecast['yhat'].values[-len(test_df):]

actual_values_train = train_df['y'].values
predicted_values_train = forecast['yhat'].values[-len(train_df):]

# Calculate MAPE on the test set
def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_test = calculate_mape(actual_values_test, predicted_values_test)

print(f'MAPE on the test set: {mape_test:.2f}%')


# In[ ]:


# Plot predicted vs original values
plt.figure(figsize=(12, 6))

plt.plot(test_df['ds'], actual_values_test, label='Original Data', linestyle='-', marker='o', color='red')
plt.plot(test_df['ds'], predicted_values_test, label='Predicted Data', linestyle='--', marker='x', color='green')

plt.plot(train_df['ds'], actual_values_train, label='Original Data', linestyle='-', marker='o', color='blue')
plt.plot(train_df['ds'], predicted_values_train, label='Predicted Data', linestyle='--', marker='x', color='green')

plt.title('Original vs Predicted Time Series Data')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


##ARIMA


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
def moving_average_arima(df, column, window):
    # Fit ARIMA model
    model = ARIMA(df[column], order=(1, 1, 1))  # ARIMA(0,0,0) corresponds to a simple moving average
    result = model.fit()

    # Get the fitted values
    fitted_values = result.fittedvalues

    # Calculate the moving average
    moving_average = fitted_values.rolling(window=window).mean()

    return moving_average,fitted_values

window_size=6
df['moving_avg_arima'],fitted_values = moving_average_arima(df, 'No.of Incidents', window_size)

# Plot the original data and the moving average
plt.plot(df['No.of Incidents'], label='Original Data')
plt.plot(df['moving_avg_arima'], label=f'Moving Average (window={window_size}) using ARIMA', linestyle='--')
plt.legend()
plt.title('Moving Average using ARIMA Model')
plt.show()

from sklearn.metrics import mean_absolute_error
mape_value = mean_absolute_error(df['No.of Incidents'], fitted_values)
print(f'MAPE: {mape_value:.2f}%')

order=(1, 0, 1)


# Plot the original data and the fitted values
plt.plot(df['No.of Incidents'], label='Original Data')
plt.plot(fitted_values, label=f'Fitted Values (ARIMA{order})', linestyle='--')
plt.legend()
plt.title(f'ARIMA Model with Non-Zero Orders {order} - MAPE: {mape_value:.2f}%')
plt.show()


# In[ ]:


# Train-test split
train_size = int(len(df) * 0.84)
train, test = df[:train_size].reset_index(drop=True), df[train_size:].reset_index(drop=True)

# Function to fit ARIMA model and calculate moving average
def moving_average_arima(train, test, column, window, order):
    # Concatenate train and test for simplicity
    combined = pd.concat([train, test], ignore_index=True)

    # Fit ARIMA model on the combined data
    model = ARIMA(combined[column], order=order)
    result = model.fit()

    # Get the fitted values on the combined data
    fitted_values_combined = result.fittedvalues

    # Calculate the moving average on the combined data
    moving_average_combined = fitted_values_combined.rolling(window=window).mean()

    # Split the results back into train and test
    moving_average_train = moving_average_combined.iloc[:len(train)]
    moving_average_test = moving_average_combined.iloc[len(train):]

    return moving_average_train, moving_average_test, result

window_size = 12
order = (1, 0, 1)

# Calculate moving averages for training and test sets
moving_avg_train, moving_avg_test, result = moving_average_arima(train, test, 'No.of Incidents', window_size, order)

# Plot the original data, predicted values, and the moving averages on the training set
plt.plot(train['No.of Incidents'], label='Training Data')
plt.plot(moving_avg_train, label=f'Moving Average (window={window_size}) on Training Set', linestyle='--')
plt.plot(result.fittedvalues[:len(train)], label='Fitted Values on Training Set', linestyle='-.')
plt.legend()
plt.title('ARIMA Model on Training Set')
plt.show()

# Plot the original data, predicted values, and the moving averages on the test set
plt.plot(test.index, test['No.of Incidents'], label='Test Data')
plt.plot(test.index, moving_avg_test, label=f'Moving Average (window={window_size}) on Test Set', linestyle='--')
plt.plot(test.index, result.fittedvalues[len(train):], label='Fitted Values on Test Set', linestyle='-.')
plt.legend()
plt.title('ARIMA Model on Test Set')
plt.show()

# Calculate MAPE for the test set
mape_value_test = mean_absolute_error(test['No.of Incidents'], result.fittedvalues[len(train):])
print(f'MAPE on Test Set: {mape_value_test:.2f}%')


# In[ ]:


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(test['No.of Incidents'],result.fittedvalues[len(train):])


# In[ ]:


result.fittedvalues[len(train):]


# In[ ]:


test['No.of Incidents']


# In[ ]:





# In[ ]:


##ARIMAX


# In[461]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid


# In[462]:


df = pd.read_excel('Time_Series_Data.xlsx',sheet_name='Raw Data',index_col='Month_Year')


# In[463]:


df.head()


# In[464]:


train_size = int(len(df) * 0.84)  # 80% for training
train_df, test_df = df[:train_size], df[train_size:]


# In[465]:


ts_train = train_df[['No.of Incidents']]
ts_test = test_df[['No.of Incidents']]


# In[466]:


from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])


# In[467]:


adf_test(ts)


# In[468]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF
plot_acf(ts)
plot_pacf(ts,lags=7)
plt.show()


# In[ ]:





# In[469]:


exogenous_data = df[['Release Type']]
# #'Client Owned','Release Type','Major Release Flag'


# In[470]:


exogenous_data.head()


# In[471]:


encoded = pd.get_dummies(df[['Release Type']], prefix='category')


# In[472]:


encoded


# In[473]:


# encoded = pd.get_dummies(df[['Major Release Flag']], prefix='category')
# exogenous_data = pd.concat([exogenous_data, encoded], axis=1)


# In[474]:


# df['Client Owned'] = df['Client Owned'].astype('category')


# In[475]:


# encoded = pd.get_dummies(df[['Client Owned']], prefix='category')
# exogenous_data = pd.concat([exogenous_data, encoded], axis=1)


# In[476]:


exogenous_data.head()


# In[477]:


train_exogenous_data, test_exogenous_data = encoded[:train_size], encoded[train_size:]


# In[478]:


train_exogenous_data


# In[479]:


ts_train


# In[480]:


# import numpy as np
# from sklearn.base import BaseEstimator, RegressorMixin
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# class SARIMAXWrapper(BaseEstimator, RegressorMixin):
#     def __init__(self, endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
#         self.endog = endog
#         self.exog = exog
#         self.order = order
#         self.seasonal_order = seasonal_order
#         self.model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)

#     def fit(self, X, y):
#         self.model = self.model.fit()
#         return self

#     def predict(self, X):
#         return self.model.predict(start=X.index[0], end=X.index[-1], exog=X)

# # Example usage
# from sklearn.feature_selection import RFE

# # Assuming y is your target variable (the time series you want to predict)
# y = ts_train  # Replace with your actual target variable

# # Assuming X is your feature matrix, including exogenous variables
# X = train_exogenous_data  # Replace with your actual feature matrix

# # SARIMAX model with wrapper
# sarimax_wrapper = SARIMAXWrapper(endog=ts_train, exog=train_exogenous_data)

# # Recursive Feature Elimination (RFE)
# selector = RFE(sarimax_wrapper)
# selector = selector.fit(X, y)

# # Get relevant exogenous variables
# relevant_exog_variables = X.columns[selector.support_]
# print("Relevant Exogenous Variables:", relevant_exog_variables)


# In[481]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# Define the parameter grid
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Grid search for ARIMA
best_aic = float("inf")
best_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(ts_train, order=param, seasonal_order=param_seasonal, exog=train_exogenous_data)
            results = mod.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, param_seasonal)
        except:
            continue

print('Best AIC:', best_aic)
print('Best Parameters:', best_params)


# In[482]:


mod = SARIMAX(ts_train,best_params[0], seasonal_order=best_params[1], exog=train_exogenous_data)
results = mod.fit()


# In[ ]:


forecast_steps = 4  # Set the number of steps to forecast
forecast = results.get_forecast(steps=forecast_steps, exog=test_exogenous_data)
predicted_values = forecast.predicted_mean


# In[ ]:


predicted_values.plot()
ts_test['No.of Incidents'].plot()


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(ts_test['No.of Incidents'],predicted_values)

