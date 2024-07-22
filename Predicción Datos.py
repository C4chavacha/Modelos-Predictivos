import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Generar datos sintéticos
np.random.seed(42)
periods = 60
dates = pd.date_range(start='2019-01-01', periods=periods, freq='M')
income = np.random.normal(loc=10000, scale=2000, size=periods)
data = pd.DataFrame({'Date': dates, 'Income': income})
data.to_csv('monthly_income.csv', index=False)

# Cargar datos
data = pd.read_csv('monthly_income.csv', parse_dates=['Date'], index_col='Date')
train = data[:48]
test = data[48:]

# ARIMA
model_arima = ARIMA(train['Income'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=12)

# Holt-Winters
model_hw = ExponentialSmoothing(train['Income'], trend='add', seasonal='add', seasonal_periods=12)
model_hw_fit = model_hw.fit()
forecast_hw = model_hw_fit.forecast(steps=12)

# LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train['Income'].values.reshape(-1, 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
X_train = []
y_train = []
for i in range(1, len(scaled_data)):
    X_train.append(scaled_data[i-1:i])
    y_train.append(scaled_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)
model_lstm.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)

inputs = data[len(data) - len(test) - 1:]['Income'].values.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(1, len(inputs)):
    X_test.append(inputs[i-1:i])
X_test = np.array(X_test)
forecast_lstm_scaled = model_lstm.predict(X_test)
forecast_lstm = scaler.inverse_transform(forecast_lstm_scaled)

# Visualización
plt.figure(figsize=(14, 8))
plt.plot(train['Income'], label='Train')
plt.plot(test['Income'], label='Test')
plt.plot(test.index, forecast_arima, label='ARIMA')
plt.plot(test.index, forecast_hw, label='Holt-Winters')
plt.plot(test.index, forecast_lstm, label='LSTM')
plt.legend()
plt.title('Income Prediction')
plt.xlabel('Date')
plt.ylabel('Income')
plt.savefig('income_predictions.png')
plt.show()

# Crear tablas de predicciones
forecast_df = pd.DataFrame({
    'Date': test.index,
    'ARIMA': forecast_arima,
    'Holt-Winters': forecast_hw,
    'LSTM': forecast_lstm.flatten()
})
forecast_df.to_csv('forecast_predictions.csv', index=False)
print(forecast_df)
