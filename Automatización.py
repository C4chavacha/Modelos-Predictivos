import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from pmdarima import auto_arima
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Generar datos sintéticos
np.random.seed(42)
periods = 60
dates = pd.date_range(start='2019-01-01', periods=periods, freq='MS')
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
forecast_arima = model_arima_fit.forecast(steps=12).round()

# Holt-Winters
model_hw = ExponentialSmoothing(train['Income'], trend='add', seasonal='add', seasonal_periods=12)
model_hw_fit = model_hw.fit()
forecast_hw = model_hw_fit.forecast(steps=12).round()

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
forecast_lstm = scaler.inverse_transform(forecast_lstm_scaled).round()

# Prophet
df_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Income': 'y'})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future_prophet = model_prophet.make_future_dataframe(periods=12, freq='MS')
forecast_prophet = model_prophet.predict(future_prophet)['yhat'].iloc[-12:].values.round()

# SARIMA
model_sarima = auto_arima(train['Income'], seasonal=True, m=12)
forecast_sarima = model_sarima.predict(n_periods=12).round()

# Ridge Regression
X_train_ridge = np.arange(len(train)).reshape(-1, 1)
y_train_ridge = train['Income'].values
X_test_ridge = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
model_ridge = Ridge()
model_ridge.fit(X_train_ridge, y_train_ridge)
forecast_ridge = model_ridge.predict(X_test_ridge).round()

# Random Forest
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train_ridge, y_train_ridge)
forecast_rf = model_rf.predict(X_test_ridge).round()

# Crear tablas de predicciones
forecast_df = pd.DataFrame({
    'Date': test.index,
    'ARIMA': forecast_arima,
    'Holt-Winters': forecast_hw,
    'LSTM': forecast_lstm.flatten(),
    'Prophet': forecast_prophet,
    'SARIMA': forecast_sarima,
    'Ridge': forecast_ridge,
    'Random Forest': forecast_rf
})

# Calcular promedios (excluyendo la columna 'Date')
forecast_df['Average'] = forecast_df[['ARIMA', 'Holt-Winters', 'LSTM', 'Prophet', 'SARIMA', 'Ridge', 'Random Forest']].mean(axis=1).round()

# Calcular totales
total_arima = forecast_df['ARIMA'].sum()
total_hw = forecast_df['Holt-Winters'].sum()
total_lstm = forecast_df['LSTM'].sum()
total_prophet = forecast_df['Prophet'].sum()
total_sarima = forecast_df['SARIMA'].sum()
total_ridge = forecast_df['Ridge'].sum()
total_rf = forecast_df['Random Forest'].sum()
total_avg = forecast_df['Average'].sum()

# Guardar gráficos
figsize = (10, 6)  # Tamaño de los gráficos

def save_plot(forecast, title, filename):
    plt.figure(figsize=figsize)
    plt.plot(train.index, train['Income'], label='Train')
    plt.plot(test.index, test['Income'], label='Test', color='orange')
    plt.plot(test.index, forecast, label='Predicted', color='green')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

save_plot(forecast_arima, 'ARIMA', 'income_predictions_arima.png')
save_plot(forecast_hw, 'Holt-Winters', 'income_predictions_hw.png')
save_plot(forecast_lstm, 'LSTM', 'income_predictions_lstm.png')
save_plot(forecast_prophet, 'Prophet', 'income_predictions_prophet.png')
save_plot(forecast_sarima, 'SARIMA', 'income_predictions_sarima.png')
save_plot(forecast_ridge, 'Ridge', 'income_predictions_ridge.png')
save_plot(forecast_rf, 'Random Forest', 'income_predictions_random_forest.png')

# Crear el contenido de la tabla en HTML
table_rows = ""
for index, row in forecast_df.iterrows():
    row_html = f"<tr><td>{row['Date'].strftime('%Y-%m')}</td>"
    for col in ['ARIMA', 'Holt-Winters', 'LSTM', 'Prophet', 'SARIMA', 'Ridge', 'Random Forest']:
        top3_class = 'class="highlight-top3"' if row[col] in forecast_df[col].nlargest(3).values else ''
        bottom3_class = 'class="highlight-bottom3"' if row[col] in forecast_df[col].nsmallest(3).values else ''
        row_html += f"<td {top3_class} {bottom3_class}>{int(row[col]):,}</td>"
    row_html += f"<td>{int(row['Average']):,}</td></tr>"
    table_rows += row_html

# Agregar fila de totales
table_rows += f"<tr><td><strong>Total</strong></td><td><strong>{int(total_arima):,}</strong></td><td><strong>{int(total_hw):,}</strong></td><td><strong>{int(total_lstm):,}</strong></td><td><strong>{int(total_prophet):,}</strong></td><td><strong>{int(total_sarima):,}</strong></td><td><strong>{int(total_ridge):,}</strong></td><td><strong>{int(total_rf):,}</strong></td><td><strong>{int(total_avg):,}</strong></td></tr>"

# Análisis de resultados
analysis_content = """
<h2>Análisis de Resultados</h2>
<p>Los diferentes modelos predictivos muestran una variabilidad significativa en sus predicciones de ingresos futuros. A continuación, se presentan algunos puntos destacados del análisis:</p>
<ul>
    <li><strong>ARIMA:</strong> Este modelo sigue de cerca las tendencias históricas, pero puede no capturar adecuadamente las fluctuaciones estacionales.</li>
    <li><strong>Holt-Winters:</strong> Este modelo maneja bien los datos con tendencias y estacionalidades, proporcionando predicciones más ajustadas en comparación con ARIMA.</li>
    <li><strong>LSTM:</strong> Como red neuronal, LSTM puede capturar patrones complejos en los datos, aunque su desempeño puede variar dependiendo de la calidad y cantidad de datos de entrenamiento.</li>
    <li><strong>Prophet:</strong> Prophet es robusto y maneja bien cambios estacionales y tendencias no lineales, mostrando predicciones consistentes.</li>
    <li><strong>SARIMA:</strong> Este modelo es una extensión de ARIMA que maneja la estacionalidad, y se comporta bien en series temporales con patrones estacionales claros.</li>
    <li><strong>Ridge Regression:</strong> La regresión Ridge maneja la multicolinealidad entre las variables predictoras, pero puede no capturar patrones no lineales tan efectivamente como otros modelos.</li>
    <li><strong>Random Forest:</strong> Este modelo de aprendizaje conjunto proporciona predicciones robustas al combinar múltiples árboles de decisión, aunque puede ser menos efectivo en series temporales con fuertes patrones estacionales.</li>
</ul>
<p>En general, los modelos que consideran componentes estacionales (como Holt-Winters y SARIMA) tienden a proporcionar predicciones más precisas para los datos con patrones estacionales marcados. Los modelos de aprendizaje automático (como LSTM y Random Forest) pueden capturar patrones más complejos, pero su efectividad depende en gran medida de la calidad y cantidad de datos de entrenamiento.</p>
"""

# Crear el contenido HTML con diseño ejecutivo y juvenil
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Predicciones de Ingresos</title>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Open+Sans:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Open Sans', sans-serif;
            background-color: #f0f0f5;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        .container {{
            width: 90%;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #004d99;
            font-family: 'Montserrat', sans-serif;
            text-align: center;
        }}
        h2 {{
            color: #004d99;
            font-family: 'Montserrat', sans-serif;
            font-size: 1.5em;
            text-align: center;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            align-items: start;
        }}
        .grid-item {{
            text-align: center;
        }}
        .grid-item p {{
            text-align: center;
            padding: 0 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #004d99;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .highlight-top3 {{
            background-color: #FFD700;
        }}
        .highlight-bottom3 {{
            background-color: #FFB6C1;
        }}
        img {{
            width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Predicciones de Ingresos Mensuales</h1>
        <div class="grid-container">
            <div class="grid-item">
                <h2>ARIMA</h2>
                <img src="income_predictions_arima.png" alt="Income Predictions with ARIMA">
                <p>El modelo ARIMA predice los ingresos futuros basándose en el patrón y las tendencias de los ingresos históricos.</p>
            </div>
            <div class="grid-item">
                <h2>Holt-Winters</h2>
                <img src="income_predictions_hw.png" alt="Income Predictions with Holt-Winters">
                <p>Este modelo es adecuado para datos con tendencias y patrones estacionales, y predice los ingresos futuros considerando estos componentes.</p>
            </div>
            <div class="grid-item">
                <h2>LSTM</h2>
                <img src="income_predictions_lstm.png" alt="Income Predictions with LSTM">
                <p>LSTM predice los ingresos futuros al aprender patrones complejos en los datos históricos.</p>
            </div>
            <div class="grid-item">
                <h2>Prophet</h2>
                <img src="income_predictions_prophet.png" alt="Income Predictions with Prophet">
                <p>Prophet es un modelo desarrollado por Facebook para la predicción de series temporales con datos históricos, utilizando componentes aditivos y multiplicativos.</p>
            </div>
            <div class="grid-item">
                <h2>SARIMA</h2>
                <img src="income_predictions_sarima.png" alt="Income Predictions with SARIMA">
                <p>SARIMA es una extensión del modelo ARIMA que soporta la estacionalidad, útil para series temporales con patrones estacionales claros.</p>
            </div>
            <div class="grid-item">
                <h2>Ridge</h2>
                <img src="income_predictions_ridge.png" alt="Income Predictions with Ridge">
                <p>Ridge Regression es una técnica de regresión lineal que utiliza regularización para manejar la multicolinealidad entre variables predictoras.</p>
            </div>
            <div class="grid-item">
                <h2>Random Forest</h2>
                <img src="income_predictions_random_forest.png" alt="Income Predictions with Random Forest">
                <p>Random Forest es un método de aprendizaje conjunto para la clasificación y la regresión que construye múltiples árboles de decisión durante el entrenamiento y da como salida la media de las predicciones de los árboles individuales.</p>
            </div>
        </div>
        <h2>Explicación de las Líneas de Train y Test</h2>
        <p>Las líneas de "Train" (azul) representan los datos históricos utilizados para entrenar los modelos predictivos. Las líneas de "Test" (naranja) representan los datos reales utilizados para evaluar la precisión de las predicciones de los modelos. Las líneas de "Predicted" (verde) muestran las predicciones de ingresos futuros generadas por cada modelo.</p>
        <h2>Predicciones de Modelos</h2>
        <table>
            <tr>
                <th>Fecha</th>
                <th>ARIMA</th>
                <th>Holt-Winters</th>
                <th>LSTM</th>
                <th>Prophet</th>
                <th>SARIMA</th>
                <th>Ridge</th>
                <th>Random Forest</th>
                <th>Promedio</th>
            </tr>
            {table_rows}
        </table>
        {analysis_content}
    </div>
</body>
</html>
"""

# Guardar el contenido HTML en un archivo
with open("predicciones.html", "w") as file:
    file.write(html_content)

print("Archivo predicciones.html creado con éxito.")
