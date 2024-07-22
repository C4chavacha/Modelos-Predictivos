import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
np.random.seed(42)
periods = 60
dates = pd.date_range(start='2019-01-01', periods=periods, freq='M')
income = np.random.normal(loc=10000, scale=2000, size=periods)  # Ingresos medios con algo de ruido

# Crear DataFrame
data = pd.DataFrame({'Date': dates, 'Income': income})

# Mostrar los primeros datos
print(data.head())

# Guardar en un archivo CSV
data.to_csv('monthly_income.csv', index=False)
