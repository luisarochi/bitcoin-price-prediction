import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('ruta_del_archivo/Bitcoin_History.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df.isnull().sum()
df.info()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Precio de Cierre', color='blue')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.title('Evolución del Precio de Cierre de Bitcoin')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Change %'], label='Cambio Porcentual Diario', color='red')
plt.xlabel('Fecha')
plt.ylabel('Cambio Porcentual Diario')
plt.title('Volatilidad Diaria de Bitcoin')
plt.grid(True)
plt.legend()
plt.show()

df['MA_7'] = df['Price'].rolling(window=7).mean()
df['MA_30'] = df['Price'].rolling(window=30).mean()

df[['Date', 'Price', 'MA_7', 'MA_30']].tail(10)
