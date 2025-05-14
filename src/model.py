import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file_path):
    """Carga los datos desde un archivo CSV"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def preprocess_data(df):
    """Preprocesa los datos (crea medias móviles y maneja NaN)"""
    df['MA_7'] = df['Price'].rolling(window=7).mean()
    df['MA_30'] = df['Price'].rolling(window=30).mean()

    # Eliminar filas con valores NaN
    df = df.dropna()
    
    return df

def train_model(X_train, y_train):
    """Entrena un modelo de regresión lineal"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo utilizando métricas comunes"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2

def plot_results(y_test, y_pred):
    """Genera un gráfico para comparar valores reales vs predicciones"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Valores Reales', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicciones', color='red')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre (USD)')
    plt.title('Comparación de Valores Reales vs Predicciones')
    plt.legend()
    plt.show()

def run_model():
    """Función principal para cargar los datos, entrenar y evaluar el modelo"""
    # Cargar y preprocesar los datos
    df = load_data('ruta_del_archivo/Bitcoin_History.csv')
    df = preprocess_data(df)
    
    # Definir las características (X) y la variable objetivo (y)
    X = df[['Open', 'High', 'Low', 'Volume', 'Change %', 'MA_7', 'MA_30']]
    y = df['Price']
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Entrenar el modelo
    model = train_model(X_train, y_train)
    
    # Evaluar el modelo
    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')
    
    # Graficar los resultados
    plot_results(y_test, model.predict(X_test))

