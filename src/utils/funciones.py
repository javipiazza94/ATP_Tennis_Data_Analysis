import glob

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os
import pickle

def guardar_pickle(objeto, nombre_archivo:str, directorio=None):
    """
    Guarda un objeto en formato pickle.

    Argumentos:
    - objeto: Objeto que se desea guardar.
    - nombre_archivo: Nombre del archivo pickle.
    - directorio (opcional): Directorio donde se guardará el archivo pickle. Si no se especifica, se guardará en el directorio actual.
    """
    if directorio:
        os.makedirs(directorio, exist_ok=True)
        ruta_completa = os.path.join(directorio, nombre_archivo)
    else:
        ruta_completa = nombre_archivo

    try:
        with open(ruta_completa, 'wb') as archivo:
            pickle.dump(objeto, archivo)
        print(f"El objeto ha sido guardado como '{ruta_completa}'.")
    except Exception as e:
        print(f"Error al guardar el objeto como '{ruta_completa}': {str(e)}")



def ConcatenarCSV(ruta):
    """
    Concatena varios archivos CSV en uno solo.

    Argumentos:
    - ruta: Ruta del directorio que contiene los archivos CSV a concatenar.

    Retorna:
    - DataFrame: Un DataFrame que contiene todos los datos de los archivos CSV concatenados.
    """
    archivos = glob.glob(ruta + "/*.csv")
    print(f"Found {len(archivos)} files.")
    lista = []

    for csv in archivos:
        df = pd.read_csv(csv, index_col=None, header=0)
        lista.append(df)

    if not lista:
        return pd.DataFrame()
        
    df = pd.concat(lista, axis=0, ignore_index=True)
    return df



def guardar_csv(df, nombre_archivo:str, ruta_destino:str):
    """
    Guarda un DataFrame en un archivo CSV.

    Argumentos:
    - df: DataFrame de Pandas a guardar.
    - nombre_archivo: Nombre del archivo CSV en el que se guardará el DataFrame.
    - ruta_destino: Ruta donde se guardará el archivo CSV.
    """
    try:
        os.makedirs(ruta_destino, exist_ok=True)
        ruta_completa = os.path.join(ruta_destino, nombre_archivo)
        df.to_csv(ruta_completa, index=False)
        print(f"El DataFrame se ha guardado correctamente en '{ruta_completa}'.")
    except Exception as e:
        print(f"Error al guardar el DataFrame en '{ruta_destino}': {str(e)}")




def entrenar_modelos(df, nombre_dataset:str, columna_objetivo:str, directorio_script:str):
    """
    Entrena varios modelos utilizando los datos proporcionados, guarda los modelos entrenados y devuelve un DataFrame con los resultados.
    
    Uses an 80/20 Train/Test split to evaluate performance.

    Argumentos:
    - df (DataFrame): DataFrame que contiene los datos de entrenamiento.
    - nombre_dataset (str): Nombre del conjunto de datos.
    - columna_objetivo (str): Nombre de la columna de la variable objetivo.
    - directorio_script (str): Directorio donde se guardarán los modelos.

    Retorna:
    - resultados (DataFrame): DataFrame con los resultados de cada modelo.
    """
         
    # Dividir los datos en características (X) y etiquetas (y)
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Inicializar los modelos
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Regresión logística': LogisticRegression(max_iter=1000, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Evaluar cada modelo
    resultados = []
    
    # Initialize encoder just in case (though XGBoost handles it, it's safer if Y is categorical)
    # However, our target is now 0/1 integers, so encoding issues should be minimal.
    
    for nombre_modelo, modelo in modelos.items():
        print(f"Training {nombre_modelo}...")
        # Entrenar el modelo
        modelo.fit(X_train, y_train)      
        
        # Calcular la precisión del modelo en TEST set
        precision = modelo.score(X_test, y_test)
        
        # Guardar el modelo entrenado
        nombre_modelo_archivo = f'{nombre_dataset}_{nombre_modelo}_model.pkl'
        guardar_pickle(modelo, nombre_modelo_archivo, directorio_script)
        print(f"El modelo '{nombre_modelo}' ha sido entrenado. Accuracy en Test: {precision:.4f}")
        
        resultados.append({'Modelo': nombre_modelo, 'Resultado': precision})  

    # Convertir los resultados a DataFrame
    resultados_df = pd.DataFrame(resultados)

    return resultados_df
