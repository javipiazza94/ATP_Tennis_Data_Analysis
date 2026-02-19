import glob
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
    print(len(archivos))
    lista = []

    for csv in archivos:
        df = pd.read_csv(csv, index_col=None, header=0)
        lista.append(df)

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
        ruta_completa = f"{ruta_destino}/{nombre_archivo}"
        df.to_csv(ruta_completa, index=False)
        print(f"El DataFrame se ha guardado correctamente en '{ruta_completa}'.")
    except Exception as e:
        print(f"Error al guardar el DataFrame en '{ruta_destino}': {str(e)}")




def entrenar_modelos_SINSPLIT_y_guardar(df, nombre_dataset:str, columna_objetivo:str, directorio_script:str):
         """
    Entrena varios modelos utilizando los datos proporcionados, guarda los modelos entrenados y devuelve un DataFrame con los resultados.

    Argumentos:
    - df (DataFrame): DataFrame que contiene los datos de entrenamiento.
    - nombre_dataset (str): Nombre del conjunto de datos.
    - columna_objetivo (str): Nombre de la columna de la variable objetivo.

    Retorna:
    - resultados (DataFrame): DataFrame con los resultados de cada modelo.
    """
         
    # Dividir los datos en características (X) y etiquetas (y)
         X = df.drop(columns=[columna_objetivo])
         y = df[columna_objetivo]

    # Inicializar los modelos
         modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'AdaBoost': AdaBoostClassifier(),
        'Regresión logística': LogisticRegression(max_iter=500),
        'Bagging': BaggingClassifier(n_estimators=100)
    }

    # Evaluar cada modelo
         resultados = []
         for nombre_modelo, modelo in modelos.items():
            # Entrenar el modelo
            modelo.fit(X, y)      
            # Calcular la precisión del modelo
            precision = modelo.score(X, y)
            # Guardar el modelo entrenado
            nombre_modelo_archivo = f'{nombre_dataset}_{nombre_modelo}_model.pkl'
            guardar_pickle(modelo, nombre_modelo_archivo, directorio_script)
            print(f"El modelo '{nombre_modelo}' ha sido entrenado y guardado como '{nombre_modelo_archivo}'.")
            
            resultados.append({'Modelo': nombre_modelo, 'Resultado': precision})  

        # Convertir los resultados a DataFrame
         resultados_df = pd.DataFrame(resultados)

         return resultados_df




def entrenar_XGBOOST(df, nombre_dataset:str, columna_objetivo:str, directorio_script:str):
    """
    Entrena varios modelos utilizando los datos proporcionados, guarda los modelos entrenados y devuelve un DataFrame con los resultados.

    Argumentos:
    - df (DataFrame): DataFrame que contiene los datos de entrenamiento.
    - nombre_dataset (str): Nombre del conjunto de datos.
    - columna_objetivo (str): Nombre de la columna de la variable objetivo.

    Retorna:
    - resultados (DataFrame): DataFrame con los resultados de cada modelo.
    """
    # Dividir los datos en características (X) y etiquetas (y)
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]

    # Inicializar el modelo XGBoost
    modelo = XGBClassifier()

    # Inicializar el codificador
    label_encoder = LabelEncoder()

    # Convertir la variable categórica a valores numéricos discretos
    y_encoded = label_encoder.fit_transform(y)

    # Entrenar el modelo con las etiquetas codificadas
    modelo.fit(X, y_encoded)

    # Hacer predicciones en el conjunto de datos de prueba
    predicciones= modelo.predict(X)

    #Pillar la precision
    xgb_accuracy = accuracy_score(y_encoded, predicciones)

    # Guardar el modelo entrenado
    nombre_modelo_archivo = f'{nombre_dataset}_XGBoost_model.pkl'
    guardar_pickle(modelo, nombre_modelo_archivo, directorio_script)
    print(f"El modelo 'XGBoost' ha sido entrenado y guardado como '{nombre_modelo_archivo}'.")

    # Crear el DataFrame de resultados
    resultados_df = pd.DataFrame({'Modelo': ['XGBoost'], 'Resultado': [xgb_accuracy]})

    return resultados_df
