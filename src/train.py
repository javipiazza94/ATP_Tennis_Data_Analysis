import pandas as pd
import sys
import os

# Obtiene la ruta absoluta del directorio que contiene este script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Agregar directorio utils al path
sys.path.append(os.path.join(directorio_actual, 'utils'))
from funciones import entrenar_modelos_SINSPLIT_y_guardar, guardar_csv

# Carga el conjunto de datos
ruta_data = os.path.join(directorio_actual, 'data', 'processed', 'dfLimpio.csv')
df = pd.read_csv(ruta_data)

print(df.head(10))

# Especifica el nombre del conjunto de datos y la columna objetivo
nombre_dataset = 'dataset_tenis'
columna_objetivo = 'winner_id'  

# Entrena los modelos y guarda los resultados
ruta_modelos = os.path.join(directorio_actual, 'model')
resultados = entrenar_modelos_SINSPLIT_y_guardar(df, nombre_dataset, columna_objetivo, ruta_modelos)
print(resultados)

guardar_csv(resultados, 'modelos.csv', ruta_modelos)
