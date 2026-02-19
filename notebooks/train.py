import pandas as pd
import sys
sys.path.append('C:/Users/puent/OneDrive/Escritorio/Javi Bootcamp/Data-Science-TBB-main/SCRIPTS/Proyectos/Machine Learning/src/utils')
from funciones import entrenar_modelos_SINSPLIT_y_guardar, guardar_csv
import os

# Obtiene la ruta absoluta del directorio que contiene este script
directorio_actual = os.path.abspath(os.getcwd())


# Carga el conjunto de datos
df = pd.read_csv(directorio_actual + '\\Data-Science-TBB-main\\SCRIPTS\\Proyectos\\Machine Learning\\src\\data\\processed\\dfLimpio.csv')

print(df.head(10))

# Especifica el nombre del conjunto de datos y la columna objetivo
nombre_dataset = 'dataset_tenis'
columna_objetivo = 'winner_id'  

# Entrena los modelos y guarda los resultados
resultados = entrenar_modelos_SINSPLIT_y_guardar(df, nombre_dataset, columna_objetivo, directorio_actual + '\\Data-Science-TBB-main\\SCRIPTS\\Proyectos\\Machine Learning\\src\\model')
print(resultados)

guardar_csv(resultados, 'modelos.csv', directorio_actual + '\\Data-Science-TBB-main\\SCRIPTS\\Proyectos\\Machine Learning\\src\\model')
