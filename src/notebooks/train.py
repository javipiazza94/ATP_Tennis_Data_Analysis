import pandas as pd
import sys
import os

# Obtiene la ruta absoluta del directorio que contiene este script
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Ruta al directorio src (dos niveles arriba de notebooks)
src_dir = os.path.abspath(os.path.join(directorio_actual, '..'))
# Ruta a utils
utils_dir = os.path.join(src_dir, 'utils')

sys.path.append(utils_dir)

from funciones import entrenar_modelos, guardar_csv

# Carga el conjunto de datos procesado (v2)
data_path = os.path.join(src_dir, 'data', 'processed', 'df_ready_for_model.csv')
print(f"Loading data from: {data_path}")

if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    print("Please run src/data/process_data_v2.py first.")
    sys.exit(1)

df = pd.read_csv(data_path)

print(df.head(10))

# Especifica el nombre del conjunto de datos y la columna objetivo
nombre_dataset = 'dataset_tenis'
columna_objetivo = 'target'  

# Directorio para guardar modelos
model_dir = os.path.join(src_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

# Entrena los modelos y guarda los resultados
print("Starting training...")
resultados = entrenar_modelos(df, nombre_dataset, columna_objetivo, model_dir)
print(resultados)

guardar_csv(resultados, 'modelos.csv', model_dir)

