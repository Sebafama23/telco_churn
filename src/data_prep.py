import pandas as pd
import numpy as np
import argparse
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline 

def initial_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza los pasos iniciales de limpieza de datos."""
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)

    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(0.0)

    # Convertir 'churn' a tipo entero.
    if df['churn'].dtype != 'int64':
        df['churn'] = df['churn'].astype(int)

    return df

def load_params(params_path):
    with open(params_path) as f:
        return yaml.safe_load(f)

# --- FUNCIÓN DE PREPROCESAMIENTO ---
def apply_preprocessing(df: pd.DataFrame):
    """Identifica y aplica One-Hot Encoding y Escalado a las columnas."""
    
    # Identificar Target (si está en el DataFrame, lo excluimos temporalmente)
    TARGET = 'churn'
    
    # Separar features del target (si existe)
    if TARGET in df.columns:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
    else:
        X = df
        y = None

    # Identificar columnas:
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    # Preprocesadores
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        # handle_unknown='ignore' es clave si el split crea nuevas categorías en el test set
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    # ColumnTransformer aplica la transformación correcta a cada columna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='passthrough' # Mantiene columnas no especificadas (aunque no debería haber)
    )
    
    # Ajustar y transformar (esto elimina todas las strings y las convierte a números)
    X_processed = preprocessor.fit_transform(X)
    
    # Convertir a DataFrame, conservando el Target si existe
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    if y is not None:
        df_processed[TARGET] = y
        
    return df_processed
# ------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Ruta del dataset crudo.")
    parser.add_argument("--params", required=True, help="Ruta al archivo params.yaml.")
    args = parser.parse_args()

    try:
        # 1. Cargar datos y aplicar limpieza inicial
        df_raw = pd.read_csv(args.raw)
        df_cleaned = initial_data_cleaning(df_raw.copy())
        
        # 2. LLAMAR LA FUNCIÓN DE PREPROCESAMIENTO Y CODIFICACIÓN
        df_processed = apply_preprocessing(df_cleaned)
        print(f"[INFO] Dataset codificado: {df_processed.shape[1]} columnas numéricas.")

        # 3. Cargar parámetros de SPLIT y RUTAS DE SALIDA
        full_params = load_params(args.params)
        split_params = full_params.get('split')
        paths = full_params.get('paths')

        if split_params is None or paths is None:
             raise KeyError("El archivo params.yaml debe tener las secciones 'split' y 'paths'.")
        
        TARGET = 'churn'
        
        # 4. División de datos
        df_train, df_test = train_test_split(
            df_processed, # <--- ¡USAMOS EL DF 100% NUMÉRICO!
            test_size=split_params['test_size'],
            random_state=split_params['random_state'],
            stratify=df_processed[TARGET] 
        )
        
        # 5. Guardar los archivos de salida
        train_output_path = paths['train_data']
        test_output_path = paths['test_data']
        
        Path(train_output_path).parent.mkdir(parents=True, exist_ok=True) 

        df_train.to_csv(train_output_path, index=False)
        df_test.to_csv(test_output_path, index=False)

        print(f"✅ Datasets de entrenamiento y prueba creados.")

    except Exception as e:
        print(f"ERROR fatal en la etapa 'prepare'. Detalle: {e}")
        exit(1)