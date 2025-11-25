import yaml
import mlflow 
import os 
import argparse
import pandas as pd
import json # Necesario para guardar métricas en el archivo
from sklearn.linear_model import LogisticRegression # Usaremos LogisticRegression (más simple)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib

# --- CONFIGURACIÓN MLFLOW ---
# Las credenciales de DAGsHub ya están en variables de entorno (MLFLOW_TRACKING_USERNAME, etc.).
mlflow.set_experiment("Telco Churn Experiment")

# --- FUNCIONES AUXILIARES ---

def load_params(params_path):
    """Carga los parámetros del modelo y split desde params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def eval_model(clf, X, y):
    """Evalúa el modelo y devuelve las métricas."""
    # Nota: Usamos roc_auc_score solo si el modelo tiene predict_proba
    try:
        y_proba = clf.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_proba)
    except AttributeError:
        roc_auc = None
        
    y_pred = clf.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }
    if roc_auc is not None:
        metrics['roc_auc'] = roc_auc
        
    return metrics

def train_and_evaluate(train_path, test_path, model_path, metrics_path, params_path):
    
    # 1. CARGAR PARÁMETROS y DATOS
    params = load_params(params_path)
    model_params = params.get('model', {})
    split_params = params.get('split', {})
    
    # Asumimos que data_prep.py generó los archivos train.csv y test.csv
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    TARGET = 'churn' # La columna objetivo de Telco Churn

    # Seleccionar solo FEATURES NUMÉRICAS (para evitar errores con One-Hot Encoder si no se aplica)
    # **ESTE ES UN PUNTO CRÍTICO. Si tienes features categóricas, necesitarás el ColumnTransformer
    # del script de tu proyecto anterior, pero por ahora simplificamos a numéricas**
    numerical_cols = df_train.select_dtypes(include=['number']).columns.drop(TARGET)
    
    X_train, y_train = df_train[numerical_cols], df_train[TARGET]
    X_test, y_test = df_test[numerical_cols], df_test[TARGET]

    # 2. ENTRENAMIENTO (Logistic Regression)
    model = LogisticRegression(
        C=model_params.get('C', 1.0),
        max_iter=model_params.get('max_iter', 200),
        random_state=split_params.get('random_state', 42),
        solver='liblinear' # Usamos un solver simple para evitar warnings
    )
    print("Entrenando modelo...")
    model.fit(X_train, y_train)

    # 3. EVALUACIÓN y LOGGING
    metrics = eval_model(model, X_test, y_test)
    print(f"Métricas finales: {metrics}")

    # Log con MLflow
    with mlflow.start_run(run_name="LogisticRegression_Telco"):
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
    # 4. GUARDAR ARTEFACTOS DVC
    joblib.dump(model, model_path)
    print(f"✅ Modelo guardado en {model_path}")

    # Guardar métricas en archivo (para DVC)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Métricas guardadas en {metrics_path}")


if __name__ == '__main__':
    # Bloque principal que DVC llama
    parser = argparse.ArgumentParser(description="Etapa de entrenamiento para DVC.")
    parser.add_argument("--data", required=True) 
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    # Rutas de los datasets de train/test (se obtienen del params.yaml)
    params = load_params(args.params)
    train_path = params['paths']['train_data']
    test_path = params['paths']['test_data']
    
    # Llamar a la función principal
    train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        model_path=args.model,
        metrics_path=args.metrics,
        params_path=args.params # Pasar params_path para logging interno
    )