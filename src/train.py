import argparse
import json
import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
import os # <-- ¡NECESARIO!

# --- NUEVOS IMPORTS PARA MLFLOW ---
import mlflow
import mlflow.sklearn
# -----------------------------------

# Forzamos la URI de tracking a DagsHub. La URL real se pasa como variable de entorno
# en tu archivo ci.yaml. Si falla, usa un backend temporal seguro.
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/SebaFama23/telco_churn.mlflow")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Telco Churn Experiment")
# -----------------------------------

# Función para cargar parámetros de un archivo YAML
def load_params(params_path):
    with open(params_path) as f:
        return yaml.safe_load(f)

# Función principal de entrenamiento
def train_and_evaluate(train_path, test_path, model_path, metrics_path, params_path):
    
    # MLflow: Inicia una nueva corrida (Run) para este entrenamiento. 
    # Usamos 'Training Run' como nombre, puedes cambiarlo.
    with mlflow.start_run(run_name="DVC-MLflow Training Run") as run:
        
        # 1. Cargar Parámetros
        params = load_params(params_path)
        model_params = params.get('model', {})
        split_params = params.get('split', {})
        
        # MLflow: Loguea los hiperparámetros del modelo y del split
        mlflow.log_params(model_params)
        mlflow.log_params(split_params)

        # 2. Cargar Datos Limpios
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        TARGET = 'churn'
        
        X_train, y_train = df_train.drop(TARGET, axis=1), df_train[TARGET]
        X_test, y_test = df_test.drop(TARGET, axis=1), df_test[TARGET]
        
        # 3. Entrenamiento (Logistic Regression)
        model = LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 200),
            penalty=model_params.get('penalty', 'l2'),      # <-- ¡NUEVA LÍNEA!
            solver=model_params.get('solver', 'liblinear'), # <-- ¡NUEVA LÍNEA!
            random_state=split_params.get('random_state', 42)
        )
        print("[INFO] Entrenando modelo...")
        model.fit(X_train, y_train)
        
        # 4. Evaluación
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        print(f"[INFO] Métricas obtenidas: {metrics}")
        
        # MLflow: Loguea las métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        # 5. Guardar Métricas (metrics.json)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"[OK] Métricas guardadas en {metrics_path}")
        
        # 6. Guardar Modelo (model.pkl)
        joblib.dump(model, model_path)
        print(f"[OK] Modelo guardado en {model_path}")
        
        # MLflow: Loguea el modelo como un artefacto
        #mlflow.sklearn.log_model(model, "model_artifact")
        #mlflow.log_artifact(metrics_path) 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    
    train_and_evaluate(args.train, args.test, args.model, args.metrics, args.params)