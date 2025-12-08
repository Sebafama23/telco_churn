# 1- Clonar y Configurar
1. Clonar el repositorio
git clone https://github.com/Sebafama23/telco_churn.git
cd telco_churn

2. Crear y activar el entorno virtual
python -m venv venv
source venv/Scripts/activate

3. Instalar dependencias (incluyendo DVC y MLflow)
pip install -r requirements.txt
pip install "dvc[all]" mlflow dagshub

4. Configurar credenciales de DagsHub localmente (requiere tu Token PAT)
#Esto solo se necesita una vez por terminal para el push/pull

dvc remote modify dagshub-storage --local auth basic
dvc remote modify dagshub-storage --local user TU_USUARIO_DAGSHUB
dvc remote modify dagshub-storage --local password TU_TOKEN_DAGSHUB


# 2- Para Realizar un Nuevo Experimento (Iteración Colaborativa). 
1. Sincronizar y Crear la Rama
#Asegúrate de estar en main y crea una nueva rama descriptiva (feat-*).

Bash
    # 1. Ir a la rama principal y sincronizar
    git checkout main
    git pull origin main

    # 2. Crear y cambiar a la nueva rama de feature (Ej: tunear un hiperparámetro)
    git switch -c feat-nuevo-modelo

2. Implementar el Experimento - Modifica el archivo params.yaml
model:
  C: 0.1 # <--- Nuevo valor de prueba
  penalty: 'l2'
  ...

3. Traer Datos, Reproducir y Registrar
Bash
    # 1. Traer datos grandes faltantes (si los hay)
    dvc pull

    # 2. Ejecutar la pipeline (generar nuevo modelo/métricas)
    dvc repro

4. Confirmar Cambios Locales
Bash
    # Añade el archivo de parámetros modificado y los metadatos generados por DVC
    git add params.yaml dvc.lock metrics.json

    # Haz commit con un mensaje descriptivo
    git commit -m "feat: Experimento de tunning con C=0.1"

5. Subir a la Nube y Crear el Pull Request (PR)
Bash
    # 1. Subir los archivos grandes a DagsHub (caché DVC)
    dvc push

    # 2. Subir el código y los metadatos a GitHub (dispara el CI)
    git push origin feat-nuevo-modelo

6. Validar y Fusionar (Merge)
Crea el Pull Request (PR) en GitHub (el enlace aparecerá en la terminal).

Validación Automática: Espera a que el CI/CD Pipeline en GitHub Actions se ejecute y muestre el ✅ verde.

Comparación: Ve a la sección Experiments en DagsHub para comparar las métricas de esta rama con las de main.

Merge: Si las métricas son superiores, fusiona el PR en la rama main. El workflow de CD promoverá automáticamente el nuevo modelo a Production en el MLflow Model Registry.