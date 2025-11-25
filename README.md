Paso	    Comando/Acción	    Herramienta	        Objetivo
0. Ambiente	        conda activate dagshub-mlops-env	        Conda	        Asegurar que todas las librerías (dvc, mlflow, etc.) sean accesibles.
1. Navegación	        cd "C:\Users\Morgan\mi_nuevo_proyecto_mlops"	        Terminal	        Ubicarte en la carpeta raíz del proyecto.
2. Edición	        Abrir VS Code y seleccionar el kernel mineria_datos.	    VS Code	        Trabajar en el código o notebooks.
3. DVC Check	        dvc status	        DVC	(Solo si ya tienes datos rastreados)        Verifica qué archivos de datos han cambiado.
4. CÓDIGO	    Editas/Agregas      archivos de código (.py, .ipynb, etc.).	        Git/VS Code	        *
5. DATOS	        Agregas/Modificas archivos de datos (e.g., data/raw.csv).	        DVC	        *
6. RASTREO DVC	        dvc add data/nuevo_dataset.csv	        DVC	        Rastrea los nuevos archivos de datos (crea/actualiza data/.dvc).
7. SUBIDA DVC	        dvc push	        DVC	        Sube los datos reales (los archivos grandes) al almacenamiento remoto de DAGsHub.
8. COMMIT GIT	        git add . git commit -m "Descripción de los cambios de código y datos"	    Git     	Rastrea los cambios de código y el puntero DVC (.dvc)
9. SUBIDA GIT	        git push	        Git	        Sube el código y los punteros a DAGsHub (la parte "visible").

MLFOW conections:
set MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
set MLFLOW_TRACKING_USERNAME="user"
set MLFLOW_TRACKING_PASSWORD="tokken"

Github Actions: 
# 1. Añadir el archivo modificado
git add README.md

# 2. Confirmar los cambios con un mensaje claro
git commit -m "Test: Validando conexión DagsHub/CI despues de arreglo"

# 3. Enviar el commit a GitHub (esto dispara la CI/CD)
git push origin main

