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
set MLFLOW_TRACKING_URI="https://dagshub.com/SebaFama23/mi_nuevo_proyecto_mlops.mlflow"
set MLFLOW_TRACKING_USERNAME="SebaFama23"
set MLFLOW_TRACKING_PASSWORD="6f5f090e18b20252f15a3fb0bce293ce0981e00b"
