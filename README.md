# final_project_xai
This repo contains the final project developed for the course "Ethics and Explainable AI".


## Workflow de Gestión de Dependencias
Este proyecto utiliza [Conda](https://docs.conda.io/en/latest/) para la gestión de dependencias y entornos virtuales. 

### Añadiendo paquete mediante Conda
Para añadir un nuevo paquete a tu entorno Conda, sigue estos pasos:
1. Activa tu entorno Conda:
   ```bash
   conda activate xai_project
   ```
2. Instala el paquete utilizando Conda:
   ```bash
   conda install nombre_del_paquete=versión
   ```
3. Actualiza el archivo `environment.yml` para reflejar los cambios. Manualmente, edita el archivo y añade:
- `- nombre_del_paquete=versión` bajo la sección `dependencies`.
4. Exporta las versiones exactas instaladas a un archivo `environment_locked.yml` para asegurar la reproducibilidad:
   ```bash
    conda env export --no-builds > environment_locked.yml
    ```

### Añadiendo paquete mediante pip
Si el paquete que deseas instalar no está disponible en los canales de Conda, puedes usar pip:
1. Activa tu entorno Conda:
    ```bash
    conda activate xai_project
    ```
2. Instala el paquete utilizando pip:
    ```bash
    pip install nombre_del_paquete==versión
    ```
3. Actualiza el archivo `environment.yml` para reflejar los cambios. Manualmente, edita el archivo y añade:
   - `- pip:` (si no existe ya)
     - `  - nombre_del_paquete==versión`
4. Exporta las versiones exactas instaladas a un archivo `requirements_frozen.txt`:
    ```bash
    pip freeze > requirements_frozen.txt
    ```
5. Actualizar el archivo `environment_locked.yml` para asegurar la reproducibilidad:
   ```bash
    conda env export --no-builds > environment_locked.yml
    ```