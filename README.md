# Reporte Automatizado de Aplicación

App de Streamlit que cruza el polígono de aplicación con los datos reales del GPS
(bitácora AgNav, Shapefile, GPX, GeoJSON o CSV) y genera un PDF de una página con
hectáreas teóricas/aplicadas, traslape, fuera del polígono, zonas omitidas, eficiencia y mapa.

## Despliegue en Streamlit Community Cloud
1. Suba `app.py`, `requirements.txt` y este `README.md` a un repositorio de GitHub.
2. En https://share.streamlit.io elija el repositorio y `app.py` como archivo principal.
3. En *Advanced settings* seleccione Python 3.11 o 3.12.

No se agrega `gdal` a `requirements.txt`: `fiona` y `pyogrio` ya incluyen GDAL en sus wheels
para Linux. Instalar `gdal` por pip exigiría `libgdal-dev` en `packages.txt` y suele fallar.
