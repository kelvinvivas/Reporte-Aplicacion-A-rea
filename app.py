import os
import zipfile
import tempfile
import struct
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import streamlit as st

st.set_page_config(page_title="Procesador AgNav - Los Jícaros", layout="wide")

st.title("🌱 Procesador GIS de Aplicación Agrícola — AgNav & Shapefile")
st.markdown("Carga tus archivos `.zip` para extraer las coordenadas GPS, calcular áreas de cobertura, traslapes y generar el reporte analítico.")

col1, col2 = st.columns(2)

with col1:
    file_shp_zip = st.file_uploader("1. Shapefile del Lote (`jicaro.zip`)", type=["zip"])

with col2:
    file_gps_zip = st.file_uploader("2. Telemetría GPS AgNav (`JicaroS1B1.zip`)", type=["zip"])

def parse_agnav_n_file(file_path):
    """Decodifica registros binarios de 68 bytes de archivos .t4x / .t5x de AgNav."""
    coords = []
    spray_states = []
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Buscar cabeceras de registro \xfb\xfb (0xFB 0xFB)
    idx = 0
    while idx < len(data) - 68:
        if data[idx:idx+2] == b'\xfb\xfb':
            rec = data[idx:idx+68]
            # Extraer enteros de coordenadas UTM (Offset 28: Este, Offset 32: Norte en metros/cm)
            easting = struct.unpack('<i', rec[28:32])[0] / 10.0
            northing = struct.unpack('<i', rec[32:36])[0] / 10.0
            spray_status = rec[20] > 0  # Estado del solenoide / activación de aspersión
            
            if 100000 < easting < 900000 and 1000000 < northing < 2000000:
                coords.append((easting, northing))
                spray_states.append(spray_status)
            idx += 68
        else:
            idx += 1
    return coords, spray_states

if file_shp_zip and file_gps_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Extraer Shapefile
        shp_dir = os.path.join(tmpdir, "shp")
        os.makedirs(shp_dir, exist_ok=True)
        with zipfile.ZipFile(file_shp_zip, 'r') as z:
            z.extractall(shp_dir)
            
        # 2. Extraer archivos AgNav
        gps_dir = os.path.join(tmpdir, "gps")
        os.makedirs(gps_dir, exist_ok=True)
        with zipfile.ZipFile(file_gps_zip, 'r') as z:
            z.extractall(gps_dir)
            
        # Cargar Shapefile
        shp_path = None
        for root, _, files in os.walk(shp_dir):
            for f in files:
                if f.endswith(".shp"):
                    shp_path = os.path.join(root, f)
                    break
                    
        if shp_path:
            gdf_lote = gpd.read_file(shp_path)
            if gdf_lote.crs != "EPSG:32616":
                gdf_lote = gdf_lote.to_crs("EPSG:32616")
                
            poligono_lote = unary_union(gdf_lote.geometry)
            area_lote_ha = poligono_lote.area / 10000.0
            
            # Buscar y decodificar todos los archivos AgNav .tXX
            swaths = []
            turn_lines = []
            ancho_faja = 15.0  # Ancho de faja configurable (metros)
            
            for root, _, files in os.walk(gps_dir):
                for f in files:
                    if '.' in f and f.split('.')[-1].startswith('t') and not f.startswith('q'):
                        full_path = os.path.join(root, f)
                        coords, states = parse_agnav_n_file(full_path)
                        
                        if len(coords) > 1:
                            line = LineString(coords)
                            turn_lines.append(line)
                            
                            # Si hay activación de aspersión, se genera el buffer de faja (15 m)
                            swath = line.buffer(ancho_faja / 2.0, cap_style=2)
                            swaths.append(swath)

            if swaths:
                union_aplicada = unary_union(swaths)
                
                # Geometría de intersecciones
                area_dentro = union_aplicada.intersection(poligono_lote)
                area_fuera = union_aplicada.difference(poligono_lote)
                
                ha_lote = area_lote_ha
                ha_dentro = area_dentro.area / 10000.0
                ha_fuera = area_fuera.area / 10000.0
                ha_total_aplicada = union_aplicada.area / 10000.0
                
                # Visualización
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Dibujar polígono base
                gdf_lote.plot(ax=ax, facecolor='#f0f0f0', edgecolor='black', linewidth=1.5)
                
                # Dibujar capa fuera del polígono (Rojo)
                if not area_fuera.is_empty:
                    gpd.GeoSeries([area_fuera]).plot(ax=ax, color='red', alpha=0.7, label='Área fuera del polígono')
                    
                # Dibujar capa aplicada dentro (Verde)
                if not area_dentro.is_empty:
                    gpd.GeoSeries([area_dentro]).plot(ax=ax, color='#2ca02c', alpha=0.8, label='Área aplicada dentro')
                    
                # Dibujar trazas de viraje (Azul)
                if turn_lines:
                    gpd.GeoSeries(turn_lines).plot(ax=ax, color='blue', linewidth=0.7, alpha=0.6, label='Trayectoria / Giros')
                    
                # Numerar bloques
                for idx, row in gdf_lote.iterrows():
                    c = row.geometry.centroid
                    ax.text(c.x, c.y, str(idx+1), fontsize=10, fontweight='bold', ha='center',
                            bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='black', lw=0.8))

                ax.set_title("Los Jícaros (GTE.-04) - Mapa de Aplicación Aérea", fontsize=12, fontweight='bold')
                ax.set_xlabel("Este (m) - WGS84 UTM 16N")
                ax.set_ylabel("Norte (m)")
                ax.grid(True, linestyle=':', alpha=0.6)
                
                col_left, col_right = st.columns([1.2, 1.0])
                
                with col_left:
                    st.pyplot(fig)
                    
                with col_right:
                    st.subheader("📊 RESULTADOS CALCULADOS DEL ARCHIVO")
                    
                    st.write(f"**Área real del polígono:** {ha_lote:.3f} ha (100.00%)")
                    st.write(f"**Área aplicada dentro:** {ha_dentro:.3f} ha ({(ha_dentro/ha_lote)*100:.2f}%)")
                    st.write(f"**Área aplicada fuera:** {ha_fuera:.3f} ha ({(ha_fuera/ha_lote)*100:.2f}%)")
                    st.write(f"**Área aplicada total real:** {ha_total_aplicada:.3f} ha")
                    
                    margen_error = ((ha_total_aplicada - ha_lote) / ha_lote) * 100
                    st.metric("Margen de Error", f"{margen_error:.2f} %")
            else:
                st.warning("Se extrajeron los archivos, pero no se encontraron trazas de coordenadas válidas en la carpeta de AgNav.")
