# -*- coding: utf-8 -*-
"""
Reporte Automatizado de Aplicación (agrícola / aérea)
================================================================
App de Streamlit que:
  1. Recibe un ZIP con el polígono de aplicación (Shapefile, GeoJSON, KML, GPKG).
  2. Recibe un ZIP con los datos reales del GPS (bitácora binaria AgNav, Shapefile,
     GPX, GeoJSON o CSV con coordenadas).
  3. Calcula hectáreas teóricas, aplicadas, traslape, fuera del polígono, zonas
     omitidas (gaps), cobertura y eficiencia.
  4. Genera un mapa estático (verde = aplicada, amarillo = traslape, rojo = fuera,
     gris rayado = omitida, azul = giros) y un PDF de UNA sola página.

Todo se procesa en memoria: los ZIP nunca se guardan de forma permanente.
"""
import datetime
import io
import os
import re
import tempfile
import traceback
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024   # protección contra "zip bombs"
AGNAV_RECORD_SIZE = 68                       # bytes por registro en n*.t##
GAP_BREAK_M = 40.0                           # salto máximo dentro de una pasada
MAX_TURN_LENGTH_M = 300.0                    # un giro más largo se considera traslado
SHARP_TURN_DEG = 45.0                        # cambio de rumbo que corta una pasada
POLY_EXT = (".shp", ".geojson", ".json", ".gpkg", ".kml", ".gpx")

COLOR_GREEN = "#2eb340"
COLOR_YELLOW = "#ffd900"
COLOR_RED = "#e61a1a"
COLOR_GAP = "#bdbdbd"
COLOR_TURN = "#1f4fff"


class DataError(Exception):
    """Error de datos del usuario (ZIP dañado, formato no válido, etc.)."""


# --------------------------------------------------------------------------
# 1. Lectura de ZIP en memoria
# --------------------------------------------------------------------------
def read_zip(uploaded, label):
    """Devuelve (dict nombre->bytes, bytes_originales_del_zip). Todo en memoria."""
    raw = uploaded.getvalue()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise DataError(f"{label}: el archivo no es un ZIP válido o está corrupto.")
    files = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            if sum(i.file_size for i in zf.infolist()) > MAX_UNCOMPRESSED_BYTES:
                raise DataError(f"{label}: el contenido descomprimido es demasiado grande.")
            bad = zf.testzip()
            if bad:
                raise DataError(f"{label}: el ZIP está dañado (archivo '{bad}').")
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if info.is_dir() or "__MACOSX" in info.filename or base.startswith("."):
                    continue
                files[info.filename] = zf.read(info)
    except DataError:
        raise
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise DataError(f"{label}: no se pudo abrir el ZIP ({exc}). "
                        "Verifique que no tenga contraseña ni esté dañado.")
    if not files:
        raise DataError(f"{label}: el ZIP está vacío.")
    return files, raw


def read_text_metadata(*file_dicts):
    """Busca ancho de faja y área aplicada reportada por AgNav en .agn/.rem/.log."""
    swath, agnav_m2 = None, None
    for files in file_dicts:
        for name, data in files.items():
            if not name.lower().endswith((".agn", ".rem", ".log", ".txt")):
                continue
            txt = data.decode("latin1", errors="ignore")
            m = re.search(r"^\s*36\s+([\d.]+)\s+.*SWATH WIDTH", txt, re.M)
            if m and swath is None:
                swath = float(m.group(1))
            m = re.search(r"^\s*10\s+([\d.]+)\s+.*SPRAYED AREA IN m2", txt, re.M | re.I)
            if m and agnav_m2 is None:
                agnav_m2 = float(m.group(1))
    return swath, agnav_m2


# --------------------------------------------------------------------------
# 2. Lectura GIS (geopandas)
# --------------------------------------------------------------------------
def _read_candidate(files, raw, name):
    """Lee un archivo vectorial del ZIP. Primero en memoria (fiona), luego temporal."""
    lower = name.lower()
    # (a) en memoria con fiona.ZipMemoryFile (shp, geojson, gpkg)
    if lower.endswith((".shp", ".geojson", ".json", ".gpkg")):
        try:
            from fiona.io import ZipMemoryFile
            with ZipMemoryFile(raw) as z:
                with z.open(name) as src:
                    crs = src.crs_wkt or None
                    gdf = gpd.GeoDataFrame.from_features(list(src), crs=crs)
            if len(gdf):
                return gdf
        except Exception:
            pass
    # (b) respaldo: carpeta temporal que se borra automáticamente
    with tempfile.TemporaryDirectory() as tmp:
        for fname, data in files.items():
            dest = os.path.join(tmp, fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
        path = os.path.join(tmp, name)
        if lower.endswith(".gpx"):
            for layer in ("track_points", "tracks", "routes", "waypoints"):
                try:
                    g = gpd.read_file(path, layer=layer)
                    if len(g):
                        return g
                except Exception:
                    continue
            raise DataError("El GPX no contiene pistas, rutas ni puntos.")
        return gpd.read_file(path)


def read_vector_zip(files, raw, label):
    cands = [n for n in files if n.lower().endswith(POLY_EXT)]
    cands.sort(key=lambda n: (not n.lower().endswith(".shp"), n))
    if not cands:
        raise DataError(f"{label}: no se encontró ningún .shp/.geojson/.kml/.gpkg/.gpx dentro del ZIP.")
    errors = []
    for name in cands:
        try:
            gdf = _read_candidate(files, raw, name)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            if len(gdf):
                return gdf
        except DataError:
            raise
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise DataError(f"{label}: no se pudo leer ningún archivo espacial. " + " | ".join(errors))


def read_csv_points(files, label):
    names = [n for n in files if n.lower().endswith((".csv", ".txt"))]
    for name in names:
        try:
            df = pd.read_csv(io.BytesIO(files[name]), sep=None, engine="python")
        except Exception:
            continue
        low = {c.lower().strip(): c for c in df.columns}
        xk = next((low[k] for k in ("lon", "lng", "long", "longitude", "longitud", "x", "este", "easting") if k in low), None)
        yk = next((low[k] for k in ("lat", "latitude", "latitud", "y", "norte", "northing") if k in low), None)
        if xk and yk:
            df = df[[xk, yk]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) >= 2:
                return np.c_[df[xk].values, df[yk].values]
    raise DataError(f"{label}: no se encontraron columnas de coordenadas (lat/lon o x/y) en el CSV.")


# --------------------------------------------------------------------------
# 3. Datos GPS: bitácora AgNav (binaria) y formatos genéricos
# --------------------------------------------------------------------------
def parse_agnav(files):
    """
    Decodifica los archivos de navegación AgNav (n*.t##, registros de 68 bytes).
      bytes 3-6   : tiempo            bytes 28-31 / 32-35 : X, Y proyectados (dm)
      bytes 7-10  : latitud  (1e-6°)  byte 39             : estado de aspersión
      bytes 11-14 : longitud (1e-6°)
    Devuelve lista de dicts {name, xy, ll, spray}.
    """
    out = []
    for name, data in sorted(files.items()):
        base = os.path.basename(name).lower()
        if not re.fullmatch(r"n[^/\\]*\.t\d+", base):
            continue
        n = len(data) // AGNAV_RECORD_SIZE
        if n < 2:
            continue
        arr = np.frombuffer(data[: n * AGNAV_RECORD_SIZE], dtype=np.uint8).reshape(n, AGNAV_RECORD_SIZE)
        u32 = lambda c: np.ascontiguousarray(arr[:, c:c + 4]).view("<u4").ravel().astype(float)
        i32 = lambda c: np.ascontiguousarray(arr[:, c:c + 4]).view("<i4").ravel().astype(float)
        lat, lon = u32(7) / 1e6, i32(11) / 1e6
        ok = (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        if ok.mean() < 0.9:
            continue
        out.append(dict(name=os.path.basename(name),
                        xy=np.c_[u32(28) / 10.0, u32(32) / 10.0],
                        ll=np.c_[lon, lat],
                        spray=arr[:, 39] > 0))
    return out


def _split_sharp(p, deg=SHARP_TURN_DEG):
    """Divide una polilínea donde el rumbo cambia bruscamente."""
    if len(p) < 3:
        return [p]
    d = np.diff(p, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.abs((h[1:] - h[:-1] + np.pi) % (2 * np.pi) - np.pi)
    cuts = [0] + list(np.where(dh > np.radians(deg))[0] + 1) + [len(p) - 1]
    return [p[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def runs_and_turns(tracks, key):
    """Pasadas de aspersión (spray activo) y giros (tramos sin aspersión entre pasadas)."""
    runs, turns = [], []
    for t in tracks:
        xy, sp = t[key], t["spray"]
        n, i, frames = len(xy), 0, []
        while i < n:
            if sp[i]:
                j = i
                while j + 1 < n and sp[j + 1] and np.hypot(*(xy[j + 1] - xy[j])) < GAP_BREAK_M:
                    j += 1
                if j > i:
                    frames.append((i, j))
                i = j + 1
            else:
                i += 1
        for a, b in frames:
            runs.extend(_split_sharp(xy[a:b + 1]))
        for (a0, b0), (a1, b1) in zip(frames[:-1], frames[1:]):
            seg = xy[b0:a1 + 1]
            if len(seg) > 1 and np.hypot(*np.diff(seg, axis=0).T).sum() < MAX_TURN_LENGTH_M:
                turns.append(seg)
    return runs, turns


def tracks_from_generic(gdf_or_xy, crs, label):
    """Convierte puntos/líneas/polígonos genéricos en pasadas (arrays XY en el CRS métrico)."""
    runs, direct = [], []
    if isinstance(gdf_or_xy, np.ndarray):          # CSV
        xy = gdf_or_xy
        if np.abs(xy[:, 0]).max() <= 180 and np.abs(xy[:, 1]).max() <= 90:
            tx, ty = Transformer.from_crs(4326, crs, always_xy=True).transform(xy[:, 0], xy[:, 1])
            xy = np.column_stack([tx, ty])
        gaps = np.where(np.hypot(*np.diff(xy, axis=0).T) > GAP_BREAK_M)[0] + 1
        for seg in np.split(xy, gaps):
            if len(seg) > 1:
                runs.extend(_split_sharp(seg))
        return runs, direct
    gdf = gdf_or_xy
    if gdf.crs is None:
        raise DataError(f"{label}: el archivo GPS no tiene sistema de coordenadas (.prj).")
    gdf = gdf.to_crs(crs)
    pts = []
    for geom in gdf.geometry:
        for g in getattr(geom, "geoms", [geom]):
            if g.geom_type == "LineString":
                runs.extend(_split_sharp(np.asarray(g.coords)[:, :2]))
            elif g.geom_type == "Point":
                pts.append((g.x, g.y))
            elif g.geom_type == "Polygon":
                direct.append(g)
    if pts:
        xy = np.asarray(pts)
        gaps = np.where(np.hypot(*np.diff(xy, axis=0).T) > GAP_BREAK_M)[0] + 1
        for seg in np.split(xy, gaps):
            if len(seg) > 1:
                runs.extend(_split_sharp(seg))
    return runs, direct


# --------------------------------------------------------------------------
# 4. Geometría de las fajas
# --------------------------------------------------------------------------
def run_quads(p, half):
    """Cuadriláteros de faja a lo largo de una polilínea (normales promediadas)."""
    d = np.diff(p, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    hv = np.empty(len(p))
    hv[0], hv[-1] = h[0], h[-1]
    for k in range(1, len(p) - 1):
        hv[k] = np.arctan2(np.sin(h[k - 1]) + np.sin(h[k]), np.cos(h[k - 1]) + np.cos(h[k]))
    nrm = np.c_[-np.sin(hv), np.cos(hv)] * half
    left, right = p + nrm, p - nrm
    return [np.array([left[i], left[i + 1], right[i + 1], right[i]]) for i in range(len(p) - 1)]


def swath_geometry(p, width):
    quads = []
    for q in run_quads(p, width / 2.0):
        poly = Polygon(q)
        if poly.area > 1e-6:
            quads.append(poly if poly.is_valid else make_valid(poly))
    return unary_union(quads) if quads else None


def polygonal(geom):
    """Se queda solo con las partes poligonales de una geometría."""
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else Polygon()


def overlap_geometry(geoms):
    """Zona cubierta por 2 o más pasadas distintas."""
    tree = STRtree(geoms)
    pieces = []
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            if j > i:
                inter = polygonal(g.intersection(geoms[j]))
                if inter.area > 0.01:
                    pieces.append(inter)
    return unary_union(pieces) if pieces else Polygon()


# --------------------------------------------------------------------------
# 5. Procesamiento principal
# --------------------------------------------------------------------------
def to_metric_crs(gdf, fallback_epsg):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=fallback_epsg)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def pick_agnav_coords(tracks, poly_u, crs, mode):
    """Elige XY del proyecto AgNav o Lat/Lon convertido, según cuál cae sobre el polígono."""
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    for t in tracks:
        t["xy_proj"] = t["xy"]
        tx, ty = tr.transform(t["ll"][:, 0], t["ll"][:, 1])
        t["xy_ll"] = np.column_stack([tx, ty])
    if mode == "XY de AgNav":
        return "xy_proj", "XY del proyecto AgNav"
    if mode == "Lat/Lon convertido":
        return "xy_ll", "Lat/Lon convertido"
    zone = poly_u.buffer(150)
    score = {}
    for key in ("xy_proj", "xy_ll"):
        hits = tot = 0
        for t in tracks:
            pts = t[key][t["spray"]]
            pts = pts[:: max(1, len(pts) // 300)]
            tot += len(pts)
            hits += sum(zone.contains(Point(x, y)) for x, y in pts)
        score[key] = hits / tot if tot else 0
    key = max(score, key=score.get)
    if score[key] < 0.2:
        raise DataError("Los datos del GPS no caen sobre el polígono. Revise que ambos ZIP "
                        "correspondan al mismo lote o cambie el modo de coordenadas.")
    return key, "XY del proyecto AgNav" if key == "xy_proj" else "Lat/Lon convertido"


def process(poly_up, gps_up, params):
    # ---- lectura de ZIP ----
    pfiles, praw = read_zip(poly_up, "Polígono")
    gfiles, graw = read_zip(gps_up, "Datos GPS")
    swath_log, agnav_m2 = read_text_metadata(pfiles, gfiles)
    width = swath_log if (params["auto_meta"] and swath_log) else params["swath"]

    # ---- polígono ----
    gdf = read_vector_zip(pfiles, praw, "Polígono")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise DataError("Polígono: el archivo no contiene geometrías de tipo polígono.")
    gdf = to_metric_crs(gdf, params["epsg"])
    crs = gdf.crs
    gdf["geometry"] = gdf.geometry.apply(lambda g: g if g.is_valid else make_valid(g))
    poly_u = polygonal(unary_union(list(gdf.geometry)))
    area_poly = poly_u.area / 1e4

    # ---- GPS ----
    source, coord_label = "", ""
    tracks = parse_agnav(gfiles)
    direct = []
    if tracks:
        key, coord_label = pick_agnav_coords(tracks, poly_u, crs, params["coord_mode"])
        runs, turns = runs_and_turns(tracks, key)
        source = f"Bitácora AgNav ({len(tracks)} archivos)"
    else:
        try:
            gps = read_vector_zip(gfiles, graw, "Datos GPS")
            runs, direct = tracks_from_generic(gps, crs, "Datos GPS")
            source = "Shapefile/GPX/GeoJSON"
        except DataError:
            xy = read_csv_points(gfiles, "Datos GPS")
            runs, direct = tracks_from_generic(xy, crs, "Datos GPS")
            source = "CSV de coordenadas"
        turns = []
        coord_label = f"Coordenadas en {crs.name}"
    if not runs and not direct:
        raise DataError("No se encontraron pasadas con aspersión activa en los datos del GPS.")

    # ---- fajas ----
    geoms = [g for g in (swath_geometry(r, width) for r in runs) if g is not None and not g.is_empty]
    geoms += [g for g in direct if not g.is_empty]
    applied = polygonal(unary_union(geoms))
    overlap = overlap_geometry(geoms) if len(geoms) > 1 else Polygon()
    inside = applied.intersection(poly_u)
    ov_in = overlap.intersection(poly_u)
    green = polygonal(inside.difference(ov_in))
    yellow = polygonal(ov_in)
    red = polygonal(applied.difference(poly_u))
    gaps = polygonal(poly_u.difference(applied))

    ha = lambda g: g.area / 1e4
    a_in, a_ov, a_out, a_gap = ha(inside), ha(yellow), ha(red), ha(gaps)
    a_total = a_in + a_out
    gross_ha = sum(np.hypot(*np.diff(r, axis=0).T).sum() for r in runs) * width / 1e4
    m = dict(
        area_poly=area_poly, area_in=a_in, area_green=ha(green), area_overlap=a_ov,
        area_out=a_out, area_gap=a_gap, area_total=a_total,
        cobertura=a_in / area_poly * 100 if area_poly else 0,
        eficiencia=ha(green) / a_total * 100 if a_total else 0,
        margen_error=(a_total - area_poly) / area_poly * 100 if area_poly else 0,
        gross_ha=gross_ha, agnav_ha=(agnav_m2 / 1e4 if agnav_m2 else None),
        n_runs=len(runs), n_turns=len(turns), width=width, source=source,
        coords=coord_label, crs=crs.to_string() if crs else "",
    )

    # ---- tabla por polígono ----
    rows = []
    label_col = params["label_col"]
    ov_tree_geom = overlap
    for idx, (_, row) in enumerate(gdf.iterrows()):
        g = row.geometry
        pa = g.area / 1e4
        ap = g.intersection(applied).area / 1e4
        ovp = g.intersection(ov_tree_geom).area / 1e4
        rows.append({
            "Polígono": (str(row[label_col]) if label_col and label_col in gdf.columns else str(idx + 1)),
            "Área (ha)": round(pa, 3), "Aplicada (ha)": round(ap, 3),
            "Traslape (ha)": round(ovp, 3), "Sin aplicar (ha)": round(max(pa - ap, 0), 3),
            "Cobertura (%)": round(ap / pa * 100, 1) if pa else 0.0,
        })
    table = pd.DataFrame(rows)

    png = make_map(gdf, poly_u, green, yellow, red, gaps, runs, turns, label_col)
    return dict(metrics=m, table=table, png=png)


# --------------------------------------------------------------------------
# 6. Mapa (matplotlib + geopandas.plot)
# --------------------------------------------------------------------------
def _nice_length(width_m):
    target = width_m / 5.0
    for v in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if v >= target:
            return v
    return 10000


def _plot_geom(ax, geom, **kw):
    if geom is not None and not geom.is_empty:
        gpd.GeoSeries([geom]).plot(ax=ax, **kw)


def make_map(gdf, poly_u, green, yellow, red, gaps, runs, turns, label_col):
    minx, miny, maxx, maxy = unary_union([poly_u, green, yellow, red]).bounds
    w, h = maxx - minx, maxy - miny
    mx, my = w * 0.04, h * 0.04
    minx, maxx, miny, maxy = minx - mx, maxx + mx, miny - my, maxy + my
    w, h = maxx - minx, maxy - miny
    fig_w = 10.0
    fig_h = float(np.clip(fig_w * h / w, 5.0, 12.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    _plot_geom(ax, gaps, facecolor="#e3e3e3", edgecolor=COLOR_GAP, hatch="////", linewidth=0, zorder=1)
    _plot_geom(ax, green, color=COLOR_GREEN, linewidth=0, zorder=2)
    _plot_geom(ax, yellow, color=COLOR_YELLOW, linewidth=0, zorder=3)
    _plot_geom(ax, red, color=COLOR_RED, linewidth=0, zorder=3)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=5)
    if runs:
        ax.add_collection(LineCollection(runs, colors="#0b3d0b", linewidths=0.25, alpha=0.6, zorder=4))
    if turns:
        ax.add_collection(LineCollection(turns, colors=COLOR_TURN, linewidths=0.7, zorder=6))
    if label_col and label_col in gdf.columns and len(gdf) <= 150:
        fs = 5.5 if len(gdf) > 15 else 8
        for _, row in gdf.iterrows():
            pt = row.geometry.representative_point()
            ax.text(pt.x, pt.y, str(row[label_col]), fontsize=fs, ha="center", va="center", zorder=7,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.grid(True, color="#d0d0d0", linewidth=0.4)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("Este (m)", fontsize=7)
    ax.set_ylabel("Norte (m)", fontsize=7)

    # barra de escala
    L = _nice_length(w)
    x0, y0 = minx + 0.03 * w, miny + 0.035 * h
    seg = L / 4.0
    ax.add_patch(Rectangle((x0 - 0.01 * w, y0 - 0.02 * h), L + 0.02 * w, 0.075 * h,
                           fc="white", ec="none", alpha=0.8, zorder=8))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, 0.012 * h,
                               fc="black" if i % 2 == 0 else "white", ec="black", lw=0.6, zorder=9))
    lab = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
    ax.text(x0, y0 + 0.02 * h, "0", fontsize=6, zorder=9)
    ax.text(x0 + L, y0 + 0.02 * h, lab, fontsize=6, ha="center", zorder=9)
    # flecha norte
    ax.annotate("", xy=(0.955, 0.965), xytext=(0.955, 0.895), xycoords="axes fraction", zorder=9,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.text(0.955, 0.975, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")

    handles = [
        Patch(fc=COLOR_GREEN, label="Aplicada (1 pasada)"),
        Patch(fc=COLOR_YELLOW, label="Traslape (2+ pasadas)"),
        Patch(fc=COLOR_RED, label="Fuera del polígono"),
        Patch(fc="#e3e3e3", ec=COLOR_GAP, hatch="////", label="Omitida (gap)"),
        Line2D([0], [0], color="black", lw=1.2, label="Límite del polígono"),
        Line2D([0], [0], color=COLOR_TURN, lw=1.2, label="Giros"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------
# 7. PDF de UNA página (reportlab, canvas con layout fijo)
# --------------------------------------------------------------------------
def fmt(v, d=2):
    return f"{v:,.{d}f}"


def build_pdf(res, meta, logo_bytes=None):
    m, png = res["metrics"], res["png"]
    img = ImageReader(io.BytesIO(png))
    iw, ih = img.getSize()
    pagesize = landscape(A4) if iw / ih > 1.35 else A4
    W, H = pagesize
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=pagesize)
    c.setTitle("Reporte Automatizado de Aplicación")
    M = 12 * mm
    dark, green = colors.HexColor("#1b5e20"), colors.HexColor("#2e7d32")

    # --- encabezado ---
    top = H - M
    c.setFillColor(green)
    c.rect(M, top - 18 * mm, W - 2 * M, 18 * mm, fill=1, stroke=0)
    lw_, lh_ = 30 * mm, 14 * mm
    c.setFillColor(colors.white)
    c.rect(M + 2 * mm, top - 16 * mm, lw_, lh_, fill=1, stroke=0)
    if logo_bytes:
        try:
            c.drawImage(ImageReader(io.BytesIO(logo_bytes)), M + 2 * mm, top - 16 * mm, lw_, lh_,
                        preserveAspectRatio=True, anchor="c", mask="auto")
        except Exception:
            logo_bytes = None
    if not logo_bytes:
        c.setFillColor(colors.HexColor("#9e9e9e"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(M + 2 * mm + lw_ / 2, top - 10 * mm, "LOGO")
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(M + lw_ + 8 * mm, top - 8 * mm, "Reporte Automatizado de Aplicación")
    c.setFont("Helvetica", 8.5)
    sub = " | ".join(x for x in (meta.get("lote"), meta.get("cliente")) if x) or "Lote sin nombre"
    c.drawString(M + lw_ + 8 * mm, top - 14 * mm, sub)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - M - 3 * mm, top - 8 * mm, "Fecha: " + datetime.date.today().strftime("%d/%m/%Y"))
    if meta.get("fecha_app"):
        c.drawRightString(W - M - 3 * mm, top - 14 * mm, "Aplicación: " + meta["fecha_app"])

    # --- KPI principales (3 tarjetas) ---
    y = top - 18 * mm - 4 * mm
    card_h, gap = 17 * mm, 4 * mm
    card_w = (W - 2 * M - 2 * gap) / 3
    kpis = [("Hectáreas teóricas", fmt(m["area_poly"]) + " ha"),
            ("Hectáreas aplicadas", fmt(m["area_in"]) + " ha"),
            ("% de Eficiencia", fmt(m["eficiencia"], 1) + " %")]
    for i, (lab, val) in enumerate(kpis):
        x = M + i * (card_w + gap)
        c.setFillColor(colors.HexColor("#f1f8e9"))
        c.setStrokeColor(green)
        c.roundRect(x, y - card_h, card_w, card_h, 3 * mm, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#555555"))
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(x + card_w / 2, y - 6 * mm, lab)
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(x + card_w / 2, y - 13.5 * mm, val)
    y -= card_h + 3 * mm

    # --- tabla de detalle (4 columnas) ---
    pct = lambda v: fmt(v / m["area_poly"] * 100, 1) + " %" if m["area_poly"] else "-"
    agn = fmt(m["agnav_ha"]) + " ha" if m["agnav_ha"] else "n/d"
    data = [
        ["Cobertura del polígono", fmt(m["cobertura"], 1) + " %", "Aplicada 1 pasada (verde)", fmt(m["area_green"]) + " ha"],
        ["Traslape 2+ pasadas (amarillo)", fmt(m["area_overlap"]) + " ha  (" + pct(m["area_overlap"]) + ")",
         "Fuera del polígono (rojo)", fmt(m["area_out"]) + " ha  (" + pct(m["area_out"]) + ")"],
        ["Zonas omitidas (gaps)", fmt(m["area_gap"]) + " ha  (" + pct(m["area_gap"]) + ")",
         "Área aplicada total", fmt(m["area_total"]) + " ha"],
        ["Margen de error", f"{m['margen_error']:+.2f} %", "Ancho de faja / pasadas / giros",
         f"{m['width']:g} m / {m['n_runs']} / {m['n_turns']}"],
        ["Área bruta (suma de fajas)", fmt(m["gross_ha"]) + " ha", "Área reportada por AgNav", agn],
    ]
    t = Table(data, colWidths=[(W - 2 * M) * f for f in (0.27, 0.23, 0.27, 0.23)], rowHeights=5.6 * mm)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8), ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")), ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e8f5e9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    tw, th = t.wrapOn(c, W - 2 * M, 100 * mm)
    t.drawOn(c, M, y - th)
    y -= th + 3 * mm

    # --- pie: observaciones y firmas ---
    foot_h = 34 * mm
    foot_top = M + foot_h
    c.setStrokeColor(colors.HexColor("#b0bec5"))
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(M, foot_top - 3 * mm, "Observaciones")
    c.rect(M, M + 11 * mm, W - 2 * M, foot_h - 17 * mm, fill=0, stroke=1)
    c.setFont("Helvetica", 8)
    note = (meta.get("notas") or "").strip()
    ty = foot_top - 10.5 * mm
    for ln in simpleSplit(note, "Helvetica", 8, W - 2 * M - 4 * mm)[:3]:
        c.drawString(M + 2 * mm, ty, ln)
        ty -= 3.6 * mm
    sig_w = (W - 2 * M - 2 * gap) / 3
    for i, lab in enumerate(("Elaborado por", "Revisado por", "Recibido por (cliente)")):
        x = M + i * (sig_w + gap)
        c.line(x + 4 * mm, M + 6 * mm, x + sig_w - 4 * mm, M + 6 * mm)
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(x + sig_w / 2, M + 2.5 * mm, lab + (": " + meta["elaborado"] if i == 0 and meta.get("elaborado") else ""))

    # --- mapa: ocupa todo el espacio libre, centrado y sin deformarse ---
    map_top, map_bottom = y, foot_top + 2 * mm
    bw, bh = W - 2 * M, map_top - map_bottom
    s = min(bw / iw, bh / ih)
    dw, dh = iw * s, ih * s
    c.drawImage(img, M + (bw - dw) / 2, map_bottom + (bh - dh) / 2, dw, dh)
    c.setFont("Helvetica-Oblique", 6.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawRightString(W - M, M - 3 * mm + 1, f"Fuente: {m['source']} | Coordenadas: {m['coords']} | CRS: {m['crs']}")
    c.showPage()          # una sola página
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------
# 8. Interfaz Streamlit
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Reporte de Aplicación", page_icon="🗺️", layout="wide")
    st.title("🗺️ Reporte Automatizado de Aplicación")
    st.caption("Cargue el polígono y los datos del GPS; la app calcula la cobertura y genera un PDF de una página.")

    with st.sidebar:
        st.header("Parámetros")
        auto_meta = st.checkbox("Leer ancho de faja y área AgNav del log (si existe)", value=True)
        swath = st.number_input("Ancho de faja (m)", 1.0, 200.0, 15.0, 0.5)
        coord_mode = st.selectbox("Coordenadas AgNav", ["Automático", "XY de AgNav", "Lat/Lon convertido"])
        epsg = st.number_input("EPSG si el polígono no trae .prj", 1000, 99999, 32616, 1)
        label_col = st.text_input("Columna para etiquetar polígonos (opcional)", "")
        st.header("Datos del reporte")
        lote = st.text_input("Lote / Finca")
        cliente = st.text_input("Cliente / Empresa")
        fecha_app = st.text_input("Fecha de aplicación")
        elaborado = st.text_input("Elaborado por")
        notas = st.text_area("Observaciones", height=90)
        logo = st.file_uploader("Logo (opcional)", type=["png", "jpg", "jpeg"])

    c1, c2 = st.columns(2)
    poly_up = c1.file_uploader("1) ZIP del polígono de aplicación", type=["zip"], key="poly")
    gps_up = c2.file_uploader("2) ZIP con los datos reales del GPS", type=["zip"], key="gps")

    if st.button("Generar reporte", type="primary", disabled=not (poly_up and gps_up)):
        params = dict(auto_meta=auto_meta, swath=swath, coord_mode=coord_mode, epsg=int(epsg),
                      label_col=label_col.strip() or None)
        try:
            with st.spinner("Procesando datos..."):
                res = process(poly_up, gps_up, params)
                res["meta"] = dict(lote=lote, cliente=cliente, fecha_app=fecha_app,
                                   elaborado=elaborado, notas=notas)
                res["pdf"] = build_pdf(res, res["meta"], logo.getvalue() if logo else None)
            st.session_state["res"] = res
        except DataError as exc:
            st.session_state.pop("res", None)
            st.error(f"⚠️ {exc}")
        except Exception as exc:  # error no previsto
            st.session_state.pop("res", None)
            st.error("Ocurrió un error inesperado al procesar los archivos. "
                     "Verifique que los datos sean válidos.")
            with st.expander("Detalle técnico"):
                st.code(traceback.format_exc())

    res = st.session_state.get("res")
    if not res:
        st.info("Suba ambos archivos ZIP y presione **Generar reporte**.")
        return
    m = res["metrics"]
    k = st.columns(5)
    k[0].metric("Hectáreas teóricas", f"{m['area_poly']:,.2f}")
    k[1].metric("Hectáreas aplicadas", f"{m['area_in']:,.2f}")
    k[2].metric("% Cobertura", f"{m['cobertura']:.1f} %")
    k[3].metric("% Eficiencia", f"{m['eficiencia']:.1f} %")
    k[4].metric("Margen de error", f"{m['margen_error']:+.2f} %")
    k = st.columns(4)
    k[0].metric("Traslape (ha)", f"{m['area_overlap']:,.2f}")
    k[1].metric("Fuera del polígono (ha)", f"{m['area_out']:,.2f}")
    k[2].metric("Omitida / gaps (ha)", f"{m['area_gap']:,.2f}")
    k[3].metric("Pasadas / giros", f"{m['n_runs']} / {m['n_turns']}")
    st.image(res["png"], use_container_width=True)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Descargar PDF (1 página)", res["pdf"], "reporte_aplicacion.pdf", "application/pdf")
    d2.download_button("⬇️ Descargar mapa (PNG)", res["png"], "mapa_aplicacion.png", "image/png")
    d3.download_button("⬇️ Descargar tabla (CSV)", res["table"].to_csv(index=False).encode("utf-8-sig"),
                       "detalle_poligonos.csv", "text/csv")
    with st.expander("Detalle por polígono"):
        st.dataframe(res["table"], use_container_width=True)


if __name__ == "__main__":
    main()
