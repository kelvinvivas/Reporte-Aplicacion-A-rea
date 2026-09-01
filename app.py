import streamlit as st
import io, re, struct, tempfile, zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

st.set_page_config(page_title="Aplicación aérea AgNav", page_icon="🚁", layout="wide")

VERDE = "#4CAF18"
AMARILLO = "#FFD700"
ROJO = "#E31A1C"
AZUL = "#4169E1"
VERDE_OSCURO = "#2F5D1F"
LOGO_PATH = Path("montelimar_logo.png")


def unzip(upload, dest):
    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
        z.extractall(dest)


def files(root, ext):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() == ext]


def load_field(root):
    shps = files(root, ".shp")
    if not shps:
        raise ValueError("No se encontró ningún archivo .shp dentro del ZIP del campo.")
    shp = next((p for p in shps if "spr" not in p.stem.lower()), shps[0])
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        raise ValueError("Falta el archivo .prj del shapefile.")
    if not gdf.crs.is_projected:
        raise ValueError("El shapefile debe estar en coordenadas proyectadas, por ejemplo UTM.")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("El shapefile no contiene polígonos válidos.")
    return shp, gdf, unary_union(gdf.geometry)


def swath_from_agn(root, fallback):
    for p in files(root, ".agn"):
        try:
            for line in p.read_text(errors="ignore").splitlines():
                if re.match(r"^\s*36\s+", line):
                    v = float(line.split()[1])
                    if 1 <= v <= 100:
                        return v
        except Exception:
            pass
    return fallback


def passes_from_spron(root, crs):
    cands = [p for p in files(root, ".shp") if "spron" in p.stem.lower() or "spray" in p.stem.lower()]
    if not cands:
        return None
    g = gpd.read_file(cands[0])
    g = g.set_crs(crs) if g.crs is None else g.to_crs(crs)
    pts = g[g.geom_type == "Point"].copy()
    if len(pts) < 2:
        return None
    tcol = next((c for c in pts.columns if c.upper() in {"GPSTIME", "TIME", "UTC", "SECONDS"}), None)
    if tcol:
        pts = pts.sort_values(tcol).reset_index(drop=True)
    lines, cur = [], [pts.geometry.iloc[0]]
    for i in range(1, len(pts)):
        p = pts.geometry.iloc[i]
        prev = pts.geometry.iloc[i - 1]
        split = p.distance(prev) > 60
        if tcol:
            try:
                dt = float(pts[tcol].iloc[i]) - float(pts[tcol].iloc[i - 1])
                split = split or dt > 1.5 or dt < 0
            except Exception:
                pass
        if split:
            if len(cur) >= 2:
                lines.append(LineString(cur))
            cur = [p]
        else:
            cur.append(p)
    if len(cur) >= 2:
        lines.append(LineString(cur))
    return (lines, cands[0].name) if lines else None


def passes_from_t20(root):
    best = None
    for p in files(root, ".t20"):
        raw = p.read_bytes()
        rows = []
        for i in range(len(raw) // 68):
            r = raw[i * 68:(i + 1) * 68]
            if r[:2] != b"\xfb\xfb":
                continue
            try:
                x = struct.unpack_from("<i", r, 28)[0] / 10
                y = struct.unpack_from("<i", r, 32)[0] / 10
                spray = r[39]
            except Exception:
                continue
            if 10000 < abs(x) < 10000000 and 10000 < abs(y) < 20000000:
                rows.append((i, x, y, spray))
        if rows:
            df = pd.DataFrame(rows, columns=["i", "x", "y", "spray"])
            active = int((df["spray"] > 0).sum())
            if best is None or active > best[0]:
                best = (active, p, df)
    if not best or best[0] < 2:
        return None
    _, path, df = best
    lines, cur, prev = [], [], None
    for row in df.itertuples(index=False):
        if row.spray > 0:
            if prev is None or row.i == prev + 1:
                cur.append((row.x, row.y))
            else:
                if len(cur) >= 2:
                    lines.append(LineString(cur))
                cur = [(row.x, row.y)]
            prev = row.i
        else:
            if len(cur) >= 2:
                lines.append(LineString(cur))
            cur, prev = [], None
    if len(cur) >= 2:
        lines.append(LineString(cur))
    return (lines, path.name) if lines else None


def get_passes(root, crs):
    a = passes_from_spron(root, crs)
    if a:
        return a[0], a[1], "SPRAY ON"
    a = passes_from_t20(root)
    if a:
        return a[0], a[1], "T20"
    raise ValueError("No se encontró un SPRAY ON compatible ni se pudo leer un archivo .t20.")


def analyze(field, passes, swath, resolution):
    strips = [l.buffer(swath / 2, cap_style=2, join_style=2) for l in passes]
    coverage = unary_union(strips)
    inside = coverage.intersection(field)
    outside = coverage.difference(field)
    missing = field.difference(coverage)

    minx, miny, maxx, maxy = field.bounds
    pad = swath * 2
    minx -= pad; miny -= pad; maxx += pad; maxy += pad
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    if width * height > 16000000:
        factor = np.sqrt((width * height) / 16000000)
        resolution *= factor
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

    transform = from_origin(minx, maxy, resolution, resolution)
    field_mask = rasterize([(field, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    count = np.zeros((height, width), dtype=np.uint16)
    for strip in strips:
        count += rasterize([(strip, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")

    overlap = (count >= 2) & (field_mask == 1)
    single = (count == 1) & (field_mask == 1)
    cell_ha = resolution * resolution / 10000

    return {
        "field": field, "passes": passes, "inside": inside, "outside": outside, "missing": missing,
        "field_ha": field.area / 10000, "inside_ha": inside.area / 10000, "outside_ha": outside.area / 10000,
        "missing_ha": missing.area / 10000, "overlap_ha": float(overlap.sum() * cell_ha),
        "single_ha": float(single.sum() * cell_ha), "overlap": overlap, "resolution": resolution,
        "transform": transform
    }


def overlap_vector(result):
    mask = result["overlap"].astype("uint8")
    geoms = []
    if mask.any():
        for geom, value in shapes(mask, mask=mask.astype(bool), transform=result["transform"]):
            if value == 1:
                geoms.append(shape(geom))
    if not geoms:
        return None
    geom = unary_union(geoms).intersection(result["field"])
    return None if geom.is_empty else geom


def make_map(result, crs, name):
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    if not result["outside"].is_empty:
        gpd.GeoSeries([result["outside"]], crs=crs).plot(ax=ax, color=AZUL, alpha=0.95, linewidth=0)
    if not result["inside"].is_empty:
        gpd.GeoSeries([result["inside"]], crs=crs).plot(ax=ax, color=VERDE, alpha=0.95, linewidth=0)
    if not result["missing"].is_empty:
        gpd.GeoSeries([result["missing"]], crs=crs).plot(ax=ax, color=AMARILLO, alpha=0.98, linewidth=0)
    ov = overlap_vector(result)
    if ov is not None:
        gpd.GeoSeries([ov], crs=crs).plot(ax=ax, color=ROJO, alpha=0.98, linewidth=0)
    gpd.GeoSeries([result["field"]], crs=crs).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.25)
    gpd.GeoSeries(result["passes"], crs=crs).plot(ax=ax, color="black", linewidth=0.18, alpha=0.25)

    b = result["field"].bounds
    pad = max(30, (b[2] - b[0]) * 0.06)
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_title("MAPA DE CALIDAD DE APLICACIÓN AÉREA", fontweight="bold", fontsize=14, pad=10)
    ax.set_xlabel("Coordenada Este (m)")
    ax.set_ylabel("Coordenada Norte (m)")
    ax.grid(True, linewidth=0.22, alpha=0.20)
    ax.legend(handles=[
        Patch(facecolor=VERDE, label="Aplicación correcta"),
        Patch(facecolor=AMARILLO, label="Sin aplicar"),
        Patch(facecolor=ROJO, label="Traslape"),
        Patch(facecolor=AZUL, label="Fuera de área")
    ], loc="lower right", fontsize=8.5, frameon=True)

    out = io.BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out


def make_pdf(result, img, name, swath, source):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(A4), rightMargin=14, leftMargin=14, topMargin=10, bottomMargin=10)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Titulo", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18,
                                 leading=20, textColor=colors.HexColor(VERDE_OSCURO), alignment=TA_CENTER, spaceAfter=1)
    field_style = ParagraphStyle("Campo", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=15,
                                 leading=17, alignment=TA_CENTER, spaceAfter=1)
    subtitle_style = ParagraphStyle("Subtitulo", parent=styles["BodyText"], fontSize=9.5, leading=11,
                                    alignment=TA_CENTER, spaceAfter=3)
    normal_style = ParagraphStyle("NormalPeq", parent=styles["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT)
    section_style = ParagraphStyle("Seccion", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10.5,
                                   leading=12, textColor=colors.white, alignment=TA_CENTER)

    field_ha = result["field_ha"]
    rows = [
        ["Resultado", "ha", "%"],
        ["Área real del campo", f'{field_ha:.2f}', "100.00"],
        ["Aplicación correcta", f'{result["single_ha"]:.2f}', f'{result["single_ha"] / field_ha * 100:.2f}'],
        ["Sin aplicar", f'{result["missing_ha"]:.2f}', f'{result["missing_ha"] / field_ha * 100:.2f}'],
        ["Traslape", f'{result["overlap_ha"]:.2f}', f'{result["overlap_ha"] / field_ha * 100:.2f}'],
        ["Fuera de área", f'{result["outside_ha"]:.2f}', f'{result["outside_ha"] / field_ha * 100:.2f}'],
        ["Área total cubierta", f'{result["inside_ha"]:.2f}', f'{result["inside_ha"] / field_ha * 100:.2f}']
    ]

    results_table = Table(rows, colWidths=[128, 48, 48], rowHeights=[24, 28, 28, 28, 28, 28, 30])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8D4")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E5F0D9")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
    ]))

    info_table = Table([
        ["Ancho de faja:", f"{swath:.1f} m"],
        ["Fuente GPS:", source],
        ["Resolución de cálculo:", f'{result["resolution"]:.2f} m']
    ], colWidths=[115, 110], rowHeights=[22, 22, 22])
    info_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0B0B0")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6)
    ]))

    result_header = Table([[Paragraph("RESULTADOS", section_style)]], colWidths=[225], rowHeights=[24])
    result_header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(VERDE_OSCURO)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    left_panel = [result_header, Spacer(1, 4), results_table, Spacer(1, 8), info_table]
    map_image = Image(img, width=420, height=330)

    logo_items = []
    if LOGO_PATH.exists():
        logo_items.append(Image(str(LOGO_PATH), width=105, height=105))
        logo_items.append(Spacer(1, 5))
    logo_items.append(Paragraph("<b>INGENIO MONTELIMAR</b>", ParagraphStyle(
        "LogoText", parent=normal_style, fontSize=9, leading=11, alignment=TA_CENTER)))

    logo_panel = Table([[logo_items]], colWidths=[125], rowHeights=[330])
    logo_panel.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#A0A0A0"))
    ]))

    layout = Table([[left_panel, map_image, logo_panel]], colWidths=[235, 430, 130])
    layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
    ]))

    story = [
        Paragraph("INFORME DE APLICACIÓN AÉREA", title_style),
        Paragraph(f"<b>{name.upper()}</b>", field_style),
        Paragraph("Comparación del plano real del campo vs. registro GPS AgNav", subtitle_style),
        Spacer(1, 4),
        layout,
        Spacer(1, 6),
        Paragraph(
            "<b>Nota:</b> El área del campo se calcula directamente a partir de la geometría real del shapefile. "
            "La aplicación correcta corresponde a la superficie con una sola cobertura. El traslape representa áreas "
            "con dos o más coberturas y está incluido dentro del área total cubierta.",
            normal_style
        )
    ]

    doc.build(story)
    output.seek(0)
    return output


st.title("🚁 Calidad de Aplicación Aérea AgNav")
st.write(
    "Sube el ZIP del plano del campo y el ZIP del GPS AgNav. El sistema calcula automáticamente área real, "
    "aplicación correcta, traslape, áreas sin aplicar y fuera del campo."
)

col1, col2 = st.columns(2)
with col1:
    field_zip = st.file_uploader("Plano del campo (.ZIP)", type=["zip"])
with col2:
    agnav_zip = st.file_uploader("GPS AgNav (.ZIP)", type=["zip"])

name = st.text_input("Nombre del campo", "Campo")
swath_default = st.number_input("Ancho de faja por defecto (m)", 1.0, 100.0, 15.0, 0.5)
resolution = st.select_slider("Resolución de cálculo del traslape (m)", options=[0.20, 0.25, 0.50, 1.00], value=0.25)
st.caption("Recomendado: 0.25 m. La resolución controla la precisión del cálculo. El mapa muestra el traslape como franjas rojas continuas.")

process = st.button("PROCESAR APLICACIÓN", type="primary", disabled=not (field_zip and agnav_zip), use_container_width=True)

if process:
    try:
        with st.spinner("Procesando aplicación..."):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                field_root = tmp / "field"; agnav_root = tmp / "agnav"
                field_root.mkdir(); agnav_root.mkdir()
                unzip(field_zip, field_root); unzip(agnav_zip, agnav_root)
                shp, field_gdf, field = load_field(field_root)
                swath = swath_from_agn(agnav_root, swath_default)
                passes, source, mode = get_passes(agnav_root, field_gdf.crs)
                result = analyze(field, passes, swath, resolution)
                img = make_map(result, field_gdf.crs, name)
                pdf = make_pdf(result, img, name, swath, f"{source} ({mode})")

        st.success("Procesamiento terminado.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Área real", f'{result["field_ha"]:.2f} ha')
        c2.metric("Área total cubierta", f'{result["inside_ha"]:.2f} ha', f'{result["inside_ha"] / result["field_ha"] * 100:.2f} %')
        c3.metric("Sin aplicar", f'{result["missing_ha"]:.2f} ha')
        c4.metric("Traslape", f'{result["overlap_ha"]:.2f} ha')

        st.image(img, caption="Mapa de calidad de aplicación aérea", use_container_width=True)

        summary = pd.DataFrame({
            "Resultado": ["Área real del campo", "Aplicación correcta", "Sin aplicar", "Traslape", "Fuera de área", "Área total cubierta"],
            "Área (ha)": [result["field_ha"], result["single_ha"], result["missing_ha"], result["overlap_ha"], result["outside_ha"], result["inside_ha"]],
            "% del campo": [100, result["single_ha"] / result["field_ha"] * 100, result["missing_ha"] / result["field_ha"] * 100,
                           result["overlap_ha"] / result["field_ha"] * 100, result["outside_ha"] / result["field_ha"] * 100,
                           result["inside_ha"] / result["field_ha"] * 100]
        })
        st.dataframe(summary.style.format({"Área (ha)": "{:.2f}", "% del campo": "{:.2f}"}), use_container_width=True, hide_index=True)
        st.info(f"GPS procesado: {source} | Método: {mode} | Ancho de faja: {swath:.1f} m | Pasadas activas: {len(passes)}")
        st.download_button("⬇️ Descargar informe PDF", data=pdf.getvalue(), file_name=f"Informe_Aplicacion_Aerea_{name}.pdf", mime="application/pdf", use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo procesar la aplicación: {e}")

st.divider()
st.caption("Colores: verde = aplicación correcta; amarillo = sin aplicar; rojo = traslape; azul = fuera de área.")
