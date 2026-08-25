import streamlit as st
import io
import re
import struct
import tempfile
import zipfile
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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Aplicación aérea AgNav",
    page_icon="🚁",
    layout="wide"
)

VERDE = "#4CAF18"
AMARILLO = "#FFD700"
ROJO = "#E31A1C"
AZUL = "#4169E1"


# ============================================================
# FUNCIONES DE ARCHIVOS
# ============================================================

def unzip(upload, dest):
    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
        z.extractall(dest)


def files(root, ext):
    return [
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() == ext
    ]


# ============================================================
# LEER SHAPEFILE DEL CAMPO
# ============================================================

def load_field(root):
    shps = files(root, ".shp")

    if not shps:
        raise ValueError(
            "No se encontró ningún archivo .shp "
            "dentro del ZIP del campo."
        )

    shp = next(
        (
            p for p in shps
            if "spr" not in p.stem.lower()
        ),
        shps[0]
    )

    gdf = gpd.read_file(shp)

    if gdf.crs is None:
        raise ValueError(
            "El shapefile no tiene sistema de coordenadas. "
            "Verifica que el archivo .prj esté dentro del ZIP."
        )

    if not gdf.crs.is_projected:
        raise ValueError(
            "El shapefile debe estar en un sistema de "
            "coordenadas proyectado, por ejemplo UTM."
        )

    gdf = gdf[
        gdf.geom_type.isin([
            "Polygon",
            "MultiPolygon"
        ])
    ]

    if gdf.empty:
        raise ValueError(
            "El shapefile no contiene polígonos válidos."
        )

    field = unary_union(gdf.geometry)

    return shp, gdf, field


# ============================================================
# LEER ANCHO DE FAJA DESDE AGNAV
# ============================================================

def swath_from_agn(root, fallback):

    for p in files(root, ".agn"):

        try:

            for line in p.read_text(
                errors="ignore"
            ).splitlines():

                if re.match(
                    r"^\s*36\s+",
                    line
                ):

                    v = float(
                        line.split()[1]
                    )

                    if 1 <= v <= 100:
                        return v

        except Exception:
            pass

    return fallback


# ============================================================
# LEER SPRAY ON EN SHAPEFILE
# ============================================================

def passes_from_spron(root, crs):

    candidates = [
        p for p in files(root, ".shp")
        if (
            "spron" in p.stem.lower()
            or "spray" in p.stem.lower()
        )
    ]

    if not candidates:
        return None

    gdf = gpd.read_file(
        candidates[0]
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        gdf = gdf.to_crs(crs)

    points = gdf[
        gdf.geom_type == "Point"
    ].copy()

    if len(points) < 2:
        return None

    time_col = next(
        (
            c for c in points.columns
            if c.upper() in {
                "GPSTIME",
                "TIME",
                "UTC",
                "SECONDS"
            }
        ),
        None
    )

    if time_col:

        points = points.sort_values(
            time_col
        ).reset_index(drop=True)

    lines = []

    current = [
        points.geometry.iloc[0]
    ]

    for i in range(
        1,
        len(points)
    ):

        p = points.geometry.iloc[i]

        previous = points.geometry.iloc[
            i - 1
        ]

        split = (
            p.distance(previous) > 60
        )

        if time_col:

            try:

                dt = (
                    float(
                        points[
                            time_col
                        ].iloc[i]
                    )
                    -
                    float(
                        points[
                            time_col
                        ].iloc[i - 1]
                    )
                )

                split = (
                    split
                    or dt > 1.5
                    or dt < 0
                )

            except Exception:
                pass

        if split:

            if len(current) >= 2:

                lines.append(
                    LineString(current)
                )

            current = [p]

        else:

            current.append(p)

    if len(current) >= 2:

        lines.append(
            LineString(current)
        )

    if not lines:
        return None

    return (
        lines,
        candidates[0].name
    )


# ============================================================
# LEER ARCHIVO BINARIO T20
# ============================================================

def passes_from_t20(root):

    best = None

    for p in files(
        root,
        ".t20"
    ):

        raw = p.read_bytes()

        rows = []

        record_size = 68

        for i in range(
            len(raw) // record_size
        ):

            r = raw[
                i * record_size:
                (i + 1) * record_size
            ]

            if r[:2] != b"\xfb\xfb":
                continue

            try:

                x = (
                    struct.unpack_from(
                        "<i",
                        r,
                        28
                    )[0]
                    / 10
                )

                y = (
                    struct.unpack_from(
                        "<i",
                        r,
                        32
                    )[0]
                    / 10
                )

                spray = r[39]

            except Exception:
                continue

            if (
                10000 < abs(x) < 10000000
                and
                10000 < abs(y) < 20000000
            ):

                rows.append(
                    (
                        i,
                        x,
                        y,
                        spray
                    )
                )

        if not rows:
            continue

        df = pd.DataFrame(
            rows,
            columns=[
                "i",
                "x",
                "y",
                "spray"
            ]
        )

        active = int(
            (
                df["spray"] > 0
            ).sum()
        )

        if (
            best is None
            or active > best[0]
        ):

            best = (
                active,
                p,
                df
            )

    if (
        best is None
        or best[0] < 2
    ):

        return None

    _, path, df = best

    lines = []

    current = []

    previous = None

    for row in df.itertuples(
        index=False
    ):

        if row.spray > 0:

            if (
                previous is None
                or row.i == previous + 1
            ):

                current.append(
                    (
                        row.x,
                        row.y
                    )
                )

            else:

                if len(current) >= 2:

                    lines.append(
                        LineString(
                            current
                        )
                    )

                current = [
                    (
                        row.x,
                        row.y
                    )
                ]

            previous = row.i

        else:

            if len(current) >= 2:

                lines.append(
                    LineString(
                        current
                    )
                )

            current = []

            previous = None

    if len(current) >= 2:

        lines.append(
            LineString(
                current
            )
        )

    if not lines:
        return None

    return (
        lines,
        path.name
    )


# ============================================================
# IDENTIFICAR PASADAS
# ============================================================

def get_passes(root, crs):

    result = passes_from_spron(
        root,
        crs
    )

    if result:

        return (
            result[0],
            result[1],
            "SPRAY ON"
        )

    result = passes_from_t20(
        root
    )

    if result:

        return (
            result[0],
            result[1],
            "T20"
        )

    raise ValueError(
        "No se encontró un SPRAY ON compatible "
        "ni se pudo leer un archivo .t20."
    )


# ============================================================
# ANÁLISIS ESPACIAL
# ============================================================

def analyze(
    field,
    passes,
    swath,
    resolution
):

    strips = [
        line.buffer(
            swath / 2,
            cap_style=2,
            join_style=2
        )
        for line in passes
    ]

    coverage = unary_union(
        strips
    )

    inside = coverage.intersection(
        field
    )

    outside = coverage.difference(
        field
    )

    missing = field.difference(
        coverage
    )

    minx, miny, maxx, maxy = (
        field.bounds
    )

    pad = swath * 2

    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = int(
        np.ceil(
            (maxx - minx)
            / resolution
        )
    )

    height = int(
        np.ceil(
            (maxy - miny)
            / resolution
        )
    )

    max_cells = 16000000

    if (
        width * height
        > max_cells
    ):

        factor = np.sqrt(
            (
                width * height
            )
            /
            max_cells
        )

        resolution *= factor

        width = int(
            np.ceil(
                (maxx - minx)
                /
                resolution
            )
        )

        height = int(
            np.ceil(
                (maxy - miny)
                /
                resolution
            )
        )

    transform = from_origin(
        minx,
        maxy,
        resolution,
        resolution
    )

    field_mask = rasterize(
        [
            (
                field,
                1
            )
        ],
        out_shape=(
            height,
            width
        ),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    count = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint16
    )

    for strip in strips:

        count += rasterize(
            [
                (
                    strip,
                    1
                )
            ],
            out_shape=(
                height,
                width
            ),
            transform=transform,
            fill=0,
            dtype="uint8"
        )

    overlap = (
        (count >= 2)
        &
        (field_mask == 1)
    )

    single = (
        (count == 1)
        &
        (field_mask == 1)
    )

    cell_ha = (
        resolution
        *
        resolution
        /
        10000
    )

    return {

        "field": field,

        "passes": passes,

        "inside": inside,

        "outside": outside,

        "missing": missing,

        "field_ha":
            field.area
            /
            10000,

        "inside_ha":
            inside.area
            /
            10000,

        "outside_ha":
            outside.area
            /
            10000,

        "missing_ha":
            missing.area
            /
            10000,

        "overlap_ha":
            float(
                overlap.sum()
                *
                cell_ha
            ),

        "single_ha":
            float(
                single.sum()
                *
                cell_ha
            ),

        "overlap":
            overlap,

        "resolution":
            resolution,

        "transform":
            transform

    }


# ============================================================
# CONVERTIR TRASLAPE A POLÍGONO
# ============================================================

def overlap_vector(result):

    mask = result[
        "overlap"
    ].astype(
        "uint8"
    )

    geometries = []

    if mask.any():

        for geom, value in shapes(

            mask,

            mask=mask.astype(
                bool
            ),

            transform=result[
                "transform"
            ]

        ):

            if value == 1:

                geometries.append(
                    shape(geom)
                )

    if not geometries:
        return None

    geometry = unary_union(
        geometries
    ).intersection(
        result["field"]
    )

    if geometry.is_empty:
        return None

    return geometry


# ============================================================
# GENERAR MAPA
# ============================================================

def make_map(
    result,
    crs,
    name
):

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            6.3
        )
    )

    # FUERA DE ÁREA
    if not result[
        "outside"
    ].is_empty:

        gpd.GeoSeries(
            [
                result[
                    "outside"
                ]
            ],
            crs=crs
        ).plot(
            ax=ax,
            color=AZUL,
            alpha=0.95,
            linewidth=0
        )

    # COBERTURA INTERNA
    if not result[
        "inside"
    ].is_empty:

        gpd.GeoSeries(
            [
                result[
                    "inside"
                ]
            ],
            crs=crs
        ).plot(
            ax=ax,
            color=VERDE,
            alpha=0.95,
            linewidth=0
        )

    # SIN APLICAR
    if not result[
        "missing"
    ].is_empty:

        gpd.GeoSeries(
            [
                result[
                    "missing"
                ]
            ],
            crs=crs
        ).plot(
            ax=ax,
            color=AMARILLO,
            alpha=0.98,
            linewidth=0
        )

    # TRASLAPE
    overlap_geom = (
        overlap_vector(
            result
        )
    )

    if overlap_geom is not None:

        gpd.GeoSeries(
            [
                overlap_geom
            ],
            crs=crs
        ).plot(
            ax=ax,
            color=ROJO,
            alpha=0.98,
            linewidth=0
        )

    # BORDE DEL CAMPO
    gpd.GeoSeries(
        [
            result[
                "field"
            ]
        ],
        crs=crs
    ).plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.3
    )

    # LÍNEA CENTRAL DE PASADAS
    gpd.GeoSeries(
        result[
            "passes"
        ],
        crs=crs
    ).plot(
        ax=ax,
        color="black",
        linewidth=0.14,
        alpha=0.13
    )

    bounds = (
        result[
            "field"
        ].bounds
    )

    pad = max(
        30,
        (
            bounds[2]
            -
            bounds[0]
        )
        *
        0.06
    )

    ax.set_xlim(
        bounds[0] - pad,
        bounds[2] + pad
    )

    ax.set_ylim(
        bounds[1] - pad,
        bounds[3] + pad
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        "Mapa de calidad de aplicación aérea - "
        +
        name,
        fontweight="bold",
        fontsize=14
    )

    ax.set_xlabel(
        "Coordenada Este (m)"
    )

    ax.set_ylabel(
        "Coordenada Norte (m)"
    )

    ax.grid(
        True,
        linewidth=0.22,
        alpha=0.20
    )

    ax.legend(
        handles=[
            Patch(
                facecolor=VERDE,
                label="Aplicación correcta"
            ),

            Patch(
                facecolor=AMARILLO,
                label="Sin aplicar"
            ),

            Patch(
                facecolor=ROJO,
                label="Traslape"
            ),

            Patch(
                facecolor=AZUL,
                label="Fuera de área"
            )
        ],
        loc="best",
        fontsize=8.5,
        frameon=True
    )

    output = io.BytesIO()

    fig.tight_layout()

    fig.savefig(
        output,
        format="png",
        dpi=240,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    output.seek(0)

    return output


# ============================================================
# GENERAR PDF DE UNA SOLA HOJA
# ============================================================

def make_pdf(
    result,
    img,
    name,
    swath,
    source
):

    output = io.BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=20,
        leftMargin=20,
        topMargin=16,
        bottomMargin=16
    )

    styles = (
        getSampleStyleSheet()
    )

    title_style = (
        styles["Title"]
    )

    title_style.fontSize = 15

    title_style.leading = 17

    title_style.spaceAfter = 3


    subtitle_style = (
        styles["Heading2"]
    )

    subtitle_style.fontSize = 11

    subtitle_style.leading = 13

    subtitle_style.spaceAfter = 2


    normal_style = (
        styles["BodyText"]
    )

    normal_style.fontSize = 8

    normal_style.leading = 10


    field_ha = result[
        "field_ha"
    ]


    rows = [

        [
            "Resultado",
            "ha",
            "%"
        ],

        [
            "Área real del campo",
            f'{field_ha:.2f}',
            "100.00"
        ],

        [
            "Aplicación correcta",

            f'{result["single_ha"]:.2f}',

            f'''
            {
                result["single_ha"]
                /
                field_ha
                *
                100
            :.2f}
            '''
        ],

        [
            "Sin aplicar",

            f'{result["missing_ha"]:.2f}',

            f'''
            {
                result["missing_ha"]
                /
                field_ha
                *
                100
            :.2f}
            '''
        ],

        [
            "Traslape",

            f'{result["overlap_ha"]:.2f}',

            f'''
            {
                result["overlap_ha"]
                /
                field_ha
                *
                100
            :.2f}
            '''
        ],

        [
            "Fuera de área",

            f'{result["outside_ha"]:.2f}',

            f'''
            {
                result["outside_ha"]
                /
                field_ha
                *
                100
            :.2f}
            '''
        ],

        [
            "Área total cubierta",

            f'{result["inside_ha"]:.2f}',

            f'''
            {
                result["inside_ha"]
                /
                field_ha
                *
                100
            :.2f}
            '''
        ]

    ]


    results_table = Table(

        rows,

        colWidths=[
            155,
            55,
            55
        ],

        rowHeights=25

    )


    results_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#D9E2F3"
                )
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "ALIGN",
                (1, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "BACKGROUND",
                (0, -1),
                (-1, -1),
                colors.HexColor(
                    "#E2F0D9"
                )
            ),

            (
                "FONTNAME",
                (0, -1),
                (-1, -1),
                "Helvetica-Bold"
            )

        ])

    )


    map_image = Image(
        img,
        width=500,
        height=285
    )


    left_panel = [

        Paragraph(
            "<b>RESULTADOS</b>",
            normal_style
        ),

        Spacer(
            1,
            5
        ),

        results_table,

        Spacer(
            1,
            9
        ),

        Paragraph(
            f'''
            <b>Ancho de faja:</b>
            {swath:.1f} m
            ''',
            normal_style
        ),

        Paragraph(
            f'''
            <b>Fuente GPS:</b>
            {source}
            ''',
            normal_style
        ),

        Paragraph(
            f'''
            <b>Resolución de cálculo:</b>
            {
                result["resolution"]
            :.2f} m
            ''',
            normal_style
        )

    ]


    layout = Table(

        [
            [
                left_panel,
                map_image
            ]
        ],

        colWidths=[
            280,
            510
        ]

    )


    layout.setStyle(

        TableStyle([

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )

        ])

    )


    story = [

        Paragraph(
            "INFORME DE APLICACIÓN AÉREA",
            title_style
        ),

        Paragraph(
            f'''
            <b>
            {name.upper()}
            </b>
            ''',
            subtitle_style
        ),

        Paragraph(
            "Comparación del plano real del campo vs. registro GPS AgNav",
            normal_style
        ),

        Spacer(
            1,
            6
        ),

        layout,

        Spacer(
            1,
            6
        ),

        Paragraph(

            "<b>Nota:</b> "
            "El área del campo se calcula directamente "
            "a partir de la geometría real del shapefile. "
            "La aplicación correcta corresponde a la "
            "superficie con una sola cobertura. "
            "El traslape representa áreas con dos o más "
            "coberturas y está incluido dentro del área "
            "total cubierta.",

            normal_style

        )

    ]


    doc.build(
        story
    )

    output.seek(0)

    return output


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

st.title(
    "🚁 Calidad de Aplicación Aérea AgNav"
)

st.write(
    "Sube el ZIP del plano del campo y el ZIP del GPS AgNav. "
    "El sistema calcula automáticamente área real, aplicación "
    "correcta, traslape, áreas sin aplicar y fuera del campo."
)


col1, col2 = st.columns(
    2
)


with col1:

    field_zip = st.file_uploader(

        "Plano del campo (.ZIP)",

        type=[
            "zip"
        ]

    )


with col2:

    agnav_zip = st.file_uploader(

        "GPS AgNav (.ZIP)",

        type=[
            "zip"
        ]

    )


name = st.text_input(
    "Nombre del campo",
    "Campo"
)


swath_default = st.number_input(

    "Ancho de faja por defecto (m)",

    min_value=1.0,

    max_value=100.0,

    value=15.0,

    step=0.5

)


resolution = st.select_slider(

    "Resolución de cálculo del traslape (m)",

    options=[
        0.20,
        0.25,
        0.50,
        1.00
    ],

    value=0.25

)


st.caption(

    "Recomendado: 0.25 m. "
    "La resolución controla la precisión del cálculo. "
    "El mapa muestra el traslape como franjas rojas continuas."

)


process = st.button(

    "PROCESAR APLICACIÓN",

    type="primary",

    disabled=not (
        field_zip
        and
        agnav_zip
    ),

    use_container_width=True

)


if process:

    try:

        with st.spinner(
            "Procesando aplicación..."
        ):

            with tempfile.TemporaryDirectory() as tmp:

                tmp = Path(tmp)

                field_root = (
                    tmp
                    /
                    "field"
                )

                agnav_root = (
                    tmp
                    /
                    "agnav"
                )

                field_root.mkdir()

                agnav_root.mkdir()


                unzip(
                    field_zip,
                    field_root
                )


                unzip(
                    agnav_zip,
                    agnav_root
                )


                shp, field_gdf, field = (
                    load_field(
                        field_root
                    )
                )


                swath = swath_from_agn(

                    agnav_root,

                    swath_default

                )


                passes, source, mode = (

                    get_passes(

                        agnav_root,

                        field_gdf.crs

                    )

                )


                result = analyze(

                    field,

                    passes,

                    swath,

                    resolution

                )


                img = make_map(

                    result,

                    field_gdf.crs,

                    name

                )


                pdf = make_pdf(

                    result,

                    img,

                    name,

                    swath,

                    f"{source} ({mode})"

                )


        st.success(
            "Procesamiento terminado."
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(

            "Área real",

            f'''
            {
                result["field_ha"]
            :.2f} ha
            '''

        )


        c2.metric(

            "Área total cubierta",

            f'''
            {
                result["inside_ha"]
            :.2f} ha
            ''',

            f'''
            {
                result["inside_ha"]
                /
                result["field_ha"]
                *
                100
            :.2f} %
            '''

        )


        c3.metric(

            "Sin aplicar",

            f'''
            {
                result["missing_ha"]
            :.2f} ha
            '''

        )


        c4.metric(

            "Traslape",

            f'''
            {
                result["overlap_ha"]
            :.2f} ha
            '''

        )


        st.image(

            img,

            caption=(
                "Mapa de calidad "
                "de aplicación aérea"
            ),

            use_container_width=True

        )


        summary = pd.DataFrame({

            "Resultado": [

                "Área real del campo",

                "Aplicación correcta",

                "Sin aplicar",

                "Traslape",

                "Fuera de área",

                "Área total cubierta"

            ],

            "Área (ha)": [

                result[
                    "field_ha"
                ],

                result[
                    "single_ha"
                ],

                result[
                    "missing_ha"
                ],

                result[
                    "overlap_ha"
                ],

                result[
                    "outside_ha"
                ],

                result[
                    "inside_ha"
                ]

            ],

            "% del campo": [

                100,

                result[
                    "single_ha"
                ]
                /
                result[
                    "field_ha"
                ]
                *
                100,

                result[
                    "missing_ha"
                ]
                /
                result[
                    "field_ha"
                ]
                *
                100,

                result[
                    "overlap_ha"
                ]
                /
                result[
                    "field_ha"
                ]
                *
                100,

                result[
                    "outside_ha"
                ]
                /
                result[
                    "field_ha"
                ]
                *
                100,

                result[
                    "inside_ha"
                ]
                /
                result[
                    "field_ha"
                ]
                *
                100

            ]

        })


        st.dataframe(

            summary.style.format({

                "Área (ha)":
                    "{:.2f}",

                "% del campo":
                    "{:.2f}"

            }),

            use_container_width=True,

            hide_index=True

        )


        st.info(

            f"GPS procesado: {source} | "
            f"Método: {mode} | "
            f"Ancho de faja: {swath:.1f} m | "
            f"Pasadas activas: {len(passes)}"

        )


        st.download_button(

            "⬇️ Descargar informe PDF",

            data=pdf.getvalue(),

            file_name=(
                f"Informe_Aplicacion_Aerea_{name}.pdf"
            ),

            mime="application/pdf",

            use_container_width=True

        )


    except Exception as e:

        st.error(
            f"No se pudo procesar la aplicación: {e}"
        )


st.divider()


st.caption(

    "Colores: verde = aplicación correcta; "
    "amarillo = sin aplicar; "
    "rojo = traslape; "
    "azul = fuera de área."

)
