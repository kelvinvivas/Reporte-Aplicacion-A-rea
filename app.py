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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

st.set_page_config(page_title="Aplicación aérea AgNav", page_icon="🚁", layout="wide")

VERDE="#4CAF18"
AMARILLO="#FFD700"
ROJO="#E31A1C"
AZUL="#4169E1"

def unzip(upload, dest):
    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
        z.extractall(dest)

def files(root, ext):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower()==ext]

def load_field(root):
    shps=files(root,".shp")
    if not shps:
        raise ValueError("No se encontró .shp en el ZIP del campo.")
    shp=next((p for p in shps if "spr" not in p.stem.lower()),shps[0])
    gdf=gpd.read_file(shp)
    if gdf.crs is None:
        raise ValueError("Falta el .prj del shapefile.")
    if not gdf.crs.is_projected:
        raise ValueError("El SHP debe estar en coordenadas proyectadas, por ejemplo UTM.")
    gdf=gdf[gdf.geom_type.isin(["Polygon","MultiPolygon"])]
    if gdf.empty:
        raise ValueError("El SHP no contiene polígonos.")
    return shp,gdf,unary_union(gdf.geometry)

def swath_from_agn(root,fallback):
    for p in files(root,".agn"):
        try:
            for line in p.read_text(errors="ignore").splitlines():
                if re.match(r"^\s*36\s+",line):
                    v=float(line.split()[1])
                    if 1<=v<=100:
                        return v
        except:
            pass
    return fallback

def passes_from_spron(root,crs):
    cands=[p for p in files(root,".shp") if "spron" in p.stem.lower() or "spray" in p.stem.lower()]
    if not cands:
        return None
    g=gpd.read_file(cands[0])
    if g.crs is None:
        g=g.set_crs(crs)
    else:
        g=g.to_crs(crs)
    pts=g[g.geom_type=="Point"].copy()
    if len(pts)<2:
        return None
    tcol=next((c for c in pts.columns if c.upper() in {"GPSTIME","TIME","UTC","SECONDS"}),None)
    if tcol:
        pts=pts.sort_values(tcol).reset_index(drop=True)
    lines=[]; cur=[pts.geometry.iloc[0]]
    for i in range(1,len(pts)):
        p=pts.geometry.iloc[i]; prev=pts.geometry.iloc[i-1]
        split=p.distance(prev)>60
        if tcol:
            try:
                dt=float(pts[tcol].iloc[i])-float(pts[tcol].iloc[i-1])
                split=split or dt>1.5 or dt<0
            except:
                pass
        if split:
            if len(cur)>=2:
                lines.append(LineString(cur))
            cur=[p]
        else:
            cur.append(p)
    if len(cur)>=2:
        lines.append(LineString(cur))
    return (lines,cands[0].name) if lines else None

def passes_from_t20(root):
    best=None
    for p in files(root,".t20"):
        raw=p.read_bytes(); rows=[]
        for i in range(len(raw)//68):
            r=raw[i*68:(i+1)*68]
            if r[:2]!=b"\xfb\xfb":
                continue
            x=struct.unpack_from("<i",r,28)[0]/10
            y=struct.unpack_from("<i",r,32)[0]/10
            s=r[39]
            if 10000<abs(x)<10000000 and 10000<abs(y)<20000000:
                rows.append((i,x,y,s))
        if rows:
            d=pd.DataFrame(rows,columns=["i","x","y","s"])
            n=int((d.s>0).sum())
            if best is None or n>best[0]:
                best=(n,p,d)
    if not best or best[0]<2:
        return None
    _,p,d=best
    lines=[]; cur=[]; prev=None
    for r in d.itertuples(index=False):
        if r.s>0:
            if prev is None or r.i==prev+1:
                cur.append((r.x,r.y))
            else:
                if len(cur)>=2:
                    lines.append(LineString(cur))
                cur=[(r.x,r.y)]
            prev=r.i
        else:
            if len(cur)>=2:
                lines.append(LineString(cur))
            cur=[]; prev=None
    if len(cur)>=2:
        lines.append(LineString(cur))
    return (lines,p.name) if lines else None

def get_passes(root,crs):
    a=passes_from_spron(root,crs)
    if a:
        return a[0],a[1],"SPRAY ON"
    a=passes_from_t20(root)
    if a:
        return a[0],a[1],"T20"
    raise ValueError("No encontré SPRAY ON compatible ni pude leer un .t20.")

def analyze(field,passes,swath,res):
    strips=[l.buffer(swath/2,cap_style=2,join_style=2) for l in passes]
    cov=unary_union(strips)
    inside=cov.intersection(field)
    outside=cov.difference(field)
    missing=field.difference(cov)

    minx,miny,maxx,maxy=field.bounds
    pad=swath*2
    minx-=pad; miny-=pad; maxx+=pad; maxy+=pad
    W=int(np.ceil((maxx-minx)/res))
    H=int(np.ceil((maxy-miny)/res))
    if W*H>16000000:
        res*=np.sqrt((W*H)/16000000)
        W=int(np.ceil((maxx-minx)/res))
        H=int(np.ceil((maxy-miny)/res))
    tr=from_origin(minx,maxy,res,res)

    fm=rasterize([(field,1)],out_shape=(H,W),transform=tr,fill=0,dtype="uint8")
    cnt=np.zeros((H,W),dtype=np.uint16)
    for s in strips:
        cnt+=rasterize([(s,1)],out_shape=(H,W),transform=tr,fill=0,dtype="uint8")

    ov=(cnt>=2)&(fm==1)
    single=(cnt==1)&(fm==1)
    cell=res*res/10000

    return dict(
        field=field,passes=passes,inside=inside,outside=outside,missing=missing,
        field_ha=field.area/10000,
        inside_ha=inside.area/10000,
        outside_ha=outside.area/10000,
        missing_ha=missing.area/10000,
        overlap_ha=float(ov.sum()*cell),
        single_ha=float(single.sum()*cell),
        ov=ov,res=res,transform=tr
    )

def overlap_vector(r):
    mask=r["ov"].astype("uint8")
    geoms=[]
    if mask.any():
        for geom,value in shapes(mask,mask=mask.astype(bool),transform=r["transform"]):
            if value==1:
                geoms.append(shape(geom))
    if not geoms:
        return None
    geom=unary_union(geoms).intersection(r["field"])
    return None if geom.is_empty else geom


def _nice_scale_length(width_m):
    target = max(width_m * 0.28, 1)
    power = 10 ** np.floor(np.log10(target))
    value = target / power
    if value < 1.5:
        nice = 1
    elif value < 3.5:
        nice = 2
    elif value < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * power


def make_map(r, crs, name):
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(9.2, 7.0))

    if not r["outside"].is_empty:
        gpd.GeoSeries([r["outside"]], crs=crs).plot(
            ax=ax, color=AZUL, alpha=0.96, linewidth=0, zorder=1
        )

    if not r["inside"].is_empty:
        gpd.GeoSeries([r["inside"]], crs=crs).plot(
            ax=ax, color=VERDE, alpha=0.96, linewidth=0, zorder=2
        )

    if not r["missing"].is_empty:
        gpd.GeoSeries([r["missing"]], crs=crs).plot(
            ax=ax, color=AMARILLO, alpha=0.99, linewidth=0, zorder=3
        )

    ovgeom = overlap_vector(r)
    if ovgeom is not None:
        gpd.GeoSeries([ovgeom], crs=crs).plot(
            ax=ax, color=ROJO, alpha=0.99, linewidth=0, zorder=4
        )

    if r["passes"]:
        gpd.GeoSeries(r["passes"], crs=crs).plot(
            ax=ax, color="black", linewidth=0.38, alpha=0.62, zorder=5
        )

    gpd.GeoSeries([r["field"]], crs=crs).plot(
        ax=ax, facecolor="none", edgecolor="black", linewidth=1.45, zorder=6
    )

    minx, miny, maxx, maxy = r["field"].bounds
    dx = maxx - minx
    dy = maxy - miny
    pad_x = max(25, dx * 0.08)
    pad_y = max(25, dy * 0.08)

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")

    fmt = FuncFormatter(lambda x, pos: f"{x:,.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.set_xlabel("Coordenada Este (m)", fontsize=10)
    ax.set_ylabel("Coordenada Norte (m)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.25)

    legend_handles = [
        Patch(facecolor=VERDE, edgecolor="#398916", label="Aplicación correcta"),
        Patch(facecolor=AMARILLO, edgecolor="#C9A900", label="Sin aplicar"),
        Patch(facecolor=ROJO, edgecolor="#A91618", label="Traslape"),
        Patch(facecolor=AZUL, edgecolor="#214F9C", label="Fuera de área"),
        Patch(facecolor="white", edgecolor="black", label="Límite del campo"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.13),
        fontsize=8.4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
        handleheight=1.5,
        labelspacing=0.75,
    )

    ax.text(
        0.94, 0.94, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", zorder=20
    )
    ax.annotate(
        "",
        xy=(0.94, 0.91),
        xytext=(0.94, 0.80),
        xycoords="axes fraction",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=5,
            headwidth=15,
            headlength=20
        ),
        zorder=20
    )

    x0, x1 = ax.get_xlim()
    visible_width = x1 - x0
    total_scale = _nice_scale_length(visible_width)
    segments = 4
    seg = total_scale / segments

    y0, y1 = ax.get_ylim()
    visible_height = y1 - y0
    bar_x0 = x0 + visible_width * 0.61
    bar_y = y0 + visible_height * 0.045
    bar_h = visible_height * 0.014

    for i in range(segments):
        rect = Rectangle(
            (bar_x0 + i * seg, bar_y),
            seg,
            bar_h,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black",
            linewidth=0.7,
            zorder=20
        )
        ax.add_patch(rect)

    for i in range(segments + 1):
        value = int(round(i * seg))
        ax.text(
            bar_x0 + i * seg,
            bar_y + bar_h * 1.65,
            f"{value}",
            ha="center", va="bottom",
            fontsize=7.5, zorder=20
        )

    ax.text(
        bar_x0 + total_scale + visible_width * 0.012,
        bar_y + bar_h * 1.65,
        "m",
        ha="left", va="bottom",
        fontsize=7.5, zorder=20
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    out = io.BytesIO()
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.985)
    fig.savefig(out, format="png", dpi=260, facecolor="white")
    plt.close(fig)
    out.seek(0)
    return out


def _draw_centered_text(c, text, x, y, w, font="Helvetica-Bold",
                        size=10, color=colors.black):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawCentredString(x + w / 2, y, text)


def _draw_result_table(c, r, x, y_top, w):
    header_h = 25
    row_h = 34
    col1 = w * 0.57
    col2 = w * 0.22
    col3 = w - col1 - col2

    data = [
        ("Área real del campo", r["field_ha"], 100.00, None),
        ("Aplicación correcta", r["single_ha"], r["single_ha"] / r["field_ha"] * 100, VERDE),
        ("Sin aplicar", r["missing_ha"], r["missing_ha"] / r["field_ha"] * 100, AMARILLO),
        ("Traslape", r["overlap_ha"], r["overlap_ha"] / r["field_ha"] * 100, ROJO),
        ("Fuera de área", r["outside_ha"], r["outside_ha"] / r["field_ha"] * 100, AZUL),
        ("Área total cubierta", r["inside_ha"], r["inside_ha"] / r["field_ha"] * 100, "TOTAL"),
    ]

    y = y_top - header_h
    c.setFillColor(colors.HexColor("#DCE8D4"))
    c.rect(x, y, w, header_h, fill=1, stroke=0)

    c.setStrokeColor(colors.HexColor("#A0A0A0"))
    c.setLineWidth(0.45)
    for vx in (x, x + col1, x + col1 + col2, x + w):
        c.line(vx, y, vx, y + header_h)
    c.line(x, y, x + w, y)
    c.line(x, y + header_h, x + w, y + header_h)

    _draw_centered_text(c, "Resultado", x, y + 8, col1, size=8)
    _draw_centered_text(c, "ha", x + col1, y + 8, col2, size=8)
    _draw_centered_text(c, "%", x + col1 + col2, y + 8, col3, size=8)

    for label, ha, pct, swatch in data:
        y -= row_h

        if swatch == "TOTAL":
            c.setFillColor(colors.HexColor("#E5F0D9"))
            c.rect(x, y, w, row_h, fill=1, stroke=0)

        c.setStrokeColor(colors.HexColor("#A0A0A0"))
        c.setLineWidth(0.45)
        c.rect(x, y, w, row_h, fill=0, stroke=1)
        c.line(x + col1, y, x + col1, y + row_h)
        c.line(x + col1 + col2, y, x + col1 + col2, y + row_h)

        text_x = x + 8
        if swatch not in (None, "TOTAL"):
            c.setFillColor(colors.HexColor(swatch))
            c.rect(x + 7, y + 10, 12, 14, fill=1, stroke=0)
            text_x = x + 25

        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold" if swatch == "TOTAL" else "Helvetica", 7.8)
        c.drawString(text_x, y + 12, label)
        c.drawCentredString(x + col1 + col2 / 2, y + 12, f"{ha:.2f}")
        c.drawCentredString(x + col1 + col2 + col3 / 2, y + 12, f"{pct:.2f}")

    return y


def _draw_info_box(c, x, y_top, w, swath, source, resolution):
    row_h = 24
    rows = [
        ("Ancho de faja:", f"{swath:.1f} m"),
        ("Fuente GPS:", source),
        ("Resolución de cálculo:", f"{resolution:.2f} m"),
    ]
    y = y_top - row_h * len(rows)

    c.setStrokeColor(colors.HexColor("#A0A0A0"))
    c.setLineWidth(0.55)
    c.rect(x, y, w, row_h * len(rows), fill=0, stroke=1)

    split = x + w * 0.55
    c.line(split, y, split, y + row_h * len(rows))

    for i, (lab, val) in enumerate(rows):
        row_y = y + row_h * (len(rows) - 1 - i)
        if i > 0:
            c.line(x, row_y + row_h, x + w, row_y + row_h)

        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(x + 8, row_y + 8.5, lab)

        c.setFont("Helvetica", 7.0)
        c.drawString(split + 7, row_y + 8.5, val)

    return y


def make_pdf(r, img, name, swath, source):
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    out = io.BytesIO()
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(out, pagesize=(page_w, page_h))

    dark_green = colors.HexColor("#285C1F")
    light_border = colors.HexColor("#9A9A9A")

    c.setStrokeColor(colors.HexColor("#555555"))
    c.setLineWidth(0.6)
    c.rect(9, 9, page_w - 18, page_h - 18, fill=0, stroke=1)

    c.setFillColor(dark_green)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, page_h - 37, "INFORME DE APLICACIÓN AÉREA")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(page_w / 2, page_h - 65, name.upper())

    c.setFont("Helvetica", 9.5)
    c.drawCentredString(
        page_w / 2,
        page_h - 84,
        "Comparación del plano real del campo vs. registro GPS AgNav"
    )

    header_line_y = page_h - 96
    c.setStrokeColor(dark_green)
    c.setLineWidth(1.2)
    c.line(10, header_line_y, page_w - 10, header_line_y)

    left_x, left_w = 18, 215
    map_x, map_w = 242, 430
    logo_x, logo_w = 681, page_w - 699
    panel_top = header_line_y - 13
    section_h = 26
    panel_bottom = 80

    c.setFillColor(dark_green)
    c.rect(left_x, panel_top - section_h, left_w, section_h, fill=1, stroke=0)
    _draw_centered_text(
        c, "RESULTADOS", left_x, panel_top - section_h + 8,
        left_w, size=10.5, color=colors.white
    )

    table_y = _draw_result_table(
        c, r, left_x, panel_top - section_h - 5, left_w
    )
    _draw_info_box(
        c, left_x, table_y - 10, left_w, swath, source, r["res"]
    )

    c.setFillColor(dark_green)
    c.rect(map_x, panel_top - section_h, map_w, section_h, fill=1, stroke=0)
    _draw_centered_text(
        c, "MAPA DE CALIDAD DE APLICACIÓN AÉREA",
        map_x, panel_top - section_h + 8, map_w,
        size=10.1, color=colors.white
    )

    map_box_y = panel_bottom + 4
    map_box_h = panel_top - section_h - map_box_y - 5
    c.setStrokeColor(light_border)
    c.setLineWidth(0.55)
    c.rect(map_x, map_box_y, map_w, map_box_h, fill=0, stroke=1)

    c.drawImage(
        ImageReader(img),
        map_x + 4,
        map_box_y + 4,
        width=map_w - 8,
        height=map_box_h - 8,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto"
    )

    logo_panel_y = panel_bottom + 4
    logo_panel_h = panel_top - logo_panel_y
    c.setStrokeColor(light_border)
    c.setLineWidth(0.55)
    c.rect(logo_x, logo_panel_y, logo_w, logo_panel_h, fill=0, stroke=1)

    logo_path = Path("montelimar_logo.png")
    if logo_path.exists():
        try:
            logo = ImageReader(str(logo_path))
            iw, ih = logo.getSize()
            max_w = logo_w - 18
            max_h = 150
            scale = min(max_w / iw, max_h / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(
                logo,
                logo_x + (logo_w - dw) / 2,
                logo_panel_y + logo_panel_h * 0.48,
                width=dw,
                height=dh,
                mask="auto"
            )
        except Exception:
            pass

    c.setFillColor(dark_green)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        logo_x + logo_w / 2,
        logo_panel_y + logo_panel_h * 0.40,
        "MONTELIMAR"
    )

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(
        logo_x + logo_w / 2,
        logo_panel_y + logo_panel_h * 0.34,
        "INGENIO MONTELIMAR"
    )

    c.setStrokeColor(light_border)
    c.setLineWidth(0.55)
    c.line(10, 68, page_w - 10, 68)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(28, 48, "Nota:")

    c.setFont("Helvetica", 7.5)
    c.drawString(
        52, 48,
        "El área del campo se calcula directamente a partir de la geometría real del shapefile. "
        "La aplicación correcta corresponde a la superficie con una sola cobertura."
    )
    c.drawString(
        52, 34,
        "El traslape representa áreas con dos o más coberturas y está incluido dentro del área total cubierta."
    )

    c.showPage()
    c.save()
    out.seek(0)
    return out


st.title("🚁 Calidad de Aplicación Aérea AgNav")
st.write(
    "Sube el ZIP del plano y el ZIP del AgNav. El sistema calcula el área geométrica real, "
    "cobertura, traslape, sin aplicar y fuera de área."
)

c1,c2=st.columns(2)
with c1:
    fzip=st.file_uploader("Plano del campo (.ZIP)",type=["zip"])
with c2:
    azip=st.file_uploader("GPS AgNav (.ZIP)",type=["zip"])

name=st.text_input("Nombre del campo","Campo")
swath_default=st.number_input("Ancho de faja por defecto (m)",1.0,100.0,15.0,.5)
res=st.select_slider(
    "Resolución de cálculo del traslape (m)",
    options=[0.20,0.25,0.50,1.00],
    value=0.25
)
st.caption("Recomendado: 0.25 m. Este valor afecta el cálculo; el mapa final muestra el traslape como franjas continuas.")

if st.button("PROCESAR APLICACIÓN",type="primary",disabled=not(fzip and azip),use_container_width=True):
    try:
        with st.spinner("Procesando..."):
            with tempfile.TemporaryDirectory() as td:
                fr=Path(td)/"field"
                ar=Path(td)/"agnav"
                fr.mkdir(); ar.mkdir()
                unzip(fzip,fr); unzip(azip,ar)
                shp,gdf,field=load_field(fr)
                swath=swath_from_agn(ar,swath_default)
                passes,source,mode=get_passes(ar,gdf.crs)
                r=analyze(field,passes,swath,res)
                img=make_map(r,gdf.crs,name)
                pdf=make_pdf(r,img,name,swath,f"{source} ({mode})")

        st.success("Listo.")
        a,b,c,d=st.columns(4)
        a.metric("Área real",f'{r["field_ha"]:.2f} ha')
        b.metric("Área total cubierta",f'{r["inside_ha"]:.2f} ha',f'{r["inside_ha"]/r["field_ha"]*100:.2f}%')
        c.metric("Sin aplicar",f'{r["missing_ha"]:.2f} ha')
        d.metric("Traslape",f'{r["overlap_ha"]:.2f} ha')

        st.image(img,use_container_width=True)
        st.download_button(
            "Descargar PDF",
            pdf.getvalue(),
            file_name=f"Informe_{name}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(str(e))

st.caption(
    "Nota: el lector .t20 reproduce el formato observado en los archivos AgNav usados para desarrollar "
    "esta herramienta; otras versiones pueden requerir adaptar el decodificador."
)
