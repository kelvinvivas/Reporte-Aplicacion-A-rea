import streamlit as st
import io, re, struct, tempfile, zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import LineString
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_origin
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

st.set_page_config(page_title="Aplicación aérea AgNav", page_icon="🚁", layout="wide")

VERDE="#4CAF18"; AMARILLO="#FFD700"; ROJO="#E31A1C"; AZUL="#4169E1"

def unzip(upload, dest):
    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
        z.extractall(dest)

def files(root, ext):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower()==ext]

def load_field(root):
    shps = files(root, ".shp")
    if not shps: raise ValueError("No se encontró .shp en el ZIP del campo.")
    shp = next((p for p in shps if "spr" not in p.stem.lower()), shps[0])
    gdf = gpd.read_file(shp)
    if gdf.crs is None: raise ValueError("Falta el .prj del shapefile.")
    if not gdf.crs.is_projected: raise ValueError("El SHP debe estar en coordenadas proyectadas, por ejemplo UTM.")
    gdf = gdf[gdf.geom_type.isin(["Polygon","MultiPolygon"])]
    if gdf.empty: raise ValueError("El SHP no contiene polígonos.")
    return shp, gdf, unary_union(gdf.geometry)

def swath_from_agn(root, fallback):
    for p in files(root, ".agn"):
        try:
            for line in p.read_text(errors="ignore").splitlines():
                if re.match(r"^\s*36\s+", line):
                    v=float(line.split()[1])
                    if 1 <= v <= 100: return v
        except: pass
    return fallback

def passes_from_spron(root, crs):
    cands=[p for p in files(root,".shp") if "spron" in p.stem.lower() or "spray" in p.stem.lower()]
    if not cands: return None
    g=gpd.read_file(cands[0])
    if g.crs is None: g=g.set_crs(crs)
    else: g=g.to_crs(crs)
    pts=g[g.geom_type=="Point"].copy()
    if len(pts)<2: return None
    tcol=next((c for c in pts.columns if c.upper() in {"GPSTIME","TIME","UTC","SECONDS"}),None)
    if tcol: pts=pts.sort_values(tcol).reset_index(drop=True)
    lines=[]; cur=[pts.geometry.iloc[0]]
    for i in range(1,len(pts)):
        p=pts.geometry.iloc[i]; prev=pts.geometry.iloc[i-1]
        split=p.distance(prev)>60
        if tcol:
            try:
                dt=float(pts[tcol].iloc[i])-float(pts[tcol].iloc[i-1])
                split=split or dt>1.5 or dt<0
            except: pass
        if split:
            if len(cur)>=2: lines.append(LineString(cur))
            cur=[p]
        else: cur.append(p)
    if len(cur)>=2: lines.append(LineString(cur))
    return (lines,cands[0].name) if lines else None

def passes_from_t20(root):
    best=None
    for p in files(root,".t20"):
        raw=p.read_bytes(); rows=[]
        for i in range(len(raw)//68):
            r=raw[i*68:(i+1)*68]
            if r[:2]!=b"\xfb\xfb": continue
            x=struct.unpack_from("<i",r,28)[0]/10
            y=struct.unpack_from("<i",r,32)[0]/10
            s=r[39]
            if 10000<abs(x)<10000000 and 10000<abs(y)<20000000:
                rows.append((i,x,y,s))
        if rows:
            d=pd.DataFrame(rows,columns=["i","x","y","s"])
            n=int((d.s>0).sum())
            if best is None or n>best[0]: best=(n,p,d)
    if not best or best[0]<2: return None
    _,p,d=best; lines=[]; cur=[]; prev=None
    for r in d.itertuples(index=False):
        if r.s>0:
            if prev is None or r.i==prev+1: cur.append((r.x,r.y))
            else:
                if len(cur)>=2: lines.append(LineString(cur))
                cur=[(r.x,r.y)]
            prev=r.i
        else:
            if len(cur)>=2: lines.append(LineString(cur))
            cur=[]; prev=None
    if len(cur)>=2: lines.append(LineString(cur))
    return (lines,p.name) if lines else None

def get_passes(root, crs):
    a=passes_from_spron(root,crs)
    if a: return a[0],a[1],"SPRAY ON"
    a=passes_from_t20(root)
    if a: return a[0],a[1],"T20"
    raise ValueError("No encontré SPRAY ON compatible ni pude leer un .t20.")

def analyze(field, passes, swath, res):
    strips=[l.buffer(swath/2,cap_style=2,join_style=2) for l in passes]
    cov=unary_union(strips)
    inside=cov.intersection(field); outside=cov.difference(field); missing=field.difference(cov)
    minx,miny,maxx,maxy=field.bounds; pad=swath*2
    minx-=pad; miny-=pad; maxx+=pad; maxy+=pad
    W=int(np.ceil((maxx-minx)/res)); H=int(np.ceil((maxy-miny)/res))
    if W*H>16000000:
        res*=np.sqrt((W*H)/16000000); W=int(np.ceil((maxx-minx)/res)); H=int(np.ceil((maxy-miny)/res))
    tr=from_origin(minx,maxy,res,res)
    fm=rasterize([(field,1)],out_shape=(H,W),transform=tr,fill=0,dtype="uint8")
    cnt=np.zeros((H,W),dtype=np.uint16)
    for s in strips: cnt+=rasterize([(s,1)],out_shape=(H,W),transform=tr,fill=0,dtype="uint8")
    ov=(cnt>=2)&(fm==1); single=(cnt==1)&(fm==1); cell=res*res/10000
    return dict(field=field,passes=passes,inside=inside,outside=outside,missing=missing,
        field_ha=field.area/10000,inside_ha=inside.area/10000,outside_ha=outside.area/10000,
        missing_ha=missing.area/10000,overlap_ha=float(ov.sum()*cell),single_ha=float(single.sum()*cell),
        ov=ov,res=res,bounds=(minx,miny,maxx,maxy))

def make_map(r, crs, name):
    fig,ax=plt.subplots(figsize=(11,6))
    if not r["outside"].is_empty: gpd.GeoSeries([r["outside"]],crs=crs).plot(ax=ax,color=AZUL)
    if not r["inside"].is_empty: gpd.GeoSeries([r["inside"]],crs=crs).plot(ax=ax,color=VERDE)
    if not r["missing"].is_empty: gpd.GeoSeries([r["missing"]],crs=crs).plot(ax=ax,color=AMARILLO)
    gpd.GeoSeries([r["field"]],crs=crs).plot(ax=ax,facecolor="none",edgecolor="black",linewidth=1.3)
    ys,xs=np.where(r["ov"]); minx,miny,maxx,maxy=r["bounds"]
    if len(xs):
        step=max(1,len(xs)//25000)
        ax.scatter(minx+(xs[::step]+.5)*r["res"],maxy-(ys[::step]+.5)*r["res"],s=1.5,c=ROJO,marker="s")
    gpd.GeoSeries(r["passes"],crs=crs).plot(ax=ax,color="black",linewidth=.2,alpha=.25)
    bx=r["field"].bounds; pad=max(30,(bx[2]-bx[0])*.06)
    ax.set_xlim(bx[0]-pad,bx[2]+pad); ax.set_ylim(bx[1]-pad,bx[3]+pad); ax.set_aspect("equal")
    ax.set_title("Mapa de calidad de aplicación aérea - "+name,fontweight="bold")
    ax.legend(handles=[Patch(facecolor=VERDE,label="Aplicación correcta"),Patch(facecolor=AMARILLO,label="Sin aplicar"),
        Patch(facecolor=ROJO,label="Traslapado"),Patch(facecolor=AZUL,label="Fuera de área")])
    out=io.BytesIO(); fig.tight_layout(); fig.savefig(out,format="png",dpi=220,bbox_inches="tight"); plt.close(fig); out.seek(0)
    return out

def make_pdf(r, img, name, swath, source):
    out=io.BytesIO(); doc=SimpleDocTemplate(out,pagesize=landscape(A4))
    styles=getSampleStyleSheet()
    rows=[["Resultado","Área (ha)","% del campo"],
          ["Área real",f'{r["field_ha"]:.2f}',"100.00 %"],
          ["Aplicación correcta",f'{r["single_ha"]:.2f}',f'{r["single_ha"]/r["field_ha"]*100:.2f} %'],
          ["Sin aplicar",f'{r["missing_ha"]:.2f}',f'{r["missing_ha"]/r["field_ha"]*100:.2f} %'],
          ["Traslapado",f'{r["overlap_ha"]:.2f}',f'{r["overlap_ha"]/r["field_ha"]*100:.2f} %'],
          ["Fuera de área",f'{r["outside_ha"]:.2f}',f'{r["outside_ha"]/r["field_ha"]*100:.2f} %'],
          ["Cobertura dentro",f'{r["inside_ha"]:.2f}',f'{r["inside_ha"]/r["field_ha"]*100:.2f} %']]
    t=Table(rows,colWidths=[260,100,130]); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightblue),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
    story=[Paragraph("INFORME DE APLICACIÓN AÉREA - "+name.upper(),styles["Title"]),Spacer(1,8),t,Spacer(1,8),
           Paragraph(f"Área geométrica real. Ancho de faja: {swath:.1f} m. Fuente GPS: {source}.",styles["BodyText"]),
           Spacer(1,8),Image(img,width=600,height=330)]
    doc.build(story); out.seek(0); return out

st.title("🚁 Calidad de Aplicación Aérea AgNav")
st.write("Sube el ZIP del plano y el ZIP del AgNav. El sistema calcula el área geométrica real, cobertura, traslape, sin aplicar y fuera de área.")

c1,c2=st.columns(2)
with c1: fzip=st.file_uploader("Plano del campo (.ZIP)",type=["zip"])
with c2: azip=st.file_uploader("GPS AgNav (.ZIP)",type=["zip"])
name=st.text_input("Nombre del campo","Campo")
swath_default=st.number_input("Ancho de faja por defecto (m)",1.0,100.0,15.0,.5)
res=st.select_slider("Resolución del traslape (m)",options=[0.20,0.25,0.50,1.00],value=0.25)

if st.button("PROCESAR APLICACIÓN",type="primary",disabled=not(fzip and azip),use_container_width=True):
    try:
        with st.spinner("Procesando..."):
            with tempfile.TemporaryDirectory() as td:
                fr=Path(td)/"field"; ar=Path(td)/"agnav"; fr.mkdir(); ar.mkdir()
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
        b.metric("Cobertura",f'{r["inside_ha"]:.2f} ha',f'{r["inside_ha"]/r["field_ha"]*100:.2f}%')
        c.metric("Sin aplicar",f'{r["missing_ha"]:.2f} ha')
        d.metric("Traslape",f'{r["overlap_ha"]:.2f} ha')
        st.image(img,use_container_width=True)
        st.download_button("Descargar PDF",pdf.getvalue(),file_name=f"Informe_{name}.pdf",mime="application/pdf")
    except Exception as e:
        st.error(str(e))

st.caption("Nota: el lector .t20 reproduce el formato observado en los archivos AgNav usados para desarrollar esta herramienta; otras versiones pueden requerir adaptar el decodificador.")
