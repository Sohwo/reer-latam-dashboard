# app.py
# Streamlit app: REER + Comercio LATAM (mapa, filtros, comparador, KPIs, diseño oscuro)
# Requisitos: streamlit, pandas, plotly, numpy
# Ejecutar: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from io import StringIO

st.set_page_config(page_title="REER & Comercio LATAM", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper: carga CSV robusta
# ---------------------------
@st.cache_data
def load_data(path="unified_for_powerbi_standard.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré el archivo '{path}'. Colócalo en la misma carpeta o actualiza la ruta.")
    # intenta con coma, si falla intenta con ; y decimal coma
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=';')
        except Exception as e:
            raise e
    # columnas esperadas: Year, Country, REER, Nominal, Exports, Imports
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # normalizar nombres
    if 'Year' not in df.columns:
        # intentar detectar una columna que sea año
        for c in df.columns:
            if c.lower() in ['year','año','date','fecha']:
                df = df.rename(columns={c:'Year'})
                break
    if 'Country' not in df.columns:
        for c in df.columns:
            if c.lower() in ['country','país','pais','pais_name']:
                df = df.rename(columns={c:'Country'})
                break
    # coerción numérica
    for col in ['REER','Nominal','Exports','Imports']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.'), errors='coerce')
    # Year int
    try:
        df['Year'] = df['Year'].astype(int)
    except:
        pass
    return df

# ---------------------------
# Pequeña tabla con ISO3 para LATAM (para el mapa)
# ---------------------------
ISO3 = {
    "Argentina":"ARG","Bolivia":"BOL","Brasil":"BRA","Chile":"CHL","Colombia":"COL",
    "Costa Rica":"CRI","Cuba":"CUB","Rep. Dominicana":"DOM","República Dominicana":"DOM","Dominican Republic":"DOM",
    "Ecuador":"ECU","El Salvador":"SLV","Guatemala":"GTM","Honduras":"HND","México":"MEX",
    "Nicaragua":"NIC","Panamá":"PAN","Panama":"PAN","Paraguay":"PRY","Perú":"PER","Peru":"PER",
    "Uruguay":"URY","Venezuela":"VEN"
}

def country_to_iso3(name):
    if pd.isna(name): return None
    s = str(name).strip()
    return ISO3.get(s, None)

# ---------------------------
# Load
# ---------------------------
st.markdown("<style>body{background-color:#0e1117; color:#e6eef8;} .stApp { background-color: #0e1117; }</style>", unsafe_allow_html=True)
st.markdown("<style> .big-title { font-size:28px !important; font-weight:700; color:#ffffff;} </style>", unsafe_allow_html=True)
st.markdown('<div class="big-title">📊 REER y Comercio Exterior — América Latina (2020-2025)</div>', unsafe_allow_html=True)
st.write("Fuente: World Bank WDI (REER, TC nominal, exportaciones/importaciones). Interactivo para explorar países, comparar y ver el ICC (Índice de Competitividad Cambiaria).", unsafe_allow_html=True)

try:
    df = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# filtro rango de años disponible
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())

# ---------------------------
# Sidebar: controls
# ---------------------------
with st.sidebar:
    st.header("Filtros")
    yr_range = st.slider("Rango de años", min_year, max_year, (2020, max_year), step=1)
    year_single = st.select_slider("Año focal (para mapa/ KPI)", options=list(range(min_year, max_year+1)), value=max_year)
    countries = sorted(df['Country'].dropna().unique())
    country_sel = st.selectbox("Selecciona un país", ["(ninguno)"] + countries, index=0)
    country_a = st.selectbox("Comparador: país A", countries, index=0)
    country_b = st.selectbox("Comparador: país B", countries, index=1 if len(countries)>1 else 0)
    st.markdown("---")
    theme_mode = st.selectbox("Paleta / tema", ["dark (recomendado)","light"], index=0)
    st.write("Tip: usa el comparador para mostrar lado a lado dos países en la exposición.")
    st.markdown("---")
    if st.button("Descargar métricas (CSV)"):
        # creamos métricas por país en el rango seleccionado
        df_metrics = compute_metrics(df, yr_range) if 'compute_metrics' in globals() else None
        if df_metrics is not None:
            csv = df_metrics.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV métricas", data=csv, file_name=f"metrics_{yr_range[0]}_{yr_range[1]}.csv")

# ---------------------------
# Data wrangling: rango seleccionado
# ---------------------------
yr_min, yr_max = yr_range
df_range = df[(df['Year'] >= yr_min) & (df['Year'] <= yr_max)].copy()

# compute % changes and ICC grouped by country/year
def compute_derived(df_input):
    dfc = df_input.copy().sort_values(['Country','Year'])
    dfc['VarExportsPct'] = dfc.groupby('Country')['Exports'].pct_change() * 100
    dfc['VarREERPct'] = dfc.groupby('Country')['REER'].pct_change() * 100
    dfc['ICC'] = dfc['VarExportsPct'] - dfc['VarREERPct']
    return dfc

dfd = compute_derived(df_range)

# metrics per country aggregated (mean ICC, mean REER change, total exports/imports in range)
metrics = dfd.groupby('Country').agg({
    'ICC':'mean',
    'VarExportsPct':'mean',
    'VarREERPct':'mean',
    'Exports':'sum',
    'Imports':'sum',
    'REER':'mean'
}).reset_index()

# add iso3
metrics['iso3'] = metrics['Country'].apply(country_to_iso3)

# ---------------------------
# Top KPIs (for selected country or region)
# ---------------------------
def kpi_for_country(country, year_focus):
    if country not in countries:
        return None
    s = df[(df['Country']==country) & (df['Year']<=year_focus)].sort_values('Year')
    if s.empty: return None
    last = s[s['Year']==s['Year'].max()].iloc[0]
    # previous year if exists
    prev_year = s[s['Year']==(last['Year']-1)]
    if not prev_year.empty:
        prev = prev_year.iloc[0]
        exports_ch = (last['Exports'] - prev['Exports']) / prev['Exports'] * 100 if prev['Exports'] and not np.isnan(prev['Exports']) else None
        reer_ch = (last['REER'] - prev['REER']) / prev['REER'] * 100 if prev['REER'] and not np.isnan(prev['REER']) else None
    else:
        exports_ch = None; reer_ch = None
    return {
        'year': int(last['Year']),
        'REER': last.get('REER', np.nan),
        'Exports': last.get('Exports', np.nan),
        'Imports': last.get('Imports', np.nan),
        'exports_%chg_lastyear': exports_ch,
        'reer_%chg_lastyear': reer_ch
    }

st.markdown("### KPIs")
kpi_cols = st.columns([1.2,1.2,1.2,1.2])
if country_sel != "(ninguno)":
    k = kpi_for_country(country_sel, year_single)
    if k:
        kpi_cols[0].metric(f"{country_sel} — REER ({k['year']})", f"{k['REER']:.1f}", delta=f"{k['reer_%chg_lastyear']:.1f}%" if k['reer_%chg_lastyear'] is not None else "")
        kpi_cols[1].metric(f"{country_sel} — Exports ({k['year']})", f"${k['Exports']/1e9:,.2f} B", delta=f"{k['exports_%chg_lastyear']:.1f}%" if k['exports_%chg_lastyear'] is not None else "")
        kpi_cols[2].metric(f"{country_sel} — Imports ({k['year']})", f"${k['Imports']/1e9:,.2f} B")
        # simple trade balance last year
        trade_bal = k['Exports'] - k['Imports'] if (k['Exports'] is not None and k['Imports'] is not None) else None
        kpi_cols[3].metric("Trade Balance", f"${trade_bal/1e9:,.2f} B" if trade_bal is not None else "N/A")
    else:
        for c in kpi_cols: c.write("No hay datos para ese país/año")
else:
    # regional KPIs aggregated for the selected year range
    region_exports = dfd['Exports'].sum()
    region_imports = dfd['Imports'].sum()
    region_reer_mean = dfd['REER'].mean()
    kpi_cols[0].metric("Región — Exports (sum)", f"${region_exports/1e9:,.2f} B")
    kpi_cols[1].metric("Región — Imports (sum)", f"${region_imports/1e9:,.2f} B")
    kpi_cols[2].metric("Región — REER (promedio)", f"{region_reer_mean:.1f}")
    kpi_cols[3].metric("Periodo", f"{yr_min}–{yr_max}")

st.markdown("---")

# ---------------------------
# MAP: mostrar ICC promedio o cambio REER en el año_single
# ---------------------------
st.markdown("### Mapa: Indicador por país (hover muestra métricas)")
map_metric = st.selectbox("Metric to color map", options=["ICC (promedio en rango)","%Δ REER (promedio)","Exports (sum)"], index=0)

if map_metric == "ICC (promedio en rango)":
    map_df = metrics.copy()
    map_df['color'] = map_df['ICC']
    color_title = "ICC (avg)"
    color_scale = "RdYlGn"
elif map_metric == "%Δ REER (promedio)":
    map_df = metrics.copy()
    map_df['color'] = map_df['VarREERPct']
    color_title = "%Δ REER (avg)"
    color_scale = "Viridis"
else:
    map_df = metrics.copy()
    map_df['color'] = map_df['Exports']
    color_title = "Exports (sum)"
    color_scale = "Blues"

# For map, require iso3
map_df['iso3'] = map_df['iso3'].fillna(map_df['Country'].map(country_to_iso3))
map_df = map_df.dropna(subset=['iso3'])
fig_map = px.choropleth(map_df, locations='iso3', color='color',
                        hover_name='Country',
                        hover_data={'color':True,'Exports':True,'Imports':True,'ICC':True},
                        color_continuous_scale=color_scale,
                        labels={'color':color_title},
                        projection='natural earth',
                        title=f"Mapa: {color_title} ({yr_min}-{yr_max})",
                        template='plotly_dark')
fig_map.update_layout(height=500, margin=dict(t=50,l=0,r=0,b=0))
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# Scatter regional: REER %chg vs Exports %chg
# ---------------------------
st.markdown("### Scatter: %Δ REER vs %Δ Exports (puntos por país-año)")
scatter_df = dfd.dropna(subset=['VarREERPct','VarExportsPct']).copy()
scatter_df['label'] = scatter_df['Country'] + " " + scatter_df['Year'].astype(str)
fig_sc = px.scatter(scatter_df, x='VarREERPct', y='VarExportsPct', color='Country',
                   hover_data=['Country','Year','Exports','Imports'],
                   title=f"%Δ REER vs %Δ Exports ({yr_min}-{yr_max})",
                   template='plotly_dark', size_max=12)
fig_sc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.6)
fig_sc.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.6)
st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("---")

# ---------------------------
# Comparador lado a lado
# ---------------------------
st.markdown("### Comparador: País A vs País B")
col_a, col_b = st.columns(2)

def plot_country_timeseries(country, df_all, ycols=['REER','Exports','Imports']):
    s = df_all[df_all['Country']==country].sort_values('Year')
    if s.empty:
        return None
    figs = {}
    # REER
    figs['reer'] = px.line(s, x='Year', y='REER', title=f"REER — {country}", template='plotly_dark')
    figs['exports'] = px.line(s, x='Year', y='Exports', title=f"Exports — {country}", template='plotly_dark')
    figs['imports'] = px.line(s, x='Year', y='Imports', title=f"Imports — {country}", template='plotly_dark')
    return figs

with col_a:
    st.subheader(country_a)
    figs_a = plot_country_timeseries(country_a, df)
    if figs_a:
        st.plotly_chart(figs_a['reer'], use_container_width=True)
        st.plotly_chart(figs_a['exports'], use_container_width=True)
    else:
        st.write("No hay datos A")

with col_b:
    st.subheader(country_b)
    figs_b = plot_country_timeseries(country_b, df)
    if figs_b:
        st.plotly_chart(figs_b['reer'], use_container_width=True)
        st.plotly_chart(figs_b['exports'], use_container_width=True)
    else:
        st.write("No hay datos B")

st.markdown("---")

# ---------------------------
# Ranking ICC table y descarga
# ---------------------------
st.markdown("### Ranking ICC (promedio en el rango seleccionado)")
ranking_table = metrics[['Country','ICC','VarExportsPct','VarREERPct','Exports','Imports','iso3']].sort_values('ICC', ascending=False)
ranking_table_display = ranking_table.copy()
ranking_table_display['Exports'] = ranking_table_display['Exports'].apply(lambda x: f"${x/1e9:,.2f} B" if pd.notna(x) else "")
ranking_table_display['Imports'] = ranking_table_display['Imports'].apply(lambda x: f"${x/1e9:,.2f} B" if pd.notna(x) else "")
st.dataframe(ranking_table_display.style.format({"ICC":"{:.2f}","VarExportsPct":"{:.2f}","VarREERPct":"{:.2f}"}), use_container_width=True)

# download raw metrics csv
csv_metrics = ranking_table.to_csv(index=False).encode('utf-8')
st.download_button("Descargar ranking ICC (CSV)", data=csv_metrics, file_name=f"ranking_ICC_{yr_min}_{yr_max}.csv")

st.markdown("""---\n**Notas:** ICC = %Δ Exports − %Δ REER.  
- ICC positivo → exportaciones crecieron más que el tipo de cambio real se apreció / o se depreciaron y exportaciones crecieron aún más.  
- ICC negativo → señales de pérdida neta de competitividad relativa.\n
Usa el comparador y el mapa para explorar casos. Para la exposición, muestra 2 países contrastantes (ej. país apreciado vs país depreciado) y comenta ICC.""", unsafe_allow_html=True)

# ---------------------------
# Footer / tips
# ---------------------------
st.markdown("### Tips para la presentación")
st.markdown("- Muestra el mapa, después filtra a 2 países con ICC opuesto y usa el comparador.") 
st.markdown("- Exporta pantallazos o graba la demo en vivo para 1–2 minutos.") 
st.markdown("- Si quieres, puedo generar una versión con diseño personalizado (logos, colores corporativos).")

# end