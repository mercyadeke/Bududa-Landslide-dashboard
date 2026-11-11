"""Bududa Landslide Risk & Early Warning Dashboard (Streamlit).

Run locally:   streamlit run app.py
Data files go in the ./data folder (see README.md).
"""
import base64
import io
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import streamlit as st
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from streamlit_folium import st_folium

import logic as L

st.set_page_config(page_title="Bududa Landslide Risk", page_icon="⛰️", layout="wide")

DATA = Path(__file__).parent / "data"
AMBER, BG, PANEL, BORDER, MUTED = "#F59E0B", "#0F141C", "#161D28", "#252E3B", "#9CA3AF"
ALERT_COLORS = {"NORMAL": "#9CA3AF", "WATCH": "#F59E0B", "WARNING": "#EF4444"}

st.markdown(f"""
<style>
.block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}}
.kpi {{background:{PANEL}; border:1px solid {BORDER}; border-radius:14px; padding:14px 16px; height:100%;}}
.kpi .lbl {{color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em;}}
.kpi .val {{font-size:1.75rem; font-weight:700; line-height:1.2; margin-top:4px;}}
.kpi .sub {{color:{MUTED}; font-size:0.8rem; margin-top:2px;}}
.pill {{display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700;
        border:1px solid; font-size:0.9rem;}}
h1, h2, h3 {{letter-spacing:-0.01em;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
def find(pattern: str, required: bool = True):
    hits = sorted(DATA.glob(pattern))
    if not hits and required:
        st.error(f"Missing data file matching **{pattern}** in the `data` folder. See README.md.")
        st.stop()
    return hits[0] if hits else None


@st.cache_data(show_spinner="Loading rasters…")
def load_rasters():
    with rasterio.open(find("*factor_classes*.tif")) as src:
        factors = src.read().astype("uint8")
        profile = {"crs": src.crs.to_string(), "transform": src.transform,
                   "height": src.height, "width": src.width}
    # population, resampled onto the factor grid if needed
    with rasterio.open(find("*population*.tif")) as src:
        pop = np.zeros((profile["height"], profile["width"]), dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=pop,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  resampling=Resampling.nearest)
        pop[~np.isfinite(pop)] = 0
        pop[pop < 0] = 0
    return factors, pop, profile


@st.cache_data(show_spinner="Loading sub-counties…")
def load_subcounties(crs: str, transform_tuple, shape):
    gdf = gpd.read_file(find("*exposure*.geojson"))
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    # Earth Engine sometimes exports mixed GeometryCollections or empty shapes,
    # which the web map cannot draw. Keep only the polygon parts.
    def polygons_only(g):
        if g is None or g.is_empty:
            return None
        if isinstance(g, (Polygon, MultiPolygon)):
            return g
        parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
        return unary_union(parts) if parts else None

    gdf["geometry"] = gdf.geometry.apply(polygons_only)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)          # fix any self-intersections
    gdf = gdf[~gdf.geometry.is_empty]
    name_col = "subcounty" if "subcounty" in gdf.columns else gdf.columns[0]
    gdf["name"] = gdf[name_col].astype(str).str.strip().str.upper()
    gdf = gdf.reset_index(drop=True)
    gdf["zid"] = np.arange(1, len(gdf) + 1)
    utm = gdf.to_crs(crs)
    transform = rasterio.Affine(*transform_tuple)
    zones = rasterize(((g, z) for g, z in zip(utm.geometry, utm.zid)),
                      out_shape=shape, transform=transform, fill=0, dtype="int32")
    cent = utm.geometry.representative_point().to_crs(4326)
    gdf["lat"], gdf["lon"] = cent.y.values, cent.x.values
    return gdf[["name", "zid", "lat", "lon", "geometry"]], zones


@st.cache_data
def load_tables():
    rain = pd.read_csv(find("*rain_rolling_alerts*.csv"), parse_dates=["date"])
    events = pd.read_csv(find("*landslide_events*.csv"))
    d = events["date"].astype(str)
    events["date_parsed"] = pd.to_datetime(d.str[:10], format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(d.str[:7], format="%Y-%m", errors="coerce"))
    events["key"] = events["subcounty"].map(L.subcounty_key)
    bpath = find("*buildings*.csv", required=False)
    buildings = None
    if bpath is not None:
        buildings = pd.read_csv(bpath)
        buildings["name"] = buildings["subcounty"].astype(str).str.strip().str.upper()
    return rain, events, buildings


factors, pop, prof = load_rasters()
transform = prof["transform"]
shape = (prof["height"], prof["width"])
pixel_m = abs(transform.a)
pixel_km2 = pixel_m * abs(transform.e) / 1e6
subs, zones = load_subcounties(prof["crs"], tuple(transform)[:6], shape)
rain, events, buildings = load_tables()

to_ll = Transformer.from_crs(prof["crs"], "EPSG:4326", always_xy=True)
to_utm = Transformer.from_crs("EPSG:4326", prof["crs"], always_xy=True)
left, top = transform.c, transform.f
right, bottom = left + transform.a * shape[1], top + transform.e * shape[0]
lon_w, lat_s = to_ll.transform(left, bottom)
lon_e, lat_n = to_ll.transform(right, top)
BOUNDS = [[lat_s, lon_w], [lat_n, lon_e]]


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Controls")

st.sidebar.subheader("1 · Rainfall scenario")
r30 = st.sidebar.slider("Rain in the past 30 days (mm)", 0, 450, 150, 5,
                        help="Soil saturation. Landslide days 2010–2024: 113–281 mm.")
r3 = st.sidebar.slider("Rain in the past 3 days (mm)", 0, 120, 20, 1,
                       help="Recent burst. Landslide days 2010–2024: 12–44 mm.")
alert = L.alert_level(r30, r3)

st.sidebar.subheader("2 · Who to warn")
cutoff_label = st.sidebar.selectbox(
    "Zones flagged for action", list(L.CUTOFF_OPTIONS),
    help="Auto: Watch → Very high zones; Warning → High + Very high zones.")
min_class = L.flagged_min_class(cutoff_label, alert)

st.sidebar.subheader("3 · Model weights")
st.sidebar.caption("How much each factor counts. Re-scaled to add up to 100%.")
weights = {}
for f in L.FACTORS:
    weights[f] = st.sidebar.slider(L.FACTOR_LABELS[f], 0, 50,
                                   int(round(L.DEFAULT_WEIGHTS[f] * 100)), 5, key=f"w_{f}") / 100
wn = L.normalise_weights(weights)

st.sidebar.subheader("4 · Spatial query")
buffer_m = st.sidebar.slider("Buffer radius (m)", 100, 2000, 500, 100,
                             help="Click the map to place the buffer centre.")
sub_names = ["(whole district)"] + sorted(subs["name"].unique().tolist())
sel_sub = st.sidebar.selectbox("Focus sub-county", sub_names)

st.sidebar.subheader("5 · Events shown")
yr = st.sidebar.slider("Years", 2010, 2024, (2010, 2024))


# --------------------------------------------------------------------------- compute
@st.cache_data(show_spinner="Re-computing susceptibility…")
def compute(weights_tuple):
    w = dict(zip(L.FACTORS, weights_tuple))
    score = L.weighted_score(factors, w, smooth_radius=2)
    cls, breaks = L.classify(score)
    return score, cls, breaks


score, cls, breaks = compute(tuple(wn[f] for f in L.FACTORS))
expo = L.exposure_by_zone(cls, pop, zones, dict(zip(subs.zid, subs.name)), min_class, pixel_km2)
inside = cls > 0
flag_mask = (cls >= min_class) if min_class else np.zeros(shape, bool)
people_total = float(pop[inside].sum())
people_flag = float(pop[inside & flag_mask].sum())
area_flag_pct = 100 * flag_mask[inside].mean() if inside.any() else 0.0

if sel_sub != "(whole district)":
    focus = expo[expo.subcounty == sel_sub]
else:
    focus = None


# --------------------------------------------------------------------------- header
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("## ⛰️ Bududa Landslide Risk & Early Warning")
    st.caption("Mt. Elgon slopes, Eastern Uganda · Where slopes may fail × when rain is dangerous × who lives there")
with c2:
    col = ALERT_COLORS[alert]
    st.markdown(f"<div style='text-align:right;margin-top:18px'><span class='pill' "
                f"style='color:{col};border-color:{col};background:{col}1A'>● {alert}</span></div>",
                unsafe_allow_html=True)


def kpi(col, label, value, sub, color="#F5F5F4"):
    col.markdown(f"<div class='kpi'><div class='lbl'>{label}</div>"
                 f"<div class='val' style='color:{color}'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


flag_txt = ("No zones flagged" if not min_class
            else f"{L.CLASS_NAMES[min_class]}{' and above' if min_class < 5 else ''} zones")
k = st.columns(5)
kpi(k[0], "Alert level", alert,
    f"Watch ≥{L.WATCH['r30']:.0f} mm/30d & ≥{L.WATCH['r3']:.0f} mm/3d", ALERT_COLORS[alert])
kpi(k[1], "Rainfall scenario", f"{r30} mm", f"30 days · {r3} mm in last 3 days")
kpi(k[2], "People to warn", f"{people_flag:,.0f}", f"{flag_txt} · of {people_total:,.0f}")
kpi(k[3], "Land flagged", f"{area_flag_pct:.0f}%", "Share of district area")
if buildings is not None and "bld_at_risk" in buildings:
    kpi(k[4], "Buildings at risk", f"{int(buildings.bld_at_risk.sum()):,}",
        f"High + Very high (default model) · of {int(buildings.bld_total.sum()):,}")
else:
    kpi(k[4], "Past landslides", f"{len(events)}", "Verified events 2010–2024")

st.write("")

# --------------------------------------------------------------------------- map
def class_png(cls_arr, flag=None):
    rgba = np.zeros(cls_arr.shape + (4,), dtype="uint8")
    for c, hexc in L.CLASS_COLORS.items():
        m = cls_arr == c
        rgba[m, :3] = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
        rgba[m, 3] = 200
    if flag is not None:
        rgba[flag, :3] = [239, 68, 68]
        rgba[flag, 3] = 150
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pop_png(p):
    v = np.clip(p / np.nanpercentile(p[p > 0], 98), 0, 1) if (p > 0).any() else p
    rgba = np.zeros(p.shape + (4,), dtype="uint8")
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 124, 58, 237
    rgba[..., 3] = (v * 220).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


map_col, side_col = st.columns([2.2, 1])

with map_col:
    st.markdown("#### Landslide susceptibility map")
    centre = [(lat_s + lat_n) / 2, (lon_w + lon_e) / 2]
    m = folium.Map(location=centre, zoom_start=12, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)
    folium.raster_layers.ImageOverlay(class_png(cls), bounds=BOUNDS, name="Susceptibility (5 classes)",
                                      opacity=0.85).add_to(m)
    if min_class:
        folium.raster_layers.ImageOverlay(class_png(np.zeros_like(cls), flag_mask), bounds=BOUNDS,
                                          name="Zones flagged for action", opacity=0.6).add_to(m)
    folium.raster_layers.ImageOverlay(pop_png(pop), bounds=BOUNDS, name="Population (WorldPop)",
                                      opacity=0.8, show=False).add_to(m)

    folium.GeoJson(
        subs[["name", "geometry"]], name="Sub-counties",
        style_function=lambda f: {
            "color": "#111827", "weight": 3 if f["properties"]["name"] == sel_sub else 0.8,
            "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(["name"], aliases=["Sub-county"]),
    ).add_to(m)

    # past landslides, placed at sub-county centres (approximate)
    ev = events[(events.date_parsed.dt.year >= yr[0]) & (events.date_parsed.dt.year <= yr[1])]
    look = subs.drop_duplicates("name").set_index("name")
    for _, e in ev.iterrows():
        if e.key in look.index:
            s = look.loc[e.key]
            folium.CircleMarker(
                [s.lat, s.lon], radius=7, color="#111827", weight=1.5, fill=True,
                fill_color="#EF4444", fill_opacity=0.9,
                tooltip=f"{e.date} · {e.subcounty} · deaths: {e.deaths_reported}").add_to(m)

    # buffer query centre: last click, else selected sub-county, else district centre
    click = st.session_state.get("click")
    if click:
        bc = (click["lat"], click["lng"])
    elif sel_sub != "(whole district)":
        s = look.loc[sel_sub]
        bc = (s.lat, s.lon)
    else:
        bc = tuple(centre)
    folium.Circle(bc, radius=buffer_m, color=AMBER, weight=2, fill=True, fill_opacity=0.1,
                  tooltip=f"Buffer {buffer_m} m").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked") and out["last_clicked"] != st.session_state.get("click"):
        st.session_state["click"] = out["last_clicked"]
        st.rerun()

    legend = " ".join(f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
                      f"border-radius:2px;margin:0 4px 0 10px'></span>{L.CLASS_NAMES[k]}"
                      for k, c in L.CLASS_COLORS.items())
    st.markdown(f"<div style='font-size:0.8rem;color:{MUTED}'>{legend}"
                f"<span style='display:inline-block;width:12px;height:12px;background:#EF4444;"
                f"border-radius:50%;margin:0 4px 0 14px'></span>Past landslide (sub-county centre)</div>",
                unsafe_allow_html=True)

with side_col:
    # spatial query result
    x, y = to_utm.transform(bc[1], bc[0])
    col_ = (x - transform.c) / transform.a
    row_ = (y - transform.f) / transform.e
    ppl = L.people_in_buffer(pop, (row_, col_), buffer_m, pixel_m)
    st.markdown("#### Spatial query")
    kpi(st, f"People within {buffer_m} m", f"{ppl:,.0f}",
        f"Centre {bc[0]:.4f}, {bc[1]:.4f} · click the map to move it", AMBER)
    st.write("")

    st.markdown("#### Who to warn" + (f" · {sel_sub.title()}" if focus is not None else ""))
    if focus is not None and len(focus):
        f0 = focus.iloc[0]
        kpi(st, "People in flagged zones", f"{f0.people_flagged:,}",
            f"{f0.pct_people_flagged}% of {f0.people_total:,} · {f0.pct_area_flagged}% of land")
        if buildings is not None:
            b = buildings[buildings.name == sel_sub]
            if len(b):
                st.caption(f"Buildings in High/Very high zones (default model): "
                           f"{int(b.bld_at_risk.iloc[0]):,} of {int(b.bld_total.iloc[0]):,}")
    if not min_class:
        st.info("Alert is NORMAL – no zones flagged. Raise the rainfall or pick a cut-off to see exposure.")
    elif len(expo):
        top = expo.head(12).iloc[::-1]
        fig = px.bar(top, x="people_flagged", y="subcounty", orientation="h",
                     color_discrete_sequence=[AMBER],
                     labels={"people_flagged": "People in flagged zones", "subcounty": ""})
        fig.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- bottom row
b1, b2 = st.columns(2)

with b1:
    st.markdown("#### Your scenario vs 15 years of rainfall")
    r = rain.dropna(subset=["rain_30d", "rain_3d"])
    ls = r[r.landslide == 1]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=r.rain_30d, y=r.rain_3d, mode="markers", name="All days 2010–2024",
                               marker=dict(size=3, color="rgba(156,163,175,0.35)")))
    fig.add_trace(go.Scatter(x=ls.rain_30d, y=ls.rain_3d, mode="markers", name="Landslide days",
                             marker=dict(size=11, color=AMBER, line=dict(width=1, color="#111827")),
                             text=ls.date.dt.strftime("%b %Y"), hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r30], y=[r3], mode="markers", name="Your scenario",
                             marker=dict(size=18, symbol="star", color=ALERT_COLORS[alert],
                                         line=dict(width=1.5, color="#111827"))))
    for t, c in ((L.WATCH, "#B45309"), (L.WARNING, "#991B1B")):
        fig.add_vline(x=t["r30"], line_dash="dash", line_color=c, line_width=1)
        fig.add_hline(y=t["r3"], line_dash="dash", line_color=c, line_width=1)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Rain in previous 30 days (mm)", yaxis_title="Rain in previous 3 days (mm)",
                      legend=dict(orientation="h", y=1.08), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    hist = r.alert.astype(str).str.upper().value_counts(normalize=True) \
        .reindex(["NORMAL", "WATCH", "WARNING"]).fillna(0) * 100
    st.caption(f"Rule applied to 2010–2024: Watch on {hist['WATCH']:.0f}% of days, Warning on "
               f"{hist['WARNING']:.0f}% · caught 9 of 10 landslide days (3 at Warning).")

with b2:
    st.markdown("#### Landslide timeline")
    ev = events[(events.date_parsed.dt.year >= yr[0]) & (events.date_parsed.dt.year <= yr[1])].copy()
    ev["deaths_high"] = pd.to_numeric(ev.get("deaths_high"), errors="coerce").fillna(0)
    ev["size"] = 8 + np.sqrt(ev.deaths_high) * 3
    if ev.empty:
        st.info("No events in the selected years.")
    fig = px.scatter(ev, x="date_parsed", y=[1] * len(ev), size="size", size_max=40,
                     color_discrete_sequence=["#EF4444"],
                     hover_data={"date": True, "subcounty": True, "deaths_reported": True,
                                 "date_parsed": False, "size": False})
    fig.update_yaxes(visible=False)
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ev[["date", "subcounty", "deaths_reported"]].rename(
        columns={"date": "Date", "subcounty": "Sub-county", "deaths_reported": "Deaths reported"}),
        hide_index=True, use_container_width=True, height=180)


# --------------------------------------------------------------------------- sources
with st.expander("📚 Data sources, parameters and assumptions", expanded=False):
    w_txt = ", ".join(f"{L.FACTOR_LABELS[f]} {wn[f]*100:.0f}%" for f in L.FACTORS)
    st.markdown(f"""
**Data sources**
- **Terrain:** Copernicus DEM GLO-30 (30 m) → slope, aspect, curvature
- **Hydrology:** MERIT Hydro (90 m) → wetness index (TWI), distance to streams (>1 km² catchment)
- **Land cover:** ESA WorldCover 2021 (10 m)
- **Rainfall:** CHIRPS daily (0.05° ≈ 5.5 km), district mean, 2010–2024
- **Population:** WorldPop 2020 (100 m), redistributed to 30 m
- **Buildings:** Google Open Buildings v3 (May 2023, confidence ≥ 0.75)
- **Boundaries:** UBOS 2024 district and sub-county boundaries
- **Landslide events:** 12 verified events 2010–2024 from news and humanitarian reports (ReliefWeb, IFRC, Daily Monitor, New Vision, Uganda Radio Network)

**Model parameters (current)**
- Weighted overlay: {w_txt}
- Each factor scored 1–5 using breaks from Bududa studies (Knapen et al. 2006; Kitutu 2010): 14–41° slopes, concave hollows, NE aspect, cropland
- Score smoothed over ~60 m; classes = percentiles within Bududa (20/40/70/90) → *relative* susceptibility
- Alert rule: **Watch** = 30-day ≥ {L.WATCH['r30']:.0f} mm and 3-day ≥ {L.WATCH['r3']:.0f} mm;
  **Warning** = 30-day ≥ {L.WARNING['r30']:.0f} mm and 3-day ≥ {L.WARNING['r3']:.0f} mm

**Assumptions & limitations**
- Shows *conditions* under which landslides are likely, not the date of the next landslide
- Only 10 dated events → thresholds are indicative, not calibrated probabilities
- CHIRPS smooths short, intense storms; one district-wide rainfall value is used
- Past landslides are shown at sub-county centres, not exact scar locations
- WorldPop 2020 population; building counts use the default weights
- Not an official warning system
""")

st.caption("GLM Geospatial Visual Analytics · Group project · Built with Streamlit, Google Earth Engine exports, folium and Plotly")
