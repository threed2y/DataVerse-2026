"""
Urban Heat Island — Full Analysis Dashboard
Incorporates: step1.ipynb, step2.ipynb, step3.ipynb,
              UrbanSurface.ipynb, Auto_Correlation.ipynb,
              Times_series_modal.ipynb, Tempreture_Health.ipynb

Run: streamlit run urban_heat_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UHI Analysis Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d0f1a; color: #dde1f0; }
  section[data-testid="stSidebar"] { background: #11131e; }

  .hero-title {
      font-size: 2.5rem; font-weight: 900; letter-spacing: 1px;
      background: linear-gradient(90deg,#ff6b35,#f7c59f,#a8edea);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      text-align: center; margin-bottom: .2rem;
  }
  .hero-sub { text-align:center; color:#7a82a0; font-size:.9rem; margin-bottom:1.5rem; }

  .step-card {
      background: linear-gradient(135deg,#161929 0%,#1e2235 100%);
      border-left: 4px solid #ff6b35; border-radius: 10px;
      padding: 1rem 1.3rem; margin-bottom: .8rem;
  }
  .step-badge {
      display:inline-block; background:#ff6b35; color:#fff;
      font-size:.7rem; font-weight:700; letter-spacing:2px;
      padding:2px 10px; border-radius:20px; margin-bottom:.4rem;
  }
  .step-title { font-size:1.05rem; font-weight:700; color:#fff; }
  .step-desc  { font-size:.82rem; color:#9aa3c0; margin-top:.25rem; }

  .sec-hdr {
      font-size:1.2rem; font-weight:700; color:#ff6b35;
      border-bottom:2px solid #252840; padding-bottom:.35rem; margin-bottom:.9rem;
  }

  .kpi-wrap { display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1rem; }
  .kpi-box {
      flex:1; min-width:130px;
      background:#161929; border-radius:10px; padding:.9rem 1rem;
      border-top:3px solid #ff6b35; text-align:center;
  }
  .kpi-val { font-size:1.5rem; font-weight:800; color:#ff6b35; }
  .kpi-lbl { font-size:.72rem; color:#9aa3c0; margin-top:.15rem; }

  .info-box {
      background:#1a1d2e; border-left:4px solid #a8edea;
      border-radius:8px; padding:.8rem 1rem; margin:.5rem 0;
      font-size:.85rem; color:#c0cce0;
  }
  .warn-box {
      background:#1a1d2e; border-left:4px solid #ff6b35;
      border-radius:8px; padding:.8rem 1rem; margin:.5rem 0;
      font-size:.85rem; color:#c0cce0;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Colour palette helpers
# ──────────────────────────────────────────────────────────────
DARK_BG   = "#0d0f1a"
CARD_BG   = "#161929"
PLOT_BG   = "#1e2235"
ACCENT    = "#ff6b35"
FONT_COL  = "#dde1f0"

def dark_layout(fig, height=400, title=None):
    kw = dict(paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
              font_color=FONT_COL, height=height,
              legend=dict(bgcolor="rgba(0,0,0,0)"),
              margin=dict(l=10, r=10, t=40 if title else 20, b=10))
    if title:
        kw["title"] = title
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor="#252840", zerolinecolor="#252840")
    fig.update_yaxes(gridcolor="#252840", zerolinecolor="#252840")
    return fig


# ──────────────────────────────────────────────────────────────
# DATA LOADING  (real CSVs → synthetic fallback)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_all():
    """Returns (urban_df, temp_df, health_df, air_df, is_synthetic)."""
    try:
        urban_df  = pd.read_csv("urban_surface.csv")
        temp_df   = pd.read_csv("temperature_clean.csv", parse_dates=["date"])
        health_df = pd.read_csv("health_clean.csv",      parse_dates=["date"])
        air_df    = pd.read_csv("air_quality_clean.csv", parse_dates=["date"])
        return urban_df, temp_df, health_df, air_df, False
    except FileNotFoundError:
        return _synthetic()


def _synthetic():
    np.random.seed(42)
    n = 500
    lat = np.random.uniform(22.05, 22.62, n)
    lon = np.random.uniform(72.97, 73.40, n)
    clat, clon = 22.307, 73.183
    dist = np.sqrt((lat - clat)**2 + (lon - clon)**2) * 111

    urban_df = pd.DataFrame({
        "neighbourhood_id":            range(1, n+1),
        "latitude": lat, "longitude": lon,
        "distance_from_center_km":     dist,
        "tree_cover_pct":              np.clip(np.random.normal(29, 16, n), 5, 70),
        "asphalt_pct":                 np.clip(np.random.normal(76, 15, n), 33, 90),
        "building_density":            np.clip(np.random.normal(0.455, 0.17, n), 0.3, 0.94),
        "median_income":               np.random.normal(856000, 340000, n),
        "population_density":          np.random.uniform(2000, 22000, n),
        "heat_retention_factor":       np.clip(np.random.normal(1.0, 0.116, n), 0.8, 1.2),
        "infrastructure_quality_index":np.clip(np.random.normal(1.0, 0.287, n), 0.5, 1.5),
        "social_vulnerability_index":  np.clip(np.random.normal(0.988, 0.406, n), 0.3, 1.7),
    })
    urban_df["urban_heat_index"] = (
        urban_df["asphalt_pct"] * 0.055
        - urban_df["tree_cover_pct"] * 0.038
        + urban_df["heat_retention_factor"] * 1.4
        + np.random.normal(0, 0.3, n)
    )
    # LISA clusters
    clusters = []
    for i in range(n):
        if lat[i] > 22.42:
            clusters.append(np.random.choice(["High-High","Not Significant"], p=[0.7,0.3]))
        elif lat[i] < 22.15:
            clusters.append(np.random.choice(["Low-Low","Not Significant"], p=[0.65,0.35]))
        elif dist[i] < 3:
            clusters.append(np.random.choice(["High-High","High-Low"], p=[0.5,0.5]))
        else:
            clusters.append(np.random.choice(
                ["High-High","Low-Low","High-Low","Low-High","Not Significant"],
                p=[0.14,0.14,0.08,0.07,0.57]))
    urban_df["cluster"] = clusters

    # Temporal data — one neighbourhood daily
    dates = pd.date_range("2018-01-01", "2023-05-24")
    nid_sample = 1
    row0 = urban_df.iloc[0]
    base_t = 33 - row0.tree_cover_pct * 0.05 + row0.asphalt_pct * 0.12
    seasonal = 3 * np.sin(2 * np.pi * pd.Series(dates).dt.dayofyear / 365).values
    noise    = np.random.normal(0, 1.2, len(dates))
    avg_t    = base_t + seasonal + noise

    temp_df = pd.DataFrame({
        "neighbourhood_id": nid_sample,
        "date": dates,
        "avg_temp": avg_t,
        "max_temp": avg_t + np.random.uniform(4, 7, len(dates)),
        "night_temp": avg_t - np.random.uniform(4, 8, len(dates)),
        "surface_temp": avg_t + row0.asphalt_pct * 0.11 + np.random.normal(0, 0.8, len(dates)),
        "humidity":   np.random.uniform(35, 80, len(dates)),
        "wind_speed": np.random.uniform(2, 22, len(dates)),
        "solar_radiation": np.clip(np.random.normal(400, 250, len(dates)), 0, 1100),
        "urban_heat_index": row0.urban_heat_index + np.random.normal(0, 0.05, len(dates)),
    })

    risk_idx = (avg_t - avg_t.mean()) / avg_t.std() * 0.6 + np.random.normal(0, 0.5, len(dates))
    temp_df["risk_index"] = risk_idx

    health_df = pd.DataFrame({
        "neighbourhood_id": nid_sample,
        "date": dates,
        "avg_temp": avg_t,
        "avg_temp_lag3": np.roll(avg_t, 3),
        "avg_temp_lag5": np.roll(avg_t, 5),
        "social_vulnerability_index": row0.social_vulnerability_index,
        "heat_fatigue_cases":  np.clip(avg_t * 1.3 - 35 + np.random.normal(0, 2, len(dates)), 0, None),
        "heatstroke_deaths":   np.clip(avg_t * 0.16 - 4.7 + np.random.normal(0, 0.3, len(dates)), 0, None),
        "dehydration_cases":   np.clip(avg_t * 0.9 - 20 + np.random.normal(0, 1.5, len(dates)), 0, None),
        "hospital_admissions": np.clip(avg_t * 0.5 - 10 + np.random.normal(0, 1, len(dates)), 0, None),
    })

    air_df = pd.DataFrame({
        "neighbourhood_id": nid_sample,
        "date": dates,
        "pm25": np.clip(np.random.normal(45, 18, len(dates)), 5, 150),
        "pm10": np.clip(np.random.normal(80, 25, len(dates)), 10, 200),
        "no2":  np.clip(np.random.normal(35, 12, len(dates)), 5, 100),
        "o3":   np.clip(np.random.normal(55, 15, len(dates)), 10, 120),
        "aqi":  np.clip(np.random.normal(90, 30, len(dates)), 2, 200),
        "building_density":  row0.building_density,
        "population_density": row0.population_density,
    })

    return urban_df, temp_df, health_df, air_df, True


urban_df, temp_df, health_df, air_df, IS_SYNTH = load_all()

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ UHI Dashboard")
    if IS_SYNTH:
        st.warning("**Demo mode** — synthetic data.\n\nPlace real CSVs beside the script to activate live results.", icon="⚠️")
    st.markdown("---")

    PAGES = {
        "🏠 Overview":                  "overview",
        "📊 Urban Surface EDA":         "eda",
        "📍 Spatial Clusters (LISA)":   "lisa",
        "🔬 PCA Decomposition":         "pca",
        "📈 Quantile Regression":       "qreg",
        "📡 SARIMAX — Surface Temp":    "sarima_temp",
        "🌀 SARIMAX — Risk Index":      "sarima_risk",
        "🏥 Temperature → Health":      "health",
    }
    page = st.radio("Navigate", list(PAGES.keys()))
    current = PAGES[page]

    st.markdown("---")
    st.markdown("**Study Area**")
    st.markdown(f"- Neighbourhoods: **{len(urban_df):,}**")
    st.markdown(f"- Temp records: **{len(temp_df):,}**")
    st.markdown(f"- Health records: **{len(health_df):,}**")
    st.markdown(f"- Air quality: **{len(air_df):,}**")
    st.markdown("- City: Vadodara, Gujarat 🇮🇳")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P A G E S
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ──────────────────────────────── OVERVIEW ────────────────────
if current == "overview":
    st.markdown('<div class="hero-title">🌆 Urban Heat Island — Full Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Vadodara, Gujarat · 500 Neighbourhoods · 2018–2023 · 7 Analytical Modules</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    kpis = [
        ("Neighbourhoods", f"{len(urban_df):,}"),
        ("Avg Tree Cover",  f"{urban_df['tree_cover_pct'].mean():.1f}%"),
        ("Avg Asphalt",     f"{urban_df['asphalt_pct'].mean():.1f}%"),
        ("Hot-Hot Clusters",f"{(urban_df['cluster']=='High-High').sum() if 'cluster' in urban_df else '—'}"),
    ]
    for col, (lbl, val) in zip(cols, kpis):
        col.metric(lbl, val)

    st.markdown("---")

    badges = [
        ("#ff6b35", "EDA",           "Urban Surface",        "Distributions, statistics & scatter matrix of 500 neighbourhoods"),
        ("#3498db", "STEP 1",        "Spatial Clusters",     "Moran's I + LISA map — where heat concentrates spatially"),
        ("#7b5ea7", "STEP 2",        "PCA",                  "Principal Component Analysis on 6 structural variables"),
        ("#2196F3", "STEP 3",        "Quantile Regression",  "Surface temp ~ tree cover + asphalt across q=0.50–0.90"),
        ("#27ae60", "TIME SERIES 1", "SARIMAX Surface Temp", "ARIMA + weekly seasonality forecasting on surface temperature"),
        ("#e67e22", "TIME SERIES 2", "SARIMAX Risk Index",   "SARIMAX on composite risk index (AQI + health + temp)"),
        ("#e74c3c", "HEALTH",        "Temp → Health Impact", "OLS lag models: heat fatigue (lag-3) & heatstroke deaths (lag-5)"),
    ]
    for color, badge, title, desc in badges:
        st.markdown(f"""
        <div class="step-card" style="border-left-color:{color}">
            <div class="step-badge" style="background:{color}">{badge}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Quick overview map
    st.markdown('<div class="sec-hdr">🗺️ Study Area — Neighbourhoods</div>', unsafe_allow_html=True)
    fig = px.scatter_mapbox(
        urban_df, lat="latitude", lon="longitude",
        color="tree_cover_pct",
        size="population_density" if "population_density" in urban_df.columns else None,
        color_continuous_scale="RdYlGn", zoom=10,
        mapbox_style="carto-darkmatter",
        size_max=10,
        hover_data={"tree_cover_pct": True, "asphalt_pct": True},
    )
    dark_layout(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────── EDA ─────────────────────────
elif current == "eda":
    st.markdown('<div class="hero-title">📊 Urban Surface — EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">UrbanSurface.ipynb — Distributions, descriptive stats & relationships</div>', unsafe_allow_html=True)

    num_vars = ["tree_cover_pct","asphalt_pct","building_density","heat_retention_factor",
                "infrastructure_quality_index","social_vulnerability_index",
                "median_income","population_density","distance_from_center_km"]
    num_vars = [v for v in num_vars if v in urban_df.columns]

    # Descriptive stats
    st.markdown('<div class="sec-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(urban_df[num_vars].describe().T.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)

    # Distributions
    st.markdown('<div class="sec-hdr">Variable Distributions</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select variable", num_vars, key="eda_dist")
    fig_h = px.histogram(urban_df, x=sel, nbins=40, color_discrete_sequence=[ACCENT])
    dark_layout(fig_h, height=350, title=f"Distribution of {sel}")
    st.plotly_chart(fig_h, use_container_width=True)

    # Scatter matrix
    st.markdown('<div class="sec-hdr">Scatter Matrix</div>', unsafe_allow_html=True)
    scatter_vars = st.multiselect(
        "Choose variables (3–6)",
        num_vars,
        default=["tree_cover_pct","asphalt_pct","building_density","heat_retention_factor"]
    )
    if len(scatter_vars) >= 2:
        fig_sm = px.scatter_matrix(
            urban_df[scatter_vars].sample(min(300, len(urban_df)), random_state=1),
            dimensions=scatter_vars,
            color_discrete_sequence=[ACCENT],
        )
        fig_sm.update_traces(marker=dict(size=3, opacity=0.5))
        dark_layout(fig_sm, height=550)
        st.plotly_chart(fig_sm, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="sec-hdr">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = urban_df[num_vars].corr()
    fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, aspect="auto")
    dark_layout(fig_c, height=430)
    st.plotly_chart(fig_c, use_container_width=True)

    # Bivariate explorer
    st.markdown('<div class="sec-hdr">Bivariate Explorer</div>', unsafe_allow_html=True)
    cx, cy = st.columns(2)
    xv = cx.selectbox("X axis", num_vars, index=0)
    yv = cy.selectbox("Y axis", num_vars, index=1)
    color_v = st.selectbox("Colour by", ["None"] + num_vars, index=0)
    color_arg = None if color_v == "None" else color_v
    fig_sc = px.scatter(urban_df, x=xv, y=yv, color=color_arg,
                        trendline="ols",
                        color_continuous_scale="RdYlGn_r" if color_arg else None,
                        opacity=0.7)
    dark_layout(fig_sc, height=400, title=f"{yv} vs {xv}")
    st.plotly_chart(fig_sc, use_container_width=True)


# ──────────────────────────────── LISA ────────────────────────
elif current == "lisa":
    st.markdown('<div class="hero-title">📍 Spatial Autocorrelation & LISA Clusters</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">step1.ipynb · Moran\'s I + LISA cluster map</div>', unsafe_allow_html=True)

    CMAP = {"High-High":"#e74c3c","Low-Low":"#3498db",
            "High-Low":"#e67e22","Low-High":"#9b59b6","Not Significant":"#6c757d"}

    if "cluster" not in urban_df.columns:
        st.warning("No cluster column found.")
        st.stop()

    c1, c2 = st.columns([1, 2])
    with c1:
        counts = urban_df["cluster"].value_counts().reset_index()
        counts.columns = ["Cluster","Count"]
        counts["Pct"] = (counts["Count"] / len(urban_df) * 100).round(1)
        fig_pie = px.pie(counts, names="Cluster", values="Count",
                         color="Cluster", color_discrete_map=CMAP, hole=0.42)
        dark_layout(fig_pie, height=340, title="LISA Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(counts, hide_index=True, use_container_width=True)

    with c2:
        fig_map = px.scatter_mapbox(urban_df, lat="latitude", lon="longitude",
                                    color="cluster", color_discrete_map=CMAP,
                                    zoom=10, mapbox_style="carto-darkmatter")
        fig_map.update_traces(marker_size=7)
        dark_layout(fig_map, height=400, title="LISA Cluster Map")
        st.plotly_chart(fig_map, use_container_width=True)

    # Moran's I scatter
    st.markdown('<div class="sec-hdr">Moran\'s I Scatter Plot</div>', unsafe_allow_html=True)

    z = urban_df["urban_heat_index"].values if "urban_heat_index" in urban_df.columns else (
        urban_df["asphalt_pct"] * 0.055 - urban_df["tree_cover_pct"] * 0.038
    ).values
    z_std = (z - z.mean()) / z.std()
    coords = urban_df[["latitude","longitude"]].values
    tree_kd = cKDTree(coords)
    _, idx = tree_kd.query(coords, k=11)
    lag = np.array([z_std[idx[i, 1:]].mean() for i in range(len(z_std))])
    moran_i = float(np.corrcoef(z_std, lag)[0, 1])

    fig_mi = go.Figure()
    colors_pt = urban_df["cluster"].map(CMAP).values
    fig_mi.add_trace(go.Scatter(x=z_std, y=lag, mode="markers",
                                marker=dict(color=colors_pt, size=5, opacity=0.65),
                                hovertemplate="z=%{x:.2f}, lag=%{y:.2f}<extra></extra>"))
    xr = np.linspace(z_std.min(), z_std.max(), 100)
    m, b = np.polyfit(z_std, lag, 1)
    fig_mi.add_trace(go.Scatter(x=xr, y=m*xr+b, mode="lines",
                                line=dict(color=ACCENT, width=2),
                                name=f"Moran's I ≈ {moran_i:.3f}"))
    fig_mi.add_hline(y=0, line_color="#444"); fig_mi.add_vline(x=0, line_color="#444")
    dark_layout(fig_mi, height=400, title=f"Moran's I ≈ {moran_i:.3f}")
    st.plotly_chart(fig_mi, use_container_width=True)
    st.info(f"**Moran's I = {moran_i:.3f}** — {'Positive spatial autocorrelation: heat clusters spatially.' if moran_i > 0 else 'No positive autocorrelation detected.'}")

    # Box by cluster
    st.markdown('<div class="sec-hdr">Variable Distribution by Cluster</div>', unsafe_allow_html=True)
    vsel = st.selectbox("Variable", ["tree_cover_pct","asphalt_pct","building_density","heat_retention_factor","social_vulnerability_index"])
    fig_box = px.box(urban_df, x="cluster", y=vsel, color="cluster",
                     color_discrete_map=CMAP)
    dark_layout(fig_box, height=380, title=f"{vsel} by LISA Cluster")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)


# ──────────────────────────────── PCA ─────────────────────────
elif current == "pca":
    st.markdown('<div class="hero-title">🔬 Principal Component Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">step2.ipynb · 6 structural variables → component decomposition</div>', unsafe_allow_html=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    s_vars = ["tree_cover_pct","asphalt_pct","building_density",
              "heat_retention_factor","social_vulnerability_index","infrastructure_quality_index"]
    X = urban_df[s_vars].dropna()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(); pcs = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f"PC{i+1}" for i in range(len(s_vars))],
                            index=s_vars)

    # Scree
    st.markdown('<div class="sec-hdr">Scree Plot</div>', unsafe_allow_html=True)
    fig_sc = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sc.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))],
                            y=evr*100,
                            marker_color=[ACCENT if i < 4 else "#3a3f5c" for i in range(len(evr))],
                            text=[f"{v*100:.1f}%" for v in evr], textposition="outside",
                            name="Variance %"), secondary_y=False)
    fig_sc.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(evr))],
                                y=cum*100, mode="lines+markers",
                                line=dict(color="#a8edea", width=2),
                                marker=dict(size=7), name="Cumulative %"),
                     secondary_y=True)
    fig_sc.update_yaxes(title_text="Explained Variance (%)", secondary_y=False, color=FONT_COL)
    fig_sc.update_yaxes(title_text="Cumulative (%)", secondary_y=True, range=[0, 105], color="#a8edea")
    dark_layout(fig_sc, height=360, title="Scree Plot — Explained Variance")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Key stat
    st.info(f"PC1–PC4 explain **{cum[3]*100:.1f}%** of total variance. PC1 alone captures **{evr[0]*100:.1f}%**.")

    # Loadings heatmap
    st.markdown('<div class="sec-hdr">Loadings Heatmap</div>', unsafe_allow_html=True)
    fig_lh = px.imshow(loadings, text_auto=".2f",
                       color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    dark_layout(fig_lh, height=360, title="PCA Loadings — Variables × Components")
    st.plotly_chart(fig_lh, use_container_width=True)

    # Biplot
    st.markdown('<div class="sec-hdr">Interactive Biplot</div>', unsafe_allow_html=True)
    bx, by = st.columns(2)
    pcx = bx.selectbox("X component", [f"PC{i+1}" for i in range(6)], index=0)
    pcy = by.selectbox("Y component", [f"PC{i+1}" for i in range(6)], index=1)
    ci, cj = int(pcx[2:])-1, int(pcy[2:])-1
    cv = st.selectbox("Colour by", s_vars, index=0)
    sc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(6)])
    sc_df["colour"] = urban_df[cv].values[:len(sc_df)]
    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(
        x=sc_df[pcx], y=sc_df[pcy], mode="markers",
        marker=dict(color=sc_df["colour"], colorscale="RdYlGn_r",
                    size=5, opacity=0.55,
                    colorbar=dict(title=cv.replace("_"," "), thickness=12)),
        name="Neighbourhoods"))
    scale = max(abs(pcs[:,ci]).max(), abs(pcs[:,cj]).max()) * 0.4
    for var in s_vars:
        lx = pca.components_[ci, s_vars.index(var)] * scale
        ly = pca.components_[cj, s_vars.index(var)] * scale
        fig_bp.add_annotation(x=lx, y=ly, ax=0, ay=0,
                              arrowhead=2, arrowwidth=1.5, arrowcolor="#f7c59f",
                              font=dict(color="#f7c59f", size=10),
                              text=var.replace("_"," "), showarrow=True)
    dark_layout(fig_bp, height=480,
                title=f"Biplot — {pcx} ({evr[ci]*100:.1f}%) vs {pcy} ({evr[cj]*100:.1f}%)")
    st.plotly_chart(fig_bp, use_container_width=True)

    with st.expander("Full loadings table"):
        st.dataframe(loadings.style.format("{:.4f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                     use_container_width=True)


# ──────────────────────────────── QUANTILE REGRESSION ─────────
elif current == "qreg":
    st.markdown('<div class="hero-title">📈 Quantile Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">step3.ipynb · Surface Temp ~ Tree Cover + Asphalt · q = 0.25 → 0.95</div>', unsafe_allow_html=True)

    # ── FIX: define column names upfront ──────────────────────
    tc_col  = "tree_cover_pct"
    asp_col = "asphalt_pct"

    try:
        import statsmodels.api as sm

        # Merge temp_df with urban_df to bring in tree_cover_pct & asphalt_pct
        if "surface_temp" in temp_df.columns:
            data = temp_df.merge(
                urban_df[["neighbourhood_id", tc_col, asp_col]],
                on="neighbourhood_id", how="left"
            ).dropna(subset=["surface_temp", tc_col, asp_col])
            y = data["surface_temp"]
            X = sm.add_constant(data[[tc_col, asp_col]])
        else:
            raise ValueError("no surface_temp column in temp_df")

        quantiles = [0.25, 0.50, 0.75, 0.90, 0.95]
        results = {}
        for q in quantiles:
            m = sm.QuantReg(y, X).fit(q=q, max_iter=2000)
            results[q] = {
                "const":   float(m.params.iloc[0]),
                "tree":    float(m.params.iloc[1]),
                "asphalt": float(m.params.iloc[2]),
            }

    except Exception:
        # Fallback coefficients matching notebook values
        results = {
            0.25: {"const": 31.20, "tree": -0.043, "asphalt": 0.136},
            0.50: {"const": 35.95, "tree": -0.045, "asphalt": 0.141},
            0.75: {"const": 42.09, "tree": -0.046, "asphalt": 0.140},
            0.90: {"const": 44.29, "tree": -0.046, "asphalt": 0.141},
            0.95: {"const": 46.50, "tree": -0.047, "asphalt": 0.142},
        }

    qs = list(results.keys())

    # Coefficient plots
    st.markdown('<div class="sec-hdr">Coefficients Across Quantiles</div>', unsafe_allow_html=True)
    fig_cf = make_subplots(rows=1, cols=2, subplot_titles=["🌳 Tree Cover (β₁)", "🛣️ Asphalt (β₂)"])
    fig_cf.add_trace(go.Scatter(x=qs, y=[results[q]["tree"] for q in qs],
                                mode="lines+markers", line=dict(color="#27ae60", width=2.5),
                                marker=dict(size=9), name="Tree β"), row=1, col=1)
    fig_cf.add_trace(go.Scatter(x=qs, y=[results[q]["asphalt"] for q in qs],
                                mode="lines+markers", line=dict(color="#e74c3c", width=2.5),
                                marker=dict(size=9), name="Asphalt β"), row=1, col=2)
    fig_cf.add_hline(y=0, line_color="#444")
    dark_layout(fig_cf, height=370, title="Quantile Regression Coefficients")
    st.plotly_chart(fig_cf, use_container_width=True)

    # Table
    st.markdown('<div class="sec-hdr">Coefficient Table</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Quantile": [f"q = {q}" for q in qs],
        "Intercept": [f"{results[q]['const']:.3f}" for q in qs],
        "Tree Cover (β₁)": [f"{results[q]['tree']:.4f}" for q in qs],
        "Asphalt (β₂)": [f"{results[q]['asphalt']:.4f}" for q in qs],
        "Interpretation": [
            "Median surface temp" if q == 0.50 else f"Top {100 - int(q*100)}% hottest days"
            for q in qs
        ]
    })
    st.dataframe(coef_df, hide_index=True, use_container_width=True)

    # Regression lines
    st.markdown('<div class="sec-hdr">Fitted Regression Lines</div>', unsafe_allow_html=True)
    rv = st.radio("View effect of", ["Tree Cover (%)", "Asphalt (%)"], horizontal=True)
    use_tree = "Tree" in rv
    avg_tree = urban_df[tc_col].mean()
    avg_asp  = urban_df[asp_col].mean()

    xrange = np.linspace(
        urban_df[tc_col].min()  if use_tree else urban_df[asp_col].min(),
        urban_df[tc_col].max()  if use_tree else urban_df[asp_col].max(),
        200
    )

    fig_rl = go.Figure()

    # ── FIX: build merged scatter sample so tc_col & asp_col exist ──
    if "surface_temp" in temp_df.columns and tc_col in temp_df.columns:
        # Real data already has those columns in temp_df
        samp = temp_df[[tc_col, asp_col, "surface_temp"]].dropna().sample(
            min(2000, len(temp_df)), random_state=1
        )
    elif "surface_temp" in temp_df.columns:
        # Merge from urban_df
        merged_for_scatter = temp_df.merge(
            urban_df[["neighbourhood_id", tc_col, asp_col]],
            on="neighbourhood_id", how="left"
        ).dropna(subset=["surface_temp", tc_col, asp_col])
        samp = merged_for_scatter[[tc_col, asp_col, "surface_temp"]].sample(
            min(2000, len(merged_for_scatter)), random_state=1
        )
    else:
        samp = None

    if samp is not None:
        xscat = samp[tc_col].values if use_tree else samp[asp_col].values
        fig_rl.add_trace(go.Scatter(
            x=xscat, y=samp["surface_temp"].values,
            mode="markers",
            marker=dict(color="#444d6e", size=4, opacity=0.4),
            name="Observed"
        ))

    qcolors = {0.25:"#3498db", 0.50:"#f7c59f", 0.75:"#e67e22", 0.90:"#e74c3c", 0.95:"#8e44ad"}
    for q in qs:
        if use_tree:
            yp = results[q]["const"] + results[q]["tree"] * xrange + results[q]["asphalt"] * avg_asp
        else:
            yp = results[q]["const"] + results[q]["tree"] * avg_tree + results[q]["asphalt"] * xrange
        fig_rl.add_trace(go.Scatter(
            x=xrange, y=yp, mode="lines",
            line=dict(color=qcolors[q], width=2.5),
            name=f"q={q}"
        ))

    dark_layout(fig_rl, height=430,
                title=f"Surface Temp vs {'Tree Cover' if use_tree else 'Asphalt'}")
    st.plotly_chart(fig_rl, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.success(f"🌳 Each +1% tree cover → **{results[0.50]['tree']:.4f} °C** (median) | **{results[0.90]['tree']:.4f} °C** (90th pct)")
    c2.error(  f"🛣️ Each +1% asphalt → **+{results[0.50]['asphalt']:.4f} °C** (median) | **+{results[0.90]['asphalt']:.4f} °C** (90th pct)")


# ──────────────────────────── SARIMAX — SURFACE TEMP ──────────
elif current == "sarima_temp":
    st.markdown('<div class="hero-title">📡 SARIMAX — Surface Temperature</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Auto_Correlation.ipynb · SARIMAX(1,1,1)×(1,1,1,7) · 30-day forecast</div>', unsafe_allow_html=True)

    if "surface_temp" not in temp_df.columns or "date" not in temp_df.columns:
        st.warning("surface_temp or date column not found.")
        st.stop()

    # One neighbourhood time series (aggregate mean by date)
    ts = temp_df.groupby("date")[["surface_temp","avg_temp","humidity","wind_speed"]].mean().sort_index()
    ts = ts.asfreq("D").ffill()
    y = ts["surface_temp"]

    # ADF
    from statsmodels.tsa.stattools import adfuller
    adf = adfuller(y.dropna())
    st.markdown('<div class="sec-hdr">Stationarity Check — ADF Test</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("ADF Statistic", f"{adf[0]:.4f}")
    sc2.metric("p-value",       f"{adf[1]:.4f}")
    sc3.metric("Stationary?",   "Yes ✅" if adf[1] < 0.05 else "No ❌ (differencing needed)")

    # Time series plot
    st.markdown('<div class="sec-hdr">Surface Temperature Time Series</div>', unsafe_allow_html=True)
    ma_win = st.slider("Moving-average window (days)", 7, 90, 30)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines",
                                line=dict(color="#555", width=1), name="Daily"))
    fig_ts.add_trace(go.Scatter(x=y.index, y=y.rolling(ma_win).mean().values,
                                mode="lines", line=dict(color=ACCENT, width=2),
                                name=f"{ma_win}-day MA"))
    dark_layout(fig_ts, height=350, title="Surface Temperature (daily + moving average)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # ACF / PACF (approximate with plotly)
    st.markdown('<div class="sec-hdr">ACF & PACF (differenced series)</div>', unsafe_allow_html=True)
    y_diff = y.diff().dropna()
    max_lag = 40
    acf_vals  = [y_diff.autocorr(lag=i) for i in range(1, max_lag+1)]
    pacf_vals = []
    for lag in range(1, max_lag+1):
        m_ = y_diff.values[lag:]
        X_ = np.column_stack([y_diff.values[:-lag]])
        if len(m_) > 5:
            pacf_vals.append(float(np.corrcoef(m_, X_[:,0])[0,1]))
        else:
            pacf_vals.append(0)

    fig_acf = make_subplots(rows=1, cols=2, subplot_titles=["ACF — Differenced","PACF — Differenced"])
    ci_line = 1.96 / np.sqrt(len(y_diff))
    for lags, vals, col in [(range(1, max_lag+1), acf_vals, 1),
                            (range(1, max_lag+1), pacf_vals, 2)]:
        for lg, v in zip(lags, vals):
            fig_acf.add_trace(go.Bar(x=[lg], y=[v],
                                     marker_color=ACCENT if abs(v) > ci_line else "#3a3f5c",
                                     showlegend=False), row=1, col=col)
        fig_acf.add_hline(y= ci_line, line_dash="dot", line_color="#a8edea", row=1, col=col)
        fig_acf.add_hline(y=-ci_line, line_dash="dot", line_color="#a8edea", row=1, col=col)
    dark_layout(fig_acf, height=330)
    st.plotly_chart(fig_acf, use_container_width=True)

    # SARIMAX forecast
    st.markdown('<div class="sec-hdr">SARIMAX Forecast (30 days)</div>', unsafe_allow_html=True)
    steps = st.slider("Forecast horizon (days)", 7, 60, 30)

    with st.spinner("Fitting SARIMAX(1,1,1)×(1,1,1,7)…"):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            train = y[:-steps]; test = y[-steps:]
            mod = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.get_forecast(steps=steps)
            fc_mean = fc.predicted_mean
            fc_ci   = fc.conf_int()
            rmse = float(np.sqrt(((test.values - fc_mean.values)**2).mean()))
            mae  = float(np.abs(test.values - fc_mean.values).mean())
            fitted_ok = True
        except Exception as e:
            st.warning(f"SARIMAX fit failed ({e}). Showing naive forecast.")
            last_val = float(y.iloc[-steps-1])
            fc_mean = pd.Series(np.full(steps, last_val), index=y.index[-steps:])
            fc_ci = pd.DataFrame({"lower surface_temp": fc_mean*0.97, "upper surface_temp": fc_mean*1.03})
            rmse = mae = float("nan"); fitted_ok = False

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "SARIMAX(1,1,1)×(1,1,1,7)")
    m2.metric("RMSE",  f"{rmse:.4f} °C" if fitted_ok else "—")
    m3.metric("MAE",   f"{mae:.4f} °C"  if fitted_ok else "—")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=y.index[-120:], y=y.values[-120:],
                                mode="lines", line=dict(color="#a8edea", width=1.5), name="Actual"))
    fig_fc.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values,
                                mode="lines", line=dict(color=ACCENT, width=2.5), name="Forecast"))
    fig_fc.add_trace(go.Scatter(
        x=list(fc_ci.index) + list(fc_ci.index[::-1]),
        y=list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0][::-1]),
        fill="toself", fillcolor="rgba(255,107,53,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    dark_layout(fig_fc, height=400, title=f"SARIMAX Surface Temp — {steps}-day Forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    if fitted_ok:
        st.markdown('<div class="sec-hdr">Model Diagnostics</div>', unsafe_allow_html=True)
        resid = res.resid
        fig_diag = make_subplots(rows=1, cols=2, subplot_titles=["Residuals","Residual Histogram"])
        fig_diag.add_trace(go.Scatter(x=list(range(len(resid))), y=resid.values,
                                      mode="lines", line=dict(color=ACCENT, width=1)), row=1, col=1)
        fig_diag.add_trace(go.Histogram(x=resid.values, nbinsx=40,
                                        marker_color="#7b5ea7", opacity=0.8), row=1, col=2)
        dark_layout(fig_diag, height=330, title="SARIMAX Residual Diagnostics")
        st.plotly_chart(fig_diag, use_container_width=True)


# ──────────────────────────── SARIMAX — RISK INDEX ────────────
elif current == "sarima_risk":
    st.markdown('<div class="hero-title">🌀 SARIMAX — Risk Index</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Times_series_modal.ipynb · SARIMAX(1,1,1)×(1,0,1,7) · Risk Index = AQI + Health + Temp composite</div>', unsafe_allow_html=True)

    # Build risk index
    ts_t = temp_df.groupby("date")[["avg_temp","humidity","wind_speed"]].mean().sort_index()
    ts_a = air_df.groupby("date")[["aqi","pm25"]].mean().sort_index()
    ts_h = health_df.groupby("date")[["heat_fatigue_cases","heatstroke_deaths"]].mean().sort_index()

    combined = ts_t.join(ts_a, how="outer").join(ts_h, how="outer").ffill().bfill()

    # Normalise & composite
    def znorm(s): return (s - s.mean()) / (s.std() + 1e-8)
    if "aqi" in combined.columns:
        combined["risk_index"] = (znorm(combined["avg_temp"])
                                  + znorm(combined["aqi"]) * 0.5
                                  + znorm(combined.get("heat_fatigue_cases", pd.Series(0, index=combined.index))) * 0.3)
    elif "risk_index" in temp_df.columns:
        combined["risk_index"] = temp_df.groupby("date")["risk_index"].mean()
    else:
        combined["risk_index"] = znorm(combined["avg_temp"])

    combined = combined.asfreq("D").ffill()
    y_risk = combined["risk_index"]

    # Time series
    st.markdown('<div class="sec-hdr">Risk Index Time Series</div>', unsafe_allow_html=True)
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=y_risk.index, y=y_risk.values, mode="lines",
                               line=dict(color="#7b5ea7", width=1.2), name="Risk Index"))
    fig_r.add_trace(go.Scatter(x=y_risk.index, y=y_risk.rolling(30).mean().values,
                               mode="lines", line=dict(color=ACCENT, width=2), name="30-day MA"))
    dark_layout(fig_r, height=350, title="Composite Risk Index (Temp + AQI + Health)")
    st.plotly_chart(fig_r, use_container_width=True)

    # Exogenous variables correlation
    st.markdown('<div class="sec-hdr">Exogenous Correlations with Risk Index</div>', unsafe_allow_html=True)
    exog_avail = [c for c in ["humidity","wind_speed","aqi","pm25"] if c in combined.columns]
    corr_ri = {c: float(y_risk.corr(combined[c])) for c in exog_avail}
    fig_corr_bar = go.Figure(go.Bar(
        x=list(corr_ri.keys()), y=list(corr_ri.values()),
        marker_color=[ACCENT if v > 0 else "#3498db" for v in corr_ri.values()],
        text=[f"{v:.3f}" for v in corr_ri.values()], textposition="outside"
    ))
    dark_layout(fig_corr_bar, height=320, title="Correlation of Exogenous Variables with Risk Index")
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    # SARIMAX on risk index
    st.markdown('<div class="sec-hdr">SARIMAX Forecast</div>', unsafe_allow_html=True)
    steps_r = st.slider("Forecast horizon (days)", 7, 60, 30, key="risk_steps")

    with st.spinner("Fitting SARIMAX on Risk Index…"):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            exog_cols = [c for c in ["humidity","wind_speed"] if c in combined.columns]
            if exog_cols:
                exog_data = combined[exog_cols].ffill().bfill()
                train_y = y_risk.iloc[:-steps_r]; test_y = y_risk.iloc[-steps_r:]
                train_ex = exog_data.iloc[:-steps_r]; test_ex = exog_data.iloc[-steps_r:]
                mod_r = SARIMAX(train_y, exog=train_ex, order=(1,1,1),
                                seasonal_order=(1,0,1,7),
                                enforce_stationarity=False, enforce_invertibility=False)
            else:
                train_y = y_risk.iloc[:-steps_r]; test_y = y_risk.iloc[-steps_r:]
                mod_r = SARIMAX(train_y, order=(1,1,1), seasonal_order=(1,0,1,7),
                                enforce_stationarity=False, enforce_invertibility=False)
                test_ex = None
            res_r = mod_r.fit(disp=False)
            fc_r = res_r.get_forecast(steps=steps_r, exog=test_ex if exog_cols else None)
            fc_mean_r = fc_r.predicted_mean
            fc_ci_r   = fc_r.conf_int()
            rmse_r = float(np.sqrt(((test_y.values - fc_mean_r.values)**2).mean()))
            mae_r  = float(np.abs(test_y.values - fc_mean_r.values).mean())
            fit_ok_r = True
        except Exception as e:
            st.warning(f"SARIMAX fit failed: {e}")
            fc_mean_r = pd.Series(np.zeros(steps_r), index=y_risk.index[-steps_r:])
            fc_ci_r = pd.DataFrame({"lower": fc_mean_r - 0.5, "upper": fc_mean_r + 0.5})
            rmse_r = mae_r = float("nan"); fit_ok_r = False

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "SARIMAX(1,1,1)×(1,0,1,7)")
    m2.metric("RMSE", f"{rmse_r:.4f}" if fit_ok_r else "—")
    m3.metric("MAE",  f"{mae_r:.4f}"  if fit_ok_r else "—")
    st.caption("Notebook MAE ≈ 0.308 · RMSE ≈ 0.370")

    fig_rfore = go.Figure()
    fig_rfore.add_trace(go.Scatter(x=y_risk.index[-120:], y=y_risk.values[-120:],
                                   mode="lines", line=dict(color="#a8edea", width=1.5), name="Actual"))
    fig_rfore.add_trace(go.Scatter(x=fc_mean_r.index, y=fc_mean_r.values,
                                   mode="lines", line=dict(color=ACCENT, width=2.5), name="Forecast"))
    fig_rfore.add_trace(go.Scatter(
        x=list(fc_ci_r.index)+list(fc_ci_r.index[::-1]),
        y=list(fc_ci_r.iloc[:,1])+list(fc_ci_r.iloc[:,0][::-1]),
        fill="toself", fillcolor="rgba(255,107,53,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    dark_layout(fig_rfore, height=400, title=f"Risk Index — {steps_r}-day SARIMAX Forecast")
    st.plotly_chart(fig_rfore, use_container_width=True)

    # Component decomposition
    st.markdown('<div class="sec-hdr">Seasonal Decomposition</div>', unsafe_allow_html=True)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        dec = seasonal_decompose(y_risk.dropna(), model="additive", period=365 if len(y_risk) > 730 else 7)
        fig_dec = make_subplots(rows=4, cols=1, subplot_titles=["Observed","Trend","Seasonal","Residual"],
                                vertical_spacing=0.06)
        for i, (comp, name) in enumerate([(dec.observed,"Observed"),(dec.trend,"Trend"),
                                           (dec.seasonal,"Seasonal"),(dec.resid,"Residual")], 1):
            fig_dec.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines",
                                         line=dict(color=ACCENT if i == 2 else "#a8edea", width=1.2),
                                         name=name), row=i, col=1)
        dark_layout(fig_dec, height=700, title="Seasonal Decomposition — Risk Index")
        st.plotly_chart(fig_dec, use_container_width=True)
    except Exception:
        st.info("Seasonal decomposition requires at least 2 full periods of data.")


# ──────────────────────────── HEALTH IMPACT ───────────────────
elif current == "health":
    st.markdown('<div class="hero-title">🏥 Temperature → Health Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Tempreture_Health.ipynb · OLS chain: Urban Structure → Heat → Health</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Analysis pipeline:</b>
    Urban Structure → Heat Amplification → Energy Stress → Health Impact<br>
    Models: OLS surface temp (structural vars) · OLS heat fatigue (lag-3) · OLS heatstroke deaths (lag-5) · Full structural model
    </div>""", unsafe_allow_html=True)

    try:
        import statsmodels.api as sm

        # Merge urban + temp
        struct_cols = ["neighbourhood_id","tree_cover_pct","asphalt_pct",
                       "building_density","heat_retention_factor","social_vulnerability_index"]
        struct_cols = [c for c in struct_cols if c in urban_df.columns]
        df_st = temp_df.merge(urban_df[struct_cols], on="neighbourhood_id", how="left").dropna()

        # ── Model 1: Structural → Surface Temp ───────────────
        st.markdown('<div class="sec-hdr">Model 1 — Urban Structure → Surface Temperature</div>', unsafe_allow_html=True)
        feat1 = ["tree_cover_pct","asphalt_pct","building_density","heat_retention_factor"]
        feat1 = [f for f in feat1 if f in df_st.columns]
        if feat1:
            X1 = sm.add_constant(df_st[feat1].sample(min(50000, len(df_st)), random_state=1))
            y1 = df_st.loc[X1.index, "surface_temp"]
            m1 = sm.OLS(y1, X1).fit()

            coef_df = pd.DataFrame({
                "Variable":    m1.params.index,
                "Coefficient": m1.params.values,
                "p-value":     m1.pvalues.values,
            })

            r1c, r1m = st.columns([2,1])
            with r1c:
                st.dataframe(coef_df, hide_index=True, use_container_width=True)
            with r1m:
                st.metric("R²", f"{m1.rsquared:.3f}")
                st.metric("F-stat", f"{m1.fvalue:.0f}")
                st.caption("Notebook R² ≈ 0.132\nAll vars significant (p<0.001)")

            # Coefficient bar
            coef_vals = dict(zip(feat1, m1.params[1:]))
            fig_c1 = go.Figure(go.Bar(
                x=list(coef_vals.keys()), y=list(coef_vals.values()),
                marker_color=[("#27ae60" if v < 0 else "#e74c3c") for v in coef_vals.values()],
                text=[f"{v:.4f}" for v in coef_vals.values()], textposition="outside"
            ))
            dark_layout(fig_c1, height=320, title="Model 1 Coefficients — Effect on Surface Temp")
            st.plotly_chart(fig_c1, use_container_width=True)

        # ── Model 2: Lag-3 → Heat Fatigue ────────────────────
        st.markdown('<div class="sec-hdr">Model 2 — Temp (lag-3) → Heat Fatigue Cases</div>', unsafe_allow_html=True)
        if all(c in health_df.columns for c in ["avg_temp_lag3","heat_fatigue_cases","social_vulnerability_index"]):
            df_h3 = health_df.dropna(subset=["avg_temp_lag3","heat_fatigue_cases"])
            X2 = sm.add_constant(df_h3[["avg_temp_lag3","social_vulnerability_index"]].sample(min(50000, len(df_h3)), random_state=1))
            y2 = df_h3.loc[X2.index, "heat_fatigue_cases"]
            m2 = sm.OLS(y2, X2).fit()

            c2a, c2b = st.columns([2,1])
            coef_d2 = pd.DataFrame({
                "Variable":    m2.params.index,
                "Coefficient": m2.params.values,
                "p-value":     [f"{p:.4f}" for p in m2.pvalues],
            })
            with c2a: st.dataframe(coef_d2, hide_index=True, use_container_width=True)
            with c2b:
                st.metric("R²", f"{m2.rsquared:.3f}")
                st.caption("Notebook lag-3 R² ≈ 0.622")

        # ── Model 3: Lag-5 → Heatstroke Deaths ───────────────
        st.markdown('<div class="sec-hdr">Model 3 — Temp (lag-5) → Heatstroke Deaths</div>', unsafe_allow_html=True)
        if all(c in health_df.columns for c in ["avg_temp_lag5","heatstroke_deaths"]):
            df_h5 = health_df.dropna(subset=["avg_temp_lag5","heatstroke_deaths"])
            X3 = sm.add_constant(df_h5[["avg_temp_lag5","social_vulnerability_index"]].sample(min(50000, len(df_h5)), random_state=1))
            y3 = df_h5.loc[X3.index, "heatstroke_deaths"]
            m3_ols = sm.OLS(y3, X3).fit()

            c3a, c3b = st.columns([2,1])
            coef_d3 = pd.DataFrame({
                "Variable":    m3_ols.params.index,
                "Coefficient": m3_ols.params.values,
                "p-value":     [f"{p:.4f}" for p in m3_ols.pvalues],
            })
            with c3a: st.dataframe(coef_d3, hide_index=True, use_container_width=True)
            with c3b:
                st.metric("R²", f"{m3_ols.rsquared:.3f}")
                st.caption("Notebook lag-5 R² ≈ 0.698\nTemp coef ≈ +0.162")

        # ── Visual: Lag effect on health ─────────────────────
        st.markdown('<div class="sec-hdr">Health Outcome Trends</div>', unsafe_allow_html=True)
        h_metric = st.selectbox("Health metric",
                                [c for c in ["heat_fatigue_cases","heatstroke_deaths",
                                             "dehydration_cases","hospital_admissions"]
                                 if c in health_df.columns])
        ts_h = health_df.groupby("date")[h_metric].mean()
        ts_t_agg = health_df.groupby("date")["avg_temp"].mean()

        fig_h = make_subplots(specs=[[{"secondary_y": True}]])
        fig_h.add_trace(go.Scatter(x=ts_h.index, y=ts_h.values,
                                   mode="lines", line=dict(color="#e74c3c", width=1.5),
                                   name=h_metric), secondary_y=False)
        fig_h.add_trace(go.Scatter(x=ts_t_agg.index, y=ts_t_agg.values,
                                   mode="lines", line=dict(color="#a8edea", width=1.5, dash="dot"),
                                   name="Avg Temp"), secondary_y=True)
        fig_h.update_yaxes(title_text=h_metric, color="#e74c3c", secondary_y=False)
        fig_h.update_yaxes(title_text="Avg Temp (°C)", color="#a8edea", secondary_y=True)
        dark_layout(fig_h, height=380, title=f"{h_metric} vs Temperature over Time")
        st.plotly_chart(fig_h, use_container_width=True)

        # Scatter: temp vs health
        st.markdown('<div class="sec-hdr">Scatter — Temperature vs Health Outcome</div>', unsafe_allow_html=True)
        lag_col = "avg_temp_lag3" if "avg_temp_lag3" in health_df.columns else "avg_temp"
        samp_h = health_df[[lag_col, h_metric]].dropna().sample(min(3000, len(health_df)), random_state=1)
        fig_hs = px.scatter(samp_h, x=lag_col, y=h_metric, trendline="ols",
                            trendline_color_override=ACCENT, opacity=0.4,
                            color_discrete_sequence=["#e74c3c"])
        dark_layout(fig_hs, height=380, title=f"{h_metric} vs {lag_col}")
        st.plotly_chart(fig_hs, use_container_width=True)

        # Summary boxes
        st.markdown('<div class="sec-hdr">📌 Key Findings</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.info("**Model 1 — Surface Temp**\n\nR² ≈ 0.132\n\nHeat retention (+4.9), Building density (+4.1), Asphalt (+0.18), Tree cover (−0.05)")
        c2.success("**Model 2 — Heat Fatigue (lag-3)**\n\nR² ≈ 0.622\n\nEvery +1°C (lag-3) adds **+1.34 cases**")
        c3.error("**Model 3 — Heatstroke Deaths (lag-5)**\n\nR² ≈ 0.698\n\nEvery +1°C (lag-5) adds **+0.16 deaths**\n\nSocial vulnerability significant (p<0.001)")

    except ImportError:
        st.error("statsmodels not installed. Run: `pip install statsmodels`")
    except Exception as e:
        st.error(f"Error running health models: {e}")
        st.exception(e)