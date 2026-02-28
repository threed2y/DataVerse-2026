import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ──────────────────────── PAGE CONFIG ────────────────────────
st.set_page_config(
    page_title="The DNA of Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────── Y2K / BRUTALIST CSS ────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Space Mono', monospace !important;
    background-color: #F4F4F0 !important;
    color: #000000 !important;
}
.stApp {
    background-color: #F4F4F0 !important;
}

/* ── Headers ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Mono', monospace !important;
    color: #000000 !important;
    letter-spacing: -0.02em;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #000000;
    border: 2px solid #000000;
    padding: 18px 20px;
    border-radius: 0px;
}
[data-testid="stMetric"] label {
    color: #39FF14 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #39FF14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 3px solid #000000;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    color: #000000 !important;
    background: transparent;
    border: 2px solid #000000;
    border-bottom: none;
    border-radius: 0px;
    padding: 10px 24px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #000000 !important;
    color: #39FF14 !important;
}

/* ── Dividers ── */
hr {
    border: 1px solid #000000 !important;
}

/* ── Select / Dropdown ── */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-family: 'Space Mono', monospace !important;
    color: #000000 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.8rem !important;
    font-weight: 700;
}
.stSelectbox [data-baseweb="select"] {
    border-radius: 0px !important;
    border: 2px solid #000000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 2px solid #000000 !important;
    border-radius: 0px !important;
}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border: 2px solid #000000;
    padding: 4px;
    background: #F4F4F0;
}

/* ── Custom blocks ── */
.era-divider {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    border-top: 3px solid #000000;
    border-bottom: 3px solid #000000;
    padding: 12px 0;
    margin: 32px 0 24px 0;
    text-transform: uppercase;
}
.subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: #555;
    margin-top: -10px;
    margin-bottom: 24px;
}
.kpi-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
    margin-bottom: 4px;
}
.footer-text {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    margin-top: 60px;
    padding: 20px;
    border-top: 1px solid #000;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────── PALETTE ────────────────────────────
NEON_GREEN = "#39FF14"
DEEP_PURPLE = "#800080"
BLACK = "#000000"
GRAY = "#888888"
BG_COLOR = "#F4F4F0"
ACCENT_CYAN = "#00FFFF"

# ──────────────────────── DATA LOADING ───────────────────────
@st.cache_data
def load_and_process_data():
    data_path = os.path.join(os.path.dirname(__file__), "songs.xlsx")
    df = pd.read_excel(data_path)

    # ── Parse dates ──
    df["release_date"] = pd.to_datetime(df["track_album_release_date"], errors="coerce")
    df["Year"] = df["release_date"].dt.year

    # Drop rows where year couldn't be extracted
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # ── Era column ──
    def assign_era(year):
        if year < 2000:
            return "The Analog Era (Pre-2000)"
        elif year <= 2012:
            return "The Genre Explosion (2000-2012)"
        else:
            return "The Streaming Era (2013-2020)"

    df["Era"] = df["Year"].apply(assign_era)

    # ── Duration in seconds for readability ──
    df["duration_s"] = df["duration_ms"] / 1000.0

    return df


df = load_and_process_data()

# ──────────────────────── PLOTLY LAYOUT HELPER ───────────────
def brutalist_layout(fig, title="", height=480, show_legend=True):
    """Apply consistent brutalist styling to every Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Space Mono, monospace", size=16, color=BLACK),
            x=0.01,
        ),
        font=dict(family="IBM Plex Mono, monospace", size=12, color=BLACK),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        showlegend=show_legend,
        legend=dict(
            font=dict(family="IBM Plex Mono, monospace", size=11, color=BLACK),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BLACK,
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BLACK,
        linewidth=1.5,
        tickfont=dict(family="IBM Plex Mono, monospace", size=11, color=BLACK),
        title_font=dict(family="Space Mono, monospace", size=13, color=BLACK),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BLACK,
        linewidth=1.5,
        tickfont=dict(family="IBM Plex Mono, monospace", size=11, color=BLACK),
        title_font=dict(family="Space Mono, monospace", size=13, color=BLACK),
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center; font-size:2.4rem; margin-bottom:0;'>"
    "🎵 THE DNA OF MUSIC</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-family:IBM Plex Mono, monospace; "
    "font-size:1.05rem; color:#555; margin-top:0;'>"
    "60 Years of Sonic Evolution &mdash; 1957 → 2020</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────── KPI METRICS ────────────────────────
analog = df[df["Era"] == "The Analog Era (Pre-2000)"]
streaming = df[df["Era"] == "The Streaming Era (2013-2020)"]

avg_dur_analog = analog["duration_s"].mean()
avg_dur_stream = streaming["duration_s"].mean()
avg_energy_analog = analog["energy"].mean()
avg_energy_stream = streaming["energy"].mean()
avg_dance_analog = analog["danceability"].mean()
avg_dance_stream = streaming["danceability"].mean()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="Avg Duration (sec)",
        value=f"{avg_dur_stream:.0f}s",
        delta=f"{avg_dur_stream - avg_dur_analog:+.0f}s vs Analog Era",
    )
with c2:
    st.metric(
        label="Avg Energy",
        value=f"{avg_energy_stream:.3f}",
        delta=f"{avg_energy_stream - avg_energy_analog:+.3f} vs Analog Era",
    )
with c3:
    st.metric(
        label="Avg Danceability",
        value=f"{avg_dance_stream:.3f}",
        delta=f"{avg_dance_stream - avg_dance_analog:+.3f} vs Analog Era",
    )

st.markdown(
    "<p class='kpi-header' style='text-align:center; margin-top:10px;'>"
    "▲ Streaming Era (2013-2020) compared with Analog Era (Pre-2000)</p>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════
#  ACT I, II & III — Chronological Story
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="era-divider">Act I · II · III — The Chronological Story</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">How <b>duration</b>, <b>energy</b>, and '
    '<b>acousticness</b> evolved year by year.</p>',
    unsafe_allow_html=True,
)

yearly = (
    df.groupby("Year")[["duration_s", "energy", "acousticness"]]
    .mean()
    .reset_index()
    .sort_values("Year")
)
# Apply rolling smoothing
for col in ["duration_s", "energy", "acousticness"]:
    yearly[col] = yearly[col].rolling(window=3, min_periods=1, center=True).mean()

era_breaks = [1999.5, 2012.5]
era_labels_pos = [
    (1978, "THE ANALOG ERA"),
    (2006, "GENRE EXPLOSION"),
    (2016, "STREAMING ERA"),
]

tab_dur, tab_energy, tab_acoustic = st.tabs(
    ["⏱ Duration", "⚡ Energy", "🎸 Acousticness"]
)

metric_map = {
    "⏱ Duration": ("duration_s", "Avg Duration (sec)", DEEP_PURPLE),
    "⚡ Energy": ("energy", "Avg Energy", NEON_GREEN),
    "🎸 Acousticness": ("acousticness", "Avg Acousticness", DEEP_PURPLE),
}

for tab, (key, (col, ylabel, color)) in zip(
    [tab_dur, tab_energy, tab_acoustic], metric_map.items()
):
    with tab:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yearly["Year"],
                y=yearly[col],
                mode="lines",
                line=dict(color=color, width=3),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
                name=ylabel,
            )
        )
        # Era break lines
        for xv in era_breaks:
            fig.add_vline(
                x=xv,
                line_dash="dash",
                line_color=BLACK,
                line_width=1.5,
                annotation_text="",
            )
        # Era annotations
        for xpos, label in era_labels_pos:
            fig.add_annotation(
                x=xpos,
                y=yearly[col].max() * 1.05,
                text=label,
                showarrow=False,
                font=dict(family="Space Mono, monospace", size=10, color=GRAY),
            )
        brutalist_layout(fig, title=ylabel + " Over Time", height=420, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  THE SONIC TENSIONS — Scatter Plot
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="era-divider">The Sonic Tensions — The Happy-Sad Paradox</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Valence (happiness) vs Energy, coloured by danceability. '
    "Do happy songs always hit hard?</p>",
    unsafe_allow_html=True,
)

era_filter = st.selectbox(
    "Filter by Era",
    options=["All Eras"] + sorted(df["Era"].unique().tolist()),
    index=0,
)

scatter_df = df if era_filter == "All Eras" else df[df["Era"] == era_filter]
# Sample for performance
if len(scatter_df) > 5000:
    scatter_df = scatter_df.sample(5000, random_state=42)

fig_scatter = px.scatter(
    scatter_df,
    x="valence",
    y="energy",
    color="danceability",
    color_continuous_scale=[DEEP_PURPLE, GRAY, NEON_GREEN],
    opacity=0.55,
    hover_data=["track_name", "track_artist", "Year"],
    labels={
        "valence": "Valence (Happiness →)",
        "energy": "Energy ↑",
        "danceability": "Danceability",
    },
)
fig_scatter.update_traces(marker=dict(size=5, line=dict(width=0)))
fig_scatter.update_coloraxes(
    colorbar=dict(
        title=dict(
            text="Danceability",
            font=dict(family="Space Mono, monospace", size=12, color=BLACK),
        ),
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=BLACK),
        outlinecolor=BLACK,
        outlinewidth=1,
        len=0.6,
    )
)
brutalist_layout(fig_scatter, title="The Happy-Sad Paradox", height=520)
st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  THE GENRE BLUR — Radar Chart
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="era-divider">The Genre Blur — Audio DNA by Genre</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Average audio profile for the top 3 most popular genres.</p>',
    unsafe_allow_html=True,
)

radar_metrics = ["danceability", "energy", "acousticness", "valence", "speechiness"]

# Top 3 genres by average track popularity
top_genres = (
    df.groupby("playlist_genre")["track_popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(3)
    .index.tolist()
)

genre_colors = [NEON_GREEN, DEEP_PURPLE, BLACK]

fig_radar = go.Figure()
for i, genre in enumerate(top_genres):
    g_df = df[df["playlist_genre"] == genre]
    vals = [g_df[m].mean() for m in radar_metrics]
    vals += vals[:1]  # close the polygon
    cats = [m.upper() for m in radar_metrics] + [radar_metrics[0].upper()]
    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            fillcolor=f"rgba({int(genre_colors[i][1:3],16)},{int(genre_colors[i][3:5],16)},{int(genre_colors[i][5:7],16)},0.12)",
            line=dict(color=genre_colors[i], width=2.5),
            name=genre.upper(),
        )
    )

fig_radar.update_layout(
    polar=dict(
        bgcolor=BG_COLOR,
        radialaxis=dict(
            visible=True,
            range=[0, 0.85],
            showgrid=False,
            linecolor=BLACK,
            tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=GRAY),
        ),
        angularaxis=dict(
            linecolor=BLACK,
            showgrid=False,
            tickfont=dict(family="Space Mono, monospace", size=12, color=BLACK),
        ),
    ),
    font=dict(family="IBM Plex Mono, monospace", size=12, color=BLACK),
    paper_bgcolor=BG_COLOR,
    height=520,
    margin=dict(l=60, r=60, t=60, b=60),
    legend=dict(
        font=dict(family="Space Mono, monospace", size=12, color=BLACK),
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BLACK,
        borderwidth=1,
    ),
    title=dict(
        text="Audio Fingerprint — Top 3 Genres",
        font=dict(family="Space Mono, monospace", size=16, color=BLACK),
        x=0.01,
    ),
)
st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  ARTIST SPOTLIGHT — Outlier Hits
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="era-divider">Artist Spotlight — The Outlier Hits</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Top 5 most popular tracks that break the mould: '
    "<b>high popularity</b> with <b>below-average energy</b> and "
    "<b>above-average acousticness</b>.</p>",
    unsafe_allow_html=True,
)

avg_energy_all = df["energy"].mean()
avg_acoustic_all = df["acousticness"].mean()

# Deduplicate songs (same track can appear in multiple playlists)
df_unique = df.drop_duplicates(subset=["track_id"])

outliers = df_unique[
    (df_unique["energy"] < avg_energy_all) & (df_unique["acousticness"] > avg_acoustic_all)
].nlargest(5, "track_popularity")

display_cols = [
    "track_name",
    "track_artist",
    "track_popularity",
    "energy",
    "acousticness",
    "valence",
    "Year",
    "Era",
]
outlier_display = outliers[display_cols].reset_index(drop=True)
outlier_display.columns = [
    "Track",
    "Artist",
    "Popularity",
    "Energy",
    "Acousticness",
    "Valence",
    "Year",
    "Era",
]

# Build a Plotly table for visual consistency
fig_table = go.Figure(
    data=[
        go.Table(
            columnwidth=[200, 140, 80, 70, 90, 70, 60, 180],
            header=dict(
                values=list(outlier_display.columns),
                fill_color=BLACK,
                font=dict(family="Space Mono, monospace", size=12, color=NEON_GREEN),
                align="left",
                line_color=BLACK,
                height=36,
            ),
            cells=dict(
                values=[outlier_display[c] for c in outlier_display.columns],
                fill_color=BG_COLOR,
                font=dict(family="IBM Plex Mono, monospace", size=12, color=BLACK),
                align="left",
                line_color=BLACK,
                height=30,
                format=[None, None, None, ".3f", ".3f", ".3f", None, None],
            ),
        )
    ]
)
fig_table.update_layout(
    paper_bgcolor=BG_COLOR,
    margin=dict(l=0, r=0, t=10, b=10),
    height=300,
)
st.plotly_chart(fig_table, use_container_width=True)

# ──────────── Fun Annotation ─────────────
if not outliers.empty:
    top_song = outliers.iloc[0]
    st.markdown(
        f"<p style='font-family:IBM Plex Mono, monospace; font-size:0.85rem; "
        f"color:#555; text-align:center;'>"
        f"🏆 <b>\"{top_song['track_name']}\"</b> by <b>{top_song['track_artist']}</b> "
        f"— Popularity {top_song['track_popularity']}, Energy {top_song['energy']:.3f}, "
        f"Acousticness {top_song['acousticness']:.3f}</p>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="footer-text">'
    "Built with Streamlit & Plotly · Data: Spotify Song Attributes · "
    "Aesthetic: Y2K Brutalist Nostalgia"
    "</p>",
    unsafe_allow_html=True,
)
