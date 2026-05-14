# ══════════════════════════════════════════════════════════════════════════════
# app.py — Churn Decision Analytics · Concesionario Automovilístico
# streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import os
import pickle
import warnings
from datetime import date
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "#1e293b",
    "axes.facecolor":    "#1e293b",
    "text.color":        "#cbd5e1",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "axes.edgecolor":    "#334155",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         True,
    "grid.alpha":        0.15,
    "grid.color":        "#334155",
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Helvetica Neue", "Arial"],
    "font.size":         10,
    "legend.facecolor":  "#0f172a",
    "legend.edgecolor":  "#334155",
    "legend.labelcolor": "#94a3b8",
})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Decision Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@700;800&family=Syne:wght@800&display=swap');

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0f172a !important;
}
.main, .stApp, section.main { background-color: #0f172a !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; }

/* Headings */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: "Manrope", "Inter", sans-serif !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em !important;
}

/* Body text in markdown containers */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stText"],
.stMarkdown p {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
    font-family: "Inter", sans-serif !important;
}

/* Labels: sliders, selects, inputs, radio */
label, [data-testid="stWidgetLabel"],
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-family: "Inter", sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.84rem; color: #94a3b8;
}
[data-testid="stSidebar"] .stButton > button {
    background: #334155 !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #38bdf8 !important;
    color: #0f172a !important;
    border-color: #38bdf8 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    font-size: 0.84rem !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p { color: #94a3b8; }

/* ── Section panels (st.container border=True) ───────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #1e293b !important;
    border: 1px solid #263348 !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.04) inset !important;
    margin-bottom: 20px !important;
}

/* ── st.metric cards ────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #0f172a;
    padding: 16px 18px;
    border-radius: 10px;
    border: 1px solid #334155;
}
[data-testid="metric-container"] { text-align: left; }
[data-testid="stMetricValue"]    { font-size: 1.45rem !important; color: #f1f5f9 !important; font-weight: 700; font-family: "Manrope", sans-serif; }
[data-testid="stMetricLabel"]    { font-size: 0.68rem; color: #64748b !important;
                                   text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricDelta"]    { font-size: 0.73rem; }

/* ── KPI cards ──────────────────────────────────────────────────────────── */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 14px; margin-bottom: 20px;
}
.kpi-card {
    background: #1e293b; padding: 22px 24px; border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25); border: 1px solid #334155;
    transition: box-shadow 0.2s, transform 0.15s;
}
.kpi-card:hover { box-shadow: 0 8px 32px rgba(56,189,248,0.12); transform: translateY(-2px); }
.kpi-label {
    color: #64748b; font-size: 0.71rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
}
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; margin: 0; font-family: "Manrope", sans-serif; }
.kpi-desc  { color: #475569; font-size: 0.71rem; margin-top: 5px; }
.kpi-card.green .kpi-value { color: #22c55e; }
.kpi-card.amber .kpi-value { color: #818cf8; }

/* ── Section headers ────────────────────────────────────────────────────── */
.sec-hdr {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 18px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 22px; margin-top: 4px;
}
.sec-num {
    background: linear-gradient(135deg, #0c4a6e, #1e3a5f);
    color: #38bdf8;
    font-size: 0.58rem; font-weight: 800;
    letter-spacing: 0.16em; text-transform: uppercase;
    padding: 5px 9px; border-radius: 8px; white-space: nowrap;
    border: 1px solid #0369a1;
}
.sec-title {
    font-size: 1.05rem; font-weight: 700; color: #f1f5f9;
    font-family: "Manrope", sans-serif; letter-spacing: -0.01em;
}
.sec-sub   { font-size: 0.72rem; color: #475569; margin-left: auto; font-weight: 400; }

/* ── Sub-section headers ────────────────────────────────────────────────── */
.subsec-hdr {
    font-size: 0.68rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-left: 2px solid #38bdf8; padding-left: 8px;
    margin: 20px 0 10px;
}

/* ── Insight panels ─────────────────────────────────────────────────────── */
.insight {
    background: #0c2340; border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 11px 14px;
    font-size: 0.82rem; color: #bae6fd; line-height: 1.65; margin-bottom: 10px;
}
.insight.green { background: #0c1a3a; border-left-color: #7dd3fc; color: #bae6fd; }
.insight.amber { background: #1e1f3a; border-left-color: #818cf8; color: #c7d2fe; }

/* ── Tarjetas de igual altura ───────────────────────────────────────────── */
.card-equal {
    height: 420px;
    overflow-y: auto;
}

/* ── Supuestos bar ──────────────────────────────────────────────────────── */
.supuestos {
    background: #0f172a; border: 1px solid #334155; border-radius: 10px;
    padding: 9px 14px; font-size: 0.77rem; color: #64748b;
    display: flex; flex-wrap: wrap; gap: 6px;
    align-items: center; margin-bottom: 20px;
}
.sup-tag {
    background: #1e293b; border: 1px solid #334155; border-radius: 5px;
    padding: 2px 8px; font-size: 0.71rem; color: #94a3b8; font-weight: 500;
}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge-alto  { background:#6d28d9; color:white; padding:2px 9px; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-medio { background:#818cf8; color:white; padding:2px 9px; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-bajo  { background:#0369a1; color:white; padding:2px 9px; border-radius:20px; font-size:0.78rem; font-weight:700; }

/* ── Cliente card ───────────────────────────────────────────────────────── */
.cliente-card {
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px; background: #1e293b;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

/* ── Implications block ─────────────────────────────────────────────────── */
.impl-block {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 12px; padding: 22px 26px;
}
.impl-ttl {
    font-size: 0.67rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;
}
.impl-item {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 12px 0; border-bottom: 1px solid #1e293b;
}
.impl-item:last-child { border-bottom: none; }
.impl-n    { font-size: 1rem; min-width: 1.5rem; }
.impl-head { font-weight: 700; color: #f1f5f9; font-size: 0.86rem; margin-bottom: 3px; }
.impl-body { color: #64748b; font-size: 0.81rem; line-height: 1.6; }

/* ── st.expander ────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background: #1e293b; border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
details[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-family: "Inter", sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
details[data-testid="stExpander"] summary:hover { color: #e2e8f0 !important; }

/* ── Inputs, selects, number inputs ────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: "Inter", sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Selectbox ──────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Slider ─────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
    border-color: #38bdf8 !important;
}


/* ── Dataframes ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] thead tr th { background: #0f172a !important; color: #94a3b8 !important; }
[data-testid="stDataFrame"] tbody tr td { color: #e2e8f0 !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background: #1e3a5f !important; }

/* ── Tabs (if any) ──────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] { color: #64748b !important; font-family: "Inter", sans-serif !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #38bdf8 !important; }

/* ── Internal dividers ──────────────────────────────────────────────────── */
hr[data-testid="stDivider"] {
    border-color: #1e293b !important;
    margin: 6px 0 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: "Inter", sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
    background: #38bdf8 !important;
    color: #0f172a !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover { background: #7dd3fc !important; }
.stButton > button[kind="secondary"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}

/* ── Expander content text ──────────────────────────────────────────────── */
details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

/* ── Footer ─────────────────────────────────────────────────────────────── */
.footer-bar {
    background: #1e293b; color: #475569;
    padding: 14px 28px; border-radius: 12px; text-align: center;
    margin-top: 24px; font-size: 0.8rem;
    border: 1px solid #334155;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
with open("pipeline_winner.pkl", "rb") as _fh:
    _meta_tmp = pickle.load(_fh)
P_TRAIN = float(_meta_tmp.get("p_train", 0.0877))
del _meta_tmp, _fh
DISC_FLOTA = 1_000.0
ALPHA_MAP  = {m: (0.07 if m in ("A", "B") else 0.10) for m in "ABCDEFGHIJK"}

C_ALTO  = "#a78bfa"   # violet-400 — gradient warm end
C_MEDIO = "#818cf8"   # indigo-400 — gradient mid
C_BAJO  = "#7dd3fc"   # sky-300   — gradient cool end
C_GRAY  = "#64748b"   # slate-500
C_BLUE  = "#38bdf8"   # sky-400   — primary accent
SEG_COLORS = {"Alto": C_ALTO, "Medio": C_MEDIO, "Bajo": C_BAJO}
INT_COLORS = {"Alta": C_ALTO, "Media": C_MEDIO, "Baja": C_BAJO, "Ninguna": C_GRAY}

ESCENARIOS = {
    "Conservador": {"p_real": 0.05,   "r": 0.15, "H": 3, "roi_min": 8.0, "budget_pct": 0.30},
    "Base":        {"p_real": 0.08,    "r": 0.10, "H": 5, "roi_min": 5.0, "budget_pct": 0.55},
    "Agresivo":    {"p_real": 0.12,   "r": 0.08, "H": 7, "roi_min": 3.0, "budget_pct": 0.70},
}

MES_ACTUAL = date.today().strftime("%B %Y").capitalize()


# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_base():
    df_pred = pd.read_csv("predicciones_nuevos_clientes.csv")
    df_raw  = pd.read_csv("nuevos_clientes.csv")[["Customer_ID", "Revisiones"]]
    costes  = pd.read_csv("Costes.csv").set_index("Modelo")
    with open("pipeline_winner.pkl", "rb") as fh:
        meta = pickle.load(fh)
    return df_pred, df_raw, costes, meta



# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────────────────────────────────────
def prior_correction(prob: np.ndarray, p_train: float, p_real: float) -> np.ndarray:
    if abs(p_train - p_real) < 1e-8:
        return prob.copy()
    odds_ratio = (p_real / (1 - p_real)) / (p_train / (1 - p_train))
    odds       = prob / np.clip(1 - prob, 1e-10, None)
    return (odds * odds_ratio) / (1 + odds * odds_ratio)


def compute_cltv_vectorized(base, alpha, n, nm_pct, p_adj, H, r):
    total = np.zeros(len(base))
    for k in range(1, H + 1):
        C_nk   = base * (1 + alpha) ** (n.astype(float) + k)
        total += (C_nk * nm_pct * (1 - p_adj) ** k) / (1 + r) ** k
    return total


def _profit_curve(prob_adj, evr, coste_mkt, thresholds):
    idx      = np.argsort(prob_adj)[::-1]
    s_prob   = prob_adj[idx]
    cs_evr   = np.concatenate([[0.0], np.cumsum(evr[idx])])
    profits  = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        k = int(np.searchsorted(-s_prob, -t, side="left"))
        profits[i] = cs_evr[k] - k * coste_mkt
    return profits


def _assign_action(seg, q):
    if seg == "Alto":
        if q == "Q4":   return ("Llamada personal + Desc. flota + Revisión gratuita", "Alta")
        elif q == "Q3": return ("Llamada personal + Desc. flota",                     "Alta")
        elif q == "Q2": return ("Email/SMS + Descuento servicio",                     "Media")
        else:           return ("Email/SMS informativo",                               "Media")
    elif seg == "Medio":
        if q in ("Q4", "Q3"): return ("Email personalizado + Desc. bajo riesgo", "Media")
        elif q == "Q2":       return ("Newsletter + Oferta estacional",           "Baja")
        else:                 return ("Newsletter",                                "Baja")
    else:
        if q in ("Q4", "Q3"): return ("Newsletter de fidelización", "Baja")
        else:                  return ("Sin acción",                 "Ninguna")


def recalculate(df_pred, df_raw, costes, p_real, H, r, roi_min, budget_pct):
    df = df_pred.copy()
    df = df.merge(df_raw, on="Customer_ID", how="left")
    df["Revisiones"] = df["Revisiones"].fillna(0).clip(lower=0).astype(int)

    df["_base"]  = df["Modelo"].map(costes["Mantenimiento_medio"]).fillna(300.0)
    df["_alpha"] = df["Modelo"].map(ALPHA_MAP).fillna(0.10)
    nm_raw       = (costes["Margen"] - costes["Comisión_Marca"] - 1.0) / 100.0
    df["_nm"]    = df["Modelo"].map(nm_raw).fillna(0.27)

    df["prob_adj"] = prior_correction(df["prob_churn"].values, P_TRAIN, p_real)

    df["cltv"] = np.round(
        compute_cltv_vectorized(
            df["_base"].values, df["_alpha"].values, df["Revisiones"].values,
            df["_nm"].values, df["prob_adj"].values, H, r,
        ), 2
    )

    # C(n) = coste real en la revisión actual (compounding)
    df["_cn"]         = df["_base"] * (1 + df["_alpha"]) ** df["Revisiones"]
    # Margen absoluto sobre C(n): misma base que _coste_mkt → ROI coherente
    df["_margen_abs"] = df["_cn"] * df["_nm"]
    # EVR = prob × margen(C(n)) — versión probabilística de TP×Margen
    df["evr"] = np.round(np.maximum(df["prob_adj"] * df["_margen_abs"], 0.0), 2)

    df["_coste_mkt"] = 0.01 * df["_cn"]
    coste_mkt_mean   = float(df["_coste_mkt"].mean())

    thresholds = np.arange(0.001, 1.000, 0.001)
    profits    = _profit_curve(df["prob_adj"].values, df["evr"].values,
                               coste_mkt_mean, thresholds)
    best_idx         = int(np.argmax(profits))
    thresh_econ      = float(thresholds[best_idx])
    pos_mask         = profits > 0
    thresh_breakeven = float(thresholds[pos_mask][0]) if pos_mask.any() else thresh_econ

    n_total  = len(df)
    n_alto   = int(n_total * budget_pct * 0.35)
    n_accion = int(n_total * budget_pct)

    rank = df["prob_adj"].rank(method="first", ascending=False).astype(int)
    seg  = pd.Series("Bajo", index=df.index)
    seg[rank <= n_accion] = "Medio"
    seg[rank <= n_alto]   = "Alto"
    df["segmento"] = seg

    try:
        df["_q"] = pd.qcut(df["cltv"], q=4, labels=["Q1","Q2","Q3","Q4"],
                           duplicates="drop").astype(str)
    except Exception:
        df["_q"] = "Q2"

    acc = df.apply(lambda r: _assign_action(r["segmento"], r["_q"]), axis=1)
    df["accion"]     = acc.map(lambda x: x[0])
    df["intensidad"] = acc.map(lambda x: x[1])

    roi_contacto = np.where(df["_coste_mkt"] > 0, df["evr"] / df["_coste_mkt"], 0.0)
    roi_excl = (df["segmento"].isin(["Alto", "Medio"])) & (roi_contacto < roi_min)
    df.loc[roi_excl, "segmento"]   = "Bajo"
    df.loc[roi_excl, "accion"]     = "Sin acción"
    df.loc[roi_excl, "intensidad"] = "Ninguna"

    df["_disc_flota"]  = np.where(
        (df["segmento"] == "Alto") & (df["_q"].isin(["Q3","Q4"])), DISC_FLOTA, 0.0
    )
    df["_coste_total"] = df["_coste_mkt"] + df["_disc_flota"]
    df["roi"]          = np.where(df["_coste_total"] > 0,
                                  df["evr"] / df["_coste_total"], 0.0)

    p_range  = np.linspace(0.005, 0.20, 40)
    thrs_c   = np.linspace(0.001, 0.999, 100)
    sens_ben = []
    for p_s in p_range:
        p_adj_s = prior_correction(df["prob_churn"].values, P_TRAIN, p_s)
        evr_s   = np.maximum(p_adj_s * df["_margen_abs"].values, 0.0)
        profs_s = _profit_curve(p_adj_s, evr_s, coste_mkt_mean, thrs_c)
        sens_ben.append(float(profs_s.max()))

    return {
        "df":          df,
        "te":          thresh_econ,
        "tb":          thresh_breakeven,
        "thresholds":  thresholds,
        "profits":     profits,
        "c_mkt":       coste_mkt_mean,
        "sensitivity": (p_range, np.array(sens_ben)),
        "p_real": p_real, "H": H, "r": r,
        "roi_min": roi_min, "budget_pct": budget_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────
def section(num, title, subtitle=None):
    sub = f"<span class='sec-sub'>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f"<div class='sec-hdr'>"
        f"<span class='sec-num'>{num}</span>"
        f"<span class='sec-title'>{title}</span>"
        f"{sub}"
        f"</div>",
        unsafe_allow_html=True,
    )


def subsection(title):
    st.markdown(f"<div class='subsec-hdr'>{title}</div>", unsafe_allow_html=True)


def badge(seg):
    cls = {"Alto": "badge-alto", "Medio": "badge-medio", "Bajo": "badge-bajo"}.get(seg, "")
    return f"<span class='{cls}'>{seg}</span>"


def kpi_cards(items):
    """items = list of (label, value, desc, variant)  — variant: '' | 'green' | 'amber' | 'navy'"""
    cards = ""
    for label, value, desc, variant in items:
        cards += (
            f"<div class='kpi-card {variant}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-desc'>{desc}</div>"
            f"</div>"
        )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def insight(text, variant=""):
    cls = f"insight {variant}" if variant else "insight"
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header(state, escenario, meta):
    df        = state["df"]
    evr_total = float(df["evr"].sum())
    roi_medio = float(df.loc[df["roi"] > 0, "roi"].mean() or 0)
    n_accion  = int((df["accion"] != "Sin acción").sum())

    st.markdown(
        f"""<div style="padding:28px 0 20px 0">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
    <div style="width:4px;align-self:stretch;background:#38bdf8;border-radius:2px;flex-shrink:0"></div>
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#38bdf8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:3px;font-family:'Inter',sans-serif">
        Concesionario &middot; Retención de clientes
      </div>
      <div style="font-size:2.8rem;font-weight:800;letter-spacing:-0.03em;line-height:1.1;font-family:'Syne','Manrope','Inter',sans-serif;background:linear-gradient(90deg,#7dd3fc 0%,#38bdf8 40%,#818cf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
        Churn Decision Analytics
      </div>
    </div>
  </div>
  <p style="color:#94a3b8;font-size:0.92rem;margin:0 0 16px 14px;max-width:640px;font-family:'Inter',sans-serif">
    Optimización económica de campañas de retención mediante modelos predictivos
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-left:14px">
    <span style="background:#0c4a6e;color:#38bdf8;border:1px solid #0369a1;border-radius:6px;padding:3px 10px;font-size:0.71rem;font-weight:700;letter-spacing:0.06em">
      {escenario.upper()}
    </span>
    <span style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:3px 10px;font-size:0.71rem">
      {MES_ACTUAL}
    </span>
    <span style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:3px 10px;font-size:0.71rem">
      {meta['winner_name']}
    </span>
    <span style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:3px 10px;font-size:0.71rem">
      AUC&nbsp;<b style="color:#e2e8f0">{meta['auc_test']:.4f}</b>
    </span>
    <span style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:3px 10px;font-size:0.71rem">
      <b style="color:#e2e8f0">{len(df):,}</b>&nbsp;clientes
    </span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    kpi_cards([
        ("EVR Total en Riesgo",   f"€{evr_total:,.0f}", "ingresos futuros en riesgo",   ""),
        ("ROI Medio Intervención",f"{roi_medio:.1f}x",  "retorno por € invertido",       "green"),
        ("Umbral Óptimo",         f"{state['te']:.3f}", "maximiza beneficio de campaña", ""),
        ("Clientes Accionables",  f"{n_accion:,}",      "con intervención planificada",  "amber"),
    ])

    te    = state["te"]
    pills = (
        f"<span class='sup-tag'>Churn: <b>{state['p_real']*100:.1f}%</b></span>"
        f"<span class='sup-tag'>Horizonte: <b>{state['H']} rev.</b></span>"
        f"<span class='sup-tag'>Tasa desc.: <b>{state['r']:.0%}</b></span>"
        f"<span class='sup-tag'>ROI mín.: <b>{state['roi_min']:.1f}x</b></span>"
        f"<span class='sup-tag'>Presupuesto: <b>{state['budget_pct']*100:.0f}% cartera</b></span>"
        f"<span class='sup-tag'>Umbral óptimo: <b>{te:.3f}</b></span>"
    )
    st.markdown(
        "<div style='margin:10px 0 8px 0;padding:10px 16px;background:#0c2340;"
        "border-left:3px solid #38bdf8;border-radius:0 8px 8px 0;"
        "font-size:0.82rem;color:#bae6fd;line-height:1.5;'>"
        "El modelo no maximiza cobertura: maximiza retorno. "
        "Contactar al segmento correcto con la acción correcta multiplica el ROI frente a una campaña masiva."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin:4px 0 8px 0;padding:10px 16px;background:#0c2340;"
        "border-left:3px solid #38bdf8;border-radius:0 8px 8px 0;"
        "font-size:0.82rem;color:#bae6fd;line-height:1.5;'>"
        "El umbral óptimo se sitúa bajo al maximizar valor esperado: el bajo coste permite intervenir con probabilidades pequeñas."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='supuestos'>&#128204;&nbsp;<b>Supuestos activos</b>&nbsp;&nbsp;{pills}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 01 — MODELO PREDICTIVO
# ─────────────────────────────────────────────────────────────────────────────
def render_modelo(state, meta):
    df       = state["df"]
    AUC      = float(meta["auc_test"])
    top_n    = max(1, int(len(df) * 0.10))
    top_df   = df.nlargest(top_n, "prob_adj")
    lift_10  = float(top_df["prob_adj"].mean()) / max(float(df["prob_adj"].mean()), 1e-9)
    gains_10 = float(top_df["evr"].sum()) / max(float(df["evr"].sum()), 1e-9) * 100

    # ── Fila 1: métricas en ancho completo
    m1, m2, m3, _ = st.columns([1, 1, 1, 3])
    m1.metric("AUC-ROC",      f"{AUC:.4f}")
    m2.metric("Lift Top 10%", f"{lift_10:.2f}x")
    m3.metric("Ganancia 10%", f"{gains_10:.1f}%",
              help="% del EVR total capturado en el 10% con mayor riesgo")

    # ── Fila 2: imagen ROC/Gains/Lift
    img_roc = "img/roc_gains_lift_dashboard.png"
    if os.path.exists(img_roc):
        st.image(img_roc, use_container_width=True)
    else:
        st.info("Exporta `img/roc_gains_lift_dashboard.png` desde el notebook (Sección 7) y recarga.")

    # ── Fila 3: distribución de scores a ancho completo
    subsection("Distribución de scores — original vs ajustado al prior real")
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.hist(df["prob_churn"].values, bins=80, color=C_BLUE, alpha=0.5,
            edgecolor="white", lw=0.3, label="Score original")
    ax.hist(df["prob_adj"].values, bins=80, color=C_ALTO, alpha=0.45,
            edgecolor="white", lw=0.3,
            label=f"Ajustado al prior real ({state['p_real']*100:.1f}%)")
    ax.axvline(state["te"], color=C_ALTO, ls="--", lw=1.4,
               label=f"Umbral óptimo {state['te']:.3f}")
    ax.set_xlabel("Probabilidad de Churn")
    ax.set_ylabel("Clientes")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Importancia de variables: gráfico a ancho completo + tarjetas debajo
    subsection("Importancia de variables")

    # Categorías estratégicas por variable (para colorear barras)
    _CAT_COLORS = {
        # ── Cliente / Perfil sociodemográfico ────────────────────────────────
        "Edad":                       ("#a78bfa", "Cliente"),
        "RENTA_MEDIA_ESTIMADA":       ("#a78bfa", "Cliente"),
        "GENERO":                     ("#a78bfa", "Cliente"),
        "STATUS_SOCIAL":              ("#a78bfa", "Cliente"),
        "PROV_DESC":                  ("#a78bfa", "Cliente"),
        "ZONA":                       ("#a78bfa", "Cliente"),
        "es_premium":                 ("#a78bfa", "Cliente"),
        "MOTIVO_VENTA":               ("#a78bfa", "Cliente"),
        # ── Producto / Vehículo ──────────────────────────────────────────────
        "PVP":                        ("#7dd3fc", "Producto"),
        "pvp_por_kw":                 ("#7dd3fc", "Producto"),
        "Kw":                         ("#7dd3fc", "Producto"),
        "Modelo":                     ("#7dd3fc", "Producto"),
        "TIPO_CARROCERIA":            ("#7dd3fc", "Producto"),
        "Equipamiento":               ("#7dd3fc", "Producto"),
        "Fuel":                       ("#7dd3fc", "Producto"),
        "Origen":                     ("#7dd3fc", "Producto"),
        # ── Servicio / Taller ────────────────────────────────────────────────
        "ENCUESTA_CLIENTE_ZONA_TALLER": ("#38bdf8", "Servicio"),
        "MANTENIMIENTO_GRATUITO":     ("#38bdf8", "Servicio"),
        "DAYS_LAST_SERVICE":          ("#38bdf8", "Servicio"),
        # ── Financiero / Contractual ─────────────────────────────────────────
        "extension_garantia_bin":     ("#818cf8", "Financiero"),
        "SEGURO_BATERIA_LARGO_PLAZO": ("#818cf8", "Financiero"),
        "FORMA_PAGO":                 ("#818cf8", "Financiero"),
    }
    _DEFAULT_COLOR = "#94a3b8"

    # Fila 1: gráfico a ancho completo
    try:
        clf  = meta["model"].steps[-1][1]
        prep = meta["model"].steps[0][1]
        fn   = prep.get_feature_names_out()
        imp  = clf.feature_importances_

        def _orig(name):
            if "__" not in name:
                return name
            _, rest = name.split("__", 1)
            for f in meta.get("cat_feats", []):
                if rest.startswith(f):
                    return f
            return rest

        fi_df = (
            pd.DataFrame({"feature": [_orig(n) for n in fn], "importance": imp})
            .groupby("feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False)
            .head(10)
            .sort_values("importance", ascending=True)
        )

        bar_colors  = [_CAT_COLORS.get(f, (_DEFAULT_COLOR, "Otro"))[0] for f in fi_df["feature"]]

        fig_fi, ax_fi = plt.subplots(figsize=(11, 3.8))
        bars = ax_fi.barh(fi_df["feature"], fi_df["importance"],
                          color=bar_colors, height=0.55)
        ax_fi.bar_label(bars, fmt="%.3f", padding=5,
                        fontsize=8.5, color="#6b7280")
        ax_fi.set_xlabel("Importancia (gain agregado)", fontsize=8.5)
        ax_fi.set_title("Top 10 variables más influyentes — coloreadas por dimensión estratégica",
                        fontsize=10, fontweight="bold", pad=12, color="#cbd5e1")
        ax_fi.tick_params(labelsize=9)
        ax_fi.spines[["top", "right", "left"]].set_visible(False)
        ax_fi.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax_fi.set_axisbelow(True)

        # Leyenda de categorías
        from matplotlib.patches import Patch
        legend_items = {
            "Servicio":   "#38bdf8",
            "Producto":   "#7dd3fc",
            "Financiero": "#818cf8",
            "Cliente":    "#a78bfa",
        }
        ax_fi.legend(
            handles=[Patch(facecolor=c, label=l) for l, c in legend_items.items()],
            fontsize=8, frameon=False, loc="lower right",
            ncol=4, handlelength=1.2, handleheight=0.9,
        )

        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)
    except Exception:
        if os.path.exists("img/feature_importance.png"):
            st.image("img/feature_importance.png", width="stretch")
        else:
            st.info("No se pudo generar el gráfico de importancia de variables.")

    # Fila 2: dos tarjetas lado a lado
    card_col1, card_col2 = st.columns(2)

    with card_col1:
        st.markdown("""
<div class="card-equal" style="margin-top:14px;background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;font-size:0.78rem;color:#cbd5e1;line-height:1.5;">
<h4 style="margin:0 0 14px 0;font-size:0.8rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.04em;">Qué aprendió el modelo sobre el churn</h4>
<p style="margin-bottom:12px;"><b style="color:#f1f5f9;">① Perfil sociodemográfico</b><br>Edad y renta estimada son las variables más predictivas. El abandono no es uniforme: varía estructuralmente por segmento de cliente.<br><span style="color:#38bdf8;font-weight:600;">→ Las campañas deben segmentarse por perfil, no solo por producto.</span></p>
<p style="margin-bottom:12px;"><b style="color:#f1f5f9;">② Segmento del vehículo (PVP y precio/potencia)</b><br>El precio del vehículo y su relación con la potencia capturan el segmento al que pertenece el cliente. Es el predictor más fuerte tras la demografía.<br><span style="color:#38bdf8;font-weight:600;">→ Entry, mid y premium tienen dinámicas de churn distintas.</span></p>
<p style="margin-bottom:12px;"><b style="color:#f1f5f9;">③ Calidad del entorno de servicio local</b><br>La satisfacción media del taller de zona influye en la retención. Las áreas con mayor valoración generan más fidelidad.<br><span style="color:#38bdf8;font-weight:600;">→ La experiencia del taller local importa tanto como el producto.</span></p>
<p style="margin-bottom:12px;"><b style="color:#f1f5f9;">④ Compromiso contractual de largo plazo</b><br>La extensión de garantía y el mantenimiento gratuito reducen el riesgo de abandono. El vínculo contractual predice lealtad.<br><span style="color:#38bdf8;font-weight:600;">→ Extender coberturas es más eficaz que descuentos puntuales.</span></p>
<p style="margin-bottom:0;"><b style="color:#f1f5f9;">⑤ Forma de pago y motivación de compra</b><br>Contado vs. financiado, y si la compra fue particular o no, diferencian patrones de fidelidad.<br><span style="color:#38bdf8;font-weight:600;margin-top:12px;display:block;">→ El canal y la motivación de compra condicionan la relación posterior.</span></p>
</div>
""", unsafe_allow_html=True)

    with card_col2:
        st.markdown("""
<div class="card-equal" style="margin-top:14px;background:#1e293b;border-radius:12px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,0.3);border-left:4px solid #38bdf8;font-size:0.78rem;color:#cbd5e1;line-height:1.5;">
<h4 style="margin:0 0 14px 0;font-size:0.8rem;font-weight:700;color:#f1f5f9;text-transform:uppercase;letter-spacing:0.04em;">Qué NO explica el modelo</h4>
<p style="margin-bottom:10px;"><b style="color:#f1f5f9;">Quejas individuales</b><br>QUEJA tiene un 57% de nulos y no figura entre las variables relevantes. El modelo no captura insatisfacción declarada porque no hay señal suficiente en los datos.</p>
<p style="margin-bottom:10px;"><b style="color:#f1f5f9;">Historial de visitas previas</b><br>DAYS_LAST_SERVICE tiene un 47% de nulos — la mayoría de clientes no tiene historial de servicio registrado. El modelo no puede usar esta señal de forma fiable.</p>
<p style="margin-bottom:10px;"><b style="color:#f1f5f9;">Antigüedad de la relación</b><br>La duración del vínculo con el concesionario no es un predictor autónomo fuerte. Lo que importa es el perfil del cliente, no cuánto tiempo lleva.</p>
<p style="margin-bottom:0;padding-top:10px;border-top:1px solid #334155;font-weight:700;color:#38bdf8;">El abandono es estructural (quién es el cliente y qué compró), no reactivo (qué experiencia tuvo).</p>
</div>
""", unsafe_allow_html=True)

    # ── Expander: métricas del modelo
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    with st.expander("Interpretación del modelo"):
        insight(
            f"<b>{meta['winner_name']}</b><br>"
            f"El modelo discrimina churn con AUC <b>{AUC:.4f}</b>.<br><br>"
            f"El <b>top 10%</b> de mayor riesgo concentra el <b>{gains_10:.1f}%</b> "
            f"del valor en riesgo, con lift <b>{lift_10:.2f}x</b> sobre selección aleatoria."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 02 — OPTIMIZACIÓN ECONÓMICA
# ─────────────────────────────────────────────────────────────────────────────
def render_economia(state):
    df      = state["df"]
    te      = state["te"]
    tb      = state["tb"]
    thrs    = state["thresholds"]
    profits = state["profits"]
    c_mkt   = state["c_mkt"]
    p_real  = state["p_real"]
    p_range, sens = state["sensitivity"]

    mask_alto = df["prob_adj"] >= te
    n_total   = len(df)
    n_con     = int(mask_alto.sum())
    evr_all   = float(df["evr"].sum())
    evr_con   = float(df.loc[mask_alto, "evr"].sum())
    cost_sin  = n_total * c_mkt
    cost_con  = n_con * c_mkt
    ben_sin   = evr_all - cost_sin
    ben_con   = evr_con - cost_con
    roi_sin   = evr_all / cost_sin if cost_sin > 0 else 0
    roi_con   = evr_con / cost_con if cost_con > 0 else 0

    col_crv, col_info = st.columns([3, 2])

    with col_crv:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(thrs, profits / 1_000, color=C_BLUE, lw=2)
        ax.axvline(te, color=C_ALTO,  ls="--", lw=1.5, label=f"Umbral óptimo {te:.3f}")
        ax.axvline(tb, color=C_MEDIO, ls="--", lw=1.2, label=f"Break-even {tb:.4f}")
        ax.axhline(0, color="#cccccc", lw=1)
        ax.fill_between(thrs, profits/1_000, 0, where=profits>0, alpha=0.07, color=C_BAJO)
        ax.set_xlabel("Umbral de clasificación")
        ax.set_ylabel("Beneficio neto (k€)")
        ax.set_title("Beneficio neto vs umbral de decisión", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        insight(
            "<b>¿Qué optimiza este gráfico?</b><br>"
            "El umbral controla cuántos clientes reciben la campaña. "
            "Seleccionar un umbral muy bajo contacta a clientes con bajo valor en riesgo. "
            "El umbral óptimo <b>maximiza el beneficio neto esperado</b> de la intervención."
        )
        r1, r2 = st.columns(2)
        r1.metric("Clientes objetivo",  f"{n_con:,}",           f"de {n_total:,} total")
        r2.metric("Ahorro en campaña",  f"€{cost_sin-cost_con:,.0f}")
        r1.metric("EVR capturado",      f"€{evr_con:,.0f}")
        r2.metric("Beneficio neto",     f"€{ben_con:,.0f}")

    st.divider()
    col_sens, col_comp = st.columns(2)

    with col_sens:
        subsection("Sensibilidad al churn rate real")
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        ax2.plot(p_range * 100, np.array(sens) / 1_000, color=C_BLUE, lw=2)
        ax2.axvline(p_real * 100, color=C_ALTO, ls="--", lw=1.4,
                    label=f"Prior actual ({p_real*100:.1f}%)")
        ax2.axhline(0, color="#cccccc", lw=1)
        ax2.set_xlabel("Churn rate real (%)")
        ax2.set_ylabel("Beneficio máximo (k€)")
        ax2.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_comp:
        subsection("Sin modelo vs con modelo")
        comp = pd.DataFrame({
            "Escenario":      ["Sin modelo (target all)", f"Con modelo (≥ {te:.3f})"],
            "Clientes":       [f"{n_total:,}", f"{n_con:,}"],
            "EVR capturado":  [f"€{evr_all:,.0f}", f"€{evr_con:,.0f}"],
            "Coste campaña":  [f"€{cost_sin:,.0f}", f"€{cost_con:,.0f}"],
            "Beneficio neto": [f"€{ben_sin:,.0f}", f"€{ben_con:,.0f}"],
            "ROI":            [f"{roi_sin:.1f}x", f"{roi_con:.1f}x"],
        })
        st.dataframe(comp, width="stretch", hide_index=True)
        insight(
            f"El modelo ahorra <b>€{cost_sin-cost_con:,.0f}</b> en costes de campaña "
            f"y mejora el beneficio neto en <b>€{ben_con-ben_sin:,.0f}</b>.",
            variant="green"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 03 — SEGMENTACIÓN ESTRATÉGICA DE CLIENTES
# ─────────────────────────────────────────────────────────────────────────────
def render_segmentacion(state):
    df = state["df"]
    te = state["te"]
    tb = state["tb"]

    col_scatter, col_stats = st.columns([3, 2])

    with col_scatter:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for seg in ["Bajo", "Medio", "Alto"]:
            m  = (df["segmento"] == seg) & (df["cltv"] >= 0)
            mn = (df["segmento"] == seg) & (df["cltv"] < 0)
            ax.scatter(df.loc[m,  "prob_adj"], df.loc[m,  "cltv"],
                       c=SEG_COLORS[seg], alpha=0.28, s=7, rasterized=True)
            if mn.any():
                ax.scatter(df.loc[mn, "prob_adj"], df.loc[mn, "cltv"],
                           c=C_GRAY, alpha=0.20, s=7, rasterized=True)
        ax.axvline(te, color=C_ALTO,  ls="--", lw=1.2, alpha=0.8,
                   label=f"Umbral óptimo {te:.3f}")
        ax.axvline(tb, color=C_MEDIO, ls="--", lw=1.2, alpha=0.8,
                   label=f"Break-even {tb:.4f}")
        ax.axhline(0, color="#cccccc", ls="-", lw=0.8)
        handles = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor=SEG_COLORS[s],
                   markersize=7, label=s)
            for s in ["Alto", "Medio", "Bajo"]
        ] + [Line2D([0],[0], marker="o", color="w", markerfacecolor=C_GRAY,
                    markersize=7, label="CLTV < 0")]
        ax.legend(handles=handles, fontsize=8, frameon=False)
        ax.set_xlabel("Probabilidad de Churn (ajustada)")
        ax.set_ylabel("CLTV (€)")
        ax.set_title("Customer Value vs Churn Risk", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        insight(
            "· <b>Retención agresiva (Alto)</b>: alta probabilidad de churn, alto valor — prioridad máxima<br>"
            "· <b>Fidelización (Medio)</b>: riesgo moderado, valor significativo — acción planificada<br>"
            "· <b>Sin acción (Bajo)</b>: bajo riesgo o CLTV negativo — no justifica inversión"
        )

    with col_stats:
        resumen = (
            df.groupby("segmento")
            .agg(N=("Customer_ID","count"),
                 Prob_media=("prob_adj","mean"),
                 CLTV_medio=("cltv","mean"),
                 EVR_total=("evr","sum"),
                 ROI_medio=("roi","mean"))
            .reindex(["Alto","Medio","Bajo"])
            .reset_index()
        )
        resumen.insert(2, "% total", resumen["N"] / resumen["N"].sum() * 100)

        fmt = resumen.copy()
        fmt["N"]          = fmt["N"].map("{:,}".format)
        fmt["% total"]    = fmt["% total"].map("{:.1f}%".format)
        fmt["Prob_media"] = fmt["Prob_media"].map("{:.3f}".format)
        fmt["CLTV_medio"] = fmt["CLTV_medio"].map("€{:.0f}".format)
        fmt["EVR_total"]  = fmt["EVR_total"].map("€{:,.0f}".format)
        fmt["ROI_medio"]  = fmt["ROI_medio"].map("{:.1f}x".format)
        fmt.columns = ["Segmento","N","% Total","Prob. media","CLTV medio","EVR total","ROI medio"]

        st.markdown("**Estadísticas por segmento**")
        st.dataframe(fmt, width="stretch", hide_index=True)

        evr_seg = df.groupby("segmento")["evr"].sum().reindex(["Alto","Medio","Bajo"]).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        bars = ax2.bar(evr_seg.index, evr_seg.values,
                       color=[SEG_COLORS[s] for s in evr_seg.index],
                       width=0.5, edgecolor="white")
        ax2.bar_label(bars, fmt="€{:,.0f}", padding=3, fontsize=8)
        ax2.set_ylabel("EVR (€)")
        ax2.set_ylim(0, float(evr_seg.max()) * 1.2)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"€{x:,.0f}"))
        ax2.set_title("EVR por segmento", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    subsection("Concentración de valor en riesgo — Curva de Lorenz")

    df_sorted = df.sort_values("prob_adj", ascending=False).reset_index(drop=True)
    n         = len(df_sorted)
    evr_total = float(df["evr"].sum())
    pct_cli   = np.arange(1, n + 1) / n * 100
    cum_evr   = (np.cumsum(df_sorted["evr"].values) / evr_total * 100
                 if evr_total > 0 else np.zeros(n))

    top10_idx = max(1, int(n * 0.10)) - 1
    top20_idx = max(1, int(n * 0.20)) - 1
    evr_top10 = float(cum_evr[top10_idx])
    evr_top20 = float(cum_evr[top20_idx])

    col_chart, col_conc = st.columns([3, 1])

    with col_chart:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(pct_cli, cum_evr, color=C_BLUE, lw=2.5, label="EVR acumulado")
        ax3.plot([0, 100], [0, 100], color=C_GRAY, ls="--", lw=1, label="Distribución uniforme")
        ax3.fill_between(pct_cli, cum_evr, pct_cli, alpha=0.08, color=C_BLUE)

        ax3.axvline(10, color=C_ALTO, ls=":", lw=1.5, alpha=0.8)
        ax3.axhline(evr_top10, color=C_ALTO, ls=":", lw=1.5, alpha=0.8)
        ax3.annotate(
            f"Top 10%: {evr_top10:.0f}% del EVR",
            xy=(10, evr_top10), xytext=(38, 18), fontsize=8, color=C_ALTO,
            arrowprops=dict(arrowstyle="->", color=C_ALTO, lw=1,
                            connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec=C_ALTO, alpha=0.85, lw=0.8),
        )
        ax3.axvline(20, color=C_MEDIO, ls=":", lw=1.2, alpha=0.7)
        ax3.axhline(evr_top20, color=C_MEDIO, ls=":", lw=1.2, alpha=0.7)
        ax3.annotate(
            f"Top 20%: {evr_top20:.0f}% del EVR",
            xy=(20, evr_top20), xytext=(55, 38), fontsize=8, color=C_MEDIO,
            arrowprops=dict(arrowstyle="->", color=C_MEDIO, lw=1,
                            connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec=C_MEDIO, alpha=0.85, lw=0.8),
        )
        ax3.set_xlabel("% Clientes (ordenados por riesgo de churn, desc.)")
        ax3.set_ylabel("% EVR acumulado")
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8, frameon=False)
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_conc:
        st.markdown("**Concentración**")
        st.metric("EVR en top 10%", f"{evr_top10:.0f}%",
                  help="% del valor concentrado en el 10% de mayor riesgo")
        st.metric("EVR en top 20%", f"{evr_top20:.0f}%",
                  help="% del valor concentrado en el 20% de mayor riesgo")
        n_accion = int((df["accion"] != "Sin acción").sum())
        st.metric("Cartera intervenida", f"{n_accion / n * 100:.0f}%",
                  help="% de clientes con intervención planificada")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — ESTRATEGIA DE CAMPAÑAS
# ─────────────────────────────────────────────────────────────────────────────
def render_campanas(state):
    df = state["df"]

    col_plan, col_logic = st.columns([3, 2])

    with col_plan:
        plan = (
            df.groupby(["accion", "intensidad"])
            .agg(
                Clientes=("Customer_ID", "count"),
                EVR_total=("evr", "sum"),
                ROI_medio=("roi", "mean"),
            )
            .reset_index()
        )
        plan["% Total"] = plan["Clientes"] / len(df) * 100
        int_order       = {"Alta": 0, "Media": 1, "Baja": 2, "Ninguna": 3}
        plan["_sort"]   = plan["intensidad"].map(int_order)
        plan            = plan.sort_values("_sort").drop("_sort", axis=1)

        plan_fmt = plan.copy()
        plan_fmt["Clientes"]  = plan_fmt["Clientes"].map("{:,}".format)
        plan_fmt["% Total"]   = plan_fmt["% Total"].map("{:.1f}%".format)
        plan_fmt["EVR_total"] = plan_fmt["EVR_total"].map("€{:,.0f}".format)
        plan_fmt["ROI_medio"] = plan_fmt["ROI_medio"].map("{:.1f}x".format)
        plan_fmt.columns = ["Acción", "Intensidad", "Clientes", "EVR Total", "ROI Medio", "% Total"]
        plan_fmt = plan_fmt[["Acción", "Clientes", "% Total", "EVR Total", "ROI Medio", "Intensidad"]]

        st.dataframe(plan_fmt, width="stretch", hide_index=True)
        st.caption(
            "**EVR Total** = ingresos en riesgo capturable por acción  ·  "
            "**ROI Medio** = retorno esperado por € invertido"
        )

    with col_logic:
        st.markdown("**Lógica de decisión**")
        insight(
            "<b>Segmento Alto — Retención agresiva</b><br>"
            "· Q4: Llamada personal + Desc. flota + Revisión gratuita<br>"
            "· Q3: Llamada personal + Descuento flota<br>"
            "· Q2: Email/SMS + Descuento servicio"
        )
        insight(
            "<b>Segmento Medio — Fidelización planificada</b><br>"
            "· Q4/Q3: Email personalizado + Descuento bajo riesgo<br>"
            "· Q2: Newsletter + Oferta estacional",
            variant="amber"
        )
        insight(
            "<b>Segmento Bajo — Acción mínima</b><br>"
            "· Q4/Q3: Newsletter de fidelización<br>"
            "· Q1/Q2: Sin acción",
            variant="green"
        )

    st.divider()
    with st.expander("Datos detallados por cliente", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        sel_seg  = f1.multiselect("Segmento", ["Alto","Medio","Bajo"],
                                   default=["Alto","Medio","Bajo"])
        sel_mod  = f2.multiselect("Modelo",   sorted(df["Modelo"].unique()),
                                   default=sorted(df["Modelo"].unique()))
        roi_min  = f3.slider("ROI mínimo",  0.0, float(df["roi"].max()), 0.0, step=0.5)
        cltv_min = f4.slider("CLTV mín. (€)",
                              float(df["cltv"].min()), float(df["cltv"].max()), 0.0, step=10.0)

        COLS = ["Customer_ID","Modelo","Edad","ZONA","prob_adj","segmento",
                "cltv","evr","roi","accion","intensidad"]
        mask = (
            df["segmento"].isin(sel_seg) & df["Modelo"].isin(sel_mod)
            & (df["roi"] >= roi_min) & (df["cltv"] >= cltv_min)
        )
        df_f = df.loc[mask, COLS].copy()
        df_f.columns = ["ID","Modelo","Edad","Zona","Prob.","Segmento",
                         "CLTV (€)","EVR (€)","ROI","Acción","Intensidad"]
        df_f["Prob."] = df_f["Prob."].round(4)

        st.caption(f"Mostrando **{len(df_f):,}** clientes tras aplicar filtros.")
        st.dataframe(
            df_f, width="stretch", hide_index=True,
            column_config={
                "Prob.":    st.column_config.ProgressColumn(
                                "Prob.", min_value=0.0, max_value=1.0, format="%.4f"),
                "CLTV (€)": st.column_config.NumberColumn("CLTV (€)", format="€%.2f"),
                "EVR (€)":  st.column_config.NumberColumn("EVR (€)",  format="€%.2f"),
            },
        )
        st.download_button(
            "⬇️  Exportar selección",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name=f"plan_comercial_{MES_ACTUAL.replace(' ','_')}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 05 — ANÁLISIS ESTRATÉGICO
# ─────────────────────────────────────────────────────────────────────────────
def render_estrategia(state):
    df    = state["df"]
    te    = state["te"]
    c_mkt = state["c_mkt"]

    n_total   = len(df)
    evr_all   = float(df["evr"].sum())
    roi_medio = float(df.loc[df["roi"] > 0, "roi"].mean() or 0)

    cost_all = n_total * c_mkt
    ben_all  = evr_all - cost_all
    roi_all  = evr_all / cost_all if cost_all > 0 else 0

    mask_ml  = df["prob_adj"] >= te
    n_ml     = int(mask_ml.sum())
    evr_ml   = float(df.loc[mask_ml, "evr"].sum())
    cost_ml  = n_ml * c_mkt
    ben_ml   = evr_ml - cost_ml
    roi_ml   = evr_ml / cost_ml if cost_ml > 0 else 0

    n_top5    = max(1, int(n_total * 0.05))
    top5_idx  = df["prob_adj"].nlargest(n_top5).index
    evr_top5  = float(df.loc[top5_idx, "evr"].sum())
    cost_top5 = n_top5 * c_mkt
    ben_top5  = evr_top5 - cost_top5
    roi_top5  = evr_top5 / cost_top5 if cost_top5 > 0 else 0

    subsection("Simulación comparativa de enfoques")
    col_tbl, col_charts = st.columns([2, 3])

    with col_tbl:
        ahorro_pct = (1 - cost_ml / cost_all) * 100 if cost_all > 0 else 0
        evr_ml_pct = evr_ml / evr_all * 100 if evr_all > 0 else 0

        sim = pd.DataFrame({
            "Estrategia":    ["Sin modelo (target all)",
                              f"Modelo ML (≥ {te:.3f})", "Top 5% riesgo"],
            "Clientes":      [f"{n_total:,}", f"{n_ml:,}", f"{n_top5:,}"],
            "Coste campaña": [f"€{cost_all:,.0f}", f"€{cost_ml:,.0f}", f"€{cost_top5:,.0f}"],
            "EVR capturado": [f"€{evr_all:,.0f}", f"€{evr_ml:,.0f}", f"€{evr_top5:,.0f}"],
            "Beneficio":     [f"€{ben_all:,.0f}", f"€{ben_ml:,.0f}", f"€{ben_top5:,.0f}"],
            "ROI":           [f"{roi_all:.1f}x", f"{roi_ml:.1f}x", f"{roi_top5:.1f}x"],
        })
        st.dataframe(sim, width="stretch", hide_index=True)
        insight(
            f"El modelo ML reduce el coste de campaña un <b>{ahorro_pct:.0f}%</b> "
            f"manteniendo el <b>{evr_ml_pct:.0f}%</b> del valor en riesgo capturado.",
            variant="green"
        )

    with col_charts:
        labels   = ["Sin modelo", "Modelo ML", "Top 5%"]
        benefits = [ben_all / 1_000, ben_ml / 1_000, ben_top5 / 1_000]
        costs    = [cost_all / 1_000, cost_ml / 1_000, cost_top5 / 1_000]
        colors   = [C_GRAY, C_BLUE, C_BAJO]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
        bars1 = ax1.bar(labels, benefits, color=colors, width=0.5, edgecolor="white")
        ax1.bar_label(bars1, labels=[f"€{v:.0f}k" for v in benefits], padding=3, fontsize=8)
        ax1.set_ylabel("Beneficio neto (k€)")
        ax1.set_title("Beneficio neto", fontsize=10, fontweight="bold")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}k"))

        bars2 = ax2.bar(labels, costs, color=colors, width=0.5, edgecolor="white")
        ax2.bar_label(bars2, labels=[f"€{v:.0f}k" for v in costs], padding=3, fontsize=8)
        ax2.set_ylabel("Coste de campaña (k€)")
        ax2.set_title("Coste de campaña", fontsize=10, fontweight="bold")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}k"))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    subsection("Implicaciones estratégicas para la dirección comercial")

    df_sorted     = df.sort_values("prob_adj", ascending=False).reset_index(drop=True)
    top10_n       = max(1, int(n_total * 0.10))
    evr_top10_pct = (float(df_sorted.iloc[:top10_n]["evr"].sum()) / evr_all * 100
                     if evr_all > 0 else 0)
    ahorro_pct2   = (1 - cost_ml / cost_all) * 100 if cost_all > 0 else 0
    evr_ml_pct2   = evr_ml / evr_all * 100 if evr_all > 0 else 0

    st.markdown(
        f"""<div class="impl-block">
<div class="impl-ttl">Conclusiones clave</div>

<div class="impl-item">
  <div class="impl-n">1&#65039;&#8419;</div>
  <div>
    <div class="impl-head">El churn no está distribuido uniformemente</div>
    <div class="impl-body">El <b>10% de clientes con mayor riesgo</b> concentra el
    <b>{evr_top10_pct:.0f}%</b> del valor total en riesgo. Actuar sin segmentación
    desperdicia presupuesto en clientes de bajo impacto económico.</div>
  </div>
</div>

<div class="impl-item">
  <div class="impl-n">2&#65039;&#8419;</div>
  <div>
    <div class="impl-head">El modelo permite asignar capital comercial eficientemente</div>
    <div class="impl-body">Interviniendo sobre el <b>{n_ml / n_total * 100:.0f}% de la cartera</b>,
    el modelo captura el <b>{evr_ml_pct2:.0f}%</b> del valor en riesgo con un ROI medio de
    <b>{roi_medio:.1f}x</b> por intervención.</div>
  </div>
</div>

<div class="impl-item">
  <div class="impl-n">3&#65039;&#8419;</div>
  <div>
    <div class="impl-head">El modelo mejora la rentabilidad del marketing</div>
    <div class="impl-body">Frente a un enfoque masivo, el modelo reduce el coste de campaña un
    <b>{ahorro_pct2:.0f}%</b> manteniendo la mayor parte del valor capturado. Cada euro
    invertido genera un retorno medio de <b>{roi_medio:.1f}x</b>.</div>
  </div>
</div>

</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 06 — BÚSQUEDA INDIVIDUAL DE CLIENTE
# ─────────────────────────────────────────────────────────────────────────────
def render_cliente(state):
    df = state["df"]
    te = state["te"]
    tb = state["tb"]
    H  = state["H"]
    r  = state["r"]

    col_inp, col_btn = st.columns([4, 1])
    query  = col_inp.text_input("Buscar cliente", placeholder="Introduce Customer ID  (ej: 103388)",
                                label_visibility="collapsed")
    buscar = col_btn.button("Buscar", use_container_width=True)

    if not (query or buscar):
        st.caption("Introduce un Customer ID para ver el análisis individual.")
        return

    try:
        cid = int(query)
    except ValueError:
        st.error("Customer ID debe ser un número entero.")
        return

    row = df[df["Customer_ID"] == cid]
    if row.empty:
        st.warning(f"Cliente **{cid}** no encontrado.")
        return

    r_d = row.iloc[0]
    seg = r_d["segmento"]
    col_card, col_just = st.columns([1, 1])

    with col_card:
        seg_color = SEG_COLORS.get(seg, C_GRAY)
        st.markdown(
            f"<div class='cliente-card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:1.1rem;font-weight:700;color:#0A2342;'>Cliente {cid}</span>"
            f"&nbsp;{badge(seg)}"
            f"</div>"
            f"<div style='color:#64748B;font-size:0.85rem;margin:4px 0 12px 0;'>"
            f"Modelo {r_d['Modelo']} · Zona {r_d['ZONA']} · Edad {r_d['Edad']}</div>"
            f"<table style='width:100%;font-size:0.9rem;border-collapse:collapse;'>"
            f"<tr><td style='color:#64748B;padding:3px 0;'>Prob. churn</td>"
            f"<td style='font-weight:600;text-align:right;'>{r_d['prob_adj']:.4f}</td></tr>"
            f"<tr><td style='color:#64748B;padding:3px 0;'>CLTV (H={H}, r={r:.0%})</td>"
            f"<td style='font-weight:600;text-align:right;"
            f"color:{'#DC2626' if r_d['cltv']<0 else '#0A2342'};'>"
            f"€{r_d['cltv']:,.2f}</td></tr>"
            f"<tr><td style='color:#64748B;padding:3px 0;'>EVR</td>"
            f"<td style='font-weight:600;text-align:right;'>€{r_d['evr']:,.2f}</td></tr>"
            f"<tr><td style='color:#64748B;padding:3px 0;'>ROI intervención</td>"
            f"<td style='font-weight:600;text-align:right;color:#059669;'>"
            f"{r_d['roi']:.2f}x</td></tr>"
            f"</table>"
            f"<div style='margin-top:14px;padding:10px 12px;background:#F8FAFC;"
            f"border-radius:6px;border-left:4px solid {seg_color};'>"
            f"<div style='font-weight:700;font-size:0.85rem;color:#0A2342;'>{r_d['accion']}</div>"
            f"<div style='font-size:0.78rem;color:#64748B;margin-top:3px;'>"
            f"Intensidad: {r_d['intensidad']}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_just:
        st.markdown("**Justificación económica**")
        if r_d["cltv"] < 0:
            st.warning(
                f"CLTV negativo (€{r_d['cltv']:.2f}).  \n"
                "El margen de este modelo no cubre costes a largo plazo. "
                "No se recomienda inversión en retención."
            )
        else:
            umbral_ref  = te if seg == "Alto" else tb
            umbral_name = "Umbral óptimo" if seg == "Alto" else "Break-even"
            superado    = r_d["prob_adj"] >= umbral_ref
            st.markdown(
                f"| Concepto | Valor |\n|---|---|\n"
                f"| Score original | `{r_d['prob_churn']:.4f}` |\n"
                f"| Score ajustado | `{r_d['prob_adj']:.4f}` |\n"
                f"| CLTV calculado | `€{r_d['cltv']:.2f}` |\n"
                f"| Valor esperado en riesgo | `€{r_d['evr']:.2f}` |\n"
                f"| {umbral_name} | `{umbral_ref:.4f}` |\n"
                f"| Umbral superado | `{'✓ Sí' if superado else '✗ No'}` |"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CARGA INICIAL
# ─────────────────────────────────────────────────────────────────────────────
df_pred, df_raw, costes, meta = load_base()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS DE USUARIO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""<div style="padding:10px 0 18px 0;border-bottom:1px solid #334155;margin-bottom:16px">
  <div style="font-size:0.6rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px">Concesionario · Retención</div>
  <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.01em;line-height:1.2;font-family:'Manrope','Inter',sans-serif">Churn Dashboard</div>
  <div style="font-size:0.76rem;color:#64748b;margin-top:4px">{MES_ACTUAL}</div>
</div>
<div style="font-size:0.65rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px">Parámetros del modelo</div>""",
        unsafe_allow_html=True,
    )

    escenario = st.radio(
        "escenario",
        options=list(ESCENARIOS.keys()),
        index=1,
        format_func=lambda x: {
            "Conservador": "⬇️  Conservador",
            "Base":        "▶️  Base",
            "Agresivo":    "⬆️  Agresivo",
        }[x],
        label_visibility="collapsed",
    )

    params = ESCENARIOS[escenario]

    with st.expander("⚙️ Ajustar supuestos", expanded=False):
        p_real_pct = st.slider("Churn estimado (%)", 0.5, 20.0,
                                float(round(params["p_real"] * 100, 2)), 0.1)
        r_pct      = st.slider("Tasa descuento (%)", 5.0, 20.0,
                                float(params["r"] * 100), 0.5)
        H          = st.slider("Horizonte (revisiones)", 3, 10, int(params["H"]), 1)
        st.markdown("---")
        roi_min    = st.slider("ROI mínimo (x)", 1.0, 15.0, float(params["roi_min"]), 0.5,
                               help="Solo se actúa sobre clientes con ROI ≥ este umbral.")
        budget_pct = st.slider("Presupuesto máximo (% cartera)", 10, 80,
                               int(params["budget_pct"] * 100), 5,
                               help="Máximo de clientes a intervenir, ordenados por EVR.") / 100

    p_real = p_real_pct / 100
    r      = r_pct / 100

    st.divider()

    uploaded = st.file_uploader(
        "📂 Cargar cartera del mes",
        type="csv",
        label_visibility="visible",
        help=(
            "CSV con el mismo formato que `predicciones_nuevos_clientes.csv`. "
            "Columnas requeridas: Customer_ID, Modelo, Edad, ZONA, prob_churn. "
            "El archivo debe incluir un registro por cliente."
        ),
    )
    if uploaded:
        st.success("✓ Cartera cargada correctamente.")

    st.divider()
    ejecutar = st.button("▶  Ejecutar planificación mensual",
                          use_container_width=True, type="primary")
    st.divider()
    st.caption(
        f"Modelo: **{meta['winner_name']}**  \n"
        f"AUC: **{meta['auc_test']:.4f}**  \n"
        f"Entrenado con prior **{P_TRAIN*100:.1f}%**"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE ESTADO
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        st.session_state["_df_cartera"] = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

if "_df_cartera" in st.session_state:
    df_pred = st.session_state["_df_cartera"]

params_key = (round(p_real, 4), H, round(r, 3),
              round(roi_min, 2), round(budget_pct, 3),
              len(df_pred), id(df_pred))

if ejecutar or "state" not in st.session_state or st.session_state.get("_pkey") != params_key:
    with st.spinner("Ejecutando planificación mensual…"):
        state = recalculate(df_pred, df_raw, costes, p_real, H, r, roi_min, budget_pct)
        st.session_state["state"] = state
        st.session_state["_pkey"] = params_key
else:
    state = st.session_state["state"]


# ─────────────────────────────────────────────────────────────────────────────
# 07 — SIMULADOR DE CAMPAÑA INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
def render_simulador(state):
    df        = state["df"].sort_values("prob_adj", ascending=False).reset_index(drop=True)
    c_mkt     = state["c_mkt"]
    n_total   = len(df)
    evr_total = float(df["evr"].sum())
    n_optimo  = int((state["df"]["prob_adj"] >= state["te"]).sum())

    # Curva completa precalculada
    cum_evr  = df["evr"].cumsum().values
    cum_cost = np.arange(1, n_total + 1) * c_mkt
    cum_ben  = cum_evr - cum_cost
    n_max_ben = int(np.argmax(cum_ben)) + 1   # N donde el beneficio es máximo

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        subsection("Panel de control")
        n_sel = st.slider(
            "Clientes a contactar",
            min_value=0, max_value=n_total,
            value=min(n_optimo, n_total),
            step=1,
            key="sim_n_clientes",
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        if n_sel == 0:
            evr_sel  = 0.0
            cost_sel = 0.0
            ben_sel  = 0.0
            roi_sel  = 0.0
        else:
            evr_sel  = float(cum_evr[n_sel - 1])
            cost_sel = float(cum_cost[n_sel - 1])
            ben_sel  = evr_sel - cost_sel
            roi_sel  = evr_sel / cost_sel if cost_sel > 0 else 0.0

        pct_cartera = n_sel / n_total * 100
        pct_evr     = evr_sel / evr_total * 100 if evr_total > 0 else 0.0

        st.metric("EVR capturado",       f"€{evr_sel:,.0f}")
        st.metric("Coste campaña",       f"€{cost_sel:,.0f}")
        st.metric("Beneficio neto",      f"€{ben_sel:,.0f}")
        st.metric("ROI",                 f"{roi_sel:.1f}x")
        st.metric("% cartera contactada", f"{pct_cartera:.1f}%")
        st.metric("% EVR capturado",      f"{pct_evr:.1f}%")

    with col_main:
        subsection("Beneficio neto vs clientes contactados")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, n_total + 1), cum_ben / 1_000,
                color=C_BLUE, lw=2, label="Beneficio neto (k€)")
        ax.fill_between(np.arange(1, n_total + 1), cum_ben / 1_000, 0,
                        where=cum_ben > 0, alpha=0.07, color=C_BLUE)

        # Punto máximo
        ax.scatter([n_max_ben], [cum_ben[n_max_ben - 1] / 1_000],
                   color=C_BAJO, s=60, zorder=5, label=f"Maximo ({n_max_ben:,} clientes)")

        # Línea óptimo automático
        if 0 < n_optimo <= n_total:
            ax.axvline(n_optimo, color=C_MEDIO, ls="--", lw=1.4,
                       label=f"Optimo automatico ({n_optimo:,})")

        # Línea selección actual
        if n_sel > 0:
            ax.axvline(n_sel, color=C_ALTO, ls="-", lw=1.8,
                       label=f"Tu seleccion ({n_sel:,})")
            ax.scatter([n_sel], [ben_sel / 1_000],
                       color=C_ALTO, s=70, zorder=6)

        ax.axhline(0, color=C_GRAY, lw=0.8)
        ax.set_xlabel("Clientes contactados (ordenados por riesgo)")
        ax.set_ylabel("Beneficio neto (k€)")
        ax.legend(fontsize=8, frameon=False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Tabla comparativa
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        subsection("Comparativa de escenarios")

        evr_opt  = float(cum_evr[n_optimo - 1]) if n_optimo > 0 else 0.0
        cost_opt = float(cum_cost[n_optimo - 1]) if n_optimo > 0 else 0.0
        ben_opt  = evr_opt - cost_opt
        roi_opt  = evr_opt / cost_opt if cost_opt > 0 else 0.0

        tabla = pd.DataFrame({
            "Escenario":  ["Optimo automatico", "Tu seleccion", "Diferencia"],
            "Clientes":   [f"{n_optimo:,}",
                           f"{n_sel:,}",
                           f"{n_sel - n_optimo:+,}"],
            "Coste":      [f"€{cost_opt:,.0f}",
                           f"€{cost_sel:,.0f}",
                           f"€{cost_sel - cost_opt:+,.0f}"],
            "EVR":        [f"€{evr_opt:,.0f}",
                           f"€{evr_sel:,.0f}",
                           f"€{evr_sel - evr_opt:+,.0f}"],
            "Beneficio":  [f"€{ben_opt:,.0f}",
                           f"€{ben_sel:,.0f}",
                           f"€{ben_sel - ben_opt:+,.0f}"],
            "ROI":        [f"{roi_opt:.1f}x",
                           f"{roi_sel:.1f}x",
                           f"{roi_sel - roi_opt:+.1f}x"],
        })
        st.dataframe(tabla, hide_index=True, use_container_width=True)

        if n_sel > 0 and ben_sel < ben_opt * 0.9:
            insight(
                f"Tu seleccion ({n_sel:,} clientes) genera un beneficio "
                f"<b>€{ben_opt - ben_sel:,.0f} inferior</b> al optimo automatico. "
                f"Considera ajustar a <b>{n_optimo:,} clientes</b>.",
                variant="amber",
            )
        elif n_sel > 0 and ben_sel >= ben_opt * 0.95:
            insight(
                f"Tu seleccion captura el <b>{pct_evr:.0f}% del EVR</b> "
                f"contactando solo al <b>{pct_cartera:.1f}%</b> de la cartera. Eficiencia alta.",
                variant="green",
            )

        insight(
            f"El umbral económico óptimo de {meta['winner_name']} ({state['te']:.3f}) contacta al "
            f"<b>{n_optimo/n_total*100:.1f}% de la cartera</b>, lo que sugiere que la prior correction "
            f"({P_TRAIN:.1%}→{state['p_real']:.0%}) diluye la capacidad discriminante del modelo en producción.",
            variant="amber",
        )


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PRINCIPAL — scroll lineal con paneles numerados
# ─────────────────────────────────────────────────────────────────────────────

# ── Header + KPIs
with st.container():
    render_header(state, escenario, meta)

# ── 01 / Modelo Predictivo
with st.container(border=True):
    section("01", "Modelo Predictivo de Churn",
            "Evaluación del clasificador sobre datos de test")
    render_modelo(state, meta)

# ── 02 / Optimización Económica
with st.container(border=True):
    section("02", "Optimización Económica de Campañas",
            "Beneficio neto vs umbral de decisión")
    render_economia(state)

# ── 03 / Segmentación Estratégica
with st.container(border=True):
    section("03", "Segmentación Estratégica de Clientes",
            "Customer Value vs Churn Risk")
    render_segmentacion(state)

# ── 04 / Estrategia de Campañas
with st.container(border=True):
    section("04", "Estrategia de Campañas del Mes",
            "Acciones comerciales por segmento y cuartil de valor")
    render_campanas(state)

# ── 05 / Análisis Estratégico
with st.container(border=True):
    section("05", "Análisis Estratégico",
            "Simulación comparativa e implicaciones para la dirección")
    render_estrategia(state)

# ── 06 / Búsqueda Individual
with st.container(border=True):
    section("06", "Búsqueda Individual de Cliente",
            "Análisis económico y recomendación por Customer ID")
    render_cliente(state)

# ── 07 / Simulador de Campaña
with st.container(border=True):
    section("07", "Simulador de Campaña Interactivo",
            "Explora escenarios: que pasa si contacto N clientes?")
    render_simulador(state)

# ── Footer
st.markdown(
    "<div class='footer-bar'>"
    "El sistema transforma <b>riesgo</b> en "
    "<b>asignación óptima de capital comercial</b> "
    "bajo restricciones reales de margen."
    "</div>",
    unsafe_allow_html=True,
)
