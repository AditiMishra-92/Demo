# =====================================================================
#  Cardiac Failure Analytics Dashboard
#  Team 2 - PythonPioneers | NumpyNinja Python Hackathon
#
#  Run:  streamlit run DashboardHeartfailure.py
#  Data: Cardiac_Cleaned_Data.xlsb (or Cardiac_Cleaned_Data.csv) in the same folder
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import html
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

st.set_page_config(page_title="Cardiac Failure Analytics", page_icon="❤️", layout="wide")

# ----------------------------- COLOURS (from our original file) -----------------------------
NAVY = "#073B4C"       # dark teal / headings
TEAL = "#0B5D6B"
GREEN = "#087F5B"
TEAL2 = "#0B7A75"
BLUE = "#087F9B"
GREYTXT = "#637B83"
BG = "#F4F9FB"
ALERT = "#D1495B"      # only for danger / death highlights
RAMP = ["#B7E4D8", "#6CC3B0", TEAL2, TEAL, NAVY]      # light = better, dark = worse
READMIT, DEATH = BLUE, ALERT                            # same meaning on every chart

# ----------------------------- STYLE -----------------------------
st.markdown(f"""
<style>
.stApp {{background:{BG};}}
section[data-testid="stSidebar"] {{background:linear-gradient(180deg,#073B4C,#0B5D6B,#087F5B);}}
section[data-testid="stSidebar"] * {{color:white !important;}}

/* Sidebar navigation as buttons (like the diabetes dashboard) */
section[data-testid="stSidebar"] div[data-testid="stRadio"], section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{width:100%;}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{gap:14px; width:100%; display:flex; flex-direction:column; align-items:stretch;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.35);
    border-radius:14px; padding:16px 18px; width:100% !important; max-width:100% !important; display:flex !important; box-sizing:border-box; justify-content:center; text-align:center; transition:0.2s;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{background:rgba(255,255,255,0.18);}}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
    background:rgba(255,255,255,0.25); border:1px solid white;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {{display:none;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div {{margin:0 auto;}}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{font-size:16px; font-weight:600;}}

.hdr {{background:linear-gradient(90deg,#073B4C,#087F5B);color:white;padding:25px 30px;border-radius:16px;margin-bottom:20px;}}
.hdr h1 {{margin:0;font-size:30px;color:white}} .hdr p {{margin:6px 0 0;opacity:.92}}
.section {{background:white;padding:20px;border-radius:15px;box-shadow:0 3px 12px rgba(0,0,0,.06);margin-bottom:18px;}}
.kpi {{background:white;padding:16px;border-radius:14px;border-left:5px solid #087F5B;box-shadow:0 3px 12px rgba(0,0,0,.06);min-height:105px;}}
.kpi .i{{font-size:25px}} .kpi .t{{font-size:13px;color:#637B83;font-weight:600}} .kpi .v{{font-size:26px;color:#073B4C;font-weight:700}}
.found {{background:#EAF5F8;border-left:5px solid #087F9B;padding:14px 16px;border-radius:9px;margin:6px 0;}}
.todo {{background:#E8F6EF;border-left:5px solid #087F5B;padding:14px 16px;border-radius:9px;margin:6px 0;}}
.badge {{display:inline-block;background:#073B4C;color:white;padding:3px 10px;border-radius:20px;font-size:12px;margin-bottom:6px;}}
.member {{background:white;border-radius:14px;padding:18px;text-align:center;box-shadow:0 3px 12px rgba(0,0,0,.06);border-top:5px solid #087F5B;}}
.member .n {{font-size:17px;font-weight:700;color:#073B4C}} .member .r {{font-size:13px;color:#637B83}}
.stTabs [data-baseweb="tab"] p {{font-size:15px;}}
/* ---------- Reference-style blocks ---------- */
.hero {{background:linear-gradient(120deg,#FFFFFF 0%,#EAF5F8 55%,#D6EFE6 100%);border-radius:22px;padding:40px 44px 0 44px;
        box-shadow:0 6px 20px rgba(7,59,76,.10);overflow:hidden;position:relative;}}
.hero .t1 {{font-size:64px;font-weight:900;color:#073B4C;line-height:1;letter-spacing:1px;margin:0;}}
.hero .t2 {{font-size:46px;font-weight:900;color:#087F5B;line-height:1.1;margin:6px 0 0 0;}}
.hero .sub {{font-size:18px;color:#0B5D6B;margin-top:14px;}}
.hero .line {{height:4px;width:70%;background:linear-gradient(90deg,#073B4C,#087F5B);border-radius:4px;margin:18px 0 26px 0;}}
.pill {{display:inline-block;background:#073B4C;color:white;font-size:26px;font-weight:800;padding:10px 30px;border-radius:14px;letter-spacing:1px;}}
.meet {{color:#087F5B;font-weight:800;font-size:18px;letter-spacing:1px;margin:10px 0 18px 0;}}
.tm {{display:flex;align-items:center;gap:14px;padding:6px 4px;}}
.tm .av {{width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;color:white;flex-shrink:0;}}
.tm .nm {{font-size:18px;font-weight:800;}} .tm .rl {{font-size:14px;color:#637B83;border-top:3px solid;padding-top:4px;margin-top:4px;}}
.herobar {{background:#073B4C;color:white;border-radius:18px 18px 0 0;display:flex;justify-content:space-around;padding:20px 10px;margin:28px -44px 0 -44px;font-size:20px;font-weight:600;}}
.bigtitle {{text-align:center;font-size:46px;font-weight:900;color:#073B4C;letter-spacing:1px;margin:0;}}
.bigtitle span {{display:inline-block;width:18%;height:3px;background:#073B4C;vertical-align:middle;margin:0 18px;border-radius:3px;}}
.lead {{max-width:900px;margin:10px auto 22px auto;text-align:center;font-size:17px;color:#0B5D6B;font-weight:500;}}
.spec {{background:#073B4C;color:white;border-radius:18px;padding:18px 18px 8px 18px;}}
.spec h3 {{color:#7FD8BE;margin:0 0 10px 0;font-size:22px;}}
.spec .row {{display:flex;gap:12px;align-items:center;border-top:1px solid rgba(255,255,255,.18);padding:10px 0;}}
.spec .ic {{font-size:24px;width:34px;text-align:center;}} .spec .k {{font-weight:700;}} .spec .v {{opacity:.9;font-size:14px;}}
.card-h {{text-align:center;}} .card-h .ic {{font-size:34px;}} .card-h .nm {{font-weight:900;font-size:15px;letter-spacing:.5px;margin:4px 0 6px 0;}}
.card-h ul {{text-align:left;font-size:13px;color:#073B4C;padding-left:18px;margin:0;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background:white;border-radius:16px !important;}}
.checkbox {{background:#EAF5F8;border-left:6px solid #073B4C;border-radius:16px;padding:22px 26px;box-shadow:0 3px 12px rgba(0,0,0,.05);margin-bottom:18px;}}
.checkbox b.h {{font-size:18px;color:#073B4C;}}
.checkbox .it {{font-size:17px;color:#073B4C;margin:16px 0;}}
.pagetitle {{font-size:44px;font-weight:800;color:#073B4C;margin:10px 0 18px 0;}}
.dash-title {{font-size:52px;font-weight:800;color:#073B4C;margin:0 0 18px 0;}}
.kpi2 {{background:white;border-radius:20px;padding:22px 24px;box-shadow:0 4px 16px rgba(7,59,76,.08);min-height:130px;}}
.kpi2 .t {{font-size:16px;color:#1F2D33;}} .kpi2 .v {{font-size:40px;color:#073B4C;margin-top:10px;}}
.sec {{font-size:32px;font-weight:700;color:#073B4C;margin:14px 0 6px 0;}}
.sec .badge {{font-size:13px;vertical-align:middle;margin-left:8px;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{color:#073B4C !important; fill:#073B4C !important;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{background:white;border-radius:10px;}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] button,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{color:#073B4C !important; -webkit-text-fill-color:#073B4C !important; fill:#073B4C !important;}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{background:white !important;border-radius:10px;}}
.qbox {{background:white;border:2px solid #087F5B;border-radius:12px;padding:12px 16px;margin:6px 0 12px 0;font-size:16px;color:#073B4C;}}
</style>
""", unsafe_allow_html=True)


# ----------------------------- TEAM LOGO (top of the left menu, every page) -----------------------------
# The logo image is stored inside this file (base64), so no extra image file is needed.
LOGO_B64 = "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"
st.markdown("""
<style>
.team-logo {margin:-70px 0 6px 0; text-align:left;}
.team-logo img {width:110px; height:110px; border-radius:50%; box-shadow:0 3px 12px rgba(0,0,0,.35); border:3px solid rgba(255,255,255,.6);}
</style>
""", unsafe_allow_html=True)


# ----------------------------- SMALL HELPERS -----------------------------
def kpi(icon, title, value, size=26):
    st.markdown(f"<div class='kpi'><div class='i'>{icon}</div><div class='t'>{title}</div>"
                f"<div class='v' style='font-size:{size}px'>{value}</div></div>", unsafe_allow_html=True)


def found(text):
    st.markdown(f"<div class='found'><b>What we found:</b> {text}</div>", unsafe_allow_html=True)


def todo(text):
    st.markdown(f"<div class='todo'><b>Action:</b> {text}</div>", unsafe_allow_html=True)


def kpi2(icon, title, value):
    """Plain white KPI card, diabetes-dashboard style."""
    st.markdown(f"<div class='kpi2'><div class='t'>{icon} {title}</div><div class='v'>{value}</div></div>",
                unsafe_allow_html=True)


def section(title, kind):
    """Tab heading with a small Descriptive / Prescriptive / Predictive label."""
    st.markdown(f"<div class='sec'>{title} <span class='badge'>{kind}</span></div>", unsafe_allow_html=True)


def badge(text):
    st.markdown(f"<span class='badge'>{text}</span>", unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, title_font_color=NAVY,
                      font_color=NAVY, margin=dict(t=60, l=10, r=10, b=10), legend_title="")
    return fig


def bar(x, y, title, colours, ytitle="% of patients", fmt=".1f", height=380):
    fig = px.bar(x=x, y=y, text_auto=fmt, color=x, color_discrete_sequence=colours, title=title)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=ytitle)
    return style(fig, height)


def two_outcomes(table, title):
    long = table.reset_index().melt(id_vars=table.index.name, var_name="Outcome", value_name="Percent")
    fig = px.bar(long, x=table.index.name, y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Readmitted in 6 months": READMIT, "Died in 6 months": DEATH, "Came back": READMIT, "Died": DEATH}, title=title)
    fig.update_layout(yaxis_title="% of patients", xaxis_title="")
    return style(fig, 400)


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------- DATA -----------------------------
HERE = Path(__file__).parent


@st.cache_data
def load_data():
    xlsb, csv = HERE / "Cardiac_Cleaned_Data.xlsb", HERE / "Cardiac_Cleaned_Data.csv"
    if xlsb.exists():
        df = pd.read_excel(xlsb, engine="pyxlsb")
    else:
        df = pd.read_csv(csv)
    new = {}

    stage_order = ["G1 (>=90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"]
    ckd = pd.cut(df["glomerular_filtration_rate"], bins=[0, 15, 30, 45, 60, 90, 1000], right=False, labels=stage_order[::-1])
    new["ckd_stage"] = pd.Categorical(ckd, categories=stage_order, ordered=True)

    cut = np.where(df["gender"] == "Male", 130, 120)
    hb = df["hemoglobin"]
    anemia = np.select([hb.isna(), hb >= cut, hb >= 110, hb >= 80], ["Missing", "No anemia", "Mild", "Moderate"], default="Severe")
    new["anemia_level"] = pd.Categorical(pd.Series(anemia).replace("Missing", np.nan),
                                         categories=["No anemia", "Mild", "Moderate", "Severe"], ordered=True)

    sbp, dbp = df["systolic_blood_pressure"], df["diastolic_blood_pressure"]
    bp_order = ["Low (<90)", "Normal", "Elevated", "High stage 1", "High stage 2"]
    bp = np.select([sbp.isna(), sbp < 90, (sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
                   ["Missing", bp_order[0], bp_order[4], bp_order[3], bp_order[2]], default=bp_order[1])
    new["bp_stage"] = pd.Categorical(pd.Series(bp).replace("Missing", np.nan), categories=bp_order, ordered=True)

    new["age"] = df["agecat"].apply(lambda s: (int(str(s).split("-")[0]) + int(str(s).split("-")[1])) / 2)
    new["male"] = (df["gender"] == "Male").astype(int)
    new["nlr"] = df["neutrophil_count"] / df["lymphocyte_count"]
    new["nlr_log"] = np.log(new["nlr"])
    new["troponin_log"] = np.log1p(df["high_sensitivity_troponin"])
    if "bnp_log" not in df.columns:
        new["bnp_log"] = np.log1p(df["brain_natriuretic_peptide"])
    return pd.concat([df, pd.DataFrame(new)], axis=1)


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the cleaned data file: {e}")
    st.stop()

# Model inputs (admission-time data only)
DEATH_FEATURES = ["nyha_cardiac_function_classification", "killip_grade", "bnp_log", "troponin_log",
                  "nlr_log", "albumin", "hemoglobin", "sodium"]
READMIT_FEATURES = ["nyha_cardiac_function_classification", "killip_grade", "systolic_blood_pressure", "pulse",
                    "respiration", "glomerular_filtration_rate", "urea", "cystatin",
                    "moderate_to_severe_chronic_kidney_disease", "bnp_log", "troponin_log", "nlr_log", "albumin",
                    "hemoglobin", "sodium", "cci_score", "diabetes", "chronic_obstructive_pulmonary_disease",
                    "age", "male", "bmi"]


def logistic():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))])


def model_set():
    return {
        "Logistic Regression": logistic(),
        "Random Forest": Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                                                    class_weight="balanced_subsample", random_state=0, n_jobs=-1))]),
        "ANN (neural network)": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                          ("model", MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=2000, random_state=0))]),
    }


@st.cache_data
def cv_probs(data, features, target, model_name, repeats=1):
    """Risk for every patient, predicted by a model that never saw that patient (5-fold cross-validation)."""
    X, y = data[features].astype(float), data[target]
    probs = []
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs.append(cross_val_predict(model_set()[model_name], X, y, cv=cv, method="predict_proba")[:, 1])
    return np.mean(probs, axis=0)


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown(f"<div class='team-logo'><img src='data:image/png;base64,{LOGO_B64}' alt='Team logo'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:48px'>❤️</div>"
                "<h2 style='text-align:center;margin:0'>Cardiac Failure</h2>"
                "<p style='text-align:center'>Team 2 • PythonPioneers</p>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["🏠 Introduction", "📘 Data Overview", "🧹 Data Cleaning & Feature Engineering",
                                   "📊 Insights", "🤖 Model Performance", "📌 Key Takeaways & Conclusion"],
                    label_visibility="collapsed")


# =====================================================================
# 1. INTRODUCTION
# =====================================================================
if page == "🏠 Introduction":
    team = [("Aditi Mishra", "Team Lead", NAVY), ("Saranya Shanmugam", "Team Member", GREEN),
            ("Sashi Laguduva", "Team Member", BLUE), ("Sudha Madhuri Basa", "Team Member", ALERT)]
    members = "".join(
        f"<div class='tm'><div class='av' style='background:{c}'>👤</div>"
        f"<div><div class='nm' style='color:{c}'>{n}</div><div class='rl' style='border-color:{c}'>{r}</div></div></div>"
        for n, r, c in team)
    heart_svg = (
        "<svg viewBox='0 0 220 200' width='300' style='position:absolute;right:50px;top:40px;opacity:.95'>"
        "<defs><linearGradient id='hg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#E86A7A'/>"
        "<stop offset='1' stop-color='#B8324A'/></linearGradient></defs>"
        "<path d='M110 185 C 30 125, 5 75, 40 38 C 70 8, 102 22, 110 50 C 118 22, 150 8, 180 38 C 215 75, 190 125, 110 185 Z' fill='url(#hg)'/>"
        "<polyline points='20,105 70,105 85,80 100,135 118,55 135,120 148,105 200,105' fill='none' stroke='white' "
        "stroke-width='7' stroke-linejoin='round' stroke-linecap='round'/></svg>")
    st.markdown(
        f"<div class='hero'>{heart_svg}"
        "<p class='t1'>CARDIAC FAILURE</p>"
        "<p class='t2'>HEART FAILURE DATASET</p>"
        "<div class='sub'>Spotting high-risk heart failure patients on the day they are admitted</div>"
        "<div class='line'></div>"
        "<div style='text-align:center'><span class='pill'>TEAM 2: PYTHONPIONEERS</span>"
        "<div class='meet'>—— MEET OUR TEAM ——</div></div>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px'>{members}</div>"
        "<div class='herobar'><span>⭐ Early Risk Detection</span><span>❤️ Better Decisions</span>"
        "<span>👥 Healthier Hearts</span></div></div>", unsafe_allow_html=True)


# =====================================================================
# 2. DATA OVERVIEW
# =====================================================================
elif page == "📘 Data Overview":
    st.markdown("<div class='bigtitle'><span></span>DATA OVERVIEW<span></span></div>"
                "<div class='lead'>This dataset links 7 hospital tables for 2,008 heart failure patients: who they are, "
                "how sick their heart is, their other diseases, 100+ blood tests, alertness, medicines, and what happened "
                "to them up to 6 months after discharge. It lets us find who needs extra care and spot them early.</div>",
                unsafe_allow_html=True)

    years = pd.to_datetime(df["admission_date"])
    spec_rows = [("👥", "Patients", f"{len(df):,} hospitalised heart failure patients"),
                 ("🗂️", "Source", "7 hospital tables, linked by patient ID"),
                 ("📅", "Admissions", f"{years.dt.year.min()} – {years.dt.year.max()}"),
                 ("⏱️", "Follow-up", "28 days, 3 months, 6 months"),
                 ("🧪", "Tests", "100+ blood tests and vital signs"),
                 ("💊", "Medicines", "25 drugs given in hospital"),
                 ("📋", "Final table", "2,008 rows × 210 columns")]
    spec = "".join(f"<div class='row'><div class='ic'>{i}</div><div><div class='k'>{k}:</div><div class='v'>{v}</div></div></div>"
                   for i, k, v in spec_rows)

    left, right = st.columns([1, 3.2])
    with left:
        st.markdown(f"<div class='spec'><h3>Cardiac Failure<br>Dataset Specifications</h3>{spec}</div>", unsafe_allow_html=True)

    def mini(fig):
        fig.update_layout(template="plotly_white", height=150, margin=dict(t=5, l=5, r=5, b=5), showlegend=False,
                          xaxis_title="", yaxis_title="", font_size=10)
        fig.update_traces(selector=dict(type="pie"), textinfo="none")
        return fig

    cards = [
        ("🧍", "DEMOGRAPHY", NAVY, ["Gender", "Age group", "Height, weight, BMI", "Occupation"],
         lambda: px.bar(df["agecat"].value_counts().sort_index(), color_discrete_sequence=[NAVY])),
        ("❤️", "CARDIAC", ALERT, ["NYHA class (symptoms)", "Killip grade (fluid/shock)", "Heart failure type", "Heart scan (LVEF)"],
         lambda: px.bar(df["nyha_cardiac_function_classification"].value_counts().sort_index(), color_discrete_sequence=[ALERT])),
        ("📜", "HISTORY", GREEN, ["Diabetes", "Kidney disease", "COPD, liver disease", "Comorbidity score"],
         lambda: px.bar(pd.Series({"Kidney": df["moderate_to_severe_chronic_kidney_disease"].mean(),
                                   "Diabetes": df["diabetes"].mean(),
                                   "COPD": df["chronic_obstructive_pulmonary_disease"].mean()}) * 100,
                        color_discrete_sequence=[GREEN])),
        ("🏥", "HOSPITAL STAY", BLUE, ["Admission type", "Days in hospital", "Death: 28d / 3m / 6m", "Readmission: 28d / 3m / 6m"],
         lambda: px.bar(pd.Series({"Came back": df["re_admission_within_6_months"].mean(),
                                   "Died": df["death_within_6_months"].mean()}) * 100,
                        color=["Came back", "Died"], color_discrete_sequence=[READMIT, DEATH])),
        ("🧪", "LABS", TEAL2, ["BNP (heart strain)", "Troponin (heart damage)", "Kidney tests (eGFR)", "Blood count, salts"],
         lambda: px.histogram(np.log10(df["brain_natriuretic_peptide"].dropna()), nbins=25, color_discrete_sequence=[TEAL2])),
        ("🧠", "RESPONSIVENESS", "#6C4AB6", ["Eye opening", "Verbal response", "Movement", "GCS score (alertness)"],
         lambda: px.pie(values=df["gcs_category"].value_counts().values, names=df["gcs_category"].value_counts().index,
                        hole=.6, color_discrete_sequence=["#6C4AB6", "#B9A6E3", "#D8CCF1", "#EDE7F8"])),
        ("💊", "PRESCRIPTIONS", "#E07A5F", ["25 medicines", "Water tablets", "Heart medicines", "Medicines per patient"],
         lambda: px.histogram(df["total_drugs"], nbins=16, color_discrete_sequence=["#E07A5F"])),
        ("✨", "DERIVED FEATURES", TEAL, ["BMI / BP groups", "Kidney stage, anemia level", "Warning flags", "NLR, comorbidity count"],
         lambda: px.pie(values=df["bmi_category"].value_counts().values, names=df["bmi_category"].value_counts().index,
                        hole=.6, color_discrete_sequence=[TEAL, "#6CC3B0", "#B7E4D8", NAVY])),
    ]
    with right:
        for row in (cards[:4], cards[4:]):
            cols = st.columns(4)
            for col, (ic, nm, colr, items, chart) in zip(cols, row):
                with col:
                    with st.container(border=True):
                        bullets = "".join(f"<li>{x}</li>" for x in items)
                        st.markdown(f"<div class='card-h'><div class='ic'>{ic}</div>"
                                    f"<div class='nm' style='color:{colr}'>{nm}</div><ul>{bullets}</ul></div>",
                                    unsafe_allow_html=True)
                        st.plotly_chart(mini(chart()), width="stretch", config={"displayModeBar": False})


# =====================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =====================================================================
elif page == "🧹 Data Cleaning & Feature Engineering":
    st.markdown("<div class='pagetitle'>🧹 Data Cleaning & Feature Engineering</div>", unsafe_allow_html=True)
    steps = ["Removed a fake patient record and joined all 7 tables into one (one row per patient)",
             "Set impossible values to blank: 0 kg weight, 0 pulse, BMI of 404, reversed blood pressure",
             "Fixed wrong units: troponin, hematocrit and heart-scan values",
             "Filled blanks only when the meaning was clear (blank breathing support = no ventilation)",
             "Kept real gaps empty: missing lab tests were not invented",
             "Changed medicines from many rows per patient to one row per patient",
             "Renamed confusing lab columns and made yes/no columns 1/0"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in steps)
    st.markdown(f"<div class='checkbox'><b class='h'>Data Cleaning Steps:</b>{items}</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#073B4C'>🧠 Engineered Features</h3>", unsafe_allow_html=True)
    feats = pd.DataFrame({
        "Feature": ["bmi_category, bp_category", "ckd_stage, anemia_level", "bnp_elevated_flag, troponin_elevated_flag",
                    "polypharmacy_flag, total_drugs", "comorbidity_count", "nlr (neutrophil ÷ lymphocyte)", "bnp_log, hs_crp_log"],
        "Purpose": ["Compare patient groups easily", "Kidney and blood health in clear stages",
                    "Quick yes/no warning signs (heart strain, heart damage)", "How many medicines each patient takes",
                    "How much extra illness a patient carries", "Free inflammation marker from the routine blood count",
                    "Stop a few extreme values from controlling the models"]})
    st.dataframe(feats, hide_index=True, width="stretch")


# =====================================================================
# 4. INSIGHTS  (guided: Insight Area -> Marker -> Outcome)
# =====================================================================
elif page == "📊 Insights":
    st.markdown("<div class='dash-title'>📊 Cardiac Failure Dashboard</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi2("🔁", "Came back (6 months)", pct(df["re_admission_within_6_months"].mean()))
    with c2: kpi2("⚠️", "Died (6 months)", pct(df["death_within_6_months"].mean()))
    with c3: kpi2("❤️", "Severe symptoms (NYHA 3–4)", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))
    with c4: kpi2("🧪", "Median BNP", f"{df['brain_natriuretic_peptide'].median():.0f}")
    st.write("")
    st.markdown("<div class='section' style='padding:12px 18px'>Choose an area, a marker and an outcome. "
                "Each insight follows <b>Marker → Evidence → Finding → What it means</b>.</div>", unsafe_allow_html=True)

    # ---------------- helper to cut a column into labelled groups ----------------
    def cut(col, bins, labels):
        return pd.cut(df[col], bins=bins, labels=labels, right=False)

    def yes_no(mask, yes, no):
        return pd.Series(np.where(mask, yes, no), index=df.index)

    def drug_any(cols):
        return df[cols].sum(axis=1) > 0

    nyha_grp = np.where(df["nyha_cardiac_function_classification"] == 4, "Symptoms at rest", "Symptoms on activity")
    kil_grp = np.where(df["killip_grade"] >= 3, "fluid/shock", "no fluid")

    # ---------------- all insight areas, markers, groups and plain-English meaning ----------------
    # Each marker: (kind, function returning groups, "what it means" text)
    AREAS = {
        "❤️ Cardiac Biomarkers": ("Prescriptive", {
            "BNP (heart strain)": (lambda: cut("brain_natriuretic_peptide", [0, 100, 500, 2000, 1e9],
                                               ["Normal (<100)", "100–500", "500–2000", "Very high (≥2000)"]),
                                   "BNP rises when the heart is stretched. Higher BNP means a more strained heart; "
                                   "use it with the bedside exam to judge how sick the patient is."),
            "Troponin (heart damage)": (lambda: cut("high_sensitivity_troponin", [0, 14.0001, 100, 1e9],
                                                    ["Normal (≤14)", "Raised (14–100)", "Very high (>100)"]),
                                        "Troponin shows heart muscle damage. Very high troponin points to acute injury "
                                        "and a patient who needs closer monitoring."),
        }),
        "🫘 Kidney Function": ("Prescriptive", {
            "Kidney stage (eGFR)": (lambda: df["ckd_stage"],
                                    "Heart and kidneys pull each other down. eGFR below 45 (stage G3b or worse) should be "
                                    "treated as high risk: careful water-tablet dosing, potassium checks, early follow-up."),
            "Urea": (lambda: cut("urea", [0, 7.1, 15, 1e9], ["Normal (<7.1)", "Raised (7.1–15)", "High (≥15)"]),
                     "Urea builds up when the kidneys are not clearing waste, often because the heart pumps poorly."),
            "Chronic kidney disease (history)": (lambda: yes_no(df["moderate_to_severe_chronic_kidney_disease"] == 1, "Yes", "No"),
                                                 "Known kidney disease adds long-term burden and limits which heart "
                                                 "medicines can be used safely."),
        }),
        "🔥 Inflammation & Nutrition": ("Prescriptive", {
            "NLR (routine blood count)": (lambda: pd.qcut(df["nlr"], 4, labels=["Lowest", "Low", "High", "Highest (≥8.7)"]),
                                          "NLR is free from the routine blood count and available for almost every patient. "
                                          "Flag NLR of 8.7 or more for closer monitoring."),
            "White blood cells": (lambda: cut("white_blood_cell", [0, 4, 10, 1e9], ["Low (<4)", "Normal (4–10)", "High (>10)"]),
                                  "A high white cell count suggests infection or stress, a common trigger of heart failure attacks."),
            "hs-CRP (special test)": (lambda: cut("hs_crp", [0, 3, 1e9], ["Normal (<3)", "High (≥3)"]),
                                      "hs-CRP measures inflammation but was not tested for about half of patients, "
                                      "so NLR is the more practical marker."),
            "Albumin (nutrition)": (lambda: cut("albumin", [0, 35, 1e9], ["Low (<35)", "Normal (≥35)"]),
                                    "Low albumin reflects poor nutrition and inflammation; these patients recover less well."),
        }),
        "🛏️ Clinical Severity (Killip / NYHA)": ("Predictive", {
            "Killip grade (fluid / shock)": (lambda: df["killip_grade"].map(lambda k: f"Killip {k}"),
                                             "A 30-second bedside exam. Killip 1 patients are low risk; Killip 3–4 "
                                             "(fluid in lungs or shock) need close monitoring."),
            "NYHA class (symptoms)": (lambda: df["nyha_cardiac_function_classification"].map(lambda k: f"NYHA {k}"),
                                      "NYHA shows how much symptoms limit daily life. Class 4 (symptoms at rest) carries the most risk."),
            "Killip + NYHA together": (lambda: pd.Series(pd.Categorical([f"{n} + {k}" for n, k in zip(nyha_grp, kil_grp)], categories=[
                                           "Symptoms on activity + no fluid", "Symptoms at rest + no fluid",
                                           "Symptoms on activity + fluid/shock", "Symptoms at rest + fluid/shock"],
                                           ordered=True), index=df.index),
                                       "Using both bedside scores together separates patients even better than either alone."),
            "Alertness (consciousness)": (lambda: yes_no(df["consciousness"] == "Clear", "Fully alert", "Not fully alert"),
                                          "Patients who are not fully alert at admission are rare but very high risk."),
        }),
        "🕰️ Current Severity vs Prior History": ("Predictive", {
            "Old heart attack": (lambda: yes_no(df["myocardial_infarction"] == 1, "Yes", "No"),
                                 "Past diagnoses tell us little about who will die. Today's bedside condition matters more."),
            "Past heart failure": (lambda: yes_no(df["congestive_heart_failure"] == 1, "Yes", "No"),
                                   "Most patients (93%) already had heart failure before. Patients newly diagnosed at this admission "
                                   "had more deaths, so a first-time diagnosis deserves extra attention."),
            "Circulation problems (PVD)": (lambda: yes_no(df["peripheral_vascular_disease"] == 1, "Yes", "No"),
                                           "Old vascular disease adds little once current severity is known."),
            "Killip grade today (compare)": (lambda: df["killip_grade"].map(lambda k: f"Killip {k}"),
                                             "Compare with the history markers: today's Killip grade shows a much bigger difference."),
        }),
        "🩸 Anemia": ("Prescriptive", {
            "Anemia level (WHO)": (lambda: df["anemia_level"],
                                   "Mild and moderate anemia add little risk, but severe anemia (Hb below 80) is a real "
                                   "warning sign: flag it at admission and correct it."),
            "Severe anemia vs rest": (lambda: pd.Series(np.where(df["anemia_level"].isna(), None,
                                                                 np.where(df["anemia_level"] == "Severe", "Severe (<80)", "Not severe")),
                                                        index=df.index),
                                      "Severe anemia makes a weak heart work much harder to deliver oxygen."),
        }),
        "🩺 Blood Pressure": ("Prescriptive", {
            "Blood pressure stage": (lambda: df["bp_stage"],
                                     "Low BP (below 90) means the pump is failing: treat as possible shock. Higher BP patients "
                                     "come back less often because their heart still has strength."),
            "Pulse": (lambda: cut("pulse", [0, 60, 100.0001, 1e9], ["Slow (<60)", "Normal (60–100)", "Fast (>100)"]),
                      "A fast pulse can mean the heart is struggling to keep up."),
        }),
        "🧂 Blood Gas & Salts": ("Prescriptive", {
            "Sodium": (lambda: cut("sodium", [0, 135, 145.0001, 1e9], ["Low (<135)", "Normal (135–145)", "High (>145)"]),
                       "Low sodium often reflects fluid overload; review fluids and water tablets."),
            "Potassium": (lambda: cut("potassium", [0, 3.5, 5.0001, 1e9], ["Low (<3.5)", "Normal (3.5–5)", "High (>5)"]),
                          "High potassium is common with weak kidneys and some heart medicines; monitor it closely."),
            "Lactate (blood gas)": (lambda: cut("lactate", [0, 2, 1e9], ["Normal (<2)", "High (≥2)"]),
                                    "High lactate means tissues are short of oxygen. Tested for about half of patients."),
            "Bicarbonate (blood gas)": (lambda: cut("standard_bicarbonate", [0, 22, 1e9], ["Low (<22)", "Normal (≥22)"]),
                                        "Low bicarbonate means acid build-up in the blood, a sign of a very sick patient."),
        }),
        "👥 Patient Profile": ("Descriptive", {
            "Age group": (lambda: df["agecat"], "Heart failure risk in this group follows how sick patients are more than their age."),
            "Gender": (lambda: df["gender"], "Women make up 58% of patients; outcomes differ little by gender."),
            "BMI group": (lambda: df["bmi_category"].astype(object),
                          "1 in 4 patients is underweight, a sign of frailty in long-term heart failure."),
            "Number of other diseases": (lambda: df["comorbidity_count"].clip(upper=3).map(
                                             {0: "0", 1: "1", 2: "2", 3: "3+"}),
                                         "More other diseases means more burden and more returns to hospital."),
        }),
        "💊 Medicines": ("Descriptive", {
            "ACE inhibitor / ARB": (lambda: yes_no(drug_any(["Benazepril hydrochloride tablet", "Valsartan Dispersible tablet"]), "Given", "Not given"),
                                    "A key long-term heart medicine, given to only about 4 in 10 patients. Differences reflect "
                                    "who was well enough to receive it, not proof the drug caused them."),
            "Beta-blocker": (lambda: yes_no(drug_any(["Metoprolol Succinate Sustained-release tablet", "metoprolol tartrate injection"]), "Given", "Not given"),
                             "Another key long-term medicine given to only about 4 in 10 patients."),
            "Spironolactone": (lambda: yes_no(df["Spironolactone tablet"] == 1, "Given", "Not given"),
                               "Given to most patients. Those not given it were often too sick or had kidney problems, so this "
                               "shows a link, not proof of cause. It needs potassium checks."),
            "Water tablet by drip (IV furosemide)": (lambda: yes_no(df["Furosemide injection"] == 1, "Given", "Not given"),
                                                     "IV water tablets are used for more congested, sicker patients."),
            "Number of medicines": (lambda: cut("total_drugs", [0, 5, 9, 13, 100], ["0–4", "5–8", "9–12", "13+"]),
                                    "Patients on very few medicines had more deaths, likely because the sickest patients died or left "
                                    "before full treatment. This shows a link, not that medicines alone made the difference."),
        }),
        "🤖 Predicted Risk (model)": ("Predictive", {
            "Predicted risk group": (None,
                                     "Our Logistic Regression model scores every patient using admission data only, tested on "
                                     "patients it never saw. The highest-risk group should get closer monitoring and early follow-up."),
        }),
    }

    OUTCOMES = {"Readmission within 28 days": "re_admission_within_28_days",
                "Readmission within 3 months": "re_admission_within_3_months",
                "Readmission within 6 months": "re_admission_within_6_months",
                "Death within 28 days": "death_within_28_days",
                "Death within 3 months": "death_within_3_months",
                "Death within 6 months": "death_within_6_months"}

    area = st.selectbox("1. Select Insight Area", list(AREAS.keys()))
    kind, markers = AREAS[area]
    marker = st.selectbox("2. Select Marker", list(markers.keys()))
    out_label = st.selectbox("3. Select Outcome", list(OUTCOMES.keys()), index=5)
    target = OUTCOMES[out_label]
    is_death = target.startswith("death")
    make_groups, meaning = markers[marker]

    # ---------------- build the groups ----------------
    if make_groups is None:   # predicted risk groups from the model
        feats = DEATH_FEATURES if is_death else READMIT_FEATURES
        with st.spinner("Scoring patients with the model..."):
            prob = cv_probs(df, feats, target, "Logistic Regression", repeats=3)
        groups = pd.Series(pd.qcut(prob, 5, labels=["Lowest", "Low", "Middle", "High", "Highest"]), index=df.index)
        model_auc = roc_auc_score(df[target], prob)
    else:
        groups = make_groups()
        model_auc = None

    data = pd.DataFrame({"group": groups, "y": df[target]}).dropna()
    if isinstance(groups.dtype, pd.CategoricalDtype):
        order = [c for c in groups.cat.categories if c in set(data["group"])]
    else:
        order = sorted(data["group"].unique(), key=str)
    summary = data.groupby("group", observed=True)["y"].agg(["mean", "size"]).reindex(order)
    summary["rate"] = summary["mean"] * 100

    # ---------------- layout: Donut | Chart | Finding ----------------
    left, mid, right = st.columns([0.9, 1.7, 1.3])

    with left:
        yes = df[target].mean() * 100
        word = "Died" if is_death else "Came back"
        fig = go.Figure(go.Pie(values=[yes, 100 - yes], labels=[word, "Did not"], hole=0.68, sort=False,
                               marker=dict(colors=[DEATH if is_death else READMIT, "#E3ECEF"]), textinfo="none"))
        fig.update_layout(title=dict(text="All 2,008 patients", font=dict(size=14, color=NAVY)), height=300,
                          margin=dict(t=40, l=5, r=5, b=5), showlegend=True,
                          legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                          annotations=[dict(text=f"<b>{yes:.1f}%</b><br>{word.lower()}", x=0.5, y=0.5,
                                            showarrow=False, font=dict(size=18, color=NAVY))])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(out_label)

    with mid:
        accent = DEATH if is_death else READMIT
        if len(summary) > 2:
            cols_used = (RAMP * 3)[:len(summary)]
        else:   # two groups: highlight the higher-risk one
            cols_used = [accent if r == summary["rate"].max() else "#9FB7BE" for r in summary["rate"]]
        fig = px.bar(x=[str(i) for i in summary.index], y=summary["rate"], text_auto=".1f",
                     color=[str(i) for i in summary.index], color_discrete_sequence=cols_used,
                     title=f"{marker}: {out_label}")
        fig.update_traces(customdata=summary["size"], hovertemplate="%{x}<br>%{y:.1f}%<br>%{customdata} patients<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="% of patients")
        st.plotly_chart(style(fig, 360), width="stretch")

    with right:
        badge(kind)
        top, low = summary["rate"].idxmax(), summary["rate"].idxmin()
        tested = len(data)
        # Evidence: chi-square test across the groups (predicted risk uses ROC-AUC)
        if model_auc is not None:
            evidence = f"Model ROC-AUC {model_auc:.2f} on unseen patients (0.5 = coin toss)"
        elif summary["size"].min() > 0 and data["y"].nunique() == 2 and len(summary) > 1:
            _, p, _, _ = stats.chi2_contingency(pd.crosstab(data["group"], data["y"]))
            evidence = f"Chi-square test, p {'< 0.001' if p < 0.001 else '= ' + format(p, '.3f')} " \
                       f"({'significant' if p < 0.05 else 'not significant'})"
        else:
            evidence = "Descriptive comparison"
        small = summary[summary["size"] < 30]
        st.markdown(
            f"<div class='sec' style='font-size:26px'>🔎 Finding</div>"
            f"<p style='font-size:16px;color:#073B4C'>Patients in <b>{html.escape(str(top))}</b> had the highest rate: "
            f"<b>{summary.loc[top, 'rate']:.1f}%</b> ({int(summary.loc[top, 'size'])} patients), "
            f"vs <b>{summary.loc[low, 'rate']:.1f}%</b> in <b>{html.escape(str(low))}</b>. "
            f"Average for all patients: {df[target].mean()*100:.1f}%.</p>"
            f"<p style='font-size:14px;color:#637B83'><b>Evidence:</b> {evidence}. "
            f"Patients with this marker: {tested:,} of {len(df):,}."
            + (f" Small groups (under 30 patients): {html.escape(', '.join(map(str, small.index)))}." if len(small) else "")
            + "</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='todo'><b>What this means:</b> {meaning}</div>", unsafe_allow_html=True)


# =====================================================================
# 5. MODEL PERFORMANCE
# =====================================================================
elif page == "🤖 Model Performance":
    st.markdown("<div class='pagetitle'>🤖 Model Performance</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='section'>
<b>How we tested:</b> only information available <b>at admission</b> was used. Each model was trained on 4/5 of the
patients and tested on the other 1/5, five times over (5-fold cross-validation), so every score comes from unseen patients.
<br><br>
<b>How to read the scores:</b>
<ul style='margin-bottom:0'>
<li><b>ROC-AUC</b>: how often the model ranks a patient who had the outcome above one who did not. 0.5 = coin toss, 1.0 = perfect.</li>
<li><b>Recall</b>: of the patients who had the outcome, how many the model flagged.</li>
<li><b>Precision</b>: of the patients the model flagged, how many really had the outcome.</li>
<li><b>Accuracy is misleading here:</b> only 3% die, so a model that says "nobody dies" is 97% accurate and useless.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    targets = {"6-month death": ("death_within_6_months", DEATH_FEATURES, df),
               "28-day death": ("death_within_28_days", DEATH_FEATURES, df),
               "6-month readmission": ("re_admission_within_6_months", READMIT_FEATURES,
                                       df[(df["outcome_during_hospitalization"] != "Dead") &
                                          (df["death_within_6_months"] == 0)].reset_index(drop=True))}
    choice = st.selectbox("Outcome to predict", list(targets.keys()))
    target, feats, data = targets[choice]
    y = data[target].values

    with st.spinner("Training and testing 3 models..."):
        rows, curves, preds = [], {}, {}
        for name in model_set():
            p = cv_probs(data, feats, target, name)
            pred = (p >= 0.5).astype(int)
            preds[name] = pred
            curves[name] = roc_curve(y, p)
            rows.append([name, roc_auc_score(y, p), recall_score(y, pred, zero_division=0),
                         precision_score(y, pred, zero_division=0), accuracy_score(y, pred)])
        rows.append(["Baseline: predict 'no' for everyone", 0.5, 0.0, 0.0, 1 - y.mean()])
    res = pd.DataFrame(rows, columns=["Model", "ROC-AUC", "Recall", "Precision", "Accuracy"])

    lr = res.set_index("Model").loc["Logistic Regression"]
    ann = res.set_index("Model").loc["ANN (neural network)"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🏆", "Chosen model", "Logistic Regression", size=20)
    with c2: kpi("📈", "ROC-AUC (chosen model)", f"{lr['ROC-AUC']:.2f}")
    with c3: kpi("🎯", "Patients caught (recall)", pct(lr["Recall"]))
    with c4: kpi("👥", "Patients with outcome", f"{int(y.sum())} of {len(y):,}")
    st.write("")

    st.dataframe(res.style.format({c: "{:.2f}" for c in ["ROC-AUC", "Recall", "Precision", "Accuracy"]}),
                 hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, colr in zip(curves, [GREEN, BLUE, NAVY]):
            fpr, tpr, _ = curves[name]
            auc = res.set_index("Model").loc[name, "ROC-AUC"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.2f})", line=dict(color=colr, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin toss", line=dict(color=ALERT, dash="dash")))
        fig.update_layout(title="ROC curve (higher and more to the left = better)",
                          xaxis_title="False alarms (rate)", yaxis_title="Patients caught (rate)")
        st.plotly_chart(style(fig, 400), width="stretch")
    with right:
        cm = confusion_matrix(y, preds["Logistic Regression"])
        cm_df = pd.DataFrame(cm, index=["Actual: no", "Actual: yes"], columns=["Flagged: no", "Flagged: yes"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=["#EAF5F8", TEAL2, NAVY],
                        title="Logistic Regression: who it flagged")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 400), width="stretch")

    found(f"All three models <b>rank</b> patients about equally well (ROC-AUC {res['ROC-AUC'][:3].min():.2f} to "
          f"{res['ROC-AUC'][:3].max():.2f}). The difference is who they actually <b>flag</b>: Logistic Regression catches "
          f"<b>{lr['Recall']*100:.0f}%</b> of these patients, while the neural network catches {ann['Recall']*100:.0f}% "
          f"and can look 'accurate' only because it says 'no' to almost everyone, like the baseline row. "
          f"So we chose <b>Logistic Regression</b>: it catches the most high-risk patients and is easy to explain to doctors.")
    todo("A flagged patient is not a diagnosis. Flags point the team to who needs a closer look first.")

    # ---------------- Key models at a glance (scores calculated live, on unseen patients) ----------------
    st.subheader("Our key models at a glance")
    alive6 = targets["6-month readmission"][2]
    glance = [
        ("Death within 28 days", "Bedside check only (Killip + NYHA)",
         df, ["killip_grade", "nyha_cardiac_function_classification"], "death_within_28_days", "Very good with just a 30-second exam"),
        ("Death within 6 months", "Bedside check + 6 routine blood tests",
         df, DEATH_FEATURES, "death_within_6_months", "Best overall death model"),
        ("Death within 28 days", "NLR from the routine blood count",
         df, ["nlr_log"], "death_within_28_days", "A free test with useful signal"),
        ("Came back within 6 months", "21 admission measures",
         alive6, READMIT_FEATURES, "re_admission_within_6_months", "Weak, but top-risk group returns 2x as often"),
    ]
    summary = pd.DataFrame({
        "What we predicted": [g[0] for g in glance],
        "Using": [g[1] for g in glance],
        "ROC-AUC": [f"{roc_auc_score(g[2][g[4]], cv_probs(g[2], g[3], g[4], 'Logistic Regression', repeats=3)):.2f}"
                    for g in glance],
        "In simple words": [g[5] for g in glance]})
    st.dataframe(summary, hide_index=True, width="stretch")

    # ---------------- Patient risk check ----------------
    st.subheader("🩺 Try it: Patient Risk Check")
    st.caption("Uses the 6-month death model. Pick a real patient or enter a new one. "
               "It supports the doctor's judgement; it does not replace it.")

    @st.cache_resource
    def final_model():
        model = logistic().fit(df[DEATH_FEATURES].astype(float), df["death_within_6_months"])
        prob = cv_probs(df, DEATH_FEATURES, "death_within_6_months", "Logistic Regression", repeats=3)
        edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
        rate = pd.Series(df["death_within_6_months"].values).groupby(np.digitize(prob, edges)).mean() * 100
        return model, edges, rate, prob

    model, edges, death_rate, cv_prob = final_model()

    ids = ["New patient (enter values)"] + sorted(df["inpatient_number"].astype(int).tolist())
    pid = st.selectbox("Patient ID", ids, help="Pick a patient from our data to fill in their admission values, "
                                                "or choose 'New patient' and type the values.")
    med = df[["brain_natriuretic_peptide", "high_sensitivity_troponin", "neutrophil_count", "lymphocyte_count",
              "albumin", "hemoglobin", "sodium", "glomerular_filtration_rate", "systolic_blood_pressure"]].median()
    if pid == ids[0]:
        d = {"nyha": 3, "killip": 2, "bnp": 750.0, "trop": 55.0, "neut": 5.0, "lymph": 1.0,
             "alb": 37.0, "hb": 115.0, "na": 139.0, "egfr": 60.0, "sbp": 130.0}
        row_i = None
    else:
        row_i = df.index[df["inpatient_number"].astype(int) == pid][0]
        r = df.loc[row_i]

        def val(col, lo, hi):
            v = r[col] if pd.notna(r[col]) else med[col]
            return float(min(max(v, lo), hi))

        d = {"nyha": int(r["nyha_cardiac_function_classification"]), "killip": int(r["killip_grade"]),
             "bnp": val("brain_natriuretic_peptide", 10, 5000), "trop": val("high_sensitivity_troponin", 0, 50000),
             "neut": val("neutrophil_count", 0.1, 50), "lymph": val("lymphocyte_count", 0.05, 20),
             "alb": val("albumin", 10, 60), "hb": val("hemoglobin", 30, 200), "na": val("sodium", 110, 160),
             "egfr": val("glomerular_filtration_rate", 1, 200), "sbp": val("systolic_blood_pressure", 50, 250)}

    with st.form(f"patient_{pid}"):
        c1, c2, c3, c4 = st.columns(4)
        nyha = c1.selectbox("NYHA class (symptoms)", [1, 2, 3, 4], index=[1, 2, 3, 4].index(d["nyha"]))
        killip = c2.selectbox("Killip grade (fluid / shock)", [1, 2, 3, 4], index=d["killip"] - 1)
        bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, d["bnp"])
        trop = c4.number_input("Troponin (pg/mL)", 0.0, 50000.0, d["trop"])
        c5, c6, c7, c8 = st.columns(4)
        neut = c5.number_input("Neutrophils (x10^9/L)", 0.1, 50.0, d["neut"])
        lymph = c6.number_input("Lymphocytes (x10^9/L)", 0.05, 20.0, d["lymph"])
        alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, d["alb"])
        hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, d["hb"])
        c9, c10, c11, _ = st.columns(4)
        na = c9.number_input("Sodium (mmol/L)", 110.0, 160.0, d["na"])
        egfr = c10.number_input("eGFR (kidney)", 1.0, 200.0, d["egfr"])
        sbp_in = c11.number_input("Systolic BP (mmHg)", 50.0, 250.0, d["sbp"])
        submitted = st.form_submit_button("Check risk", type="primary")

    if submitted:
        nlr_val = neut / lymph
        model_inputs = {"nyha": nyha, "killip": killip, "bnp": bnp, "trop": trop, "neut": neut, "lymph": lymph,
                        "alb": alb, "hb": hbv, "na": na}
        if row_i is not None and all(model_inputs[k] == d[k] for k in model_inputs):
            # Existing patient, model values unchanged: use the risk from a model that never saw this patient
            score = cv_prob[row_i]
        else:
            x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(nlr_val), alb, hbv, na]],
                             columns=DEATH_FEATURES)
            score = model.predict_proba(x)[0, 1]
        grp = int(np.digitize(score, edges))
        names = ["Lowest", "Low", "Middle", "High", "Highest"]
        colours = [RAMP[0], RAMP[1], "#F2C14E", "#E07A5F", ALERT]
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Risk group: <span style='color:{colours[grp]}'>{names[grp]}</span>", unsafe_allow_html=True)
            kpi("⚠️", "Similar patients who died within 6 months", f"{death_rate.iloc[grp]:.1f}%")
            if row_i is not None:
                died = df.loc[row_i, "death_within_6_months"] == 1
                back = df.loc[row_i, "re_admission_within_6_months"] == 1
                st.markdown(f"<div class='found'><b>What really happened to patient {pid}:</b><br>"
                            f"Died within 6 months: <b>{'Yes' if died else 'No'}</b><br>"
                            f"Came back within 6 months: <b>{'Yes' if back else 'No'}</b></div>",
                            unsafe_allow_html=True)
            flags = [("Fluid in lungs or shock (Killip 3-4)", killip >= 3), ("Symptoms at rest (NYHA IV)", nyha == 4),
                     ("Low blood pressure (below 90)", sbp_in < 90), ("Weak kidneys (eGFR below 45)", egfr < 45),
                     ("Severe anemia (hemoglobin below 80)", hbv < 80), (f"High NLR ({nlr_val:.1f})", nlr_val >= 8.7)]
            shown = [n for n, on in flags if on]
            st.write("")
            st.markdown("**Warning signs:**")
            for n in shown:
                st.error(n)
            if not shown:
                st.success("No warning signs.")
        with right:
            st.plotly_chart(bar(names, death_rate.values, "Deaths within 6 months by risk group (%)", colours, height=320),
                            width="stretch")


# =====================================================================
# 6. KEY TAKEAWAYS & CONCLUSION
# =====================================================================
elif page == "📌 Key Takeaways & Conclusion":
    st.markdown("<div class='pagetitle'>📌 Key Takeaways</div>", unsafe_allow_html=True)
    take = ["Coming back to hospital (38.5% in 6 months) is a much bigger problem than death (2.8%)",
            "How sick the patient is today matters most: 27% of Killip 4 patients died within 6 months vs 0.8% of Killip 1",
            "Heart and organ warning signs: very high troponin, weak kidneys, high potassium, severe anemia and low sodium",
            "Simple routine tests work best: NLR is free and available for 99% of patients, while hs-CRP and blood gas were missing for half",
            "Only about 4 in 10 patients get the key long-term heart medicines (ACE inhibitor/ARB, beta-blocker)",
            "Our simple, explainable model (Logistic Regression) catches 70% of 6-month deaths; the neural network caught none"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in take)
    st.markdown(f"<div class='checkbox'><b class='h'>Key Clinical Findings:</b>{items}</div>", unsafe_allow_html=True)

    st.markdown("<div class='pagetitle' style='font-size:36px'>🏁 Conclusion</div>", unsafe_allow_html=True)
    concl = ["With tests the hospital already does on day 1 (bedside check + routine blood tests), it can spot high-risk patients early",
             "Acting on these warning signs can save lives, free ICU beds and reduce readmissions through early follow-up",
             "Limits: one hospital's data and few deaths; results show links, not proof of cause"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in concl)
    st.markdown(f"<div class='checkbox'>{items}</div>", unsafe_allow_html=True)
