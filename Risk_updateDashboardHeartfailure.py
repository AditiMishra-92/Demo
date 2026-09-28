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
.qbox {{background:white;border:2px solid #087F5B;border-radius:12px;padding:12px 16px;margin:6px 0 12px 0;font-size:16px;color:#073B4C;}}
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


def question(text):
    st.markdown(f"<div class='qbox'>❓ <b>Question:</b> {text}</div>", unsafe_allow_html=True)


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
                 color_discrete_map={"Readmitted in 6 months": READMIT, "Died in 6 months": DEATH}, title=title)
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
    st.markdown("<div style='text-align:center;font-size:48px'>❤️</div>"
                "<h2 style='text-align:center;margin:0'>Cardiac Failure</h2>"
                "<p style='text-align:center'>Team 2 • PythonPioneers</p>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["🏠 Introduction", "📘 Data Overview", "🧹 Data Cleaning & Features",
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
elif page == "🧹 Data Cleaning & Features":
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
# 4. INSIGHTS
# =====================================================================
elif page == "📊 Insights":
    st.markdown("<div class='pagetitle'>📊 Insights</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🔁", "Came back within 6 months", pct(df["re_admission_within_6_months"].mean()))
    with c2: kpi("⚠️", "Died within 6 months", pct(df["death_within_6_months"].mean()))
    with c3: kpi("❤️", "Severe symptoms (NYHA III–IV)", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))
    with c4: kpi("🧪", "Median BNP (heart strain)", f"{df['brain_natriuretic_peptide'].median():.0f} pg/mL")
    st.write("")

    d28 = "death_within_28_days"
    tabs = st.tabs(["👥 Patient Profile", "💊 Medicines", "🫘 Kidneys", "🩸 Anemia", "🩺 Blood Pressure",
                    "🛏️ Killip → Death", "🧪 NLR → Death", "🔁 Readmission Risk"])

    # ---------- Descriptive ----------
    with tabs[0]:
        badge("Descriptive")
        question("Who are our patients and how sick are they?")
        m = pd.Series({
            "High BNP (heart strain)": df["bnp_elevated_flag"].mean(),
            "Heart muscle damage": df["troponin_elevated_flag"].mean(),
            "Anemia": df["anemia_level"].isin(["Mild", "Moderate", "Severe"]).sum() / df["anemia_level"].notna().sum(),
            "Weak kidneys (eGFR < 60)": (df["glomerular_filtration_rate"] < 60).sum() / df["glomerular_filtration_rate"].notna().sum(),
            "Underweight": (df["bmi_category"] == "Underweight").mean(),
            "Diabetes": df["diabetes"].mean()}).sort_values() * 100
        fig = px.bar(m, orientation="h", text_auto=".0f", color_discrete_sequence=[TEAL2], title="How common each problem is (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="")
        st.plotly_chart(style(fig, 330), width="stretch")
        found("Mostly elderly (73% aged 69+). 92% have a strained heart and 84% heart muscle damage. "
              "Anemia (61%) is more common than diabetes (23%).")

    with tabs[1]:
        badge("Descriptive")
        question("Do patients get the recommended heart medicines?")
        drugs = pd.Series({
            "Water tablets (diuretic)": ((df["Furosemide injection"] + df["Furosemide tablet"] + df["Torasemide tablet"] +
                                          df["Hydrochlorothiazide tablet"]) > 0).mean(),
            "Spironolactone": df["Spironolactone tablet"].mean(),
            "ACE inhibitor / ARB": ((df["Benazepril hydrochloride tablet"] + df["Valsartan Dispersible tablet"]) > 0).mean(),
            "Beta-blocker": ((df["Metoprolol Succinate Sustained-release tablet"] + df["metoprolol tartrate injection"]) > 0).mean(),
        }).sort_values() * 100
        fig = px.bar(drugs, orientation="h", text_auto=".0f", color_discrete_sequence=[GREEN], title="Medicines given (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="")
        st.plotly_chart(style(fig, 300), width="stretch")
        found("96% get water tablets, but only about 4 in 10 get the key long-term heart medicines.")
        todo("Check every suitable patient is on the recommended medicines before going home.")

    # ---------- Prescriptive ----------
    with tabs[2]:
        badge("Prescriptive")
        question("Do weaker kidneys lead to more deaths and readmissions?")
        g = df.dropna(subset=["ckd_stage"]).groupby("ckd_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Outcomes by kidney stage (worse →)"), width="stretch")
        found("Deaths rise from 1.6% (healthy kidneys) to 9.2% (kidney failure). Readmission peaks at 50%.")
        todo("eGFR below 45 = high risk: check potassium and see the patient within 2 weeks of discharge.")

    with tabs[3]:
        badge("Prescriptive")
        question("Does anemia (low hemoglobin) raise the risk?")
        g = df.dropna(subset=["anemia_level"]).groupby("anemia_level", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Outcomes by anemia level"), width="stretch")
        found("Mild and moderate anemia add little risk. Severe anemia (Hb below 80) nearly triples deaths (6.8% vs 2.5%).")
        todo("Flag hemoglobin below 80 at admission and correct it.")

    with tabs[4]:
        badge("Prescriptive")
        question("Is low blood pressure at admission dangerous?")
        g = df.dropna(subset=["bp_stage"]).groupby("bp_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Outcomes by blood pressure group"), width="stretch")
        found("Only 20 patients had BP below 90, but 9 in 10 had symptoms at rest and 8 in 10 had fluid in the lungs or shock.")
        todo("Treat BP below 90 as possible shock: move to close monitoring.")

    # ---------- Predictive ----------
    with tabs[5]:
        badge("Predictive")
        question("Can a quick bedside check at admission (Killip grade) predict who dies within 28 days?")
        kil = df.groupby("killip_grade")[d28].mean() * 100
        st.plotly_chart(bar([f"Killip {x}" for x in kil.index], kil.values, "Deaths within 28 days by Killip grade (%)",
                            RAMP[1:], height=340), width="stretch")
        auc_k = roc_auc_score(df[d28], cv_probs(df, ["killip_grade"], d28, "Logistic Regression", repeats=3))
        found(f"None of 527 Killip 1 patients died; 1 in 4 Killip 4 patients died. Model score (ROC-AUC) {auc_k:.2f}. "
              "Old diagnoses (past heart attack) did not help: about 2% deaths either way.")
        todo("Killip 1 can go to a normal ward; Killip 3–4 need close monitoring.")

    with tabs[6]:
        badge("Predictive")
        question("Can a free inflammation marker from the routine blood count (NLR) predict early death?")
        q = pd.qcut(df["nlr"], 4, labels=["Lowest NLR", "Low", "High", "Highest NLR"])
        nq = df.groupby(q, observed=True)[d28].mean() * 100
        st.plotly_chart(bar(list(nq.index.astype(str)), nq.values, "Deaths within 28 days by NLR level (%)",
                            RAMP[1:], height=340), width="stretch")
        auc_n = roc_auc_score(df[d28], cv_probs(df, ["nlr_log"], d28, "Logistic Regression", repeats=3))
        found(f"Model score (ROC-AUC) {auc_n:.2f}. Highest NLR group: 8 times the early deaths of the lowest (3.4% vs 0.4%). "
              "The special test hs-CRP did not help: over half the patients never had it.")
        todo("Calculate NLR for every patient and flag NLR of 8.7 or more.")

    with tabs[7]:
        badge("Predictive")
        question("Can admission data predict who will come back to hospital within 6 months?")
        alive = df[(df["outcome_during_hospitalization"] != "Dead") & (df["death_within_6_months"] == 0)].reset_index(drop=True)
        with st.spinner("Scoring patients..."):
            prob = cv_probs(alive, READMIT_FEATURES, "re_admission_within_6_months", "Logistic Regression", repeats=3)
        groups = pd.qcut(prob, 5, labels=["Lowest risk", "Low", "Middle", "High", "Highest risk"])
        by_g = alive.groupby(groups, observed=True)["re_admission_within_6_months"].mean() * 100
        st.plotly_chart(bar(list(by_g.index.astype(str)), by_g.values, "Patients who came back, by predicted risk group (%)",
                            RAMP, height=340), width="stretch")
        auc_r = roc_auc_score(alive["re_admission_within_6_months"], prob)
        found(f"Harder to predict (ROC-AUC {auc_r:.2f}), but the highest-risk group came back about twice as often "
              f"({by_g.iloc[-1]:.0f}% vs {by_g.iloc[0]:.0f}%). Drivers: severe symptoms, weak kidneys, other diseases.")
        todo("Give the highest-risk group a follow-up call and an early clinic visit.")


# =====================================================================
# 5. MODEL PERFORMANCE
# =====================================================================
elif page == "🤖 Model Performance":
    st.markdown("<div class='pagetitle'>🤖 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='found'>Models use <b>only admission data</b> and are scored on <b>patients they never saw</b> "
                "(5-fold cross-validation). <b>ROC-AUC</b>: 0.5 = coin toss, 1.0 = perfect. "
                "<b>Recall</b>: share of real cases the model caught. Accuracy is misleading: only 3% die, "
                "so saying \"nobody dies\" is 97% accurate and useless.</div>", unsafe_allow_html=True)

    targets = {"6-month death": ("death_within_6_months", DEATH_FEATURES, df),
               "28-day death": ("death_within_28_days", DEATH_FEATURES, df),
               "6-month readmission": ("re_admission_within_6_months", READMIT_FEATURES,
                                       df[(df["outcome_during_hospitalization"] != "Dead") &
                                          (df["death_within_6_months"] == 0)].reset_index(drop=True))}
    choice = st.selectbox("Outcome to predict", list(targets.keys()))
    target, feats, data = targets[choice]
    y = data[target].values

    with st.spinner("Testing 3 models..."):
        rows, curves = [], {}
        for name in model_set():
            p = cv_probs(data, feats, target, name)
            pred = (p >= 0.5).astype(int)
            curves[name] = roc_curve(y, p)
            rows.append([name, roc_auc_score(y, p), recall_score(y, pred, zero_division=0),
                         precision_score(y, pred, zero_division=0), accuracy_score(y, pred)])
        rows.append(["Baseline: predict 'no' for everyone", 0.5, 0.0, 0.0, 1 - y.mean()])
    res = pd.DataFrame(rows, columns=["Model", "ROC-AUC", "Recall", "Precision", "Accuracy"])
    lr = res.set_index("Model").loc["Logistic Regression"]
    ann = res.set_index("Model").loc["ANN (neural network)"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🏆", "Chosen model", "Logistic Regression", size=20)
    with c2: kpi("📈", "ROC-AUC", f"{lr['ROC-AUC']:.2f}")
    with c3: kpi("🎯", "Real cases caught (recall)", pct(lr["Recall"]))
    with c4: kpi("👥", "Patients with outcome", f"{int(y.sum())} of {len(y):,}")
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(res.style.format({c: "{:.2f}" for c in ["ROC-AUC", "Recall", "Precision", "Accuracy"]}),
                     hide_index=True, width="stretch")
        found(f"All three models rank patients about equally well, but Logistic Regression <b>catches {lr['Recall']*100:.0f}%</b> "
              f"of real cases while the neural network catches {ann['Recall']*100:.0f}%. "
              f"Logistic Regression is also easy to explain, so we chose it.")
    with right:
        fig = go.Figure()
        for name, colr in zip(curves, [GREEN, BLUE, NAVY]):
            fpr, tpr, _ = curves[name]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=colr, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin toss", line=dict(color=ALERT, dash="dash")))
        fig.update_layout(title="ROC curve (higher = better)", xaxis_title="False alarms", yaxis_title="Real cases caught")
        st.plotly_chart(style(fig, 360), width="stretch")

    with st.expander("🩺 Try it: Patient Risk Check (6-month death model)"):
        @st.cache_resource
        def final_model():
            model = logistic().fit(df[DEATH_FEATURES].astype(float), df["death_within_6_months"])
            prob = cv_probs(df, DEATH_FEATURES, "death_within_6_months", "Logistic Regression", repeats=3)
            edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
            rate = pd.Series(df["death_within_6_months"].values).groupby(np.digitize(prob, edges)).mean() * 100
            return model, edges, rate, prob

        model, edges, death_rate, cv_prob = final_model()

        # ---- Choose an existing patient (values fill in automatically) or enter a new one ----
        ids = ["New patient (enter values)"] + sorted(df["inpatient_number"].astype(int).tolist())
        pid = st.selectbox("Patient ID", ids, help="Pick a patient from our data to fill in their admission values, "
                                                    "or choose 'New patient' and type the values.")
        med = df[DEATH_FEATURES + ["brain_natriuretic_peptide", "high_sensitivity_troponin", "neutrophil_count",
                                   "lymphocyte_count"]].median()
        if pid == ids[0]:
            d = {"nyha": 3, "killip": 2, "bnp": 750.0, "trop": 55.0, "neut": 5.0, "lymph": 1.0,
                 "alb": 37.0, "hb": 115.0, "na": 139.0}
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
                 "alb": val("albumin", 10, 60), "hb": val("hemoglobin", 30, 200), "na": val("sodium", 110, 160)}

        with st.form(f"patient_{pid}"):
            c1, c2, c3, c4 = st.columns(4)
            nyha = c1.selectbox("NYHA class", [1, 2, 3, 4], index=[1, 2, 3, 4].index(d["nyha"]))
            killip = c2.selectbox("Killip grade", [1, 2, 3, 4], index=d["killip"] - 1)
            bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, d["bnp"])
            trop = c4.number_input("Troponin (pg/mL)", 0.0, 50000.0, d["trop"])
            c5, c6, c7, c8 = st.columns(4)
            neut = c5.number_input("Neutrophils", 0.1, 50.0, d["neut"])
            lymph = c6.number_input("Lymphocytes", 0.05, 20.0, d["lymph"])
            alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, d["alb"])
            hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, d["hb"])
            na = st.number_input("Sodium (mmol/L)", 110.0, 160.0, d["na"])
            submitted = st.form_submit_button("Check risk", type="primary")

        if submitted:
            entered = {"nyha": nyha, "killip": killip, "bnp": bnp, "trop": trop, "neut": neut, "lymph": lymph,
                       "alb": alb, "hb": hbv, "na": na}
            if row_i is not None and entered == d:
                # Existing patient, values unchanged: use the risk from a model that never saw this patient
                score = cv_prob[row_i]
            else:
                x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(neut / lymph), alb, hbv, na]],
                                 columns=DEATH_FEATURES)
                score = model.predict_proba(x)[0, 1]
            grp = int(np.digitize(score, edges))
            names = ["Lowest", "Low", "Middle", "High", "Highest"]
            colours = [RAMP[0], RAMP[1], "#F2C14E", "#E07A5F", ALERT]
            a, b = st.columns([1, 1.4])
            with a:
                st.markdown(f"### Risk group: <span style='color:{colours[grp]}'>{names[grp]}</span>", unsafe_allow_html=True)
                kpi("⚠️", "Similar patients who died within 6 months", f"{death_rate.iloc[grp]:.1f}%")
                if row_i is not None:
                    died = df.loc[row_i, "death_within_6_months"] == 1
                    back = df.loc[row_i, "re_admission_within_6_months"] == 1
                    st.write("")
                    st.markdown(f"<div class='found'><b>What really happened to patient {pid}:</b><br>"
                                f"Died within 6 months: <b>{'Yes' if died else 'No'}</b><br>"
                                f"Came back within 6 months: <b>{'Yes' if back else 'No'}</b></div>",
                                unsafe_allow_html=True)
            with b:
                st.plotly_chart(bar(names, death_rate.values, "Deaths within 6 months by risk group (%)", colours, height=280),
                                width="stretch")


# =====================================================================
# 6. KEY TAKEAWAYS & CONCLUSION
# =====================================================================
elif page == "📌 Key Takeaways & Conclusion":
    st.markdown("<div class='pagetitle'>📌 Key Takeaways</div>", unsafe_allow_html=True)
    take = ["Coming back to hospital (38.5%) is a bigger problem than death (2.8%)",
            "A quick bedside check (Killip grade) at admission spots most patients who will die",
            "Weak kidneys, severe anemia and low blood pressure raise the risk the most",
            "NLR from the routine blood count is a free, useful early-warning test",
            "Only 4 in 10 patients get the key long-term heart medicines",
            "A simple, explainable model (Logistic Regression) worked best"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in take)
    st.markdown(f"<div class='checkbox'><b class='h'>Key Clinical Findings:</b>{items}</div>", unsafe_allow_html=True)

    st.markdown("<div class='pagetitle' style='font-size:36px'>🏁 Conclusion</div>", unsafe_allow_html=True)
    concl = ["With tests the hospital already does on day 1, it can spot high-risk heart failure patients early",
             "Acting on these warning signs can save lives, free ICU beds and reduce readmissions",
             "Limits: one hospital's data, few deaths; results show links, not proof of cause"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in concl)
    st.markdown(f"<div class='checkbox'>{items}</div>", unsafe_allow_html=True)
