# =====================================================================
#  Heart Failure Risk Dashboard  |  Team 2 - Python Pioneers
#  NumpyNinja Python Hackathon - Category 5 (Demo of insights in Python)
#
#  How to run:
#     pip install streamlit plotly pandas numpy scipy scikit-learn
#     streamlit run app.py
#  Keep Cardiac_Cleaned_Data.csv in the same folder as this file.
# =====================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title="Heart Failure Risk Dashboard", layout="wide")

# ---------- Same colour system as our notebooks ----------
BLUE = "#2a78d6"      # readmission / single series
ORANGE = "#eb6834"    # death
REF = "#d03b3b"       # cut-off lines
GREY = "#b8b6b0"      # "normal" / comparison group
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # light = better, dark = worse

DATA_FILE = "Cardiac_Cleaned_Data.csv"


# =====================================================================
# 1. LOAD DATA + create the group columns used in the notebooks
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    # Groups from the prescriptive notebook
    stage_order = ["G1 (>=90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"]
    df["ckd_stage"] = pd.cut(df["glomerular_filtration_rate"], bins=[0, 15, 30, 45, 60, 90, 1000],
                             right=False, labels=stage_order[::-1])
    df["ckd_stage"] = pd.Categorical(df["ckd_stage"], categories=stage_order, ordered=True)

    anemia_order = ["No anemia", "Mild", "Moderate", "Severe"]
    cut = np.where(df["gender"] == "Male", 130, 120)
    hb = df["hemoglobin"]
    anemia = np.select([hb.isna(), hb >= cut, hb >= 110, hb >= 80],
                       ["Missing", "No anemia", "Mild", "Moderate"], default="Severe")
    df["anemia_level"] = pd.Categorical(pd.Series(anemia).replace("Missing", np.nan),
                                        categories=anemia_order, ordered=True)

    sbp, dbp = df["systolic_blood_pressure"], df["diastolic_blood_pressure"]
    bp_order = ["Low (<90)", "Normal (<120/80)", "Elevated (120-129)", "Stage 1 (130-139/80-89)", "Stage 2 (>=140/90)"]
    bp = np.select([sbp.isna(), sbp < 90, (sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
                   ["Missing", bp_order[0], bp_order[4], bp_order[3], bp_order[2]], default=bp_order[1])
    df["bp_stage"] = pd.Categorical(pd.Series(bp).replace("Missing", np.nan), categories=bp_order, ordered=True)

    df["nyha_iv"] = (df["nyha_cardiac_function_classification"] == 4).astype(int)
    df["killip_3_4"] = (df["killip_grade"] >= 3).astype(int)
    df["on_inotrope"] = (df[["Milrinone injection", "Dobutamine hydrochloride injection",
                             "Isoprenaline Hydrochloride injection"]].sum(axis=1) > 0).astype(int)

    # Predictors from the predictive notebook
    df["age"] = df["agecat"].apply(lambda s: (int(s.split("-")[0]) + int(s.split("-")[1])) / 2)
    df["male"] = (df["gender"] == "Male").astype(int)
    df["unconscious"] = (df["consciousness"] != "Clear").astype(int)
    df["right_or_both_hf"] = (df["type_of_heart_failure"] != "Left").astype(int)
    df["nlr"] = df["neutrophil_count"] / df["lymphocyte_count"]
    df["nlr_log"] = np.log(df["nlr"])
    df["wbc_log"] = np.log(df["white_blood_cell"])
    df["troponin_log"] = np.log1p(df["high_sensitivity_troponin"])
    df["hs_crp_missing"] = df["hs_crp"].isna().astype(int)

    # In-hospital death = died in hospital OR left against advice and died within 2 days
    left_and_died = (df["outcome_during_hospitalization"] == "DischargeAgainstOrder") & (df["death_within_28_days"] == 1)
    days_after = df["time_of_death__days_from_admission"] - df["dischargeday"]
    df["in_hospital_death"] = ((df["outcome_during_hospitalization"] == "Dead") |
                               (left_and_died & (days_after <= 2))).astype(int)
    return df


# =====================================================================
# 2. SMALL HELPER FUNCTIONS
# =====================================================================
def pct(x):
    return f"{x * 100:.1f}%"


def outcome_chart(table, title):
    """Readmission (blue) and death (orange) % for each group - same as notebook charts."""
    long = table.reset_index().melt(id_vars=table.index.name, var_name="Outcome", value_name="Percent")
    fig = px.bar(long, x=table.index.name, y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Readmission 6m (%)": BLUE, "Death 6m (%)": ORANGE}, title=title)
    fig.update_layout(yaxis_title="% of patients", xaxis_title="", legend_title="", height=420)
    return fig


def chi2_test(a, b):
    """Chi-square test + Cramer's V."""
    t = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(t)
    v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
    return chi2, p, v


def fisher_test(flag, outcome):
    t = pd.crosstab(flag, outcome).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    return stats.fisher_exact(t)


def show_evidence(rows):
    """Nice table of statistical tests."""
    out = pd.DataFrame(rows, columns=["What we tested", "Test", "Result", "p-value"])
    out["Significant?"] = out["p-value"].apply(lambda p: "Yes" if p < 0.05 else "No")
    out["p-value"] = out["p-value"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    st.dataframe(out, hide_index=True, width="stretch")


def make_model():
    """Same model settings as the predictive notebook."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))


@st.cache_data
def cv_score(data, features, target, repeats=3):
    """Repeated stratified 5-fold cross-validation. Returns ROC-AUC, PR-AUC and each patient's risk."""
    X = data[features].astype(float)
    y = data[target]
    aucs, praucs, probs = [], [], []
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
        praucs.append(average_precision_score(y, p))
        probs.append(p)
    return round(np.mean(aucs), 3), round(np.mean(praucs), 3), np.mean(probs, axis=0)


def insight_box(text):
    st.info(text)


def action_box(text):
    st.success("**Recommendation:** " + text)


df = load_data()

# =====================================================================
# 3. SIDEBAR - navigation + filters
# =====================================================================
st.sidebar.title("Heart Failure Dashboard")
st.sidebar.caption("Team 2 - Python Pioneers")
page = st.sidebar.radio("Go to", [
    "1. Introduction",
    "2. Data Cleaning & Feature Engineering",
    "3. Patient Profile (Descriptive)",
    "4. What Drives Risk (Prescriptive)",
    "5. Predicting Outcomes (Predictive)",
    "6. Patient Risk Check",
    "7. Key Takeaways & Conclusion",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters (pages 3 and 4)")
gender_pick = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique()),
                                     default=sorted(df["gender"].dropna().unique()))
age_pick = st.sidebar.multiselect("Age group", sorted(df["agecat"].dropna().unique()),
                                  default=sorted(df["agecat"].dropna().unique()))
dff = df[df["gender"].isin(gender_pick) & df["agecat"].isin(age_pick)]
st.sidebar.write(f"Patients selected: **{len(dff)}** of {len(df)}")


# =====================================================================
# PAGE 1 - INTRODUCTION
# =====================================================================
if page == "1. Introduction":
    st.title("Heart Failure: Who Comes Back, Who Is at Risk, and What Can the Hospital Do?")

    st.markdown("""
**The problem.** Heart failure patients are admitted very sick, and many return to hospital soon after
discharge. Each readmission costs the hospital money and bed space, and each early death may have been
preventable with closer monitoring. The hospital needs to know **which patients to watch closely at
admission and after discharge**.

**What this dashboard does.** It turns the data of 2,008 hospitalised heart failure patients into
answers a care team can act on:
- Which patients are most likely to be readmitted or die within 6 months?
- Which organ markers (kidney, blood, blood pressure) raise that risk?
- Can we flag high-risk patients **at admission**, before it is too late?

**Who can use it.** Cardiologists and ward doctors (triage and ICU decisions), discharge nurses
(follow-up planning) and hospital managers (bed planning and quality targets).
""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Readmitted within 6 months", pct(df["re_admission_within_6_months"].mean()))
    c3.metric("Died within 6 months", pct(df["death_within_6_months"].mean()))
    c4.metric("Admitted as NYHA III-IV", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))

    st.subheader("About the data")
    st.markdown("""
Seven linked tables (joined on `inpatient_number`) from a hospital heart failure registry:

| Table | What it holds |
|---|---|
| Demography | Gender, age group, weight, height, BMI, occupation |
| Cardiac | NYHA class, Killip grade, heart failure type, echo measurements |
| History | Other diseases (diabetes, kidney disease, COPD...) and CCI score |
| Hospitalization | Admission, stay length, death and readmission outcomes |
| Labs | 100+ blood tests and vital signs at admission |
| Responsiveness | Glasgow Coma Scale and consciousness |
| Prescriptions | 25 drugs given during the stay |
""")

    st.subheader("How to read this dashboard")
    st.markdown("""
1. **Data Cleaning** - what we fixed and why the numbers can be trusted.
2. **Patient Profile** - who the patients are (baseline).
3. **What Drives Risk** - kidney function, anemia and blood pressure vs outcomes, with statistical tests.
4. **Predicting Outcomes** - machine learning models tested on patients they have not seen.
5. **Patient Risk Check** - enter a new patient's admission values and see their risk group.
6. **Key Takeaways** - the answer to "so what?" for the hospital.

Colours are the same everywhere: **blue = readmission**, **orange = death**, darker blue = more severe.
""")


# =====================================================================
# PAGE 2 - DATA CLEANING & FEATURE ENGINEERING
# =====================================================================
elif page == "2. Data Cleaning & Feature Engineering":
    st.title("Data Cleaning & Feature Engineering")
    st.markdown("We cleaned 7 raw tables into **one table with one row per patient** "
                "(2,008 patients, 210 columns). Every step had a clinical reason.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw tables", "7")
    c2.metric("Cleaning steps", "27")
    c3.metric("New features created", "16")
    c4.metric("Duplicate patients", "0")

    st.subheader("Before vs after: why cleaning mattered")
    ba = pd.DataFrame({
        "Problem": ["Fake patient (ID 5) in demography", "Impossible weight/height (0 kg, 0.35 m)",
                    "Troponin in wrong unit (ng/mL)", "Mitral E/A typed in cm/s",
                    "E/A ratio blank though E and A exist", "Prescriptions: many rows per patient",
                    "BNP stuck at machine limit (5000)"],
        "Before": ["2,009 rows", "BMI up to 404", "0.3% show heart damage", "Mean E = 4.85 m/s (impossible)",
                   "393 patients with E/A", "15,362 rows (1 patient counted ~8 times)", "Looks like a normal value"],
        "After": ["2,008 rows, matches all tables", "BMI 13-39, mean 21", "84% show heart damage (expected)",
                  "Mean E = 1.06 m/s (normal)", "536 patients (+36%)", "1 row per patient, 25 drug columns",
                  "Flag bnp_at_limit (100 patients)"],
    })
    st.dataframe(ba, hide_index=True, width="stretch")

    tab1, tab2, tab3 = st.tabs(["Cleaning steps", "Feature engineering", "Checks before modelling"])
    with tab1:
        steps = pd.DataFrame({
            "Group": ["Remove wrong data", "Remove wrong data", "Fix units", "Fix units", "Fix units",
                      "Fill blanks (only when meaning is clear)", "Fill blanks (only when meaning is clear)",
                      "Fill blanks (only when meaning is clear)", "Keep blanks on purpose", "Keep blanks on purpose",
                      "Standardise", "Standardise", "Reshape and merge"],
            "What we did": ["Dropped fake ID 5; set weight <20 kg and height <1 m to null, recalculated BMI",
                            "Set impossible values to null: zero pulse/BP, systolic < diastolic, LVEDD <10, tricuspid pressure <10",
                            "LV diameter renamed to mm", "Mitral E and A cm/s -> m/s",
                            "Hematocrit fraction -> %, troponin ng/mL -> pg/mL",
                            "Occupation blank -> 'Unknown'", "Respiratory support blank -> 'None' (no ventilation)",
                            "E/A ratio calculated from E / A (exact, not a guess)",
                            "Time of death / readmission time (blank = event did not happen)",
                            "Lab tests not done (sicker patients were tested more, so mean-filling would mislead)",
                            "Text case, 1/0 for Type II respiratory failure, clear lab column names",
                            "Admission date text -> date",
                            "Prescriptions pivoted to one row per patient, then all 7 tables left-merged"],
        })
        st.dataframe(steps, hide_index=True, width="stretch")
        st.caption("Columns kept but not analysed: leukemia (everyone = 0), cholinesterase (all blank), "
                   "body_temperature_blood_gas (always 37, machine default).")

        miss = (df[["lvef", "hs_crp", "ea", "brain_natriuretic_peptide", "hemoglobin",
                    "glomerular_filtration_rate", "albumin", "killip_grade"]].isna().mean() * 100).sort_values()
        fig = px.bar(miss, orientation="h", text_auto=".1f", title="Missing values kept on purpose (% of patients)",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(showlegend=False, xaxis_title="% missing", yaxis_title="", height=380)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fe = pd.DataFrame({
            "New feature": ["bmi_category, obesity_flag", "bp_category", "gcs_category, gcs_impaired_flag",
                            "lvedd_enlarged_flag", "ea_category", "respiratory_support_flag",
                            "total_drugs, polypharmacy_flag, medication_burden", "troponin_elevated_flag",
                            "bnp_log, bnp_elevated_flag", "hs_crp_log, hs_crp_inflammation_flag", "comorbidity_count",
                            "NLR (neutrophil / lymphocyte), age midpoint (model notebook)"],
            "Rule used": ["WHO BMI cut-offs", "SBP <90 low, >=140 or DBP >=90 high", "GCS 15 normal ... <=8 severe",
                          "LVEDD > 56 mm", "E/A <0.8 / 0.8-1.5 / >1.5", "IMV or NIMV given",
                          "5+ drugs = polypharmacy", "> 14 pg/mL", "log for skew, >100 pg/mL high",
                          "log for skew, >3 mg/L high", "Count of 6 key diseases", "Inflammation marker from routine blood count"],
            "Why it helps": ["Groups are easier to compare than raw numbers", "Tests the 'low BP = pump failure' idea",
                             "Flags brain hypoperfusion", "Enlarged heart = worse failure", "Heart filling pattern",
                             "Marks the sickest patients", "Treatment burden per patient", "Heart muscle damage",
                             "Stops a few huge values from controlling models", "Same, plus inflammation flag",
                             "Total disease burden", "Available for 98.7% of patients"],
        })
        st.dataframe(fe, hide_index=True, width="stretch")
        col = st.selectbox("See how a feature splits the patients",
                           ["bmi_category", "bp_category", "gcs_category", "ea_category", "medication_burden"])
        counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="Patients")
        counts[col] = counts[col].astype(str)
        st.plotly_chart(px.bar(counts, x=col, y="Patients", text_auto=True, color_discrete_sequence=[BLUE]),
                        width="stretch")

    with tab3:
        st.markdown("""
- **Leakage check:** outcome columns (death, readmission, emergency return, discharge destination),
  discharge day, and all drugs / oxygen / ventilation given *during* the stay are **never** used as
  predictors, because they are not known at admission.
- **One row per patient:** 2,008 rows = 2,008 unique IDs, no duplicate rows.
- **Model-time filling:** missing lab values are filled with the median **inside** each training fold,
  so test patients never leak into training.
""")


# =====================================================================
# PAGE 3 - PATIENT PROFILE (DESCRIPTIVE)
# =====================================================================
elif page == "3. Patient Profile (Descriptive)":
    st.title("Who Are the Patients? (Baseline)")
    if len(dff) == 0:
        st.warning("No patients match the filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(dff))
    c2.metric("Aged 69+", pct(dff["agecat"].isin(["69-79", "79-89", "89-110"]).mean()))
    c3.metric("Female", pct((dff["gender"] == "Female").mean()))
    c4.metric("Median BNP (pg/mL)", f"{dff['brain_natriuretic_peptide'].median():.0f}")

    # Outcomes over time
    out = pd.DataFrame({
        "Time": ["28 days", "3 months", "6 months"] * 2,
        "Outcome": ["Readmission"] * 3 + ["Death"] * 3,
        "Percent": [dff[c].mean() * 100 for c in ["re_admission_within_28_days", "re_admission_within_3_months",
                                                   "re_admission_within_6_months", "death_within_28_days",
                                                   "death_within_3_months", "death_within_6_months"]]})
    fig = px.bar(out, x="Time", y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Readmission": BLUE, "Death": ORANGE},
                 title="Readmission is the bigger problem: it climbs fast after discharge")
    fig.update_layout(yaxis_title="% of patients", height=400)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        ag = dff.groupby(["agecat", "gender"]).size().reset_index(name="Patients")
        fig = px.bar(ag, x="agecat", y="Patients", color="gender", barmode="group",
                     color_discrete_map={"Female": BLUE, "Male": ORANGE}, title="Age and gender")
        fig.update_layout(xaxis_title="Age group", height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        sev = pd.crosstab(dff["nyha_cardiac_function_classification"], dff["killip_grade"])
        sev.index = [f"NYHA {i}" for i in sev.index]
        sev.columns = [f"Killip {c}" for c in sev.columns]
        fig = px.imshow(sev, text_auto=True, color_continuous_scale=["#f4f8fd", BLUE, "#0d366b"],
                        title="Severity at admission (NYHA x Killip)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        m = pd.Series({
            "High BNP (>100)": dff["bnp_elevated_flag"].mean(),
            "High troponin (>14)": dff["troponin_elevated_flag"].mean(),
            "Anemia": dff["anemia_level"].isin(["Mild", "Moderate", "Severe"]).sum() / dff["anemia_level"].notna().sum(),
            "Reduced kidney (eGFR<60)": (dff["glomerular_filtration_rate"] < 60).sum() / dff["glomerular_filtration_rate"].notna().sum(),
            "Underweight (BMI<18.5)": (dff["bmi_category"] == "Underweight").mean(),
            "Diabetes": dff["diabetes"].mean(),
        }).sort_values() * 100
        fig = px.bar(m, orientation="h", text_auto=".0f", color_discrete_sequence=[BLUE],
                     title="How common are the key problems? (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="", height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        drugs = pd.Series({
            "Diuretic": ((dff["Furosemide injection"] + dff["Furosemide tablet"] + dff["Torasemide tablet"] +
                          dff["Hydrochlorothiazide tablet"]) > 0).mean(),
            "Spironolactone (MRA)": dff["Spironolactone tablet"].mean(),
            "ACEi / ARB": ((dff["Benazepril hydrochloride tablet"] + dff["Valsartan Dispersible tablet"]) > 0).mean(),
            "Beta-blocker": ((dff["Metoprolol Succinate Sustained-release tablet"] +
                              dff["metoprolol tartrate injection"]) > 0).mean(),
        }).sort_values() * 100
        fig = px.bar(drugs, orientation="h", text_auto=".0f", color_discrete_sequence=[RAMP[3]],
                     title="Guideline medicines given (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="", height=380)
        st.plotly_chart(fig, width="stretch")

    insight_box("Most patients are elderly (73% aged 69+), 82% arrive in NYHA class III-IV, and 92% have high BNP. "
                "Death is low (2.8% by 6 months) but **38.5% are readmitted within 6 months**. "
                "Only about 38% get an ACE inhibitor/ARB or a beta-blocker, so guideline treatment is a gap.")


# =====================================================================
# PAGE 4 - WHAT DRIVES RISK (PRESCRIPTIVE)
# =====================================================================
elif page == "4. What Drives Risk (Prescriptive)":
    st.title("What Drives Risk? Kidney, Blood and Blood Pressure")
    st.markdown("Each tab tests one organ-level marker against heart failure severity and outcomes, "
                "with statistical evidence from our data. Use the sidebar filters to check a sub-group.")
    if len(dff) < 50:
        st.warning("Too few patients selected for reliable tests. Widen the filters.")
        st.stop()

    t1, t2, t3 = st.tabs(["P1. Kidney function", "P2. Anemia", "P3. Blood pressure"])

    # ---------- P1 Kidney ----------
    with t1:
        st.subheader("Does worse kidney function go with more severe heart failure and worse outcomes?")
        st.caption("Why: in cardiorenal syndrome a weak heart starves the kidneys, and weak kidneys hold salt and "
                   "water, which loads the heart more (higher BNP, worse NYHA).")
        k = dff.dropna(subset=["ckd_stage"])
        g = k.groupby("ckd_stage", observed=True)
        outc = pd.DataFrame({"Readmission 6m (%)": g["re_admission_within_6_months"].mean() * 100,
                             "Death 6m (%)": g["death_within_6_months"].mean() * 100})
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(outcome_chart(outc, "Readmission and death by CKD stage (worse ->)"), width="stretch")
        with right:
            prof = pd.DataFrame({"Patients": g.size(), "Median BNP": g["brain_natriuretic_peptide"].median(),
                                 "NYHA IV (%)": g["nyha_iv"].mean() * 100,
                                 "High K+ >5 (%)": g["potassium"].apply(lambda s: (s > 5).mean() * 100),
                                 "Median urea": g["urea"].median()}).round(1)
            st.markdown("**Severity profile by CKD stage**")
            st.dataframe(prof, width="stretch")
        rows = []
        for col, name in [("brain_natriuretic_peptide", "eGFR vs BNP"), ("potassium", "eGFR vs potassium"),
                          ("urea", "eGFR vs urea")]:
            d = k[["glomerular_filtration_rate", col]].dropna()
            r, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            rows.append((name, "Spearman", f"rho = {r:.2f}", p))
        for col, name in [("nyha_iv", "CKD stage vs NYHA IV"), ("re_admission_within_6_months", "CKD stage vs readmission"),
                          ("death_within_6_months", "CKD stage vs death")]:
            chi2, p, v = chi2_test(k["ckd_stage"], k[col])
            rows.append((name, "Chi-square", f"Cramer's V = {v:.2f}", p))
        show_evidence(rows)
        insight_box("6-month death rises from about 1.6% at G1 to 9% at G5, and readmission climbs from 31% to 50% at G3b. "
                    "Readmission drops at G4-G5 because more of these patients die first (competing risk).")
        action_box("Treat eGFR < 45 (G3b or worse) as high risk: careful diuretic dosing, potassium checks, "
                   "and follow-up within 2 weeks of discharge.")

    # ---------- P2 Anemia ----------
    with t2:
        st.subheader("Is anemia linked to worse symptoms and outcomes, and does severe anemia stand out?")
        st.caption("Why: hemoglobin carries oxygen. With anemia, a weak heart must pump harder to deliver the same oxygen.")
        a = dff.dropna(subset=["anemia_level"]).copy()
        g = a.groupby("anemia_level", observed=True)
        outc = pd.DataFrame({"Readmission 6m (%)": g["re_admission_within_6_months"].mean() * 100,
                             "Death 6m (%)": g["death_within_6_months"].mean() * 100})
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(outcome_chart(outc, "Readmission and death by anemia level (WHO)"), width="stretch")
        with right:
            prof = pd.DataFrame({"Patients": g.size(), "NYHA IV (%)": g["nyha_iv"].mean() * 100,
                                 "Median BNP": g["brain_natriuretic_peptide"].median(),
                                 "Median stay (days)": g["dischargeday"].median()}).round(1)
            st.markdown("**Profile by anemia level**")
            st.dataframe(prof, width="stretch")
        rows = []
        for col, name in [("nyha_iv", "Anemia level vs NYHA IV"), ("re_admission_within_6_months", "Anemia level vs readmission"),
                          ("death_within_6_months", "Anemia level vs death")]:
            chi2, p, v = chi2_test(a["anemia_level"], a[col])
            rows.append((name, "Chi-square", f"Cramer's V = {v:.2f}", p))
        a["severe"] = (a["anemia_level"] == "Severe").astype(int)
        orr, p = fisher_test(a["severe"], a["death_within_6_months"])
        rows.append(("Severe anemia vs rest: death 6m", "Fisher exact", f"Odds ratio = {orr:.1f}", p))
        show_evidence(rows)
        insight_box("Mixed result, reported honestly: anemia as a general label is **not** significantly linked to outcomes. "
                    "But **severe anemia (Hb < 80 g/L)** has about 3x the 6-month death rate of the other levels.")
        action_box("Flag Hb < 80 at admission for correction (iron studies, transfusion if needed) and closer follow-up.")

    # ---------- P3 BP ----------
    with t3:
        st.subheader("Do patients with low blood pressure have more congestion, shock and worse outcomes?")
        st.caption("Why: blood pressure is the output of the heart's pump. When the pump fails badly, BP falls "
                   "and fluid backs up into the lungs.")
        b = dff.dropna(subset=["bp_stage"]).copy()
        b["low_bp"] = (b["bp_stage"] == "Low (<90)").astype(int)
        g = b.groupby("bp_stage", observed=True)
        outc = pd.DataFrame({"Readmission 6m (%)": g["re_admission_within_6_months"].mean() * 100,
                             "Death 6m (%)": g["death_within_6_months"].mean() * 100})
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(outcome_chart(outc, "Readmission and death by BP stage"), width="stretch")
        with right:
            comp = pd.DataFrame({
                "Group": ["Low BP (SBP<90)"] * 3 + ["All other BP"] * 3,
                "Measure": ["Killip III-IV", "NYHA IV", "IV inotrope"] * 2,
                "Percent": list(b[b.low_bp == 1][["killip_3_4", "nyha_iv", "on_inotrope"]].mean() * 100) +
                           list(b[b.low_bp == 0][["killip_3_4", "nyha_iv", "on_inotrope"]].mean() * 100)})
            fig = px.bar(comp, x="Measure", y="Percent", color="Group", barmode="group", text_auto=".0f",
                         color_discrete_map={"Low BP (SBP<90)": ORANGE, "All other BP": GREY},
                         title=f"Low BP (n={b['low_bp'].sum()}) vs everyone else")
            fig.update_layout(height=420, xaxis_title="", yaxis_title="% of patients")
            st.plotly_chart(fig, width="stretch")
        rows = []
        if b["low_bp"].sum() > 0:
            for col, name in [("killip_3_4", "Low BP vs rest: Killip III-IV"), ("nyha_iv", "Low BP vs rest: NYHA IV"),
                              ("on_inotrope", "Low BP vs rest: IV inotrope"), ("death_within_6_months", "Low BP vs rest: death 6m")]:
                orr, p = fisher_test(b["low_bp"], b[col])
                rows.append((name, "Fisher exact", f"Odds ratio = {orr:.1f}", p))
        chi2, p, v = chi2_test(b["bp_stage"], b["re_admission_within_6_months"])
        rows.append(("BP stage vs readmission", "Chi-square", f"Cramer's V = {v:.2f}", p))
        show_evidence(rows)
        insight_box("Low BP is a strong red flag: 80% are Killip III-IV and 90% NYHA IV. Yet they did **not** get IV "
                    "inotropes more often (possible treatment gap). Readmission **falls** as BP rises (the 'BP paradox').")
        action_box("Treat SBP < 90 as possible cardiogenic shock (ICU-level monitoring, review inotropes). "
                   "Watch normal and low BP patients for readmission, not only high BP.")


# =====================================================================
# PAGE 5 - PREDICTING OUTCOMES (PREDICTIVE)
# =====================================================================
elif page == "5. Predicting Outcomes (Predictive)":
    st.title("Can We Predict Bad Outcomes at Admission?")
    st.markdown("""
Every model uses **only admission-time data**, is logistic regression with balanced class weights, and is
scored with **5-fold cross-validation** (on patients the model has not seen).
We report **ROC-AUC** (0.5 = coin toss, 1.0 = perfect), not accuracy: predicting "everyone survives" is
97% accurate and useless.
""")
    with st.spinner("Training models (first time only)..."):
        t13, t14, t15, t16 = st.tabs(["Q13. Readmission", "Q14. 6-month death", "Q15. History vs severity",
                                      "Q16. Inflammation"])

        # ---------- Q13 ----------
        with t13:
            st.subheader("Can admission data predict 6-month readmission?")
            q13 = df[(df["outcome_during_hospitalization"] != "Dead") & (df["death_within_6_months"] == 0)].reset_index(drop=True)
            f13 = ["nyha_cardiac_function_classification", "killip_grade", "systolic_blood_pressure", "pulse", "respiration",
                   "glomerular_filtration_rate", "urea", "cystatin", "moderate_to_severe_chronic_kidney_disease",
                   "bnp_log", "troponin_log", "nlr_log", "albumin", "hemoglobin", "sodium", "cci_score", "diabetes",
                   "chronic_obstructive_pulmonary_disease", "age", "male", "bmi"]
            auc, prauc, prob = cv_score(q13, f13, "re_admission_within_6_months")
            groups = pd.qcut(prob, 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
            by_g = q13.groupby(groups, observed=True)["re_admission_within_6_months"].mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", auc)
            c2.metric("Highest-risk fifth readmitted", f"{by_g.iloc[-1]:.1f}%")
            c3.metric("Lowest-risk fifth readmitted", f"{by_g.iloc[0]:.1f}%")
            fig = px.bar(x=by_g.index.astype(str), y=by_g.values, text_auto=".1f", color=by_g.index.astype(str),
                         color_discrete_sequence=RAMP[1:], title="Actual readmission by predicted risk group")
            fig.add_hline(y=q13["re_admission_within_6_months"].mean() * 100, line_dash="dash", line_color=REF,
                          annotation_text="average")
            fig.update_layout(showlegend=False, xaxis_title="Predicted risk group", yaxis_title="% readmitted", height=400)
            st.plotly_chart(fig, width="stretch")
            insight_box("The prediction is weak (ROC-AUC about 0.61) but still useful: the top-risk fifth is readmitted "
                        "about **twice as often** as the bottom fifth. Readmission is driven by long-term burden "
                        "(NYHA class, kidney function, other diseases), not by the acute crisis.")
            action_box("Use the risk group to prioritise follow-up calls and early clinic visits after discharge.")

        # ---------- Q14 ----------
        with t14:
            st.subheader("Which combination of markers best predicts 6-month death?")
            domains = {
                "Severity": ["nyha_cardiac_function_classification", "killip_grade"],
                "Biomarkers": ["bnp_log", "troponin_log", "nlr_log", "albumin", "hemoglobin", "sodium"],
                "Renal": ["glomerular_filtration_rate", "urea", "cystatin", "moderate_to_severe_chronic_kidney_disease"],
                "Comorbidity": ["cci_score", "diabetes", "chronic_obstructive_pulmonary_disease",
                                "myocardial_infarction", "type_ii_respiratory_failure"],
                "Vitals": ["systolic_blood_pressure", "pulse", "respiration"],
                "Clinical": ["age", "male", "bmi", "unconscious"]}
            steps, used = [], []
            for name, cols in domains.items():
                used = used + cols
                a_, _, _ = cv_score(df, used, "death_within_6_months")
                steps.append({"Model": ("+ " if steps else "") + name, "Markers": len(used), "ROC-AUC": a_})
            steps = pd.DataFrame(steps)
            fig = px.line(steps, x="Model", y="ROC-AUC", markers=True, text="ROC-AUC",
                          title="Adding more domains does not keep helping", color_discrete_sequence=[BLUE])
            fig.update_traces(textposition="top center")
            best = steps["ROC-AUC"].idxmax()
            fig.add_scatter(x=[steps["Model"][best]], y=[steps["ROC-AUC"][best]], mode="markers",
                            marker=dict(size=16, color=ORANGE), name="Best")
            fig.update_layout(height=420, yaxis_range=[0.6, 0.9])
            st.plotly_chart(fig, width="stretch")
            st.dataframe(steps, hide_index=True, width="stretch")
            insight_box("Severity alone gives ROC-AUC about 0.75. Adding the biomarker panel gives the **best model (about 0.80, "
                        "8 markers)**. Adding more after that makes it worse: with only 57 deaths, extra markers add noise. "
                        "Killip grade and BNP carry most of the weight.")
            action_box("Collect a short list at admission (NYHA, Killip, BNP, troponin, NLR, albumin, hemoglobin, sodium). "
                       "This is the model behind the **Patient Risk Check** page.")

        # ---------- Q15 ----------
        with t15:
            st.subheader("Does past cardiac history predict in-hospital death, or does current severity?")
            st.caption("In-hospital death = 11 recorded deaths + 20 patients who left against medical advice and died "
                       "within 2 days (31 events).")
            history = ["myocardial_infarction", "congestive_heart_failure", "peripheral_vascular_disease",
                       "cerebrovascular_disease", "right_or_both_hf"]
            severity = ["nyha_cardiac_function_classification", "killip_grade"]
            res = pd.DataFrame({"Model": ["History only", "Severity only", "History + severity"],
                                "ROC-AUC": [cv_score(df, history, "in_hospital_death")[0],
                                            cv_score(df, severity, "in_hospital_death")[0],
                                            cv_score(df, history + severity, "in_hospital_death")[0]]})
            left, right = st.columns(2)
            with left:
                fig = px.bar(res, x="Model", y="ROC-AUC", text_auto=".2f", color="Model",
                             color_discrete_sequence=[GREY, BLUE, RAMP[4]], title="Model ROC-AUC")
                fig.add_hline(y=0.5, line_dash="dash", line_color=REF, annotation_text="coin toss")
                fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=400)
                st.plotly_chart(fig, width="stretch")
            with right:
                kil = df.groupby("killip_grade")["in_hospital_death"].mean() * 100
                fig = px.bar(x=[f"Killip {k}" for k in kil.index], y=kil.values, text_auto=".1f",
                             color=[f"Killip {k}" for k in kil.index], color_discrete_sequence=RAMP[1::1],
                             title="In-hospital death by Killip grade (%)")
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="% died", height=400)
                st.plotly_chart(fig, width="stretch")
            insight_box("**The present beats the past.** History alone is no better than a coin toss (about 0.49). "
                        "Current severity reaches about 0.87. Killip grade 1: **0 of 527 patients died** in hospital; "
                        "grade 4: 1 in 4 died.")
            action_box("Base triage on the bedside Killip/NYHA exam, not on the list of past diagnoses. "
                       "Killip 1 is a safe rule-out; Killip 4 needs ICU-level care.")

        # ---------- Q16 ----------
        with t16:
            st.subheader("Can inflammation markers (hs-CRP, WBC, NLR) predict early death?")
            sets = {"All three": ["hs_crp_log", "hs_crp_missing", "wbc_log", "nlr_log"], "NLR only": ["nlr_log"],
                    "WBC only": ["wbc_log"], "hs-CRP only": ["hs_crp_log", "hs_crp_missing"]}
            rows = []
            for tname, t in {"In-hospital death": "in_hospital_death", "28-day death": "death_within_28_days"}.items():
                for mname, cols in sets.items():
                    rows.append({"Outcome": tname, "Markers": mname, "ROC-AUC": cv_score(df, cols, t)[0]})
            res16 = pd.DataFrame(rows)
            left, right = st.columns(2)
            with left:
                fig = px.bar(res16, x="Markers", y="ROC-AUC", color="Outcome", barmode="group", text_auto=".2f",
                             color_discrete_map={"In-hospital death": ORANGE, "28-day death": RAMP[4]},
                             title="ROC-AUC by marker")
                fig.add_hline(y=0.5, line_dash="dash", line_color=REF)
                fig.update_layout(yaxis_range=[0, 1], height=400)
                st.plotly_chart(fig, width="stretch")
            with right:
                avail = (df[["nlr", "white_blood_cell", "hs_crp"]].notna().mean() * 100).rename(
                    {"nlr": "NLR", "white_blood_cell": "WBC", "hs_crp": "hs-CRP"})
                fig = px.bar(x=avail.index, y=avail.values, text_auto=".1f", color=avail.index,
                             color_discrete_sequence=[BLUE, BLUE, GREY], title="Marker measured for (% of patients)")
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="%", height=400, yaxis_range=[0, 110])
                st.plotly_chart(fig, width="stretch")
            insight_box("NLR alone (about 0.72) works as well as all three together. hs-CRP is almost useless here, "
                        "because it was **not measured for 53%** of patients. NLR comes free with every routine blood count.")
            action_box("Calculate NLR for every admission and flag the top quarter (NLR >= 8.7) for closer monitoring.")


# =====================================================================
# PAGE 6 - PATIENT RISK CHECK (the predictive tool)
# =====================================================================
elif page == "6. Patient Risk Check":
    st.title("Patient Risk Check (at admission)")
    st.markdown("Enter a patient's admission values. The tool uses our **best 6-month death model (Q14)** and the "
                "**red flags found in the prescriptive analysis**. It supports clinical judgement; it does not replace it.")

    features = ["nyha_cardiac_function_classification", "killip_grade", "bnp_log", "troponin_log",
                "nlr_log", "albumin", "hemoglobin", "sodium"]
    target = "death_within_6_months"

    @st.cache_resource
    def train_final_model():
        model = make_model().fit(df[features].astype(float), df[target])
        # cross-validated risk for every patient -> used to build the 5 risk groups honestly
        _, _, prob = cv_score(df, features, target)
        edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
        grp = np.digitize(prob, edges)
        death_rate = pd.Series(df[target].values).groupby(grp).mean() * 100
        return model, edges, death_rate

    model, edges, death_rate = train_final_model()

    with st.form("patient"):
        c1, c2, c3, c4 = st.columns(4)
        nyha = c1.selectbox("NYHA class", [2, 3, 4], index=1)
        killip = c2.selectbox("Killip grade", [1, 2, 3, 4], index=1)
        bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, 750.0)
        trop = c4.number_input("hs-Troponin (pg/mL)", 0.0, 50000.0, 55.0)
        c5, c6, c7, c8 = st.columns(4)
        neut = c5.number_input("Neutrophils (x10^9/L)", 0.1, 50.0, 5.0)
        lymph = c6.number_input("Lymphocytes (x10^9/L)", 0.05, 20.0, 1.0)
        alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, 37.0)
        hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, 115.0)
        c9, c10, c11, _ = st.columns(4)
        na = c9.number_input("Sodium (mmol/L)", 110.0, 160.0, 139.0)
        egfr = c10.number_input("eGFR (mL/min)", 1.0, 200.0, 60.0)
        sbp_in = c11.number_input("Systolic BP (mmHg)", 50.0, 250.0, 130.0)
        submitted = st.form_submit_button("Check risk")

    if submitted:
        nlr_val = neut / lymph
        x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(nlr_val), alb, hbv, na]], columns=features)
        score = model.predict_proba(x)[0, 1]
        group = int(np.digitize(score, edges))
        names = ["Lowest", "Low", "Middle", "High", "Highest"]
        colours = [RAMP[0], RAMP[1], "#f2c14e", ORANGE, REF]

        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Risk group: <span style='color:{colours[group]}'>{names[group]}</span>",
                        unsafe_allow_html=True)
            st.metric("6-month death rate of similar patients", f"{death_rate.iloc[group]:.1f}%",
                      delta=f"{death_rate.iloc[group] - df[target].mean() * 100:+.1f} pts vs average",
                      delta_color="inverse")
            st.caption("The risk group comes from where this patient's model score falls among our 2,008 patients. "
                       "The % is the real death rate seen in that group.")
        with right:
            fig = px.bar(x=names, y=death_rate.values, text_auto=".1f", title="Actual 6-month death by risk group",
                         color=names, color_discrete_sequence=colours)
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="% died", height=320)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Red flags (from our analysis)")
        flags = [
            ("Killip grade 3-4 (fluid in lungs / shock)", killip >= 3, "28-day death rises sharply; Killip 4 = 1 in 4 die"),
            ("NYHA class IV (symptoms at rest)", nyha == 4, "Risk jumps at class IV"),
            ("Systolic BP < 90 (possible shock)", sbp_in < 90, "P3: consider ICU monitoring and inotrope review"),
            ("eGFR < 45 (CKD G3b or worse)", egfr < 45, "P1: diuretic dosing, potassium checks, follow-up in 2 weeks"),
            ("Hemoglobin < 80 g/L (severe anemia)", hbv < 80, "P2: about 3x 6-month death; correct anemia"),
            ("NLR >= 8.7 (top quarter)", nlr_val >= 8.7, "Q16: early death risk rises across NLR quartiles"),
            ("Albumin < 35 g/L", alb < 35, "Low protein / inflammation"),
        ]
        n_flags = sum(f[1] for f in flags)
        st.write(f"**{n_flags} of {len(flags)} red flags present** (NLR = {nlr_val:.1f})")
        for name, present, why in flags:
            if present:
                st.error(f"{name} - {why}")
        if n_flags == 0:
            st.success("No red flags. Standard care and routine follow-up.")


# =====================================================================
# PAGE 7 - KEY TAKEAWAYS & CONCLUSION
# =====================================================================
elif page == "7. Key Takeaways & Conclusion":
    st.title("Key Takeaways & Conclusion")

    st.subheader("Key findings")
    st.markdown("""
1. **Readmission, not death, is the main burden.** 38.5% are readmitted within 6 months vs 2.8% who die.
2. **Current severity beats past history.** Killip grade and NYHA class at admission predict death far better
   than old diagnoses (ROC-AUC 0.87 vs 0.49). Killip 1 = zero in-hospital deaths among 527 patients.
3. **Kidneys matter.** Death rises from about 1.6% (G1) to 9% (G5); readmission peaks at G3b (50%).
4. **Only severe anemia is dangerous.** Mild/moderate anemia adds little; Hb < 80 roughly triples 6-month death.
5. **Low BP is rare but serious**, and those patients did not get more inotropes: a possible treatment gap.
6. **Simple markers win.** NLR from a routine blood count beats hs-CRP, which is missing for half the patients.
7. **Fewer, better markers.** An 8-marker model (severity + biomarkers) beats a 24-marker model.
8. **Guideline treatment gap.** Only 19% received the full recommended combination (ACEi/ARB + beta-blocker + MRA).
""")

    st.subheader("What the hospital should do")
    st.markdown("""
| When | Action | Based on |
|---|---|---|
| Admission (triage) | Use Killip + NYHA to route patients: Killip 1 -> ward, Killip 4 / SBP < 90 -> ICU | Q15, P3 |
| First blood test | Flag NLR >= 8.7, Hb < 80, eGFR < 45 | Q16, P2, P1 |
| During stay | Review guideline medicines and inotrope use in low-BP patients | Descriptive Q10, P3 |
| Discharge | High readmission-risk group -> follow-up call and clinic within 2 weeks | Q13, P1 |
""")

    st.subheader("Conclusion")
    st.markdown("""
Heart failure patients in this hospital arrive old and very sick, and more than a third come back within
6 months. The data shows the risk can be spotted **at admission** with a few routine, cheap measurements
(bedside Killip/NYHA exam and a basic blood panel). Acting on these flags can help the hospital reduce
early deaths, plan ICU beds, and cut readmissions through targeted follow-up.
""")

    st.subheader("Limitations")
    st.markdown("""
- Single-hospital data; only 57 deaths within 6 months, so death models have wide uncertainty.
- Many tests (LVEF, hs-CRP) were missing for over half the patients.
- The data shows association, not proof of cause.
- The risk tool is a decision-support demo and has not been validated on outside patients.
""")

    with st.expander("How we built this dashboard (technical notes)"):
        st.markdown("""
- **Why Streamlit + Plotly:** pure Python, so the same pandas code from our notebooks runs here, and the charts
  are interactive (hover, zoom) with no web development needed.
- **Idea:** we asked "who would open this dashboard and what decision would they make?" That turned the
  analysis into a triage tool plus a patient risk check, instead of just a summary of the data.
- **Obstacles and fixes:**
  - *Slow model training on every click* -> `st.cache_data` / `st.cache_resource` so models train once.
  - *Only 11 in-hospital deaths* -> added patients who left against advice and died within 2 days (31 events).
  - *Rare outcomes make accuracy misleading* -> ROC-AUC, balanced class weights, cross-validation.
  - *Model probabilities are inflated by class balancing* -> the risk tool shows the **real death rate of the
    patient's risk group** instead of a raw probability.
  - *Missing labs* -> median filling inside the model pipeline only, so no leakage.
- **Filters** in the sidebar re-run the descriptive and prescriptive charts and statistical tests for any
  gender / age sub-group.
""")
