# =====================================================================
#  Heart Failure Risk Dashboard  |  Team 2 - Python Pioneers
#  NumpyNinja Python Hackathon - Category 5 (Demo of insights in Python)
#
#  How to run:
#     pip install -r requirements.txt
#     streamlit run app.py
#  Keep Cardiac_Cleaned_Data.csv in the same folder as this file.
# =====================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

st.set_page_config(page_title="Heart Failure Risk Dashboard", layout="wide")

# ---------- Same colour system as our notebooks ----------
BLUE = "#2a78d6"      # readmission
ORANGE = "#eb6834"    # death
REF = "#d03b3b"       # reference lines
GREY = "#b8b6b0"      # comparison group
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # light = better, dark = worse

DATA_FILE = "Cardiac_Cleaned_Data.csv"


# =====================================================================
# LOAD DATA + create the groups we need
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    new = {}

    # Kidney stage (eGFR)
    stage_order = ["G1 (>=90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"]
    ckd = pd.cut(df["glomerular_filtration_rate"], bins=[0, 15, 30, 45, 60, 90, 1000],
                 right=False, labels=stage_order[::-1])
    new["ckd_stage"] = pd.Categorical(ckd, categories=stage_order, ordered=True)

    # Anemia level (WHO)
    cut = np.where(df["gender"] == "Male", 130, 120)
    hb = df["hemoglobin"]
    anemia = np.select([hb.isna(), hb >= cut, hb >= 110, hb >= 80],
                       ["Missing", "No anemia", "Mild", "Moderate"], default="Severe")
    new["anemia_level"] = pd.Categorical(pd.Series(anemia).replace("Missing", np.nan),
                                         categories=["No anemia", "Mild", "Moderate", "Severe"], ordered=True)

    # Blood pressure stage
    sbp, dbp = df["systolic_blood_pressure"], df["diastolic_blood_pressure"]
    bp_order = ["Low (<90)", "Normal", "Elevated", "High stage 1", "High stage 2"]
    bp = np.select([sbp.isna(), sbp < 90, (sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
                   ["Missing", bp_order[0], bp_order[4], bp_order[3], bp_order[2]], default=bp_order[1])
    new["bp_stage"] = pd.Categorical(pd.Series(bp).replace("Missing", np.nan), categories=bp_order, ordered=True)

    # Model inputs
    new["age"] = df["agecat"].apply(lambda s: (int(s.split("-")[0]) + int(s.split("-")[1])) / 2)
    new["male"] = (df["gender"] == "Male").astype(int)
    new["nlr"] = df["neutrophil_count"] / df["lymphocyte_count"]
    new["nlr_log"] = np.log(new["nlr"])
    new["troponin_log"] = np.log1p(df["high_sensitivity_troponin"])
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def make_model():
    """Logistic regression, same settings as our predictive notebook."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))


@st.cache_data
def risk_scores(data, features, target, repeats=3):
    """Each patient's risk, predicted by a model that never saw that patient (5-fold cross-validation)."""
    X, y = data[features].astype(float), data[target]
    probs = []
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs.append(cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1])
    return np.mean(probs, axis=0)


def pct(x):
    return f"{x * 100:.1f}%"


def bar(x, y, title, colours, ytitle="% of patients", fmt=".1f", height=380):
    fig = px.bar(x=x, y=y, text_auto=fmt, color=x, color_discrete_sequence=colours, title=title)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=ytitle, height=height)
    return fig


def two_outcomes(table, title):
    """Readmission (blue) and death (orange) per group."""
    long = table.reset_index().melt(id_vars=table.index.name, var_name="Outcome", value_name="Percent")
    fig = px.bar(long, x=table.index.name, y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Readmitted in 6 months": BLUE, "Died in 6 months": ORANGE}, title=title)
    fig.update_layout(yaxis_title="% of patients", xaxis_title="", legend_title="", height=400)
    return fig


df = load_data()

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("Heart Failure Dashboard")
st.sidebar.caption("Team 2 - Python Pioneers")
page = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Preparation",
    "Who Are the Patients?",
    "What Raises the Risk?",
    "Spotting High-Risk Patients Early",
    "Patient Risk Check",
    "Key Takeaways",
])


# =====================================================================
# PAGE: INTRODUCTION
# =====================================================================
if page == "Introduction":
    st.title("Heart Failure: Which Patients Need Extra Care?")
    st.markdown("""
Heart failure patients arrive at hospital very sick, and **more than 1 in 3 come back within 6 months**.
Every return costs a bed and money, and some early deaths could be prevented with closer watching.

**Our goal:** help the hospital spot the high-risk patients **on the day they are admitted**,
using simple tests it already does.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients studied", f"{len(df):,}")
    c2.metric("Came back within 6 months", pct(df["re_admission_within_6_months"].mean()))
    c3.metric("Died within 6 months", pct(df["death_within_6_months"].mean()))
    c4.metric("Arrived with severe symptoms", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))

    st.markdown("""
**Who can use this dashboard:** doctors at admission, nurses planning discharge, and hospital managers
planning beds.

**Data:** 7 hospital tables (patient details, heart tests, other diseases, blood tests, hospital stay,
alertness, medicines) joined into one table with one row per patient.
""")


# =====================================================================
# PAGE: DATA PREPARATION (short)
# =====================================================================
elif page == "Data Preparation":
    st.title("Data Preparation")

    left, right = st.columns(2)
    with left:
        st.subheader("Cleaning, in short")
        st.markdown("""
- **Removed impossible values:** a fake patient, 0 kg weight, 0 pulse, BMI of 404.
- **Fixed wrong units:** troponin, hematocrit and heart-scan values were in the wrong unit.
- **Filled blanks only when the meaning was clear:** e.g. blank breathing support = no ventilation.
- **Kept real gaps empty:** we did not invent missing lab results.
- **Joined 7 tables into 1:** one row per patient (2,008 patients).
""")
    with right:
        st.subheader("Why it mattered")
        ba = pd.DataFrame({
            "Example": ["Patients showing heart damage (troponin)", "Highest BMI", "Medicine rows per patient"],
            "Before": ["0.3% (wrong unit)", "404 (impossible)", "about 8 (patients counted 8 times)"],
            "After": ["84% (as expected)", "39", "1"]})
        st.dataframe(ba, hide_index=True, width="stretch")

    st.subheader("Feature engineering: new columns we added, and why")
    st.markdown("""
- **Groups instead of raw numbers** (BMI group, blood pressure group, kidney stage, anemia level):
  easier to compare and explain, e.g. "death rate in kidney stage G5".
- **Yes/No warning flags** (high BNP, high troponin, enlarged heart, many medicines):
  each one answers a simple clinical question.
- **Number of other diseases per patient:** one number for how much extra illness a patient carries.
- **NLR** (a ratio from the routine blood count): a cheap inflammation marker available for almost everyone.
- **Log of very skewed values** (BNP, troponin): stops a few extreme patients from controlling the model.
""")


# =====================================================================
# PAGE: WHO ARE THE PATIENTS?
# =====================================================================
elif page == "Who Are the Patients?":
    st.title("Who Are the Patients?")

    out = pd.DataFrame({
        "Time": ["28 days", "3 months", "6 months"] * 2,
        "Outcome": ["Came back"] * 3 + ["Died"] * 3,
        "Percent": [df[c].mean() * 100 for c in ["re_admission_within_28_days", "re_admission_within_3_months",
                                                  "re_admission_within_6_months", "death_within_28_days",
                                                  "death_within_3_months", "death_within_6_months"]]})
    fig = px.bar(out, x="Time", y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Came back": BLUE, "Died": ORANGE},
                 title="Coming back to hospital is the bigger problem")
    fig.update_layout(yaxis_title="% of patients", xaxis_title="", legend_title="", height=380)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        m = pd.Series({
            "High BNP (heart under strain)": df["bnp_elevated_flag"].mean(),
            "Heart muscle damage (troponin)": df["troponin_elevated_flag"].mean(),
            "Anemia": df["anemia_level"].isin(["Mild", "Moderate", "Severe"]).sum() / df["anemia_level"].notna().sum(),
            "Weak kidneys (eGFR < 60)": (df["glomerular_filtration_rate"] < 60).sum() / df["glomerular_filtration_rate"].notna().sum(),
            "Underweight": (df["bmi_category"] == "Underweight").mean(),
        }).sort_values() * 100
        fig = px.bar(m, orientation="h", text_auto=".0f", color_discrete_sequence=[BLUE],
                     title="How common each problem is (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="", height=360)
        st.plotly_chart(fig, width="stretch")
    with right:
        drugs = pd.Series({
            "Water tablets (diuretic)": ((df["Furosemide injection"] + df["Furosemide tablet"] + df["Torasemide tablet"] +
                                          df["Hydrochlorothiazide tablet"]) > 0).mean(),
            "Spironolactone": df["Spironolactone tablet"].mean(),
            "ACE inhibitor / ARB": ((df["Benazepril hydrochloride tablet"] + df["Valsartan Dispersible tablet"]) > 0).mean(),
            "Beta-blocker": ((df["Metoprolol Succinate Sustained-release tablet"] +
                              df["metoprolol tartrate injection"]) > 0).mean(),
        }).sort_values() * 100
        fig = px.bar(drugs, orientation="h", text_auto=".0f", color_discrete_sequence=[RAMP[3]],
                     title="Recommended heart medicines given (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="", height=360)
        st.plotly_chart(fig, width="stretch")

    st.info("""
- Most patients are **elderly (73% aged 69+)** and **82% arrive with severe symptoms**.
- Few die (2.8% in 6 months), but **38.5% come back** within 6 months.
- Almost everyone gets water tablets, but **only about 4 in 10 get the key long-term heart medicines**.
""")


# =====================================================================
# PAGE: WHAT RAISES THE RISK? (prescriptive insights)
# =====================================================================
elif page == "What Raises the Risk?":
    st.title("What Raises the Risk?")
    st.markdown("We checked three body systems that affect the heart. Each finding was confirmed with statistical tests.")

    t1, t2, t3 = st.tabs(["Kidneys", "Anemia (low hemoglobin)", "Blood pressure"])

    with t1:
        k = df.dropna(subset=["ckd_stage"])
        g = k.groupby("ckd_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "As kidneys get weaker, more patients die"), width="stretch")
        st.info("""
**What we found:** weak kidneys and a weak heart pull each other down.
Deaths rise from **1.6%** (healthy kidneys) to **9.2%** (kidney failure), and returns peak at **50%** in stage G3b.

**What to do:** treat patients with eGFR below 45 as high risk. Check potassium, dose water tablets carefully,
and see them again within 2 weeks of going home.
""")

    with t2:
        a = df.dropna(subset=["anemia_level"])
        g = a.groupby("anemia_level", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Only severe anemia stands out"), width="stretch")
        st.info("""
**What we found:** mild and moderate anemia are very common but add little risk.
**Severe anemia (hemoglobin below 80)** nearly **triples** the 6-month death rate (6.8% vs about 2.5%).

**What to do:** flag hemoglobin below 80 at admission and correct it (iron, transfusion if needed).
""")

    with t3:
        b = df.dropna(subset=["bp_stage"])
        g = b.groupby("bp_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Low blood pressure is rare but dangerous"), width="stretch")
        st.info("""
**What we found:** only 20 patients arrived with low blood pressure (below 90), but **9 in 10 had symptoms at rest**
and **8 in 10 had fluid in the lungs or shock**. Surprisingly, they did **not** get heart-support drips more often.
Patients with higher blood pressure came back **less** often, because their heart still has pumping strength.

**What to do:** treat blood pressure below 90 as possible shock and move the patient to close monitoring.
""")


# =====================================================================
# PAGE: SPOTTING HIGH-RISK PATIENTS EARLY (predictive insights)
# =====================================================================
elif page == "Spotting High-Risk Patients Early":
    st.title("Can We Spot High-Risk Patients on Day 1?")
    st.markdown("We used only information available **at admission**, and tested our models on patients they had never seen.")

    d28 = "death_within_28_days"
    t1, t2, t3, t4 = st.tabs(["Bedside check", "Now vs past", "Blood test", "Who will come back?"])

    # ---- Killip + NYHA ----
    with t1:
        left, right = st.columns(2)
        with left:
            kil = df.groupby("killip_grade")[d28].mean() * 100
            st.plotly_chart(bar([f"Killip {k}" for k in kil.index], kil.values,
                                "Deaths within 28 days by Killip grade (%)", RAMP[1:]), width="stretch")
        with right:
            nyha4 = np.where(df["nyha_cardiac_function_classification"] == 4, "Symptoms at rest", "Symptoms on activity")
            k34 = np.where(df["killip_grade"] >= 3, "Fluid in lungs / shock", "No / mild fluid")
            grid = (pd.crosstab(nyha4, k34, values=df[d28], aggfunc="mean") * 100).round(1)
            grid = grid.loc[["Symptoms on activity", "Symptoms at rest"], ["No / mild fluid", "Fluid in lungs / shock"]]
            fig = px.imshow(grid, text_auto=True, color_continuous_scale=["#f4f8fd", ORANGE],
                            title="Deaths within 28 days (%): combine the two bedside scores")
            fig.update_layout(height=380, coloraxis_showscale=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        st.info("""
**What we found:** a 30-second bedside exam (Killip grade) sorts patients very well.
**None of 527 Killip 1 patients died** within 28 days, while **1 in 4 Killip 4 patients died**.
Adding symptom level (NYHA) makes it sharper: 0.4% vs **11.9%** deaths, a **30 times** difference.

**What to do:** Killip 1 patients can safely go to a normal ward. Killip 4 patients need ICU-level care.
""")

    # ---- Past history vs current state ----
    with t2:
        hist = pd.Series({
            "Old heart attack": df.loc[df["myocardial_infarction"] == 1, d28].mean(),
            "No old heart attack": df.loc[df["myocardial_infarction"] == 0, d28].mean(),
            "Past heart failure": df.loc[df["congestive_heart_failure"] == 1, d28].mean(),
            "No past heart failure": df.loc[df["congestive_heart_failure"] == 0, d28].mean(),
        }) * 100
        left, right = st.columns(2)
        with left:
            st.plotly_chart(bar(list(hist.index), hist.values, "Past history: death rate hardly changes (%)",
                                [GREY, GREY, GREY, GREY]), width="stretch")
        with right:
            st.plotly_chart(bar(["Killip 1", "Killip 2", "Killip 3", "Killip 4"],
                                (df.groupby("killip_grade")[d28].mean() * 100).values,
                                "Condition today: death rate changes a lot (%)", RAMP[1:]), width="stretch")
        st.info("""
**What we found:** a patient's **past** (old heart attack, earlier heart failure) tells us almost nothing
about who will die: about 2% either way. How sick the patient is **today** tells us almost everything.

**What to do:** decide the level of care from today's bedside exam, not from the list of old diagnoses.
""")

    # ---- NLR vs hs-CRP ----
    with t3:
        left, right = st.columns(2)
        with left:
            q = pd.qcut(df["nlr"], 4, labels=["Lowest NLR", "Low", "High", "Highest NLR"])
            nq = df.groupby(q, observed=True)[d28].mean() * 100
            st.plotly_chart(bar(list(nq.index.astype(str)), nq.values,
                                "Deaths within 28 days by NLR level (%)", RAMP[1:5]), width="stretch")
        with right:
            avail = pd.Series({"NLR (routine blood count)": df["nlr"].notna().mean() * 100,
                               "hs-CRP (special test)": df["hs_crp"].notna().mean() * 100})
            st.plotly_chart(bar(list(avail.index), avail.values, "How many patients had the test (%)",
                                [BLUE, GREY]), width="stretch")
        st.info("""
**What we found:** NLR comes free with the routine blood count. Patients in the highest NLR group had
**8 times** the early death rate of the lowest group (3.4% vs 0.4%).
The well-known inflammation test (hs-CRP) did not help, because **more than half of patients were never tested**.

**What to do:** calculate NLR for every patient and flag NLR of 8.7 or more.
""")

    # ---- Readmission risk groups ----
    with t4:
        q13 = df[(df["outcome_during_hospitalization"] != "Dead") & (df["death_within_6_months"] == 0)].reset_index(drop=True)
        f13 = ["nyha_cardiac_function_classification", "killip_grade", "systolic_blood_pressure", "pulse", "respiration",
               "glomerular_filtration_rate", "urea", "cystatin", "moderate_to_severe_chronic_kidney_disease",
               "bnp_log", "troponin_log", "nlr_log", "albumin", "hemoglobin", "sodium", "cci_score", "diabetes",
               "chronic_obstructive_pulmonary_disease", "age", "male", "bmi"]
        with st.spinner("Scoring patients..."):
            prob = risk_scores(q13, f13, "re_admission_within_6_months")
        groups = pd.qcut(prob, 5, labels=["Lowest risk", "Low", "Middle", "High", "Highest risk"])
        by_g = q13.groupby(groups, observed=True)["re_admission_within_6_months"].mean() * 100
        st.plotly_chart(bar(list(by_g.index.astype(str)), by_g.values,
                            "Patients who actually came back, by predicted risk group (%)", RAMP[1:]), width="stretch")
        st.info(f"""
**What we found:** coming back is harder to predict than death, because it also depends on things outside the
hospital (home support, taking medicines). Still, our model's **highest-risk group came back about twice as often**
({by_g.iloc[-1]:.0f}%) as the lowest-risk group ({by_g.iloc[0]:.0f}%). Main drivers: severe symptoms, weak kidneys
and other diseases.

**What to do:** give the highest-risk group a follow-up phone call and an early clinic visit after discharge.
""")


# =====================================================================
# PAGE: PATIENT RISK CHECK
# =====================================================================
elif page == "Patient Risk Check":
    st.title("Patient Risk Check")
    st.markdown("Enter a new patient's admission values to see their risk group and warning signs. "
                "This supports the doctor's judgement; it does not replace it.")

    features = ["nyha_cardiac_function_classification", "killip_grade", "bnp_log", "troponin_log",
                "nlr_log", "albumin", "hemoglobin", "sodium"]
    target = "death_within_6_months"

    @st.cache_resource
    def train_final_model():
        model = make_model().fit(df[features].astype(float), df[target])
        prob = risk_scores(df, features, target)
        edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
        death_rate = pd.Series(df[target].values).groupby(np.digitize(prob, edges)).mean() * 100
        return model, edges, death_rate

    model, edges, death_rate = train_final_model()

    with st.form("patient"):
        c1, c2, c3, c4 = st.columns(4)
        nyha = c1.selectbox("NYHA class (symptoms)", [2, 3, 4], index=1)
        killip = c2.selectbox("Killip grade (fluid / shock)", [1, 2, 3, 4], index=1)
        bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, 750.0)
        trop = c4.number_input("Troponin (pg/mL)", 0.0, 50000.0, 55.0)
        c5, c6, c7, c8 = st.columns(4)
        neut = c5.number_input("Neutrophils (x10^9/L)", 0.1, 50.0, 5.0)
        lymph = c6.number_input("Lymphocytes (x10^9/L)", 0.05, 20.0, 1.0)
        alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, 37.0)
        hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, 115.0)
        c9, c10, c11, _ = st.columns(4)
        na = c9.number_input("Sodium (mmol/L)", 110.0, 160.0, 139.0)
        egfr = c10.number_input("eGFR (kidney)", 1.0, 200.0, 60.0)
        sbp_in = c11.number_input("Systolic BP (mmHg)", 50.0, 250.0, 130.0)
        submitted = st.form_submit_button("Check risk")

    if submitted:
        nlr_val = neut / lymph
        x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(nlr_val), alb, hbv, na]], columns=features)
        group = int(np.digitize(model.predict_proba(x)[0, 1], edges))
        names = ["Lowest", "Low", "Middle", "High", "Highest"]
        colours = [RAMP[0], RAMP[1], "#f2c14e", ORANGE, REF]

        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Risk group: <span style='color:{colours[group]}'>{names[group]}</span>",
                        unsafe_allow_html=True)
            st.metric("Similar patients who died within 6 months", f"{death_rate.iloc[group]:.1f}%",
                      delta=f"{death_rate.iloc[group] - df[target].mean() * 100:+.1f} vs average",
                      delta_color="inverse")
        with right:
            st.plotly_chart(bar(names, death_rate.values, "Deaths within 6 months by risk group (%)", colours,
                                height=300), width="stretch")

        st.subheader("Warning signs")
        flags = [
            ("Fluid in lungs or shock (Killip 3-4)", killip >= 3),
            ("Symptoms at rest (NYHA IV)", nyha == 4),
            ("Low blood pressure (below 90)", sbp_in < 90),
            ("Weak kidneys (eGFR below 45)", egfr < 45),
            ("Severe anemia (hemoglobin below 80)", hbv < 80),
            (f"High NLR ({nlr_val:.1f}, 8.7 or more)", nlr_val >= 8.7),
        ]
        shown = [name for name, present in flags if present]
        for name in shown:
            st.error(name)
        if not shown:
            st.success("No warning signs. Standard care and routine follow-up.")


# =====================================================================
# PAGE: KEY TAKEAWAYS
# =====================================================================
elif page == "Key Takeaways":
    st.title("Key Takeaways")

    st.subheader("What we learned")
    st.markdown("""
1. **Coming back is the bigger problem than dying:** more than 1 in 3 patients return within 6 months.
2. **How sick the patient is today matters more than their past.** A quick bedside check finds most of the patients who die.
3. **Weak kidneys, severe anemia and low blood pressure** are the three warning signs that raise the risk most.
4. **Simple routine tests work best.** NLR from the normal blood count beat a special test that half the patients never had.
5. **Many patients miss key long-term heart medicines**, which may be one reason so many come back.
""")

    st.subheader("What the hospital should do")
    st.markdown("""
1. **When the patient arrives:** do the quick bedside check. Mild cases go to the ward; patients with fluid in the lungs,
   shock or low blood pressure go to close monitoring.
2. **After the first blood test:** flag weak kidneys, severe anemia and high NLR for extra care.
3. **During the stay:** make sure patients get the recommended heart medicines.
4. **Before going home:** high-risk patients get a follow-up call and a clinic visit within 2 weeks.
""")

    st.subheader("Conclusion")
    st.success("With tests the hospital already does on day 1, it can spot the patients most likely to die or come back, "
               "and give them the right level of care early. This can save lives, free up ICU beds and reduce returns.")

    with st.expander("How we built this (for questions)"):
        st.markdown("""
- **Tools:** Python, pandas, scikit-learn and Streamlit, so our notebook code runs directly in the dashboard.
- **Idea:** we asked "who would open this and what would they decide?", so we built a risk check, not just charts.
- **Challenges:** very few deaths in the data, so we measured how well the model *ranks* patients instead of
  simple accuracy, and tested it on unseen patients. Missing lab results were filled only inside the model training.
- **Limits:** data from one hospital; it shows links, not proof of cause.
""")
