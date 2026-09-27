import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# Title & Intro
# -----------------------------
st.set_page_config(page_title="Heart Dashboard Demo", layout="wide")
st.title("❤️ Heart Health Demo Dashboard")
st.write("A simple Streamlit demo app to test deployment on Streamlit Cloud.")

# -----------------------------
# Sample Heart Dataset
# -----------------------------
st.subheader("Sample Heart Data")

np.random.seed(42)
data = pd.DataFrame({
    "Age": np.random.randint(30, 80, 50),
    "RestingBP": np.random.randint(90, 160, 50),
    "Cholesterol": np.random.randint(150, 300, 50),
    "HeartRate": np.random.randint(60, 120, 50),
})

st.dataframe(data)

# -----------------------------
# Plot 1: Cholesterol Distribution
# -----------------------------
st.subheader("Cholesterol Distribution")
fig1 = px.histogram(data, x="Cholesterol", nbins=20, color_discrete_sequence=["red"])
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Plot 2: Heart Rate vs Age
# -----------------------------
st.subheader("Heart Rate vs Age")
fig2 = px.scatter(
    data,
    x="Age",
    y="HeartRate",
    color="HeartRate",
    color_continuous_scale="Reds",
    title="Heart Rate by Age"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Summary Stats
# -----------------------------
st.subheader("Summary Statistics")
st.write(data.describe())
