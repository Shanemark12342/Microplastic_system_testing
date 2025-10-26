# app.py — Streamlit Microplastic Pollution Risk Dashboard (Resilient Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from fpdf import FPDF

st.set_page_config(page_title="Microplastic Risk Prediction Dashboard", layout="wide")

# ===================== UTILITIES =====================

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel file.")
        return None

    df.columns = [c.strip() for c in df.columns]
    rename_map = {}

    for col in df.columns:
        c = col.lower()
        if "dominant_risk_type" in c:
            rename_map[col] = "risk"
        elif "mp_count" in c or "microplastic" in c or "mp_conc" in c:
            rename_map[col] = "mp_conc"

    df.rename(columns=rename_map, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "mp_conc" in df.columns:
        df["mp_conc"] = pd.to_numeric(df["mp_conc"], errors="coerce")

    return df


def make_risk_label(df):
    if "risk" in df.columns:
        y = df["risk"].astype(str)
        target_col = "risk"
    elif "mp_conc" in df.columns:
        y = pd.cut(df["mp_conc"], bins=[-1, 10, 30, 100],
                   labels=["Low", "Medium", "High"])
        target_col = "mp_conc"
    else:
        st.error("Dataset must contain 'Dominant_Risk_Type' or 'MP_Count (items/individual)'.")
        st.stop()
    return y, target_col


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Auto-adjusts test size to prevent ValueErrors."""
    n_samples = len(X)
    if n_samples < 5:
        st.error("❌ Not enough samples to train/test (need at least 5 rows after cleaning).")
        st.stop()
    if n_samples * test_size < 1:
        test_size = max(1 / n_samples, 0.2)
        st.warning(f"⚠️ Adjusted test size to {test_size:.2f} to fit data.")
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    except ValueError:
        st.warning("⚠️ Using non-stratified random split due to imbalance.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ===================== SESSION STATE =====================

if "data" not in st.session_state:
    st.session_state["data"] = None

# ===================== SIDEBAR =====================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📤 Upload Dataset", "📊 Data Analysis", "🤖 Prediction Dashboard", "📄 Reports"]
)

# ===================== HOME PAGE =====================

if page == "🏠 Home":
    st.title("🌊 Microplastic Pollution Risk Prediction System")
    st.markdown("""
    This interactive system uses **machine learning** to assess and visualize microplastic pollution risks.
    
    **Features include:**
    - Upload and explore datasets  
    - Train predictive models (Random Forest / XGBoost)  
    - View pollution heatmaps and trends  
    - Generate downloadable reports
    """)

# ===================== UPLOAD =====================

elif page == "📤 Upload Dataset":
    st.title("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state["data"] = df
            st.success("✅ Dataset uploaded successfully.")
            st.dataframe(df.head())

# ===================== DATA ANALYSIS =====================

elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    df = st.session_state.get("data")
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.warning("No numeric features found for correlation.")
        else:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if "mp_conc" in df.columns:
            st.subheader("MP Concentration Distribution")
            fig2 = px.histogram(df, x="mp_conc", nbins=30, title="Distribution of MP Concentration")
            st.plotly_chart(fig2)

# ===================== PREDICTION DASHBOARD =====================

elif page == "🤖 Prediction Dashboard":
    st.title("🤖 Prediction Dashboard")

    df = st.session_state.get("data")
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        y, target_col = make_risk_label(df)

        df = df.dropna(subset=[target_col])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No valid numeric features for training.")
            st.stop()

        X = df[numeric_cols].fillna(0)
        y = y.loc[X.index]

        test_size = st.slider("Test size fraction", 0.1, 0.5, 0.2, 0.05)
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size)
            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2f}**")

            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            if {"Latitude", "Longitude"}.issubset(df.columns):
                st.subheader("Predicted Risk Map")
                df["Prediction"] = model.predict(X)
                fig_map = px.scatter_mapbox(
                    df,
                    lat="Latitude", lon="Longitude",
                    color="Prediction",
                    color_continuous_scale=["green", "orange", "red"],
                    zoom=2, mapbox_style="open-street-map",
                    hover_data=["Prediction"]
                )
                st.plotly_chart(fig_map)

# ===================== REPORTS =====================

elif page == "📄 Reports":
    st.title("📄 Generate Reports")
    df = st.session_state.get("data")

    if df is None:
        st.warning("⚠️ Please upload and analyze data first.")
    else:
        if st.button("Download Summary (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Microplastic Pollution Risk Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "This automatically generated report summarizes model performance and pollution risk levels.")
            pdf.output("Microplastic_Risk_Report.pdf")

            with open("Microplastic_Risk_Report.pdf", "rb") as f:
                st.download_button("Download Report", f, file_name="Microplastic_Risk_Report.pdf")
