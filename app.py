# app.py — Streamlit Microplastic Pollution Risk Dashboard

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

# ===================== UTILITY FUNCTIONS =====================

@st.cache_data
def load_data(file):
    """Load CSV/Excel, standardize column names, and auto-convert numeric data."""
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

    # Convert numeric-like text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "mp_conc" in df.columns:
        df["mp_conc"] = pd.to_numeric(df["mp_conc"], errors="coerce")

    return df


def make_risk_label(df):
    """Detect or create a risk label column."""
    colnames = [c.lower() for c in df.columns]

    if "risk" in df.columns:
        y = df["risk"]
        target_col = "risk"
    elif "mp_conc" in df.columns:
        y = pd.cut(df["mp_conc"], bins=[-1, 10, 30, 100],
                   labels=["Low", "Medium", "High"])
        target_col = "mp_conc"
    else:
        matched = [c for c in colnames if "dominant" in c or "mp_count" in c]
        if matched:
            st.warning(f"⚠️ Column '{matched[0]}' recognized but renamed internally.")
            df.rename(columns={matched[0]: "mp_conc"}, inplace=True)
            y = pd.cut(df["mp_conc"], bins=[-1, 10, 30, 100],
                       labels=["Low", "Medium", "High"])
            target_col = "mp_conc"
        else:
            st.error("No valid target found. Dataset must contain 'Dominant_Risk_Type' or 'MP_Count (items/individual)'.")
            st.stop()

    return y, target_col


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Fallback to non-stratified split if class imbalance occurs."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        st.warning("⚠️ Not enough samples per class for stratified split — using random split.")
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

# ===================== PAGE: HOME =====================

if page == "🏠 Home":
    st.title("🌊 Microplastic Pollution Risk Prediction System")
    st.markdown("""
    This interactive web-based system assesses **microplastic pollution risk**
    across various environments using predictive analytics and data mining.

    **Modules:**
    - Upload and analyze environmental datasets  
    - Train machine learning models (Random Forest, XGBoost)  
    - Generate pollution risk maps  
    - Download analytical reports
    """)

# ===================== PAGE: UPLOAD =====================

elif page == "📤 Upload Dataset":
    st.title("📤 Upload Dataset")

    uploaded_file = st.file_uploader("Upload your CSV or Excel dataset", type=["csv", "xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state["data"] = df
            st.success("✅ Dataset successfully uploaded and cached.")
            st.dataframe(df.head())
            st.write(f"**Columns:** {list(df.columns)}")

# ===================== PAGE: DATA ANALYSIS =====================

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
            st.warning("⚠️ No numeric columns found for correlation heatmap.")
        else:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("Distribution of MP Concentration")
        if "mp_conc" in df.columns:
            fig2 = px.histogram(df, x="mp_conc", nbins=30, title="Distribution of MP Concentration")
            st.plotly_chart(fig2)
        else:
            st.warning("No 'mp_conc' column found for visualization.")

# ===================== PAGE: PREDICTION DASHBOARD =====================

elif page == "🤖 Prediction Dashboard":
    st.title("🤖 Prediction Dashboard")

    df = st.session_state.get("data")
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        y, target_col = make_risk_label(df)

        # Convert all numeric-like columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No valid numeric features found for model training.")
            st.stop()

        X = df[numeric_cols].dropna()
        y = y.loc[X.index]

        X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)

        if st.button("Train Model"):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model trained successfully — Accuracy: **{acc:.2f}**")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, preds))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Risk map (if lat/lon exist)
            if {"Latitude", "Longitude"}.issubset(df.columns):
                st.subheader("Geographic Risk Map")
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
            else:
                st.info("No Latitude/Longitude columns found — skipping map.")

# ===================== PAGE: REPORTS =====================

elif page == "📄 Reports":
    st.title("📄 Generate Reports")

    df = st.session_state.get("data")
    if df is None:
        st.warning("⚠️ Please upload and analyze data first.")
    else:
        st.write("Generate a summary report (PDF/Excel) with findings and predictions.")
        if st.button("Download Summary (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Microplastic Pollution Risk Report", ln=True, align="C")
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, "This report summarizes risk predictions and environmental indicators.")
            pdf.output("Microplastic_Risk_Report.pdf")
            with open("Microplastic_Risk_Report.pdf", "rb") as f:
                st.download_button("Download Report", f, file_name="Microplastic_Risk_Report.pdf")
