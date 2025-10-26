# ============================================================
# File: app.py
# Microplastic Pollution Risk Prediction Dashboard (v3 – Fixed Heatmap)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fpdf import FPDF

# ============================================================
# --- Streamlit Config ---
# ============================================================
st.set_page_config(page_title="Microplastic Risk Prediction", layout="wide")
st.title("🌍 Microplastic Pollution Risk Prediction Dashboard")
st.sidebar.title("🔧 Navigation")

# ============================================================
# --- Session State ---
# ============================================================
for key in ["data", "pred_df", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# --- Utility Functions ---
# ============================================================
@st.cache_data
def load_data(file):
    """Load CSV/Excel and normalize column names."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel file.")
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


def generate_sample_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "Latitude": np.random.uniform(-10, 10, 100),
        "Longitude": np.random.uniform(100, 120, 100),
        "pH": np.random.uniform(6, 9, 100),
        "Turbidity": np.random.uniform(1, 10, 100),
        "Population_Density": np.random.uniform(100, 10000, 100),
        "MP_Count (items/individual)": np.random.uniform(5, 100, 100),
        "Dominant_Risk_Type": np.random.choice(["Low", "Medium", "High"], 100)
    })
    return df


def make_risk_label(df):
    cols_lower = [c.lower() for c in df.columns]
    risk_col, conc_col = None, None

    for c in df.columns:
        if "risk" in c.lower():
            risk_col = c
            break
    for c in df.columns:
        if "mp_count" in c.lower() or "mp_conc" in c.lower():
            conc_col = c
            break

    if risk_col:
        st.info(f"✅ Detected risk column: `{risk_col}`")
        y = df[risk_col]
        return y, risk_col
    elif conc_col:
        st.info(f"✅ Detected MP concentration column: `{conc_col}`")
        y = pd.cut(df[conc_col], bins=[-1, 10, 30, 100],
                   labels=["Low", "Medium", "High"])
        return y, conc_col
    else:
        st.error("Dataset must contain either 'Dominant_Risk_Type' or 'MP_Count (items/individual)' column.")
        st.stop()


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        st.warning("⚠️ Not enough samples per class — using random split instead.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_pdf_report(metrics, high_risk_sites):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Microplastic Risk Assessment Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, "Model Performance Metrics:", ln=True)
    for k, v in metrics.items():
        pdf.cell(200, 10, f"{k}: {v:.2f}", ln=True)
    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, "High-Risk Zones Identified:", ln=True)
    for _, row in high_risk_sites.iterrows():
        pdf.cell(200, 10, f"Lat: {row['Latitude']}, Lon: {row['Longitude']}", ln=True)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

# ============================================================
# --- Navigation ---
# ============================================================
menu = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📤 Upload Dataset", "📊 Data Analysis", "🤖 Prediction Dashboard", "📈 Reports"]
)

if st.sidebar.button("🔄 Reset Session"):
    for key in ["data", "pred_df", "metrics"]:
        st.session_state[key] = None
    st.experimental_rerun()

# ============================================================
# --- Home ---
# ============================================================
if menu == "🏠 Home":
    st.header("Welcome to the Microplastic Pollution Risk Prediction System")
    st.write("""
    This platform helps assess microplastic pollution risk through data mining and visualization.
    Upload your dataset or use a sample to explore:
    - Data analysis and visualization  
    - Predictive modeling using Random Forest / XGBoost  
    - Risk classification maps and summary reports  
    """)

# ============================================================
# --- Upload Dataset ---
# ============================================================
elif menu == "📤 Upload Dataset":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state["data"] = df
            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            st.dataframe(df.head())
    elif st.session_state["data"] is not None:
        st.info("Using previously loaded dataset.")
        st.dataframe(st.session_state["data"].head())
    else:
        st.warning("Or use the built-in sample dataset below.")
        if st.button("📘 Load Sample Dataset"):
            st.session_state["data"] = generate_sample_dataset()
            st.success("Sample dataset loaded successfully!")

# ============================================================
# --- Data Analysis (Fixed Heatmap) ---
# ============================================================
elif menu == "📊 Data Analysis":
    st.header("Data Exploration and Visualization")
    if st.session_state["data"] is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["data"].copy()
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all"))

        st.subheader("Correlation Heatmap")
        # Convert numeric-looking strings to numbers
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

        if numeric_df.empty:
            st.warning("⚠️ No numeric columns found for correlation heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

# ============================================================
# --- Prediction Dashboard ---
# ============================================================
elif menu == "🤖 Prediction Dashboard":
    st.header("Predictive Modeling and Risk Classification")
    if st.session_state["data"] is None:
        st.warning("Please upload or load data first.")
    else:
        df = st.session_state["data"].copy()
        y, target_col = make_risk_label(df)

        # Ensure numeric features
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty or y.empty:
            st.error("❌ No valid numeric features found for model training.")
            st.stop()

        y = y.loc[X.index]
        X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

        model_choice = st.radio("Select Model:", ["Random Forest", "XGBoost"], horizontal=True)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = RandomForestClassifier(random_state=42) if model_choice == "Random Forest" else XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="macro"),
                    "Recall": recall_score(y_test, y_pred, average="macro"),
                    "F1-score": f1_score(y_test, y_pred, average="macro"),
                }
                st.session_state["metrics"] = metrics
                st.subheader("Model Performance")
                st.write(metrics)

                df["Predicted_Risk"] = model.predict(X)
                st.session_state["pred_df"] = df
                st.success("✅ Model trained successfully!")

        if st.session_state["pred_df"] is not None:
            pred_df = st.session_state["pred_df"]
            st.subheader("📍 Predicted Risk Map")
            fig = px.scatter_mapbox(
                pred_df,
                lat="Latitude",
                lon="Longitude",
                color="Predicted_Risk",
                color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
                zoom=3,
                height=600,
                mapbox_style="carto-positron"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# --- Reports ---
# ============================================================
elif menu == "📈 Reports":
    st.header("Generate Summary Report")
    if st.session_state["pred_df"] is None:
        st.warning("Please train a model first in the Prediction Dashboard.")
    else:
        metrics = st.session_state["metrics"]
        pred_df = st.session_state["pred_df"]
        high_risk_sites = pred_df[pred_df["Predicted_Risk"] == "High"]
        pdf_bytes = generate_pdf_report(metrics, high_risk_sites)
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="Microplastic_Risk_Report.pdf")
        st.subheader("Model Summary")
        st.write(metrics)
        st.subheader("High Risk Locations")
        st.dataframe(high_risk_sites[["Latitude", "Longitude", "Predicted_Risk"]])
