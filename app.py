# ============================================================
# File: app.py
# Microplastic Pollution Risk Prediction Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fpdf import FPDF
import io

# ============================================================
# --- Streamlit Configuration ---
# ============================================================
st.set_page_config(page_title="Microplastic Risk Prediction", layout="wide")

st.title("🌍 Microplastic Pollution Risk Prediction Dashboard")
st.sidebar.title("🔧 Navigation")

# ============================================================
# --- Initialize session state (persistent data across pages) ---
# ============================================================
for key in ["data", "pred_df", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# --- Utility Functions ---
# ============================================================
@st.cache_data
def load_data(file):
    """Load CSV/Excel and standardize column names."""
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
    """Auto-detect or create a risk label from available columns."""
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
            st.error("Dataset must contain 'Dominant_Risk_Type' or 'MP_Count (items/individual)'.")
            st.stop()

    return y, target_col


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Safe stratified split with fallback."""
    from sklearn.model_selection import train_test_split
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        st.warning("⚠️ Not enough samples per class for stratified split — using random split instead.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_pdf_report(metrics, high_risk_sites):
    """Generate PDF summary report."""
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
# --- Sidebar Navigation ---
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
# --- Page 1: Home ---
# ============================================================
if menu == "🏠 Home":
    st.header("Welcome to the Microplastic Pollution Risk Prediction System")
    st.write("""
    This platform analyzes environmental datasets to predict and visualize microplastic pollution risks.
    Upload your dataset or use a sample to explore:
    - Statistical data analysis  
    - Predictive modeling using Random Forest / XGBoost  
    - Risk visualization and downloadable reports  
    """)

# ============================================================
# --- Page 2: Upload Dataset ---
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
        st.info("Using dataset from session memory.")
        st.dataframe(st.session_state["data"].head())
    else:
        st.warning("Or use the sample dataset below.")
        if st.button("📘 Load Sample Dataset"):
            st.session_state["data"] = generate_sample_dataset()
            st.success("Sample dataset loaded successfully!")

# ============================================================
# --- Page 3: Data Analysis ---
# ============================================================
elif menu == "📊 Data Analysis":
    st.header("Data Exploration and Visualization")
    if st.session_state["data"] is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["data"]
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots()
        im = ax.imshow(numeric_df.corr(), cmap="coolwarm")
        ax.set_xticks(range(len(numeric_df.columns)))
        ax.set_xticklabels(numeric_df.columns, rotation=90)
        ax.set_yticks(range(len(numeric_df.columns)))
        ax.set_yticklabels(numeric_df.columns)
        st.pyplot(fig)

# ============================================================
# --- Page 4: Prediction Dashboard ---
# ============================================================
elif menu == "🤖 Prediction Dashboard":
    st.header("Predictive Modeling and Risk Classification")
    if st.session_state["data"] is None:
        st.warning("Please upload or load data first.")
    else:
        df = st.session_state["data"].copy()
        y, target_col = make_risk_label(df)

        X = df.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

        model_choice = st.radio("Select Model:", ["Random Forest", "XGBoost"], horizontal=True)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
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
# --- Page 5: Reports ---
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
