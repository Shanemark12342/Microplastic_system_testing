# app.py
"""
Microplastic Risk Prediction Dashboard
Streamlit-based predictive analytics system for assessing microplastic pollution risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fpdf import FPDF
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Initialize session state
# --------------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state["data"] = None
if "pred_df" not in st.session_state:
    st.session_state["pred_df"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None

# --------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel file.")
        return None
    return df


def generate_sample_dataset(n=200, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "latitude": np.random.uniform(-90, 90, n),
        "longitude": np.random.uniform(-180, 180, n),
        "pH": np.random.uniform(6, 9, n),
        "turbidity": np.random.uniform(0, 100, n),
        "population_density": np.random.uniform(50, 5000, n),
        "mp_conc": np.random.uniform(0, 50, n)
    })
    df["risk"] = pd.cut(df["mp_conc"], bins=[-1, 10, 30, 100], labels=["Low", "Medium", "High"])
    return df


def auto_preprocess(df: pd.DataFrame):
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return df, num_cols, cat_cols


def make_risk_label(df):
    """Auto-detect target column or create synthetic"""
    if "risk" in df.columns:
        y = df["risk"]
        target_col = "risk"
    else:
        if "mp_conc" not in df.columns:
            st.error("Dataset must include either 'risk' or 'mp_conc' column.")
            st.stop()
        y = pd.cut(df["mp_conc"], bins=[-1, 10, 30, 100], labels=["Low", "Medium", "High"])
        target_col = "mp_conc"
    return y, target_col


def build_model_pipeline(num_cols, cat_cols, clf_choice="rf"):
    if clf_choice == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    return model


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Safely split data; fallback to non-stratified if class imbalance"""
    from sklearn.model_selection import train_test_split
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        st.warning("⚠️ Not enough samples per class for stratified split — using random split instead.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def create_pdf_report(summary_text, pred_df, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Microplastic Pollution Risk Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(10)
    pdf.cell(0, 10, "Model Metrics:", ln=True)
    for k, v in metrics.items():
        pdf.cell(0, 10, f"{k.capitalize()}: {v:.3f}", ln=True)
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


def create_excel_report(df, pred_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Original Data")
        pred_df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"])

if st.sidebar.button("🔄 Reset Session"):
    for key in ["data", "pred_df", "metrics"]:
        st.session_state[key] = None
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.caption("🌊 Environmental Microplastic Risk Analyzer")

# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
if page == "Home":
    st.title("🌍 Microplastic Pollution Risk Assessment Dashboard")
    st.markdown("""
    This system predicts microplastic pollution risk (Low / Medium / High)
    based on environmental indicators using **Random Forest** and **XGBoost** models.
    """)
    st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e", use_container_width=True)
    st.info("Navigate using the sidebar to upload data, analyze, and predict risks.")

# --------------------------------------------------------------------------
# Upload Dataset
# --------------------------------------------------------------------------
elif page == "Upload Dataset":
    st.header("📤 Upload Environmental Dataset")
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

# --------------------------------------------------------------------------
# Data Analysis
# --------------------------------------------------------------------------
elif page == "Data Analysis":
    st.header("📊 Data Analysis & Visualization")

    if st.session_state["data"] is None:
        st.warning("Please upload or load data first.")
    else:
        df = st.session_state["data"]
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        corr = df.select_dtypes(include=np.number).corr()
        im = ax.imshow(corr, cmap="coolwarm")
        plt.colorbar(im)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

        if "mp_conc" in df.columns:
            st.subheader("Distribution of Microplastic Concentration")
            st.plotly_chart(px.histogram(df, x="mp_conc", nbins=20, title="Microplastic Concentration Distribution"))

# --------------------------------------------------------------------------
# Prediction Dashboard
# --------------------------------------------------------------------------
elif page == "Prediction Dashboard":
    st.header("🤖 Prediction Dashboard")

    if st.session_state["data"] is None:
        st.warning("Upload or load dataset first.")
    else:
        df = st.session_state["data"]

        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        clf_choice = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost"])

        df, num_cols, cat_cols = auto_preprocess(df)
        y, target_col = make_risk_label(df)

        if len(set(y)) < 2:
            st.error("Not enough unique risk classes to train model.")
            st.stop()

        X = df[num_cols]
        model = build_model_pipeline(num_cols, cat_cols, "rf" if clf_choice == "Random Forest" else "xgb")

        X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_size)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate_model(y_test, preds)
        st.session_state["metrics"] = metrics
        st.subheader("Model Performance")
        st.write(pd.DataFrame([metrics]))

        pred_df = X_test.copy()
        pred_df["predicted_risk"] = preds
        st.session_state["pred_df"] = pred_df

        st.subheader("🌎 Risk Map")
        if "latitude" in df.columns and "longitude" in df.columns:
            pred_df["color"] = pred_df["predicted_risk"].map({
                "Low": [0, 255, 0],
                "Medium": [255, 165, 0],
                "High": [255, 0, 0]
            })
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=pred_df,
                        get_position=["longitude", "latitude"],
                        get_fill_color="color",
                        get_radius=50000,
                    ),
                ],
            ))
        st.success("Prediction complete.")

# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
elif page == "Reports":
    st.header("📑 Reports")

    if st.session_state["pred_df"] is None:
        st.warning("Please run prediction first.")
    else:
        pred_df = st.session_state["pred_df"]
        df = st.session_state["data"]
        metrics = st.session_state["metrics"] or {}
        summary_text = "The predictive model has successfully classified pollution risk zones."

        col1, col2 = st.columns(2)
        with col1:
            excel_bytes = create_excel_report(df, pred_df)
            st.download_button(
                "📥 Download Excel Report",
                data=excel_bytes,
                file_name="microplastic_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            pdf_bytes = create_pdf_report(summary_text, pred_df, metrics)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="microplastic_report.pdf",
                mime="application/pdf"
            )
        st.success("Reports generated successfully.")
