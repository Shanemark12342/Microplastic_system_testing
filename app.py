import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF

# --- Streamlit Page Config ---
st.set_page_config(page_title="Microplastic Risk System", page_icon="🌍", layout="wide")

# --- Helper Functions ---

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        st.error("Please upload a CSV or Excel file.")
        return None

def preprocess_data(df):
    if df is None:
        return None
    st.info(f"Initial dataset shape: {df.shape}")
    df = df.dropna().drop_duplicates()
    if 'location' in df.columns and not pd.api.types.is_numeric_dtype(df['location']):
        df['location_encoded'] = df['location'].astype('category').cat.codes
    st.success("Preprocessing complete!")
    return df

def safe_train_test_split(X, y, test_size=0.3, random_state=42):
    """Prevent stratify errors when few samples or classes."""
    if len(y.unique()) < 2 or len(y) < 5:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(df, target_col, features, model_type):
    X = df[features].select_dtypes(include=np.number)
    y = df[target_col]

    # Convert categorical target to numeric
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype('category').cat.codes

    if len(y.unique()) <= 1:
        st.error("Target column must have at least two classes.")
        return None, None, None

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    st.success(f"Model trained: {model_type}")
    st.write(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    st.session_state['model'] = model
    st.session_state['pred_df'] = pd.DataFrame({
        "True": y_test,
        "Predicted": y_pred
    })
    return model, X_test, y_test

# --- App UI ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload", "Analysis", "Predict"])

if page == "Home":
    st.title("🌍 Microplastic Pollution Risk System")
    st.markdown("""
    This app analyzes and predicts microplastic pollution risk using your environmental dataset.
    Upload your data, explore patterns, train a predictive model, and visualize risk on an interactive map.
    """)

elif page == "Upload":
    st.title("Upload Dataset")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    if file:
        df = load_data(file)
        st.session_state['df'] = preprocess_data(df)
        st.dataframe(st.session_state['df'].head())

elif page == "Analysis":
    st.title("Data Analysis")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            fig, ax = plt.subplots()
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation.")
    else:
        st.info("Upload data first.")

elif page == "Predict":
    st.title("Prediction Dashboard")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']

        # Rename automatically for compatibility
        if 'Dominant_Risk_Type' in df.columns and 'Risk_Level' not in df.columns:
            df['Risk_Level'] = df['Dominant_Risk_Type']
        if 'MP_Count (items/individual)' in df.columns and 'mp_conc' not in df.columns:
            df['mp_conc'] = df['MP_Count (items/individual)']

        target = st.selectbox("Select Target Column", [c for c in df.columns if df[c].nunique() <= 10])
        features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target])

        model_type = st.radio("Choose Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model"):
            if features and target:
                model, X_test, y_test = train_model(df, target, features, model_type)
                if model:
                    pred_df = st.session_state['pred_df']
                    st.dataframe(pred_df.head())

                    st.subheader("Prediction Distribution")
                    fig = px.histogram(pred_df, x="Predicted", color="Predicted", title="Predicted Risk Distribution")
                    st.plotly_chart(fig)

                    if 'latitude' in df.columns and 'longitude' in df.columns:
                        st.subheader("Predicted Risk Map")
                        map_df = pd.concat([X_test, pred_df], axis=1)
                        fig_map = px.scatter_mapbox(
                            map_df,
                            lat="latitude",
                            lon="longitude",
                            color="Predicted",
                            color_discrete_sequence=["green", "orange", "red"],
                            zoom=2,
                            title="Predicted Risk Map"
                        )
                        fig_map.update_layout(mapbox_style="carto-positron")
                        st.plotly_chart(fig_map)
                    else:
                        st.info("Latitude/Longitude not found for mapping.")
            else:
                st.warning("Select target and features first.")
    else:
        st.info("Please upload data first.")
