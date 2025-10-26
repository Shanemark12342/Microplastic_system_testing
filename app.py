import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF  # You'll need to install fpdf: pip install fpdf

# --- Configuration ---
st.set_page_config(
    page_title="Microplastic Pollution Risk System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming (Minimalist, blue/green) ---
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(#2e7bcf, #2cb4a0);
        color: white;
    }
    .Widget>label {
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0056b3; /* Darker blue for headers */
    }
    .stButton>button {
        background-color: #2cb4a0; /* Green for buttons */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #208e7a;
    }
    .stFileUploader>label {
        color: #0056b3;
    }
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg { /* This targets the Streamlit main content area directly */
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads data from CSV or Excel."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    """Data cleaning and preprocessing."""
    if df is None:
        return None

    st.subheader("Data Preprocessing Steps (Applied Automatically)")
    st.info(f"Initial dataset shape: {df.shape}")

    # Drop rows with any missing values
    original_rows = df.shape[0]
    df.dropna(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} rows with missing values.")

    # Drop duplicates
    original_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} duplicate rows.")

    # Basic feature engineering (if 'location' column exists)
    if 'location' in df.columns:
        df['location_encoded'] = df['location'].astype('category').cat.codes
        st.write("- Encoded 'location' column.")

    # Normalize numerical columns (for clustering)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        st.write("- Normalized numerical columns for clustering.")

    st.success("Preprocessing complete!")
    return df

def perform_clustering(df, features, n_clusters=3):
    """Performs K-Means clustering."""
    if not features:
        st.error("No features selected for clustering.")
        return None, None

    X = df[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    st.success(f"Clustering complete! Silhouette Score: {silhouette_avg:.2f}")
    return clusters, kmeans

def train_model(df, target_column, features, model_type, use_cv=False, cv_folds=5):
    """Model training with optional K-fold cross-validation."""
    if df is None or target_column not in df.columns or not features:
        st.error("Cannot train model. Please check data, target, and features.")
        return None, None, None, None

    X = df[features]
    y = df[target_column]

    # Ensure target column is numerical for classification
    if not pd.api.types.is_numeric_dtype(y):
        try:
            y = pd.Categorical(y, categories=['Low', 'Medium', 'High'], ordered=True).codes
            st.info(f"Target column '{target_column}' converted to numerical categories: 0=Low, 1=Medium, 2=High.")
        except:
            st.error(f"Could not convert target column '{target_column}' to numerical categories.")
            return None, None, None, None

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    if model:
        if use_cv:
            # K-fold Cross-Validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            st.write(f"Cross-Validation Accuracy Scores: {cv_scores}")
            st.write(f"Mean CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

            # Train on full data for final model
            model.fit(X, y)
            y_pred = model.predict(X)  # Predictions on full data for reporting

            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.success(f"Model '{model_type}' trained successfully!")
        st.write(f"**Model Performance:**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        # Store results for reporting
        st.session_state['model_report'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y if use_cv else y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True),
            'cv_used': use_cv,
            'cv_scores': cv_scores.tolist() if use_cv else None
        }
        return model, X if use_cv else X_test, y if use_cv else y_test
    return None, None, None

def generate_report(df, predictions, model_results, plot_buffer, clusters=None):
    """Generates a PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Microplastic Pollution Risk Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This report summarizes the analysis, clustering, predictions for microplastic pollution risk based on the provided dataset and models.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Analysis Overview", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"Dataset processed: {st.session_state.get('uploaded_filename', 'N/A')} with {df.shape[0]} rows and {df.shape[1]} columns.")
    pdf.multi_cell(0, 7, f"Key findings narrative: (Example: The analysis identified clusters of high-risk areas and strong correlations between pH levels and microplastic risk.)")
    pdf.ln(5)

    if clusters is not None:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Clustering Results", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Performed K-Means clustering into {len(np.unique(clusters))} clusters. Silhouette Score: {st.session_state.get('silhouette_score', 'N/A'):.2f}")
        pdf.ln(5)

    if model_results:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"3. Predictive Model Performance ({model_results['model_type']})", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 7, f"Accuracy: {model_results['accuracy']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Precision (Weighted): {model_results['precision']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Recall (Weighted): {model_results['recall']:.2f}", 0, 1)
        pdf.cell(0, 7, f"F1-Score (Weighted): {model_results['f1_score']:.2f}", 0, 1)
        if model_results['cv_used']:
            pdf.cell(0, 7, f"Mean CV Accuracy: {np.mean(model_results['cv_scores']):.2f}", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Classification Report:", 0, 1)
        pdf.set_font("Arial", "", 10)
        for class_name, metrics in model_results['classification_report'].items():
            if isinstance(metrics, dict):
                pdf.cell(0, 5, f"  {class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}, Support={metrics['support']}", 0, 1)
            else:
                pdf.cell(0, 5, f"  {class_name}: {metrics:.2f}", 0, 1)
        pdf.ln(5)

    if plot_buffer:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "4. Visualizations and Risk Map", 0, 1)
        pdf.ln(2)
        pdf.image(plot_buffer, x=10, y=pdf.get_y(), w=190)
        pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "5. Identified High-Risk Zones & Mitigation Strategies", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, "Based on clustering and predictions, high-risk zones are identified. Suggested mitigation strategies include enhanced waste management, public awareness campaigns, and stricter industrial regulations.")
    pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'test_data' not in st.session_state:
    st.session_state['test_data'] = None
if 'test_labels' not in st.session_state:
    st.session_state['test_labels'] = None
if 'model_report' not in st.session_state:
    st.session_state['model_report'] = None
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = "No file uploaded"
if 'clusters' not in st.session_state:
    st.session_state['clusters'] = None
if 'silhouette_score' not in st.session_state:
    st.session_state['silhouette_score'] = None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Clustering", "Prediction Dashboard", "Reports"]
)

st.sidebar.subheader("Input Variables (Global)")
if st.session_state['processed_df'] is not None:
    all_columns = st.session_state['processed_df'].columns.tolist()
    location_col = st.sidebar.selectbox("Select Location Column", ['None'] + all_columns, key='sidebar_location')
    pollution_indicators = st.sidebar.multiselect("Select Pollution Indicators", all_columns, key='sidebar_indicators')
    temporal_col = st.sidebar.selectbox("Select Temporal Column", ['None'] + all_columns, key='sidebar_temporal')

    if location_col != 'None':
        st.session_state['location_col'] = location_col
    else:
        st.session_state['location_col'] = None

    st.session_state['pollution_indicators'] = pollution_indicators
    if temporal_col != 'None':
        st.session_state['temporal_col'] = temporal_col
    else:
        st.session_state['temporal_col'] = None
else:
    st.sidebar.info("Upload a dataset first to select variables.")

# --- Main Content Area ---

if page == "Home":
    st.title("Welcome to the Microplastic Pollution Risk Assessment System")
    st.image("https://via.placeholder.com/700x300.png?text=Environmental+Sustainability", use_column_width=True)
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This platform leverages advanced predictive analytics to assess and visualize the risk levels of microplastic pollution across various environments.
        Utilizing Streamlit for an interactive user experience and powerful data mining algorithms like Random Forest, XGBoost, and K-Means clustering,
        we provide insights into pollution trends, potential high-risk zones, and actionable mitigation strategies.
        </p>
        <p style='font-size: 1.1em;'>
        Navigate through the sections to upload your data, preprocess it, perform clustering, generate predictions with cross-validation, and download comprehensive reports.
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Key Features:")
    st.markdown("""
    - **Data Upload & Preprocessing:** Seamlessly upload and clean environmental datasets.
    - **Descriptive Analytics:** Understand your data with interactive charts and statistics.
    - **Clustering:** Group data into clusters for pattern discovery.
    - **Predictive Modeling:** Classify microplastic pollution risk with K-fold cross-validation.
    - **Interactive Dashboards:** Visualize pollution intensity with heatmaps and time-series graphs.
    - **Comprehensive Reporting:** Download detailed reports in PDF or Excel.
    """)

elif page == "Upload Dataset":
    st.title("Upload Your Environmental Dataset")
    st.markdown("Please upload your environmental data file in CSV or Excel format (.csv, .xls, .xlsx).")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        st.session_state['uploaded_filename'] = uploaded_file.name
        with st.spinner("Loading and preprocessing data..."):
            st.session_state['df'] = load_data(uploaded_file)
            if st.session_state['df'] is not None:
                st.write("Original Data Preview:")
                st.dataframe(st.session_state['df'].head())
                st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy())

                if st.session_state['processed_df'] is not None:
                    st.success("Dataset loaded and preprocessed successfully!")
                    st.write("Processed Data Preview:")
                    st.dataframe(st.session_state['processed_df'].head())
                    st.write(f"Processed dataset shape: {st.session_state['processed_df'].shape}")
                else:
                    st.error("Data preprocessing failed.")
            else:
                st.error("Failed to load data.")
    else:
        st.info("Awaiting file upload.")

elif page == "Data Analysis":
    st.title("Data Analysis & Exploration")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df']
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

        st.subheader("Data Distribution")
        selected_column_dist = st.selectbox("Select a column to view its distribution:", df.columns)
        if selected_column_dist:
            fig = px.histogram(df, x=selected_column_dist, title=f'Distribution of {selected_column_dist}')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto",
                                 color_continuous_scale=px.colors.sequential.Plasma,
                                 title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns found for correlation analysis.")

        st.subheader("Time-series Trends (if temporal data is available)")
        if st.session_state['temporal_col'] and st.session_state['temporal_col'] in df.columns:
            try:
                df[st.session_state['temporal_col']] = pd.to_datetime(df'['st.session_state['temporal_col']
