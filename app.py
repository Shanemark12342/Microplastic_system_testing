import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF
import io

# Set page config
st.set_page_config(page_title="Microplastic Pollution Risk Assessment", layout="wide")

# Custom CSS for theme (blue and green tones)
st.markdown("""
<style>
    .main {background-color: #e0f7fa;}
    .sidebar .sidebar-content {background-color: #b2dfdb;}
    h1, h2, h3 {color: #004d40;}
    .stButton>button {background-color: #00796b; color: white;}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🌊 Microplastic Pollution Risk Assessment Platform")

# Sidebar menu
menu = st.sidebar.selectbox("Navigation", ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

# Home Page
if menu == "Home":
    st.header("Welcome to the Microplastic Pollution Risk Assessment Platform")
    st.write("""
    This platform uses predictive analytics to assess microplastic pollution risks across environments.
    Upload your data, analyze it, generate predictions, and download reports.
    
    **Key Features:**
    - Data upload and preprocessing
    - Descriptive statistics and visualizations
    - Machine learning predictions (Random Forest & XGBoost)
    - Interactive maps and dashboards
    - Downloadable reports
    
    Navigate using the sidebar to get started!
    """)
    st.image("https://via.placeholder.com/800x400/00796b/ffffff?text=Microplastic+Pollution+Awareness", use_column_width=True)  # Placeholder image

# Upload Dataset Page
elif menu == "Upload Dataset":
    st.header("Upload Your Dataset")
    st.write("Upload a CSV or Excel file containing environmental data (e.g., pH, turbidity, population density, location, etc.).")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['data'] = df
        st.success("Data uploaded successfully!")
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

# Data Analysis Page
elif menu == "Data Analysis":
    st.header("Data Analysis")
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        st.subheader("Data Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select a column for histogram", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[selected_col], bins=20, color='#00796b', alpha=0.7)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        
        st.subheader("Correlations")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        st.subheader("Time-Series Trends (if temporal data exists)")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            selected_ts_col = st.selectbox("Select column for time-series", numeric_cols)
            fig = px.line(df, x=df.index, y=selected_ts_col, title=f"Trend of {selected_ts_col}")
            st.plotly_chart(fig)
    else:
        st.warning("Please upload a dataset first.")

# Prediction Dashboard Page
elif menu == "Prediction Dashboard":
    st.header("Prediction Dashboard")
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        
        # Sidebar for model selection and features
        st.sidebar.subheader("Model Configuration")
        model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
        features = st.sidebar.multiselect("Select Features", df.columns.tolist(), default=[col for col in df.columns if col != 'risk'])
        target = st.sidebar.selectbox("Select Target (Risk Level)", df.columns.tolist(), index=df.columns.tolist().index('risk') if 'risk' in df.columns else 0)
        
        if features and target:
            X = df[features]
            y = df[target]
            
            # Preprocessing: Handle missing values, encode if needed
            X = X.fillna(X.mean())
            if y.dtype == 'object':
                y = y.astype('category').cat.codes  # Encode categorical target
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model Training
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                import xgboost as xgb
                model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), random_state=42)
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred_full = model.predict(X)
            
            # Metrics
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average='weighted')
            rec = recall_score(y_test, pred, average='weighted')
            f1 = f1_score(y_test, pred, average='weighted')
            
            st.session_state['model'] = model
            st.session_state['results'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
            st.session_state['predictions'] = pred_full
            
            # Display Metrics
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1-Score", f"{f1:.2f}")
            
            # Risk Classification
            st.subheader("Risk Classifications")
            risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            df['Predicted Risk'] = [risk_labels.get(p, 'Unknown') for p in pred_full]
            st.dataframe(df[['Predicted Risk']].head())
            
            # Visualizations
            st.subheader("Pollution Intensity Heatmap")
            if len(features) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(X.corr(), annot=True, cmap='viridis', ax=ax)
                st.pyplot(fig)
            
            st.subheader("Risk Map (if location data available)")
            if 'lat' in df.columns and 'lon' in df.columns:
                df['Predicted Risk'] = df['Predicted Risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
                fig = px.scatter_mapbox(df, lat='lat', lon='lon', color='Predicted Risk', 
                                        color_continuous_scale=['green', 'orange', 'red'], 
                                        mapbox_style="open-street-map", title="Risk Map")
                st.plotly_chart(fig)
            else:
                st.info("Add 'lat' and 'lon' columns for map visualization.")
            
            st.subheader("Bar Chart of Risk Levels")
            risk_counts = df['Predicted Risk'].value_counts()
            fig = px.bar(risk_counts, x=risk_counts.index, y=risk_counts.values, color=risk_counts.index,
                         color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
            st.plotly_chart(fig)
    else:
        st.warning("Please upload a dataset first.")

# Reports Page
elif menu == "Reports":
    st.header("Generate and Download Reports")
    if st.session_state['results'] is not None and st.session_state['data'] is not None:
        results = st.session_state['results']
        df = st.session_state['data']
        
        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Microplastic Pollution Risk Assessment Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt="Model Performance:", ln=True)
        pdf.cell(200, 10, txt=f"Accuracy: {results['accuracy']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Precision: {results['precision']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Recall: {results['recall']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"F1-Score: {results['f1']:.2f}", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Summary of Findings:", ln=True)
        pdf.multi_cell(0, 10, txt="High-risk areas are typically near industrial zones. Mitigation strategies include reducing plastic waste and monitoring water quality.")
        pdf.ln(10)
        pdf.cell(200, 10, txt="High-Risk Zones: [List based on predictions]", ln=True)
        
        # Download PDF
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button("Download PDF Report", pdf_buffer, file_name="pollution_report.pdf", mime="application/pdf")
        
        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            results_df = pd.DataFrame([results])
            results_df.to_excel(writer, sheet_name='Results', index=False)
        output.seek(0)
        st.download_button("Download Excel Report", output, file_name="pollution_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("Please run predictions first to generate a report.")
