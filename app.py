import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
import base64
from PIL import Image
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(page_title="Microplastic Pollution Risk Assessment", page_icon="🌊", layout="wide")

# Custom CSS for theme
st.markdown("""
<style>
    .main {
        background-color: #e8f4f8;
    }
    .sidebar .sidebar-content {
        background-color: #d1ecf1;
    }
    h1, h2, h3 {
        color: #2e8b57;
    }
    .stButton>button {
        background-color: #20b2aa;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Menu", ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"])

# Global variables
data = None
model = None
predictions = None
report_data = {}

# Home Page
if page == "Home":
    st.title("🌊 Microplastic Pollution Risk Assessment Platform")
    st.markdown("""
    Welcome to our predictive analytics platform for assessing microplastic pollution risks across environments.
    
    This system uses advanced machine learning to analyze environmental data and predict pollution risk levels.
    
    **Features:**
    - Upload and analyze datasets
    - Visualize pollution trends
    - Predict risk levels with Random Forest and XGBoost
    - Generate downloadable reports
    
    Navigate using the sidebar to get started!
    """)
    st.image("https://via.placeholder.com/800x400/2e8b57/ffffff?text=Microplastic+Pollution+Awareness", use_column_width=True)

# Upload Dataset Page
elif page == "Upload Dataset":
    st.title("📤 Upload Dataset")
    st.markdown("Upload your environmental data in CSV or Excel format. The dataset should include columns like location, pH, turbidity, population_density, and risk_level (for training).")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.session_state['data'] = data
        st.success("Dataset uploaded successfully!")
        st.dataframe(data.head())

# Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
    else:
        data = st.session_state['data']
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(data.describe())
        
        # Correlations
        st.subheader("Correlation Matrix")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots()
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)
        
        # Distribution charts
        st.subheader("Distribution Charts")
        col1, col2 = st.columns(2)
        with col1:
            if 'pH' in data.columns:
                fig = px.histogram(data, x='pH', title="pH Distribution")
                st.plotly_chart(fig)
        with col2:
            if 'turbidity' in data.columns:
                fig = px.histogram(data, x='turbidity', title="Turbidity Distribution")
                st.plotly_chart(fig)

# Prediction Dashboard Page
elif page == "Prediction Dashboard":
    st.title("🔮 Prediction Dashboard")
    if 'data' not in st.session_state:
        st.error("Please upload a dataset first.")
    else:
        data = st.session_state['data']
        
        # Sidebar for model selection
        st.sidebar.subheader("Model Settings")
        algorithm = st.sidebar.selectbox("Algorithm", ["Random Forest", "XGBoost"])
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        
        # Prepare data
        if 'risk_level' in data.columns:
            features = data.drop('risk_level', axis=1).select_dtypes(include=[np.number])
            target = data['risk_level']
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
            
            if algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = xgb.XGBClassifier(random_state=42)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            st.session_state['model'] = model
            st.session_state['predictions'] = predictions
            st.session_state['accuracy'] = accuracy
            st.session_state['precision'] = precision
            st.session_state['recall'] = recall
            st.session_state['f1'] = f1
            
            # Model performance
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{precision:.2f}")
            col3.metric("Recall", f"{recall:.2f}")
            col4.metric("F1-Score", f"{f1:.2f}")
            
            # Risk map (assuming lat/lon columns)
            if 'latitude' in data.columns and 'longitude' in data.columns:
                st.subheader("Risk Map")
                data['predicted_risk'] = model.predict(features)
                risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                data['color'] = data['predicted_risk'].map(risk_colors)
                
                m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)
                for idx, row in data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        color=row['color'],
                        fill=True,
                        fill_color=row['color'],
                        popup=f"Risk: {row['predicted_risk']}"
                    ).add_to(m)
                st_folium(m, width=700, height=500)
            
            # Classification results
            st.subheader("Classification Results")
            fig = px.bar(data['predicted_risk'].value_counts(), title="Predicted Risk Levels")
            st.plotly_chart(fig)
            
            # Time-series if temporal data exists
            if 'date' in data.columns:
                st.subheader("Pollution Trends Over Time")
                data['date'] = pd.to_datetime(data['date'])
                fig = px.line(data, x='date', y='turbidity', title="Turbidity Over Time")
                st.plotly_chart(fig)
        else:
            st.error("Dataset must contain a 'risk_level' column for training.")

# Reports Page
elif page == "Reports":
    st.title("📄 Reports")
    if 'data' not in st.session_state or 'model' not in st.session_state:
        st.error("Please upload data and run predictions first.")
    else:
        data = st.session_state['data']
        accuracy = st.session_state['accuracy']
        precision = st.session_state['precision']
        recall = st.session_state['recall']
        f1 = st.session_state['f1']
        
        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Microplastic Pollution Risk Assessment Report", ln=True, align='C')
        pdf.cell(200, 10, txt="", ln=True)
        pdf.cell(200, 10, txt=f"Model Accuracy: {accuracy:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Precision: {precision:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Recall: {recall:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"F1-Score: {f1:.2f}", ln=True)
        pdf.cell(200, 10, txt="", ln=True)
        pdf.cell(200, 10, txt="High-risk zones: Areas with predicted 'High' risk.", ln=True)
        pdf.cell(200, 10, txt="Mitigation: Implement stricter pollution controls.", ln=True)
        
        # Download PDF
        pdf_output = pdf.output(dest='S').encode('latin1')
        b64 = base64.b64encode(pdf_output).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=False)
        excel_data = output.getvalue()
        b64_excel = base64.b64encode(excel_data).decode()
        href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="data.xlsx">Download Excel Data</a>'
        st.markdown(href_excel, unsafe_allow_html=True)
