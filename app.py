import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # XGBoost alternative
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import io
import base64
# For PDF generation (you'd need to install reportlab or fpdf)
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet

# --- Configuration and Theme ---
st.set_page_config(
    page_title="Microplastic Risk Assessment System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the environmental theme (blue and green tones)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #e0f7fa; /* Light cyan background */
    }
    .sidebar .sidebar-content {
        background: #00796b; /* Dark teal for sidebar */
        color: white;
    }
    .Widget>label {
        color: #004d40; /* Darker teal for widget labels */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #004d40; /* Darker teal for headers */
    }
    .stButton>button {
        background-color: #26a69a; /* Medium teal for buttons */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #009688; /* Slightly darker teal on hover */
    }
    .css-1d391kg { /* Main content area */
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 3.5rem;
        background-color: #e0f7fa; /* Light cyan background */
    }
    .css-1lcbmhc { /* Sidebar background */
        background-color: #00796b;
    }
    .css-vk32pt { /* Sidebar header/title */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = 'Risk_Level' # Default target

# --- Helper Functions (Backend Logic) ---

@st.cache_data # Cache this function to avoid re-running on every interaction
def generate_sample_data(num_samples=100):
    """Generates synthetic environmental data for demonstration."""
    np.random.seed(42)
    data = {
        'Location_ID': [f'Site_{i:03d}' for i in range(num_samples)],
        'Latitude': np.random.uniform(20, 50, num_samples),
        'Longitude': np.random.uniform(-120, -70, num_samples),
        'pH': np.random.uniform(6.5, 8.5, num_samples),
        'Turbidity': np.random.uniform(0.5, 50, num_samples), # NTU
        'Water_Temp_C': np.random.uniform(10, 30, num_samples),
        'Population_Density_km2': np.random.uniform(50, 5000, num_samples),
        'Industrial_Activity_Index': np.random.uniform(0, 10, num_samples),
        'Plankton_Density': np.random.uniform(100, 1000, num_samples), # cells/mL
        'Season': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], num_samples),
        'Year': np.random.randint(2018, 2023, num_samples)
    }
    df_sample = pd.DataFrame(data)

    # Introduce some correlation to create risk levels
    # Higher turbidity, population, industrial activity -> higher risk
    risk_score = (df_sample['Turbidity'] * 0.1 +
                  df_sample['Population_Density_km2'] * 0.005 +
                  df_sample['Industrial_Activity_Index'] * 5)

    df_sample['Risk_Level'] = pd.cut(risk_score,
                                     bins=[0, 200, 500, np.inf],
                                     labels=['Low', 'Medium', 'High'],
                                     right=False)
    # Ensure some data has missing values for preprocessing demo
    for col in ['pH', 'Turbidity', 'Water_Temp_C']:
        df_sample.loc[df_sample.sample(frac=0.05).index, col] = np.nan

    return df_sample

@st.cache_data
def clean_and_transform_data(df_input):
    """KDD Phase 2: Preprocessing & Transformation."""
    df_cleaned = df_input.copy()

    # Handle missing values (simple imputation for demo)
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

    # Encode categorical features if any
    for col in df_cleaned.select_dtypes(include='object').columns:
        if col != st.session_state.target_col and col not in ['Location_ID']: # Don't encode Location_ID or target
            le = LabelEncoder()
            df_cleaned[col + '_encoded'] = le.fit_transform(df_cleaned[col])
            # Drop original categorical column for modeling, keep for display
            # df_cleaned = df_cleaned.drop(columns=[col])

    return df_cleaned

@st.cache_resource # Cache the model itself
def train_model(X, y, model_type='RandomForest'):
    """KDD Phase 4: Data Mining."""
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        # You'd need to install xgboost: pip install xgboost
        # model = GradientBoostingClassifier(n_estimators=100, random_state=42) # Using GBC as an alternative if xgboost is not installed
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        st.error("Invalid model type specified.")
        return None

    model.fit(X, y)
    return model

@st.cache_data
def evaluate_model(model, X_test, y_test):
    """KDD Phase 5: Evaluation."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def get_table_download_link(df, filename, file_format="csv"):
    """Generates a link for downloading a dataframe."""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    elif file_format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
    else:
        href = ""
    return href

def generate_pdf_report_content(df_results, model_metrics):
    """Generates dummy content for a PDF report."""
    # In a real app, you'd use ReportLab or FPDF here to create a real PDF.
    # For now, it's just a placeholder string.
    report_content = f"""
    Microplastic Pollution Risk Assessment Report

    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    1. Introduction
    This report summarizes the findings from the Microplastic Risk Assessment System.
    The system analyzed environmental data to predict microplastic pollution risk levels.

    2. Data Overview
    Total data points analyzed: {len(df_results)}
    Predicted Risk Distribution:
    {df_results['Predicted_Risk'].value_counts().to_string()}

    3. Predictive Model Performance
    Model Type: Random Forest (or XGBoost if selected)
    Accuracy: {model_metrics['accuracy']:.2f}
    Precision (weighted): {model_metrics['precision']:.2f}
    Recall (weighted): {model_metrics['recall']:.2f}
    F1-Score (weighted): {model_metrics['f1']:.2f}

    4. Identified High-Risk Zones
    (Based on predicted 'High' risk)
    {df_results[df_results['Predicted_Risk'] == 'High'][['Location_ID', 'Latitude', 'Longitude', 'Turbidity', 'Population_Density_km2']].head().to_string()}
    ... (More details would go here)

    5. Suggested Mitigation Strategies
    - Implement stricter waste management protocols in high-risk urban areas.
    - Promote public awareness campaigns on responsible plastic disposal.
    - Invest in wastewater treatment upgrades to filter microplastics.
    - Conduct regular monitoring of pollution indicators in identified medium-risk zones.

    6. Conclusion
    The system provides actionable insights into microplastic pollution risks,
    enabling informed decision-making for environmental protection.
    """
    return report_content.strip()

# --- Streamlit UI ---

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"])

# --- Home Page ---
if page == "Home":
    st.title("🌊 Microplastic Risk Assessment System")
    st.markdown("### A Web-based Predictive Analytics Platform")
    st.write(
        """
        Welcome to the Microplastic Risk Assessment System. This platform helps assess and visualize
        the risk levels of microplastic pollution across various environments using advanced
        data mining algorithms. Our goal is to transform raw environmental data into actionable
        insights for monitoring, policy-making, and public awareness.

        **System Features:**
        - **Data Upload:** Easily upload your environmental datasets.
        - **Data Analysis:** Explore your data with descriptive statistics and interactive visualizations.
        - **Predictive Modeling:** Utilize machine learning (Random Forest, XGBoost) to classify pollution risks (Low, Medium, High).
        - **Interactive Dashboard:** Visualize pollution hotspots with geographic maps and trend charts.
        - **Comprehensive Reports:** Generate and download summary reports with findings and mitigation strategies.

        Navigate through the sidebar to get started!
        """
    )
    st.image("https://images.unsplash.com/photo-1547826039-bb7d35366471?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
             caption="Clean oceans, our shared goal.", use_column_width=True)

# --- Upload Dataset Page ---
elif page == "Upload Dataset":
    st.title("⬆️ Upload Your Dataset")
    st.write("Upload your environmental data in CSV or Excel format.")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else: # .xlsx
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("First 5 rows of your data:")
            st.dataframe(st.session_state.df.head())

            # Sidebar for selecting target column
            all_cols = st.session_state.df.columns.tolist()
            if 'Risk_Level' in all_cols:
                default_target_index = all_cols.index('Risk_Level')
            else:
                default_target_index = 0 if len(all_cols) > 0 else None

            selected_target = st.sidebar.selectbox(
                "Select your target (Risk Level) column:",
                options=all_cols,
                index=default_target_index
            )
            if selected_target:
                st.session_state.target_col = selected_target
                st.sidebar.write(f"Target column set to: **{st.session_state.target_col}**")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
    else:
        st.info("No file uploaded. Please upload a dataset or use sample data.")
        if st.button("Generate Sample Data"):
            st.session_state.df = generate_sample_data()
            st.success("Sample data generated!")
            st.write("First 5 rows of sample data:")
            st.dataframe(st.session_state.df.head())
            st.session_state.target_col = 'Risk_Level'
            st.sidebar.write(f"Target column set to: **{st.session_state.target_col}** (from sample data)")


# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    if st.session_state.df is not None:
        st.write("### Descriptive Statistics")
        st.dataframe(st.session_state.df.describe())

        st.write("### Data Information")
        buffer = io.StringIO()
        st.session_state.df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write("### Missing Values")
        missing_df = st.session_state.df.isnull().sum().to_frame(name='Missing Count')
        missing_df['Percentage'] = (missing_df['Missing Count'] / len(st.session_state.df)) * 100
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        if missing_df['Missing Count'].sum() == 0:
            st.info("No missing values found in the dataset.")

        st.write("### Feature Selection for Analysis")
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if st.session_state.target_col in numeric_cols:
            numeric_cols.remove(st.session_state.target_col)
        
        # Ensure 'Latitude' and 'Longitude' are available if they exist in the numeric columns
        available_geo_cols = [col for col in ['Latitude', 'Longitude'] if col in numeric_cols]
        if available_geo_cols:
            for col in available_geo_cols:
                numeric_cols.remove(col) # Remove from general numeric if we want specific handling

        selected_features_for_viz = st.sidebar.multiselect(
            "Select features for correlation & distribution:",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        st.session_state.feature_cols = selected_features_for_viz # Store for prediction

        if selected_features_for_viz:
            st.write("### Correlation Heatmap")
            corr_df = st.session_state.df[selected_features_for_viz + [st.session_state.target_col] if st.session_state.target_col in st.session_state.df.columns else selected_features_for_viz].corr(numeric_only=True)
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

            st.write("### Distribution Charts")
            for col in selected_features_for_viz:
                if st.session_state.df[col].dtype in ['int64', 'float64']:
                    fig = px.histogram(st.session_state.df, x=col, marginal="rug",
                                       title=f'Distribution of {col}',
                                       color_discrete_sequence=['#26a69a'])
                    st.plotly_chart(fig)
                else:
                    st.write(f"Distribution for non-numeric column '{col}' is not shown.")
        else:
            st.info("Please select features from the sidebar to visualize correlations and distributions.")
    else:
        st.warning("Please upload a dataset on the 'Upload Dataset' page first.")

# --- Prediction Dashboard Page ---
elif page == "Prediction Dashboard":
    st.title("📈 Prediction Dashboard")

    if st.session_state.df is None:
        st.warning("Please upload a dataset or generate sample data on the 'Upload Dataset' page.")
    elif st.session_state.target_col not in st.session_state.df.columns:
         st.error(f"Target column '{st.session_state.target_col}' not found in the dataset. Please select the correct target column on the 'Upload Dataset' page.")
    else:
        st.write("### Predictive Risk Assessment")

        cleaned_df = clean_and_transform_data(st.session_state.df.copy())

        # Identify all potential feature columns, excluding target and known IDs
        all_potential_features = [col for col in cleaned_df.columns if col not in [st.session_state.target_col, 'Location_ID', 'Latitude', 'Longitude']]
        
        # Update session_state.feature_cols to include encoded columns and numeric ones from the cleaned df
        current_numeric_features = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        current_numeric_features = [f for f in current_numeric_features if f not in ['Latitude', 'Longitude', st.session_state.target_col]]
        
        # Also add encoded categorical columns
        encoded_cols = [col for col in cleaned_df.columns if col.endswith('_encoded')]
        
        # Combine all suitable numeric and encoded features
        st.session_state.feature_cols = sorted(list(set(current_numeric_features + encoded_cols)))
        
        if not st.session_state.feature_cols:
            st.warning("No suitable numeric or encoded features found for training. Please check your dataset.")
            st.stop()
            
        X = cleaned_df[st.session_state.feature_cols]
        
        # Ensure target is encoded for classification
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(cleaned_df[st.session_state.target_col])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.selectbox("Choose a classification model:", ("RandomForest", "XGBoost"))
        
        if st.sidebar.button("Train Model and Predict"):
            with st.spinner("Training model and making predictions..."):
                st.session_state.model = train_model(X_train, y_train, model_type=model_choice)
                if st.session_state.model:
                    accuracy, precision, recall, f1 = evaluate_model(st.session_state.model, X_test, y_test)
                    st.session_state.model_accuracy = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

                    # Make predictions on the *entire* cleaned dataset
                    full_predictions_encoded = st.session_state.model.predict(X)
                    st.session_state.predictions = le_target.inverse_transform(full_predictions_encoded)
                    
                    # Store predictions in the original DataFrame for display
                    st.session_state.df['Predicted_Risk'] = st.session_state.predictions
                    st.session_state.df['Prediction_Confidence'] = np.max(st.session_state.model.predict_proba(X), axis=1)

                    st.success("Model trained and predictions generated!")
        
        if st.session_state.predictions is not None:
            st.write("### Model Performance Scorecard")
            if st.session_state.model_accuracy:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{st.session_state.model_accuracy['accuracy']:.2f}")
                col2.metric("Precision", f"{st.session_state.model_accuracy['precision']:.2f}")
                col3.metric("Recall", f"{st.session_state.model_accuracy['recall']:.2f}")
                col4.metric("F1-Score", f"{st.session_state.model_accuracy['f1']:.2f}")
            else:
                st.info("Train the model to see performance metrics.")

            st.write("### Predicted Risk Distribution")
            risk_counts = pd.Series(st.session_state.predictions).value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
            fig_risk = px.bar(risk_counts,
                              x=risk_counts.index,
                              y=risk_counts.values,
                              title="Distribution of Predicted Risk Levels",
                              labels={'x': 'Risk Level', 'y': 'Count'},
                              color=risk_counts.index,
                              color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
            st.plotly_chart(fig_risk)

            # --- Geographic Visualization ---
            st.write("### Microplastic Risk Map")
            
            # Check for Latitude and Longitude columns
            if 'Latitude' in st.session_state.df.columns and 'Longitude' in st.session_state.df.columns:
                
                # Replace with your actual Mapbox token
                # You can get one from mapbox.com
                # It's recommended to set this as an environment variable in production
                mapbox_access_token = st.secrets["mapbox_token"] if "mapbox_token" in st.secrets else "pk.YOUR_MAPBOX_TOKEN_HERE" # Replace with your token

                if mapbox_access_token == "pk.YOUR_MAPBOX_TOKEN_HERE":
                    st.warning("Please replace 'pk.YOUR_MAPBOX_TOKEN_HERE' with your actual Mapbox access token to enable the map.")
                else:
                    px.set_mapbox_access_token(mapbox_access_token)

                    # Create a copy with the relevant columns for the map
                    map_df = st.session_state.df[['Location_ID', 'Latitude', 'Longitude', 'Predicted_Risk', 'Prediction_Confidence'] + st.session_state.feature_cols].copy()
                    
                    # Define color scale for risk levels
                    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    
                    fig_map = px.scatter_mapbox(map_df,
                                                lat="Latitude",
                                                lon="Longitude",
                                                color="Predicted_Risk",
                                                size="Prediction_Confidence", # Use confidence to vary marker size
                                                hover_name="Location_ID",
                                                hover_data={'Predicted_Risk': True,
                                                            'Prediction_Confidence': ':.2f',
                                                            'Latitude': ':.2f',
                                                            'Longitude': ':.2f'} | {col: True for col in st.session_state.feature_cols[:3]}, # Add first few features to hover
                                                color_discrete_map=color_map,
                                                zoom=3,
                                                height=500,
                                                title="Predicted Microplastic Risk Locations")
                    fig_map.update_layout(mapbox_style="mapbox://styles/mapbox/light-v10", # light-v10, dark-v10, streets-v11, outdoors-v11, satellite-v9
                                        margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig_map)
            else:
                st.warning("Latitude and Longitude columns are required for the geographic risk map.")
        else:
            st.info("Train the model to see predictions and the risk map.")

# --- Reports Page ---
elif page == "Reports":
    st.title("📄 Reports")

    if st.session_state.df is None or st.session_state.predictions is None:
        st.warning("Please upload data, analyze it, and run predictions first.")
    else:
        st.write("### Generate Comprehensive Reports")
        st.write("Download detailed summary reports of the analysis and predictions.")

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                # In a real application, integrate with ReportLab or FPDF
                # For this demo, it's just a text file presented as a PDF download
                pdf_content = generate_pdf_report_content(st.session_state.df, st.session_state.model_accuracy)
                
                # To simulate PDF download, we'll encode text content for now
                b64_pdf = base64.b64encode(pdf_content.encode('utf-8')).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="microplastic_report.pdf">Download PDF Report</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")
                st.expander("View Report Content (Text representation)") \
                    .text(pdf_content)

        if st.button("Generate Excel Report (Raw Data with Predictions)"):
            with st.spinner("Generating Excel report..."):
                report_df = st.session_state.df.copy()
                if 'Predicted_Risk' in report_df.columns:
                    st.markdown(get_table_download_link(report_df, "microplastic_predictions", "excel"), unsafe_allow_html=True)
                    st.success("Excel report generated successfully!")
                else:
                    st.warning("No 'Predicted_Risk' column found. Please run predictions first.")

        st.write("---")
        st.write("### Summary of Current Predictions:")
        if 'Predicted_Risk' in st.session_state.df.columns:
            st.dataframe(st.session_state.df[['Location_ID', 'Latitude', 'Longitude', 'Predicted_Risk', 'Prediction_Confidence'] + st.session_state.feature_cols[:3]].head(10))
            st.write(f"Total entries: {len(st.session_state.df)}")
        else:
            st.info("No predictions available yet. Please navigate to the 'Prediction Dashboard' and train
