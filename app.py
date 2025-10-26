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
from fpdf import FPDF # You'll need to install fpdf2: pip install fpdf2

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
    """Placeholder for data cleaning and preprocessing."""
    if df is None:
        return None

    st.subheader("Data Preprocessing Steps (Applied Automatically)")
    st.info(f"Initial dataset shape: {df.shape}")

    # Example: Drop rows with any missing values (for simplicity)
    original_rows = df.shape[0]
    df.dropna(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} rows with missing values.")

    # Example: Drop duplicates
    original_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    st.write(f"- Removed {original_rows - df.shape[0]} duplicate rows.")

    # Example: Basic feature engineering (if 'location' column exists)
    # This might not be needed if 'location' is just used for display
    if 'location' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['location']):
            df['location_encoded'] = df['location'].astype('category').cat.codes
            st.write("- Encoded 'location' column to 'location_encoded'.")

    st.success("Preprocessing complete!")
    return df

def train_model(df_original, target_column, features, model_type):
    """Trains a machine learning model and stores results in session state."""
    df = df_original.copy() # Work on a copy of the dataframe

    if df is None or target_column not in df.columns or not features:
        st.error("Cannot train model. Please check data, target, and features.")
        return None, None, None

    # Ensure all features are numeric for training
    # For now, we'll only select numeric features already passed
    X = df[features]
    
    # Handle the target variable (y)
    y = df[target_column]

    # Store original unique labels for mapping and reporting
    unique_labels = None
    if not pd.api.types.is_numeric_dtype(y):
        try:
            unique_labels = sorted(y.unique().tolist()) # Get original unique labels here
            st.session_state['unique_labels_map'] = {i: label for i, label in enumerate(unique_labels)} # Store for later
            category_type = pd.CategoricalDtype(categories=unique_labels, ordered=True)
            y = y.astype(category_type).cat.codes
            st.info(f"Target column '{target_column}' converted to numerical categories: {st.session_state['unique_labels_map']}")
        except Exception as e:
            st.error(f"Could not convert target column '{target_column}' to numerical categories. Error: {e}")
            return None, None, None
    else:
        # If target is already numeric, ensure unique_labels_map is not set or handle as purely numeric
        unique_labels = sorted(y.unique().tolist())
        st.session_state['unique_labels_map'] = {label: label for label in unique_labels} # Map numeric to itself


    # Ensure y is a Pandas Series for the stratify logic to work correctly
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index) # Maintain index if converting

    # Check for number of unique classes after conversion
    if len(y.unique()) <= 1:
        st.warning(f"The target column '{target_column}' has only one unique class after preprocessing. Cannot perform stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=None)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        objective_choice = 'multi:softmax' if len(y.unique()) > 2 else 'binary:logistic'
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective=objective_choice, random_state=42)

    if model:
        model.fit(X_train, y_train)
        y_pred_numeric = model.predict(X_test)

        # Convert numeric predictions and true labels back to original string labels
        if unique_labels and isinstance(st.session_state.get('unique_labels_map'), dict) and any(isinstance(v, str) for v in st.session_state['unique_labels_map'].values()):
            # Map back using the stored unique_labels_map values
            y_pred_labels = pd.Series(y_pred_numeric).map(st.session_state['unique_labels_map']).values
            y_test_labels = pd.Series(y_test).map(st.session_state['unique_labels_map']).values
            target_names_for_report = unique_labels
        else:
            y_pred_labels = y_pred_numeric # Fallback for purely numeric targets
            y_test_labels = y_test
            target_names_for_report = [str(x) for x in sorted(np.unique(y_test))]


        accuracy = accuracy_score(y_test, y_pred_numeric)
        precision = precision_score(y_test, y_pred_numeric, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_numeric, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_numeric, average='weighted', zero_division=0)

        st.success(f"Model '{model_type}' trained successfully!")
        st.write(f"**Model Performance:**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        # Store results in session state
        st.session_state['model'] = model
        st.session_state['test_data'] = X_test # Keep X_test for display context
        st.session_state['test_labels'] = y_test_labels # Store labels for display
        st.session_state['predictions'] = y_pred_labels # Store labels for display

        st.session_state['model_report'] = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred_numeric, target_names=target_names_for_report, output_dict=True)
        }
        return model, X_test, y_test_labels # Return labels for immediate use
    return None, None, None

def generate_report(df, predictions, model_results, plot_buffer):
    """Generates a PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Microplastic Pollution Risk Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "This report summarizes the analysis and predictions for microplastic pollution risk based on the provided dataset and predictive model.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Analysis Overview", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"Dataset processed: {st.session_state.get('uploaded_filename', 'N/A')} with {df.shape[0]} rows and {df.shape[1]} columns.")
    pdf.multi_cell(0, 7, f"Key findings narrative: (Example: The analysis identified a strong correlation between pH levels and microplastic risk. High-risk zones are predominantly found in coastal urban areas.)")
    pdf.ln(5)

    if model_results:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"2. Predictive Model Performance ({model_results['model_type']})", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 7, f"Accuracy: {model_results['accuracy']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Precision (Weighted): {model_results['precision']:.2f}", 0, 1)
        pdf.cell(0, 7, f"Recall (Weighted): {model_results['recall']:.2f}", 0, 1)
        pdf.cell(0, 7, f"F1-Score (Weighted): {model_results['f1_score']:.2f}", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Classification Report:", 0, 1)
        pdf.set_font("Arial", "", 10)
        # Assuming the classification_report in model_results is already a dict
        report_data = model_results['classification_report']

        # Get target names in the correct order from session_state if available
        target_names_ordered = list(st.session_state.get('unique_labels_map', {}).values())
        if not target_names_ordered: # Fallback if map not present or numeric target
            target_names_ordered = [k for k in report_data.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
        for class_name in target_names_ordered:
            if class_name in report_data and isinstance(report_data[class_name], dict):
                metrics = report_data[class_name]
                pdf.cell(0, 5, f"  {class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}, Support={metrics['support']}", 0, 1)
        # Also print averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_data and isinstance(report_data[avg_type], dict): # Check if it's a dict for metrics
                 metrics = report_data[avg_type]
                 pdf.cell(0, 5, f"  {avg_type.replace('_', ' ').title()}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}", 0, 1)
        pdf.ln(5)


    if plot_buffer:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Visualizations and Risk Map", 0, 1)
        pdf.ln(2)
        # Add the image from the buffer
        # Ensure the image fits within the PDF page width (e.g., 190mm if page width is 210mm)
        pdf.image(BytesIO(plot_buffer), x=10, y=pdf.get_y(), w=190)
        pdf.ln(10) # Adjust line break after image


    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Identified High-Risk Zones & Mitigation Strategies", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, "Based on the predictions, high-risk zones (e.g., coordinates X, Y or specific locations A, B) are identified in areas with high industrial discharge and dense population. Suggested mitigation strategies include enhanced waste management, public awareness campaigns, and stricter industrial regulations.")
    pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')


# --- Initialize Session State ---
# Initialize all session state variables to None or appropriate defaults
if 'df' not in st.session_state: st.session_state['df'] = None
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'model' not in st.session_state: st.session_state['model'] = None
if 'predictions' not in st.session_state: st.session_state['predictions'] = None
if 'test_data' not in st.session_state: st.session_state['test_data'] = None
if 'test_labels' not in st.session_state: st.session_state['test_labels'] = None
if 'model_report' not in st.session_state: st.session_state['model_report'] = None
if 'uploaded_filename' not in st.session_state: st.session_state['uploaded_filename'] = "No file uploaded"
if 'location_col' not in st.session_state: st.session_state['location_col'] = None
if 'pollution_indicators' not in st.session_state: st.session_state['pollution_indicators'] = []
if 'temporal_col' not in st.session_state: st.session_state['temporal_col'] = None
if 'risk_map_plot_buffer' not in st.session_state: st.session_state['risk_map_plot_buffer'] = None
if 'unique_labels_map' not in st.session_state: st.session_state['unique_labels_map'] = {}


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"]
)

st.sidebar.subheader("Input Variables (Global)")
if st.session_state['processed_df'] is not None:
    all_columns = st.session_state['processed_df'].columns.tolist()
    
    # Filter out columns that are primarily identifiers or non-numeric for pollution indicators
    suggested_pollution_cols = [col for col in all_columns if pd.api.types.is_numeric_dtype(st.session_state['processed_df'][col]) and col not in ['latitude', 'longitude']]

    location_col_options = ['None'] + [col for col in all_columns if st.session_state['processed_df'][col].nunique() < 100 or not pd.api.types.is_numeric_dtype(st.session_state['processed_df'][col])]
    location_col_index = 0
    if st.session_state['location_col'] in location_col_options:
        location_col_index = location_col_options.index(st.session_state['location_col'])
    location_col = st.sidebar.selectbox("Select Location Column (for map/hover)", location_col_options, key='sidebar_location', index=location_col_index)
    
    pollution_indicators = st.sidebar.multiselect("Select Pollution Indicators", suggested_pollution_cols, default=st.session_state['pollution_indicators'] if st.session_state['pollution_indicators'] else [], key='sidebar_indicators')
    
    temporal_col_options = ['None'] + [col for col in all_columns if pd.api.types.is_datetime64_any_dtype(st.session_state['processed_df'][col]) or pd.api.types.is_string_dtype(st.session_state['processed_df'][col])] # Allow string for potential conversion
    temporal_col_index = 0
    if st.session_state['temporal_col'] in temporal_col_options:
        temporal_col_index = temporal_col_options.index(st.session_state['temporal_col'])
    temporal_col = st.sidebar.selectbox("Select Temporal Column", temporal_col_options, key='sidebar_temporal', index=temporal_col_index)

    st.session_state['location_col'] = location_col if location_col != 'None' else None
    st.session_state['pollution_indicators'] = pollution_indicators
    st.session_state['temporal_col'] = temporal_col if temporal_col != 'None' else None
else:
    st.sidebar.info("Upload a dataset first to select variables.")


# --- Main Content Area ---

if page == "Home":
    st.title("Welcome to the Microplastic Pollution Risk Assessment System")
    # Using an actual image from Unsplash for better visual appeal
    st.image("https://images.unsplash.com/photo-1596499870503-452f1e403d7c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80", caption="Sustainable Oceans", use_column_width=True)
    st.markdown("""
        <p style='font-size: 1.1em;'>
        This platform leverages advanced predictive analytics to assess and visualize the risk levels of microplastic pollution across various environments.
        Utilizing Streamlit for an interactive user experience and powerful data mining algorithms like Random Forest and XGBoost,
        we provide insights into pollution trends, potential high-risk zones, and actionable mitigation strategies.
        </p>
        <p style='font-size: 1.1em;'>
        Navigate through the sections to upload your data, analyze it, generate predictions, and download comprehensive reports.
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Key Features:")
    st.markdown("""
    - **Data Upload & Preprocessing:** Seamlessly upload and clean environmental datasets.
    - **Descriptive Analytics:** Understand your data with interactive charts and statistics.
    - **Predictive Modeling:** Classify microplastic pollution risk (Low, Medium, High).
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
                st.session_state['processed_df'] = preprocess_data(st.session_state['df'].copy()) # Pass a copy

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
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty and len(numeric_df.columns) > 1: # Ensure at least two numeric columns
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto",
                                 color_continuous_scale=px.colors.sequential.Plasma,
                                 title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns found for correlation analysis.")

        st.subheader("Time-series Trends (if temporal data is available)")
        if st.session_state['temporal_col'] and st.session_state['temporal_col'] in df.columns:
            try:
                df[st.session_state['temporal_col']] = pd.to_datetime(df[st.session_state['temporal_col']])
                # Group by temporal column and average selected indicators
                if st.session_state['pollution_indicators']:
                    trend_df = df.groupby(st.session_state['temporal_col'])[st.session_state['pollution_indicators']].mean().reset_index()
                    fig_time = px.line(trend_df, x=st.session_state['temporal_col'], y=st.session_state['pollution_indicators'],
                                       title="Pollution Indicator Trends Over Time")
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("Select pollution indicators in the sidebar to view time trends.")
            except Exception as e:
                st.warning(f"Could not plot time series. Ensure '{st.session_state['temporal_col']}' is a valid date column. Error: {e}")
        else:
            st.info("Select a temporal column in the sidebar to view time-series trends.")

    else:
        st.warning("Please upload a dataset on the 'Upload Dataset' page first.")

elif page == "Prediction Dashboard":
    st.title("Prediction Dashboard")

    if st.session_state['processed_df'] is not None:
        df = st.session_state['processed_df'].copy() # Use a copy to avoid modifying original

        st.subheader("Model Configuration")
        target_column_options = df.columns.tolist()
        # Filter out potential ID columns or non-numeric for target, focusing on low cardinality
        target_column_options = [col for col in target_column_options if df[col].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[col])]
        target_column = st.selectbox("Select Target Variable (e.g., 'Risk_Level')", target_column_options)

        # Automatically exclude target, location, and temporal columns from features list
        available_features = [col for col in df.columns if col != target_column
                              and col != st.session_state.get('location_col')
                              and col != st.session_state.get('temporal_col')
                              and not pd.api.types.is_datetime64_any_dtype(df[col])] # Exclude datetime
        # Filter features to include only numeric ones for ML models (simplification)
        numeric_features = df[available_features].select_dtypes(include=np.number).columns.tolist()
        feature_columns = st.multiselect("Select Features for Prediction (Numeric Recommended)", numeric_features, default=numeric_features)


        model_type = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model & Generate Predictions"):
            if target_column and feature_columns:
                with st.spinner(f"Training {model_type} model..."):
                    # Call train_model, which now stores results directly in session_state
                    model, X_test, y_test_labels = train_model(df, target_column, feature_columns, model_type)

                    if model:
                        st.success("Model trained and predictions generated!")
                    else:
                        st.error("Model training failed. Please check your selections.")
            else:
                st.warning("Please select a target variable and at least one feature.")

        # --- Display Predictions and Visualizations ---
        if st.session_state['model'] is not None and st.session_state['predictions'] is not None:
            st.subheader("Prediction Results & Dashboard")

            # 1. Prediction Sample Table
            st.markdown("#### Sample of Predicted Risks")
            prediction_display_df = pd.DataFrame(st.session_state['test_data'].copy())
            prediction_display_df['True_Risk'] = st.session_state['test_labels']
            prediction_display_df['Predicted_Risk'] = st.session_state['predictions']
            st.dataframe(prediction_display_df.head(10)) # Show top 10 predictions


            # 2. Risk Distribution Bar Chart
            st.markdown("#### Predicted Risk Distribution")
            predicted_risk_counts = pd.Series(st.session_state['predictions']).value_counts()
            # Ensure consistent order for risk levels if they are categorical
            risk_level_order = ['Low', 'Medium', 'High'] # Define expected order
            predicted_risk_counts = predicted_risk_counts.reindex(risk_level_order).fillna(0) # Reindex and fill missing with 0

            fig_risk_dist = px.bar(predicted_risk_counts,
                                   x=predicted_risk_counts.index,
                                   y=predicted_risk_counts.values,
                                   labels={'x': 'Risk Level', 'y': 'Number of Sites'},
                                   color=predicted_risk_counts.index,
                                   color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                                   title="Distribution of Predicted Risk Levels")
            st.plotly_chart(fig_risk_dist, use_container_width=True)


            # 3. Geographic Heatmap (Risk Map)
            st.markdown("#### Microplastic Pollution Risk Map")
            location_col = st.session_state.get('location_col')

            # Create a DataFrame for mapping that combines X_test (features) with predictions
            map_df = st.session_state['test_data'].copy()
            map_df['Predicted_Risk'] = st.session_state['predictions']
            map_df['True_Risk'] = st.session_state['test_labels']

            # --- Robust way to get Lat/Lon for the test set ---
            # Try to merge with the original processed_df using the index
            # This assumes your processed_df still has the original index from before train_test_split
            if ('latitude' in df.columns and 'longitude' in df.columns):
                # If X_test inherited original indices, merge directly
                if map_df.index.isin(df.index).all():
                    # Ensure df has the columns before attempting to select them
                    cols_to_merge = ['latitude', 'longitude']
                    if location_col and location_col in df.columns:
                        cols_to_merge.append(location_col)
                    map_df = map_df.merge(df[cols_to_merge], left_index=True, right_index=True, how='left')
                else:
                    st.warning("Original DataFrame index not fully preserved in test set. Attempting to add lat/lon from original df based on available features (less reliable).")
                    # Fallback: Merge on available columns if index merge fails or is not applicable
                    merge_cols = [col for col in map_df.columns if col in df.columns and col not in ['Predicted_Risk', 'True_Risk']]
                    if merge_cols:
                        cols_to_get = ['latitude', 'longitude']
                        if location_col and location_col in df.columns:
                            cols_to_get.append(location_col)
                        
                        df_for_merge = df[cols_to_get + merge_cols].drop_duplicates()
                        map_df = map_df.merge(df_for_merge, on=merge_cols, how='left', suffixes=('', '_df'))
                        
                        # Handle potential duplicate columns after merge
                        for c in ['latitude', 'longitude']:
                            if f'{c}_df' in map_df.columns:
                                map_df[c] = map_df[c].fillna(map_df[f'{c}_df'])
                                map_df = map_df.drop(columns=[f'{c}_df'])
                        if location_col and f'{location_col}_df' in map_df.columns:
                             map_df[location_col] = map_df[location_col].fillna(map_df[f'{location_col}_df'])
                             map_df = map_df.drop(columns=[f'{location_col}_df'])
                    else:
                        st.error("Cannot merge for latitude/longitude without common columns between test data and original data.")

                # Drop rows where latitude or longitude are still missing after merge attempts
                map_df.dropna(subset=['latitude', 'longitude'], inplace=True)

            if 'latitude' in map_df.columns and 'longitude' in map_df.columns and not map_df.empty:
    risk_color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="Predicted_Risk",
        color_discrete_map=risk_color_map,
        size_max=15,
        zoom=1,
        hover_name=location_col if location_col else None,
        hover_data={
            "Predicted_Risk": True,
            "True_Risk": True,
            "latitude": ':.2f',
            "longitude": ':.2f'
        } if location_col else {
            "Predicted_Risk": True,
            "True_Risk": True,
            "latitude": ':.2f',
            "longitude": ':.2f'
        }
    )

    fig_map.update_layout(
        mapbox_style="carto-positron",
        title="Geographical Distribution of Predicted Microplastic Risk",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Latitude/Longitude data not available for mapping.")
