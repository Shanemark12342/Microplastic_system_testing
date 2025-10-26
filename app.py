# File: app.py
"""
Streamlit app for Microplastic Pollution Risk Analytics
Requirements (pip): streamlit pandas numpy scikit-learn plotly matplotlib fpdf openpyxl xlsxwriter
Optional (for faster boosting): xgboost
Run: streamlit run app.py
"""
from io import BytesIO
import base64
import math
import datetime
import tempfile

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from fpdf import FPDF
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import KMeans

# Optional xgboost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# -------------------------
# Helper utilities
# -------------------------

def load_data(uploaded_file):
    """Read CSV or Excel into DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)  # try CSV fallback
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def preprocess(df, numeric_cols=None, categorical_cols=None, fill_method='mean', scale=True, dropna_thresh=0.0):
    """Clean data: drop duplicates, handle missing values, encode categorical, scale numeric.
    Returns: X (DataFrame), encoders dict, scaler (or None), preprocess_info dict
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # drop columns with too many missing values
    if dropna_thresh > 0:
        thresh = int((1 - dropna_thresh) * len(df))
        df = df.dropna(axis=1, thresh=thresh)

    # identify columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # numeric imputer
    if fill_method in ('mean', 'median'):
        strategy = fill_method
    else:
        strategy = 'mean'
    num_imp = SimpleImputer(strategy=strategy)
    if numeric_cols:
        try:
            df[numeric_cols] = num_imp.fit_transform(df[numeric_cols])
        except Exception:
            # if single column, ensure 2D
            df[numeric_cols] = np.array([num_imp.fit_transform(df[[c]])[:,0] for c in numeric_cols]).T

    # categorical encoding (LabelEncoder)
    encoders = {}
    for col in categorical_cols:
        if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
            le = LabelEncoder()
            df[col] = df[col].fillna("MISSING")
            try:
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            except Exception:
                # fallback: map unique to ints
                uniq = {k: i for i, k in enumerate(df[col].astype(str).unique())}
                df[col] = df[col].astype(str).map(uniq)
                encoders[col] = uniq

    scaler = None
    if scale and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    preprocess_info = dict(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    return df, encoders, scaler, preprocess_info

def kfold_cv_and_train(model, X, y, cv=5):
    """Perform KFold CV and return dict of metrics and final refitted model (on whole data)."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=kf)
    acc = accuracy_score(y, y_pred_cv)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred_cv, average='weighted', zero_division=0)
    cm = confusion_matrix(y, y_pred_cv)
    # fit model on whole data
    model.fit(X, y)
    return {
        'model': model,
        'cv_accuracy': acc,
        'cv_precision': prec,
        'cv_recall': rec,
        'cv_f1': f1,
        'confusion_matrix': cm,
        'y_cv_pred': y_pred_cv
    }

def assign_risk_from_probs(proba, classes):
    """Map probabilities to risk labels Low/Medium/High:
    - if class names are known mapping used, else use highest-prob class and map by ordering.
    """
    # if classes are numeric or strings 'Low','Medium','High'
    idx = np.argmax(proba, axis=1)
    preds = [classes[i] for i in idx]
    # normalize to Low/Medium/High if possible
    def norm_label(lbl):
        s = str(lbl).lower()
        if 'low' in s: return 'Low'
        if 'med' in s or 'midd' in s: return 'Medium'
        if 'high' in s: return 'High'
        # fallback: map by class order
        return lbl
    return [norm_label(p) for p in preds]

def cluster_and_label(df_features, n_clusters=3, scaling=True, random_state=42):
    """KMeans clustering, then label clusters as Low/Medium/High based on cluster centers' pollution score.
    Pollution score computed as mean of features (assumes higher -> worse)."""
    X = df_features.copy()
    if scaling:
        st_local_scaler = StandardScaler()
        X = pd.DataFrame(st_local_scaler.fit_transform(X), columns=X.columns)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    # compute center 'pollution' score as mean of center values
    center_scores = centers.mean(axis=1)
    order = np.argsort(center_scores)  # low->high
    # map cluster index to risk label
    rank_to_label = {order[0]: 'Low', order[1]: 'Medium', order[-1]: 'High'}
    labeled = [rank_to_label[lbl] for lbl in labels]
    return labels, labeled, kmeans, center_scores

def df_to_excel_bytes(df_dict):
    """Given a dict of {sheet_name: df}, return bytes of an Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
        writer.save()
    return output.getvalue()

def generate_pdf_report(title, description, summary_text, metrics: dict, df_sample: pd.DataFrame = None):
    """Create a simple PDF report and return bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, description)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Model Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in metrics.items():
        pdf.cell(0, 6, f"{k}: {v}", ln=True)
    if df_sample is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 6, "Sample Predictions (first 20 rows)", ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", size=8)
        table = df_sample.head(20)
        # Print table header
        col_width = pdf.w / max(1, len(table.columns)) - 4
        for c in table.columns:
            pdf.cell(col_width, 6, str(c)[:15], border=1)
        pdf.ln()
        for _, row in table.iterrows():
            for item in row:
                txt = str(item)[:15]
                pdf.cell(col_width, 6, txt, border=1)
            pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Microplastic Risk Dashboard", layout="wide")
st.title("Microplastic Pollution Risk Analytics Platform")
st.markdown("A Streamlit dashboard for data upload, preprocessing, modeling, visualization, and report generation.")

# Top navigation
nav = st.selectbox("Navigation", ["Home", "Upload Dataset", "Data Analysis", "Modeling & Prediction", "Reports"])

# Initialize session state containers
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = {}
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None

# Sidebar controls
st.sidebar.header("Controls")
file_uploader = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])
if file_uploader is not None:
    df_in = load_data(file_uploader)
    if df_in is not None:
        st.session_state['df_raw'] = df_in
        st.success(f"Loaded {file_uploader.name} — {df_in.shape[0]} rows, {df_in.shape[1]} cols")

# Sidebar model/config options
st.sidebar.subheader("Preprocessing")
fill_method = st.sidebar.selectbox("Numeric imputation", ['mean', 'median'])
scale_option = st.sidebar.checkbox("Scale numeric features (StandardScaler)", value=True)
dropna_thresh = st.sidebar.slider("Drop columns with > missing %", 0.0, 0.9, 0.0, step=0.05)

st.sidebar.subheader("Modeling")
model_choice = st.sidebar.selectbox("Primary model", ["RandomForest", "XGBoost (if available)"])
if model_choice == "XGBoost (if available)" and not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost not installed — will fallback to RandomForest.")
cv_folds = st.sidebar.slider("K-Fold (CV)", 2, 10, 5)
run_button = st.sidebar.button("Run Training / Analysis")

# -------------------------
# Home page
# -------------------------
if nav == "Home":
    st.header("Welcome")
    st.markdown("""
    **Purpose:** This app lets you upload environmental measurements and generate risk classifications (Low/Medium/High) for microplastic pollution using supervised classification (Random Forest / XGBoost) or clustering if no labels are available.
    
    **Workflow:** Upload → Preprocess → Analyze → Train & Predict → Download Reports.
    """)
    st.markdown("**Quick tips:**")
    st.markdown("- Ensure your data contains numeric pollution indicators (e.g., pH, turbidity, microplastic_count) and optionally latitude/longitude columns for mapping.")
    st.markdown("- If you have historical labeled risk column (Low/Medium/High or 0/1/2), choose it as Target in Modeling page to do supervised training.")
    st.info("Data is stored only in your browser session (ephemeral) and not uploaded to external servers by this app.")

# -------------------------
# Upload Dataset page
# -------------------------
elif nav == "Upload Dataset":
    st.header("Upload Dataset")
    st.write("Upload CSV or Excel (.xls/.xlsx). The app will attempt to infer types.")
    uploaded = file_uploader  # from sidebar
    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw']
        st.subheader("Preview")
        st.dataframe(df.head(200))
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        if st.button("Show dtypes"):
            st.write(df.dtypes)
        if st.button("Download sample cleaned as Excel"):
            bytes_xl = df_to_excel_bytes({"raw": df})
            b64 = base64.b64encode(bytes_xl).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="dataset.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Use the file uploader in the sidebar to upload a dataset.")

# -------------------------
# Data Analysis page
# -------------------------
elif nav == "Data Analysis":
    st.header("Data Analysis & Visualization")
    df = st.session_state.get('df_raw')
    if df is None:
        st.warning("Upload a dataset first (sidebar).")
    else:
        st.subheader("Overview")
        st.write("First 10 rows:")
        st.dataframe(df.head(10))
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe(include='all').T)
        st.subheader("Missing values per column")
        miss = df.isna().sum().rename("missing_count").to_frame()
        miss['missing_pct'] = (miss['missing_count'] / len(df) * 100).round(2)
        st.dataframe(miss.sort_values('missing_pct', ascending=False))

        # Correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.subheader("Correlation heatmap (numeric features)")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature distributions")
            feature = st.selectbox("Select numeric feature", numeric_cols)
            fig2 = px.histogram(df, x=feature, nbins=30, marginal="box")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric columns detected for correlation / distributions.")

        # If lat/lon present, show map
        possible_lat = [c for c in df.columns if 'lat' in c.lower()]
        possible_lon = [c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()]
        if possible_lat and possible_lon:
            lat_col = possible_lat[0]
            lon_col = possible_lon[0]
            st.subheader("Geographic scatter")
            fig_map = px.scatter_geo(df, lat=lat_col, lon=lon_col, hover_name=df.columns[0],
                                     size_max=9)
            st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# Modeling & Prediction page
# -------------------------
elif nav == "Modeling & Prediction":
    st.header("Modeling & Prediction")
    df = st.session_state.get('df_raw')
    if df is None:
        st.warning("Upload dataset first (sidebar).")
    else:
        st.subheader("Select features and target")
        cols = df.columns.tolist()
        with st.form("feature_form"):
            lat_col = st.selectbox("Latitude column (optional)", options=[None] + cols, index=0)
            lon_col = st.selectbox("Longitude column (optional)", options=[None] + cols, index=0)
            date_col = st.selectbox("Temporal/date column (optional)", options=[None] + cols, index=0)
            target_col = st.selectbox("Target column (if labeled risk present)", options=[None] + cols, index=0)
            feature_cols = st.multiselect("Select features for modeling (numeric preferred)", options=cols, default=[c for c in cols if c not in (lat_col, lon_col, date_col, target_col)][:6])
            submitted = st.form_submit_button("Prepare Data")
        if submitted:
            if not feature_cols:
                st.error("Select at least one feature.")
            else:
                # Preprocess
                numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
                categorical_features = [c for c in feature_cols if c not in numeric_features]
                df_proc, encoders, scaler, info = preprocess(df[feature_cols + ([target_col] if target_col else [])],
                                                              numeric_cols=numeric_features,
                                                              categorical_cols=categorical_features,
                                                              fill_method=fill_method,
                                                              scale=scale_option,
                                                              dropna_thresh=dropna_thresh)
                st.session_state['df_processed'] = df_proc
                st.success("Preprocessing complete.")
                st.write("Processed features preview:")
                st.dataframe(df_proc.head(10))

                # If target present -> supervised
                if target_col:
                    st.subheader("Supervised classification workflow")
                    # ensure target in df_proc
                    if target_col not in df_proc.columns:
                        st.error("Target column not in processed dataframe.")
                    else:
                        # Encode target if necessary
                        y = df_proc[target_col]
                        if y.dtype == object or not pd.api.types.is_integer_dtype(y):
                            le_t = LabelEncoder()
                            y_enc = le_t.fit_transform(y.astype(str))
                            classes = list(le_t.classes_)
                        else:
                            # numeric labels
                            y_enc = y.values
                            classes = sorted(list(pd.Series(y_enc).unique()))
                        X = df_proc.drop(columns=[target_col])
                        st.write("Class distribution:")
                        st.dataframe(pd.Series(y_enc).value_counts().rename("count").to_frame())

                        # model selection
                        models_to_run = []
                        if model_choice == "RandomForest" or not XGBOOST_AVAILABLE:
                            models_to_run.append(("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)))
                        if model_choice.startswith("XGBoost") and XGBOOST_AVAILABLE:
                            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                            models_to_run.append(("XGBoost", clf))

                        all_results = {}
                        if st.button("Run K-Fold Training (this may take a moment)"):
                            for name, m in models_to_run:
                                st.info(f"Training {name} with {cv_folds}-fold CV")
                                res = kfold_cv_and_train(m, X, y_enc, cv=cv_folds)
                                # store
                                all_results[name] = res
                                st.success(f"{name} CV F1: {res['cv_f1']:.3f}, Acc: {res['cv_accuracy']:.3f}")
                                # confusion matrix
                                fig_cm = go.Figure(data=go.Heatmap(z=res['confusion_matrix'],
                                                                   x=[str(c) for c in classes],
                                                                   y=[str(c) for c in classes],
                                                                   colorscale='Viridis'))
                                fig_cm.update_layout(title=f"Confusion Matrix ({name})", xaxis_title="Predicted", yaxis_title="Actual")
                                st.plotly_chart(fig_cm, use_container_width=True)
                            # pick first model as trained_model in session
                            if all_results:
                                st.session_state['model_results'] = all_results
                                st.session_state['trained_model'] = list(all_results.values())[0]['model']
                                st.success("Models trained and saved in session.")
                        # Prediction / Map
                        if st.session_state.get('trained_model') is not None:
                            st.subheader("Predict & Map")
                            model = st.session_state['trained_model']
                            X_full = X.copy()
                            # predict probabilities if supported
                            try:
                                proba = model.predict_proba(X_full)
                            except Exception:
                                proba = None
                            y_pred = model.predict(X_full)
                            # Map labels
                            if proba is not None:
                                risk_labels = assign_risk_from_probs(proba, classes)
                            else:
                                # convert numeric preds to labels if possible
                                risk_labels = []
                                for p in y_pred:
                                    lab = str(p)
                                    if isinstance(p, (int, np.integer)):
                                        # map numeric to Low/Medium/High if 3 classes
                                        if len(classes) == 3:
                                            mp = {0:'Low',1:'Medium',2:'High'}
                                            risk_labels.append(mp.get(p, lab))
                                        else:
                                            risk_labels.append(lab)
                                    else:
                                        s = str(p).lower()
                                        if 'low' in s: risk_labels.append('Low')
                                        elif 'med' in s: risk_labels.append('Medium')
                                        elif 'high' in s: risk_labels.append('High')
                                        else: risk_labels.append(lab)
                            result_df = df.copy()
                            result_df['predicted_label'] = risk_labels
                            st.dataframe(result_df[[c for c in result_df.columns if c in (lat_col, lon_col) or c in feature_cols][:10]].head(10))
                            if lat_col and lon_col and lat_col in result_df.columns and lon_col in result_df.columns:
                                st.subheader("Risk Map")
                                fig_map = px.scatter_geo(result_df, lat=lat_col, lon=lon_col,
                                                         color='predicted_label',
                                                         hover_data=feature_cols + ['predicted_label'],
                                                         title="Predicted Risk by Location")
                                st.plotly_chart(fig_map, use_container_width=True)
                            # store for reporting
                            st.session_state['last_prediction_df'] = result_df

                else:
                    # Unsupervised clustering
                    st.subheader("Unsupervised clustering -> risk assignment")
                    n_clusters = st.slider("Number of clusters (3 recommended)", 2, 6, 3)
                    if st.button("Run Clustering"):
                        features_df = df_proc[feature_cols]
                        labels, labeled, kmeans_model, center_scores = cluster_and_label(features_df, n_clusters=n_clusters, scaling=scale_option)
                        out_df = df.copy()
                        out_df['cluster'] = labels
                        out_df['risk_label'] = labeled
                        st.dataframe(out_df.head(20))
                        st.session_state['last_prediction_df'] = out_df
                        st.success("Clustering complete.")
                        if lat_col and lon_col and lat_col in out_df.columns and lon_col in out_df.columns:
                            fig_map = px.scatter_geo(out_df, lat=lat_col, lon=lon_col, color='risk_label',
                                                     hover_data=feature_cols + ['cluster'], title="Cluster-based Risk Map")
                            st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# Reports page
# -------------------------
elif nav == "Reports":
    st.header("Generate & Download Reports")
    df_proc = st.session_state.get('df_processed')
    last_pred = st.session_state.get('last_prediction_df')
    models = st.session_state.get('model_results', {})
    if not models and last_pred is None:
        st.info("No model runs or predictions available. Visit Modeling & Prediction and run training/prediction first.")
    else:
        st.subheader("Model Scorecards")
        if models:
            for name, res in models.items():
                st.markdown(f"**{name}**")
                st.write({
                    "CV Accuracy": round(res['cv_accuracy'], 4),
                    "CV Precision": round(res['cv_precision'], 4),
                    "CV Recall": round(res['cv_recall'], 4),
                    "CV F1": round(res['cv_f1'], 4)
                })
        if last_pred is not None:
            st.subheader("Sample predictions")
            st.dataframe(last_pred.head(50))
            if st.button("Download predictions as Excel"):
                bytes_xl = df_to_excel_bytes({"predictions": last_pred})
                st.download_button("Download Excel", data=bytes_xl, file_name="predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            # generate PDF summary
            if st.button("Generate PDF report (summary)"):
                title = "Microplastic Pollution Risk Report"
                desc = "Auto-generated summary report from the Microplastic Risk Dashboard."
                summary = "This report summarizes predicted risk levels and model performance."
                # prepare metrics
                metrics = {}
                for name, res in models.items():
                    metrics[f"{name} CV F1"] = f"{res['cv_f1']:.3f}"
                    metrics[f"{name} CV Acc"] = f"{res['cv_accuracy']:.3f}"
                pdf_bytes = generate_pdf_report(title, desc, summary, metrics, df_sample=last_pred)
                st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

# End of app
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for microplastic pollution risk analytics. Data is ephemeral and processed locally in your browser session.")
st.sidebar.markdown("© Microplastic Risk Dashboard")
