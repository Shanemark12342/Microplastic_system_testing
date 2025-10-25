# app.py
"""
Streamlit app: Microplastic Pollution Risk Dashboard
Path: app.py

Requirements (pip):
streamlit pandas numpy scikit-learn matplotlib plotly pydeck fpdf openpyxl xgboost (optional)

Run:
streamlit run app.py
"""

from io import BytesIO
import tempfile
import base64
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Optional xgboost
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False

# For interactive maps
import pydeck as pdk

# PDF generation
from fpdf import FPDF

# ---------- Helpers ----------

def set_page_config():
    st.set_page_config(page_title="Microplastic Risk Dashboard",
                       layout="wide",
                       initial_sidebar_state="expanded")

def theme_css():
    st.markdown(
        """
        <style>
        /* Minimal blue/green theme accents */
        .stApp header {background: linear-gradient(90deg, #e6f4ff, #e8f7ea);}
        .stButton>button {border-radius:10px;}
        .reportview-container {background: linear-gradient(180deg,#ffffff,#f7fffb)}
        </style>
        """, unsafe_allow_html=True
    )

def load_data_uploader():
    uploaded = st.file_uploader("Upload CSV / Excel dataset", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state['df'] = df
            st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error("Error reading file: " + str(e))

def generate_sample_dataset(n=200, seed=42):
    np.random.seed(seed)
    lats = np.random.uniform(-5.0, 5.0, n) + 10.0  # sample latitude band
    lons = np.random.uniform(100.0, 110.0, n)
    pH = np.random.normal(7.8, 0.4, n)
    turbidity = np.abs(np.random.normal(5, 3, n))
    population = np.random.poisson(2000, n)
    mp_conc = (0.2 * population/2000) + (0.5 * (turbidity/10.0)) + np.random.normal(0, 0.1, n)
    season = np.random.choice(["Dry", "Wet"], size=n)
    df = pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "pH": pH.round(2),
        "turbidity": turbidity.round(2),
        "population_density": population,
        "mp_conc": (mp_conc * 100).round(3),  # ppm-like units
        "season": season,
        "site_id": [f"SITE_{i+1}" for i in range(n)]
    })
    return df

def preview_data(df, n=5):
    st.dataframe(df.head(n))

def detect_latlon_columns(df):
    candidates = df.columns.str.lower()
    lat_col = None
    lon_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ('lat', 'latitude'):
            lat_col = c
        if cl in ('lon', 'long', 'longitude'):
            lon_col = c
    # fallback: numeric columns that look like coords by range
    if not lat_col:
        for c in df.select_dtypes(include='number').columns:
            if df[c].between(-90, 90).all() and df[c].mean() != 0:
                lat_col = c
                break
    if not lon_col:
        for c in df.select_dtypes(include='number').columns:
            if df[c].between(-180, 180).all() and df[c].mean() != 0 and c != lat_col:
                lon_col = c
                break
    return lat_col, lon_col

def auto_preprocess(df, numeric_impute='median'):
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    # impute
    for c in num_cols:
        if df[c].isna().any():
            if numeric_impute == 'median':
                df[c].fillna(df[c].median(), inplace=True)
            else:
                df[c].fillna(df[c].mean(), inplace=True)
    for c in cat_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "NA", inplace=True)
    return df, num_cols, cat_cols

def make_risk_label(df, mp_col=None):
    # If risk column exists, return as-is
    for possible in ['risk','risk_level','pollution_risk']:
        if possible in df.columns:
            return df[ possible ].astype(str)
    # Try to find common mp concentration column
    if not mp_col:
        for cand in ['mp_conc','microplastic_ppm','microplastic','concentration','mp_concentration']:
            if cand in df.columns:
                mp_col = cand
                break
    if not mp_col:
        # fallback: pick numeric column with name containing 'conc' or 'mp'
        for c in df.select_dtypes(include=['number']).columns:
            if 'conc' in c.lower() or 'mp' in c.lower():
                mp_col = c
                break
    if not mp_col:
        # Unable to auto-generate risk
        return None, mp_col
    # make quantile bins
    quantiles = df[mp_col].quantile([0.33, 0.66]).values
    def label(v):
        if v <= quantiles[0]:
            return "Low"
        elif v <= quantiles[1]:
            return "Medium"
        else:
            return "High"
    return df[mp_col].apply(label), mp_col

def build_model_pipeline(num_cols, cat_cols, clf_choice='rf', rf_params=None):
    # numeric and categorical transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ], remainder='drop'
    )
    if clf_choice == 'xgb' and _XGBOOST_AVAILABLE:
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, **(rf_params or {}))
    else:
        clf = RandomForestClassifier(n_estimators=rf_params.get('n_estimators',100) if rf_params else 100,
                                     max_depth=rf_params.get('max_depth', None) if rf_params else None,
                                     random_state=42)
    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])
    return pipe

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_corr_heatmap(df, numeric_cols):
    fig, ax = plt.subplots(figsize=(6,5))
    corr = df[numeric_cols].corr()
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    return fig

def create_excel_report(df, predictions_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='data', index=False)
        predictions_df.to_excel(writer, sheet_name='predictions', index=False)
    processed_data = output.getvalue()
    return processed_data

def create_pdf_report(summary_text, top_high_risk_df, image_paths=[]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Microplastic Pollution - Analysis Report", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(6)
    if not top_high_risk_df.empty:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Top High-Risk Sites", ln=True)
        pdf.set_font("Arial", size=10)
        # Simple table: site_id, latitude, longitude, predicted_risk, confidence
        top_high_risk_df = top_high_risk_df.head(10)
        col_names = top_high_risk_df.columns.tolist()
        for idx, row in top_high_risk_df.iterrows():
            line = " | ".join([f"{c}: {row[c]}" for c in col_names])
            pdf.multi_cell(0, 6, line)
    # Append images
    for p in image_paths:
        try:
            pdf.add_page()
            pdf.image(p, w=180)
        except Exception:
            continue
    out = pdf.output(dest='S').encode('latin-1')
    return out

def save_plot_tmp(fig, name='plot.png'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return tmp.name

# ---------- Streamlit App ----------

def main():
    set_page_config()
    theme_css()
    st.title("🌊 Microplastic Pollution Risk — Predictive Dashboard")
    # Top nav simulation
    tab = st.selectbox("", ["Home", "Upload Dataset", "Data Analysis", "Prediction Dashboard", "Reports"], index=0, key="topnav")

    # Sidebar controls
    st.sidebar.header("Controls")
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'preprocessed' not in st.session_state:
        st.session_state['preprocessed'] = {'df': None, 'num_cols': [], 'cat_cols': []}
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None

    if tab == "Home":
        st.markdown("""
        **Overview:** This web app performs predictive analytics to classify microplastic pollution risk as **Low**, **Medium**, or **High** using Random Forest or XGBoost.
        - Upload environmental datasets (CSV/Excel)
        - Auto-clean, transform, and feature-engineer
        - Train models, evaluate, visualize, and download reports
        """)
        st.info("Tip: Upload a CSV with columns like latitude, longitude, mp_conc (microplastic concentration), pH, turbidity, population_density, date.")
        if st.button("Load sample dataset"):
            st.session_state['df'] = generate_sample_dataset()
            st.success("Sample dataset generated.")
            preview_data(st.session_state['df'])
    elif tab == "Upload Dataset":
        st.markdown("### Upload or generate data")
        load_data_uploader()
        st.write("Or generate a sample dataset for testing:")
        if st.button("Generate sample dataset"):
            st.session_state['df'] = generate_sample_dataset()
            st.success("Sample dataset generated.")
        if st.session_state['df'] is not None:
            st.write("Preview:")
            preview_data(st.session_state['df'], n=8)

    elif tab == "Data Analysis":
        df = st.session_state['df']
        if df is None:
            st.warning("No dataset loaded. Go to Upload Dataset tab.")
            return
        st.header("Preprocessing")
        numeric_impute = st.selectbox("Numeric imputation", options=['median','mean'], index=0)
        if st.button("Run preprocessing"):
            df_clean, num_cols, cat_cols = auto_preprocess(df, numeric_impute=numeric_impute)
            st.session_state['preprocessed'] = {'df': df_clean, 'num_cols': num_cols, 'cat_cols': cat_cols}
            st.success(f"Preprocessing complete — {len(num_cols)} numeric cols, {len(cat_cols)} categorical cols.")
        if st.session_state['preprocessed']['df'] is not None:
            df_clean = st.session_state['preprocessed']['df']
            st.subheader("Descriptive statistics")
            st.dataframe(df_clean.describe(include='all').T)
            st.subheader("Correlation heatmap (numeric)")
            if st.session_state['preprocessed']['num_cols']:
                fig = plot_corr_heatmap(df_clean, st.session_state['preprocessed']['num_cols'])
                _ = save_plot_tmp(fig, 'corr.png')
            else:
                st.info("No numeric columns detected for correlation.")
            st.subheader("Distribution charts")
            cols = st.multiselect("Select numeric features to plot distributions", st.session_state['preprocessed']['num_cols'], max_selections=3)
            for c in cols:
                fig, ax = plt.subplots()
                ax.hist(df_clean[c].dropna(), bins=30)
                ax.set_title(c)
                st.pyplot(fig)

    elif tab == "Prediction Dashboard":
        df = st.session_state['df']
        if df is None:
            st.warning("No dataset loaded. Go to Upload Dataset tab.")
            return
        # Ensure preprocessing
        if st.session_state['preprocessed']['df'] is None:
            st.warning("Please run preprocessing on Data Analysis tab first, or click below to auto-preprocess.")
            if st.button("Auto-preprocess now"):
                df_clean, num_cols, cat_cols = auto_preprocess(df)
                st.session_state['preprocessed'] = {'df': df_clean, 'num_cols': num_cols, 'cat_cols': cat_cols}
                st.success("Preprocessing done.")
            else:
                return
        df_clean = st.session_state['preprocessed']['df']
        st.subheader("Select columns for modeling")
        suggested_lat, suggested_lon = detect_latlon_columns(df_clean)
        lat_col = st.selectbox("Latitude column", options=[None] + list(df_clean.columns), index=(1 if suggested_lat else 0))
        lon_col = st.selectbox("Longitude column", options=[None] + list(df_clean.columns), index=(1 if suggested_lon else 0))
        date_col = st.selectbox("Date/time column (optional)", options=[None] + list(df_clean.columns))
        # choose features
        default_feats = [c for c in df_clean.columns if c not in (lat_col, lon_col, date_col, 'site_id')]
        features = st.multiselect("Features to use for modeling", default_feats, default=default_feats[:6] if len(default_feats)>0 else default_feats)
        target_col = st.selectbox("Existing target column (risk) if any", options=[None] + list(df_clean.columns))
        # Risk auto-generation
        if target_col is None:
            make_label = st.button("Auto-generate risk labels from microplastic concentration")
            if make_label:
                labels, used_mp_col = make_risk_label(df_clean)
                if labels is None:
                    st.error("Could not auto-generate risk. Please provide a target column or ensure there is an mp_conc-like column.")
                    return
                df_clean = df_clean.copy()
                df_clean['risk'] = labels
                target_col = 'risk'
                st.session_state['preprocessed']['df'] = df_clean
                st.success(f"Risk labels created using `{used_mp_col}`.")
        if target_col is None:
            st.info("Provide or create a target (risk) column to train model.")
            return

        # Model options
        st.subheader("Model & training")
        model_choice = st.selectbox("Model", options=['RandomForest'] + (['XGBoost'] if _XGBOOST_AVAILABLE else []))
        n_estimators = st.number_input("n_estimators", value=100, min_value=10, max_value=2000, step=10)
        max_depth = st.number_input("max_depth (None=0)", value=0, min_value=0, max_value=100, step=1)
        test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.5, value=0.2)
        do_train = st.button("Train model")
        if do_train:
            if len(features) == 0:
                st.error("Select at least one feature.")
            else:
                X = df_clean[features].copy()
                y = df_clean[target_col].astype(str).copy()
                # ensure no nan in y
                if y.isna().any():
                    st.error("Target column contains missing values. Clean target first.")
                else:
                    # split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                    # determine numeric/categorical within features
                    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
                    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
                    rf_params = {'n_estimators': n_estimators}
                    if max_depth > 0:
                        rf_params['max_depth'] = int(max_depth)
                    clf_choice = 'xgb' if (model_choice == 'XGBoost' and _XGBOOST_AVAILABLE) else 'rf'
                    pipeline = build_model_pipeline(num_cols, cat_cols, clf_choice=clf_choice, rf_params=rf_params)
                    with st.spinner("Training model..."):
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        metrics = evaluate_model(y_test, y_pred)
                        st.success("Training done.")
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        st.write("Precision / Recall / F1 (weighted):",
                                 f"{metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}")
                        st.write("Classification report:")
                        st.text(classification_report(y_test, y_pred, zero_division=0))
                        st.write("Confusion matrix:")
                        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
                        st.write(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))
                        # Save model and metadata
                        st.session_state['model'] = pipeline
                        st.session_state['model_meta'] = {'features': features, 'num_cols': num_cols, 'cat_cols': cat_cols, 'target': target_col}
        # Prediction on full dataset
        if st.session_state['model'] is not None:
            st.subheader("Predictions on dataset")
            model = st.session_state['model']
            meta = st.session_state['model_meta']
            X_all = df_clean[meta['features']].copy()
            preds = model.predict(X_all)
            try:
                probs = model.predict_proba(X_all).max(axis=1)
            except Exception:
                probs = np.array([math.nan]*len(preds))
            pred_df = df_clean.copy()
            pred_df['predicted_risk'] = preds
            pred_df['pred_confidence'] = probs
            st.session_state['predictions'] = pred_df
            st.dataframe(pred_df[[c for c in ['site_id','latitude','longitude','predicted_risk','pred_confidence'] if c in pred_df.columns]].head(30))

            # Map visualization
            if lat_col and lon_col and lat_col in pred_df.columns and lon_col in pred_df.columns:
                st.subheader("Risk map")
                color_by = {'Low': [34,139,34], 'Medium': [255,165,0], 'High': [255,0,0]}
                pred_df['color'] = pred_df['predicted_risk'].map(color_by).apply(lambda x: [int(i) for i in x] if isinstance(x, (list,tuple)) else [0,0,0])
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=pred_df,
                    get_position=[lon_col, lat_col],
                    get_fill_color="color",
                    get_radius=500,
                    pickable=True,
                    auto_highlight=True
                )
                view = pdk.ViewState(latitude=float(pred_df[lat_col].mean()), longitude=float(pred_df[lon_col].mean()), zoom=6, pitch=0)
                r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Site: {site_id}\nRisk: {predicted_risk}\nConf: {pred_confidence}"})
                st.pydeck_chart(r)
            else:
                st.info("Latitude/Longitude columns not selected or not found — map unavailable.")

    elif tab == "Reports":
        st.header("Generate & Download Reports")
        if st.session_state['predictions'] is None:
            st.warning("No predictions available. Train a model on Prediction Dashboard first.")
            return
        pred_df = st.session_state['predictions']
        top_high_risk = pred_df[pred_df['predicted_risk']=='High'].sort_values('pred_confidence', ascending=False)
        st.subheader("High-risk sites")
        st.dataframe(top_high_risk.head(20))
        if st.button("Download Excel report"):
            excel_bytes = create_excel_report(st.session_state['preprocessed']['df'], pred_df)
            st.download_button("Download .xlsx", excel_bytes, file_name="microplastic_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("Download PDF report"):
            # create small summary and save a plot image
            summary = ("This report summarizes predicted microplastic pollution risk across sampled sites.\n"
                       "Model: " + ("XGBoost" if (_XGBOOST_AVAILABLE and isinstance(st.session_state['model'].named_steps['clf'], xgb.XGBClassifier)) else "RandomForest") + ".\n")
            # create a plot of counts
            fig, ax = plt.subplots()
            pred_df['predicted_risk'].value_counts().reindex(['Low','Medium','High']).plot(kind='bar', ax=ax)
            ax.set_title("Predicted Risk Counts")
            imgpath = save_plot_tmp(fig, 'risk_counts.png')
            pdf_bytes = create_pdf_report(summary, top_high_risk, image_paths=[imgpath])
            st.download_button("Download PDF", pdf_bytes, file_name="microplastic_report.pdf", mime="application/pdf")
            os.unlink(imgpath)

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Models: RandomForest / XGBoost • All data processed in-session only.")

if __name__ == "__main__":
    main()
