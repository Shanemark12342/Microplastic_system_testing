# tests/test_app.py
"""
Unit tests for Microplastic Risk Dashboard (Streamlit app)
Run with: pytest -v
"""

import pytest
import pandas as pd
import numpy as np
from app import (
    generate_sample_dataset, auto_preprocess, make_risk_label,
    build_model_pipeline, evaluate_model, create_excel_report, create_pdf_report
)


@pytest.fixture(scope="module")
def sample_df():
    """Generate reusable sample dataset"""
    return generate_sample_dataset(n=100, seed=123)


def test_generate_sample_dataset(sample_df):
    """Check basic shape and required columns"""
    assert not sample_df.empty
    assert "latitude" in sample_df.columns
    assert "longitude" in sample_df.columns
    assert "mp_conc" in sample_df.columns


def test_auto_preprocess_fills_missing(sample_df):
    """Ensure missing values are filled"""
    df = sample_df.copy()
    df.loc[0, "pH"] = np.nan
    df_clean, num_cols, cat_cols = auto_preprocess(df)
    assert df_clean["pH"].isna().sum() == 0
    assert isinstance(num_cols, list)
    assert isinstance(cat_cols, list)


def test_make_risk_label_creates_labels(sample_df):
    """Ensure risk labels are generated"""
    labels, col = make_risk_label(sample_df)
    assert labels.isin(["Low", "Medium", "High"]).all()
    assert col == "mp_conc"


def test_build_model_pipeline_and_fit(sample_df):
    """Train simple RandomForest model"""
    df, _, _ = auto_preprocess(sample_df)
    y, _ = make_risk_label(df)
    X = df[["pH", "turbidity", "population_density"]]
    pipeline = build_model_pipeline(["pH", "turbidity", "population_density"], [], clf_choice="rf")
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model_returns_metrics():
    """Ensure all metrics computed correctly"""
    y_true = ["Low", "Medium", "High", "High"]
    y_pred = ["Low", "Medium", "Low", "High"]
    metrics = evaluate_model(y_true, y_pred)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1"])
    assert 0 <= metrics["accuracy"] <= 1


def test_create_excel_and_pdf_report(sample_df):
    """Generate and verify downloadable report bytes"""
    pred_df = sample_df.copy()
    pred_df["predicted_risk"] = np.random.choice(["Low", "Medium", "High"], size=len(pred_df))
    excel_bytes = create_excel_report(sample_df, pred_df)
    assert isinstance(excel_bytes, bytes)
    pdf_bytes = create_pdf_report("Summary text", pred_df.head(), [])
    assert isinstance(pdf_bytes, bytes)
