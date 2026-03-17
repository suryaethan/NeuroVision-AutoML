"""
NeuroVision AutoML - Streamlit Dashboard
Interactive web UI for running the AutoML pipeline and exploring results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

from neurovision.engine import NeuroVisionEngine

# ---- Page Config ----
st.set_page_config(
    page_title="NeuroVision AutoML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---- Session State ----
if "results" not in st.session_state:
    st.session_state.results = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if "df" not in st.session_state:
    st.session_state.df = None

# ---- Header ----
st.markdown('<div class="main-header">🧠 NeuroVision AutoML</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#888;'>Drop your dataset — AI handles the rest</p>",
    unsafe_allow_html=True
)
st.divider()

# ---- Sidebar ----
with st.sidebar:
    st.header("🔧 Configuration")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload any CSV dataset"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

        target_col = st.selectbox(
            "🎯 Target Column",
            options=df.columns.tolist(),
            index=len(df.columns) - 1
        )

        problem_type = st.selectbox(
            "🤖 Problem Type",
            options=["Auto Detect", "classification", "regression"]
        )
        if problem_type == "Auto Detect":
            problem_type = None

        test_size = st.slider("📈 Test Split", 0.1, 0.4, 0.2, 0.05)
        detect_anomalies = st.checkbox("🔍 Detect Anomalies", value=True)
        run_shap = st.checkbox("💡 SHAP Explainability", value=True)

        run_btn = st.button("🚀 Run AutoML Pipeline", use_container_width=True)

        if run_btn:
            with st.spinner("🧠 Training all models..."):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    df.to_csv(f.name, index=False)
                    tmp_path = f.name

                engine = NeuroVisionEngine()
                results = engine.run(
                    data_path=tmp_path,
                    target_col=target_col,
                    problem_type=problem_type,
                    test_size=test_size,
                    detect_anomalies=detect_anomalies,
                    explain=run_shap,
                )
                os.unlink(tmp_path)
                st.session_state.results = results
                st.session_state.engine = engine
                st.success("🏆 Pipeline complete!")
    else:
        st.info("📂 Upload a CSV file to start")

# ---- Main Content ----
results = st.session_state.results
df = st.session_state.df

if results is None:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🔧 Smart Preprocessing
        - Auto null handling
        - Categorical encoding
        - Feature engineering
        - DateTime extraction
        """)
    with col2:
        st.markdown("""
        ### 🤖 7+ ML Models
        - XGBoost, LightGBM
        - Random Forest
        - SVM, KNN
        - Linear models
        """)
    with col3:
        st.markdown("""
        ### 💡 SHAP Explainability
        - Global feature importance
        - Per-prediction explanations
        - Model leaderboard
        - Anomaly detection
        """)
else:
    # Show results
    leaderboard = results["leaderboard"]
    problem_type = results["problem_type"]
    best_name = results["best_model_name"]

    st.success(f"🏆 Best Model: **{best_name}** | Problem: **{problem_type.upper()}**")

    # Metrics
    best = leaderboard[0]
    if problem_type == "classification":
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Accuracy", f"{best.get('accuracy', 0):.2%}")
        m2.metric("📊 F1 Score", f"{best.get('f1', 0):.4f}")
        m3.metric("📡 AUC-ROC", f"{best.get('roc_auc', 0):.4f}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 RMSE", f"{best.get('rmse', 0):.4f}")
        m2.metric("📊 MAE", f"{best.get('mae', 0):.4f}")
        m3.metric("⭐ R2 Score", f"{best.get('r2', 0):.4f}")

    st.divider()

    # Leaderboard
    st.subheader("📊 Model Leaderboard")
    lb_data = []
    for i, entry in enumerate(leaderboard, 1):
        row = {"Rank": i, "Model": entry["model_name"]}
        if problem_type == "classification":
            row["Accuracy"] = f"{entry.get('accuracy', 0):.4f}"
            row["F1 Score"] = f"{entry.get('f1', 0):.4f}"
            row["AUC-ROC"] = f"{entry.get('roc_auc', 0):.4f}"
        else:
            row["RMSE"] = f"{entry.get('rmse', 0):.4f}"
            row["MAE"] = f"{entry.get('mae', 0):.4f}"
            row["R2"] = f"{entry.get('r2', 0):.4f}"
        lb_data.append(row)
    st.dataframe(pd.DataFrame(lb_data), use_container_width=True)

    # SHAP plot
    if "shap_values" in results and results["shap_values"] is not None:
        st.divider()
        st.subheader("💡 SHAP Feature Importance")
        shap_data = results["shap_values"]
        feature_names = results["feature_names"]
        mean_abs = np.abs(shap_data["shap_values"]).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_abs
        }).sort_values("Importance", ascending=True).tail(15)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Mean |SHAP| Feature Importance (Top 15)",
            color="Importance",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data preview
    if df is not None:
        st.divider()
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 **Shape:**", df.shape)
        with col2:
            st.write("❗ **Missing Values:**", df.isnull().sum().sum())
