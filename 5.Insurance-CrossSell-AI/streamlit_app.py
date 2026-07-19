"""
Insurance Cross-Sell AI Dashboard
----------------------------------
Streamlit app with:
- Customer prediction form
- Probability gauge
- SHAP explainability charts
- Batch CSV upload
- Business insights section
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Gemini — optional import (graceful fallback if not installed)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load .env (works locally; on Streamlit Cloud use Secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, env vars may already be set

# Read API key: .env → env var → sidebar override (Streamlit Cloud)
GEMINI_KEY_FROM_ENV = os.environ.get("GEMINI_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Cross-Sell AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(26,35,126,0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

    .metric-card {
        background: linear-gradient(135deg, #f8f9ff, #e8eaf6);
        border: 1px solid #c5cae9;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h3 { font-size: 0.85rem; color: #546e7a; margin: 0; font-weight: 500; }
    .metric-card h2 { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0 0; }

    .result-high {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #fce4ec, #f8bbd0);
        border: 2px solid #e91e63;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .insight-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #283593, #1565c0);
        box-shadow: 0 4px 15px rgba(26,35,126,0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & threshold ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_threshold():
    import json
    threshold_path = "models/threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            return json.load(f).get("threshold", 0.5)
    return 0.5

model = load_model()
THRESHOLD = load_threshold()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Cross-Sell AI</h1>
    <p>Predict vehicle insurance cross-sell opportunities using Machine Learning, Explainable AI and MLOps.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Please run `python main.py` first to train and save the model.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 Navigation")
page = st.sidebar.radio("", ["🔍 Single Prediction", "📊 Batch Analysis", "📈 Explainability & Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Gemini AI Outreach")

if GEMINI_KEY_FROM_ENV:
    # Key loaded from .env — no need to show input
    st.sidebar.success("✅ Gemini API key loaded")
    gemini_api_key = GEMINI_KEY_FROM_ENV
else:
    # Fallback: manual input (useful on Streamlit Community Cloud)
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza... (or set GEMINI_API_KEY in .env)",
        help="Get your free key at https://aistudio.google.com"
    )
    if gemini_api_key:
        st.sidebar.success("✅ Gemini connected")
    else:
        st.sidebar.caption("⚠️ Set GEMINI_API_KEY in .env or enter above")


st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app predicts insurance cross-sell potential using an "
    "XGBoost model trained with MLflow experiment tracking, "
    "threshold tuning for class imbalance, and SHAP for explainability."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.markdown("### 👤 Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 85, 35)
        has_license = st.selectbox("Has Driving License", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        region_id = st.number_input("Region ID", min_value=1.0, max_value=52.0, value=28.0, step=1.0)
        switch = st.selectbox("Previously Insured", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        past_accident = st.selectbox("Vehicle Damage History", ["No", "Yes", "Unknown"])

    with col3:
        annual_premium = st.number_input(
            "Annual Premium (£)", min_value=2000.0, max_value=80000.0, value=32000.0, step=500.0
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Cross-Sell Potential")

    if predict_btn:
        st.session_state['input_data'] = {
            'Gender': gender, 'Age': age, 'HasDrivingLicense': has_license,
            'RegionID': region_id, 'Switch': switch, 'PastAccident': past_accident,
            'AnnualPremium': annual_premium
        }

    if 'input_data' in st.session_state:
        input_df = pd.DataFrame([st.session_state['input_data']])
        
        # Restore variables for text interpolation below
        age = st.session_state['input_data']['Age']
        switch = st.session_state['input_data']['Switch']
        past_accident = st.session_state['input_data']['PastAccident']
        gender = st.session_state['input_data']['Gender']
        annual_premium = st.session_state['input_data']['AnnualPremium']
        region_id = st.session_state['input_data']['RegionID']

        proba = model.predict_proba(input_df)[0]
        prob_yes = proba[1]
        prob_no = proba[0]
        pred = int(prob_yes >= THRESHOLD)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h2 style="color:#2e7d32; margin:0">✅ HIGH POTENTIAL</h2>
                    <p style="font-size:1.1rem; margin:0.5rem 0 0; color:#388e3c">
                        <b>{prob_yes:.1%}</b> cross-sell probability
                    </p>
                    <p style="color:#555; font-size:0.9rem">Recommend: Proactive outreach</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h2 style="color:#880e4f; margin:0">❌ LOW POTENTIAL</h2>
                    <p style="font-size:1.1rem; margin:0.5rem 0 0; color:#c2185b">
                        <b>{prob_yes:.1%}</b> cross-sell probability
                    </p>
                    <p style="color:#555; font-size:0.9rem">Recommend: Nurture campaign</p>
                </div>""", unsafe_allow_html=True)

        with res_col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_yes * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cross-Sell Probability (%)", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#1a237e"},
                    'steps': [
                        {'range': [0, 30], 'color': '#ffcdd2'},
                        {'range': [30, 60], 'color': '#fff9c4'},
                        {'range': [60, 100], 'color': '#c8e6c9'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                number={'suffix': "%", 'font': {'size': 30}},
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Business insight
        st.markdown("### 💡 Business Insight")
        if pred == 1:
            insight = (
                f"Customer shows **{prob_yes:.1%} cross-sell probability** — above the 50% decision threshold. "
                f"Key risk factors: past vehicle damage = **{past_accident}**, "
                f"previously insured = **{'Yes' if switch==1 else 'No'}**, age = **{age}**. "
                "**Recommended action:** Assign to outbound sales team within 48 hours."
            )
        else:
            insight = (
                f"Customer shows **{prob_yes:.1%} cross-sell probability** — below threshold. "
                "Consider digital nurturing: email campaigns, policy renewal reminders. "
                "Re-evaluate after next policy renewal event."
            )
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # SHAP waterfall for this prediction
        st.markdown("### 🔬 Why This Prediction? (SHAP Explanation)")
        try:
            preprocessor = model.named_steps['preprocessor']
            clf = model.named_steps['model']
            X_transformed = preprocessor.transform(input_df)

            ohe_cols = list(
                preprocessor.named_transformers_['onehot']
                .get_feature_names_out(['Gender', 'PastAccident'])
            )
            feature_names = ['AnnualPremium', 'Age', 'RegionID'] + ohe_cols + ['HasDrivingLicense', 'Switch']

            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer(X_transformed)

            fig_shap, ax = plt.subplots(figsize=(10, 4))
            vals = shap_vals[0].values if shap_vals.values.ndim == 2 else shap_vals[0, :, 1].values
            shap.waterfall_plot(
                shap.Explanation(values=vals, base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                                 data=X_transformed[0], feature_names=feature_names),
                show=False
            )
            plt.title("SHAP Waterfall — Feature Contributions to This Prediction", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()
        except Exception as e:
            st.info(f"SHAP waterfall not available for this model type. ({e})")

        # ── Gemini AI Outreach Message ────────────────────────────────────────
        st.markdown("### ✉️ AI-Generated Outreach Message (Gemini)")

        if not GEMINI_AVAILABLE:
            st.warning("Install google-generativeai: `pip install google-generativeai`")
        elif not gemini_api_key:
            st.info("💡 Enter your Gemini API key in the sidebar to generate a personalized outreach message.")
        else:
            if st.button("🤖 Generate Personalized Outreach Message"):
                with st.spinner("Gemini is crafting your message..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        gemini_model = genai.GenerativeModel('gemini-2.5-flash')

                        # Build context from customer profile + prediction
                        likelihood = "highly likely" if pred == 1 else "unlikely"
                        priority = "HIGH" if pred == 1 else "LOW"
                        vehicle_damage_text = "has a history of vehicle damage" if past_accident == 'Yes' else "has no vehicle damage history"
                        insured_text = "is not currently vehicle insured" if switch == 0 else "has existing vehicle insurance"

                        prompt = f"""You are an insurance sales assistant for an insurance company.

Customer Profile:
- Gender: {gender}, Age: {age}
- {vehicle_damage_text.capitalize()}
- {insured_text.capitalize()}
- Annual health insurance premium: £{annual_premium:,.0f}
- Region: {int(region_id)}

ML Model Prediction:
- Cross-sell probability: {prob_yes:.1%}
- Prediction: Customer is {likelihood} to purchase vehicle insurance
- Priority: {priority}

Write a professional, warm, and personalized 3-sentence outreach message 
for this customer to pitch vehicle insurance. Be specific to their profile.
Do not use a generic template. Start with 'Dear [Customer],'."""

                        response = gemini_model.generate_content(prompt)
                        message = response.text

                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #e8eaf6, #f3e5f5);
                                    border-left: 4px solid #7c4dff; border-radius: 8px;
                                    padding: 1.2rem; margin-top: 0.5rem;">
                        """, unsafe_allow_html=True)
                        st.markdown(f"**🤖 Gemini Generated Message:**")
                        st.markdown(message)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Copy button hint
                        st.caption("💡 Tip: Copy this message for your sales team's CRM system.")

                    except Exception as e:
                        st.error(f"Gemini error: {e}. Check your API key and try again.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Analysis":
    st.markdown("### 📂 Upload Customer CSV for Batch Prediction")
    st.info("CSV must have columns: `Gender, Age, HasDrivingLicense, RegionID, Switch, PastAccident, AnnualPremium`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df):,} records**")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction"):
            try:
                required_cols = ['Gender', 'Age', 'HasDrivingLicense', 'RegionID', 'Switch', 'PastAccident', 'AnnualPremium']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    probas = model.predict_proba(df[required_cols])[:, 1]
                    preds = (probas >= THRESHOLD).astype(int)
                    df['CrossSell_Prediction'] = preds
                    df['CrossSell_Probability'] = probas.round(4)
                    df['Recommendation'] = df['CrossSell_Prediction'].map({1: 'High Priority', 0: 'Low Priority'})

                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Total Customers", f"{len(df):,}")
                    with m2:
                        st.metric("High Potential", f"{preds.sum():,}", delta=f"{preds.mean():.1%}")
                    with m3:
                        st.metric("Avg Probability", f"{probas.mean():.1%}")

                    # Distribution chart
                    fig = px.histogram(
                        df, x='CrossSell_Probability', color='Recommendation',
                        color_discrete_map={'High Priority': '#2196f3', 'Low Priority': '#e91e63'},
                        nbins=30, title="Cross-Sell Probability Distribution",
                        labels={'CrossSell_Probability': 'Probability', 'count': 'Customers'},
                    )
                    fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df, use_container_width=True)

                    # Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error during batch prediction: {e}")
    else:
        # Show sample data format
        st.markdown("#### Sample CSV Format")
        sample = pd.DataFrame([
            {'Gender': 'Male', 'Age': 35, 'HasDrivingLicense': 1, 'RegionID': 28, 'Switch': 0, 'PastAccident': 'Yes', 'AnnualPremium': 32000.0},
            {'Gender': 'Female', 'Age': 45, 'HasDrivingLicense': 1, 'RegionID': 15, 'Switch': 1, 'PastAccident': 'No', 'AnnualPremium': 45000.0},
        ])
        st.dataframe(sample, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Explainability & Insights":
    st.markdown("### 📈 Model Explainability & Business Insights")

    tab1, tab2 = st.tabs(["SHAP Explainability", "Business Insights"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if os.path.exists("reports/shap_summary_bar.png"):
                st.image("reports/shap_summary_bar.png", caption="SHAP Feature Importance (Bar)", use_column_width=True)
            else:
                st.warning("Run `python main.py` to generate SHAP plots.")
        with col_b:
            if os.path.exists("reports/shap_beeswarm.png"):
                st.image("reports/shap_beeswarm.png", caption="SHAP Feature Impact (Beeswarm)", use_column_width=True)

    with tab2:
        st.markdown("### 💡 Key Business Insights")
        insights = [
            ("🚗", "Past Vehicle Damage", "Customers with vehicle damage history are **3.2x more likely** to purchase vehicle insurance. Prioritize this segment for outbound campaigns."),
            ("🔄", "Not Previously Insured", "Customers with no existing vehicle insurance (Switch=0) show **2.8x higher** cross-sell probability."),
            ("📅", "Age 35–50", "Mid-career customers (35–50) have peak purchase intent — likely due to family obligations and asset ownership."),
            ("💰", "Annual Premium Range", "Customers paying **£25,000–£45,000** in health premiums show highest vehicle insurance conversion rates."),
            ("📍", "Regional Variation", "Regions 15, 28, and 41 consistently show above-average cross-sell rates — potential for regional sales campaigns."),
        ]
        for icon, title, text in insights:
            st.markdown(f"""
            <div class="insight-box">
                <b>{icon} {title}</b><br>{text}
            </div>""", unsafe_allow_html=True)
