# ── FraudShield AI — Streamlit Dashboard ─────────────────
# Author: Suman Das | PNB 11 yrs | MTech IAR Jadavpur University
# Fraud Detection Dashboard with Live Scoring + SHAP Explainability

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title = "FraudShield AI",
    page_icon  = "🛡️",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .fraud-badge {
        background: #e53935;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .legit-badge {
        background: #00bcd4;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Hardcoded Dataset Statistics ──────────────────────────
TOTAL_TX       = 283726
FRAUD_COUNT    = 473
LEGIT_COUNT    = 283253
FRAUD_RATE     = 0.1667
AMOUNT_AT_RISK = 33694.79
IMBALANCE      = 598

FRAUD_BY_HOUR = {
    0: 4,  1: 8,  2: 19, 3: 12, 4: 14,
    5: 6,  6: 3,  7: 5,  8: 7,  9: 4,
    10: 2, 11: 3, 12: 5, 13: 4, 14: 6,
    15: 5, 16: 4, 17: 7, 18: 8, 19: 6,
    20: 5, 21: 4, 22: 3, 23: 2
}

LEGIT_BY_HOUR = {
    0: 5100,  1: 3200,  2: 1300,  3: 820,
    4: 1340,  5: 2100,  6: 4200,  7: 8900,
    8: 14200, 9: 18400, 10: 20800, 11: 19600,
    12: 18200, 13: 17800, 14: 16900, 15: 15800,
    16: 15200, 17: 14800, 18: 13900, 19: 12400,
    20: 10800, 21: 8900,  22: 7200,  23: 6100
}

FRAUD_EXAMPLE = {
    'V1': -2.3122, 'V2': 1.9519,  'V3': -1.6098,
    'V4': 3.9979,  'V5': -0.5220, 'V6': -1.4265,
    'V7': -2.5374, 'V8': 1.3918,  'V9': -2.7700,
    'V10': -2.7722,'V11': 3.2020, 'V12': -2.8992,
    'V13': -0.5950,'V14': -4.2954,'V15': 0.3900,
    'V16': -1.1407,'V17': -2.8300,'V18': -0.0168,
    'V19': 0.4160, 'V20': 0.0600, 'V21': 0.1294,
    'V22': -0.2143,'V23': -0.0339,'V24': 0.0840,
    'V25': 0.1285, 'V26': -0.1885,'V27': 0.0677,
    'V28': -0.0426,'Amount': 1.0
}

LEGIT_EXAMPLE = {
    'V1': -1.3598, 'V2': -0.0728, 'V3': 2.5363,
    'V4': 1.3781,  'V5': -0.3383, 'V6': 0.4624,
    'V7': 0.2396,  'V8': 0.0987,  'V9': 0.3638,
    'V10': 0.0908, 'V11': -0.5516,'V12': -0.6178,
    'V13': -0.9913,'V14': -0.3111,'V15': 1.4681,
    'V16': -0.4704,'V17': 0.2076, 'V18': 0.0258,
    'V19': 0.4039, 'V20': 0.2514, 'V21': -0.0183,
    'V22': 0.2778, 'V23': -0.1105,'V24': 0.0669,
    'V25': 0.1285, 'V26': -0.1891,'V27': 0.1336,
    'V28': -0.0211,'Amount': 149.62
}

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model = joblib.load(
        'models/fraudshield_xgb_model.pkl')
    scaler    = joblib.load(
        'models/fraudshield_scaler.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, scaler, explainer

FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + \
               ['Log_Amount', 'Hour']

# ── Header ────────────────────────────────────────────────
st.markdown("# 🛡️ FraudShield AI")
st.markdown("### Bank Fraud Detection System "
            "with Explainable AI")
st.markdown("---")

try:
    xgb_model, scaler, explainer = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🔍 Live Fraud Scorer",
    "📈 Model Performance"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Transaction Overview")
    st.caption("Statistics pre-computed from "
               "283,726 credit card transactions")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{TOTAL_TX:,}")
    with col2:
        st.metric("Fraud Cases",
                  f"{FRAUD_COUNT:,}",
                  delta=f"{FRAUD_RATE:.3f}% fraud rate",
                  delta_color="inverse")
    with col3:
        st.metric("Amount at Risk",
                  f"${AMOUNT_AT_RISK:,.2f}")
    with col4:
        st.metric("Imbalance Ratio", f"{IMBALANCE}:1")

    st.markdown("---")

    col1, col2 = st.columns(2)
    hours = list(range(24))

    with col1:
        st.markdown("#### Fraud Rate by Hour of Day")
        fraud_rate_hour = [
            FRAUD_BY_HOUR.get(h, 0) /
            (FRAUD_BY_HOUR.get(h, 0) +
             LEGIT_BY_HOUR.get(h, 1)) * 100
            for h in hours
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = hours,
            y    = fraud_rate_hour,
            mode = 'lines+markers',
            fill = 'tozeroy',
            line = dict(color='#e53935', width=2),
            name = 'Fraud Rate %'
        ))
        fig.update_layout(
            xaxis_title   = "Hour of Day",
            yaxis_title   = "Fraud Rate (%)",
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Fraud Count by Hour of Day")
        fig = px.bar(
            x     = hours,
            y     = [FRAUD_BY_HOUR.get(h, 0)
                     for h in hours],
            color = [FRAUD_BY_HOUR.get(h, 0)
                     for h in hours],
            color_continuous_scale = 'Reds',
            labels = {'x': 'Hour', 'y': 'Fraud Count'}
        )
        fig.update_layout(
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 300,
            showlegend    = False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Transaction Amount Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Legit Mean Amount",   "$88.35")
        st.metric("Legit Median Amount", "$22.00")
    with col2:
        st.metric("Fraud Mean Amount",   "$123.87")
        st.metric("Fraud Median Amount", "$9.82")
    with col3:
        st.metric("Peak Fraud Hour",  "2AM (1.45% rate)")
        st.metric("Safest Hour",      "10AM (0.05% rate)")

    st.markdown("#### 🔍 Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⏰ **Peak Fraud Hours**\n\n"
                "2AM, 4AM, 3AM show 30x higher "
                "fraud rate than 10AM")
    with col2:
        st.info("💰 **Amount Pattern**\n\n"
                "Fraud median $9.82 vs legit $22.00 — "
                "threshold evasion behaviour")
    with col3:
        st.info("⚖️ **Class Imbalance**\n\n"
                "598:1 ratio — accuracy is meaningless. "
                "Model uses Precision, Recall, F1")

# ══════════════════════════════════════════════════════════
# TAB 2 — LIVE FRAUD SCORER
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Live Transaction Fraud Scorer")
    st.markdown("Score any transaction in real time "
                "with SHAP explainability.")

    if not models_loaded:
        st.error("Models not loaded. Cannot score.")
    else:
        st.markdown("#### Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value = 0.0,
                max_value = 50000.0,
                value     = 150.0,
                step      = 0.01
            )
            hour = st.slider(
                "Hour of Day (0-23)",
                min_value = 0,
                max_value = 23,
                value     = 14
            )

        with col2:
            st.markdown("**Key PCA Features**")
            v14 = st.number_input(
                "V14 (strongest fraud signal)",
                value=0.0, step=0.01, format="%.4f")
            v4  = st.number_input(
                "V4", value=0.0,
                step=0.01, format="%.4f")
            v12 = st.number_input(
                "V12", value=0.0,
                step=0.01, format="%.4f")
            v10 = st.number_input(
                "V10", value=0.0,
                step=0.01, format="%.4f")
            v3  = st.number_input(
                "V3", value=0.0,
                step=0.01, format="%.4f")

        st.markdown("#### Quick Test")
        col1, col2 = st.columns(2)
        with col1:
            use_fraud = st.button(
                "🔴 Load Real Fraud Example",
                use_container_width=True)
        with col2:
            use_legit = st.button(
                "🟢 Load Legitimate Example",
                use_container_width=True)

        score_btn = st.button(
            "🔍 SCORE TRANSACTION",
            type                = "primary",
            use_container_width = True
        )

        if score_btn or use_fraud or use_legit:
            features = np.zeros(30)

            if use_fraud:
                example = FRAUD_EXAMPLE
                amount  = example['Amount']
                for i in range(1, 29):
                    features[i-1] = example[f'V{i}']
                features[28] = np.log1p(amount)
                features[29] = 2

            elif use_legit:
                example = LEGIT_EXAMPLE
                amount  = example['Amount']
                for i in range(1, 29):
                    features[i-1] = example[f'V{i}']
                features[28] = np.log1p(amount)
                features[29] = 10

            else:
                features[13] = v14
                features[3]  = v4
                features[11] = v12
                features[9]  = v10
                features[2]  = v3
                features[28] = np.log1p(amount)
                features[29] = hour

            features_df = pd.DataFrame(
                [features], columns=FEATURE_COLS)
            scaled      = scaler.transform(features_df)
            scaled_df   = pd.DataFrame(
                scaled, columns=FEATURE_COLS)

            prob     = float(
                xgb_model.predict_proba(scaled)[0][1])
            decision = "FRAUD" if prob >= 0.5 \
                       else "LEGITIMATE"
            confidence = (
                "HIGH"   if prob > 0.8 or prob < 0.2
                else "MEDIUM" if prob > 0.6 or prob < 0.4
                else "LOW"
            )

            st.markdown("---")
            st.markdown("#### 🎯 Scoring Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                if decision == "FRAUD":
                    st.error("🚨 **FRAUD DETECTED**")
                else:
                    st.success("✅ **LEGITIMATE**")
            with col2:
                st.metric("Fraud Probability",
                          f"{prob:.1%}")
            with col3:
                st.metric("Confidence", confidence)

            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {'text': "Fraud Probability (%)"},
                gauge = {
                    'axis'  : {'range': [0, 100]},
                    'bar'   : {'color': '#e53935'
                               if prob > 0.5
                               else '#00bcd4'},
                    'steps' : [
                        {'range': [0, 30],
                         'color': '#1a3a1a'},
                        {'range': [30, 70],
                         'color': '#3a3a1a'},
                        {'range': [70, 100],
                         'color': '#3a1a1a'}
                    ],
                    'threshold': {
                        'line' : {'color': 'white',
                                  'width': 2},
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                height        = 250,
                paper_bgcolor = '#0a0c10',
                font_color    = 'white'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🧠 Why This Decision?")
            shap_vals = explainer.shap_values(scaled_df)[0]

            top_idx  = np.argsort(
                np.abs(shap_vals))[::-1][:8]
            shap_df  = pd.DataFrame({
                'Feature'    : [FEATURE_COLS[i]
                                for i in top_idx],
                'SHAP Value' : [shap_vals[i]
                                for i in top_idx],
                'Direction'  : ['🔴 Toward Fraud'
                                if shap_vals[i] > 0
                                else '🔵 Toward Legit'
                                for i in top_idx]
            })

            colors = ['#e53935' if v > 0 else '#00bcd4'
                      for v in shap_df['SHAP Value']]

            fig = go.Figure(go.Bar(
                x            = shap_df['SHAP Value'],
                y            = shap_df['Feature'],
                orientation  = 'h',
                marker_color = colors,
                text         = shap_df['Direction'],
                textposition = 'outside'
            ))
            fig.update_layout(
                title         = "Top Feature Contributions",
                xaxis_title   = "SHAP Value",
                plot_bgcolor  = '#0a0c10',
                paper_bgcolor = '#0a0c10',
                font_color    = 'white',
                height        = 350
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(shap_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Model Performance")

    st.markdown("#### Complete Model Comparison")
    results_data = {
        'Model'     : ['Logistic Regression',
                       'Random Forest',
                       'XGBoost Base ⭐',
                       'XGBoost Tuned V1',
                       'XGBoost Tuned V2',
                       'Autoencoder V1',
                       'Autoencoder V2'],
        'Precision' : [0.0521, 0.5938, 0.9494,
                       0.9351, 0.9250, 0.1373, 0.1338],
        'Recall'    : [0.8737, 0.8000, 0.7895,
                       0.7579, 0.7789, 0.4105, 0.4000],
        'F1'        : [0.0983, 0.6816, 0.8621,
                       0.8372, 0.8457, 0.2058, 0.2005],
        'AUC-ROC'   : [0.9597, 0.9825, 0.9733,
                       0.9787, 0.9738, 0.9289, 0.9335],
        'Caught'    : [83, 76, 75, 72, 74, 39, 38],
        'FP'        : [1510, 52, 4, 5, 6, 245, 246]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Precision Comparison")
        fig = px.bar(
            results_df,
            x     = 'Model',
            y     = 'Precision',
            color = 'Precision',
            color_continuous_scale = 'Reds',
            text  = results_df['Precision'].apply(
                lambda x: f"{x:.3f}")
        )
        fig.update_layout(
            plot_bgcolor    = '#0a0c10',
            paper_bgcolor   = '#0a0c10',
            font_color      = 'white',
            height          = 350,
            showlegend      = False,
            xaxis_tickangle = -30
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### F1 Score Comparison")
        fig = px.bar(
            results_df,
            x     = 'Model',
            y     = 'F1',
            color = 'F1',
            color_continuous_scale = 'Blues',
            text  = results_df['F1'].apply(
                lambda x: f"{x:.3f}")
        )
        fig.update_layout(
            plot_bgcolor    = '#0a0c10',
            paper_bgcolor   = '#0a0c10',
            font_color      = 'white',
            height          = 350,
            showlegend      = False,
            xaxis_tickangle = -30
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### False Alarms per Model "
                "(Lower is Better)")
    fig = px.bar(
        results_df,
        x     = 'Model',
        y     = 'FP',
        color = 'FP',
        color_continuous_scale = 'Reds',
        text  = 'FP'
    )
    fig.update_layout(
        plot_bgcolor    = '#0a0c10',
        paper_bgcolor   = '#0a0c10',
        font_color      = 'white',
        height          = 300,
        showlegend      = False,
        xaxis_tickangle = -30
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🧠 SHAP Feature Importance")
    col1, col2 = st.columns(2)

    with col1:
        shap_data = {
            'Feature'    : ['V14', 'V4', 'V12',
                            'V10', 'V3', 'V11',
                            'V17', 'V16'],
            'Importance' : [0.85, 0.72, 0.65,
                            0.58, 0.52, 0.48,
                            0.42, 0.38]
        }
        shap_fig_df = pd.DataFrame(shap_data)
        fig = px.bar(
            shap_fig_df,
            x           = 'Importance',
            y           = 'Feature',
            orientation = 'h',
            color       = 'Importance',
            color_continuous_scale = 'Reds'
        )
        fig.update_layout(
            title         = "Top SHAP Features",
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 350,
            showlegend    = False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### V14 Key Rule")
        st.error("**V14 < -3** → High Fraud Probability\n\n"
                 "Mean fraud V14  : **-6.271**\n\n"
                 "Mean legit V14  : **+0.029**\n\n"
                 "Separation      : **6.3 std deviations**")

        st.markdown("##### Production Strategy")
        st.info("**Layer 1 — XGBoost**\n\n"
                "94.94% precision · 4 false alarms\n\n"
                "Best for known fraud patterns")
        st.warning("**Layer 2 — Autoencoder**\n\n"
                   "AUC-ROC 0.9289\n\n"
                   "Best for new unknown patterns")

    st.markdown("---")
    st.markdown(
        "**FraudShield AI** | "
        "Suman Das | PNB 11 yrs | "
        "MTech IAR Jadavpur University | "
        "suman.ju.ai@gmail.com"
    )
