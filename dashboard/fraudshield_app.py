# ── FraudShield AI — Streamlit Dashboard ─────────────────
# Author: Suman Das | PNB 11 yrs | MTech IAR Jadavpur University
# Fraud Detection Dashboard with Live Scoring + SHAP Explainability

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e53935;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
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

# ── Load Data and Models ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    df = df.drop_duplicates()
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour']       = (df['Time'] / 3600).astype(int) % 24
    return df

@st.cache_resource
def load_models():
    xgb_model = joblib.load('fraudshield_xgb_model.pkl')
    scaler    = joblib.load('fraudshield_scaler.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, scaler, explainer

# ── Feature List ──────────────────────────────────────────
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + \
               ['Log_Amount', 'Hour']

# ── Header ────────────────────────────────────────────────
st.markdown("# 🛡️ FraudShield AI")
st.markdown("### Bank Fraud Detection System with Explainable AI")
st.markdown("---")

# ── Load data and models ──────────────────────────────────
try:
    df                        = load_data()
    xgb_model, scaler, explainer = load_models()
    models_loaded             = True
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

    # ── KPI Metrics ───────────────────────────────────────
    fraud_df   = df[df['Class'] == 1]
    legit_df   = df[df['Class'] == 0]
    fraud_rate = len(fraud_df) / len(df) * 100
    amount_at_risk = fraud_df['Amount'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions",
                  f"{len(df):,}")
    with col2:
        st.metric("Fraud Cases",
                  f"{len(fraud_df):,}",
                  delta=f"{fraud_rate:.3f}% fraud rate",
                  delta_color="inverse")
    with col3:
        st.metric("Amount at Risk",
                  f"${amount_at_risk:,.2f}")
    with col4:
        st.metric("Imbalance Ratio",
                  f"{len(legit_df)//len(fraud_df)}:1")

    st.markdown("---")

    # ── Fraud by Hour ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fraud Rate by Hour of Day")
        fraud_by_hour = df[df['Class']==1].groupby(
            'Hour').size()
        legit_by_hour = df[df['Class']==0].groupby(
            'Hour').size()
        fraud_rate_hour = (fraud_by_hour /
                          (fraud_by_hour +
                           legit_by_hour) * 100
                          ).fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = fraud_rate_hour.index,
            y    = fraud_rate_hour.values,
            mode = 'lines+markers',
            fill = 'tozeroy',
            line = dict(color='#e53935', width=2),
            name = 'Fraud Rate %'
        ))
        fig.update_layout(
            xaxis_title = "Hour of Day",
            yaxis_title = "Fraud Rate (%)",
            plot_bgcolor = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color   = 'white',
            height = 300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Transaction Amount Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x       = legit_df['Amount'].clip(upper=500),
            name    = 'Legitimate',
            opacity = 0.7,
            marker_color = '#00bcd4',
            nbinsx  = 50
        ))
        fig.add_trace(go.Histogram(
            x       = fraud_df['Amount'].clip(upper=500),
            name    = 'Fraud',
            opacity = 0.7,
            marker_color = '#e53935',
            nbinsx  = 30
        ))
        fig.update_layout(
            barmode      = 'overlay',
            xaxis_title  = "Amount ($)",
            yaxis_title  = "Count",
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 300
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Fraud Count by Hour Bar ───────────────────────────
    st.markdown("#### Fraud Count by Hour of Day")
    fig = px.bar(
        x     = fraud_by_hour.index,
        y     = fraud_by_hour.values,
        color = fraud_by_hour.values,
        color_continuous_scale = 'Reds',
        labels = {'x': 'Hour', 'y': 'Fraud Count'}
    )
    fig.update_layout(
        plot_bgcolor  = '#0a0c10',
        paper_bgcolor = '#0a0c10',
        font_color    = 'white',
        height        = 250,
        showlegend    = False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Key Insights ──────────────────────────────────────
    st.markdown("#### 🔍 Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⏰ **Peak Fraud Hours**\n\n"
                "2AM, 4AM, 3AM show 30x higher "
                "fraud rate than 10AM")
    with col2:
        st.info("💰 **Amount Pattern**\n\n"
                "Fraud median amount is lower — "
                "threshold evasion behaviour detected")
    with col3:
        st.info("⚖️ **Class Imbalance**\n\n"
                "598:1 ratio — accuracy is meaningless. "
                "Model uses Precision, Recall, F1")

# ══════════════════════════════════════════════════════════
# TAB 2 — LIVE FRAUD SCORER
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Live Transaction Fraud Scorer")
    st.markdown("Enter transaction details to score "
                "in real time with SHAP explanation.")

    if not models_loaded:
        st.error("Models not loaded. Cannot score.")
    else:
        # ── Input Form ────────────────────────────────────
        st.markdown("#### Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value   = 0.0,
                max_value   = 50000.0,
                value       = 150.0,
                step        = 0.01
            )
            hour = st.slider(
                "Hour of Day (0-23)",
                min_value = 0,
                max_value = 23,
                value     = 14
            )

        with col2:
            st.markdown("**Key PCA Features**")
            v14 = st.number_input("V14 (strongest fraud signal)",
                                   value=0.0, step=0.01,
                                   format="%.4f")
            v4  = st.number_input("V4",
                                   value=0.0, step=0.01,
                                   format="%.4f")
            v12 = st.number_input("V12",
                                   value=0.0, step=0.01,
                                   format="%.4f")
            v10 = st.number_input("V10",
                                   value=0.0, step=0.01,
                                   format="%.4f")
            v3  = st.number_input("V3",
                                   value=0.0, step=0.01,
                                   format="%.4f")

        # ── Use Real Example Buttons ──────────────────────
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

        # ── Score Button ──────────────────────────────────
        score_btn = st.button(
            "🔍 SCORE TRANSACTION",
            type             = "primary",
            use_container_width = True
        )

        if score_btn or use_fraud or use_legit:
            # Build feature array
            features = np.zeros(30)

            if use_fraud:
                fraud_sample = df[df['Class']==1].iloc[0]
                for i, col in enumerate(
                        [f'V{j}' for j in range(1, 29)]):
                    features[i] = fraud_sample[col]
                features[28] = np.log1p(
                    fraud_sample['Amount'])
                features[29] = int(
                    fraud_sample['Time'] / 3600) % 24
                amount = fraud_sample['Amount']

            elif use_legit:
                legit_sample = df[df['Class']==0].iloc[0]
                for i, col in enumerate(
                        [f'V{j}' for j in range(1, 29)]):
                    features[i] = legit_sample[col]
                features[28] = np.log1p(
                    legit_sample['Amount'])
                features[29] = int(
                    legit_sample['Time'] / 3600) % 24
                amount = legit_sample['Amount']

            else:
                v_cols = [f'V{i}' for i in range(1, 29)]
                for i, col in enumerate(v_cols):
                    features[i] = 0.0
                # Override with user inputs
                features[13] = v14
                features[3]  = v4
                features[11] = v12
                features[9]  = v10
                features[2]  = v3
                features[28] = np.log1p(amount)
                features[29] = hour

            # Scale and predict
            features_df  = pd.DataFrame(
                [features], columns=FEATURE_COLS)
            scaled       = scaler.transform(features_df)
            scaled_df    = pd.DataFrame(
                scaled, columns=FEATURE_COLS)

            prob     = float(
                xgb_model.predict_proba(scaled)[0][1])
            decision = "FRAUD" if prob >= 0.5 \
                       else "LEGITIMATE"

            # ── Results Display ───────────────────────────
            st.markdown("---")
            st.markdown("#### 🎯 Scoring Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                if decision == "FRAUD":
                    st.error(f"🚨 **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **LEGITIMATE**")
            with col2:
                st.metric("Fraud Probability",
                          f"{prob:.1%}")
            with col3:
                confidence = (
                    "HIGH"   if prob > 0.8 or prob < 0.2
                    else "MEDIUM" if prob > 0.6 or prob < 0.4
                    else "LOW"
                )
                st.metric("Confidence", confidence)

            # ── Probability Gauge ─────────────────────────
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

            # ── SHAP Explanation ──────────────────────────
            st.markdown("#### 🧠 Why This Decision?")
            shap_vals = explainer.shap_values(scaled_df)[0]

            top_idx   = np.argsort(
                np.abs(shap_vals))[::-1][:8]
            shap_df   = pd.DataFrame({
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
                x           = shap_df['SHAP Value'],
                y           = shap_df['Feature'],
                orientation = 'h',
                marker_color = colors,
                text        = shap_df['Direction'],
                textposition = 'outside'
            ))
            fig.update_layout(
                title        = "Top Feature Contributions",
                xaxis_title  = "SHAP Value",
                plot_bgcolor  = '#0a0c10',
                paper_bgcolor = '#0a0c10',
                font_color    = 'white',
                height        = 350
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Feature Table ─────────────────────────────
            st.dataframe(
                shap_df.style.applymap(
                    lambda x: 'color: #e53935'
                    if '🔴' in str(x)
                    else 'color: #00bcd4',
                    subset=['Direction']
                ),
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Model Performance")

    # ── Results Table ─────────────────────────────────────
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

    # ── Bar Charts ────────────────────────────────────────
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
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 350,
            showlegend    = False,
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
            plot_bgcolor  = '#0a0c10',
            paper_bgcolor = '#0a0c10',
            font_color    = 'white',
            height        = 350,
            showlegend    = False,
            xaxis_tickangle = -30
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── False Alarms ──────────────────────────────────────
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
        plot_bgcolor  = '#0a0c10',
        paper_bgcolor = '#0a0c10',
        font_color    = 'white',
        height        = 300,
        showlegend    = False,
        xaxis_tickangle = -30
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Summary ──────────────────────────────────────
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
            title        = "Top SHAP Features",
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

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**FraudShield AI** | "
        "Suman Das | PNB 11 yrs | "
        "MTech IAR Jadavpur University | "
        "suman.ju.ai@gmail.com"
    )
```

---

## requirements.txt

Create a file called **`requirements.txt`** with:
```
streamlit
pandas
numpy
plotly
joblib
shap
xgboost
scikit-learn
```

---

## Deploy to Streamlit Cloud

**Step 1** — Create new GitHub repo called:
```
fraudshield-ai
```

**Step 2** — Upload these 4 files:
```
fraudshield_app.py
requirements.txt
fraudshield_xgb_model.pkl
fraudshield_scaler.pkl