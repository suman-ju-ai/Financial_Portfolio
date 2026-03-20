# Financial AI Portfolio 🧠🛡️
### FinSight AI + FraudShield AI — Complete Financial AI System

**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · XGBoost · Reinforcement Learning · Streamlit · FastAPI
**Domain:** 11 yrs Banking (PNB) + MTech IAR (Jadavpur University, CGPA 9.79)

## 🚀 Live Demo
👉 **https://finsight-ai-9qwq8uhnwckfp2vxx9ch4b.streamlit.app/**

---

## Project 1 — FinSight AI 📈
### Financial Time Series Forecasting + RL Trading Agent

End-to-end Financial AI system combining deep learning
forecasting with reinforcement learning portfolio optimization.

### FinSight Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster, walk-forward validation, scaling pipeline
- [x] Day 3 — Temporal Fusion Transformer + attention heatmap
- [x] Day 4 — RL Environment, Gymnasium TradingEnv, fair baseline comparison
- [x] Day 5 — PPO V1+V2+V3 trained, BalancedTradingEnv, bear market analysis
- [x] Day 6 — Streamlit dashboard live, 3 tabs, permanent public URL
- [x] Day 7 — Upwork + Fiverr profile launched, LinkedIn published

### Assets Covered
NIFTY50 · Reliance · TCS · Bitcoin (2019-2024)

### Forecasting Results — Reliance 2024
| Model          | MAE      | RMSE     | Directional Acc |
|----------------|----------|----------|-----------------|
| ARIMA baseline | 0.011502 | 0.013421 | ~50%            |
| LSTM V2        | 0.015139 | 0.018931 | 49.7%           |
| TFT            | 0.012382 | 0.018418 | 53.8%           |

### RL Trading Results — Fair Comparison on 2024 Bear Market
| Agent          | Total Return | Sharpe | Max Drawdown | Behaviour             |
|----------------|-------------|--------|--------------|-----------------------|
| Random Agent   | -25.26%     | —      | —            | Random baseline       |
| Buy & Hold     | -18.08%     | -0.963 | -24.46%      | Market benchmark      |
| PPO V1 ⭐      | 0.00%       | 0.000  | 0.00%        | Cash preservation     |
| PPO V2 ⭐      | 0.00%       | 0.000  | 0.00%        | Cash preservation     |
| PPO V3 ⭐      | -5.47%      | -0.745 | -7.79%       | Active trading        |

All 3 PPO agents outperformed Random Agent and Buy & Hold
on 2024 bear market data.

PPO V1 and V2 beat random agent by +25.26%.
PPO V3 beat random agent by +19.79%.
PPO V3 beat Buy & Hold by +12.61%.
Max drawdown reduced from 24.46% to 7.79%.

### RL Agent Evolution
- PPO V1: Learned cash preservation — rational in bear market
- PPO V2: Deeper network [128,128,64] — same cash preservation
- PPO V3: BalancedTradingEnv with inactivity penalty (-0.0002)
  broke cash-preservation bias — active trading achieved
  (38 buys, 205 sells across 243 trading days)

### Random Agent Baselines
Training data (2019-2022 bull market):
- Total steps    : 935
- Final return   : +21.48%
- Stop loss hit  : False

Test data (2024 bear market — fair comparison):
- Total steps    : 215
- Final return   : -25.26%
- Stop loss hit  : False

Note: Step count difference (935 vs 215) confirmed different
datasets were used. Fair comparison requires same test data.

### TFT Attention Insight
TFT independently discovered three financially meaningful patterns:
- Days 0-25  : Near zero attention — old data is noise
- Day 30 spike: Monthly options expiry cycle detected
- Days 50-60 : Highest attention — recency matters most

---

## Project 2 — FraudShield AI 🛡️
### Bank Fraud Detection with Explainable AI

End-to-end fraud detection system handling extreme class
imbalance (598:1) with SHAP explainability.

### FraudShield Progress
- [x] Day 1-2 — EDA, SMOTE, 5 models trained and compared
- [x] Day 3   — SHAP Explainability + V14 analysis
- [ ] Day 4   — Autoencoder Anomaly Detection
- [ ] Day 5   — FastAPI Real-Time Scoring Endpoint
- [ ] Day 6   — Streamlit Fraud Analyst Dashboard
- [ ] Day 7   — GitHub Polish + New Fiverr Gig

### Dataset
Credit Card Fraud Detection (Kaggle)
283,726 transactions · 473 frauds · 598:1 imbalance ratio

### FraudShield Model Results
| Model               | Precision | Recall | F1     | False Alarms |
|---------------------|-----------|--------|--------|--------------|
| Logistic Regression | 5.21%     | 87.37% | 0.0983 | 1,510        |
| Random Forest       | 59.38%    | 80.00% | 0.6816 | 52           |
| XGBoost Base ⭐     | 94.94%    | 78.95% | 0.8621 | 4            |
| XGBoost Tuned V1    | 93.51%    | 75.79% | 0.8372 | 5            |
| XGBoost Tuned V2    | 92.50%    | 77.89% | 0.8457 | 6            |

XGBoost achieved 94.94% precision with only 4 false alarms
per 56,651 legitimate transactions.

### SHAP Explainability Results
- Top 5 features  : V14, V4, V12, V10, V3
- V14 separation  : 6.3 standard deviations
- Fraud prob      : 99.6% correctly identified
- Legit prob      : 0.0% correctly identified
- V14 < -3        : High fraud probability (mean=-6.271)
- V14 > -1        : Low fraud probability (mean=+0.029)

### Fraud Temporal Patterns
- Peak fraud hours : 2AM, 4AM, 3AM
- Lowest fraud     : 10AM, 10PM, 12AM
- Fraud rate ratio : 30x higher at 2AM vs 10AM

---

## Key Technical Decisions

### FinSight AI
- Walk-forward validation — no data leakage
- Log returns instead of raw prices — ensures stationarity
- ADF test — statistically confirmed stationarity
- StandardScaler with inverse transform — fair comparison
- Dropout 0.3 + gradient clipping — overfitting control
- Sharpe-adjusted reward — penalises volatility not just losses
- Transaction cost 0.1% — realistic trading simulation
- Stop loss at 70% — prevents catastrophic drawdown
- PPO clip_range=0.2 — stable policy updates
- Inactivity penalty -0.0002 — forces active trading
- Fair baseline: same test data for all agent comparisons

### FraudShield AI
- SMOTE oversampling — 598:1 imbalance handled
- scale_pos_weight=599 — XGBoost native imbalance handling
- Stratified train/test split — preserves fraud ratio
- Log transform on Amount — reduces skewness
- Hour of day feature — temporal fraud pattern signal
- SHAP TreeExplainer — regulatory-grade explainability

---

financial-ai-portfolio/
│
├── notebooks/
│   ├── FinSight_Day1.ipynb
│   ├── FinSight_Day2_LSTM.ipynb
│   ├── FinSight_Day3_TFT.ipynb
│   ├── FinSight_Day4_RL_Environment.ipynb
│   ├── FinSight_Day5_PPO_Agent.ipynb
│   ├── FinSight_Day6_Dashboard.ipynb
│   ├── FraudShield_Day1_Day2_EDA_Models.ipynb
│   └── FraudShield_Day3_SHAP.ipynb
│
├── dashboard/
│   ├── app.py
│   └── requirements.txt
│
├── models/
│   ├── ppo_trading_agent.zip
│   ├── ppo_finsight_v2_optimized.zip
│   ├── ppo_balanced_agent_v3.zip
│   └── fraudshield_xgb_model.pkl
│
├── visuals/
│   ├── tft_attention.png
│   ├── final_equity_comparison.png
│   ├── ppo_backtest_results.png
│   ├── random_agent_2024.png
│   ├── class_distribution.png
│   ├── fraud_by_hour.png
│   ├── complete_model_comparison.png
│   ├── shap_summary.png
│   ├── shap_fraud_explanation.png
│   ├── shap_legit_explanation.png
│   └── shap_v14_analysis.png
│
└── README.md

## Contact
📧 suman.ju.ai@gmail.com
🔗 linkedin.com/in/suman-das-6b0749276
💻 github.com/suman-ju-ai
