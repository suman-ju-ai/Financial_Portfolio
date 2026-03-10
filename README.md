# FinSight AI 🧠
### Financial Time Series Forecasting + GNN + RL Trading Agent
**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · GNN · Reinforcement Learning · Streamlit
**Domain:** 10 years Banking (PNB) + MTech AI (Jadavpur University, CGPA 9.79)

## Project Goal
End-to-end Financial AI system combining:
- LSTM / GRU / Temporal Fusion Transformer for price forecasting
- Graph Neural Network for inter-asset relationship modelling
- PPO Reinforcement Learning agent for portfolio optimization
- Live Streamlit dashboard

## Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster, walk-forward validation, scaling pipeline
- [ ] Day 3 — Temporal Fusion Transformer
- [ ] Day 4 — RL Environment design
- [ ] Day 5 — PPO Agent training
- [ ] Day 6 — Streamlit dashboard
- [ ] Day 7 — GitHub polish + profile launch

## Assets Covered
NIFTY50 · Reliance · TCS · Bitcoin (2019-2024)

## Results Table
| Model          | MAE      | RMSE     | Directional Acc |
|----------------|----------|----------|-----------------|
| ARIMA baseline | 0.011502 | 0.013421 | ~50%            |
| LSTM V2        | 0.015139 | 0.018931 | 49.7%           |
| TFT            | --       | --       | --              |
| GNN+LSTM       | --       | --       | --              |

Note: Both ARIMA and LSTM achieve ~50% directional accuracy —
expected for financial returns (random walk behaviour).
Pipeline rigour and explainability (Day 3 TFT) is the real value.

## Key Technical Decisions
- Walk-forward validation — no data leakage
- Log returns instead of raw prices — ensures stationarity
- ADF test — statistically confirmed stationarity
- StandardScaler with inverse transform — fair model comparison
- Dropout 0.3 + gradient clipping — overfitting control

## Project Structure
notebooks/
  FinSight_Day1.ipynb      — EDA + ARIMA baseline
  FinSight_Day2_LSTM.ipynb — LSTM training pipeline
models/
  (saved weights added from Day 3 onwards)

## Contact
📧 suman.ju.ai@gmail.com
🔗 linkedin.com/in/suman-das-6b0749276
