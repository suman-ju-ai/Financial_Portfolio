# FinSight AI 🧠
### Financial Time Series Forecasting + GNN + RL Trading Agent
**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · GNN · Reinforcement Learning · Streamlit
**Domain:** 10 years Banking (PNB) + MTech AI (Jadavpur University, CGPA 9.81)

## Project Goal
End-to-end Financial AI system combining:
- LSTM / GRU / Temporal Fusion Transformer for price forecasting
- Graph Neural Network for inter-asset relationship modelling
- PPO Reinforcement Learning agent for portfolio optimization
- Live Streamlit dashboard

## Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster, walk-forward validation, scaling pipeline
- [x] Day 3 — Temporal Fusion Transformer + attention heatmap
- [x] Day 4 — RL Environment, Gymnasium TradingEnv, random agent baseline
- [ ] Day 5 — PPO Agent training
- [ ] Day 6 — Streamlit dashboard
- [ ] Day 7 — GitHub polish + profile launch

## Assets Covered
NIFTY50 · Reliance · TCS · Bitcoin (2019-2024)

## Results Table
| Model                   | MAE      | RMSE     | Directional Acc      |
|-------------------------|----------|----------|----------------------|
| ARIMA baseline          | 0.011502 | 0.013421 | ~50%                 |
| LSTM V2                 | 0.015139 | 0.018931 | 49.7%                |
| TFT                     | 0.012382 | 0.018418 | 53.8%                |
| Random Agent (baseline) | N/A      | N/A      | +70.40% total return |
| PPO Agent               | --       | --       | --                   |

Note: ARIMA/LSTM/TFT evaluated on MAE and RMSE for return prediction.
RL agents evaluated on total return and Sharpe ratio — different objective.
Random agent achieved +70.40% purely from Reliance market beta (bull run 2019-2022).
PPO agent must beat this on RISK-ADJUSTED basis using Sharpe ratio.

## Key Technical Decisions
- Walk-forward validation — no data leakage
- Log returns instead of raw prices — ensures stationarity
- ADF test — statistically confirmed stationarity
- StandardScaler with inverse transform — fair model comparison
- Dropout 0.3 + gradient clipping — overfitting control
- Sharpe-adjusted reward — penalises volatility not just losses
- Transaction cost 0.1% — realistic trading simulation
- Stop loss at 70% — prevents catastrophic drawdown

## TFT Attention Insight
TFT independently discovered three financially meaningful patterns:
- Days 0-25: Near zero attention — old data is noise
- Day 30 spike: Monthly options expiry cycle detected
- Days 50-60: Highest attention — recency matters most
The model learned these patterns without being explicitly programmed.

## Project Structure
notebooks/
  FinSight_Day1.ipynb               — EDA + ARIMA baseline
  FinSight_Day2_LSTM.ipynb          — LSTM training pipeline
  FinSight_Day3_TFT.ipynb           — TFT + attention heatmap
  FinSight_Day4_RL_Environment.ipynb — Custom Gymnasium environment
models/
  (saved weights added from Day 5 onwards)
visuals/
  tft_attention.png                 — Attention heatmap
  random_agent_baseline.png         — Random agent equity curve

## Contact
📧 suman.ju.ai@gmail.com
🔗 linkedin.com/in/suman-das-6b0749276
