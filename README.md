# FinSight AI 🧠
### Financial Time Series Forecasting + GNN + RL Trading Agent

**Author:** Suman Das — Senior Applied Scientist, Financial AI
**Stack:** Python · PyTorch · GNN · Reinforcement Learning · Streamlit
**Domain:** 10 years Banking (PNB) + MTech AI (Jadavpur University, CGPA 9.79)

---

## Project Goal
End-to-end Financial AI system combining:
- LSTM / GRU / Temporal Fusion Transformer for price forecasting
- Graph Neural Network for inter-asset relationship modelling
- PPO Reinforcement Learning agent for portfolio optimization
- Live Streamlit dashboard with explainable AI

---

## Progress
- [x] Day 1 — Data pipeline, EDA, stationarity tests, ARIMA baseline
- [x] Day 2 — LSTM forecaster (walk-forward CV, PyTorch)
- [x] Day 3 — Temporal Fusion Transformer + attention heatmap
- [ ] Day 4 — RL Environment design (Gymnasium)
- [ ] Day 5 — PPO Agent training + backtest
- [ ] Day 6 — Streamlit dashboard deployment
- [ ] Day 7 — GitHub polish + Upwork profile launch

---

## Assets Covered
NIFTY50 · Reliance Industries · TCS · Bitcoin (2019–2024)

---

## Results Table

| Model | MAE | RMSE | Directional Acc | Notes |
|-------|-----|------|-----------------|-------|
| ARIMA baseline | 0.011502 | 0.013421 | ~50% | Statistical baseline |
| LSTM V2 | 0.015139 | 0.018931 | 49.7% | PyTorch, 2-layer, walk-forward CV |
| TFT | 0.012382 | 0.018418 | 53.8% | Attention heatmap extracted |
| GNN + LSTM | -- | -- | -- | Coming Day 4-5 |
| PPO RL Agent | -- | -- | -- | Coming Day 5 |

---

## Key Research Finding — TFT Attention Analysis

The TFT attention mechanism reveals that **Reliance Industries exhibits
strong recency-weighted behaviour** — the model assigns near-zero attention
to observations older than 30 days, with peak attention concentrated in
the 3-7 day window.

This finding is consistent with the **monthly options expiry cycle**
in Indian equity markets and suggests that short-term momentum is a
stronger predictor than long-term historical patterns for this asset.

> This pattern was discovered independently by the model from price data
> alone — no domain knowledge was injected. It validates that TFT learns
> economically meaningful structure, not just statistical noise.

---

## Architecture
```
Raw OHLCV Data (4 assets, 2019-2024)
        ↓
Feature Engineering (RSI, MACD, Bollinger Bands, Volume Ratio)
        ↓
   ┌────────────┬─────────────┐
   │  LSTM/TFT  │     GNN     │  ← Temporal + Relational
   │ (per asset)│(inter-asset)│
   └─────┬──────┴──────┬──────┘
         │   Fusion    │
         └──────┬──────┘
                ↓
         PPO RL Agent
                ↓
      Portfolio Decisions
      (Buy / Hold / Sell)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, pytorch-forecasting |
| Time Series | LSTM, GRU, Temporal Fusion Transformer |
| Graph Learning | PyTorch Geometric (GNN, GAT) |
| RL | Stable-Baselines3, Gymnasium |
| Indicators | pandas-ta |
| Dashboard | Streamlit, Plotly |
| Experiment Tracking | MLflow |
| Deployment | Streamlit Cloud |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Day1_Data_Pipeline.ipynb` | Data download, EDA, stationarity, ARIMA baseline |
| `Day2_LSTM.ipynb` | PyTorch LSTM, walk-forward CV, results |
| `Day3_TFT.ipynb` | Temporal Fusion Transformer, attention heatmap |
| `Day4_RL_Env.ipynb` | Coming tomorrow |
| `Day5_PPO_Agent.ipynb` | Coming Day 5 |

---

## Contact
📧 suman.ju.ai@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/suman-das-6b0749276)
📍 Kolkata, India

---

*Built as part of a career transition from 10 years banking (PNB)
to Financial AI — combining domain expertise with cutting-edge ML.*
