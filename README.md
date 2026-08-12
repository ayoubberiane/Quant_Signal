# QuantSignal

### Quantitative Market Intelligence & Portfolio Research Platform

QuantSignal is an open-source quantitative finance project designed to analyze financial markets using data analytics, statistical methods, quantitative factors, and algorithmic portfolio research.

The project aims to transform raw financial-market data into systematic investment signals while emphasizing transparency, reproducibility, and rigorous backtesting.

> **Note:** QuantSignal is an educational and research project. It does not provide financial advice or recommendations to buy or sell securities.

---

## 🎯 Project Objective

QuantSignal investigates whether quantitative signals derived from historical market data can be used to identify attractive risk-adjusted opportunities across different asset classes.

The platform will progressively develop from a financial data pipeline into a complete quantitative research system.

### Core Research Questions

- Which assets currently exhibit the strongest quantitative characteristics?
- Can momentum and volatility factors identify differences in expected performance?
- How do market regimes affect quantitative signals?
- How can portfolio risk be measured and controlled?
- Do strategies remain effective after transaction costs and realistic assumptions?
- How does a systematic strategy compare with traditional benchmarks?

---

## 🚧 Project Status

**Current Stage: Phase 1 — Data & Factor Research**

### Roadmap

- [x] Repository architecture
- [ ] Financial data ingestion
- [ ] Data cleaning and validation
- [ ] Factor construction
- [ ] Quantitative asset ranking
- [ ] Exploratory data analysis
- [ ] Portfolio construction
- [ ] Risk analytics
- [ ] Backtesting engine
- [ ] Market-regime detection
- [ ] Machine-learning research
- [ ] Interactive dashboard
- [ ] Automated data updates
- [ ] Cloud deployment

---

## 📊 Assets

The initial research universe includes multiple asset classes:

| Asset | Category |
|---|---|
| SPY | US Large Cap |
| QQQ | US Technology |
| IWM | US Small Cap |
| EFA | Developed Markets |
| EEM | Emerging Markets |
| TLT | US Treasury Bonds |
| GLD | Gold |
| SLV | Silver |
| USO | Oil |
| VNQ | Real Estate |
| BTC-USD | Bitcoin |
| ETH-USD | Ethereum |
| EURUSD=X | EUR/USD |
| JPY=X | USD/JPY |
| ^VIX | Market Volatility |

The universe may be expanded as the project develops.

---

## 🧮 Quantitative Factors

The first version of QuantSignal will investigate several quantitative factors.

### Momentum

- 1-month momentum
- 3-month momentum
- 12-month momentum

### Risk

- Rolling volatility
- Annualized volatility
- Maximum drawdown
- Correlation

### Composite QuantSignal Score

The factors will eventually be standardized and combined into a composite score:

QuantScore =

0.20 × Z(Momentum 1M)  
+ 0.30 × Z(Momentum 3M)  
+ 0.40 × Z(Momentum 12M)  
− 0.10 × Z(Volatility)

The factor weights will be treated as research parameters and evaluated through historical testing rather than assumed to be optimal.

---

## 🏗️ Architecture

```text
Financial Data
      │
      ▼
┌─────────────────────┐
│ Data Ingestion      │
│ Python + APIs       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Cleaning       │
│ Validation          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │
│ Momentum / Risk     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ QuantSignal Ranking │
│ Asset Scoring       │
└──────────┬──────────┘
           │
           ▼
      Research Output
