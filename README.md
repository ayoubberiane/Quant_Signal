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

**Current stage: Phase 2 — Backtesting & Validation**

### Roadmap

- [x] Repository architecture
- [x] Financial data ingestion
- [x] Data cleaning and validation
- [x] Factor construction
- [x] Quantitative asset ranking
- [ ] Exploratory data analysis
- [x] Portfolio construction (top-N equal weight)
- [x] Risk analytics (vol, drawdown, Sharpe, Sortino, Calmar)
- [x] Backtesting engine (costs, benchmarks, cost sweep)
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

Each asset gets four factors, computed in `src/features/factors.py`:

| Factor | Definition | Column |
|---|---|---|
| Momentum | 5-day total return | `return_5d` |
| Trend | 20-day MA above 50-day MA (0 or 1) | `trend_signal` |
| Volume | volume over its own 20-day average | `volume_ratio` |
| Risk | 20-day realised volatility, annualised | `volatility_20d` |

Also produced as diagnostics: `daily_return`, `ma_20`, `ma_50`,
`momentum_signal`.

### Composite QuantSignal Score

```
QuantScore = 0.30 * Z(return_5d)
           + 0.30 * Z(trend_signal)
           + 0.20 * Z(volume_ratio)
           - 0.20 * Z(volatility_20d)
```

`Z(.)` is a **cross-sectional** z-score: each factor is standardised across
the assets present on one date, after winsorising at the 5th and 95th
percentiles. Defined once in `src/scoring.py` and shared by the ranking and
the backtest, so the two cannot drift apart.

**Standardisation is not cosmetic.** The four factors live on different
scales: `return_5d` is order 0.01, `trend_signal` is 0 or 1, and
`volume_ratio` is unbounded above. Summed raw, each factor's influence comes
from its units rather than its weight. Measured across 7 assets:

| factor | intended | effective, raw | effective, standardised |
|---|---|---|---|
| `return_5d` | 30% | 3.3% | 29.3% |
| `trend_signal` | 30% | 40.3% | 25.4% |
| `volume_ratio` | 20% | 54.9% | 24.7% |
| `volatility_20d` | 20% | 1.6% | 20.6% |

Unstandardised, the ranking was effectively sorting on volume spikes.
`scoring.effective_weights()` reports this table on live data, so the fault
is visible rather than silent.

`--method rank` substitutes a rank transform for the z-score. With only
7 assets, winsorising clips very little, so a z-score remains somewhat
outlier-sensitive; if the two methods disagree on the ordering, that
ordering rests on a few extreme observations.

**The weights are research parameters, not assumptions of optimality.** They
have not been fitted out-of-sample. Fitting four weights on 7 assets
overfits easily, and any in-sample result describes the past rather than
evidencing an edge.

### What a score is not

A QuantScore ranks assets against each other on one date. It is not a return
forecast and carries no claim about magnitude: 1.5 means an asset screens
well relative to this universe today, nothing more.

---

## 📉 Backtesting

`src/backtesting/engine.py` simulates the strategy; `backtest.py` runs it.

```bash
python src/backtesting/backtest.py                                    # top 3, monthly, 5bp
python src/backtesting/backtest.py --top-n 1 --frequency daily --cost-bps 0
python src/backtesting/backtest.py --top-n 5 --frequency weekly --cost-bps 10
```

**Commitments the engine enforces:**

- **Timing.** Weights come from the score on date *t* and earn the return
  from *t* to *t+1*. A test confirms that results up to *t* do not change
  when later dates are appended.
- **Costs.** Turnover is two-way (`sum |target − drifted|`), so a full
  switch between two assets is a turnover of 2.0 and costs
  `2.0 × bps / 10,000`. Costs are never omitted from a reported figure.
- **Weight drift.** Between rebalances the portfolio is held, so weights
  drift with returns. Resetting them daily would understate turnover, and
  so understate cost.
- **Benchmarks.** Every run reports equal-weight across the universe and
  SPY buy-and-hold alongside the strategy. A strategy that cannot beat
  equal weight has not shown anything, since equal weight uses no signal.
- **Cost sensitivity.** Every run sweeps 0 to 50bp. A Sharpe that collapses
  between 0 and 10bp is a cost artefact; the break-even cost is how much
  slippage the edge absorbs before vanishing.
- **Frequency comparison.** Daily, weekly and monthly side by side at a
  fixed cost, so the trade-off between signal freshness and turnover is
  measured rather than assumed.

### Known limitations

Recorded rather than hidden:

- **Survivorship bias.** The universe is fixed and currently-listed.
- **No out-of-sample weight fitting.** Walk-forward evaluation is not built.
- **Equal weighting within the top N**, because the score is an ordering,
  not a magnitude.
- **Costs are a flat bps assumption**, with no spread, market impact or
  borrow modelled.
- **Daily close-to-close only**, so intraday execution is not represented.

---

## 🏗️ Architecture

```
Polygon.io daily aggregates (adjusted)
      |
      v  src/ingestion/market_data.py      -> data/raw/<TICKER>.csv
+---------------------+
| Ingestion           |  configurable range, rate-limit aware
+----------+----------+
           |
           v  src/cleaning/clean_data.py   -> data/processed/<T>_clean.csv
+---------------------+
| Cleaning            |  dedupe, dropna, price validation
+----------+----------+
           |
           v  src/features/factors.py      -> data/processed/<T>_features.csv
+---------------------+
| Features            |  momentum, trend, volume, volatility
+----------+----------+
           |
           +--------------------------+
           v                          v
  src/ranking/rank_assets.py   src/backtesting/backtest.py
  cross-sectional ranking      simulation, costs, benchmarks
           |                          |
           v                          v
  asset_ranking.csv            backtest_results.csv
```

Both consumers import the score from `src/scoring.py`.

## Repository layout

```
src/
  scoring.py              standardisation and composite score (shared)
  ingestion/market_data.py Polygon download, caching, rate limits
  cleaning/clean_data.py   validation and cleaning
  features/factors.py      factor construction
  ranking/rank_assets.py   cross-sectional ranking
  backtesting/engine.py    portfolio simulation, costs, metrics
  backtesting/backtest.py  backtest entry point
tests/
  test_factors.py          33 tests: features, scoring, ranking
  test_engine.py           32 tests: simulation, costs, benchmarks
data/
  raw/                     downloaded prices (gitignored)
  processed/               cleaned data and results (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.lock.txt

cp .env.example .env     # then add your Polygon key
python -m pytest tests/ -v
```

### Dependencies

| File | What it holds |
|---|---|
| `requirements.txt` | runtime packages, unpinned |
| `requirements-dev.txt` | runtime + `pytest`, `jupyter`, `pip-tools` |
| `requirements.lock.txt` | every package pinned to an exact, tested version with hashes (generated) |

Install from the lock file so everyone, and CI, runs the same versions. To add
or upgrade a package, edit `requirements.txt` or `requirements-dev.txt`, then
regenerate the lock with Python 3.13 and commit both files:

```bash
pip-compile --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url \
    --output-file requirements.lock.txt requirements-dev.txt
```

Add `--upgrade` to move every pin to the latest release. The lock is resolved
on Linux; if `pip install` fails on macOS or Windows, fall back to
`pip install -r requirements-dev.txt`.

`.env` is gitignored and must stay that way. Editor swap files (`*.swp`)
are ignored too: a nano swap file of `.env` once leaked a live API key into
this repository's history.

## Licence

MIT
