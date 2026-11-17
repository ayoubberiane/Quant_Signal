# QuantSignal — working notes for Claude

Educational quantitative research project. Ranks assets cross-sectionally
on momentum, trend, volume and volatility factors, then backtests the
ranking. **Not investment advice, and no output should be phrased as a
recommendation to buy or sell.**

## Commands

```bash
source .venv/bin/activate

python -m pytest -q                  # 82 tests
python -m pytest -q -m "not slow"    # 75 tests, ~15s; skips the weight fits

python src/ingestion/market_data.py  # Polygon download -> data/raw/
python src/cleaning/clean_data.py    # -> data/processed/<T>_clean.csv
python src/features/factors.py       # -> data/processed/<T>_features.csv
python src/ranking/rank_assets.py    # cross-sectional ranking
python src/backtesting/backtest.py   # simulation with costs and benchmarks
python src/validation/validate.py    # walk-forward out-of-sample validation
```

Each stage reads the previous stage's output from `data/processed/`, so run
them in order. `data/` is gitignored.

## Invariants — do not break these without a deliberate decision

**The score is defined once, in `src/scoring.py`.** Both
`ranking/rank_assets.py` and `backtesting/backtest.py` import it. It used to
be written out separately in each file; a test now fails if the two diverge.
Don't reintroduce a local copy.

**Standardisation is cross-sectional, across assets on one date.** Never
along an asset's own history. In the backtest and validator this means
`by="date"`: pooling dates lets a later distribution inform an earlier
score, which is look-ahead.

This matters more than it looks. The four factors live on different scales
(`return_5d` ~0.01, `trend_signal` 0 or 1, `volume_ratio` unbounded). Summed
raw, `return_5d`'s 30% weight moved 3.3% of the ranking and `volume_ratio`'s
20% moved 54.9% — the ranking was sorting on volume spikes.
`scoring.effective_weights()` reports this, and a test fails if the gap
reopens.

**`zscore` returns zeros, not NaN or inf, on a cross-section with no
dispersion.** Load-bearing for `trend_signal`, which is binary: in a broad
uptrend every asset reads 1, the standard deviation is 0, and a naive
z-score would destroy the whole score rather than just that term.

**Timing: score at *t*, return *t*→*t+1*.** Tests assert that results up to
*t* don't change when later dates are appended. Any change to the
simulation loop must keep that true.

**Turnover is two-way:** `sum |target - drifted|`. A full A-to-B switch is
2.0 and costs `2.0 * bps / 10_000`. There is no compensating factor of two
anywhere else, so don't add one.

**Weights drift between rebalances.** Resetting them daily would understate
turnover and so understate cost.

**`engine.simulate` and `engine.run_backtest` must agree exactly.** The
weight search calls `simulate` directly on pre-pivoted matrices for speed. A
test pins the equivalence; if it fails, fitted results are coming from a
different engine than the tests cover.

**`trend_signal` reads 0 during moving-average warm-up**, meaning "not yet
computable" rather than "downtrend". Consumers must drop those rows.

## Secrets

`.env` holds `POLYGON_API_KEY` and is gitignored, as are `*.swp` and
`.env.*` (except `.env.example`). A nano swap file of `.env` once leaked a
live key into this repository's public history. Never commit a key, never
print one, never put one in a commit message or issue.

## Known open items

- **The leaked Polygon key still needs rotating** at the provider. Removing
  the file did not undo the exposure; the value is in git history.
- **Exploratory data analysis** is the one unchecked Phase 1 roadmap item.
- **The factor set was chosen with hindsight.** Walk-forward validation
  covers the weights, not the choice of factors.
- **Survivorship bias:** the universe is fixed and currently-listed.
- **`^VIX`** appears in the README's universe but not in `market_data.py`.
  It is not investable and mean-reverts, so it belongs as a regime input
  rather than a ranked asset.
- **Costs are a flat bps assumption** — no spread, market impact or borrow.
- **Close-to-close only**, so intraday execution isn't represented.

## Style

Comments explain *why*, not *what*. Several non-obvious choices here
(two-way turnover, constrained weight-grid signs, coarse grid, zeros on
degenerate z-scores) are deliberate and documented at their definitions —
read those before changing them. Tests assert hand-derived values rather
than only that code runs, so a regression surfaces as a wrong number.
