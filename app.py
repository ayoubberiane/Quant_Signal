"""QuantSignal dashboard (Gradio), for Hugging Face Spaces or local use.

    pip install -r requirements-app.txt
    python app.py

Three tabs: today's ranking, a backtest lab, and a Chronos-Bolt forecast
tested as a factor. Prices come from Yahoo Finance, so no API key is
needed. Everything else reuses the research code in src/ unchanged.
"""

from __future__ import annotations

import sys
import threading
from datetime import date
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtesting.backtest import (  # noqa: E402
    BENCHMARK_TICKER,
    calculate_forward_returns,
    calculate_quant_score,
)
from backtesting.engine import (  # noqa: E402
    DEFAULT_COST_BPS,
    _rebalance_dates,
    buy_and_hold,
    equal_weight_benchmark,
    performance,
    run_backtest,
)
from features.factors import generate_features  # noqa: E402
from forecasting import chronos_signal as cs  # noqa: E402
from ingestion import yahoo  # noqa: E402
from ingestion.market_data import TICKERS  # noqa: E402
from ranking.rank_assets import REQUIRED_FACTORS, calculate_score  # noqa: E402

DISCLAIMER = (
    "Educational research project. Past performance on one fixed universe "
    "of currently-listed ETFs; not investment advice."
)

PALETTE = {
    "strategy": "#2563eb",
    "gross": "#93c5fd",
    "chronos": "#d97706",
    "equal": "#6b7280",
    "bench": "#059669",
}

# --------------------------------------------------------------------------
# Data, cached for the day
# --------------------------------------------------------------------------

_cache: dict = {}
_data_lock = threading.Lock()


def load_data() -> dict:
    """Download prices once per day and build every derived frame.

    Locked: the three tabs load at the same moment, and without the lock
    each would download everything from Yahoo in parallel.
    """
    with _data_lock:
        return _load_data()


def _load_data() -> dict:
    today = date.today()
    if _cache.get("day") == today:
        return _cache

    raw = yahoo.download(TICKERS)
    features = pd.concat(
        [generate_features(group.sort_values("date"))
         for _, group in raw.groupby("ticker")],
        ignore_index=True,
    )
    panel = calculate_forward_returns(calculate_quant_score(features))
    closes = raw.pivot_table(index="date", columns="ticker", values="close")

    _cache.clear()
    _cache.update(day=today, features=features, panel=panel, closes=closes,
                  forecasts={})
    return _cache


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

PCT = ["Total Return", "Annualised Return", "Annualised Volatility",
       "Maximum Drawdown", "Win Rate"]
RATIO = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]


def metrics_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    table = pd.DataFrame({name: performance(s) for name, s in series.items()}).T
    out = pd.DataFrame(index=table.index)
    for column in PCT:
        out[column] = table[column].map(lambda x: f"{x:.1%}")
    for column in RATIO:
        out[column] = table[column].map(lambda x: f"{x:.2f}")
    return out.rename_axis("Series").reset_index()


def equity_figure(series: dict[str, tuple[pd.Series, str]], title: str):
    fig = go.Figure()
    for name, (returns, colour) in series.items():
        curve = (1.0 + returns.fillna(0.0)).cumprod()
        fig.add_trace(go.Scatter(x=curve.index, y=curve, name=name,
                                 line=dict(color=colour, width=2)))
    fig.update_layout(title=title, yaxis_title="Growth of $1",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=-0.15),
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def drawdown_figure(returns: pd.Series):
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy",
                               line=dict(color="#dc2626", width=1),
                               name="Drawdown"))
    fig.update_layout(title="Strategy drawdown (net)", yaxis_tickformat=".0%",
                      template="plotly_white", height=260,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


# --------------------------------------------------------------------------
# Tab 1: ranking
# --------------------------------------------------------------------------

def ranking_view():
    data = load_data()
    latest = (data["features"].dropna(subset=REQUIRED_FACTORS + ["ma_50"])
              .sort_values("date").groupby("ticker").tail(1)
              .set_index("ticker"))
    scored = calculate_score(latest).sort_values("quant_score",
                                                 ascending=False)
    table = pd.DataFrame({
        "Rank": range(1, len(scored) + 1),
        "Asset": scored.index,
        "Close": scored["close"].map(lambda x: f"{x:,.2f}"),
        "5-day return": scored["return_5d"].map(lambda x: f"{x:+.2%}"),
        "Volatility (20d, ann.)": scored["volatility_20d"].map(
            lambda x: f"{x:.1%}"),
        "Volume vs 20d avg": scored["volume_ratio"].map(lambda x: f"{x:.2f}x"),
        "Trend (MA20 > MA50)": scored["trend_signal"].map(
            lambda x: "up" if x == 1 else "down"),
        "Score": scored["quant_score"].map(lambda x: f"{x:+.2f}"),
    })
    as_of = pd.to_datetime(latest["date"]).max().date()
    note = (f"**As of {as_of}.** The score compares the assets with each "
            "other on one day: 30% 5-day momentum, 30% trend, 20% volume "
            "spike, minus 20% volatility, each standardised across assets.")
    return note, table


# --------------------------------------------------------------------------
# Tab 2: backtest lab
# --------------------------------------------------------------------------

def backtest_view(top_n: int, frequency: str, cost_bps: float):
    panel = load_data()["panel"]
    result = run_backtest(panel, top_n=int(top_n), frequency=frequency,
                          cost_bps=cost_bps)
    equal = equal_weight_benchmark(panel).reindex(result.index)
    bench = buy_and_hold(panel, BENCHMARK_TICKER).reindex(result.index)

    curves = {
        "Strategy (net of costs)": (result["net_return"], PALETTE["strategy"]),
        "Strategy (before costs)": (result["gross_return"], PALETTE["gross"]),
        "Equal weight, all assets": (equal, PALETTE["equal"]),
        f"{BENCHMARK_TICKER} buy & hold": (bench, PALETTE["bench"]),
    }
    years = len(result) / 252
    summary = (
        f"Holding the top **{int(top_n)}**, rebalanced **{frequency}**, "
        f"paying **{cost_bps:g} bp** per trade. Turnover "
        f"{result['turnover'].sum() / years:.1f}x a year; costs took "
        f"{result['cost'].sum():.2%} in total. Current holdings: "
        f"**{result['holdings'].iloc[-1] or 'none'}**."
    )
    return (equity_figure(curves, "Strategy vs benchmarks"),
            drawdown_figure(result["net_return"]),
            metrics_table({k: v for k, (v, _) in curves.items()}),
            summary)


# --------------------------------------------------------------------------
# Tab 3: Chronos forecast
# --------------------------------------------------------------------------

def forecast_view(ticker: str, horizon: int):
    data = load_data()
    closes = data["closes"]
    bands = cs.forecast_bands(closes[[ticker]], cs.load_chronos(),
                              horizon=int(horizon))
    if ticker not in bands:
        return go.Figure(), f"Not enough history for {ticker}."
    band = bands[ticker]
    history = closes[ticker].dropna().tail(250)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history, name="Close",
                             line=dict(color=PALETTE["strategy"], width=2)))
    fig.add_trace(go.Scatter(x=band.index, y=band["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band.index, y=band["p10"], fill="tonexty",
                             fillcolor="rgba(217,119,6,0.18)",
                             line=dict(width=0), name="80% range"))
    fig.add_trace(go.Scatter(x=band.index, y=band["p50"], name="Median forecast",
                             line=dict(color=PALETTE["chronos"], width=2,
                                       dash="dash")))
    fig.update_layout(title=f"{ticker}: Chronos-Bolt forecast, "
                            f"{int(horizon)} trading days",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15),
                      margin=dict(l=40, r=20, t=50, b=40))

    last = history.iloc[-1]
    end = band.iloc[-1]
    text = (f"Last close **{last:,.2f}**. Median in {int(horizon)} days: "
            f"**{end['p50']:,.2f}** ({end['p50'] / last - 1:+.1%}); "
            f"80% range {end['p10']:,.2f} to {end['p90']:,.2f}. "
            "The width of the band is the model's uncertainty, and it is "
            "usually wide: that is the honest part of the forecast.")
    return fig, text


def chronos_backtest_view(top_n: int, cost_bps: float,
                          progress=gr.Progress()):
    horizon = 20
    data = load_data()
    panel, closes = data["panel"], data["closes"]

    key = horizon
    if key not in data["forecasts"]:
        dates = sorted(_rebalance_dates(
            pd.DatetimeIndex(sorted(panel["date"].unique())), "monthly"))
        data["forecasts"][key] = cs.expected_returns(
            closes, cs.load_chronos(), dates, horizon=horizon,
            progress=lambda f, d: progress(f, desc=d))
    forecasts = data["forecasts"][key]

    chronos_panel = cs.forecast_panel(panel, forecasts)
    start = forecasts.index.min()
    factor_panel = panel[panel["date"] >= start]

    chronos = run_backtest(chronos_panel, top_n=int(top_n),
                           frequency="monthly", cost_bps=cost_bps)
    factor = run_backtest(factor_panel, top_n=int(top_n),
                          frequency="monthly", cost_bps=cost_bps)
    index = chronos.index.intersection(factor.index)
    equal = equal_weight_benchmark(factor_panel).reindex(index)
    bench = buy_and_hold(factor_panel, BENCHMARK_TICKER).reindex(index)

    curves = {
        "Chronos factor (net)": (chronos["net_return"].reindex(index),
                                 PALETTE["chronos"]),
        "QuantSignal score (net)": (factor["net_return"].reindex(index),
                                    PALETTE["strategy"]),
        "Equal weight, all assets": (equal, PALETTE["equal"]),
        f"{BENCHMARK_TICKER} buy & hold": (bench, PALETTE["bench"]),
    }

    # Score both signals on the same month-end dates.
    score_wide = panel.pivot_table(index="date", columns="ticker",
                                   values="quant_score")
    ic_chronos = cs.information_coefficient(forecasts, closes, horizon)
    ic_factor = cs.information_coefficient(
        score_wide.reindex(forecasts.index), closes, horizon)

    def describe(ic: pd.Series) -> str:
        if ic.empty:
            return "n/a"
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) \
            if len(ic) > 1 and ic.std(ddof=1) > 0 else 0.0
        return f"{ic.mean():+.3f} (t = {t:.1f}, {len(ic)} months)"

    text = (
        f"Each month-end, Chronos forecasts every asset's next {horizon} "
        f"trading days using only prices up to that day. The top "
        f"{int(top_n)} by forecast return are held for the month, at "
        f"{cost_bps:g} bp per trade.\n\n"
        f"**Information coefficient** (rank correlation between signal and "
        f"the next {horizon} days' return; 0 means no skill):\n"
        f"- Chronos: {describe(ic_chronos)}\n"
        f"- QuantSignal score: {describe(ic_factor)}\n\n"
        "With 7 assets and a few dozen months, a |t| below about 2 is "
        "indistinguishable from luck. Read the result that way."
    )
    return (equity_figure(curves, "Chronos factor vs QuantSignal score"),
            metrics_table({k: v for k, (v, _) in curves.items()}),
            text)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

with gr.Blocks(title="QuantSignal") as demo:
    gr.Markdown(
        "# QuantSignal\n"
        f"Systematic ranking and backtesting across {len(TICKERS)} ETFs "
        f"({', '.join(TICKERS)}), with a Hugging Face forecasting model "
        f"tested as a factor. *{DISCLAIMER}*"
    )

    with gr.Tab("Today's ranking"):
        rank_note = gr.Markdown()
        rank_table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Tab("Backtest lab"):
        with gr.Row():
            top_n = gr.Slider(1, len(TICKERS), value=3, step=1,
                              label="Assets held (top N)")
            frequency = gr.Radio(["daily", "weekly", "monthly"],
                                 value="monthly", label="Rebalance")
            cost = gr.Slider(0, 50, value=DEFAULT_COST_BPS, step=1,
                             label="Cost per trade (bp)")
        bt_summary = gr.Markdown()
        bt_equity = gr.Plot()
        bt_drawdown = gr.Plot()
        bt_table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Tab("AI forecast (Chronos)"):
        gr.Markdown(
            "[Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small)"
            " is Amazon's pretrained time-series model on Hugging Face. It "
            "forecasts prices it was never trained on. Below: its forecast "
            "for one asset, then a test of whether it ranks assets better "
            "than the hand-built score."
        )
        with gr.Row():
            fc_ticker = gr.Dropdown(TICKERS, value=TICKERS[0], label="Asset")
            fc_horizon = gr.Slider(5, 60, value=20, step=5,
                                   label="Horizon (trading days)")
        fc_plot = gr.Plot()
        fc_text = gr.Markdown()

        gr.Markdown("### Does it work as a signal?")
        with gr.Row():
            ch_top_n = gr.Slider(1, len(TICKERS), value=3, step=1,
                                 label="Assets held (top N)")
            ch_cost = gr.Slider(0, 50, value=DEFAULT_COST_BPS, step=1,
                                label="Cost per trade (bp)")
        ch_button = gr.Button("Run walk-forward test",
                              variant="primary")
        ch_text = gr.Markdown()
        ch_plot = gr.Plot()
        ch_table = gr.Dataframe(interactive=False, wrap=True)

    bt_inputs = [top_n, frequency, cost]
    bt_outputs = [bt_equity, bt_drawdown, bt_table, bt_summary]
    fc_inputs = [fc_ticker, fc_horizon]

    demo.load(ranking_view, outputs=[rank_note, rank_table])
    demo.load(backtest_view, inputs=bt_inputs, outputs=bt_outputs)
    demo.load(forecast_view, inputs=fc_inputs, outputs=[fc_plot, fc_text])
    for control in bt_inputs:
        control.change(backtest_view, inputs=bt_inputs, outputs=bt_outputs)
    for control in fc_inputs:
        control.change(forecast_view, inputs=fc_inputs,
                       outputs=[fc_plot, fc_text])
    ch_button.click(chronos_backtest_view, inputs=[ch_top_n, ch_cost],
                    outputs=[ch_plot, ch_table, ch_text])


# Warm up before serving, so the first visitor does not wait for the
# download and the model load. Failures here are left to the views, which
# report them in the page.
try:
    load_data()
    cs.load_chronos()
except Exception as error:  # noqa: BLE001
    print(f"Warm-up failed, will retry on first request: {error}")

if __name__ == "__main__":
    demo.launch()
