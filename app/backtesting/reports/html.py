"""Self-contained HTML reporting with interactive Plotly charts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.backtesting.engine.portfolio import BacktestResult
from app.backtesting.metrics.performance import (
    equity_and_drawdown,
    monthly_returns,
    stock_performance_summary,
)


def write_strategy_report(
    result: BacktestResult,
    output_dir: str | Path,
    *,
    benchmark_name: str,
    limitations: list[str],
    sensitivity: pd.DataFrame | None = None,
    random_baseline: pd.DataFrame | None = None,
    monte_carlo_results: pd.DataFrame | None = None,
) -> Path:
    """Write CSV/JSON artifacts and a compact plain-language HTML report."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(directory / "trade_log.csv", index=False)
    stock_summary = stock_performance_summary(result.trades)
    stock_summary.to_csv(directory / "stock_performance_summary.csv", index=False)
    result.periods.to_csv(directory / "period_returns.csv", index=False)
    result.skipped_orders.to_csv(directory / "skipped_orders.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
    )
    if sensitivity is not None:
        sensitivity.to_csv(directory / "parameter_sensitivity.csv", index=False)
    if random_baseline is not None:
        random_baseline.to_csv(directory / "random_portfolio_baseline.csv", index=False)
    if monte_carlo_results is not None:
        monte_carlo_results.to_csv(directory / "monte_carlo.csv", index=False)

    figures: list[str] = []
    if not result.periods.empty:
        curve = equity_and_drawdown(result.periods)
        equity_figure = go.Figure()
        equity_figure.add_scatter(
            x=curve["exit_date"], y=curve["portfolio_equity"], name="Net strategy"
        )
        equity_figure.add_scatter(
            x=curve["exit_date"], y=curve["gross_equity"], name="Gross strategy"
        )
        equity_figure.add_scatter(
            x=curve["exit_date"], y=curve["benchmark_equity"], name=benchmark_name
        )
        equity_figure.update_layout(title="Equity curve", template="plotly_white")
        figures.append(equity_figure.to_html(full_html=False, include_plotlyjs="cdn"))
        drawdown = px.area(
            curve, x="exit_date", y="drawdown", title="Drawdown", template="plotly_white"
        )
        figures.append(drawdown.to_html(full_html=False, include_plotlyjs=False))
        monthly = monthly_returns(result.periods)
        if not monthly.empty:
            heatmap = px.imshow(
                monthly,
                text_auto=".1%",
                aspect="auto",
                color_continuous_scale="RdYlGn",
                title="Monthly net returns",
            )
            figures.append(heatmap.to_html(full_html=False, include_plotlyjs=False))
    if sensitivity is not None and not sensitivity.empty:
        pivot = sensitivity.pivot(index="stop_loss_pct", columns="top_n", values="sharpe")
        figure = px.imshow(
            pivot, text_auto=".2f", aspect="auto", title="Parameter sensitivity: Sharpe"
        )
        figures.append(figure.to_html(full_html=False, include_plotlyjs=False))

    verdict = _verdict(result.metrics, random_baseline)
    metrics_table = "".join(
        f"<tr><th>{html.escape(key.replace('_', ' ').title())}</th><td>{_format(value)}</td></tr>"
        for key, value in result.metrics.items()
        if not isinstance(value, (dict, list))
    )
    caveats = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    stock_tables = _stock_attribution_html(stock_summary)
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>NATIP Backtest</title>"
        "<style>body{font-family:Inter,Arial,sans-serif;max-width:1180px;margin:28px "
        "auto;color:#202124}table{border-collapse:collapse;width:100%;max-width:760px}"
        "th,td{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left}"
        ".verdict{border-left:5px solid #0f9d79;background:#f3faf7;padding:14px 18px}"
        ".warning{background:#fff6e5;padding:12px}</style></head>"
        f"<body><h1>NATIP backtest summary</h1><div class=\"verdict\"><strong>"
        f"{html.escape(verdict)}</strong></div>"
        "<p>Trades use signals known at close T and enter no earlier than the next session "
        "open. Results are research evidence, not a return guarantee.</p>"
        f"<h2>Performance</h2><table>{metrics_table}</table>{''.join(figures)}"
        f"{stock_tables}"
        f"<h2>Known limitations</h2><div class=\"warning\"><ul>{caveats}</ul></div>"
        "</body></html>"
    )
    path = directory / "summary_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_signal_report(
    quantiles: pd.DataFrame,
    ic_periods: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Write signal-research charts and return the HTML path."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    quantiles.to_csv(directory / "quantile_returns.csv", index=False)
    ic_periods.to_csv(directory / "ic_timeseries.csv", index=False)
    clean = quantiles[quantiles["quantile"] != "top_minus_bottom"].copy()
    figures = []
    if not clean.empty:
        figure = px.bar(
            clean,
            x="quantile",
            y="mean_forward_return",
            color="horizon",
            barmode="group",
            title="Forward return by signal quantile",
        )
        figures.append(figure.to_html(full_html=False, include_plotlyjs="cdn"))
    if not ic_periods.empty:
        figure = px.line(
            ic_periods, x="date", y="ic", color="horizon", title="Information coefficient"
        )
        figures.append(figure.to_html(full_html=False, include_plotlyjs=False))
    path = directory / "signal_report.html"
    path.write_text(
        "<!doctype html><html><body>" + "".join(figures) + "</body></html>", encoding="utf-8"
    )
    return path


def _verdict(metrics: dict[str, object], random_baseline: pd.DataFrame | None) -> str:
    if not metrics:
        return "INSUFFICIENT DATA: no auditable portfolio periods were produced."
    sharpe = float(metrics.get("sharpe") or 0)
    drawdown = float(metrics.get("max_drawdown") or 0)
    beats_random = None
    if random_baseline is not None and not random_baseline.empty:
        beats_random = sharpe > float(random_baseline["sharpe"].quantile(0.95))
    if sharpe >= 1 and drawdown > -0.35 and beats_random is not False:
        return (
            "PROMISING EDGE: positive after-cost evidence, subject to robustness and data "
            "limitations."
        )
    return "NO CREDIBLE EDGE YET: after-cost or robustness evidence is insufficient."


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _stock_attribution_html(summary: pd.DataFrame) -> str:
    """Render concise best/worst stock attribution tables."""

    if summary.empty:
        return "<h2>Stock attribution</h2><p>No executed trades were available.</p>"
    supported = summary.loc[summary["sample_quality"] == "SUPPORTED"]
    ranked = supported if not supported.empty else summary
    strongest = ranked.head(10)
    weakest = ranked.tail(10).sort_values("total_net_contribution")

    def render(frame: pd.DataFrame) -> str:
        rows = []
        for row in frame.itertuples(index=False):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.symbol))}</td>"
                f"<td>{int(row.trades)}</td>"
                f"<td>{row.total_net_contribution:.2%}</td>"
                f"<td>{row.average_net_return:.2%}</td>"
                f"<td>{row.median_net_return:.2%}</td>"
                f"<td>{row.win_rate:.2%}</td>"
                f"<td>{row.worst_trade_return:.2%}</td>"
                f"<td>{html.escape(str(row.data_review))}</td>"
                f"<td>{html.escape(str(row.sample_quality))}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr><th>Stock</th><th>Trades</th><th>Net contribution</th>"
            "<th>Avg trade</th><th>Median trade</th><th>Win rate</th>"
            "<th>Worst trade</th><th>Data review</th><th>Sample</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    return (
        "<h2>Stock attribution</h2>"
        "<p>Ranked by total net portfolio contribution. Returns cover only periods when the "
        "strategy selected the stock; they are not buy-and-hold returns.</p>"
        f"<h3>Strongest contributors</h3>{render(strongest)}"
        f"<h3>Weakest contributors</h3>{render(weakest)}"
        "<p>The complete attribution is available in "
        "<code>stock_performance_summary.csv</code>.</p>"
    )
