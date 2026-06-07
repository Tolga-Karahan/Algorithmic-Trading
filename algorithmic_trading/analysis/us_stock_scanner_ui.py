"""Interactive UI for the US stock scanner.

Pick a scan mode, tune its parameters, optionally restrict by market cap or
date range, click Run, and see the results in a sortable table.

Run:
    poetry run python -m algorithmic_trading.analysis.us_stock_scanner_ui
Then open http://127.0.0.1:8050 in a browser.
"""

import datetime as dt
import traceback

import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
from tqdm import tqdm as _real_tqdm

import algorithmic_trading.analysis.us_stock_scanner as scanner


# Shared progress state, fed by a tqdm subclass below.
# Single-user UI, so module-level mutable state is fine.
_SCAN_PROGRESS = {"current": 0, "total": 0, "desc": ""}


class _ProgressTqdm(_real_tqdm):
    """Mirrors every tqdm update into _SCAN_PROGRESS for the UI poller."""
    def update(self, n=1):
        super().update(n)
        _SCAN_PROGRESS["current"] = int(self.n)
        _SCAN_PROGRESS["total"] = int(self.total or 0)
        _SCAN_PROGRESS["desc"] = str(self.desc or "")


# Replace the scanner module's tqdm reference so its internal `tqdm(...)`
# calls feed our progress dict instead of the vanilla bar.
scanner.tqdm = _ProgressTqdm
from algorithmic_trading.analysis.us_stock_scanner import (
    _parse_market_cap,
    _prepare_universe,
    _load_quote_metrics_cache,
    _load_min_volume_cache,
    _load_ticker_cache,
    MARKETCAP_CACHE_TTL_SECONDS,
    MIN_VOLUME_CACHE_TTL_SECONDS,
    TICKER_CACHE_TTL_SECONDS,
    scan_uptrend,
    scan_earnings,
    scan_revisions,
    scan_target_hikes,
    scan_squeeze,
    compute_economic_releases,
    get_upcoming_earnings,
    DEFAULT_MIN_SURPRISE_PCT,
    DEFAULT_MIN_REVISION_ACCEL,
    DEFAULT_MIN_UP7D,
    DEFAULT_TARGET_LOOKBACK_DAYS,
    DEFAULT_MIN_TARGET_RAISERS,
    DEFAULT_MIN_TARGET_RAISE_PCT,
    DEFAULT_MIN_DAILY_VOLUME,
    DEFAULT_MIN_ETF_ASSETS,
    DEFAULT_SQUEEZE_BARS,
    DEFAULT_SQUEEZE_MAX_RANGE_PCT,
    DEFAULT_SQUEEZE_BREAKOUT_PCT,
    DEFAULT_SQUEEZE_DIRECTION,
    DEFAULT_SQUEEZE_TIMEFRAME,
    SQUEEZE_TIMEFRAMES,
    DEFAULT_SQUEEZE_MAX_LOOKBACK,
    DEFAULT_SQUEEZE_MIN_BARS,
    MIN_DAILY_GAIN_PCT,
    MIN_VOL_RATIO,
)


HIDDEN = {"display": "none"}
VISIBLE = {"display": "block", "marginTop": "12px"}
LABEL_STYLE = {"fontWeight": "600", "marginRight": "8px", "color": "#334155"}
ROW_STYLE = {"marginBottom": "8px"}

CARD_STYLE = {
    "background": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)",
    "border": "1px solid #e2e8f0",
}

TABLE_STYLE_CELL = {
    "padding": "10px",
    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
    "fontSize": "13px",
    "border": "none",
    "borderBottom": "1px solid #f1f5f9",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": "#1e293b",
    "color": "white",
    "fontWeight": "600",
    "fontSize": "12px",
    "letterSpacing": "0.05em",
    "textTransform": "uppercase",
    "border": "none",
    "padding": "12px 10px",
}


def _labelled(label: str, child):
    return html.Div([html.Label(label, style=LABEL_STYLE), child], style=ROW_STYLE)


def _build_startup_overlay():
    return html.Div(
        id="startup-overlay",
        style={
            "position": "fixed", "top": 0, "left": 0,
            "width": "100vw", "height": "100vh",
            "background": "rgba(248, 250, 252, 0.97)",
            "zIndex": "9999",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "backdropFilter": "blur(4px)",
        },
        children=[
            dcc.Loading(
                type="circle", color="#3b82f6",
                children=html.Div(
                    style={
                        "textAlign": "center", "maxWidth": "560px", "padding": "32px",
                        "background": "white", "borderRadius": "12px",
                        "boxShadow": "0 10px 25px rgba(0,0,0,0.1)",
                        "border": "1px solid #e2e8f0",
                    },
                    children=[
                        html.H2("Warming up", style={
                            "color": "#0f172a", "letterSpacing": "-0.02em", "marginTop": "16px",
                        }),
                        html.P(
                            "Fetching the US ticker list, market caps, and 30-day min daily "
                            "volumes from Yahoo Finance. The browser will stay on this screen "
                            "while we populate the on-disk caches.",
                            style={"color": "#475569", "lineHeight": "1.6"},
                        ),
                        html.P(
                            "First startup of the day takes ~1-2 minutes. Subsequent runs within "
                            "24 hours are instant (data cached on disk).",
                            style={"color": "#94a3b8", "fontSize": "13px",
                                   "marginTop": "16px", "lineHeight": "1.5"},
                        ),
                        # The hidden child below is what dcc.Loading watches for the spinner.
                        html.Div(id="startup-sink", style={"display": "none"}),
                    ],
                ),
            ),
        ],
    )


def _build_layout():
    return html.Div(
        style={
            "maxWidth": "1100px",
            "margin": "20px auto",
            "padding": "0 16px",
            "fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "color": "#0f172a",
        },
        children=[
            _build_startup_overlay(),
            dcc.Store(id="startup-trigger", data="init"),
            html.H1("US Stock Scanner", style={
                "marginBottom": "24px", "color": "#0f172a", "letterSpacing": "-0.02em",
            }),
            html.Div(
                style=CARD_STYLE,
                children=[
                    _labelled(
                        "Scan mode",
                        dcc.Dropdown(
                            id="mode",
                            options=[
                                {"label": "Uptrend (price + volume breakout)", "value": "uptrend"},
                                {"label": "Earnings (surprise beat)", "value": "earnings"},
                                {"label": "Revisions (upward EPS revision acceleration)", "value": "revisions"},
                                {"label": "Targets (price-target hikes)", "value": "targets"},
                                {"label": "Squeeze (tight consolidation → breakout)", "value": "squeeze"},
                            ],
                            value="uptrend",
                            clearable=False,
                            style={"width": "420px"},
                        ),
                    ),
                    _labelled(
                        "Min market cap (e.g. 5B, 100M, leave blank for no filter)",
                        dcc.Input(
                            id="market-cap", type="text", value="5B",
                            placeholder="5B", style={"width": "120px"},
                        ),
                    ),
                    _labelled(
                        "Min daily volume in last 30 trading days (shares; 0 = no filter)",
                        dcc.Input(
                            id="min-daily-volume", type="number",
                            value=DEFAULT_MIN_DAILY_VOLUME, step=10000,
                            min=0, style={"width": "140px"},
                        ),
                    ),
                    _labelled(
                        "Min ETF net assets (ETFs always included; raise to e.g. 999T to exclude)",
                        dcc.Input(
                            id="min-etf-assets", type="text",
                            value="15B", style={"width": "120px"},
                        ),
                    ),
                    _labelled(
                        "Date range (only used by uptrend / earnings / squeeze)",
                        dcc.DatePickerRange(
                            id="date-range",
                            display_format="YYYY-MM-DD",
                            clearable=True,
                        ),
                    ),

                    # --- Mode-specific parameters --------------------------------
                    html.Div(
                        id="uptrend-params",
                        children=[
                            html.H4("Uptrend parameters"),
                            _labelled(
                                "Min daily gain %",
                                dcc.Input(id="min-gain", type="number",
                                          value=MIN_DAILY_GAIN_PCT, step=0.5, style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min volume ratio (today vs 20d avg)",
                                dcc.Input(id="min-vol-ratio", type="number",
                                          value=MIN_VOL_RATIO, step=0.1, style={"width": "100px"}),
                            ),
                        ],
                    ),
                    html.Div(
                        id="earnings-params",
                        children=[
                            html.H4("Earnings parameters"),
                            _labelled(
                                "Min EPS surprise %",
                                dcc.Input(id="min-surprise", type="number",
                                          value=DEFAULT_MIN_SURPRISE_PCT, step=1.0, style={"width": "100px"}),
                            ),
                        ],
                    ),
                    html.Div(
                        id="revisions-params",
                        children=[
                            html.H4("Revisions parameters"),
                            _labelled(
                                "Min acceleration (7d pace / 30d pace)",
                                dcc.Input(id="min-accel", type="number",
                                          value=DEFAULT_MIN_REVISION_ACCEL, step=0.1, style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min upward revisions in last 7d",
                                dcc.Input(id="min-up7d", type="number",
                                          value=DEFAULT_MIN_UP7D, step=1, style={"width": "100px"}),
                            ),
                        ],
                    ),
                    html.Div(
                        id="squeeze-params",
                        children=[
                            html.H4("Squeeze parameters"),
                            _labelled(
                                "Max consolidation bars (before breakout)",
                                dcc.Input(id="squeeze-bars-input", type="number",
                                          value=DEFAULT_SQUEEZE_BARS, step=1, min=2,
                                          style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min consolidation bars (shortest stretch we'll accept)",
                                dcc.Input(id="squeeze-min-bars-input", type="number",
                                          value=DEFAULT_SQUEEZE_MIN_BARS, step=1, min=2,
                                          style={"width": "100px"}),
                            ),
                            _labelled(
                                "Max bars ago the breakout can be (0 = strictly last bar)",
                                dcc.Input(id="squeeze-max-lookback-input", type="number",
                                          value=DEFAULT_SQUEEZE_MAX_LOOKBACK, step=1, min=0,
                                          style={"width": "100px"}),
                            ),
                            _labelled(
                                "Max consolidation range %",
                                dcc.Input(id="squeeze-max-range-input", type="number",
                                          value=DEFAULT_SQUEEZE_MAX_RANGE_PCT, step=0.5,
                                          style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min breakout %",
                                dcc.Input(id="squeeze-breakout-input", type="number",
                                          value=DEFAULT_SQUEEZE_BREAKOUT_PCT, step=0.5,
                                          style={"width": "100px"}),
                            ),
                            _labelled(
                                "Direction",
                                dcc.Dropdown(
                                    id="squeeze-direction-input",
                                    options=[
                                        {"label": "Both", "value": "both"},
                                        {"label": "Up (breakout above)", "value": "up"},
                                        {"label": "Down (breakdown below)", "value": "down"},
                                    ],
                                    value=DEFAULT_SQUEEZE_DIRECTION,
                                    clearable=False,
                                    style={"width": "240px"},
                                ),
                            ),
                            _labelled(
                                "Timeframe (candle interval)",
                                dcc.Dropdown(
                                    id="squeeze-timeframe-input",
                                    options=[{"label": tf, "value": tf} for tf in SQUEEZE_TIMEFRAMES],
                                    value=DEFAULT_SQUEEZE_TIMEFRAME,
                                    clearable=False,
                                    style={"width": "140px"},
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        id="targets-params",
                        children=[
                            html.H4("Targets parameters"),
                            _labelled(
                                "Lookback window (days)",
                                dcc.Input(id="target-lookback", type="number",
                                          value=DEFAULT_TARGET_LOOKBACK_DAYS, step=1, style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min distinct firms raising",
                                dcc.Input(id="min-raisers", type="number",
                                          value=DEFAULT_MIN_TARGET_RAISERS, step=1, style={"width": "100px"}),
                            ),
                            _labelled(
                                "Min median raise %",
                                dcc.Input(id="min-raise-pct", type="number",
                                          value=DEFAULT_MIN_TARGET_RAISE_PCT, step=1.0, style={"width": "100px"}),
                            ),
                        ],
                    ),

                    html.Button(
                        "Run scan",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "16px", "padding": "10px 24px",
                            "background": "#2563eb", "color": "white",
                            "border": "none", "borderRadius": "6px",
                            "fontSize": "15px", "cursor": "pointer",
                        },
                    ),
                ],
            ),

            dcc.Store(id="scan-trigger"),
            dcc.Interval(id="progress-poll", interval=500, disabled=False),
            html.Div(id="status", style={"margin": "16px 0", "fontWeight": "600", "color": "#475569"}),
            html.Div(id="progress-display", style={
                "margin": "0 0 12px 0", "fontSize": "13px", "color": "#3b82f6",
                "fontFamily": "ui-monospace, SFMono-Regular, monospace",
            }),
            html.Div(style=CARD_STYLE, children=[
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=dash_table.DataTable(
                        id="results-table",
                        data=[],
                        columns=[],
                        page_size=50,
                        sort_action="native",
                        fixed_rows={"headers": True},
                        style_table={
                            "overflowX": "auto", "overflowY": "auto",
                            "maxHeight": "70vh", "borderRadius": "8px",
                        },
                        style_cell=TABLE_STYLE_CELL,
                        style_header=TABLE_STYLE_HEADER,
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
                        ],
                    ),
                ),
            ]),

            # ============================================================
            # Upcoming dates section (earnings + economic releases)
            # ============================================================
            html.Hr(style={"margin": "32px 0", "border": "none", "borderTop": "1px solid #cbd5e1"}),
            html.H2("Upcoming Important Dates", style={"color": "#0f172a", "letterSpacing": "-0.02em"}),
            html.Div(
                style=CARD_STYLE,
                children=[
                    _labelled(
                        "Days ahead to scan",
                        dcc.Input(id="cal-days", type="number", value=30, step=1,
                                  min=1, max=180, style={"width": "100px"}),
                    ),
                    _labelled(
                        "Earnings: min market cap (uses same syntax as main scan)",
                        dcc.Input(id="cal-market-cap", type="text", value="5B",
                                  style={"width": "120px"}),
                    ),
                    html.Button(
                        "Refresh calendars",
                        id="cal-refresh-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "12px", "padding": "10px 24px",
                            "background": "#16a34a", "color": "white",
                            "border": "none", "borderRadius": "6px",
                            "fontSize": "15px", "cursor": "pointer",
                        },
                    ),
                ],
            ),
            html.Div(id="cal-status", style={"margin": "12px 0", "fontWeight": "600", "color": "#475569"}),

            html.H3("Economic releases", style={"marginTop": "16px", "color": "#334155"}),
            html.Div(style=CARD_STYLE, children=[
                dash_table.DataTable(
                    id="econ-table", data=[], columns=[], page_size=30,
                    sort_action="native",
                    fixed_rows={"headers": True},
                    style_table={"overflowY": "auto", "maxHeight": "60vh"},
                    style_cell=TABLE_STYLE_CELL,
                    style_header=TABLE_STYLE_HEADER,
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
                    ],
                ),
            ]),

            html.H3("Earnings calendar (large caps)", style={"marginTop": "24px", "color": "#334155"}),
            html.Div(style=CARD_STYLE, children=[
                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="earnings-cal-table", data=[], columns=[], page_size=50,
                        sort_action="native",
                        fixed_rows={"headers": True},
                        style_table={
                            "overflowX": "auto", "overflowY": "auto", "maxHeight": "70vh",
                        },
                        style_cell=TABLE_STYLE_CELL,
                        style_header=TABLE_STYLE_HEADER,
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
                        ],
                    ),
                ),
            ]),
        ],
    )


app = dash.Dash(__name__)
app.title = "US Stock Scanner"
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                min-height: 100vh;
                background:
                  radial-gradient(at 20% 0%, #dbeafe 0px, transparent 50%),
                  radial-gradient(at 80% 0%, #fae8ff 0px, transparent 50%),
                  radial-gradient(at 80% 100%, #cffafe 0px, transparent 50%),
                  linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                background-attachment: fixed;
                color: #0f172a;
            }
            input[type=text], input[type=number] {
                border: 1px solid #cbd5e1 !important;
                border-radius: 6px !important;
                padding: 6px 10px !important;
                font-size: 14px !important;
            }
            input:focus { outline: 2px solid #3b82f6 !important; outline-offset: -1px; }
            button:hover { filter: brightness(1.08); }
            button:active { transform: translateY(1px); }
            /* DatePickerRange — kill the wide auto-stretched container + clearbutton reservation */
            .DateRangePicker { display: inline-block !important; width: auto !important; }
            .DateRangePickerInput {
                border: 1px solid #cbd5e1 !important;
                border-radius: 6px !important;
                background: white !important;
                padding: 0 !important;
                display: inline-flex !important;
                align-items: center !important;
            }
            .DateInput { width: 120px !important; background: transparent !important; }
            .DateInput_input {
                font-size: 14px !important;
                padding: 6px 10px !important;
                border-bottom: none !important;
                background: transparent !important;
            }
            .DateRangePickerInput_arrow { padding: 0 6px !important; }
            .DateRangePickerInput_clearDates,
            .DateRangePickerInput_clearDates_default { display: none !important; }
            .DateRangePickerInput_calendarIcon { display: none !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
app.layout = _build_layout()


@app.callback(
    Output("uptrend-params", "style"),
    Output("earnings-params", "style"),
    Output("revisions-params", "style"),
    Output("targets-params", "style"),
    Output("squeeze-params", "style"),
    Input("mode", "value"),
)
def _toggle_param_panels(mode):
    return (
        VISIBLE if mode == "uptrend" else HIDDEN,
        VISIBLE if mode == "earnings" else HIDDEN,
        VISIBLE if mode == "revisions" else HIDDEN,
        VISIBLE if mode == "targets" else HIDDEN,
        VISIBLE if mode == "squeeze" else HIDDEN,
    )


def _missing_caches() -> list[str]:
    """Names of caches that are missing or stale and need warming."""
    missing = []
    if _load_ticker_cache(TICKER_CACHE_TTL_SECONDS) is None:
        missing.append("US ticker list (NASDAQ Trader)")
    if _load_quote_metrics_cache(MARKETCAP_CACHE_TTL_SECONDS) is None:
        missing.append("market caps")
    if _load_min_volume_cache(MIN_VOLUME_CACHE_TTL_SECONDS) is None:
        missing.append("30-day min daily volumes")
    return missing


@app.callback(
    Output("startup-overlay", "style"),
    Output("startup-sink", "children"),
    Input("startup-trigger", "data"),
    prevent_initial_call=False,
)
def _startup_warmup(_):
    """On first page load: check cache freshness, warm anything stale, then hide overlay."""
    missing = _missing_caches()
    if not missing:
        return {"display": "none"}, "done"

    try:
        default_cap = _parse_market_cap("5B")
        _prepare_universe(
            refresh_tickers=False,
            min_market_cap_usd=default_cap,
            refresh_market_caps=False,
            min_daily_volume=200_000,
        )
    except Exception:
        # Hide overlay anyway so user can interact; errors will surface on Run.
        pass
    return {"display": "none"}, "done"


@app.callback(
    Output("progress-display", "children"),
    Input("progress-poll", "n_intervals"),
)
def _update_progress(_):
    p = _SCAN_PROGRESS
    total = p.get("total") or 0
    current = p.get("current") or 0
    desc = p.get("desc") or ""
    if total <= 0 or current >= total:
        return ""
    pct = current / total * 100
    bar_width = 24
    filled = int(bar_width * current / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"⏳ {desc}  {bar}  {current:,} / {total:,}  ({pct:5.1f}%)"


_MODE_DESCRIPTIONS = {
    "uptrend":   "early-uptrend setups",
    "earnings":  "earnings surprises",
    "revisions": "upward revision acceleration",
    "targets":   "price-target hikes",
    "squeeze":   "tight-consolidation breakouts",
}


@app.callback(
    Output("scan-trigger", "data"),
    Output("status", "children"),
    Input("run-btn", "n_clicks"),
    State("mode", "value"),
    State("market-cap", "value"),
    State("min-daily-volume", "value"),
    State("min-etf-assets", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("min-gain", "value"),
    State("min-vol-ratio", "value"),
    State("min-surprise", "value"),
    State("min-accel", "value"),
    State("min-up7d", "value"),
    State("target-lookback", "value"),
    State("min-raisers", "value"),
    State("min-raise-pct", "value"),
    State("squeeze-bars-input", "value"),
    State("squeeze-min-bars-input", "value"),
    State("squeeze-max-lookback-input", "value"),
    State("squeeze-max-range-input", "value"),
    State("squeeze-breakout-input", "value"),
    State("squeeze-direction-input", "value"),
    State("squeeze-timeframe-input", "value"),
    prevent_initial_call=True,
)
def _prep_scan(n_clicks, mode, market_cap_str, min_daily_volume,
               min_etf_assets_str,
               start_date, end_date,
               min_gain, min_vol_ratio,
               min_surprise, min_accel, min_up7d,
               target_lookback, min_raisers, min_raise_pct,
               squeeze_bars, squeeze_min_bars, squeeze_max_lookback,
               squeeze_max_range, squeeze_breakout, squeeze_direction, squeeze_timeframe):
    """Fast: parse params, prep universe, show descriptive status, hand off to executor."""
    cap_usd = None
    if market_cap_str and market_cap_str.strip():
        try:
            cap_usd = _parse_market_cap(market_cap_str)
        except Exception as e:
            return no_update, f"❌ Invalid market cap: {e}"

    vol_filter = float(min_daily_volume) if min_daily_volume and float(min_daily_volume) > 0 else None

    etf_assets_filter = DEFAULT_MIN_ETF_ASSETS
    if min_etf_assets_str and min_etf_assets_str.strip():
        try:
            etf_assets_filter = _parse_market_cap(min_etf_assets_str)
        except Exception as e:
            return no_update, f"❌ Invalid min ETF assets: {e}"

    try:
        tickers = _prepare_universe(
            refresh_tickers=False,
            min_market_cap_usd=cap_usd,
            refresh_market_caps=False,
            min_daily_volume=vol_filter,
            min_etf_assets=etf_assets_filter,
        )
    except Exception as e:
        return no_update, f"❌ Failed to load universe: {e}"

    # Build a descriptive status that mirrors the CLI's "Scanning N tickers..." line
    if mode in ("uptrend", "earnings", "squeeze"):
        if start_date and end_date:
            window = f"using window {start_date} → {end_date}"
        else:
            window = "(previous trading day)"
        status = f"⏳ Scanning {len(tickers)} tickers {window} for {_MODE_DESCRIPTIONS[mode]}..."
    else:
        status = f"⏳ Scanning {len(tickers)} tickers for {_MODE_DESCRIPTIONS[mode]}..."

    payload = {
        "mode": mode,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "min_gain": min_gain,
        "min_vol_ratio": min_vol_ratio,
        "min_surprise": min_surprise,
        "min_accel": min_accel,
        "min_up7d": min_up7d,
        "target_lookback": target_lookback,
        "min_raisers": min_raisers,
        "min_raise_pct": min_raise_pct,
        "squeeze_bars": squeeze_bars,
        "squeeze_min_bars": squeeze_min_bars,
        "squeeze_max_lookback": squeeze_max_lookback,
        "squeeze_max_range": squeeze_max_range,
        "squeeze_breakout": squeeze_breakout,
        "squeeze_direction": squeeze_direction,
        "squeeze_timeframe": squeeze_timeframe,
    }
    return payload, status


@app.callback(
    Output("results-table", "data"),
    Output("results-table", "columns"),
    Output("status", "children", allow_duplicate=True),
    Input("scan-trigger", "data"),
    prevent_initial_call=True,
)
def _execute_scan(trigger):
    """Slow: actually runs the scan. Triggered when _prep_scan writes to the store."""
    if not trigger:
        return no_update, no_update, no_update

    mode = trigger["mode"]
    tickers = trigger["tickers"]

    scanner.SCAN_START = dt.date.fromisoformat(trigger["start_date"]) if trigger["start_date"] else None
    scanner.SCAN_END = dt.date.fromisoformat(trigger["end_date"]) if trigger["end_date"] else None
    if mode == "uptrend":
        scanner.MIN_DAILY_GAIN_PCT = float(trigger["min_gain"])
        scanner.MIN_VOL_RATIO = float(trigger["min_vol_ratio"])

    try:
        if mode == "uptrend":
            df = scan_uptrend(tickers)
        elif mode == "earnings":
            df = scan_earnings(tickers, float(trigger["min_surprise"]))
        elif mode == "revisions":
            df = scan_revisions(tickers, float(trigger["min_accel"]), int(trigger["min_up7d"]))
        elif mode == "targets":
            df = scan_target_hikes(
                tickers,
                int(trigger["target_lookback"]),
                int(trigger["min_raisers"]),
                float(trigger["min_raise_pct"]),
            )
        elif mode == "squeeze":
            df = scan_squeeze(
                tickers,
                int(trigger["squeeze_bars"]),
                float(trigger["squeeze_max_range"]),
                float(trigger["squeeze_breakout"]),
                trigger["squeeze_direction"],
                trigger["squeeze_timeframe"],
                max_lookback=int(trigger["squeeze_max_lookback"]),
                min_consol_bars=int(trigger["squeeze_min_bars"]),
            )
        else:
            return [], [], f"Unknown mode: {mode}"
    except Exception:
        return no_update, no_update, f"❌ Scan failed:\n{traceback.format_exc()}"

    if df is None or df.empty:
        return [], [], f"{mode} scan: no matches (universe size: {len(tickers)})"

    records = df.astype(object).where(df.notna(), None).to_dict("records")
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, dt.date):
                r[k] = v.isoformat()
    columns = [{"name": c, "id": c} for c in df.columns]
    return records, columns, f"✅ {mode} scan: {len(df)} match(es) (universe size: {len(tickers)})"


@app.callback(
    Output("econ-table", "data"),
    Output("econ-table", "columns"),
    Output("earnings-cal-table", "data"),
    Output("earnings-cal-table", "columns"),
    Output("cal-status", "children"),
    Input("cal-refresh-btn", "n_clicks"),
    State("cal-days", "value"),
    State("cal-market-cap", "value"),
    prevent_initial_call=True,
)
def _refresh_calendars(n_clicks, days_ahead, market_cap_str):
    days = int(days_ahead or 30)

    # --- Economic releases (instant) ---
    econ = compute_economic_releases(days)
    econ_records = [{"Date": e["Date"].isoformat(), "Event": e["Event"], "Source": e["Source"]}
                    for e in econ]
    econ_columns = [{"name": c, "id": c} for c in ("Date", "Event", "Source")]

    # --- Earnings calendar (slow — iterates large-caps) ---
    try:
        cap_usd = _parse_market_cap(market_cap_str) if market_cap_str and market_cap_str.strip() else None
    except Exception as e:
        return econ_records, econ_columns, [], [], f"❌ Invalid market cap: {e}"

    try:
        tickers = _prepare_universe(
            refresh_tickers=False,
            min_market_cap_usd=cap_usd,
            refresh_market_caps=False,
            min_daily_volume=DEFAULT_MIN_DAILY_VOLUME,
        )
        earnings_rows = get_upcoming_earnings(tickers, days_ahead=days)
    except Exception as e:
        return econ_records, econ_columns, [], [], f"❌ Earnings calendar failed: {e}"

    earnings_records = [
        {
            "Date": r["Date"].isoformat(),
            "Ticker": r["Ticker"],
            "EPS Estimate": r["EPS Estimate"],
            "Is Estimated Date": "yes" if r["Is Estimated Date"] else "no",
        }
        for r in earnings_rows
    ]
    earnings_columns = [{"name": c, "id": c}
                       for c in ("Date", "Ticker", "EPS Estimate", "Is Estimated Date")]

    status = (
        f"✅ {len(econ_records)} economic release(s), "
        f"{len(earnings_records)} earnings report(s) "
        f"in next {days} days from a universe of {len(tickers)} tickers."
    )
    return econ_records, econ_columns, earnings_records, earnings_columns, status


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)
