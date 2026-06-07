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

import algorithmic_trading.analysis.us_stock_scanner as scanner
from algorithmic_trading.analysis.us_stock_scanner import (
    _parse_market_cap,
    _prepare_universe,
    scan_uptrend,
    scan_earnings,
    scan_revisions,
    scan_target_hikes,
    compute_economic_releases,
    get_upcoming_earnings,
    DEFAULT_MIN_SURPRISE_PCT,
    DEFAULT_MIN_REVISION_ACCEL,
    DEFAULT_MIN_UP7D,
    DEFAULT_TARGET_LOOKBACK_DAYS,
    DEFAULT_MIN_TARGET_RAISERS,
    DEFAULT_MIN_TARGET_RAISE_PCT,
    DEFAULT_MIN_AVG_VOLUME,
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
                        "Min 3-month avg daily volume (shares; 0 = no filter)",
                        dcc.Input(
                            id="min-avg-volume", type="number",
                            value=DEFAULT_MIN_AVG_VOLUME, step=10000,
                            min=0, style={"width": "140px"},
                        ),
                    ),
                    _labelled(
                        "Date range (only used by uptrend / earnings)",
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
            html.Div(id="status", style={"margin": "16px 0", "fontWeight": "600", "color": "#475569"}),
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
                        style_table={"overflowX": "auto", "borderRadius": "8px"},
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
                        style_table={"overflowX": "auto"},
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
    Input("mode", "value"),
)
def _toggle_param_panels(mode):
    return (
        VISIBLE if mode == "uptrend" else HIDDEN,
        VISIBLE if mode == "earnings" else HIDDEN,
        VISIBLE if mode == "revisions" else HIDDEN,
        VISIBLE if mode == "targets" else HIDDEN,
    )


_MODE_DESCRIPTIONS = {
    "uptrend":   "early-uptrend setups",
    "earnings":  "earnings surprises",
    "revisions": "upward revision acceleration",
    "targets":   "price-target hikes",
}


@app.callback(
    Output("scan-trigger", "data"),
    Output("status", "children"),
    Input("run-btn", "n_clicks"),
    State("mode", "value"),
    State("market-cap", "value"),
    State("min-avg-volume", "value"),
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
    prevent_initial_call=True,
)
def _prep_scan(n_clicks, mode, market_cap_str, min_avg_volume, start_date, end_date,
               min_gain, min_vol_ratio,
               min_surprise, min_accel, min_up7d,
               target_lookback, min_raisers, min_raise_pct):
    """Fast: parse params, prep universe, show descriptive status, hand off to executor."""
    cap_usd = None
    if market_cap_str and market_cap_str.strip():
        try:
            cap_usd = _parse_market_cap(market_cap_str)
        except Exception as e:
            return no_update, f"❌ Invalid market cap: {e}"

    vol_filter = float(min_avg_volume) if min_avg_volume and float(min_avg_volume) > 0 else None

    try:
        tickers = _prepare_universe(
            refresh_tickers=False,
            min_market_cap_usd=cap_usd,
            refresh_market_caps=False,
            min_avg_volume=vol_filter,
        )
    except Exception as e:
        return no_update, f"❌ Failed to load universe: {e}"

    # Build a descriptive status that mirrors the CLI's "Scanning N tickers..." line
    if mode in ("uptrend", "earnings"):
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
            min_avg_volume=DEFAULT_MIN_AVG_VOLUME,
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
