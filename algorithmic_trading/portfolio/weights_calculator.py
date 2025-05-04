import argparse
import yfinance as yf
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stock analysis and optionally save results to Excel."
    )

    parser.add_argument(
        "--save_to_excel",
        action="store_true",
        default=False,
        help="If provided, saves the output DataFrame to an Excel file",
    )

    parser.add_argument(
        "--tickers_file",
        type=str,
        default=None,
        help="If provided, read tickers from a file",
    )

    return parser.parse_args()


def get_tickers(path):
    """Read tickers from a local file."""
    if path:
        print("Reading tickers from the specified file!")
        with open(path, "r") as f:
            return f.read().split("\n")[:-1]
    else:
        print("No tickers file is provided, returning default tickers!")
        return [
            "MSFT",
            "META",
            "NVDA",
            "AMZN",
            "GOOGL",
            "LB",
            "AMD",
            "FIX",
            "ASML",
            "PLTR",
            "RKLB",
            "ADBE",
            "SOFI",
            "UBER",
        ]


def calculate_metrics(tickers):
    results = []

    # for ticker in ["FIX"]:
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            balance = stock.balance_sheet
            cashflow = stock.cashflow

            price = info.get("currentPrice")
            market_cap = info.get("marketCap")
            total_debt = info.get("totalDebt")
            cash = info.get("totalCash")

            pe_ratio = info.get("forwardPE")
            eps_current = (
                financials.loc["Basic EPS"].iloc[0]
                if not np.isnan(financials.loc["Basic EPS"].iloc[0])
                else financials.loc["Basic EPS"].iloc[1]
            )
            eps_past = (
                financials.loc["Basic EPS"].iloc[1]
                if not np.isnan(financials.loc["Basic EPS"].iloc[0])
                else financials.loc["Basic EPS"].iloc[2]
            )
            eps_growth = (
                (eps_current - eps_past) / eps_past
                if eps_current and eps_past
                else None
            )
            peg_ratio = pe_ratio / (eps_growth * 100) if pe_ratio and eps_growth else 0

            revenue = (
                financials.loc["Total Revenue"].iloc[0]
                if "Total Revenue" in financials.index
                else None
            )
            prev_revenue = (
                financials.loc["Total Revenue"].iloc[1]
                if "Total Revenue" in financials.index and len(financials.columns) > 1
                else None
            )
            revenue_growth = (
                (revenue - prev_revenue) / prev_revenue
                if revenue and prev_revenue
                else None
            )

            net_income = (
                financials.loc["Net Income"].iloc[0]
                if "Net Income" in financials.index
                else None
            )
            net_margin = net_income / revenue if net_income and revenue else None

            fcf = (
                cashflow.loc["Free Cash Flow"].iloc[0]
                if "Free Cash Flow" in cashflow.index
                else None
            )
            fcf_prev = (
                cashflow.loc["Free Cash Flow"].iloc[1]
                if "Free Cash Flow" in cashflow.index
                else None
            )
            fcf_growth = (fcf - fcf_prev) / fcf_prev if fcf and fcf_prev else None
            fcf_margin = fcf / revenue if fcf and revenue else None

            roe = info.get("returnOnEquity")

            results.append(
                {
                    "Ticker": ticker,
                    "Forward PE": pe_ratio,
                    "PEG Ratio": peg_ratio,
                    "Net Margin": net_margin,
                    "FCF Margin": fcf_margin,
                    "ROE": roe,
                    "EPS Growth": eps_growth,
                    "Revenue Growth": revenue_growth,
                    "FCF Growth": fcf_growth,
                }
            )

        except Exception as e:
            results.append({"Ticker": ticker, "Error": str(e)})

    return pd.DataFrame(results)


# Scoring Function using ROE instead of ROIC
def compute_scores(df):
    df_ranked = df.copy()
    metrics = {
        "Value": ["Forward PE", "PEG Ratio"],
        "Profitability": ["Net Margin", "FCF Margin", "ROE"],
        "Growth": ["EPS Growth", "Revenue Growth", "FCF Growth"],
    }

    for metric_group, cols in metrics.items():
        ranks = []
        for col in cols:
            if col in df_ranked.columns:
                if metric_group == "Value":
                    ranks.append(df_ranked[col].rank(ascending=False))  # Lower is better
                else:
                    ranks.append(
                        df_ranked[col].rank(ascending=True)
                    )  # Higher is better
        df_ranked[f"{metric_group} Score"] = sum(ranks) / len(ranks)

    # Final Composite Score
    df_ranked["Composite Score"] = (
        df_ranked["Value Score"] * 0.4
        + df_ranked["Profitability Score"] * 0.3
        + df_ranked["Growth Score"] * 0.3
    )

    # Normalize weights
    df_ranked["Weight"] = (
        df_ranked["Composite Score"] / df_ranked["Composite Score"].sum()
    ) * 100
    df_ranked.sort_values(by="Weight", ascending=False, inplace=True)
    df_ranked.reset_index(inplace=True)

    return df_ranked


if __name__ == "__main__":
    args = parse_args()
    tickers = get_tickers(args.tickers_file)
    df = calculate_metrics(tickers)
    df_scored = compute_scores(df)
    print(
        df_scored[
            [
                "Ticker",
                "Forward PE",
                "PEG Ratio",
                "Net Margin",
                "FCF Margin",
                "ROE",
                "EPS Growth",
                "Revenue Growth",
                "FCF Growth",
                "Value Score",
                "Profitability Score",
                "Growth Score",
                "Composite Score",
                "Weight",
            ]
        ]
    )

    if args.save_to_excel:
        df_scored.to_excel("scored_quantitative_portfolio.xlsx", index=False)
