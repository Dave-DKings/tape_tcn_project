"""
Utility script to build quarterly fundamental features using Alpha Vantage.
Restored from original implementation but with strict rate limiting to respect
the 5 requests/minute burst limit.

The script downloads quarterly statements, computes FCFE, revenue
and NCFO deltas (plus FCFE sign) per the formulas described in the Tesi thesis,
and writes a tidy CSV of the form:

    Date, Ticker, FCFE, Revenue, NCFO

along with the delta columns and FCFE sign. The CSV can then be referenced by
`config.py` via `feature_params["fundamental_features"]["data_path"]`.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    # Prefer relative import when executed from project root.
    from .config import (  # type: ignore
        ASSET_TICKERS,
        FUNDAMENTAL_FEATURES_CONFIG,
        ALPHA_VANTAGE_API_KEY,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from config import ASSET_TICKERS, FUNDAMENTAL_FEATURES_CONFIG, ALPHA_VANTAGE_API_KEY  # type: ignore

logger = logging.getLogger("fundamental_feature_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EPS = 1e-9
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Global request counter for rate limiting
REQUEST_COUNT = 0

def _rate_limit_check():
    """Ensure we don't exceed 5 requests per minute."""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    if REQUEST_COUNT > 0 and REQUEST_COUNT % 5 == 0:
        logger.info(f"Request count {REQUEST_COUNT}: Pausing for 60 seconds to respect burst limit...")
        time.sleep(60)

def _safe_get(series: pd.Series, *candidates: str) -> pd.Series:
    """
    Retrieve a row from a statement DataFrame using multiple candidate keys.
    Returns an empty Series if none of the keys are present.
    """
    for key in candidates:
        if key in series.index:
            return series.loc[key]
    return pd.Series(dtype="float64")

def _convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame values are floats."""
    return df.apply(pd.to_numeric, errors="coerce")

def _compute_working_capital_change(balance_sheet_t: pd.DataFrame) -> pd.Series:
    """
    Compute change in working capital from the balance sheet when the cash-flow
    statement does not provide it.
    """
    current_assets = _safe_get(balance_sheet_t, "Total Current Assets")
    current_liabilities = _safe_get(balance_sheet_t, "Total Current Liabilities")
    if current_assets.empty or current_liabilities.empty:
        return pd.Series(dtype="float64")
    working_capital = current_assets - current_liabilities
    return working_capital.diff()

def _compute_net_borrowings(cashflow_t: pd.DataFrame) -> pd.Series:
    """Prefer Net Borrowings line; fall back to debt issuance minus repayments."""
    net_borrowings = _safe_get(cashflow_t, "Net Borrowings")
    if not net_borrowings.empty:
        return net_borrowings
    issuance = _safe_get(
        cashflow_t,
        "Issuance Of Debt",
        "Issuance Of Long Term Debt",
        "Proceeds From Issuance Of Long Term Debt",
    )
    repayments = _safe_get(
        cashflow_t,
        "Repayment Of Debt",
        "Repayments Of Long Term Debt",
        "Repayments Of Long Term Capital Leases",
    )
    if issuance.empty and repayments.empty:
        return pd.Series(dtype="float64")
    issuance = issuance.fillna(0.0)
    repayments = repayments.fillna(0.0)
    return issuance - repayments

def _get_operating_cf(cashflow_t: pd.DataFrame) -> pd.Series:
    candidates = [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Net Cash Provided By Operating Activities",
    ]
    return _safe_get(cashflow_t, *candidates)

@dataclass
class FundamentalSeries:
    ticker: str
    frame: pd.DataFrame

def call_alpha_vantage(function: str, symbol: str) -> Dict:
    # 1. Rate Limit Check
    _rate_limit_check()

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    
    logger.info(f"Fetching {function} for {symbol}...")
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if "Note" in data:
        logger.warning(f"Alpha Vantage Note: {data['Note']}")
        # Often notes indicate usage limits, but we are handling it with sleep.
    
    if "quarterlyReports" not in data:
        # Check for Information (Rate Limit)
        if "Information" in data:
             logger.error(f"Alpha Vantage Rate Limit Hit: {data['Information']}")
        raise ValueError(f"Unexpected Alpha Vantage payload for {symbol} ({function}): {data}")
        
    return data["quarterlyReports"]

def reports_to_dataframe(reports: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(reports)
    # Filter dates? 
    # 'fiscalDateEnding' is the index.
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df = df.set_index("fiscalDateEnding")
    df = _convert_to_numeric(df)
    return df

def build_fundamental_frame(ticker: str) -> Optional[FundamentalSeries]:
    """
    Download quarterly statements for a ticker and compute FCFE / revenue / NCFO
    along with their deltas.
    """
    try:
        income_reports = call_alpha_vantage("INCOME_STATEMENT", ticker)
        balance_reports = call_alpha_vantage("BALANCE_SHEET", ticker)
        cash_reports = call_alpha_vantage("CASH_FLOW", ticker)
    except Exception as exc:
        logger.warning("Failed to fetch Alpha Vantage data for %s: %s", ticker, exc)
        return None

    income_t = reports_to_dataframe(income_reports)
    balance_t = reports_to_dataframe(balance_reports)
    cash_flow_t = reports_to_dataframe(cash_reports)

    net_income = income_t.get("netIncome")
    depreciation = cash_flow_t.get("depreciationAndAmortization")
    if depreciation is None:
        depreciation = cash_flow_t.get("depreciation")
    capex = cash_flow_t.get("capitalExpenditures")
    change_wc = cash_flow_t.get("changeInWorkingCapital")
    if change_wc is None:
        change_wc = _compute_working_capital_change(balance_t)
    net_borrowings = cash_flow_t.get("netBorrowings")
    if net_borrowings is None:
        net_borrowings = _compute_net_borrowings(cash_flow_t)
    operating_cf = cash_flow_t.get("operatingCashflow")
    if operating_cf is None:
        operating_cf = _get_operating_cf(cash_flow_t)
    revenue = income_t.get("totalRevenue")
    if revenue is None:
        revenue = income_t.get("operatingRevenue")

    required = [net_income, depreciation, capex, change_wc, net_borrowings, operating_cf, revenue]
    if any(series is None or series.empty for series in required):
        logger.warning("Missing essential Alpha Vantage fields for %s; skipping.", ticker)
        return None

    combined_index = sorted(
        set(net_income.index)
        & set(depreciation.index)
        & set(capex.index)
        & set(change_wc.index)
        & set(net_borrowings.index)
        & set(operating_cf.index)
        & set(revenue.index)
    )

    if not combined_index:
        logger.warning("No common quarterly dates across statements for %s", ticker)
        return None

    data = pd.DataFrame(index=pd.to_datetime(combined_index))
    data["NetIncome"] = net_income.reindex(data.index)
    data["Depreciation"] = depreciation.reindex(data.index)
    data["Capex"] = capex.reindex(data.index)
    data["ChangeWC"] = change_wc.reindex(data.index)
    data["NetBorrowings"] = net_borrowings.reindex(data.index)
    data["OperatingCF"] = operating_cf.reindex(data.index)
    data["Revenue"] = revenue.reindex(data.index)

    data = data.sort_index()
    data = data.dropna(how="all")

    if data.empty:
        logger.warning("Fundamental dataframe empty after aligning statements for %s", ticker)
        return None

    data["Capex"] = data["Capex"].fillna(0.0)
    data["ChangeWC"] = data["ChangeWC"].fillna(0.0)
    data["NetBorrowings"] = data["NetBorrowings"].fillna(0.0)

    data["FCFE"] = (
        data["NetIncome"]
        + data["Depreciation"]
        - data["Capex"]
        - data["ChangeWC"]
        + data["NetBorrowings"]
    )
    data["NCFO"] = data["OperatingCF"]

    def _relative_change(series: pd.Series) -> pd.Series:
        return (series - series.shift(1)) / (series.abs() + EPS)

    data["FCFE_Delta"] = _relative_change(data["FCFE"])
    data["Revenue_Delta"] = _relative_change(data["Revenue"])
    data["NCFO_Delta"] = _relative_change(data["NCFO"])
    data["FCFE_Sign"] = np.sign(data["FCFE"]).fillna(0.0)

    result = data[["FCFE", "Revenue", "NCFO", "FCFE_Delta", "Revenue_Delta", "NCFO_Delta", "FCFE_Sign"]]
    result = result.dropna(how="all")

    if result.empty:
        logger.warning("Final result empty for ticker %s", ticker)
        return None

    result = result.reset_index().rename(columns={"index": "Date"})
    result["Ticker"] = ticker
    return FundamentalSeries(ticker=ticker, frame=result)


def build_dataset(tickers: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        logger.info("Processing fundamentals for %s", ticker)
        series = build_fundamental_frame(ticker)
        if series is None:
            continue
        frames.append(series.frame)

    if not frames:
        raise RuntimeError("No fundamental data could be collected for the provided tickers.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quarterly fundamentals CSV for PPO pipeline.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=ASSET_TICKERS,
        help="List of tickers to process (defaults to ASSET_TICKERS from config).",
    )
    parser.add_argument(
        "--output",
        default=FUNDAMENTAL_FEATURES_CONFIG.get("data_path", "data/quarterly_fundamentals.csv"),
        help="Output CSV path for the generated fundamentals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Starting Alpha Vantage Fetch for {len(args.tickers)} tickers.")
    logger.info("Rate limit enforced: 60s pause after every 5 requests.")
    
    dataset = build_dataset(args.tickers)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dataset.to_csv(args.output, index=False)
    logger.info("Fundamental dataset written to %s (%d rows)", args.output, len(dataset))


if __name__ == "__main__":  # pragma: no cover
    main()
