"""
Build quarterly fundamental features from per-ticker CSV files (Compustat-style).

Inputs: folder of CSVs (one per ticker) with columns such as:
  datadate, tic, niq, saleq/revtq, dpq, capxy, oancfy, dlcchy, dltisy, dltry, wcapq

Outputs: CSV with columns:
  Date, Ticker, FCFE, Revenue, NCFO

Key behavior:
- Detect fiscal year-end month per ticker.
- Detect whether Y-suffixed fields behave like YTD monotonic accumulators.
- Convert only those accumulator fields to quarterly flows.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import ASSET_TICKERS, BASE_DATA_PATH
except Exception:
    try:
        from config import ASSET_TICKERS, BASE_DATA_PATH
    except Exception:
        ASSET_TICKERS = []
        BASE_DATA_PATH = "data"

logger = logging.getLogger("fundamental_csv_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

YTD_COLUMNS = [
    "capxy",    # capex (YTD)
    "oancfy",   # operating cash flow (YTD)
    "dpcy",     # depreciation (YTD)
    "dpy",      # depreciation (YTD alt)
    "dlcchy",   # current debt changes (YTD)
    "dltisy",   # long-term debt issuance (YTD)
    "dltry",    # long-term debt reduction (YTD)
]


def detect_fye_month(dates: pd.Series) -> int:
    """Detect fiscal year-end month by most frequent datadate month."""
    months = dates.dt.month.dropna()
    if months.empty:
        return 12
    return int(months.mode().iloc[0])


def assign_fiscal_year(dates: pd.Series, fye_month: int) -> pd.Series:
    """Assign fiscal year label based on fiscal year-end month."""
    years = dates.dt.year.astype(int)
    months = dates.dt.month.astype(int)
    fy = years.copy()
    # If quarter end month is after fiscal year-end month, it belongs to next FY.
    fy[months > fye_month] = fy[months > fye_month] + 1
    return fy


def _is_ytd_like(df: pd.DataFrame, col: str, fy_col: str = "_fy", min_years: int = 3) -> bool:
    """
    Heuristic detection for YTD-like cumulative fields.
    A field is considered YTD-like if it is mostly non-decreasing within fiscal years.
    """
    if col not in df.columns:
        return False

    tested_years = 0
    monotonic_years = 0

    for _, g in df.groupby(fy_col):
        g = g.sort_values("datadate")
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(vals) < 3:
            continue
        tested_years += 1
        diffs = vals.diff().dropna()
        # Allow minor noise; mostly non-decreasing suggests YTD accumulation.
        non_decreasing_ratio = float((diffs >= -1e-9).mean()) if len(diffs) else 0.0
        if non_decreasing_ratio >= 0.75:
            monotonic_years += 1

    if tested_years < min_years:
        return False

    return (monotonic_years / tested_years) >= 0.7


def ytd_to_quarterly(df: pd.DataFrame, col: str, fy_col: str = "_fy") -> pd.Series:
    """Convert a YTD column into quarterly flows within each fiscal year."""
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    out = pd.Series(index=df.index, dtype="float64")
    for fy, g in df.groupby(fy_col):
        g = g.sort_values("datadate")
        vals = g[col].astype(float)
        q = vals - vals.shift(1)
        q.iloc[0] = vals.iloc[0]
        out.loc[g.index] = q
    return out


def load_ticker_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    if "datadate" not in df.columns:
        raise ValueError(f"Missing datadate column in {path}")
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["datadate"]).sort_values("datadate")
    return df


def _pick_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def build_from_folder(folder: Path, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    frames: List[pd.DataFrame] = []
    tickers_set = {t.upper() for t in tickers} if tickers else None

    for csv_path in sorted(folder.glob("*.csv")):
        try:
            df = load_ticker_csv(csv_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", csv_path.name, exc)
            continue

        ticker = None
        if "tic" in df.columns:
            ticker = str(df["tic"].iloc[0]).strip().upper()
        if not ticker:
            ticker = csv_path.stem.upper()

        if tickers_set and ticker not in tickers_set:
            continue

        df["ticker"] = ticker

        # Ensure numeric
        for col in df.columns:
            if col in ("datadate", "ticker", "tic"):
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Detect fiscal year and convert YTD columns
        fye_month = detect_fye_month(df["datadate"])
        df["_fy"] = assign_fiscal_year(df["datadate"], fye_month)

        ytd_detected: List[str] = []
        for col in YTD_COLUMNS:
            if col in df.columns and _is_ytd_like(df, col, "_fy"):
                df[col + "_q"] = ytd_to_quarterly(df, col, "_fy")
                ytd_detected.append(col)
            elif col in df.columns:
                # Keep as-is if it does not behave like a cumulative field.
                df[col + "_q"] = df[col]

        if ytd_detected:
            logger.info("%s | de-accumulated YTD columns: %s", ticker, ytd_detected)
        else:
            logger.info("%s | no monotonic YTD columns detected for de-accumulation", ticker)

        # Map required components
        net_income = df.get("niq")
        revenue = _pick_existing(df, ["revtq", "saleq", "revt"])

        depreciation = _pick_existing(df, ["dpq", "dpcy_q", "dpy_q", "dp"])
        capex = _pick_existing(df, ["capxq", "capxy_q", "capxy"])
        ocf = _pick_existing(df, ["oancfq", "oancfy_q", "oancfy"])

        # Net borrowings: issuance - repayment + current debt changes
        issuance = _pick_existing(df, ["dltisy_q", "dltisy"])
        repay = _pick_existing(df, ["dltry_q", "dltry"])
        curr = _pick_existing(df, ["dlcchy_q", "dlcchy"])
        issuance = issuance if issuance is not None else pd.Series(0.0, index=df.index)
        repay = repay if repay is not None else pd.Series(0.0, index=df.index)
        curr = curr if curr is not None else pd.Series(0.0, index=df.index)
        net_borrow = issuance.fillna(0.0) - repay.fillna(0.0) + curr.fillna(0.0)

        # Change in working capital (quarterly delta)
        if "wcapq" in df.columns:
            change_wc = pd.to_numeric(df["wcapq"], errors="coerce").diff().fillna(0.0)
        else:
            change_wc = pd.Series(0.0, index=df.index)

        if net_income is None or revenue is None or ocf is None or capex is None:
            logger.warning("Missing core fields for %s. Required: niq, (revtq/saleq), oancfy(_q), capxy(_q)", ticker)
            continue

        net_income = pd.to_numeric(net_income, errors="coerce").fillna(0.0)
        revenue = pd.to_numeric(revenue, errors="coerce").fillna(0.0)
        ocf = pd.to_numeric(ocf, errors="coerce").fillna(0.0)
        capex = pd.to_numeric(capex, errors="coerce").fillna(0.0)
        depreciation = pd.to_numeric(depreciation, errors="coerce").fillna(0.0) if depreciation is not None else pd.Series(0.0, index=df.index)

        fcfe = net_income + depreciation - capex - change_wc + net_borrow

        out = pd.DataFrame({
            "Date": df["datadate"].dt.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "FCFE": fcfe.values,
            "Revenue": revenue.values,
            "NCFO": ocf.values,
        })
        frames.append(out)

    if not frames:
        raise RuntimeError("No fundamentals could be built from the provided CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fundamentals from CSV folder.")
    parser.add_argument(
        "--input",
        default=str(Path(BASE_DATA_PATH) / "fundamentals"),
        help="Folder containing per-ticker CSVs",
    )
    parser.add_argument(
        "--output",
        default=str(Path(BASE_DATA_PATH) / "quarterly_fundamentals.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter. If omitted, all CSVs in input folder are processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = Path(args.input)
    tickers = args.tickers if args.tickers else None
    out = build_from_folder(input_folder, tickers)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Wrote fundamentals CSV: %s (%d rows)", args.output, len(out))


if __name__ == "__main__":
    main()
