# Fundamental Data Column Reference

This document describes the columns found in the raw fundamentals CSV files in this folder (`cat.csv`, `googl.csv`, `jnj.csv`, `jpm.csv`, `lin.csv`, `msft.csv`, `nee.csv`, `pg.csv`, `unh.csv`, `xom.csv`).

## Raw Column Dictionary

| Column | Full Name | Meaning |
|---|---|---|
| `costat` | Company Status | Company activity status flag (for example, active/inactive). |
| `curcdq` | Currency Code (Quarterly) | Reporting currency for quarterly items (for example, `USD`). |
| `datafmt` | Data Format | Financial statement reporting format code. |
| `indfmt` | Industry Format | Industry reporting format code. |
| `consol` | Consolidation Code | Consolidation scope indicator (for example, consolidated statements). |
| `tic` | Ticker Symbol | Equity ticker identifier (for example, `MSFT`). |
| `datadate` | Data Date | Fiscal period end date for the row. |
| `gvkey` | Global Company Key | Company identifier used in Compustat-style datasets. |
| `dd1q` | Long-Term Debt Due in One Year | Debt maturing within one year at quarter end. |
| `dlttq` | Long-Term Debt - Total | Long-term debt balance at quarter end. |
| `dpq` | Depreciation and Amortization - Total | Quarterly depreciation/amortization expense. |
| `findlcq` | Finance Division Long-Term Debt Current | Finance-division debt due within current liabilities. |
| `findltq` | Finance Division Debt Long-Term | Finance-division long-term debt field from source export. |
| `niq` | Net Income (Loss) | Quarterly net income (or loss). |
| `revtq` | Revenue - Total | Quarterly total revenue. |
| `saleq` | Sales/Turnover (Net) | Quarterly net sales/turnover. |
| `wcapq` | Working Capital (Balance Sheet) | Working capital level at quarter end. |
| `capxy` | Capital Expenditures | Fiscal year-to-date capital expenditures (de-accumulated to quarterly in pipeline). |
| `dlcchy` | Changes in Current Debt | Fiscal year-to-date changes in current debt (de-accumulated to quarterly in pipeline). |
| `dltisy` | Long-Term Debt - Issuance | Fiscal year-to-date long-term debt issuance (de-accumulated to quarterly in pipeline). |
| `dltry` | Long-Term Debt - Reduction | Fiscal year-to-date long-term debt reduction (de-accumulated to quarterly in pipeline). |
| `dpcy` | Depreciation and Amortization - Statement of Cash Flows | Fiscal year-to-date cash-flow statement depreciation/amortization (de-accumulated to quarterly in pipeline). |
| `dpy` | Depreciation and Amortization - Total | Fiscal year-to-date depreciation/amortization total (de-accumulated to quarterly in pipeline). |
| `oancfy` | Operating Activities - Net Cash Flow | Fiscal year-to-date operating cash flow (de-accumulated to quarterly in pipeline). |
| `wcapcy` | Working Capital Change - Other - Increase/(Decrease) | Fiscal year-to-date working capital change item. |

## Columns Used by the Project Pipeline

The pipeline output schema is:
- `Date`
- `Ticker`
- `FCFE`
- `Revenue`
- `NCFO`

### Revenue Mapping
- Preferred source: `revtq`
- Fallback source: `saleq`

### NCFO (Net Cash Flow from Operations) Mapping
- Preferred source: `oancfy` after YTD-to-quarterly conversion (`oancfy_q`)

### FCFE (Free Cash Flow to Equity) Construction
The script computes:

`FCFE = NetIncome + Depreciation - CapEx - DeltaWorkingCapital + NetBorrowings`

Where:
- `NetIncome` -> `niq`
- `Depreciation` -> `dpq` (fallback: `dpcy_q` or `dpy_q`)
- `CapEx` -> `capxy_q` (or `capxq` if present)
- `DeltaWorkingCapital` -> `diff(wcapq)`
- `NetBorrowings` -> `dltisy_q - dltry_q + dlcchy_q`

## Monotonic Accumulation Fix (YTD to Quarterly)

The following Y-suffix columns are treated as fiscal year-to-date accumulators when they behave monotonically within fiscal year:
- `capxy`
- `oancfy`
- `dpcy`
- `dpy`
- `dlcchy`
- `dltisy`
- `dltry`

Conversion rule inside each fiscal year:
- `quarter_value(t) = ytd_value(t) - ytd_value(t-1)`
- first quarter of fiscal year uses `quarter_value = ytd_value`

This removes YTD accumulation and aligns features to quarterly signal construction.

## Notes
- `findlcq` and `findltq` are retained in raw files for reference but are not required for current FCFE/Revenue/NCFO extraction.
- If a required field is missing for a ticker/date, fallback logic or zero-filling is applied by the builder script where appropriate.
