
import requests
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

try:
    from src.config import ALPHA_VANTAGE_API_KEY
except ImportError:
    from config import ALPHA_VANTAGE_API_KEY

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

def call_alpha_vantage_raw(function: str, symbol: str):
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    print(f"Fetching {function} for {symbol}...")
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        data = response.json()
        if "quarterlyReports" not in data:
            print(f"ERROR: 'quarterlyReports' not found in response for {function}")
            print(f"Keys found: {list(data.keys())}")
            print(f"Full response: {data}")
            return None
        return data["quarterlyReports"]
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def inspect_ticker(ticker):
    functions = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]
    
    for func in functions:
        reports = call_alpha_vantage_raw(func, ticker)
        if reports:
            df = pd.DataFrame(reports)
            print(f"\n--- {func} Columns ---")
            print(df.columns.tolist())
            
            # Check for specific expected columns
            if func == "INCOME_STATEMENT":
                print(f"Check 'netIncome': {'netIncome' in df.columns}")
                print(f"Check 'totalRevenue': {'totalRevenue' in df.columns}")
            elif func == "CASH_FLOW":
                print(f"Check 'depreciationAndAmortization': {'depreciationAndAmortization' in df.columns}")
                print(f"Check 'capitalExpenditures': {'capitalExpenditures' in df.columns}")
                print(f"Check 'changeInWorkingCapital': {'changeInWorkingCapital' in df.columns}")
                print(f"Check 'operatingCashflow': {'operatingCashflow' in df.columns}")

if __name__ == "__main__":
    inspect_ticker("AAPL")
