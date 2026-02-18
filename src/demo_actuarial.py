import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_active_config
from data_utils import DataProcessor

def run_demo():
    print("Loading configuration...")
    config = get_active_config('phase1')
    
    print("Initializing DataProcessor...")
    processor = DataProcessor(config)
    
    print("Loading data...")
    df = processor.load_ohlcv_data()
    
    print("Adding Actuarial Features...")
    df = processor.add_actuarial_features(df)
    
    # Show a specific example: AAPL during COVID crash (March 2020)
    print("\n" + "="*80)
    print("EXAMPLE: AAPL during COVID Crash (March 2020)")
    print("Notice how 'Prob_30d' drops and 'Severity' spikes as the crash deepens.")
    print("="*80)
    
    aapl = df[df['Ticker'] == 'AAPL'].copy()
    aapl['Date'] = pd.to_datetime(aapl['Date'])
    
    # Filter for the crash period
    mask = (aapl['Date'] >= '2020-02-20') & (aapl['Date'] <= '2020-04-01')
    subset = aapl.loc[mask, ['Date', 'Close', 'Actuarial_Expected_Recovery', 'Actuarial_Prob_30d', 'Actuarial_Reserve_Severity']]
    
    # Format for nice printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(subset.to_string(index=False))

if __name__ == "__main__":
    run_demo()
