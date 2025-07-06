import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

def fetch_currency_data():
    """
    Fetches last 12 months of currency data vs USD from Yahoo Finance
    and saves to CSV files (only if CSV doesn't exist)
    """
    
    # Currency pairs - Yahoo Finance uses format like 'EURUSD=X'
    currencies = {
        'EUR': 'EURUSD=X',    # Euro
        'CAD': 'CADUSD=X',    # Canadian Dollar  
        'CNY': 'CNYUSD=X',    # Chinese Yuan
        'MXN': 'MXNUSD=X',    # Mexican Peso
        'JPY': 'JPYUSD=X',    # Japanese Yen
        'GBP': 'GBPUSD=X',    # British Pound
        'KRW': 'KRWUSD=X',    # Korean Won
        'TWD': 'TWDUSD=X',    # Taiwan Dollar
        'INR': 'INRUSD=X',    # Indian Rupee
        'AUD': 'AUDUSD=X',    # Australian Dollar
        'CHF': 'CHFUSD=X',    # Swiss Franc
        'SEK': 'SEKUSD=X',    # Swedish Krona
        'BRL': 'BRLUSD=X',    # Brazilian Real
        'SGD': 'SGDUSD=X',    # Singapore Dollar
        'VND': 'VNDUSD=X',    # Vietnamese Dong
        'NOK': 'NOKUSD=X',    # Norwegian Krone
        'NZD': 'NZDUSD=X',    # New Zealand Dollar
        'THB': 'THBUSD=X',    # Thai Baht
    }
    
    # Create data directory if it doesn't exist
    if not os.path.exists('currency_data'):
        os.makedirs('currency_data')
    
    # Calculate date range (last 12 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Fetching currency data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    for currency_code, ticker in currencies.items():
        csv_filename = f'currency_data/{currency_code}_USD_12m.csv'
        
        # Check if CSV already exists
        if os.path.exists(csv_filename):
            print(f"✓ {currency_code}/USD data already exists - skipping API call")
            continue
        
        try:
            print(f"📡 Fetching {currency_code}/USD data...")
            
            # Fetch data from Yahoo Finance
            ticker_obj = yf.Ticker(ticker)
            hist_data = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist_data.empty:
                print(f"❌ No data found for {currency_code}/USD")
                continue
            
            # Clean up the data
            hist_data.reset_index(inplace=True)
            hist_data['Currency'] = currency_code
            hist_data['Pair'] = f"{currency_code}/USD"
            
            # Reorder columns for better readability
            column_order = ['Date', 'Currency', 'Pair', 'Open', 'High', 'Low', 'Close', 'Volume']
            hist_data = hist_data[column_order]
            
            # Save to CSV
            hist_data.to_csv(csv_filename, index=False)
            print(f"✅ {currency_code}/USD data saved to {csv_filename} ({len(hist_data)} rows)")
            
            # Add small delay to be respectful to API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error fetching {currency_code}/USD: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("Currency data fetch complete!")
    
    # Summary of files created
    csv_files = [f for f in os.listdir('currency_data') if f.endswith('.csv')]
    print(f"Total CSV files: {len(csv_files)}")
    print("Files created:")
    for file in sorted(csv_files):
        print(f"  - {file}")

def combine_all_data():
    """
    Optional: Combine all currency data into a single CSV
    """
    csv_files = [f for f in os.listdir('currency_data') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found to combine.")
        return
    
    combined_filename = 'currency_data/ALL_CURRENCIES_12m.csv'
    
    if os.path.exists(combined_filename):
        print(f"Combined file {combined_filename} already exists - skipping")
        return
    
    all_data = []
    
    for file in csv_files:
        if file.startswith('ALL_'):  # Skip the combined file itself
            continue
        
        filepath = os.path.join('currency_data', file)
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(combined_filename, index=False)
        print(f"✅ Combined data saved to {combined_filename} ({len(combined_df)} total rows)")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import yfinance
        import pandas
    except ImportError as e:
        print("❌ Required packages not installed. Please run:")
        print("pip install yfinance pandas")
        exit(1)
    
    print("🚀 Starting currency data fetch...")
    fetch_currency_data()
    
    # Optionally combine all data
    print("\n" + "=" * 60)
    combine_response = input("Would you like to combine all data into one CSV? (y/n): ")
    if combine_response.lower() in ['y', 'yes']:
        combine_all_data()
    
    print("\n✨ Script complete!")
