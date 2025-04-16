import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Define the stock symbols to download
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Calculate date range (5 years of data)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"Downloading stock data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download data for each symbol
for symbol in symbols:
    print(f"Downloading data for {symbol}...")
    data = yf.download(
        symbol,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )
    
    # Save to CSV
    file_path = os.path.join('data', 'raw', f'{symbol}_data.csv')
    data.to_csv(file_path)
    print(f"Saved {len(data)} rows to {file_path}")

# Create a combined dataset
print("Creating combined stock dataset...")
combined_data = pd.DataFrame()

for symbol in symbols:
    file_path = os.path.join('data', 'raw', f'{symbol}_data.csv')
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Add symbol as a column
        data['Symbol'] = symbol
        combined_data = pd.concat([combined_data, data])

# Save combined data
combined_file_path = os.path.join('data', 'raw', 'stock_data.csv')
combined_data.to_csv(combined_file_path)
print(f"Saved combined dataset with {len(combined_data)} rows to {combined_file_path}")

print("Download complete!")
