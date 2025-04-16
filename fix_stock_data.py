import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_stock_data(file_path):
    """
    Fix the structure and data types of the stock data CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing stock data
    """
    try:
        logger.info(f"Fixing data in {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        # Load data from CSV
        data = pd.read_csv(file_path)
        
        # Check if the data has the expected structure
        if 'Ticker' in data.columns and 'Date' in data.columns:
            logger.info("Data already has the expected structure")
            fixed_data = data
        else:
            logger.info("Restructuring the data")
            
            # If the first row contains column headers
            if 'Price' in data.columns and data.iloc[0].values[0] == 'Ticker':
                # Extract the actual data, skipping header rows
                fixed_data = pd.DataFrame(data.values[2:], columns=data.columns)
                
                # Rename columns if needed
                if 'Price' in fixed_data.columns and 'Close' in fixed_data.columns:
                    fixed_data = fixed_data.drop(columns=['Price'])
            else:
                fixed_data = data
        
        # Convert date column to datetime
        if 'Date' in fixed_data.columns:
            fixed_data['Date'] = pd.to_datetime(fixed_data['Date'], errors='coerce')
        
        # Convert numeric columns to float
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in fixed_data.columns:
            if col in numeric_cols or any(keyword in col.lower() for keyword in ['price', 'open', 'high', 'low', 'close', 'volume', 'adj']):
                try:
                    fixed_data[col] = pd.to_numeric(fixed_data[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
        
        # Drop rows with NaN values in important columns
        if 'Close' in fixed_data.columns:
            fixed_data = fixed_data.dropna(subset=['Close'])
        
        # Fill remaining NaN values
        fixed_data = fixed_data.fillna(method='ffill').fillna(method='bfill')
        
        # Save the fixed data
        fixed_data.to_csv(file_path, index=False)
        logger.info(f"Fixed data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error fixing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Fix the stock data
    data_dir = os.path.join(os.getcwd(), 'data', 'raw')
    stock_data_path = os.path.join(data_dir, 'stock_data.csv')
    fix_stock_data(stock_data_path)
    
    # Also fix individual stock files if they exist
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        file_path = os.path.join(data_dir, f'{ticker}_data.csv')
        if os.path.exists(file_path):
            fix_stock_data(file_path)
