"""
Data Acquisition Module

This module provides functions to download and load historical stock data
from various sources including Yahoo Finance and local CSV files.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_stock_data(ticker, start_date, end_date=None, interval='1d', save_path=None):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    interval : str, optional
        Data interval. Options: '1d', '1wk', '1mo', etc.
    save_path : str, optional
        Path to save the downloaded data as CSV. If None, data is not saved.
        
    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns: Open, High, Low, Close, Volume, etc.
    """
    try:
        logger.info(f"Downloading data for {ticker} from {start_date} to {end_date or 'today'}")
        
        # If end_date is not provided, use current date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Download data using yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if stock_data.empty:
            logger.warning(f"No data found for {ticker} in the specified date range")
            return None
            
        logger.info(f"Successfully downloaded {len(stock_data)} records for {ticker}")
        
        # Save data if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stock_data.to_csv(save_path)
            logger.info(f"Data saved to {save_path}")
            
        return stock_data
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        raise

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing stock data
        
    Returns:
    --------
    pandas.DataFrame
        Stock data loaded from the CSV file
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Load data from CSV
        data = pd.read_csv(file_path)
        
        # If the first column is a date, set it as index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        elif data.columns[0] == 'Unnamed: 0':
            data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def download_market_index(index_ticker, start_date, end_date=None, save_path=None):
    """
    Download market index data (e.g., S&P 500, NASDAQ) from Yahoo Finance.
    
    Parameters:
    -----------
    index_ticker : str
        Index ticker symbol (e.g., '^GSPC' for S&P 500)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    save_path : str, optional
        Path to save the downloaded data as CSV. If None, data is not saved.
        
    Returns:
    --------
    pandas.DataFrame
        Historical index data
    """
    return download_stock_data(index_ticker, start_date, end_date, save_path=save_path)

def merge_stock_with_index(stock_data, index_data, on_index=True):
    """
    Merge stock data with market index data.
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        Stock price data
    index_data : pandas.DataFrame
        Market index data
    on_index : bool, optional
        If True, merge on the index (Date). If False, merge on a common column.
        
    Returns:
    --------
    pandas.DataFrame
        Merged data containing both stock and index information
    """
    try:
        logger.info("Merging stock data with market index data")
        
        if on_index:
            # Merge on the index (Date)
            merged_data = pd.merge(stock_data, index_data, 
                                   left_index=True, right_index=True, 
                                   suffixes=('', '_index'))
        else:
            # Merge on a common column (assuming 'Date' is the common column)
            merged_data = pd.merge(stock_data, index_data, 
                                   on='Date', 
                                   suffixes=('', '_index'))
            
        logger.info(f"Successfully merged data. Result has {len(merged_data)} records")
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging stock data with index data: {str(e)}")
        raise

def download_multiple_stocks(tickers, start_date, end_date=None, interval='1d', save_dir=None):
    """
    Download data for multiple stock tickers in parallel.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    interval : str, optional
        Data interval
    save_dir : str, optional
        Directory to save individual stock data files
        
    Returns:
    --------
    dict
        Dictionary mapping tickers to their respective DataFrames
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def download_single(ticker):
        save_path = os.path.join(save_dir, f"{ticker}_stock.csv") if save_dir else None
        return ticker, download_stock_data(ticker, start_date, end_date, interval, save_path)
    
    stock_data = {}
    with ThreadPoolExecutor() as executor:
        for ticker, data in executor.map(download_single, tickers):
            if data is not None:
                stock_data[ticker] = data
    
    return stock_data

def validate_data_quality(data):
    """
    Validate and clean the downloaded stock data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned and validated data
    """
    # Check for missing values
    missing_count = data.isnull().sum()
    if missing_count.any():
        logger.warning(f"Found missing values:\n{missing_count[missing_count > 0]}")
        
    # Forward fill missing values (common practice for time series)
    data = data.ffill()
    
    # Remove any remaining rows with missing values
    data = data.dropna()
    
    # Check for duplicate indices
    if data.index.duplicated().any():
        logger.warning("Found duplicate dates in data")
        data = data[~data.index.duplicated(keep='first')]
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    return data

def add_market_indicators(stock_data, include_vix=True, include_treasury=True):
    """
    Add market indicators like VIX and Treasury yields to the stock data.
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        Base stock data
    include_vix : bool
        Whether to include VIX volatility index
    include_treasury : bool
        Whether to include Treasury yield data
        
    Returns:
    --------
    pandas.DataFrame
        Enhanced stock data with market indicators
    """
    enhanced_data = stock_data.copy()
    
    if include_vix:
        try:
            vix_data = download_stock_data('^VIX', 
                                         stock_data.index[0].strftime('%Y-%m-%d'),
                                         stock_data.index[-1].strftime('%Y-%m-%d'))
            if vix_data is not None:
                enhanced_data = merge_stock_with_index(enhanced_data, 
                                                     vix_data[['Close']].rename(
                                                         columns={'Close': 'VIX'}))
        except Exception as e:
            logger.warning(f"Could not add VIX data: {str(e)}")
    
    if include_treasury:
        try:
            treasury_data = download_stock_data('^TNX',  # 10-year Treasury yield
                                              stock_data.index[0].strftime('%Y-%m-%d'),
                                              stock_data.index[-1].strftime('%Y-%m-%d'))
            if treasury_data is not None:
                enhanced_data = merge_stock_with_index(enhanced_data,
                                                     treasury_data[['Close']].rename(
                                                         columns={'Close': 'Treasury10Y'}))
        except Exception as e:
            logger.warning(f"Could not add Treasury data: {str(e)}")
    
    return enhanced_data

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    
    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Example: Download Apple stock data for the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    apple_data = download_stock_data(
        'AAPL', 
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        save_path=os.path.join(raw_dir, 'apple_stock.csv')
    )
    
    # Example: Download S&P 500 data for the same period
    sp500_data = download_market_index(
        '^GSPC',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        save_path=os.path.join(raw_dir, 'sp500_index.csv')
    )
    
    # Merge the data
    if apple_data is not None and sp500_data is not None:
        merged_data = merge_stock_with_index(apple_data, sp500_data)
        merged_data.to_csv(os.path.join(raw_dir, 'apple_with_sp500.csv'))
