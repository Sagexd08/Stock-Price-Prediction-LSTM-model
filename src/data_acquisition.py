import os
import numpy as np
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

def clean_column_name(col):
    """Clean column name by removing special characters and handling tuples."""
    if isinstance(col, tuple):
        # Take the first element of the tuple (column name)
        return str(col[0]).strip("'()")
    return str(col).strip("'()")

def clean_and_flatten_data(data, column_name):
    """Clean and flatten data from a DataFrame column."""
    try:
        # Convert to numpy array and flatten if multi-dimensional
        values = data[column_name].values
        if values.ndim > 1:
            values = values.flatten()
        return pd.Series(values, index=data.index, dtype=float)
    except Exception as e:
        logger.warning(f"Error cleaning column {column_name}: {str(e)}")
        return pd.Series(index=data.index, dtype=float)

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

        # Ensure index is datetime
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)

        # Clean column names
        stock_data.columns = [clean_column_name(col) for col in stock_data.columns]
        
        # Create a new DataFrame with cleaned data
        cleaned_data = pd.DataFrame(index=stock_data.index)
        
        # Process each numeric column
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in numeric_cols:
            if col in stock_data.columns:
                cleaned_data[col] = clean_and_flatten_data(stock_data, col)

        # Add ticker column
        cleaned_data['Ticker'] = ticker

        # Handle missing values
        cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')

        # Save data if save_path is provided
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                
                # Reset index to make date a column
                save_data = cleaned_data.copy()
                save_data.index.name = 'Date'
                save_data.reset_index(inplace=True)
                
                # Save to CSV making sure data types are preserved
                save_data.to_csv(save_path, index=False, float_format='%.4f', date_format='%Y-%m-%d')
                logger.info(f"Data saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving data to {save_path}: {str(e)}")
                # Continue execution even if saving fails

        return cleaned_data

    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        raise

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.
    """
    try:
        logger.info(f"Loading data from {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        # Read CSV with proper date parsing
        data = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Clean column names
        data.columns = [clean_column_name(col) for col in data.columns]
        
        # Set date as index
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)

        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in data.columns:
            if col in numeric_cols:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {str(e)}")

        # Clean up data
        data = data.replace([np.inf, -np.inf], np.nan)
        if 'Close' in data.columns:
            data = data.dropna(subset=['Close'])
        data = data.fillna(method='ffill').fillna(method='bfill')

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
