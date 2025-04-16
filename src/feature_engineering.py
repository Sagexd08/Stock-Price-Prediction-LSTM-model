"""
Feature Engineering Module

This module provides functions to compute technical indicators and other
advanced features for stock price prediction.
"""

import numpy as np
import pandas as pd
# Temporarily comment out pandas_ta import due to compatibility issues
# import pandas_ta as ta
# Using basic pandas and numpy functions instead
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_technical_indicators(df, close_col='Close', high_col='High', low_col='Low', volume_col='Volume', include_all=False):
    """
    Add technical indicators to the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing stock price data
    close_col : str, optional
        Name of the column containing closing prices
    high_col : str, optional
        Name of the column containing high prices
    low_col : str, optional
        Name of the column containing low prices
    volume_col : str, optional
        Name of the column containing volume data
    include_all : bool, optional
        If True, include a comprehensive set of indicators. If False, include only basic ones.

    Returns:
    --------
    pandas.DataFrame
        Dataframe with added technical indicators
    """
    try:
        logger.info("Adding technical indicators to the dataframe")

        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()

        # Check if required columns exist
        required_cols = [col for col in [close_col, high_col, low_col, volume_col] if col is not None]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Some indicators may not be calculated.")

        # Basic indicators (always included)
        if close_col in df.columns:
            # Simple Moving Averages
            result_df[f'SMA_5'] = df[close_col].rolling(window=5).mean()
            result_df[f'SMA_20'] = df[close_col].rolling(window=20).mean()
            result_df[f'SMA_50'] = df[close_col].rolling(window=50).mean()

            # Exponential Moving Averages
            result_df[f'EMA_5'] = df[close_col].ewm(span=5, adjust=False).mean()
            result_df[f'EMA_20'] = df[close_col].ewm(span=20, adjust=False).mean()

            # MACD (Moving Average Convergence Divergence) - simplified version
            ema12 = df[close_col].ewm(span=12, adjust=False).mean()
            ema26 = df[close_col].ewm(span=26, adjust=False).mean()
            result_df['MACD'] = ema12 - ema26
            result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()
            result_df['MACD_Hist'] = result_df['MACD'] - result_df['MACD_Signal']

            # RSI (Relative Strength Index) - simplified version
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            result_df[f'RSI_14'] = 100 - (100 / (1 + rs))

        # Advanced indicators (included if include_all=True or if specific columns are available)
        if include_all:
            if all(col in df.columns for col in [close_col, high_col, low_col]):
                # Bollinger Bands
                sma20 = df[close_col].rolling(window=20).mean()
                std20 = df[close_col].rolling(window=20).std()
                result_df['BBL_20_2.0'] = sma20 - 2 * std20  # Lower band
                result_df['BBM_20_2.0'] = sma20              # Middle band
                result_df['BBU_20_2.0'] = sma20 + 2 * std20  # Upper band
                result_df['BBB_20_2.0'] = result_df['BBU_20_2.0'] - result_df['BBL_20_2.0']  # Bandwidth
                result_df['BBP_20_2.0'] = (df[close_col] - result_df['BBL_20_2.0']) / result_df['BBB_20_2.0']  # %B

                # ATR (Average True Range) - simplified version
                high_low = df[high_col] - df[low_col]
                high_close = (df[high_col] - df[close_col].shift()).abs()
                low_close = (df[low_col] - df[close_col].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                result_df[f'ATR_14'] = true_range.rolling(window=14).mean()

                # Stochastic Oscillator - simplified version
                low_14 = df[low_col].rolling(window=14).min()
                high_14 = df[high_col].rolling(window=14).max()
                result_df['STOCH_K'] = 100 * ((df[close_col] - low_14) / (high_14 - low_14))
                result_df['STOCH_D'] = result_df['STOCH_K'].rolling(window=3).mean()

            if all(col in df.columns for col in [close_col, volume_col]):
                # OBV (On-Balance Volume) - simplified version
                obv = pd.Series(0, index=df.index)
                price_change = df[close_col].diff()
                obv.iloc[1:] = np.where(price_change.iloc[1:] > 0, df[volume_col].iloc[1:],
                                       np.where(price_change.iloc[1:] < 0, -df[volume_col].iloc[1:], 0))
                result_df[f'OBV'] = obv.cumsum()

                # Volume-weighted Average Price - simplified version
                if all(col in df.columns for col in [high_col, low_col]):
                    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
                    result_df[f'VWAP'] = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()

        logger.info(f"Added {len(result_df.columns) - len(df.columns)} technical indicators")
        return result_df

    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        raise

def add_statistical_features(df, price_col='Close', window_sizes=[5, 10, 20]):
    """
    Add statistical features calculated over rolling windows.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing stock price data
    price_col : str, optional
        Name of the column to calculate statistics for
    window_sizes : list, optional
        List of window sizes for rolling calculations

    Returns:
    --------
    pandas.DataFrame
        Dataframe with added statistical features
    """
    try:
        logger.info("Adding statistical features to the dataframe")

        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()

        if price_col not in df.columns:
            logger.warning(f"Column {price_col} not found in dataframe. Cannot add statistical features.")
            return result_df

        # Calculate returns
        result_df[f'Returns'] = df[price_col].pct_change()

        # Calculate log returns
        result_df[f'LogReturns'] = np.log(df[price_col] / df[price_col].shift(1))

        for window in window_sizes:
            # Rolling statistics on returns
            result_df[f'Volatility_{window}'] = result_df['Returns'].rolling(window=window).std()
            result_df[f'Skewness_{window}'] = result_df['Returns'].rolling(window=window).skew()
            result_df[f'Kurtosis_{window}'] = result_df['Returns'].rolling(window=window).kurt()

            # Rolling statistics on price
            result_df[f'PriceStd_{window}'] = df[price_col].rolling(window=window).std()
            result_df[f'PriceMean_{window}'] = df[price_col].rolling(window=window).mean()

            # Price momentum (percent change over window)
            result_df[f'Momentum_{window}'] = df[price_col].pct_change(periods=window)

        logger.info(f"Added {len(result_df.columns) - len(df.columns)} statistical features")
        return result_df

    except Exception as e:
        logger.error(f"Error adding statistical features: {str(e)}")
        raise

def add_lag_features(df, cols_to_lag, lag_periods=[1, 2, 3, 5, 10]):
    """
    Add lagged values of specified columns as features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing stock price data
    cols_to_lag : list
        List of column names to create lag features for
    lag_periods : list, optional
        List of lag periods to use

    Returns:
    --------
    pandas.DataFrame
        Dataframe with added lag features
    """
    try:
        logger.info("Adding lag features to the dataframe")

        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()

        # Check if columns exist
        missing_cols = [col for col in cols_to_lag if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Lag features will not be created for these columns.")
            cols_to_lag = [col for col in cols_to_lag if col not in missing_cols]

        # Create lag features
        for col in cols_to_lag:
            for lag in lag_periods:
                result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        logger.info(f"Added {len(result_df.columns) - len(df.columns)} lag features")
        return result_df

    except Exception as e:
        logger.error(f"Error adding lag features: {str(e)}")
        raise

def normalize_features(df, method='standard', exclude_cols=None):
    """
    Normalize or standardize features in the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing features
    method : str, optional
        Normalization method. Options: 'standard', 'minmax'
    exclude_cols : list, optional
        List of columns to exclude from normalization

    Returns:
    --------
    pandas.DataFrame
        Dataframe with normalized features
    scaler : object
        The fitted scaler object for later use
    """
    try:
        logger.info(f"Normalizing features using {method} scaling")

        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()

        # Determine columns to normalize
        if exclude_cols is None:
            exclude_cols = []

        # Filter out non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        # Select the appropriate scaler
        if method.lower() == 'standard':
            scaler = StandardScaler()
        elif method.lower() == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Fit and transform the data
        if cols_to_normalize:
            # Check if we have any data to normalize
            if df[cols_to_normalize].shape[0] == 0:
                logger.error("Cannot normalize an empty dataset. Ensure your data has at least one row.")
                raise ValueError("Empty dataset after preprocessing. Check your data filtering steps.")

            # Fill NaN values before normalization
            data_to_normalize = df[cols_to_normalize].fillna(0)

            # Log data shape before normalization
            logger.info(f"Data shape before normalization: {data_to_normalize.shape}")

            # Fit and transform
            result_df[cols_to_normalize] = scaler.fit_transform(data_to_normalize)
            logger.info(f"Normalized {len(cols_to_normalize)} features")
        else:
            logger.warning("No columns to normalize")

        return result_df, scaler

    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        raise

def reduce_dimensionality(df, n_components=0.95, exclude_cols=None):
    """
    Reduce dimensionality of the feature set using PCA.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing features
    n_components : float or int, optional
        Number of components to keep. If float between 0 and 1, represents the variance to preserve.
    exclude_cols : list, optional
        List of columns to exclude from PCA

    Returns:
    --------
    pandas.DataFrame
        Dataframe with reduced dimensionality
    pca : object
        The fitted PCA object for later use
    """
    try:
        logger.info(f"Reducing dimensionality using PCA")

        # Determine columns for PCA
        if exclude_cols is None:
            exclude_cols = []

        # Filter out non-numeric columns and excluded columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_for_pca = [col for col in numeric_cols if col not in exclude_cols]

        if not cols_for_pca:
            logger.warning("No columns available for PCA")
            return df.copy(), None

        # Fill NaN values with 0 for PCA
        data_for_pca = df[cols_for_pca].fillna(0)

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_for_pca)

        # Create a new dataframe with PCA components
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
            index=df.index
        )

        # Add excluded columns back
        for col in exclude_cols:
            if col in df.columns:
                pca_df[col] = df[col]

        logger.info(f"Reduced dimensionality from {len(cols_for_pca)} to {pca_result.shape[1]} components")
        return pca_df, pca

    except Exception as e:
        logger.error(f"Error reducing dimensionality: {str(e)}")
        raise

def validate_data(data):
    """Validate input data before processing."""
    if data is None or len(data) == 0:
        raise ValueError("Empty or None dataset provided")

    # Check for required columns
    required_cols = ['Close']  # Add other required columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for numeric data
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    # Check for NaN values
    nan_cols = data.columns[data.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Warning: NaN values found in columns: {nan_cols}")

    return data

def handle_data_errors(data):
    """Handle common data issues."""
    try:
        result_df = data.copy()

        # Handle string/object columns that should be numeric
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in result_df.columns:
                # Remove any currency symbols and commas
                if pd.api.types.is_object_dtype(result_df[col]):
                    result_df[col] = result_df[col].replace('[\$,]', '', regex=True)
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        # Handle percentage and volume columns
        if 'Volume' in result_df.columns and pd.api.types.is_object_dtype(result_df['Volume']):
            result_df['Volume'] = result_df['Volume'].replace('[\%,]', '', regex=True)
            result_df['Volume'] = pd.to_numeric(result_df['Volume'], errors='coerce')

        # Fill NaN values
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')

        return result_df

    except Exception as e:
        raise ValueError(f"Error handling data conversions: {str(e)}")

def prepare_features(df, target_col='Close', include_technical=True, include_statistical=True,
                    include_lags=True, normalize=True, reduce_dim=False, forecast_horizon=5):
    """
    Comprehensive function to prepare features for model training.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing raw stock price data
    target_col : str, optional
        Name of the target column to predict
    include_technical : bool, optional
        Whether to include technical indicators
    include_statistical : bool, optional
        Whether to include statistical features
    include_lags : bool, optional
        Whether to include lag features
    normalize : bool, optional
        Whether to normalize features
    reduce_dim : bool, optional
        Whether to reduce dimensionality using PCA
    forecast_horizon : int, optional
        Number of steps ahead to forecast

    Returns:
    --------
    pandas.DataFrame
        Dataframe with prepared features
    dict
        Dictionary containing fitted transformers (scalers, PCA, etc.)
    """
    try:
        logger.info("Preparing features for model training")
        
        # Make a copy of the input data
        result_df = df.copy()
        transformers = {}

        # Ensure target column exists
        if target_col not in result_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Clean and validate data
        for col in result_df.columns:
            if pd.api.types.is_object_dtype(result_df[col]):
                # Try to convert object columns to numeric
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        # Handle NaN values in target column
        target_nulls = result_df[target_col].isnull().sum() > 0
        if target_nulls:
            logger.warning(f"Found NaN values in target column {target_col}. Filling with forward/backward fill.")
            result_df[target_col] = result_df[target_col].fillna(method='ffill').fillna(method='bfill')

        # Drop rows where target column is still NaN
        result_df = result_df.dropna(subset=[target_col])

        if len(result_df) == 0:
            raise ValueError("No valid data remains after cleaning target column")

        # Fill NaN values in other columns
        result_df = result_df.fillna(method='ffill')
        result_df = result_df.fillna(method='bfill')
        result_df = result_df.fillna(0)  # Fill any remaining NaN with 0

        # Add technical indicators if requested
        if include_technical:
            try:
                result_df = add_technical_indicators(result_df)
                logger.info(f"Added technical indicators. New shape: {result_df.shape}")
            except Exception as e:
                logger.warning(f"Error adding technical indicators: {str(e)}. Continuing without them.")

        # Add statistical features if requested
        if include_statistical:
            try:
                result_df = add_statistical_features(result_df, price_col=target_col)
                logger.info(f"Added statistical features. New shape: {result_df.shape}")
            except Exception as e:
                logger.warning(f"Error adding statistical features: {str(e)}. Continuing without them.")

        # Add lag features if requested
        if include_lags:
            try:
                cols_to_lag = [target_col]
                if 'Volume' in result_df.columns:
                    cols_to_lag.append('Volume')
                result_df = add_lag_features(result_df, cols_to_lag)
                logger.info(f"Added lag features. New shape: {result_df.shape}")
            except Exception as e:
                logger.warning(f"Error adding lag features: {str(e)}. Continuing without them.")

        # Create target variable
        result_df[f'Target_{forecast_horizon}'] = result_df[target_col].shift(-forecast_horizon)

        # Drop rows with NaN in target
        result_df = result_df.dropna(subset=[f'Target_{forecast_horizon}'])

        # Normalize features if requested
        if normalize and len(result_df) > 0:
            try:
                result_df, scaler = normalize_features(result_df, exclude_cols=[f'Target_{forecast_horizon}'])
                transformers['scaler'] = scaler
                logger.info("Features normalized")
            except Exception as e:
                logger.warning(f"Error normalizing features: {str(e)}. Continuing without normalization.")

        # Reduce dimensionality if requested
        if reduce_dim and len(result_df.columns) > 10:
            try:
                exclude_cols = [f'Target_{forecast_horizon}']
                result_df, pca = reduce_dimensionality(result_df, exclude_cols=exclude_cols)
                transformers['pca'] = pca
                logger.info("Dimensionality reduction completed successfully")
            except Exception as e:
                logger.warning(f"Error reducing dimensionality: {str(e)}. Continuing without dimension reduction.")

        logger.info(f"Feature preparation complete. Final dataframe shape: {result_df.shape}")
        return result_df, transformers

    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def calculate_technical_indicators(data):
    # Implement SMA, EMA, RSI, MACD, Bollinger Bands, etc.
    pass

def generate_statistical_features(data, windows=[5, 10, 20]):
    # Rolling statistics, volatility measures, etc.
    pass

def create_lagged_features(data, lag_periods=[1, 5, 10]):
    # Create lagged versions of key features
    pass

if __name__ == "__main__":
    # Example usage
    import os
    from data_acquisition import load_stock_data

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)

    # Example: Load stock data
    stock_data_path = os.path.join(raw_dir, 'stock_data.csv')
    if os.path.exists(stock_data_path):
        stock_data = load_stock_data(stock_data_path)

        # Prepare features
        if stock_data is not None:
            # For this example, let's assume the first column after the index is the closing price
            if len(stock_data.columns) > 0:
                target_col = stock_data.columns[0]

                # Prepare features
                processed_data, transformers = prepare_features(
                    stock_data,
                    target_col=target_col,
                    include_technical=False,  # Set to False since we don't have OHLCV data
                    include_statistical=True,
                    include_lags=True,
                    normalize=True,
                    reduce_dim=False,
                    forecast_horizon=5
                )

                # Save processed data
                processed_data.to_csv(os.path.join(processed_dir, 'processed_stock_data.csv'))
                print(f"Processed data saved with {processed_data.shape[1]} features")
