"""
Data Preparation Module

This module provides functions to prepare data for training LSTM models,
including sequence generation and train/validation/test splitting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sequences(data, seq_length, target_col, feature_cols=None):
    """
    Create sequences for single-step time series forecasting.

    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe containing features and target
    seq_length : int
        Length of input sequences
    target_col : str
        Name of the target column
    feature_cols : list, optional
        List of feature column names. If None, all columns except target_col are used.

    Returns:
    --------
    numpy.ndarray
        Input sequences (X)
    numpy.ndarray
        Target values (y)
    """
    try:
        logger.info(f"Creating sequences with length {seq_length}")

        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]

        # Extract features and target
        features = data[feature_cols].values

        # Ensure target column is in the dataframe
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            logger.warning(f"Target column '{target_col}' is not numeric. Attempting to convert.")
            try:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
            except Exception as e:
                raise ValueError(f"Error converting target column to numeric: {str(e)}")

        # Check for NaN values in target
        if data[target_col].isnull().any():
            logger.warning(f"Target column '{target_col}' contains NaN values. Filling with forward fill.")
            data[target_col] = data[target_col].fillna(method='ffill').fillna(method='bfill')

        target = data[target_col].values

        # Verify we have numeric data
        if not np.issubdtype(np.asarray(features).dtype, np.number) or not np.issubdtype(np.asarray(target).dtype, np.number):
            raise ValueError("Features or target contain non-numeric values")

        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])

        logger.info(f"Created {len(X)} sequences")
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        raise

def train_val_test_split(sequences, targets, val_size=0.15, test_size=0.15):
    """
    Split sequences and targets into training, validation, and test sets,
    respecting the temporal order to avoid data leakage.

    Parameters:
    -----------
    sequences : numpy.ndarray
        Input sequences with shape (n_samples, seq_length, n_features)
    targets : numpy.ndarray
        Target values with shape (n_samples,) for single-step forecasting
        or (n_samples, forecast_horizon) for multi-step forecasting
    val_size : float, optional
        Proportion of data to use for validation
    test_size : float, optional
        Proportion of data to use for testing

    Returns:
    --------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    try:
        logger.info(f"Splitting sequences with val_size={val_size}, test_size={test_size}")

        n = len(sequences)
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - val_size))

        X_train = sequences[:val_start_idx]
        y_train = targets[:val_start_idx]

        X_val = sequences[val_start_idx:test_start_idx]
        y_val = targets[val_start_idx:test_start_idx]

        X_test = sequences[test_start_idx:]
        y_test = targets[test_start_idx:]

        logger.info(f"Split complete. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"Error splitting sequences: {str(e)}")
        raise

def normalize_data(train_data, val_data, test_data):
    """
    Normalize features using StandardScaler, fitting only on training data
    to prevent data leakage.

    Parameters:
    -----------
    train_data : numpy.ndarray
        Training data with shape (n_samples, seq_length, n_features)
    val_data : numpy.ndarray
        Validation data with shape (n_samples, seq_length, n_features)
    test_data : numpy.ndarray
        Test data with shape (n_samples, seq_length, n_features)

    Returns:
    --------
    tuple
        (normalized_train, normalized_val, normalized_test, scaler)
    """
    try:
        logger.info("Normalizing sequence data")
        from sklearn.preprocessing import StandardScaler

        # Get dimensions
        n_train, seq_length, n_features = train_data.shape

        # Reshape to 2D for scaling
        train_reshaped = train_data.reshape(-1, n_features)
        val_reshaped = val_data.reshape(-1, n_features)
        test_reshaped = test_data.reshape(-1, n_features)

        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_reshaped)

        # Transform all datasets
        train_normalized = scaler.transform(train_reshaped).reshape(n_train, seq_length, n_features)
        val_normalized = scaler.transform(val_reshaped).reshape(val_data.shape)
        test_normalized = scaler.transform(test_reshaped).reshape(test_data.shape)

        logger.info("Data normalization complete")
        return train_normalized, val_normalized, test_normalized, scaler

    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        raise

def create_multistep_sequences(data, seq_length, target_col, forecast_horizon, feature_cols=None):
    """
    Create sequences for multi-step time series forecasting.

    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe containing features and target
    seq_length : int
        Length of input sequences
    target_col : str
        Name of the target column
    forecast_horizon : int
        Number of steps ahead to forecast
    feature_cols : list, optional
        List of feature column names. If None, all columns except target_col are used.

    Returns:
    --------
    numpy.ndarray
        Input sequences (X)
    numpy.ndarray
        Target sequences (y) with shape (n_samples, forecast_horizon)
    """
    try:
        logger.info(f"Creating multi-step sequences with length {seq_length} and forecast horizon {forecast_horizon}")

        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]

        # Extract features and target
        features = data[feature_cols].values

        # Ensure target column is in the dataframe
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            logger.warning(f"Target column '{target_col}' is not numeric. Attempting to convert.")
            try:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
            except Exception as e:
                raise ValueError(f"Error converting target column to numeric: {str(e)}")

        # Check for NaN values in target
        if data[target_col].isnull().any():
            logger.warning(f"Target column '{target_col}' contains NaN values. Filling with forward fill.")
            data[target_col] = data[target_col].fillna(method='ffill').fillna(method='bfill')

        target = data[target_col].values

        # Verify we have numeric data
        if not np.issubdtype(np.asarray(features).dtype, np.number) or not np.issubdtype(np.asarray(target).dtype, np.number):
            raise ValueError("Features or target contain non-numeric values")

        X, y = [], []

        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length:i+seq_length+forecast_horizon])

        logger.info(f"Created {len(X)} multi-step sequences")
        return np.array(X), np.array(y)

    except Exception as e:
        logger.error(f"Error creating multi-step sequences: {str(e)}")
        raise

def time_series_train_val_test_split(data, val_size=0.15, test_size=0.15):
    """
    Split time series data into training, validation, and test sets,
    respecting the temporal order.

    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data
    val_size : float, optional
        Proportion of data to use for validation
    test_size : float, optional
        Proportion of data to use for testing

    Returns:
    --------
    tuple
        (train_data, val_data, test_data)
    """
    try:
        logger.info(f"Splitting time series data with val_size={val_size}, test_size={test_size}")

        n = len(data)
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - val_size))

        train_data = data.iloc[:val_start_idx].copy()
        val_data = data.iloc[val_start_idx:test_start_idx].copy()
        test_data = data.iloc[test_start_idx:].copy()

        logger.info(f"Split complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data

    except Exception as e:
        logger.error(f"Error splitting time series data: {str(e)}")
        raise

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset.

        Parameters:
        -----------
        X : numpy.ndarray
            Input sequences with shape (n_samples, seq_length, n_features)
        y : numpy.ndarray
            Target values with shape (n_samples,) for single-step forecasting
            or (n_samples, forecast_horizon) for multi-step forecasting
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training data
    X_val, y_val : numpy.ndarray
        Validation data
    X_test, y_test : numpy.ndarray
        Test data
    batch_size : int, optional
        Batch size for DataLoaders

    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    try:
        logger.info(f"Creating DataLoaders with batch_size={batch_size}")

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Created DataLoaders. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)} batches")
        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating DataLoaders: {str(e)}")
        raise

def prepare_data_for_training(data, target_col, seq_length=20, forecast_horizon=1,
                             val_size=0.15, test_size=0.15, batch_size=32, feature_cols=None):
    """
    Comprehensive function to prepare data for model training.

    Parameters:
    -----------
    data : pandas.DataFrame
        Processed data with features and target
    target_col : str
        Name of the target column
    seq_length : int, optional
        Length of input sequences
    forecast_horizon : int, optional
        Number of steps ahead to forecast
    val_size : float, optional
        Proportion of data to use for validation
    test_size : float, optional
        Proportion of data to use for testing
    batch_size : int, optional
        Batch size for DataLoaders
    feature_cols : list, optional
        List of feature column names. If None, all columns except target_col are used.

    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader, feature_dim)
    """
    try:
        logger.info("Preparing data for model training")

        # Make a copy of the data to avoid modifying the original
        data = data.copy()

        # Log initial data state
        logger.info(f"Columns before numeric conversion: {list(data.columns)}")
        logger.info(f"Data types before numeric conversion:\n{data.dtypes}")

        # Convert all columns to numeric, excluding the index
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Log data state after numeric conversion
        logger.info(f"Columns after numeric conversion: {list(data.columns)}")
        logger.info(f"Data types after numeric conversion:\n{data.dtypes}")

        # Drop any columns that are entirely non-numeric
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data = data[numeric_cols]

        # Log columns after dropping non-numeric
        logger.info(f"Columns after dropping non-numeric: {list(data.columns)}")

        # Check if target column exists in numeric columns
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in numeric columns")

        # Handle NaN values in target column
        if data[target_col].isnull().any():
            logger.warning(f"Target column '{target_col}' contains NaN values. Filling NaNs with forward fill.")
            data[target_col] = data[target_col].fillna(method='ffill')
            if data[target_col].isnull().any():
                logger.warning(f"NaNs remain in target column '{target_col}' after forward fill. Filling remaining NaNs with 0.")
                data[target_col] = data[target_col].fillna(0)

        # Split data into train/val/test sets
        train_data, val_data, test_data = time_series_train_val_test_split(
            data, val_size=val_size, test_size=test_size
        )

        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            # Ensure all feature columns are present in the data
            feature_cols = [col for col in feature_cols if col in data.columns]

        feature_dim = len(feature_cols)
        logger.info(f"Using {feature_dim} features for training")

        # Create sequences
        if forecast_horizon == 1:
            # Single-step forecasting
            X_train, y_train = create_sequences(train_data, seq_length, target_col, feature_cols)
            X_val, y_val = create_sequences(val_data, seq_length, target_col, feature_cols)
            X_test, y_test = create_sequences(test_data, seq_length, target_col, feature_cols)
        else:
            # Multi-step forecasting
            X_train, y_train = create_multistep_sequences(train_data, seq_length, target_col, forecast_horizon, feature_cols)
            X_val, y_val = create_multistep_sequences(val_data, seq_length, target_col, forecast_horizon, feature_cols)
            X_test, y_test = create_multistep_sequences(test_data, seq_length, target_col, forecast_horizon, feature_cols)

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
        )

        logger.info("Data preparation complete")
        return train_loader, val_loader, test_loader, feature_dim

    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import os
    from feature_engineering import prepare_features
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
                processed_data, _ = prepare_features(
                    stock_data,
                    target_col=target_col,
                    include_technical=False,  # Set to False since we don't have OHLCV data
                    include_statistical=True,
                    include_lags=True,
                    normalize=True,
                    reduce_dim=False,
                    forecast_horizon=5
                )

                # Prepare data for training
                train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
                    processed_data,
                    target_col=f'Target_5',  # Target column created by prepare_features
                    seq_length=20,
                    forecast_horizon=1,  # Single-step forecasting
                    batch_size=32
                )

                print(f"Data prepared for training. Feature dimension: {feature_dim}")
