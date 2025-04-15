"""s error occurs when the data preparation process cannot find numeric columns to perform aggregation operations. Let me help you fix this issue in the model training section.
Data Preparation Module
Here's how to resolve this:
This module provides functions to prepare data for training LSTM models,
including sequence generation and train/validation/test splitting.suggests the DataFrame might have non-numeric data types. Add these checks before processing:
"""
def prepare_features(data, target_col, **kwargs):
import numpy as npto avoid modifying original data
import pandas as pd)
from sklearn.model_selection import train_test_split
import torchdata types and convert if necessary
from torch.utils.data import Dataset, DataLoader'float64', 'int64']).columns
import loggingeric_columns) == 0:
        # Try to convert string columns that might be numeric
# Configure loggingdf.columns:
logging.basicConfig(
    level=logging.INFO, = pd.to_numeric(df[col], errors='coerce')
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)               continue
logger = logging.getLogger(__name__)
    # Verify numeric columns after conversion
def create_sequences(data, seq_length, target_col, feature_cols=None):olumns
    """len(numeric_columns) == 0:
    Create sequences for single-step time series forecasting.aset after conversion attempts")
    
    Parameters:rget column is numeric
    -----------ol not in df.columns:
    data : pandas.DataFrameTarget column '{target_col}' not found in dataset")
        Dataframe containing features and target
    seq_length : int
        Length of input sequencesmeric(df[target_col], errors='coerce')
    target_col : str
        Name of the target columnnot convert target column '{target_col}' to numeric type")
    feature_cols : list, optional
        List of feature column names. If None, all columns except target_col are used.
Copy
    Returns:
    --------rsor
    numpy.ndarray
        Input sequences (X)feature engineering:
    numpy.ndarray
        Target values (y)
    """Validate input data before processing."""
    try:ata is None or len(data) == 0:
        logger.info(f"Creating sequences with length {seq_length}")
    
        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None: Add other required columns
            feature_cols = [col for col in data.columns if col != target_col]
    if missing_cols:
        # Extract features and targetquired columns: {missing_cols}")
        features = data[feature_cols].values
        target = data[target_col].values
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        X, y = [], []ls) == 0:
        raise ValueError("No numeric columns found in dataset")
        for i in range(len(data) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])y()].tolist()
    if nan_cols:
        logger.info(f"Created {len(X)} sequences")ns: {nan_cols}")
        return np.array(X), np.array(y)p rows with NaN
        data = data.fillna(method='ffill').fillna(method='bfill')
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        raise

def train_val_test_split(sequences, targets, val_size=0.15, test_size=0.15):
    """
    Split sequences and targets into training, validation, and test sets,
    respecting the temporal order to avoid data leakage.
Modify the model training section to include these checks:
    Parameters:
    -----------repare Data & Train Model", key="train_model"):
    sequences : numpy.ndarrayg data and training model..."):
        Input sequences with shape (n_samples, seq_length, n_features)
    targets : numpy.ndarrayValidation
        Target values with shape (n_samples,) for single-step forecasting
        or (n_samples, forecast_horizon) for multi-step forecasting
    val_size : float, optional
        Proportion of data to use for validation
    test_size : float, optionalPerforming feature engineering...")
        Proportion of data to use for testing
            
    Returns:# Ensure we have numeric data for the target column
    --------if target_col not in data.columns:
    tuple       target_col = 'Close'  # Default to Close if target not found
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """     # Convert target column to numeric if needed
    try:    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        logger.info(f"Splitting sequences with val_size={val_size}, test_size={test_size}")
            # Process features
        n = len(sequences), transformers = prepare_features(
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - val_size))
                include_technical=include_technical,
        X_train = sequences[:val_start_idx]_statistical,
        y_train = targets[:val_start_idx],
                normalize=normalize
        X_val = sequences[val_start_idx:test_start_idx]
        y_val = targets[val_start_idx:test_start_idx]
            # Verify processed data
        X_test = sequences[test_start_idx:]n(processed_data.select_dtypes(include=['float64', 'int64']).columns) == 0:
        y_test = targets[test_start_idx:]ric features generated during processing")
                
        logger.info(f"Split complete. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
            
    except Exception as e:s e:
        logger.error(f"Error splitting sequences: {str(e)}")l training: {str(e)}")
        raiseeturn

def normalize_data(train_data, val_data, test_data):
    """
    Normalize features using StandardScaler, fitting only on training data
    to prevent data leakage.
python
    Parameters:ing for specific cases:
    -----------
    train_data : numpy.ndarray
        Training data with shape (n_samples, seq_length, n_features)
    val_data : numpy.ndarray
        Validation data with shape (n_samples, seq_length, n_features)
    test_data : numpy.ndarrayn', 'High', 'Low', 'Close', 'Adj Close']
        Test data with shape (n_samples, seq_length, n_features)
            if col in data.columns:
    Returns:    # Remove any currency symbols and commas
    --------    if data[col].dtype == 'object':
    tuple           data[col] = data[col].replace('[\$,]', '', regex=True)
        (normalized_train, normalized_val, normalized_test, scaler)erce')
    """ 
    try:# Handle percentage columns
        logger.info("Normalizing sequence data")lume'].dtype == 'object':
        from sklearn.preprocessing import StandardScaler,]', '', regex=True)
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        # Get dimensions
        n_train, seq_length, n_features = train_data.shape
    
        # Reshape to 2D for scaling
        train_reshaped = train_data.reshape(-1, n_features) {str(e)}")
        val_reshaped = val_data.reshape(-1, n_features)
        test_reshaped = test_data.reshape(-1, n_features)
Copy
        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_reshaped)
These changes will:
        # Transform all datasets
        train_normalized = scaler.transform(train_reshaped).reshape(n_train, seq_length, n_features)
        val_normalized = scaler.transform(val_reshaped).reshape(val_data.shape)
        test_normalized = scaler.transform(test_reshaped).reshape(test_data.shape)

        logger.info("Data normalization complete")
        return train_normalized, val_normalized, test_normalized, scaler
Provide clear error messages
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        raise
Add proper error handling for data conversion issues
def create_multistep_sequences(data, seq_length, target_col, forecast_horizon, feature_cols=None):
    """re to call handle_data_errors() before feature engineering:
    Create sequences for multi-step time series forecasting.
data = handle_data_errors(data)
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe containing features and target
    seq_length : int
        Length of input sequences
    target_col : strthe "No numeric types to aggregate" error by ensuring your data is properly formatted before any aggregation operations are performed.
        Name of the target column
    forecast_horizon : intr data source to ensure it's providing the expected format and that all required columns are present and properly formatted."""
        Number of steps ahead to forecast
    feature_cols : list, optional
        List of feature column names. If None, all columns except target_col are used.
including sequence generation and train/validation/test splitting.
    Returns:
    --------
    numpy.ndarrayp
        Input sequences (X)
    numpy.ndarrayl_selection import train_test_split
        Target sequences (y) with shape (n_samples, forecast_horizon)
    """rch.utils.data import Dataset, DataLoader
    try:ogging
        logger.info(f"Creating multi-step sequences with length {seq_length} and forecast horizon {forecast_horizon}")
# Configure logging
        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
)
        # Extract features and target
        features = data[feature_cols].values
        target = data[target_col].valuesarget_col, feature_cols=None):
    """
        X, y = [], []for single-step time series forecasting.

        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length:i+seq_length+forecast_horizon])
        Dataframe containing features and target
        logger.info(f"Created {len(X)} multi-step sequences")
        return np.array(X), np.array(y)
    target_col : str
    except Exception as e: column
        logger.error(f"Error creating multi-step sequences: {str(e)}")
        raiseof feature column names. If None, all columns except target_col are used.

def time_series_train_val_test_split(data, val_size=0.15, test_size=0.15):
    """-----
    Split time series data into training, validation, and test sets,
    respecting the temporal order.
    numpy.ndarray
    Parameters:values (y)
    -----------
    data : pandas.DataFrame
        Time series dataeating sequences with length {seq_length}")
    val_size : float, optional
        Proportion of data to use for validationll columns except target_col
    test_size : float, optional:
        Proportion of data to use for testingta.columns if col != target_col]

    Returns:tract features and target
    --------ures = data[feature_cols].values
    tuplearget = data[target_col].values
        (train_data, val_data, test_data)
    """ X, y = [], []
    try:
        logger.info(f"Splitting time series data with val_size={val_size}, test_size={test_size}")
            X.append(features[i:i+seq_length])
        n = len(data)target[i+seq_length])
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - val_size))
        return np.array(X), np.array(y)
        train_data = data.iloc[:val_start_idx].copy()
        val_data = data.iloc[val_start_idx:test_start_idx].copy()
        test_data = data.iloc[test_start_idx:].copy()(e)}")
        raise
        logger.info(f"Split complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_dataal_size=0.15, test_size=0.15):
    """
    except Exception as e:rgets into training, validation, and test sets,
        logger.error(f"Error splitting time series data: {str(e)}")
        raise
    Parameters:
class TimeSeriesDataset(Dataset):
    """uences : numpy.ndarray
    PyTorch Dataset for time series data.ples, seq_length, n_features)
    """gets : numpy.ndarray
    def __init__(self, X, y):ape (n_samples,) for single-step forecasting
        """(n_samples, forecast_horizon) for multi-step forecasting
        Initialize the dataset.
        Proportion of data to use for validation
        Parameters:at, optional
        -----------of data to use for testing
        X : numpy.ndarray
            Input sequences with shape (n_samples, seq_length, n_features)
        y : numpy.ndarray
            Target values with shape (n_samples,) for single-step forecasting
            or (n_samples, forecast_horizon) for multi-step forecasting
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)ze={val_size}, test_size={test_size}")

    def __len__(self):ces)
        return len(self.X)nt(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - val_size))
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]idx]
        y_train = targets[:val_start_idx]
def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """ X_val = sequences[val_start_idx:test_start_idx]
    Create PyTorch DataLoaders for training, validation, and testing.

    Parameters:= sequences[test_start_idx:]
    -----------= targets[test_start_idx:]
    X_train, y_train : numpy.ndarray
        Training data"Split complete. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    X_val, y_val : numpy.ndarray X_val, y_val, X_test, y_test
        Validation data
    X_test, y_test : numpy.ndarray
        Test dataror(f"Error splitting sequences: {str(e)}")
    batch_size : int, optional
        Batch size for DataLoaders
def normalize_data(train_data, val_data, test_data):
    Returns:
    --------e features using StandardScaler, fitting only on training data
    tupleevent data leakage.
        (train_loader, val_loader, test_loader)
    """ameters:
    try:-------
        logger.info(f"Creating DataLoaders with batch_size={batch_size}")
        Training data with shape (n_samples, seq_length, n_features)
        # Create datasetsray
        train_dataset = TimeSeriesDataset(X_train, y_train)n_features)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)eatures)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    """
        logger.info(f"Created DataLoaders. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)} batches")
        return train_loader, val_loader, test_loader
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {str(e)}")
        raisein, seq_length, n_features = train_data.shape

def prepare_data_for_training(data, target_col, seq_length=20, forecast_horizon=1,
                             val_size=0.15, test_size=0.15, batch_size=32, feature_cols=None):
    """ val_reshaped = val_data.reshape(-1, n_features)
    Comprehensive function to prepare data for model training.

    Parameters:caler on training data only
    -----------= StandardScaler()
    data : pandas.DataFrameshaped)
        Processed data with features and target
    target_col : strall datasets
        Name of the target column.transform(train_reshaped).reshape(n_train, seq_length, n_features)
    seq_length : int, optionalr.transform(val_reshaped).reshape(val_data.shape)
        Length of input sequencestransform(test_reshaped).reshape(test_data.shape)
    forecast_horizon : int, optional
        Number of steps ahead to forecastomplete")
    val_size : float, optionald, val_normalized, test_normalized, scaler
        Proportion of data to use for validation
    test_size : float, optional
        Proportion of data to use for testing: {str(e)}")
    batch_size : int, optional
        Batch size for DataLoaders
    feature_cols : list, optionalta, seq_length, target_col, forecast_horizon, feature_cols=None):
        List of feature column names. If None, all columns except target_col are used.
    Create sequences for multi-step time series forecasting.
    Returns:
    --------rs:
    tuple------
        (train_loader, val_loader, test_loader, feature_dim)
    """ Dataframe containing features and target
    try:length : int
        logger.info("Preparing data for model training")
        et_col : str
        # Make a copy of the data
        data = data.copy()
        Number of steps ahead to forecast
        # Log initial data state
        logger.info(f"Columns before numeric conversion: {list(data.columns)}")ames. If None, all columns except target_col are used.
        logger.info(f"Data types before numeric conversion:\n{data.dtypes}")
        
        # Convert all columns to numeric, excluding the index
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            y.ndarray
        # Log data state after numeric conversion_samples, forecast_horizon)
        logger.info(f"Columns after numeric conversion: {list(data.columns)}")
        logger.info(f"Data types after numeric conversion:\n{data.dtypes}")    try:
            uences with length {seq_length} and forecast horizon {forecast_horizon}")
        # Drop any columns that are entirely non-numeric
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columnsns except target_col
        data = data[numeric_cols]f feature_cols is None:
                    feature_cols = [col for col in data.columns if col != target_col]
        # Log columns after dropping non-numeric
        logger.info(f"Columns after dropping non-numeric: {list(data.columns)}")arget
        
        # Check if target column exists in numeric columnst = data[target_col].values
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in numeric columns")
            
        # Handle NaN values in target columnlength - forecast_horizon + 1):
        if data[target_col].isnull().any():
            logger.warning(f"Target column '{target_col}' contains NaN values. Filling NaNs with forward fill.")            y.append(target[i+seq_length:i+seq_length+forecast_horizon])
            data[target_col] = data[target_col].fillna(method='ffill')
            if data[target_col].isnull().any():n(X)} multi-step sequences")
                logger.warning(f"NaNs remain in target column '{target_col}' after forward fill. Filling remaining NaNs with 0.")y)
                data[target_col] = data[target_col].fillna(0)

        # Split data into train/val/test sets
        train_data, val_data, test_data = time_series_train_val_test_split(
            data, val_size=val_size, test_size=test_size
        )

        # If feature_cols is not provided, use all columns except target_col
        if feature_cols is None:    respecting the temporal order.
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            # Ensure all feature columns are present in the data
            feature_cols = [col for col in feature_cols if col in data.columns]: pandas.DataFrame
        Time series data
        feature_dim = len(feature_cols)
        logger.info(f"Using {feature_dim} features for training")
    test_size : float, optional
        # Create sequences to use for testing
        if forecast_horizon == 1:
            # Single-step forecasting
            X_train, y_train = create_sequences(train_data, seq_length, target_col, feature_cols)    --------
            X_val, y_val = create_sequences(val_data, seq_length, target_col, feature_cols)
            X_test, y_test = create_sequences(test_data, seq_length, target_col, feature_cols), val_data, test_data)
        else:
            # Multi-step forecasting
            X_train, y_train = create_multistep_sequences(train_data, seq_length, target_col, forecast_horizon, feature_cols) with val_size={val_size}, test_size={test_size}")
            X_val, y_val = create_multistep_sequences(val_data, seq_length, target_col, forecast_horizon, feature_cols)
            X_test, y_test = create_multistep_sequences(test_data, seq_length, target_col, forecast_horizon, feature_cols)
size))
        # Create data loadersze))
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size[:val_start_idx].copy()
        )st_start_idx].copy()
        test_data = data.iloc[test_start_idx:].copy()
        logger.info("Data preparation complete")
        return train_loader, val_loader, test_loader, feature_dim}, Val: {len(val_data)}, Test: {len(test_data)}")
st_data
    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")    except Exception as e:
        raiseor splitting time series data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import os
    from feature_engineering import prepare_features    PyTorch Dataset for time series data.
    from data_acquisition import load_stock_data

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
e (n_samples, seq_length, n_features)
    # Example: Load stock data
    stock_data_path = os.path.join(raw_dir, 'stock_data.csv')t values with shape (n_samples,) for single-step forecasting
    if os.path.exists(stock_data_path):            or (n_samples, forecast_horizon) for multi-step forecasting
        stock_data = load_stock_data(stock_data_path)

        # Prepare featuresype=torch.float32)
        if stock_data is not None:
            # For this example, let's assume the first column after the index is the closing price
            if len(stock_data.columns) > 0:
                target_col = stock_data.columns[0]
__(self, idx):
                # Prepare features        return self.X[idx], self.y[idx]
                processed_data, _ = prepare_features(
                    stock_data,def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):




















                print(f"Data prepared for training. Feature dimension: {feature_dim}")                )                    batch_size=32                    forecast_horizon=1,  # Single-step forecasting                    seq_length=20,                    target_col=f'Target_5',  # Target column created by prepare_features                    processed_data,                train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(                # Prepare data for training                )                    forecast_horizon=5                    reduce_dim=False,                    normalize=True,                    include_lags=True,                    include_statistical=True,                    include_technical=False,  # Set to False since we don't have OHLCV data                    target_col=target_col,    """
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
if __name__ == "__main__":
    # Example usage
    --------
    import os
    from feature_engineering import prepare_features
    from data_acquisition import load_stock_data

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
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
            raise ValueError(f"Target column '{target_col}' not found in numeric columns")
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
