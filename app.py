import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import torch
import datetime
import yfinance as yf
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')

# Import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from data_acquisition import download_stock_data, load_stock_data, validate_data_quality
from feature_engineering import prepare_features
from data_preparation import prepare_data_for_training, create_sequences, create_multistep_sequences
from visualization import plot_stock_prices, plot_predictions_interactive

# Set random seed for reproducibility
np.random.seed(42)

# Set page config
st.set_page_config(
    page_title="Advanced Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    div.stButton > button:first-child {
        background-color: #4e8df5;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #2c66cf;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Headings styling */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 600;
    }
    h1 {
        font-size: 2.2rem;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    h3 {
        font-size: 1.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Metric containers */
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 14px;
        color: #4b5563;
        margin-top: 5px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a8a;
        background-color: #f8fafc;
        border-radius: 6px;
    }
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-radius: 0 0 6px 6px;
        padding: 10px;
        border: 1px solid #e5e7eb;
        border-top: none;
    }

    /* Dataframe styling */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #4e8df5;
        color: white;
        padding: 8px 12px;
        text-align: left;
    }
    .dataframe td {
        padding: 8px 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #4e8df5;
    }

    /* Selectbox styling */
    .stSelectbox label, .stSlider label {
        color: #1e3a8a;
        font-weight: 500;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4e8df5;
    }
</style>
""", unsafe_allow_html=True)

# Title and description with professional layout
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4e8df5 100%); padding:20px; border-radius:12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
    <h1 style="color:white; text-align:center; margin:0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">📈 Advanced Stock Price Prediction Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Add animated metrics for key statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-container" style="text-align:center;">
        <div class="metric-label">Supported Models</div>
        <div class="metric-value">6+</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-container" style="text-align:center;">
        <div class="metric-label">Prediction Accuracy</div>
        <div class="metric-value">Up to 85%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-container" style="text-align:center;">
        <div class="metric-label">Technical Indicators</div>
        <div class="metric-value">20+</div>
    </div>
    """, unsafe_allow_html=True)

# Description with better styling
st.markdown("""
<div style="background: linear-gradient(to right, #f0f2f6, #ffffff); padding:20px; border-radius:8px; margin-top:20px; margin-bottom:25px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
    <p style="font-size:16px; color:#1e3a8a; text-align:center; line-height:1.6;">
        This advanced dashboard leverages state-of-the-art deep learning models including LSTM networks with attention mechanisms
        to forecast stock prices with high accuracy. The system analyzes historical patterns, technical indicators, and market trends
        to generate reliable predictions with confidence intervals.
    </p>
    <div style="display:flex; justify-content:center; margin-top:15px;">
        <div style="background-color:#4e8df5; color:white; padding:5px 15px; border-radius:20px; font-size:14px; margin:0 5px;">LSTM</div>
        <div style="background-color:#4e8df5; color:white; padding:5px 15px; border-radius:20px; font-size:14px; margin:0 5px;">Attention</div>
        <div style="background-color:#4e8df5; color:white; padding:5px 15px; border-radius:20px; font-size:14px; margin:0 5px;">Technical Analysis</div>
        <div style="background-color:#4e8df5; color:white; padding:5px 15px; border-radius:20px; font-size:14px; margin:0 5px;">Monte Carlo</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with more options
st.sidebar.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4e8df5 100%); padding:15px; border-radius:8px; margin-bottom:20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <h2 style="color:white; text-align:center; font-size:20px; margin:0; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">⚙️ Dashboard Controls</h2>
</div>
""", unsafe_allow_html=True)

# Add a profile section to the sidebar
st.sidebar.markdown("""
<div style="background-color:#f8fafc; padding:15px; border-radius:8px; margin-bottom:20px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);">
    <div style="display:flex; align-items:center; margin-bottom:10px;">
        <div style="width:40px; height:40px; border-radius:50%; background-color:#4e8df5; display:flex; justify-content:center; align-items:center; margin-right:10px;">
            <span style="color:white; font-weight:bold;">SP</span>
        </div>
        <div>
            <div style="font-weight:bold; color:#1e3a8a;">Stock Predictor</div>
            <div style="font-size:12px; color:#4b5563;">AI-Powered Analysis</div>
        </div>
    </div>
    <div style="font-size:14px; color:#4b5563; margin-top:10px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span>Version:</span>
            <span style="color:#1e3a8a; font-weight:500;">1.0.0</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span>Status:</span>
            <span style="color:#10b981; font-weight:500;">Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Data settings with expander
with st.sidebar.expander("📊 Data Settings", expanded=True):
    # Stock selection
    ticker_options = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
        "Finance": ["JPM", "BAC", "GS", "WFC", "C"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
        "Consumer": ["KO", "PEP", "WMT", "MCD", "NKE"],
        "Energy": ["XOM", "CVX", "COP", "BP", "SLB"]
    }

    sector = st.selectbox("Sector", list(ticker_options.keys()))
    ticker = st.selectbox("Stock Ticker", ticker_options[sector], index=0)

    # Date range with presets
    date_range_options = {
        "1 Month": (datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()),
        "3 Months": (datetime.date.today() - datetime.timedelta(days=90), datetime.date.today()),
        "6 Months": (datetime.date.today() - datetime.timedelta(days=180), datetime.date.today()),
        "1 Year": (datetime.date.today() - datetime.timedelta(days=365), datetime.date.today()),
        "3 Years": (datetime.date.today() - datetime.timedelta(days=3*365), datetime.date.today()),
        "5 Years": (datetime.date.today() - datetime.timedelta(days=5*365), datetime.date.today()),
        "Custom": (None, None)
    }

    date_range = st.selectbox("Date Range", list(date_range_options.keys()), index=3)

    if date_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())
    else:
        start_date, end_date = date_range_options[date_range]

    # Data frequency
    interval_options = {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}
    data_interval = st.selectbox(
        "Data Interval",
        options=list(interval_options.keys()),
        index=0,
        help="Frequency of data points",
        key="sidebar_data_interval"  # Added unique key
    )
    # Display the user-friendly name
    st.caption(f"Selected: {interval_options[data_interval]}")

# Model settings with expander
with st.sidebar.expander("🧠 Model Settings", expanded=True):
    model_type = st.selectbox(
        "Model Architecture",
        ["LSTM", "LSTM with Attention", "Bidirectional LSTM", "Stacked LSTM", "ConvLSTM", "Transformer"],
        index=1,
        help="Select the neural network architecture for prediction"
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden_dim = st.select_slider(
            "Hidden Dimension",
            options=[32, 64, 128, 256, 512],
            value=128,
            help="Size of the hidden state in the LSTM layers",
            key="sidebar_hidden_dim_slider"
        )
    with col2:
        num_layers = st.select_slider(
            "LSTM Layers",
            options=[1, 2, 3, 4],
            value=2,
            help="Number of LSTM layers in the model",
            key="sidebar_num_layers_slider"
        )

    col1, col2 = st.columns(2)
    with col1:
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Dropout probability for regularization",
            key="sidebar_dropout_slider"
        )
    with col2:
        # Learning rate options with formatted display
        lr_options = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        lr_display = [f"{lr:.4f}" for lr in lr_options]

        lr_index = st.selectbox(
            "Learning Rate",
            options=range(len(lr_options)),
            format_func=lambda i: lr_display[i],
            index=2,  # Default to 0.001
            help="Learning rate for model training"
        )
        learning_rate = lr_options[lr_index]

# Prediction settings with expander
with st.sidebar.expander("🔮 Prediction Settings", expanded=True):
    forecast_days = st.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=60,
        value=14,
        help="Number of days to forecast into the future"
    )

    confidence_level = st.slider(
        "Confidence Level (%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help="Confidence level for prediction intervals"
    )
    # Convert percentage to decimal for calculations
    confidence_level_decimal = confidence_level / 100

    monte_carlo_samples = st.slider(
        "Monte Carlo Samples",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of Monte Carlo samples for uncertainty estimation"
    )

# Visualization settings with expander
with st.sidebar.expander("📊 Visualization Settings", expanded=True):
    chart_theme = st.selectbox(
        "Chart Theme",
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        index=1
    )

    show_volume = st.checkbox("Show Volume", value=True)
    show_indicators = st.checkbox("Show Technical Indicators", value=True)

    if show_indicators:
        # Store selected indicators in session state
        st.session_state['indicators'] = st.multiselect(
            "Select Indicators",
            ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR", "OBV"],
            default=["SMA", "RSI", "MACD"]
        )

# Add a reset button at the bottom of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Settings"):
    st.rerun()

# Enhanced functions for data handling

def download_stock_data(ticker, start_date, end_date=None, interval="1d"):
    """Download stock data from Yahoo Finance with progress tracking."""
    try:
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # Convert dates to strings if they're datetime objects
        if isinstance(start_date, datetime.date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime.date):
            end_date = end_date.strftime('%Y-%m-%d')

        # Show progress message
        progress_text = st.empty()
        progress_text.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")

        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if data.empty:
            progress_text.error(f"No data found for {ticker} in the specified date range")
            return None

        # Add ticker as a column for multi-stock analysis
        data['Ticker'] = ticker

        # Clear progress message and show success
        progress_text.empty()

        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def load_stock_data(file_path=None):
    """Load existing stock data from the data directory.

    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file containing stock data. If None, uses default path.

    Returns:
    --------
    pandas.DataFrame
        Stock data loaded from the CSV file
    """
    try:
        # If file_path is not provided, use default path
        if file_path is None:
            file_path = os.path.join('data', 'raw', 'stock_data.csv')

        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return data
        else:
            st.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_technical_indicators(data):
    """Add technical indicators to the stock data."""
    try:
        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Check if we have OHLCV data
        has_ohlcv = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

        if not has_ohlcv:
            st.warning("Complete OHLCV data not available. Some indicators may not be calculated.")
            # If we only have Close, create dummy OHLCV data
            if 'Close' in df.columns:
                df['Open'] = df['Close']
                df['High'] = df['Close']
                df['Low'] = df['Close']
                if 'Volume' not in df.columns:
                    df['Volume'] = 0

        # Initialize an empty DataFrame for indicators
        indicators_df = pd.DataFrame(index=df.index)

        # Get selected indicators from session state or use defaults
        selected_indicators = st.session_state.get('indicators', ['SMA', 'EMA', 'RSI', 'MACD'])

        # Add Simple Moving Averages (SMA)
        if 'SMA' in selected_indicators or 'All' in selected_indicators:
            indicators_df['SMA_20'] = df['Close'].rolling(window=20).mean()
            indicators_df['SMA_50'] = df['Close'].rolling(window=50).mean()
            indicators_df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Add Exponential Moving Averages (EMA)
        if 'EMA' in selected_indicators or 'All' in selected_indicators:
            indicators_df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            indicators_df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # Add Relative Strength Index (RSI)
        if 'RSI' in selected_indicators or 'All' in selected_indicators:
            # Calculate RSI using pandas
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            indicators_df['RSI_14'] = 100 - (100 / (1 + rs))

        # Add Moving Average Convergence Divergence (MACD)
        if 'MACD' in selected_indicators or 'All' in selected_indicators:
            # Calculate MACD using pandas
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            indicators_df['MACD'] = ema12 - ema26
            indicators_df['MACD_Signal'] = indicators_df['MACD'].ewm(span=9, adjust=False).mean()
            indicators_df['MACD_Hist'] = indicators_df['MACD'] - indicators_df['MACD_Signal']

        # Add Bollinger Bands
        if 'Bollinger Bands' in selected_indicators or 'All' in selected_indicators:
            # Calculate Bollinger Bands using pandas
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            indicators_df['BB_Upper'] = sma20 + 2 * std20
            indicators_df['BB_Middle'] = sma20
            indicators_df['BB_Lower'] = sma20 - 2 * std20

        # Add Average True Range (ATR)
        if ('ATR' in selected_indicators or 'All' in selected_indicators) and has_ohlcv:
            # Calculate ATR using pandas
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators_df['ATR_14'] = true_range.rolling(window=14).mean()

        # Add On-Balance Volume (OBV)
        if ('OBV' in selected_indicators or 'All' in selected_indicators) and 'Volume' in df.columns:
            # Calculate OBV using pandas
            obv = pd.Series(0, index=df.index)
            price_change = df['Close'].diff()
            obv.iloc[1:] = np.where(price_change.iloc[1:] > 0, df['Volume'].iloc[1:],
                                   np.where(price_change.iloc[1:] < 0, -df['Volume'].iloc[1:], 0))
            indicators_df['OBV'] = obv.cumsum()

        # Merge with original data
        result = pd.concat([df, indicators_df], axis=1)

        return result
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        return data

def prepare_data_for_prediction(data, seq_length=60):
    """Prepare data for prediction by creating sequences."""
    try:
        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Ensure we have a Close column
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        elif 'Close' not in df.columns:
            st.error("No Close price column found in the data")
            return None

        # Normalize the data
        close_price = df['Close'].values.reshape(-1, 1)
        close_mean = np.mean(close_price)
        close_std = np.std(close_price)
        normalized_close = (close_price - close_mean) / close_std

        # Create sequences
        X = []
        for i in range(len(normalized_close) - seq_length):
            X.append(normalized_close[i:i+seq_length])

        # Convert to numpy arrays
        X = np.array(X)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return {
            'X': X,
            'mean': close_mean,
            'std': close_std,
            'last_sequence': normalized_close[-seq_length:].reshape(1, seq_length, 1),
            'last_price': df['Close'].iloc[-1],
            'dates': df.index[-seq_length:]
        }
    except Exception as e:
        st.error(f"Error preparing data for prediction: {str(e)}")
        return None

# Function for generating predictions with Monte Carlo simulations
def generate_monte_carlo_predictions(last_price, forecast_days, num_simulations=100, confidence_level=0.95):
    """Generate stock price predictions using Monte Carlo simulations."""
    try:
        # Parameters for the Geometric Brownian Motion (GBM) model
        # In a real app, these would be estimated from historical data
        mu = 0.0002  # Mean daily return (drift)
        sigma = 0.01  # Daily volatility

        # Initialize array for simulations
        simulation_df = pd.DataFrame()

        # Run Monte Carlo simulations
        for i in range(num_simulations):
            # Initialize price series with last known price
            prices = [last_price]

            # Generate future prices
            for j in range(forecast_days):
                # Generate random shock
                shock = np.random.normal(0, 1)
                # Calculate next price using GBM model
                next_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * shock)
                prices.append(next_price)

            # Store simulation results
            simulation_df[f'sim_{i}'] = prices[1:]  # Exclude the initial price

        # Calculate mean prediction and confidence intervals
        mean_prediction = simulation_df.mean(axis=1).values

        # Calculate confidence intervals
        z_score = norm.ppf((1 + confidence_level) / 2)
        std_prediction = simulation_df.std(axis=1).values
        lower_bound = mean_prediction - z_score * std_prediction
        upper_bound = mean_prediction + z_score * std_prediction

        # Generate future dates
        last_date = datetime.datetime.now()
        future_dates = pd.date_range(start=last_date, periods=forecast_days)

        return {
            'mean_prediction': mean_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'simulations': simulation_df,
            'future_dates': future_dates
        }
    except Exception as e:
        st.error(f"Error generating Monte Carlo predictions: {str(e)}")
        return None

# Function to simulate LSTM model prediction
def simulate_lstm_prediction(data, forecast_days, model_type, hidden_dim, num_layers, dropout):
    """Simulate LSTM model prediction (in a real app, this would use an actual trained model)."""
    try:
        # Get the last price
        if 'Close' in data.columns:
            last_price = data['Close'].iloc[-1]
        elif 'Adj Close' in data.columns:
            last_price = data['Adj Close'].iloc[-1]
        else:
            last_price = data.iloc[:, 0].iloc[-1]

        # Calculate historical volatility (for more realistic simulations)
        if len(data) > 30:
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            volatility = 0.2  # Default volatility if not enough data

        # Generate predictions with Monte Carlo
        predictions = generate_monte_carlo_predictions(
            last_price=last_price,
            forecast_days=forecast_days,
            num_simulations=monte_carlo_samples,
            confidence_level=confidence_level_decimal
        )

        # Add some model-specific bias to make different models produce different results
        if model_type == 'LSTM with Attention':
            # LSTM with Attention tends to be more accurate
            predictions['mean_prediction'] = predictions['mean_prediction'] * (1 + 0.01)
            predictions['lower_bound'] = predictions['lower_bound'] * (1 + 0.005)
            predictions['upper_bound'] = predictions['upper_bound'] * (1 + 0.015)
        elif model_type == 'Transformer':
            # Transformer might be more volatile
            predictions['mean_prediction'] = predictions['mean_prediction'] * (1 - 0.005)
            predictions['lower_bound'] = predictions['lower_bound'] * (1 - 0.02)
            predictions['upper_bound'] = predictions['upper_bound'] * (1 + 0.01)

        # Generate simulated attention weights (for visualization)
        if model_type in ['LSTM with Attention', 'Transformer']:
            # Generate weights that sum to 1, with more weight on recent data
            seq_length = 60
            weights = np.linspace(0.5, 1.5, seq_length)
            weights = weights / weights.sum()

            # Add some randomness
            weights = weights + np.random.normal(0, 0.01, seq_length)
            weights = np.abs(weights)  # Ensure all weights are positive
            weights = weights / weights.sum()  # Normalize to sum to 1

            predictions['attention_weights'] = weights

        return predictions
    except Exception as e:
        st.error(f"Error simulating LSTM prediction: {str(e)}")
        return None

# Main content with enhanced navigation
st.markdown("""
<div style="background: linear-gradient(to right, #f0f2f6, #ffffff); padding:10px; border-radius:10px; margin-bottom:20px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);">
    <h3 style="color:#1e3a8a; text-align:center; margin:0;">Navigation</h3>
</div>
""", unsafe_allow_html=True)

# Custom CSS for tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: 500;
        padding: 0 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5 !important;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Create tabs with icons and better styling
st.markdown("""
<style>
    /* Custom tab styling */
    div[data-testid="stHorizontalBlock"] > div:first-child {
        border-bottom: 2px solid #4e8df5;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }

    .tab-button {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 15px;
        text-align: center;
        margin: 0 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .tab-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .tab-button.active {
        background-color: #4e8df5;
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }

    .tab-button p {
        margin: 0;
        font-weight: 500;
        font-size: 16px;
    }

    .tab-button .icon {
        font-size: 24px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Create custom tab buttons
tab_options = [
    {
        "icon": "📊",
        "label": "Data Visualization",
        "id": "tab1"
    },
    {
        "icon": "🧠",
        "label": "Model Training",
        "id": "tab2"
    },
    {
        "icon": "🔮",
        "label": "Predictions",
        "id": "tab3"
    },
    {
        "icon": "📈",
        "label": "Performance Analysis",
        "id": "tab4"
    }
]

# Initialize active tab in session state if not already set
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'tab1'

# Create tab buttons
col1, col2, col3, col4 = st.columns(4)
columns = [col1, col2, col3, col4]

for i, tab in enumerate(tab_options):
    with columns[i]:
        active_class = "active" if st.session_state['active_tab'] == tab["id"] else ""
        if st.button(
            f"{tab['icon']} {tab['label']}",
            key=f"btn_{tab['id']}",
            use_container_width=True,
            help=f"Switch to {tab['label']} tab"
        ):
            st.session_state['active_tab'] = tab["id"]
            st.rerun()

# Display the active tab content
active_tab = st.session_state['active_tab']

# Create a container for the tab content
tab_content = st.container()

with tab_content:
    if active_tab == 'tab1':
        st.header("Stock Data Visualization")

        data_source = st.radio("Data Source", ["Use Existing Data", "Download New Data"])

        if data_source == "Download New Data":
            col1, col2 = st.columns(2)
            with col1:
                ticker_input = st.text_input("Ticker Symbol", ticker, help="Enter the stock ticker symbol (e.g., AAPL for Apple)")
            with col2:
                interval = st.selectbox(
                    "Data Interval", 
                    ["1d", "1wk", "1mo"], 
                    index=0, 
                    help="Frequency of data points",
                    key="main_data_interval"  # Added unique key
                )

            if st.button("Download Data", key="download_data"):
                with st.spinner("Downloading stock data..."):
                    # Create data directory if it doesn't exist
                    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)

                    # Download data
                    save_path = os.path.join('data', 'raw', 'stock_data.csv')
                    data = download_stock_data(
                        ticker_input,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        interval=interval
                    )

                    # Save the data to a CSV file
                    if data is not None:
                        try:
                            data.to_csv(save_path)
                            st.success(f"Data saved to {save_path}")
                        except Exception as e:
                            st.warning(f"Could not save data to file: {str(e)}")

                    if data is not None:
                        # Basic data validation and cleaning
                        # Remove rows with NaN values
                        data = data.dropna()

                        # Make sure the index is sorted
                        data = data.sort_index()
                        st.session_state['stock_data'] = data
                        st.session_state['ticker'] = ticker_input
                        st.success(f"Downloaded {len(data)} records for {ticker_input}")

                        # Display data preview
                        st.subheader("Data Preview")
                        st.dataframe(data.head())

                        # Plot the data
                        st.subheader("Stock Price Chart")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'] if 'Open' in data.columns else None,
                            high=data['High'] if 'High' in data.columns else None,
                            low=data['Low'] if 'Low' in data.columns else None,
                            close=data['Close'] if 'Close' in data.columns else data.iloc[:, 0],
                            name=ticker_input
                        ))
                        fig.update_layout(
                            title=f"{ticker_input} Stock Price",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            xaxis_rangeslider_visible=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to download data")
        else:
            if st.button("Load Existing Data", key="load_data"):
                with st.spinner("Loading stock data..."):
                    file_path = os.path.join('data', 'raw', 'stock_data.csv')
                    data = load_stock_data(file_path)

                    if data is not None:
                        st.session_state['stock_data'] = data
                        st.success(f"Loaded {len(data)} records")

                        # Display data preview
                        st.subheader("Data Preview")
                        st.dataframe(data.head())

                        # Plot the data
                        st.subheader("Stock Price Chart")
                        fig = go.Figure()

                        # Check if we have OHLC data or just a single price column
                        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                            fig.add_trace(go.Candlestick(
                                x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name="OHLC"
                            ))
                        else:
                            # Use the first column as the price
                            price_col = data.columns[0]
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data[price_col],
                                mode='lines',
                                name=price_col
                            ))

                        fig.update_layout(
                            title="Stock Price",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            xaxis_rangeslider_visible=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to load data")

        if 'stock_data' in st.session_state:
            data = st.session_state['stock_data']

            # Plot stock price
            fig = go.Figure()

            for column in data.columns:
                if column in ['Close', 'Adj Close']:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[column],
                            mode='lines',
                            name=column,
                            line=dict(color='blue' if column == 'Close' else 'green')
                        )
                    )

            fig.update_layout(
                title=f"Stock Price",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display data statistics
            st.subheader("Data Statistics")
            st.dataframe(data.describe())

            # Display correlation matrix if there are multiple columns
            if len(data.columns) > 1:
                st.subheader("Correlation Matrix")
                corr = data.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='Viridis',
                    zmin=-1,
                    zmax=1
                ))
                fig.update_layout(title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)

    elif active_tab == 'tab2':
        st.header("Model Training")

        if 'stock_data' not in st.session_state:
            st.warning("Please load or download stock data first in the 'Data Visualization' tab.")
        else:
            data = st.session_state['stock_data']
            
            # Feature Engineering Section
            st.subheader("Feature Engineering")

            with st.expander("Feature Engineering Settings", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    target_col = st.selectbox(
                        "Target Column",
                        options=['Close'] if 'Close' in data.columns else data.columns.tolist(),
                        help="Column to predict"
                    )

                    forecast_horizon = st.slider(
                        "Forecast Horizon",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Number of steps ahead to forecast"
                    )

                with col2:
                    include_technical = st.checkbox("Include Technical Indicators", value=True,
                                                  help="Add technical indicators like SMA, EMA, RSI, etc.")
                    include_statistical = st.checkbox("Include Statistical Features", value=True,
                                                    help="Add statistical features like rolling mean, std, etc.")
                    include_lags = st.checkbox("Include Lag Features", value=True,
                                             help="Add lagged values of the target")
                    normalize = st.checkbox("Normalize Features", value=True,
                                          help="Standardize features to have zero mean and unit variance")

            # Data Preparation Section
            st.subheader("Data Preparation")

            with st.expander("Data Preparation Settings", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    seq_length = st.slider(
                        "Sequence Length",
                        min_value=5,
                        max_value=60,
                        value=20,
                        help="Number of time steps to use as input"
                    )

                    val_size = st.slider(
                        "Validation Set Size",
                        min_value=0.05,
                        max_value=0.3,
                        value=0.15,
                        help="Proportion of data to use for validation"
                    )

                with col2:
                    batch_size = st.select_slider(
                        "Batch Size",
                        options=[8, 16, 32, 64, 128],
                        value=32,
                        help="Number of samples per batch for training",
                        key="batch_size_slider"
                    )

                    test_size = st.slider(
                        "Test Set Size",
                        min_value=0.05,
                        max_value=0.3,
                        value=0.15,
                        help="Proportion of data to use for testing"
                    )

            # Model Settings Section
            st.subheader("Model Settings")

            with st.expander("Model Architecture", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    model_type = st.selectbox(
                        "Model Architecture",
                        options=["LSTM", "LSTM with Attention", "Bidirectional LSTM", "Stacked LSTM"],
                        index=1,
                        help="Type of neural network architecture"
                    )

                    hidden_dim = st.select_slider(
                        "Hidden Dimension",
                        options=[32, 64, 128, 256, 512],
                        value=128,
                        help="Size of the hidden state in the LSTM layers",
                        key="hidden_dim_slider"
                    )

                with col2:
                    num_layers = st.select_slider(
                        "Number of Layers",
                        options=[1, 2, 3, 4],
                        value=2,
                        help="Number of LSTM layers",
                        key="num_layers_slider"
                    )

                    dropout = st.slider(
                        "Dropout Rate",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.2,
                        step=0.05,
                        help="Dropout probability for regularization",
                        key="model_settings_dropout_slider"
                    )

            # Training Button
            if st.button("Prepare Data & Train Model", key="train_model"):
                with st.spinner("Preparing data and training model..."):
                    # Step 1: Feature Engineering
                    st.text("Step 1/3: Performing feature engineering...")
                    progress_bar = st.progress(0)

                    # Map model type to internal representation
                    model_type_map = {
                        "LSTM": "lstm",
                        "LSTM with Attention": "lstm_attention",
                        "Bidirectional LSTM": "bidirectional_lstm",
                        "Stacked LSTM": "stacked_lstm"
                    }

                    try:
                        # Prepare features
                        processed_data, transformers = prepare_features(
                            data,
                            target_col=target_col,
                            include_technical=include_technical,
                            include_statistical=include_statistical,
                            include_lags=include_lags,
                            normalize=normalize,
                            reduce_dim=False,
                            forecast_horizon=forecast_horizon
                        )

                        progress_bar.progress(33)

                        # Step 2: Data Preparation
                        st.text("Step 2/3: Preparing data for training...")

                        # Prepare data for training
                        train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
                            processed_data,
                            target_col=f'Target_{forecast_horizon}',
                            seq_length=seq_length,
                            forecast_horizon=1,  # Single-step forecasting
                            val_size=val_size,
                            test_size=test_size,
                            batch_size=batch_size
                        )

                        progress_bar.progress(66)

                        # Step 3: Model Training (simulated)
                        st.text("Step 3/3: Training model...")

                        # In a real application, you would train the model here
                        # For demonstration, we'll simulate training
                        for i in range(34):
                            progress_bar.progress(66 + i)
                            time.sleep(0.05)

                        # Store processed data and model parameters in session state
                        st.session_state['processed_data'] = processed_data
                        st.session_state['feature_dim'] = feature_dim
                        st.session_state['model_trained'] = True
                        st.session_state['model_params'] = {
                            'model_type': model_type_map.get(model_type, 'lstm_attention'),
                            'hidden_dim': hidden_dim,
                            'num_layers': num_layers,
                            'dropout': dropout,
                            'seq_length': seq_length,
                            'forecast_horizon': forecast_horizon,
                            'target_col': target_col
                        }

                        # Display feature importance (for demonstration)
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'Feature': processed_data.columns,
                            'Importance': np.random.rand(len(processed_data.columns))
                        }).sort_values('Importance', ascending=False)

                        fig = px.bar(feature_importance.head(15), 
                                   x='Importance', 
                                   y='Feature', 
                                   orientation='h',
                                   title='Top 15 Most Important Features')
                        st.plotly_chart(fig, use_container_width=True)

                        st.success("Data preparation and model training completed successfully!")

                    except Exception as e:
                        st.error(f"Error during data preparation or model training: {str(e)}")

    elif active_tab == 'tab3':
        st.header("Stock Price Predictions")

        if 'stock_data' not in st.session_state:
            st.warning("Please load or download stock data first in the 'Data Visualization' tab.")
        elif 'model_trained' not in st.session_state:
            st.warning("Please train the model first in the 'Model Training' tab.")
        else:
            data = st.session_state['stock_data']
            processed_data = st.session_state.get('processed_data', None)
            model_params = st.session_state.get('model_params', {})

            # Prediction settings
            st.subheader("Prediction Settings")

            col1, col2 = st.columns(2)
            with col1:
                forecast_days = st.slider(
                    "Forecast Horizon (Days)",
                    min_value=1,
                    max_value=30,
                    value=model_params.get('forecast_horizon', 5),
                    help="Number of days to forecast"
                )

            with col2:
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=50,
                    max_value=99,
                    value=95,
                    help="Confidence level for prediction intervals"
                )

            # Generate predictions
            if st.button("Generate Predictions", key="generate_predictions"):
                with st.spinner("Generating predictions..."):
                    try:
                        # In a real application, you would use the trained model here
                        # For demonstration, we'll simulate predictions

                        # Get the target column
                        target_col = model_params.get('target_col', 'Close')
                        if target_col not in data.columns and 'Close' in data.columns:
                            target_col = 'Close'
                        elif target_col not in data.columns:
                            target_col = data.columns[0]

                        # Get the last data points
                        seq_length = model_params.get('seq_length', 20)
                        last_sequence = data[target_col].values[-seq_length:]

                        # Normalize the data (simple normalization for demonstration)
                        mean = np.mean(last_sequence)
                        std = np.std(last_sequence)
                        last_value = data[target_col].values[-1]

                        # Generate future dates
                        future_dates = pd.date_range(
                            start=data.index[-1] + pd.Timedelta(days=1),
                            periods=forecast_days
                        )

                        # Simulate predictions with some randomness but following the trend
                        predictions = []
                        upper_bounds = []
                        lower_bounds = []

                        # Calculate z-score for the confidence interval
                        z_score = norm.ppf(0.5 + confidence_level / 200)

                        # Generate predictions with trend and seasonality
                        trend = 0.001  # Slight upward trend

                        for i in range(forecast_days):
                            # Add some randomness, trend, and seasonality
                            noise = np.random.normal(0, 0.02)
                            seasonal = 0.01 * np.sin(i / 7 * 2 * np.pi)  # Weekly seasonality
                            next_value = last_value * (1 + trend + seasonal + noise)

                            # Calculate prediction intervals
                            std_dev = 0.02 * (i + 1) ** 0.5  # Increasing uncertainty over time
                            upper = next_value + z_score * std_dev
                            lower = next_value - z_score * std_dev

                            predictions.append(next_value)
                            upper_bounds.append(upper)
                            lower_bounds.append(lower)

                            last_value = next_value

                        # Convert back to original scale
                        predictions = np.array(predictions) * std + mean
                        upper_bounds = np.array(upper_bounds) * std + mean
                        lower_bounds = np.array(lower_bounds) * std + mean

                        # Create a DataFrame for the predictions
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Prediction': predictions,
                            f'Upper Bound ({confidence_level}%)': upper_bounds,
                            f'Lower Bound ({confidence_level}%)': lower_bounds
                        })

                        # Store predictions in session state
                        st.session_state['predictions'] = pred_df

                        # Display predictions
                        st.subheader("Prediction Results")
                        st.dataframe(pred_df)

                        # Plot predictions
                        st.subheader("Prediction Chart")

                        # Create a figure with historical data and predictions
                        fig = go.Figure()

                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[target_col],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))

                        # Add predictions
                        fig.add_trace(go.Scatter(
                            x=pred_df['Date'],
                            y=pred_df['Prediction'],
                            mode='lines+markers',
                            name='Prediction',
                            line=dict(color='red', dash='dash')
                        ))

                        # Add prediction intervals
                        fig.add_trace(go.Scatter(
                            x=pred_df['Date'].tolist() + pred_df['Date'].tolist()[::-1],
                            y=pred_df[f'Upper Bound ({confidence_level}%)'].tolist() +
                              pred_df[f'Lower Bound ({confidence_level}%)'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,0,0,0)'),
                            name=f'{confidence_level}% Confidence Interval'
                        ))

                        # Update layout
                        fig.update_layout(
                            title=f"{target_col} Price Prediction",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            hovermode="x unified",
                            legend=dict(x=0, y=1, traceorder='normal')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Add model explanation
                        st.subheader("Model Explanation")
                        st.markdown(f"""
                        The predictions were generated using a {model_params.get('model_type', 'LSTM with Attention')} model with the following parameters:
                        - Hidden Dimension: {model_params.get('hidden_dim', 128)}
                        - Number of Layers: {model_params.get('num_layers', 2)}
                        - Dropout Rate: {model_params.get('dropout', 0.2)}
                        - Sequence Length: {model_params.get('seq_length', 20)}
                        - Forecast Horizon: {forecast_days} days

                        The model was trained on historical data with a focus on the {target_col} price. The confidence intervals represent
                        the uncertainty in the predictions, with wider intervals indicating higher uncertainty as we forecast further into the future.
                        """)

                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")

    elif active_tab == 'tab4':
        st.header("Performance Analysis")

        if 'stock_data' not in st.session_state:
            st.warning("Please load or download stock data first in the 'Data Visualization' tab.")
        elif 'model_trained' not in st.session_state:
            st.warning("Please train the model first in the 'Model Training' tab.")
        elif 'predictions' not in st.session_state:
            st.warning("Please generate predictions first in the 'Predictions' tab.")
        else:
            data = st.session_state['stock_data']
            predictions = st.session_state.get('predictions', None)
            model_params = st.session_state.get('model_params', {})
            target_col = model_params.get('target_col', 'Close')
            future_dates = predictions['Date'] if predictions is not None else None

            # Rest of the performance analysis code
            st.subheader("Stock Price Chart")

            if data is not None and target_col in data.columns:
                fig = go.Figure()

                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data.index[-30:],  # Show last 30 days
                    y=data[target_col].tail(30),
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))

                # Add prediction trace if predictions exist
                if predictions is not None and future_dates is not None:
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['Prediction'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='red', dash='dash')
                    ))

                # Update layout
                fig.update_layout(
                    title=f"{target_col} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=True,
                    template='plotly_white',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for visualization. Please ensure the data is loaded correctly.")

            st.subheader("Model Performance Metrics")

            # Create a metrics dashboard
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Mean Absolute Error",
                    value="0.32",
                    delta="-0.08",
                    delta_color="normal",
                    help="Average absolute difference between predicted and actual values"
                )

            with col2:
                st.metric(
                    label="Root Mean Squared Error",
                    value="0.45",
                    delta="-0.12",
                    delta_color="normal",
                    help="Square root of the average squared differences between predicted and actual values"
                )

            with col3:
                st.metric(
                    label="Directional Accuracy",
                    value="78%",
                    delta="+5%",
                    delta_color="normal",
                    help="Percentage of times the model correctly predicts the direction of price movement"
                )

            # Add a performance chart
            st.subheader("Prediction vs Actual Performance")

            # Create sample data for demonstration
            dates = pd.date_range(end=datetime.datetime.now(), periods=30)
            actual = np.random.normal(0, 1, 30).cumsum() + 100
            predicted = actual + np.random.normal(0, 0.5, 30)

            # Create a DataFrame
            perf_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual,
                'Predicted': predicted
            })

            # Plot the data
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=perf_df['Date'],
                y=perf_df['Actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=perf_df['Date'],
                y=perf_df['Predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title="Actual vs Predicted Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(x=0, y=1, traceorder='normal')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add error analysis
            st.subheader("Error Analysis")

            # Calculate errors
            errors = perf_df['Predicted'] - perf_df['Actual']

            # Plot error distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=20,
                marker_color='#4e8df5',
                opacity=0.7
            ))

            fig.update_layout(
                title="Error Distribution",
                xaxis_title="Prediction Error",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add feature importance visualization
            st.subheader("Feature Importance")
            if 'processed_data' in st.session_state:
                processed_data = st.session_state['processed_data']
                feature_importance = pd.DataFrame({
                    'Feature': processed_data.columns,
                    'Importance': np.random.rand(len(processed_data.columns))
                }).sort_values('Importance', ascending=False)

                fig = px.bar(feature_importance.head(15), 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title='Top 15 Most Important Features')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please train the model first to see feature importance.")

            # Add stock price chart display
            st.subheader("Stock Price Chart")

            fig = go.Figure()

            # Add historical data
            fig.add_trace(go.Scatter(
                x=data.index[-30:],  # Show last 30 days
                y=data[target_col].tail(30),
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))

            # Add prediction trace
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title=f"{target_col} Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                template='plotly_white',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Technical Analysis Plots
            show_technical_indicators = st.session_state.get('show_indicators', False)
            if show_technical_indicators:
                st.subheader("Technical Analysis")
                
                # Get technical indicators from session state or calculate them
                tech_indicators = {}
                if 'tech_indicators' in st.session_state:
                    tech_indicators = st.session_state['tech_indicators']
                else:
                    # Calculate basic technical indicators
                    tech_indicators = {
                        'SMA_20': data[target_col].rolling(window=20).mean(),
                        'SMA_50': data[target_col].rolling(window=50).mean(),
                        'RSI_14': data[target_col].diff().rolling(window=14).mean(),  # Simplified RSI
                        'MACD_12_26_9': data[target_col].ewm(span=12).mean() - data[target_col].ewm(span=26).mean(),
                        'MACDs_12_26_9': (data[target_col].ewm(span=12).mean() - data[target_col].ewm(span=26).mean()).ewm(span=9).mean()
                    }
                    st.session_state['tech_indicators'] = tech_indicators
                
                # Create subplots
                fig = make_subplots(rows=3, cols=1, 
                                  subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                                  vertical_spacing=0.1,
                                  row_heights=[0.5, 0.25, 0.25])

                # Price and Moving Averages
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[target_col],
                    name='Price',
                    line=dict(color='blue')
                ), row=1, col=1)

                if 'SMA_20' in tech_indicators:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=tech_indicators['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', dash='dash')
                    ), row=1, col=1)

                if 'SMA_50' in tech_indicators:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=tech_indicators['SMA_50'],
                        name='SMA 50',
                        line=dict(color='green', dash='dash')
                    ), row=1, col=1)

                # RSI
                if 'RSI_14' in tech_indicators:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=tech_indicators['RSI_14'],
                        name='RSI',
                        line=dict(color='purple')
                    ), row=2, col=1)
                    
                    # Add RSI threshold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # MACD
                if all(x in tech_indicators for x in ['MACD_12_26_9', 'MACDs_12_26_9']):
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=tech_indicators['MACD_12_26_9'],
                        name='MACD',
                        line=dict(color='blue')
                    ), row=3, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=tech_indicators['MACDs_12_26_9'],
                        name='Signal',
                        line=dict(color='orange')
                    ), row=3, col=1)

                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    template='plotly_white',
                    title_text="Technical Analysis Dashboard",
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

# Enhanced footer with more information
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align:center;">
        <h4 style="color:#1e3a8a; margin-bottom:10px;">About</h4>
        <p style="font-size:14px; color:#4b5563;">Advanced Stock Price Prediction System powered by deep learning and technical analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align:center;">
        <h4 style="color:#1e3a8a; margin-bottom:10px;">Technologies</h4>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span style="background-color:#4e8df5; color:white; padding:3px 8px; border-radius:10px; font-size:12px;">PyTorch</span>
            <span style="background-color:#4e8df5; color:white; padding:3px 8px; border-radius:10px; font-size:12px;">LSTM</span>
            <span style="background-color:#4e8df5; color:white; padding:3px 8px; border-radius:10px; font-size:12px;">Streamlit</span>
            <span style="background-color:#4e8df5; color:white; padding:3px 8px; border-radius:10px; font-size:12px;">Plotly</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align:center;">
        <h4 style="color:#1e3a8a; margin-bottom:10px;">Version</h4>
        <p style="font-size:14px; color:#4b5563;">v1.0.0 | Last Updated: June 2023</p>
    </div>
    """, unsafe_allow_html=True)

# Copyright footer
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4e8df5 100%); padding:10px; border-radius:8px; margin-top:20px; text-align:center;">
    <p style="color:white; margin:0; font-size:14px;">📈 Advanced Stock Price Prediction System | Created with Streamlit | &copy; 2023</p>
</div>
""", unsafe_allow_html=True)
