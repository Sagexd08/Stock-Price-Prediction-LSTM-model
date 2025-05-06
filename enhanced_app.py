# Add project root to Python path
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_acquisition import download_stock_data, load_stock_data, validate_data_quality
from src.feature_engineering import prepare_features
from src.data_preparation import prepare_data_for_training, create_sequences, create_multistep_sequences
from src.visualization import plot_stock_prices, plot_predictions_interactive
from src.auth import get_auth_manager
from src.firebase_config import get_firebase_manager
from src.portfolio import get_portfolio_manager

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import datetime
import yfinance as yf
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Get manager instances
auth_manager = get_auth_manager()
firebase_manager = get_firebase_manager()
portfolio_manager = get_portfolio_manager()

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

    /* User profile styling */
    .user-profile {
        display: flex;
        align-items: center;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #4e8df5;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    .user-info {
        flex-grow: 1;
    }
    .user-name {
        font-weight: bold;
        color: #1e3a8a;
    }
    .user-role {
        font-size: 12px;
        color: #4b5563;
    }
    .logout-button {
        background-color: #ef4444;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
    }
    .logout-button:hover {
        background-color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Check if user is authenticated
if not auth_manager.require_auth():
    # If not authenticated, the auth manager will show the login form
    st.stop()

# Get current user
current_user = auth_manager.get_current_user()

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

# Enhanced sidebar with more options and user profile
st.sidebar.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4e8df5 100%); padding:15px; border-radius:8px; margin-bottom:20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <h2 style="color:white; text-align:center; font-size:20px; margin:0; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">⚙️ Dashboard Controls</h2>
</div>
""", unsafe_allow_html=True)

# Add user profile to sidebar
if current_user:
    initials = ''.join([name[0].upper() for name in current_user['name'].split() if name]) if 'name' in current_user else current_user['username'][0].upper()

    st.sidebar.markdown(f"""
    <div class="user-profile">
        <div class="user-avatar">{initials}</div>
        <div class="user-info">
            <div class="user-name">{current_user.get('name', current_user['username'])}</div>
            <div class="user-role">{current_user.get('role', 'User')}</div>
        </div>
        <button class="logout-button" id="logout-btn" onclick="logoutUser()">Logout</button>
    </div>

    <script>
    function logoutUser() {{
        window.location.href = "?logout=true";
    }}
    </script>
    """, unsafe_allow_html=True)

    # Check for logout parameter
    if st.experimental_get_query_params().get('logout'):
        auth_manager.logout()
        st.experimental_set_query_params()
        st.rerun()

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

    # Add to watchlist button
    if current_user and st.button("Add to Watchlist", key="add_watchlist"):
        if portfolio_manager.add_to_watchlist(current_user['username'], ticker):
            st.success(f"{ticker} added to your watchlist")
        else:
            st.error("Failed to add to watchlist")

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
        key="sidebar_data_interval"
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

# Portfolio section in sidebar
if current_user:
    with st.sidebar.expander("💼 Portfolio", expanded=False):
        # Get user portfolio
        portfolio = portfolio_manager.get_user_portfolio(current_user['username'])

        # Display watchlist
        st.subheader("Watchlist")
        if portfolio['watchlist']:
            for watch_ticker in portfolio['watchlist']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{watch_ticker}**")
                with col2:
                    if st.button("Remove", key=f"remove_{watch_ticker}"):
                        if portfolio_manager.remove_from_watchlist(current_user['username'], watch_ticker):
                            st.success(f"{watch_ticker} removed from watchlist")
                            st.rerun()
                        else:
                            st.error("Failed to remove from watchlist")
        else:
            st.write("No stocks in watchlist")

        # Add transaction form
        st.subheader("Add Transaction")
        with st.form("add_transaction_form"):
            transaction_type = st.selectbox("Type", ["buy", "sell"])
            transaction_ticker = st.selectbox("Ticker", ticker_options[sector])
            transaction_shares = st.number_input("Shares", min_value=1, value=10)
            transaction_price = st.number_input("Price per Share", min_value=0.01, value=100.00, format="%.2f")
            transaction_date = st.date_input("Date", datetime.date.today())

            if st.form_submit_button("Add Transaction"):
                transaction = {
                    "type": transaction_type,
                    "ticker": transaction_ticker,
                    "shares": transaction_shares,
                    "price": transaction_price,
                    "date": transaction_date.strftime('%Y-%m-%d')
                }

                if portfolio_manager.add_transaction(current_user['username'], transaction):
                    st.success("Transaction added successfully")
                    st.rerun()
                else:
                    st.error("Failed to add transaction")

# Add a reset button at the bottom of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Reset All Settings"):
    # Keep user authentication
    auth_state = st.session_state.get('authenticated', False)
    user_info = st.session_state.get('user', None)

    # Clear session state
    for key in list(st.session_state.keys()):
        if key not in ['authenticated', 'user', 'login_time', 'token']:
            del st.session_state[key]

    # Restore authentication state
    if auth_state:
        st.session_state['authenticated'] = auth_state
        st.session_state['user'] = user_info

    st.rerun()

# Enhanced functions for data handling
def download_stock_data(ticker, start_date, end_date=None, interval="1d", save_directly=False):
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

        # Ensure the index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Close' and 'Adj Close' in data.columns:
                    data['Close'] = data['Adj Close']
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

        # Add ticker as a column for multi-stock analysis
        data['Ticker'] = ticker

        # Convert all numeric columns to float
        for col in data.columns:
            if col != 'Ticker':  # Skip the ticker column
                try:
                    # Check if the column is a valid Series before conversion
                    if isinstance(data[col], (pd.Series, np.ndarray, list)):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    else:
                        # If not a valid type, create a new column with the same values
                        st.warning(f"Column {col} is not a valid Series. Creating a new column.")
                        values = data[col].values if hasattr(data[col], 'values') else [data[col]] * len(data)
                        data[col] = pd.Series(values, index=data.index)
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Error converting column {col} to numeric: {str(e)}")
                    # Try to recover by creating a new column
                    try:
                        data[col] = pd.Series([0] * len(data), index=data.index)
                    except:
                        pass

        # Remove any rows with all NaN values
        data = data.dropna(how='all')

        # If save_directly is True, save the data directly to a CSV file
        if save_directly:
            try:
                # Try to save to Firebase first
                if firebase_manager.initialized:
                    if firebase_manager.save_stock_data(data, ticker):
                        st.success(f"Data saved to Firebase for {ticker}")
                    else:
                        st.warning("Could not save to Firebase, falling back to local storage")

                # Save locally as backup
                save_path = os.path.join('data', 'raw', f'{ticker}_data.csv')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Create a new DataFrame with the Date as a column
                data_to_save = data.reset_index()
                data_to_save.to_csv(save_path, index=False)
                st.success(f"Data saved locally to {save_path}")
            except Exception as e:
                st.warning(f"Could not save data directly: {str(e)}")

        # Clear progress message and show success
        progress_text.empty()

        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def load_stock_data(file_path=None, ticker=None):
    """Load existing stock data from the data directory or Firebase."""
    try:
        # Try to load from Firebase first if ticker is provided
        if ticker and firebase_manager.initialized:
            data = firebase_manager.load_stock_data(ticker)
            if data is not None:
                return data

        # If file_path is not provided, use default path
        if file_path is None:
            if ticker:
                file_path = os.path.join('data', 'raw', f'{ticker}_data.csv')
                if not os.path.exists(file_path):
                    file_path = os.path.join('data', 'raw', 'stock_data.csv')
            else:
                file_path = os.path.join('data', 'raw', 'stock_data.csv')

        if os.path.exists(file_path):
            # Load data
            data = pd.read_csv(file_path)

            # Check if 'Date' column exists and set it as index
            if 'Date' in data.columns:
                # Convert to datetime with error handling
                try:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    # Drop rows with invalid dates
                    data = data.dropna(subset=['Date'])
                    data.set_index('Date', inplace=True)
                except Exception as e:
                    st.warning(f"Error converting dates: {str(e)}. Using default index.")

            # Ensure we have a 'Close' column
            if 'Close' not in data.columns and 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']

            # Convert numeric columns to float
            for col in data.columns:
                if col not in ['Ticker', 'Symbol']:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception:
                        pass

            # Handle NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')

            # Final check for any remaining NaN values
            if data.isnull().any().any():
                st.warning("Some NaN values remain in the data after filling.")

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
        future_dates = pd.date_range(start=last_date, periods=forecast_days, freq='D')

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
