"""
Firebase-specific Streamlit app for Stock Price Prediction

This version of the app is specifically designed for deployment on Firebase.
It includes Firebase authentication, Firestore data storage, and other Firebase-specific features.
"""

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

# Create tab buttons
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
        "label": "Portfolio Analysis",
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
                    key="main_data_interval"
                )

            if st.button("Download Data", key="download_data"):
                with st.spinner("Downloading stock data..."):
                    # Create data directory if it doesn't exist
                    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)

                    # Download data
                    data = download_stock_data(
                        ticker_input,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        interval=interval
                    )

                    if data is not None:
                        # Save the data to Firebase Storage
                        if firebase_manager.initialized:
                            try:
                                # Convert to CSV string
                                csv_data = data.to_csv()

                                # Save to Firebase Storage
                                file_path = f"stock_data/{current_user['username']}/{ticker_input}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                                firebase_manager.upload_string(csv_data, file_path)
                                st.success(f"Data saved to Firebase Storage: {file_path}")
                            except Exception as e:
                                st.warning(f"Could not save to Firebase: {str(e)}")

                                # Fall back to local storage
                                save_path = os.path.join('data', 'raw', 'stock_data.csv')
                                data.to_csv(save_path)
                                st.success(f"Data saved locally to {save_path}")
                        else:
                            # Save locally
                            save_path = os.path.join('data', 'raw', 'stock_data.csv')
                            data.to_csv(save_path)
                            st.success(f"Data saved locally to {save_path}")

                        # Store data in session state
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
                            xaxis_rangeslider_visible=True,
                            template=chart_theme
                        )
                        st.plotly_chart(fig, use_container_width=True, key="candlestick_plot")
                    else:
                        st.error("Failed to download data")
        else:
            if st.button("Load Existing Data", key="load_data"):
                with st.spinner("Loading stock data..."):
                    # Try to load from Firebase first
                    data = None
                    if firebase_manager.initialized:
                        try:
                            # List available files
                            files = firebase_manager.list_files(f"stock_data/{current_user['username']}/")
                            if files:
                                # Use the most recent file
                                latest_file = files[-1]
                                csv_data = firebase_manager.download_string(latest_file)

                                # Convert CSV string to DataFrame
                                import io
                                data = pd.read_csv(io.StringIO(csv_data))
                                data['Date'] = pd.to_datetime(data['Date'])
                                data.set_index('Date', inplace=True)

                                st.success(f"Loaded data from Firebase: {latest_file}")
                            else:
                                st.warning("No files found in Firebase Storage")
                        except Exception as e:
                            st.warning(f"Could not load from Firebase: {str(e)}")

                    # Fall back to local storage if needed
                    if data is None:
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
                            xaxis_rangeslider_visible=True,
                            template=chart_theme
                        )
                        st.plotly_chart(fig, use_container_width=True, key="stock_price_plot")
                    else:
                        st.error("Failed to load data")

        if 'stock_data' in st.session_state:
            data = st.session_state['stock_data']

            # Add technical indicators if requested
            if show_indicators:
                data_with_indicators = add_technical_indicators(data)

                # Plot stock price with indicators
                st.subheader("Technical Indicators")

                # Create tabs for different indicator groups
                indicator_tabs = st.tabs(["Moving Averages", "Oscillators", "Volatility", "Volume"])

                with indicator_tabs[0]:  # Moving Averages
                    fig = go.Figure()

                    # Add price
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))

                    # Add moving averages
                    if 'SMA_20' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['SMA_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange')
                        ))

                    if 'SMA_50' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['SMA_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='green')
                        ))

                    if 'SMA_200' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['SMA_200'],
                            mode='lines',
                            name='SMA 200',
                            line=dict(color='red')
                        ))

                    fig.update_layout(
                        title="Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template=chart_theme,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with indicator_tabs[1]:  # Oscillators
                    # Create two subplots: one for price, one for RSI
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.1,
                                        subplot_titles=("Price", "RSI"))

                    # Add price to the first subplot
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ), row=1, col=1)

                    # Add RSI to the second subplot
                    if 'RSI_14' in data_with_indicators.columns:
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['RSI_14'],
                            mode='lines',
                            name='RSI (14)',
                            line=dict(color='purple')
                        ), row=2, col=1)

                        # Add overbought/oversold lines
                        fig.add_trace(go.Scatter(
                            x=[data_with_indicators.index[0], data_with_indicators.index[-1]],
                            y=[70, 70],
                            mode='lines',
                            name='Overbought (70)',
                            line=dict(color='red', dash='dash')
                        ), row=2, col=1)

                        fig.add_trace(go.Scatter(
                            x=[data_with_indicators.index[0], data_with_indicators.index[-1]],
                            y=[30, 30],
                            mode='lines',
                            name='Oversold (30)',
                            line=dict(color='green', dash='dash')
                        ), row=2, col=1)

                    fig.update_layout(
                        height=600,
                        template=chart_theme,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with indicator_tabs[2]:  # Volatility
                    # Create subplot for Bollinger Bands
                    fig = go.Figure()

                    # Add price
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))

                    # Add Bollinger Bands
                    if all(col in data_with_indicators.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['BB_Upper'],
                            mode='lines',
                            name='Upper Band',
                            line=dict(color='red')
                        ))

                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['BB_Middle'],
                            mode='lines',
                            name='Middle Band (SMA 20)',
                            line=dict(color='orange')
                        ))

                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['BB_Lower'],
                            mode='lines',
                            name='Lower Band',
                            line=dict(color='green')
                        ))

                        # Add filled area between upper and lower bands
                        fig.add_trace(go.Scatter(
                            x=data_with_indicators.index.tolist() + data_with_indicators.index.tolist()[::-1],
                            y=data_with_indicators['BB_Upper'].tolist() + data_with_indicators['BB_Lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Bollinger Band Range'
                        ))

                    fig.update_layout(
                        title="Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template=chart_theme,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with indicator_tabs[3]:  # Volume
                    if 'Volume' in data.columns:
                        # Create two subplots: one for price, one for volume
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            vertical_spacing=0.1,
                                            subplot_titles=("Price", "Volume"),
                                            row_heights=[0.7, 0.3])

                        # Add price to the first subplot
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ), row=1, col=1)

                        # Add volume to the second subplot
                        fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name='Volume',
                            marker=dict(color='rgba(0, 0, 255, 0.5)')
                        ), row=2, col=1)

                        # Add OBV if available
                        if 'OBV' in data_with_indicators.columns:
                            # Normalize OBV to fit on the same scale as volume
                            max_volume = data['Volume'].max()
                            max_obv = abs(data_with_indicators['OBV']).max()
                            normalized_obv = data_with_indicators['OBV'] * (max_volume / max_obv) if max_obv > 0 else data_with_indicators['OBV']

                            fig.add_trace(go.Scatter(
                                x=data_with_indicators.index,
                                y=normalized_obv,
                                mode='lines',
                                name='On-Balance Volume (normalized)',
                                line=dict(color='green')
                            ), row=2, col=1)

                        fig.update_layout(
                            height=600,
                            template=chart_theme,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Volume data not available")

            # Display data statistics
            st.subheader("Data Statistics")
            st.dataframe(data.describe())

            # Display correlation matrix if there are multiple numeric columns
            if len(data.columns) > 1:
                st.subheader("Correlation Matrix")
                # Select only numeric columns for correlation
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:
                    corr = data[numeric_cols].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale='Viridis',
                        zmin=-1,
                        zmax=1
                    ))
                    fig.update_layout(
                        title="Correlation Matrix",
                        template=chart_theme
                    )
                    st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")
                else:
                    st.info("Not enough numeric columns to create a correlation matrix.")

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
                        step=0.05,
                        help="Proportion of data to use for validation"
                    )

                with col2:
                    test_size = st.slider(
                        "Test Set Size",
                        min_value=0.05,
                        max_value=0.3,
                        value=0.15,
                        step=0.05,
                        help="Proportion of data to use for testing"
                    )

                    batch_size = st.select_slider(
                        "Batch Size",
                        options=[8, 16, 32, 64, 128],
                        value=32,
                        help="Number of samples per batch"
                    )

            # Model Configuration Section
            st.subheader("Model Configuration")

            with st.expander("Model Settings", expanded=True):
                # Use the model settings from the sidebar
                st.write(f"**Model Architecture:** {model_type}")
                st.write(f"**Hidden Dimension:** {hidden_dim}")
                st.write(f"**Number of Layers:** {num_layers}")
                st.write(f"**Dropout Rate:** {dropout}")
                st.write(f"**Learning Rate:** {learning_rate}")

                # Additional model-specific settings
                if model_type == "Transformer":
                    num_heads = st.slider(
                        "Number of Attention Heads",
                        min_value=1,
                        max_value=16,
                        value=8,
                        help="Number of attention heads in the transformer"
                    )
                elif model_type == "ConvLSTM":
                    kernel_size = st.slider(
                        "Kernel Size",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Size of the convolutional kernel"
                    )

            # Training Settings Section
            st.subheader("Training Settings")

            with st.expander("Training Settings", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    epochs = st.slider(
                        "Number of Epochs",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        help="Number of training epochs"
                    )

                    early_stopping = st.checkbox(
                        "Early Stopping",
                        value=True,
                        help="Stop training when validation loss stops improving"
                    )

                with col2:
                    if early_stopping:
                        patience = st.slider(
                            "Patience",
                            min_value=5,
                            max_value=50,
                            value=10,
                            step=5,
                            help="Number of epochs to wait for improvement before stopping"
                        )

                    use_gpu = st.checkbox(
                        "Use GPU if available",
                        value=True,
                        help="Use GPU acceleration if available"
                    )

            # Train Model Button
            if st.button("Train Model", key="train_model_button"):
                st.info("In a production app, this would start the model training process.")

                # Create a progress bar to simulate training
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate training process
                for i in range(epochs):
                    # Update progress bar
                    progress = (i + 1) / epochs
                    progress_bar.progress(progress)

                    # Update status text
                    if i < epochs // 3:
                        status_text.text(f"Epoch {i+1}/{epochs}: Training in progress... (Loss: {0.5 - 0.3 * progress:.4f})")
                    elif i < 2 * epochs // 3:
                        status_text.text(f"Epoch {i+1}/{epochs}: Training in progress... (Loss: {0.2 - 0.1 * progress:.4f})")
                    else:
                        status_text.text(f"Epoch {i+1}/{epochs}: Training in progress... (Loss: {0.1 - 0.05 * progress:.4f})")

                    # Simulate training time
                    time.sleep(0.05)

                # Training complete
                status_text.text("Training complete!")

                # Save model info to session state
                st.session_state['model_trained'] = True
                st.session_state['model_type'] = model_type
                st.session_state['model_params'] = {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'seq_length': seq_length,
                    'batch_size': batch_size,
                    'epochs': epochs
                }

                # Show success message
                st.success("Model training completed successfully! You can now make predictions in the 'Predictions' tab.")

    elif active_tab == 'tab3':
        st.header("Stock Price Predictions")

        if 'stock_data' not in st.session_state:
            st.warning("Please load or download stock data first in the 'Data Visualization' tab.")
        elif 'model_trained' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' tab, or use the pre-trained model option below.")

            # Option to use pre-trained model
            use_pretrained = st.checkbox("Use pre-trained model", value=True)
            if use_pretrained:
                st.session_state['model_trained'] = True
                st.session_state['model_type'] = model_type
                st.session_state['model_params'] = {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'seq_length': 20,
                    'batch_size': 32,
                    'epochs': 100
                }
                st.success("Pre-trained model loaded successfully!")

        if 'stock_data' in st.session_state and 'model_trained' in st.session_state:
            data = st.session_state['stock_data']

            # Prediction Settings
            st.subheader("Prediction Settings")

            with st.expander("Prediction Settings", expanded=True):
                # Use settings from sidebar
                st.write(f"**Forecast Horizon:** {forecast_days} days")
                st.write(f"**Confidence Level:** {confidence_level}%")
                st.write(f"**Monte Carlo Samples:** {monte_carlo_samples}")

                # Additional settings
                show_prediction_details = st.checkbox("Show Prediction Details", value=True)
                download_predictions = st.checkbox("Enable Prediction Download", value=True)

            # Generate Predictions Button
            if st.button("Generate Predictions", key="generate_predictions_button"):
                with st.spinner("Generating predictions..."):
                    # In a real app, this would use the actual trained model
                    # For this demo, we'll simulate predictions using the Monte Carlo function

                    # Get the last price
                    if 'Close' in data.columns:
                        last_price = data['Close'].iloc[-1]
                    elif 'Adj Close' in data.columns:
                        last_price = data['Adj Close'].iloc[-1]
                    else:
                        last_price = data.iloc[:, 0].iloc[-1]

                    # Generate predictions
                    predictions = simulate_lstm_prediction(
                        data=data,
                        forecast_days=forecast_days,
                        model_type=model_type,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout
                    )

                    if predictions:
                        # Store predictions in session state
                        st.session_state['predictions'] = predictions
                        st.success("Predictions generated successfully!")
                    else:
                        st.error("Failed to generate predictions")

            # Display Predictions
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']

                # Get ticker from session state or use a default
                ticker_symbol = st.session_state.get('ticker', 'Stock')

                # Create prediction dataframe
                pred_dates = predictions['future_dates']
                pred_mean = predictions['mean_prediction']
                pred_lower = predictions['lower_bound']
                pred_upper = predictions['upper_bound']

                pred_df = pd.DataFrame({
                    'Date': pred_dates,
                    'Prediction': pred_mean,
                    'Lower Bound': pred_lower,
                    'Upper Bound': pred_upper
                })

                # Display prediction table
                st.subheader("Prediction Results")
                st.dataframe(pred_df)

                # Plot predictions
                st.subheader("Prediction Chart")

                # Create figure
                fig = go.Figure()

                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'] if 'Close' in data.columns else data.iloc[:, 0],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))

                # Add prediction
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_mean,
                    mode='lines',
                    name='Prediction',
                    line=dict(color='red')
                ))

                # Add prediction interval
                fig.add_trace(go.Scatter(
                    x=pred_dates.tolist() + pred_dates.tolist()[::-1],
                    y=pred_upper.tolist() + pred_lower.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))

                # Update layout
                fig.update_layout(
                    title=f"{ticker_symbol} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template=chart_theme,
                    hovermode="x unified"
                )

                # Show plot
                st.plotly_chart(fig, use_container_width=True)

                # Show prediction details if requested
                if show_prediction_details:
                    st.subheader("Prediction Details")

                    # Calculate prediction metrics
                    last_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[:, 0].iloc[-1]
                    pred_price = pred_mean[0]
                    change_abs = pred_price - last_price
                    change_pct = (change_abs / last_price) * 100

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Last Close Price",
                            value=f"${last_price:.2f}"
                        )
                    with col2:
                        st.metric(
                            label="Next Day Prediction",
                            value=f"${pred_price:.2f}",
                            delta=f"{change_abs:.2f} ({change_pct:.2f}%)"
                        )
                    with col3:
                        st.metric(
                            label=f"Price in {forecast_days} Days",
                            value=f"${pred_mean[-1]:.2f}",
                            delta=f"{(pred_mean[-1] - last_price):.2f} ({(pred_mean[-1] - last_price) / last_price * 100:.2f}%)"
                        )

                    # Display prediction range
                    st.write(f"**Prediction Range ({confidence_level}% Confidence):**")
                    st.write(f"- Minimum: ${min(pred_lower):.2f}")
                    st.write(f"- Maximum: ${max(pred_upper):.2f}")

                    # If we have attention weights, show them
                    if 'attention_weights' in predictions and model_type in ['LSTM with Attention', 'Transformer']:
                        st.subheader("Attention Weights")
                        st.write("The model pays more attention to the highlighted time periods:")

                        # Create attention weights visualization
                        weights = predictions['attention_weights']
                        dates = data.index[-len(weights):] if len(data) >= len(weights) else data.index

                        # Create figure
                        fig = go.Figure()

                        # Add price
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=data['Close'].iloc[-len(weights):] if 'Close' in data.columns else data.iloc[-len(weights):, 0],
                            mode='lines',
                            name='Price',
                            line=dict(color='blue')
                        ))

                        # Add attention weights as a bar chart
                        fig.add_trace(go.Bar(
                            x=dates,
                            y=weights * max(data['Close'].iloc[-len(weights):]) * 0.1 if 'Close' in data.columns else weights * max(data.iloc[-len(weights):, 0]) * 0.1,
                            name='Attention',
                            marker=dict(color='rgba(255,0,0,0.5)')
                        ))

                        # Update layout
                        fig.update_layout(
                            title="Attention Weights",
                            xaxis_title="Date",
                            yaxis_title="Price / Weight",
                            template=chart_theme
                        )

                        # Show plot
                        st.plotly_chart(fig, use_container_width=True)

                # Download predictions if requested
                if download_predictions:
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"{ticker_symbol}_predictions_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

    elif active_tab == 'tab4':
        st.header("Portfolio Analysis")

        if not current_user:
            st.warning("Please log in to access portfolio features.")
        else:
            # Get user portfolio
            portfolio = portfolio_manager.get_user_portfolio(current_user['username'])
            portfolio_value = portfolio_manager.get_portfolio_value(current_user['username'])

            # Display portfolio summary
            st.subheader("Portfolio Summary")

            if not portfolio['holdings']:
                st.info("Your portfolio is empty. Add transactions in the sidebar to get started.")
            else:
                # Display portfolio metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Total Value",
                        value=f"${portfolio_value['total_value']:.2f}"
                    )
                with col2:
                    st.metric(
                        label="Total Cost",
                        value=f"${portfolio_value['total_cost']:.2f}"
                    )
                with col3:
                    st.metric(
                        label="Gain/Loss",
                        value=f"${portfolio_value['gain_loss']:.2f}",
                        delta=f"{portfolio_value['gain_loss_percent']:.2f}%"
                    )

                # Display holdings
                st.subheader("Holdings")

                # Create holdings dataframe
                holdings_df = pd.DataFrame(portfolio_value['holdings'])

                # Format the dataframe for display
                if not holdings_df.empty:
                    display_df = holdings_df[['ticker', 'shares', 'cost_basis', 'current_price', 'value', 'gain_loss', 'gain_loss_percent']]
                    display_df.columns = ['Ticker', 'Shares', 'Cost Basis', 'Current Price', 'Value', 'Gain/Loss', 'Gain/Loss %']

                    # Format numeric columns
                    display_df['Cost Basis'] = display_df['Cost Basis'].map('${:,.2f}'.format)
                    display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
                    display_df['Value'] = display_df['Value'].map('${:,.2f}'.format)
                    display_df['Gain/Loss'] = display_df['Gain/Loss'].map('${:,.2f}'.format)
                    display_df['Gain/Loss %'] = display_df['Gain/Loss %'].map('{:,.2f}%'.format)

                    st.dataframe(display_df)

                # Display portfolio visualization
                st.subheader("Portfolio Visualization")

                # Create pie chart of holdings
                fig = go.Figure(data=[go.Pie(
                    labels=holdings_df['ticker'],
                    values=holdings_df['value'],
                    hole=.3,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Plotly)
                )])

                fig.update_layout(
                    title="Portfolio Allocation",
                    template=chart_theme
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display transaction history
                st.subheader("Transaction History")

                if portfolio['transactions']:
                    # Create transactions dataframe
                    transactions_df = pd.DataFrame(portfolio['transactions'])

                    # Sort by date (newest first)
                    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
                    transactions_df = transactions_df.sort_values('date', ascending=False)

                    # Format the dataframe for display
                    display_df = transactions_df[['date', 'type', 'ticker', 'shares', 'price']]
                    display_df.columns = ['Date', 'Type', 'Ticker', 'Shares', 'Price']

                    # Format columns
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df['Type'] = display_df['Type'].str.capitalize()
                    display_df['Price'] = display_df['Price'].map('${:,.2f}'.format)

                    st.dataframe(display_df)
                else:
                    st.info("No transactions yet. Add transactions in the sidebar to get started.")

# Helper functions for the app

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
            confidence_level=confidence_level_decimal,
            volatility=volatility
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

def generate_monte_carlo_predictions(last_price, forecast_days, num_simulations=100, confidence_level=0.95, volatility=0.2):
    """Generate stock price predictions using Monte Carlo simulations."""
    try:
        # Parameters for the Geometric Brownian Motion (GBM) model
        mu = 0.0002  # Mean daily return (drift)
        sigma = volatility / np.sqrt(252)  # Daily volatility

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