"""
Deployment Module

This module provides functions for deploying trained models,
including API endpoints and a simple dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, timedelta
import logging
import threading
import time
import schedule

# Flask for API
from flask import Flask, request, jsonify

# Streamlit for dashboard (imported only when needed)
# import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelServer:
    """
    Class for serving the trained model via API.
    """
    def __init__(self, model_path, scaler_path=None, seq_length=20, feature_cols=None):
        """
        Initialize the model server.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        scaler_path : str, optional
            Path to the fitted scaler
        seq_length : int, optional
            Length of input sequences
        feature_cols : list, optional
            List of feature column names
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and scaler
        self.load_model()
        
    def load_model(self):
        """
        Load the trained model and scaler.
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Import here to avoid circular imports
            from model import create_model
            
            # Load model metadata
            metadata_path = os.path.join(os.path.dirname(self.model_path), 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_type = metadata.get('model_type', 'lstm')
                input_dim = metadata.get('input_dim', 10)
                hidden_dim = metadata.get('hidden_dim', 64)
                num_layers = metadata.get('num_layers', 2)
                output_dim = metadata.get('output_dim', 1)
                dropout_prob = metadata.get('dropout_prob', 0.2)
                
                # Create model
                self.model = create_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
                
                # Load model weights
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully")
                
                # Load feature columns
                if 'feature_cols' in metadata:
                    self.feature_cols = metadata['feature_cols']
                
                # Load sequence length
                if 'seq_length' in metadata:
                    self.seq_length = metadata['seq_length']
            else:
                logger.warning(f"Model metadata not found at {metadata_path}")
                logger.warning("Using default model parameters")
                
                # Use default parameters
                from model import LSTMModel
                self.model = LSTMModel(10, 64, 2, 1, 0.2)
                
                # Load model weights
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            
            # Load scaler if available
            if self.scaler_path and os.path.exists(self.scaler_path):
                logger.info(f"Loading scaler from {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
            
        Returns:
        --------
        torch.Tensor
            Preprocessed data ready for model input
        """
        try:
            # Select features if feature_cols is provided
            if self.feature_cols:
                # Check if all required features are present
                missing_cols = [col for col in self.feature_cols if col not in data.columns]
                if missing_cols:
                    logger.warning(f"Missing columns: {missing_cols}")
                    # Add missing columns with zeros
                    for col in missing_cols:
                        data[col] = 0
                
                data = data[self.feature_cols]
            
            # Apply scaler if available
            if self.scaler:
                data_scaled = self.scaler.transform(data.values)
                data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
            else:
                data_scaled = data
            
            # Create sequences
            if len(data_scaled) < self.seq_length:
                logger.warning(f"Input data length ({len(data_scaled)}) is less than sequence length ({self.seq_length})")
                # Pad with zeros
                padding = pd.DataFrame(
                    np.zeros((self.seq_length - len(data_scaled), data_scaled.shape[1])),
                    columns=data_scaled.columns
                )
                data_scaled = pd.concat([padding, data_scaled])
            
            # Take the last seq_length rows
            data_scaled = data_scaled.iloc[-self.seq_length:].values
            
            # Convert to tensor
            X = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            return X
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
            
        Returns:
        --------
        dict
            Dictionary containing predictions
        """
        try:
            # Preprocess data
            X = self.preprocess_data(data)
            X = X.to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'attention'):
                    y_pred, attention_weights = self.model(X)
                    attention_weights = attention_weights.cpu().numpy()
                else:
                    y_pred = self.model(X)
                    attention_weights = None
            
            # Convert to numpy
            y_pred = y_pred.cpu().numpy()
            
            # Create result dictionary
            result = {
                'prediction': y_pred[0].tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add attention weights if available
            if attention_weights is not None:
                result['attention_weights'] = attention_weights[0].tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

class ModelRetrainer:
    """
    Class for periodically retraining the model with new data.
    """
    def __init__(self, model_dir, data_dir, retrain_interval_days=7):
        """
        Initialize the model retrainer.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save trained models
        data_dir : str
            Directory containing data
        retrain_interval_days : int, optional
            Interval in days between retraining
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.retrain_interval_days = retrain_interval_days
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize scheduler
        self.scheduler = schedule.Scheduler()
        self.scheduler.every(retrain_interval_days).days.at("02:00").do(self.retrain_model)
        
        # Initialize thread
        self.thread = None
        self.stop_event = threading.Event()
    
    def retrain_model(self):
        """
        Retrain the model with new data.
        """
        try:
            logger.info("Retraining model with new data")
            
            # Import here to avoid circular imports
            from data_acquisition import load_stock_data
            from feature_engineering import prepare_features
            from data_preparation import prepare_data_for_training
            from train import train_with_best_params
            
            # Load data
            raw_dir = os.path.join(self.data_dir, 'raw')
            stock_data_path = os.path.join(raw_dir, 'stock_data.csv')
            
            if not os.path.exists(stock_data_path):
                logger.warning(f"Data file not found at {stock_data_path}")
                return
            
            stock_data = load_stock_data(stock_data_path)
            
            if stock_data is None or stock_data.empty:
                logger.warning("No data loaded")
                return
            
            # Prepare features
            target_col = stock_data.columns[0]  # Assuming first column is the target
            
            processed_data, transformers = prepare_features(
                stock_data,
                target_col=target_col,
                include_technical=False,  # Set to False if we don't have OHLCV data
                include_statistical=True,
                include_lags=True,
                normalize=True,
                reduce_dim=False,
                forecast_horizon=5
            )
            
            # Save scaler
            scaler = transformers.get('scaler')
            if scaler:
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")
            
            # Prepare data for training
            train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
                processed_data,
                target_col=f'Target_5',  # Target column created by prepare_features
                seq_length=20,
                forecast_horizon=1,  # Single-step forecasting
                batch_size=32
            )
            
            # Define model parameters
            model_type = 'lstm_attention'
            hidden_dim = 64
            num_layers = 2
            output_dim = 1
            dropout_prob = 0.2
            
            # Define best parameters (could be loaded from previous optimization)
            best_params = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout_prob': dropout_prob,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            }
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Train model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = os.path.join(self.model_dir, f'{model_type}_model_{timestamp}.pth')
            
            model, history, test_metrics = train_with_best_params(
                model_type, feature_dim, output_dim, train_loader, val_loader, test_loader,
                best_params, device, num_epochs=50, model_save_path=model_save_path
            )
            
            # Save model metadata
            metadata = {
                'model_type': model_type,
                'input_dim': feature_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'output_dim': output_dim,
                'dropout_prob': dropout_prob,
                'feature_cols': processed_data.columns.tolist(),
                'seq_length': 20,
                'training_date': timestamp,
                'test_metrics': test_metrics
            }
            
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model retrained and saved to {model_save_path}")
            logger.info(f"Model metadata saved to {metadata_path}")
            
            # Create a symlink to the latest model
            latest_model_path = os.path.join(self.model_dir, f'{model_type}_model_latest.pth')
            if os.path.exists(latest_model_path):
                os.remove(latest_model_path)
            
            # Copy the model instead of creating a symlink (more compatible)
            import shutil
            shutil.copy2(model_save_path, latest_model_path)
            
            logger.info(f"Latest model symlink updated to {latest_model_path}")
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
    
    def start(self):
        """
        Start the retraining thread.
        """
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Retraining thread is already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Model retrainer started with interval of {self.retrain_interval_days} days")
    
    def stop(self):
        """
        Stop the retraining thread.
        """
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            logger.info("Model retrainer stopped")
    
    def _run_scheduler(self):
        """
        Run the scheduler in a loop.
        """
        while not self.stop_event.is_set():
            self.scheduler.run_pending()
            time.sleep(1)

# Flask API
app = Flask(__name__)

# Global variables
model_server = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions.
    
    Expects a JSON with the following structure:
    {
        "data": {
            "feature1": [value1, value2, ...],
            "feature2": [value1, value2, ...],
            ...
        }
    }
    
    Returns a JSON with predictions.
    """
    try:
        # Get input data
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Make prediction
        result = model_server.predict(df)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    API endpoint for checking the health of the service.
    """
    return jsonify({'status': 'healthy'})

def start_api_server(model_path, scaler_path=None, host='0.0.0.0', port=5000):
    """
    Start the API server.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    scaler_path : str, optional
        Path to the fitted scaler
    host : str, optional
        Host to bind the server to
    port : int, optional
        Port to bind the server to
    """
    global model_server
    
    try:
        # Initialize model server
        model_server = ModelServer(model_path, scaler_path)
        
        # Start Flask app
        app.run(host=host, port=port)
        
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

def create_dashboard():
    """
    Create a Streamlit dashboard for visualizing predictions.
    
    This function should be called from a separate script.
    """
    try:
        # Import Streamlit
        import streamlit as st
        import requests
        import plotly.graph_objects as go
        
        st.title('Stock Price Prediction Dashboard')
        
        # Sidebar
        st.sidebar.header('Settings')
        
        # Select stock
        stock_options = ['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']
        selected_stock = st.sidebar.selectbox('Select Stock', stock_options)
        
        # Select prediction horizon
        horizon = st.sidebar.slider('Prediction Horizon (days)', 1, 30, 5)
        
        # Load historical data
        @st.cache_data
        def load_data():
            # This would typically load data from a database or API
            # For this example, we'll load from a CSV file
            data_path = os.path.join('data', 'raw', 'stock_data.csv')
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                data['Date'] = pd.to_datetime(data.iloc[:, 0])
                data.set_index('Date', inplace=True)
                return data
            return None
        
        data = load_data()
        
        if data is not None:
            # Display historical data
            st.subheader('Historical Data')
            st.line_chart(data[selected_stock])
            
            # Make prediction
            if st.sidebar.button('Predict'):
                st.subheader('Prediction')
                
                # Prepare data for prediction
                recent_data = data.tail(20)  # Last 20 days
                
                # Convert to format expected by API
                api_data = {
                    'data': {col: recent_data[col].tolist() for col in recent_data.columns}
                }
                
                # Call prediction API
                try:
                    response = requests.post('http://localhost:5000/predict', json=api_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        
                        # Create future dates
                        last_date = data.index[-1]
                        future_dates = [last_date + timedelta(days=i+1) for i in range(len(prediction))]
                        
                        # Create DataFrame with predictions
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Prediction': prediction
                        })
                        pred_df.set_index('Date', inplace=True)
                        
                        # Combine historical data and predictions
                        combined_df = pd.concat([data[[selected_stock]].tail(30), pred_df])
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=data.index[-30:],
                            y=data[selected_stock].tail(30),
                            mode='lines',
                            name='Historical'
                        ))
                        
                        # Predictions
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=prediction,
                            mode='lines+markers',
                            name='Prediction',
                            line=dict(dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_stock} Price Prediction',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Display prediction values
                        st.subheader('Prediction Values')
                        st.dataframe(pred_df)
                        
                        # If attention weights are available, display them
                        if 'attention_weights' in result:
                            st.subheader('Attention Weights')
                            
                            attention_df = pd.DataFrame({
                                'Date': data.index[-20:],
                                'Weight': result['attention_weights']
                            })
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=attention_df['Date'],
                                y=attention_df['Weight'],
                                name='Attention Weight'
                            ))
                            
                            fig.update_layout(
                                title='Attention Weights',
                                xaxis_title='Date',
                                yaxis_title='Weight',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig)
                    else:
                        st.error(f"Error making prediction: {response.text}")
                
                except Exception as e:
                    st.error(f"Error connecting to prediction API: {str(e)}")
        else:
            st.error("No data available")
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction Deployment')
    parser.add_argument('--mode', type=str, choices=['api', 'dashboard', 'retrain'], 
                        default='api', help='Deployment mode')
    parser.add_argument('--model_dir', type=str, default='../models', 
                        help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, default='../data', 
                        help='Directory containing data')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host for API server')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port for API server')
    
    args = parser.parse_args()
    
    # Set paths
    model_dir = os.path.abspath(args.model_dir)
    data_dir = os.path.abspath(args.data_dir)
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    if args.mode == 'api':
        # Find the latest model
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if not model_files:
            logger.error(f"No model files found in {model_dir}")
            exit(1)
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        
        model_path = os.path.join(model_dir, model_files[0])
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler not found at {scaler_path}")
            scaler_path = None
        
        logger.info(f"Starting API server with model {model_path}")
        start_api_server(model_path, scaler_path, args.host, args.port)
        
    elif args.mode == 'dashboard':
        logger.info("Starting dashboard")
        create_dashboard()
        
    elif args.mode == 'retrain':
        logger.info("Starting model retrainer")
        retrainer = ModelRetrainer(model_dir, data_dir)
        retrainer.retrain_model()  # Run once immediately
        retrainer.start()  # Start scheduler
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping model retrainer")
            retrainer.stop()
