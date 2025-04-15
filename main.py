"""
Main script for Stock Price Prediction Model

This script provides a command-line interface to run different components
of the stock price prediction system.
"""

import os
import argparse
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory {directory} is ready")

def download_data(args):
    """Download stock data."""
    from src.data_acquisition import download_stock_data
    
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
    
    data = download_stock_data(
        ticker,
        start_date,
        end_date,
        save_path=f'data/raw/{ticker}_data.csv'
    )
    
    if data is not None:
        logger.info(f"Downloaded {len(data)} records for {ticker}")
    else:
        logger.error(f"Failed to download data for {ticker}")

def process_data(args):
    """Process raw data and prepare features."""
    from src.data_acquisition import load_stock_data
    from src.feature_engineering import prepare_features
    
    input_file = args.input_file
    output_file = args.output_file
    target_col = args.target_col
    
    logger.info(f"Processing data from {input_file}")
    
    # Load data
    data = load_stock_data(input_file)
    
    if data is None:
        logger.error(f"Failed to load data from {input_file}")
        return
    
    # Prepare features
    processed_data, transformers = prepare_features(
        data,
        target_col=target_col,
        include_technical=args.technical,
        include_statistical=args.statistical,
        include_lags=args.lags,
        normalize=args.normalize,
        reduce_dim=args.reduce_dim,
        forecast_horizon=args.horizon
    )
    
    # Save processed data
    processed_data.to_csv(output_file)
    logger.info(f"Processed data saved to {output_file}")
    
    # Save scaler if available
    if 'scaler' in transformers and args.normalize:
        import joblib
        scaler_path = os.path.join(os.path.dirname(output_file), 'scaler.pkl')
        joblib.dump(transformers['scaler'], scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

def train_model(args):
    """Train a model."""
    from src.data_acquisition import load_stock_data
    from src.feature_engineering import prepare_features
    from src.data_preparation import prepare_data_for_training
    from src.model import create_model
    from src.train import train_with_best_params, optimize_hyperparameters
    
    input_file = args.input_file
    model_type = args.model_type
    target_col = args.target_col
    
    logger.info(f"Training {model_type} model using data from {input_file}")
    
    # Load data
    if input_file.endswith('.csv'):
        data = load_stock_data(input_file)
    else:
        # Assume it's already processed
        data = pd.read_csv(input_file, index_col=0)
    
    if data is None:
        logger.error(f"Failed to load data from {input_file}")
        return
    
    # Prepare features if needed
    if args.prepare_features:
        processed_data, _ = prepare_features(
            data,
            target_col=target_col,
            include_technical=args.technical,
            include_statistical=args.statistical,
            include_lags=args.lags,
            normalize=args.normalize,
            reduce_dim=args.reduce_dim,
            forecast_horizon=args.horizon
        )
    else:
        processed_data = data
    
    # Prepare data for training
    train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
        processed_data,
        target_col=f'Target_{args.horizon}' if args.prepare_features else target_col,
        seq_length=args.seq_length,
        forecast_horizon=args.output_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Optimize hyperparameters if requested
    if args.optimize:
        logger.info("Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(
            model_type, feature_dim, args.output_size, train_loader, val_loader, device,
            n_trials=args.n_trials, timeout=args.timeout
        )
    else:
        # Use default hyperparameters
        best_params = {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout_prob': args.dropout_prob,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay
        }
    
    # Train model with best parameters
    model_save_path = os.path.join('models', f'{model_type}_model.pth')
    model, history, test_metrics = train_with_best_params(
        model_type, feature_dim, args.output_size, train_loader, val_loader, test_loader,
        best_params, device, num_epochs=args.epochs, model_save_path=model_save_path
    )
    
    logger.info(f"Model trained and saved to {model_save_path}")
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save model metadata
    import json
    metadata = {
        'model_type': model_type,
        'input_dim': feature_dim,
        'hidden_dim': best_params['hidden_dim'],
        'num_layers': best_params['num_layers'],
        'output_dim': args.output_size,
        'dropout_prob': best_params['dropout_prob'],
        'feature_cols': processed_data.drop(columns=[f'Target_{args.horizon}' if args.prepare_features else target_col]).columns.tolist(),
        'seq_length': args.seq_length,
        'training_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'test_metrics': test_metrics
    }
    
    metadata_path = os.path.join('models', 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model metadata saved to {metadata_path}")

def evaluate_model(args):
    """Evaluate a trained model."""
    from src.data_acquisition import load_stock_data
    from src.feature_engineering import prepare_features
    from src.data_preparation import prepare_data_for_training
    from src.model import create_model
    from src.evaluate import evaluate_model_comprehensive
    
    model_path = args.model_path
    input_file = args.input_file
    output_dir = args.output_dir
    
    logger.info(f"Evaluating model from {model_path} using data from {input_file}")
    
    # Load model metadata
    import json
    metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata.get('model_type', 'lstm')
        input_dim = metadata.get('input_dim', 10)
        hidden_dim = metadata.get('hidden_dim', 64)
        num_layers = metadata.get('num_layers', 2)
        output_dim = metadata.get('output_dim', 1)
        dropout_prob = metadata.get('dropout_prob', 0.2)
        seq_length = metadata.get('seq_length', 20)
        feature_cols = metadata.get('feature_cols', None)
    else:
        logger.warning(f"Model metadata not found at {metadata_path}")
        logger.warning("Using command line arguments for model parameters")
        
        model_type = args.model_type
        input_dim = None  # Will be determined from data
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        output_dim = args.output_size
        dropout_prob = args.dropout_prob
        seq_length = args.seq_length
        feature_cols = None
    
    # Load data
    if input_file.endswith('.csv'):
        data = load_stock_data(input_file)
    else:
        # Assume it's already processed
        data = pd.read_csv(input_file, index_col=0)
    
    if data is None:
        logger.error(f"Failed to load data from {input_file}")
        return
    
    # Prepare features if needed
    if args.prepare_features:
        processed_data, _ = prepare_features(
            data,
            target_col=args.target_col,
            include_technical=args.technical,
            include_statistical=args.statistical,
            include_lags=args.lags,
            normalize=args.normalize,
            reduce_dim=args.reduce_dim,
            forecast_horizon=args.horizon
        )
    else:
        processed_data = data
    
    # Prepare data for training
    _, _, test_loader, feature_dim = prepare_data_for_training(
        processed_data,
        target_col=f'Target_{args.horizon}' if args.prepare_features else args.target_col,
        seq_length=seq_length,
        forecast_horizon=output_dim,
        val_size=0.15,
        test_size=0.15,
        batch_size=args.batch_size
    )
    
    # Update input_dim if not available from metadata
    if input_dim is None:
        input_dim = feature_dim
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Evaluate model
    metrics = evaluate_model_comprehensive(model, test_loader, device, output_dir=output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"Metrics: {metrics}")

def predict(args):
    """Make predictions using a trained model."""
    from src.data_acquisition import load_stock_data, download_stock_data
    from src.feature_engineering import prepare_features
    from src.model import create_model
    from src.deploy import ModelServer
    from src.visualization import plot_predictions_interactive
    
    model_path = args.model_path
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    output_file = args.output_file
    
    logger.info(f"Making predictions for {ticker} from {start_date} to {end_date}")
    
    # Download data if needed
    if args.download:
        data = download_stock_data(
            ticker,
            start_date,
            end_date,
            save_path=f'data/raw/{ticker}_prediction_data.csv'
        )
    else:
        # Load data from file
        data = load_stock_data(args.input_file)
    
    if data is None:
        logger.error("Failed to get data for prediction")
        return
    
    # Initialize model server
    model_server = ModelServer(model_path)
    
    # Make prediction
    result = model_server.predict(data)
    
    # Save prediction
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Prediction saved to {output_file}")
    
    # Plot prediction
    if args.plot:
        plot_predictions_interactive(data, result['prediction'], ticker)

def deploy_api(args):
    """Deploy the model as an API."""
    from src.deploy import start_api_server
    
    model_path = args.model_path
    host = args.host
    port = args.port
    
    logger.info(f"Deploying model from {model_path} as API on {host}:{port}")
    
    # Find scaler if available
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = None
    
    # Start API server
    start_api_server(model_path, scaler_path, host, port)

def deploy_dashboard(args):
    """Deploy the model as a dashboard."""
    from src.deploy import create_dashboard
    
    logger.info("Starting dashboard")
    
    # Start dashboard
    create_dashboard()

def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description='Stock Price Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup directories')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download stock data')
    download_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    download_parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw data')
    process_parser.add_argument('--input-file', type=str, required=True, help='Input file path')
    process_parser.add_argument('--output-file', type=str, required=True, help='Output file path')
    process_parser.add_argument('--target-col', type=str, required=True, help='Target column name')
    process_parser.add_argument('--technical', action='store_true', help='Include technical indicators')
    process_parser.add_argument('--statistical', action='store_true', help='Include statistical features')
    process_parser.add_argument('--lags', action='store_true', help='Include lag features')
    process_parser.add_argument('--normalize', action='store_true', help='Normalize features')
    process_parser.add_argument('--reduce-dim', action='store_true', help='Reduce dimensionality')
    process_parser.add_argument('--horizon', type=int, default=5, help='Forecast horizon')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--input-file', type=str, required=True, help='Input file path')
    train_parser.add_argument('--model-type', type=str, default='lstm_attention', 
                             choices=['lstm', 'lstm_attention', 'stacked_lstm_attention', 'multi_step_lstm'],
                             help='Model type')
    train_parser.add_argument('--target-col', type=str, required=True, help='Target column name')
    train_parser.add_argument('--prepare-features', action='store_true', help='Prepare features')
    train_parser.add_argument('--technical', action='store_true', help='Include technical indicators')
    train_parser.add_argument('--statistical', action='store_true', help='Include statistical features')
    train_parser.add_argument('--lags', action='store_true', help='Include lag features')
    train_parser.add_argument('--normalize', action='store_true', help='Normalize features')
    train_parser.add_argument('--reduce-dim', action='store_true', help='Reduce dimensionality')
    train_parser.add_argument('--horizon', type=int, default=5, help='Forecast horizon')
    train_parser.add_argument('--seq-length', type=int, default=20, help='Sequence length')
    train_parser.add_argument('--output-size', type=int, default=1, help='Output size')
    train_parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size')
    train_parser.add_argument('--test-size', type=float, default=0.15, help='Test set size')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    train_parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    train_parser.add_argument('--dropout-prob', type=float, default=0.2, help='Dropout probability')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    train_parser.add_argument('--n-trials', type=int, default=20, help='Number of optimization trials')
    train_parser.add_argument('--timeout', type=int, default=3600, help='Optimization timeout in seconds')
    train_parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    evaluate_parser.add_argument('--model-path', type=str, required=True, help='Model file path')
    evaluate_parser.add_argument('--input-file', type=str, required=True, help='Input file path')
    evaluate_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    evaluate_parser.add_argument('--target-col', type=str, required=True, help='Target column name')
    evaluate_parser.add_argument('--prepare-features', action='store_true', help='Prepare features')
    evaluate_parser.add_argument('--technical', action='store_true', help='Include technical indicators')
    evaluate_parser.add_argument('--statistical', action='store_true', help='Include statistical features')
    evaluate_parser.add_argument('--lags', action='store_true', help='Include lag features')
    evaluate_parser.add_argument('--normalize', action='store_true', help='Normalize features')
    evaluate_parser.add_argument('--reduce-dim', action='store_true', help='Reduce dimensionality')
    evaluate_parser.add_argument('--horizon', type=int, default=5, help='Forecast horizon')
    evaluate_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    evaluate_parser.add_argument('--model-type', type=str, default='lstm_attention', 
                               choices=['lstm', 'lstm_attention', 'stacked_lstm_attention', 'multi_step_lstm'],
                               help='Model type (used if metadata not available)')
    evaluate_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension (used if metadata not available)')
    evaluate_parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers (used if metadata not available)')
    evaluate_parser.add_argument('--output-size', type=int, default=1, help='Output size (used if metadata not available)')
    evaluate_parser.add_argument('--dropout-prob', type=float, default=0.2, help='Dropout probability (used if metadata not available)')
    evaluate_parser.add_argument('--seq-length', type=int, default=20, help='Sequence length (used if metadata not available)')
    evaluate_parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', type=str, required=True, help='Model file path')
    predict_parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    predict_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    predict_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    predict_parser.add_argument('--input-file', type=str, help='Input file path')
    predict_parser.add_argument('--output-file', type=str, help='Output file path')
    predict_parser.add_argument('--download', action='store_true', help='Download data')
    predict_parser.add_argument('--plot', action='store_true', help='Plot prediction')
    
    # Deploy API command
    deploy_api_parser = subparsers.add_parser('deploy-api', help='Deploy model as API')
    deploy_api_parser.add_argument('--model-path', type=str, required=True, help='Model file path')
    deploy_api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    deploy_api_parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    
    # Deploy dashboard command
    deploy_dashboard_parser = subparsers.add_parser('deploy-dashboard', help='Deploy model as dashboard')
    
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == 'setup':
        setup_directories()
    elif args.command == 'download':
        download_data(args)
    elif args.command == 'process':
        process_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'deploy-api':
        deploy_api(args)
    elif args.command == 'deploy-dashboard':
        deploy_dashboard(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
