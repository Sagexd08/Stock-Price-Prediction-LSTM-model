"""
Evaluation Module

This module provides functions for evaluating trained models,
including metrics calculation, visualization, and uncertainty quantification.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    try:
        logger.info("Calculating evaluation metrics")
        
        # Mean Squared Error (MSE)
        mse = mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # R-squared (coefficient of determination)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        
        logger.info(f"Metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%, R2={r2:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def plot_predictions(y_true, y_pred, dates=None, title='Actual vs Predicted Values', save_path=None):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values
    dates : numpy.ndarray, optional
        Dates corresponding to the values
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting predictions")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if dates is not None:
            ax.plot(dates, y_true, label='Actual', marker='o', markersize=3, linestyle='-', alpha=0.7)
            ax.plot(dates, y_pred, label='Predicted', marker='x', markersize=3, linestyle='-', alpha=0.7)
            ax.set_xlabel('Date')
            
            # Set x-axis ticks
            if len(dates) > 20:
                # Show fewer ticks for readability
                tick_indices = np.linspace(0, len(dates) - 1, 10, dtype=int)
                ax.set_xticks([dates[i] for i in tick_indices])
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.plot(y_true, label='Actual', marker='o', markersize=3, linestyle='-', alpha=0.7)
            ax.plot(y_pred, label='Predicted', marker='x', markersize=3, linestyle='-', alpha=0.7)
            ax.set_xlabel('Time Step')
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        raise

def plot_residuals(y_true, y_pred, title='Residuals Plot', save_path=None):
    """
    Plot residuals (errors) between actual and predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting residuals")
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals over time
        axes[0].plot(residuals, marker='o', markersize=3, linestyle='', alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='-')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].axvline(x=0, color='r', linestyle='-')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting residuals: {str(e)}")
        raise

def plot_error_distribution(y_true, y_pred, title='Error Distribution', save_path=None):
    """
    Plot the distribution of prediction errors.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting error distribution")
        
        # Calculate errors
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors / y_true) * 100  # Percentage error
        
        # Remove infinite values (from division by zero)
        rel_errors = rel_errors[np.isfinite(rel_errors)]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute errors
        sns.histplot(abs_errors, kde=True, ax=axes[0])
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Absolute Errors')
        axes[0].grid(True, alpha=0.3)
        
        # Relative errors (percentage)
        sns.histplot(rel_errors, kde=True, ax=axes[1])
        axes[1].set_xlabel('Relative Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Relative Errors')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting error distribution: {str(e)}")
        raise

def plot_scatter(y_true, y_pred, title='Actual vs Predicted Scatter Plot', save_path=None):
    """
    Create a scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Creating scatter plot")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add R-squared value
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating scatter plot: {str(e)}")
        raise

def monte_carlo_dropout_prediction(model, X, n_samples=100, device='cpu'):
    """
    Generate predictions with uncertainty using Monte Carlo dropout.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model with dropout layers
    X : torch.Tensor
        Input data
    n_samples : int, optional
        Number of Monte Carlo samples
    device : str or torch.device, optional
        Device to run the model on
        
    Returns:
    --------
    tuple
        (mean_prediction, std_prediction)
    """
    try:
        logger.info(f"Generating Monte Carlo dropout predictions with {n_samples} samples")
        
        # Move model to device
        model.to(device)
        
        # Set model to evaluation mode but enable dropout
        model.eval()
        
        # Enable dropout during inference
        def enable_dropout(model):
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        
        enable_dropout(model)
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                    output, _ = model(X)
                else:
                    output = model(X)
                predictions.append(output.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions)
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
        
    except Exception as e:
        logger.error(f"Error generating Monte Carlo dropout predictions: {str(e)}")
        raise

def plot_prediction_intervals(y_true, y_pred, y_std, dates=None, confidence=0.95, title='Prediction Intervals', save_path=None):
    """
    Plot predictions with confidence intervals.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth values
    y_pred : numpy.ndarray
        Predicted values (mean)
    y_std : numpy.ndarray
        Standard deviation of predictions
    dates : numpy.ndarray, optional
        Dates corresponding to the values
    confidence : float, optional
        Confidence level (e.g., 0.95 for 95% confidence)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info(f"Plotting prediction intervals with {confidence*100}% confidence")
        
        # Calculate z-score for the given confidence level
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        # Plot actual values
        ax.plot(x, y_true, label='Actual', marker='o', markersize=3, linestyle='-', alpha=0.7)
        
        # Plot predicted values
        ax.plot(x, y_pred, label='Predicted', marker='x', markersize=3, linestyle='-', alpha=0.7)
        
        # Plot confidence intervals
        lower_bound = y_pred - z * y_std
        upper_bound = y_pred + z * y_std
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, label=f'{confidence*100}% Confidence Interval')
        
        if dates is not None:
            ax.set_xlabel('Date')
            
            # Set x-axis ticks
            if len(dates) > 20:
                # Show fewer ticks for readability
                tick_indices = np.linspace(0, len(dates) - 1, 10, dtype=int)
                ax.set_xticks([dates[i] for i in tick_indices])
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_xlabel('Time Step')
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting prediction intervals: {str(e)}")
        raise

def visualize_attention(model, X, dates=None, save_path=None):
    """
    Visualize attention weights from a model with attention mechanism.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model with attention mechanism
    X : torch.Tensor
        Input data
    dates : numpy.ndarray, optional
        Dates corresponding to the input sequence
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Visualizing attention weights")
        
        # Check if model has attention
        if not hasattr(model, 'attention'):
            logger.warning("Model does not have an attention mechanism")
            return None
        
        # Set model to evaluation mode
        model.eval()
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = model(X)
        
        # Convert to numpy
        attention_weights = attention_weights.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot attention weights
        if attention_weights.ndim == 3:
            # If we have a batch of sequences, take the first one
            attention_weights = attention_weights[0]
        
        seq_length = attention_weights.shape[0]
        
        if dates is not None and len(dates) == seq_length:
            x = dates
            ax.set_xlabel('Date')
        else:
            x = np.arange(seq_length)
            ax.set_xlabel('Time Step')
        
        ax.plot(x, attention_weights, marker='o', markersize=5, linestyle='-')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Attention Weights Visualization')
        ax.grid(True, alpha=0.3)
        
        # Highlight the time steps with highest attention
        top_indices = np.argsort(attention_weights.flatten())[-3:]  # Top 3 attention weights
        for idx in top_indices:
            ax.annotate(f'Time step {idx}', 
                       xy=(x[idx], attention_weights.flatten()[idx]),
                       xytext=(10, 10),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing attention weights: {str(e)}")
        raise

def evaluate_model_comprehensive(model, test_loader, device, output_dir=None):
    """
    Perform comprehensive evaluation of a trained model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : torch.device
        Device to evaluate on (CPU or GPU)
    output_dir : str, optional
        Directory to save evaluation results and plots
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    try:
        logger.info("Performing comprehensive model evaluation")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Collect predictions and ground truth
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Store predictions, targets, and inputs
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_inputs.append(inputs.cpu().numpy())
        
        # Concatenate results
        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_targets)
        X = np.concatenate(all_inputs)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Generate plots
        if output_dir is not None:
            # Predictions plot
            plot_predictions(y_true, y_pred, title='Actual vs Predicted Values',
                            save_path=os.path.join(output_dir, 'predictions.png'))
            
            # Residuals plot
            plot_residuals(y_true, y_pred, title='Residuals Analysis',
                          save_path=os.path.join(output_dir, 'residuals.png'))
            
            # Error distribution plot
            plot_error_distribution(y_true, y_pred, title='Error Distribution',
                                   save_path=os.path.join(output_dir, 'error_distribution.png'))
            
            # Scatter plot
            plot_scatter(y_true, y_pred, title='Actual vs Predicted Scatter Plot',
                        save_path=os.path.join(output_dir, 'scatter.png'))
            
            # Generate uncertainty estimates using Monte Carlo dropout
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            mean_pred, std_pred = monte_carlo_dropout_prediction(model, X_tensor, n_samples=100, device=device)
            
            # Plot prediction intervals
            plot_prediction_intervals(y_true, mean_pred, std_pred, confidence=0.95,
                                     title='Predictions with 95% Confidence Intervals',
                                     save_path=os.path.join(output_dir, 'prediction_intervals.png'))
            
            # Visualize attention weights if model has attention
            if hasattr(model, 'attention'):
                # Use a single batch for visualization
                batch_inputs = X_tensor[:1]  # Take the first sample
                visualize_attention(model, batch_inputs,
                                   save_path=os.path.join(output_dir, 'attention_weights.png'))
            
            # Save metrics to file
            with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
        
        logger.info("Comprehensive evaluation complete")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during comprehensive evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import os
    import torch
    from data_acquisition import load_stock_data
    from feature_engineering import prepare_features
    from data_preparation import prepare_data_for_training
    from model import create_model
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
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
                
                # Define model parameters
                model_type = 'lstm_attention'
                hidden_dim = 64
                num_layers = 2
                output_dim = 1  # Single-step forecasting
                dropout_prob = 0.2
                
                # Create model
                model = create_model(model_type, feature_dim, hidden_dim, num_layers, output_dim, dropout_prob)
                
                # Load trained model if exists
                model_path = os.path.join(models_dir, f'{model_type}_model.pth')
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"Loaded model from {model_path}")
                    
                    # Evaluate model
                    output_dir = os.path.join(results_dir, model_type)
                    metrics = evaluate_model_comprehensive(model, test_loader, device, output_dir)
                    
                    print(f"Evaluation metrics: {metrics}")
                else:
                    logger.warning(f"No trained model found at {model_path}")
