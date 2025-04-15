"""
Training Module

This module provides functions for training LSTM models for stock price prediction,
including hyperparameter optimization and model evaluation.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               num_epochs, device, early_stopping_patience=10, model_save_path=None):
    """
    Train the model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    num_epochs : int
        Number of epochs to train for
    device : torch.device
        Device to train on (CPU or GPU)
    early_stopping_patience : int, optional
        Number of epochs to wait for improvement before stopping
    model_save_path : str, optional
        Path to save the best model
        
    Returns:
    --------
    dict
        Dictionary containing training history
    """
    try:
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Move model to device
        model.to(device)
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                        outputs, _ = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item() * inputs.size(0)
            
            # Calculate average validation loss
            val_loss = val_loss / len(val_loader.dataset)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print statistics
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                       f"Time: {epoch_time:.2f}s")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                
                # Save the best model
                if model_save_path is not None:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"Model saved to {model_save_path}")
            else:
                early_stopping_counter += 1
                logger.info(f"EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training complete")
        return history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to evaluate on (CPU or GPU)
        
    Returns:
    --------
    tuple
        (test_loss, predictions, targets)
    """
    try:
        logger.info("Evaluating model on test data")
        
        # Move model to device
        model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        test_loss = 0.0
        predictions = []
        targets_list = []
        
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
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update statistics
                test_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # Calculate average test loss
        test_loss = test_loss / len(test_loader.dataset)
        
        # Concatenate predictions and targets
        predictions = np.concatenate(predictions)
        targets_list = np.concatenate(targets_list)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        return test_loss, predictions, targets_list
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions
    targets : numpy.ndarray
        Ground truth values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    try:
        logger.info("Calculating evaluation metrics")
        
        # Mean Squared Error (MSE)
        mse = np.mean((predictions - targets) ** 2)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - targets))
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = targets != 0
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        
        # R-squared (coefficient of determination)
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
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

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    save_path : str, optional
        Path to save the plot
    """
    try:
        logger.info("Plotting training history")
        
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_predictions(predictions, targets, dates=None, save_path=None):
    """
    Plot model predictions against ground truth.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions
    targets : numpy.ndarray
        Ground truth values
    dates : numpy.ndarray, optional
        Dates corresponding to predictions and targets
    save_path : str, optional
        Path to save the plot
    """
    try:
        logger.info("Plotting predictions")
        
        plt.figure(figsize=(12, 6))
        
        if dates is not None:
            plt.plot(dates, targets, label='Actual')
            plt.plot(dates, predictions, label='Predicted')
            plt.xlabel('Date')
        else:
            plt.plot(targets, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.xlabel('Time Step')
        
        plt.ylabel('Value')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True)
        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        raise

def objective(trial, model_type, input_dim, output_dim, train_loader, val_loader, device, num_epochs=50):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Optuna trial object
    model_type : str
        Type of model to create
    input_dim : int
        Number of input features
    output_dim : int
        Number of output dimensions
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    device : torch.device
        Device to train on (CPU or GPU)
    num_epochs : int, optional
        Number of epochs to train for
        
    Returns:
    --------
    float
        Validation loss
    """
    try:
        # Import here to avoid circular imports
        from src.model import create_model
        
        # Define hyperparameters to optimize
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create model
        model = create_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Train the model
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    if isinstance(model, nn.Module) and hasattr(model, 'attention'):
                        outputs, _ = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item() * inputs.size(0)
            
            # Calculate average validation loss
            val_loss = val_loss / len(val_loader.dataset)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Report intermediate value
            trial.report(val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Error in objective function: {str(e)}")
        raise

def optimize_hyperparameters(model_type, input_dim, output_dim, train_loader, val_loader, 
                            device, n_trials=100, timeout=None, study_name=None):
    """
    Optimize hyperparameters using Optuna.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    input_dim : int
        Number of input features
    output_dim : int
        Number of output dimensions
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    device : torch.device
        Device to train on (CPU or GPU)
    n_trials : int, optional
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds
    study_name : str, optional
        Name of the study
        
    Returns:
    --------
    dict
        Dictionary containing best hyperparameters
    """
    try:
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create a study name if not provided
        if study_name is None:
            study_name = f"stock_prediction_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a study
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, model_type, input_dim, output_dim, train_loader, val_loader, device),
            n_trials=n_trials,
            timeout=timeout
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best validation loss: {best_value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        raise

def train_with_best_params(model_type, input_dim, output_dim, train_loader, val_loader, test_loader,
                          best_params, device, num_epochs=100, model_save_path=None):
    """
    Train a model with the best hyperparameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    input_dim : int
        Number of input features
    output_dim : int
        Number of output dimensions
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    best_params : dict
        Dictionary containing best hyperparameters
    device : torch.device
        Device to train on (CPU or GPU)
    num_epochs : int, optional
        Number of epochs to train for
    model_save_path : str, optional
        Path to save the best model
        
    Returns:
    --------
    tuple
        (model, history, test_metrics)
    """
    try:
        logger.info(f"Training model with best hyperparameters for {num_epochs} epochs")
        
        # Import here to avoid circular imports
        from model import create_model
        
        # Extract hyperparameters
        hidden_dim = best_params['hidden_dim']
        num_layers = best_params['num_layers']
        dropout_prob = best_params['dropout_prob']
        learning_rate = best_params['learning_rate']
        weight_decay = best_params['weight_decay']
        
        # Create model
        model = create_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Train the model
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, device, early_stopping_patience=15, model_save_path=model_save_path
        )
        
        # Load the best model
        if model_save_path is not None and os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
        
        # Evaluate the model on test data
        test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, device)
        
        # Calculate metrics
        test_metrics = calculate_metrics(predictions, targets)
        
        # Plot training history
        if model_save_path is not None:
            plot_dir = os.path.dirname(model_save_path)
            plot_training_history(history, save_path=os.path.join(plot_dir, 'training_history.png'))
            plot_predictions(predictions, targets, save_path=os.path.join(plot_dir, 'predictions.png'))
        
        return model, history, test_metrics
        
    except Exception as e:
        logger.error(f"Error training with best hyperparameters: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import os
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
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
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
                output_dim = 1  # Single-step forecasting
                
                # Optimize hyperparameters
                best_params = optimize_hyperparameters(
                    model_type, feature_dim, output_dim, train_loader, val_loader, device,
                    n_trials=20, timeout=3600  # Limit to 20 trials or 1 hour
                )
                
                # Train with best hyperparameters
                model_save_path = os.path.join(models_dir, f'{model_type}_model.pth')
                model, history, test_metrics = train_with_best_params(
                    model_type, feature_dim, output_dim, train_loader, val_loader, test_loader,
                    best_params, device, num_epochs=100, model_save_path=model_save_path
                )
                
                # Print test metrics
                print(f"Test metrics: {test_metrics}")
