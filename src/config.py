"""
Configuration Module

This module provides functions for loading and managing configuration settings.
"""

import os
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        logger.info(f"Loading configuration from {config_path}")
        
        # Check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found at {config_path}")
            logger.warning("Using default configuration")
            return get_default_config()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.warning("Using default configuration")
        return get_default_config()

def get_default_config():
    """
    Get default configuration.
    
    Returns:
    --------
    dict
        Default configuration dictionary
    """
    return {
        'data': {
            'ticker': 'AAPL',
            'start_date': '2018-01-01',
            'end_date': None,
            'interval': '1d',
            'use_kaggle_data': True,
            'kaggle_dataset': 'mrsimple07/stock-price-prediction'
        },
        'features': {
            'include_technical': True,
            'include_statistical': True,
            'include_lags': True,
            'normalize': True,
            'reduce_dim': False,
            'forecast_horizon': 5
        },
        'data_prep': {
            'seq_length': 20,
            'val_size': 0.15,
            'test_size': 0.15,
            'batch_size': 32
        },
        'model': {
            'model_type': 'lstm_attention',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout_prob': 0.2,
            'output_size': 1,
            'conv_lstm': {
                'kernel_size': 3
            },
            'transformer': {
                'nhead': 8
            }
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'epochs': 100,
            'early_stopping_patience': 10,
            'use_gpu': True,
            'optimize': False,
            'n_trials': 20,
            'timeout': 3600
        },
        'evaluation': {
            'metrics': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2'],
            'plot_predictions': True,
            'plot_residuals': True,
            'plot_attention': True,
            'monte_carlo_samples': 100
        },
        'deployment': {
            'api_host': '0.0.0.0',
            'api_port': 5000,
            'dashboard_port': 8501,
            'retrain_interval_days': 7
        }
    }

def save_config(config, config_path='config.yaml'):
    """
    Save configuration to a YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str, optional
        Path to the configuration file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        logger.info(f"Saving configuration to {config_path}")
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Configuration saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

def update_config(config, updates):
    """
    Update configuration with new values.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    updates : dict
        Dictionary of updates
        
    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    try:
        logger.info("Updating configuration")
        
        # Update configuration recursively
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        # Update configuration
        updated_config = update_dict(config.copy(), updates)
        
        logger.info("Configuration updated successfully")
        return updated_config
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return config

def get_model_params(config):
    """
    Get model parameters from configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict
        Model parameters dictionary
    """
    try:
        logger.info("Getting model parameters from configuration")
        
        # Get model parameters
        model_params = {
            'model_type': config['model']['model_type'],
            'hidden_dim': config['model']['hidden_dim'],
            'num_layers': config['model']['num_layers'],
            'output_dim': config['model']['output_size'],
            'dropout_prob': config['model']['dropout_prob']
        }
        
        # Add model-specific parameters
        if model_params['model_type'] == 'conv_lstm':
            model_params['kernel_size'] = config['model']['conv_lstm']['kernel_size']
        elif model_params['model_type'] == 'transformer':
            model_params['nhead'] = config['model']['transformer']['nhead']
        
        logger.info(f"Model parameters: {model_params}")
        return model_params
        
    except Exception as e:
        logger.error(f"Error getting model parameters: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    config = load_config()
    print(config)
    
    # Update configuration
    updates = {
        'model': {
            'hidden_dim': 128,
            'num_layers': 3
        }
    }
    updated_config = update_config(config, updates)
    print(updated_config['model'])
    
    # Save configuration
    save_config(updated_config, 'config_updated.yaml')
    
    # Get model parameters
    model_params = get_model_params(updated_config)
    print(model_params)
