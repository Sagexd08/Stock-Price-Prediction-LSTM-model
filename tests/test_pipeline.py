"""
Test module for the entire pipeline
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_acquisition import load_stock_data
from src.feature_engineering import prepare_features
from src.data_preparation import prepare_data_for_training
from src.model import create_model
from src.train import train_model, evaluate_model, calculate_metrics
from src.evaluate import plot_predictions

class TestPipeline(unittest.TestCase):
    """
    Test cases for the entire pipeline
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create data directory
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200)
        self.stock_data = pd.DataFrame({
            'Stock_1': np.random.normal(100, 5, 200),
            'Stock_2': np.random.normal(50, 3, 200),
            'Stock_3': np.random.normal(200, 10, 200),
            'Stock_4': np.random.normal(75, 4, 200),
            'Stock_5': np.random.normal(150, 8, 200)
        }, index=dates)
        
        # Save sample data
        self.stock_data_path = os.path.join(self.data_dir, 'raw', 'test_stock_data.csv')
        self.stock_data.to_csv(self.stock_data_path)
    
    def tearDown(self):
        """
        Clean up test environment
        """
        # Remove test files
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.test_dir)
    
    def test_full_pipeline(self):
        """
        Test the entire pipeline
        """
        # Load data
        stock_data = load_stock_data(self.stock_data_path)
        
        # Check if data is loaded
        self.assertIsNotNone(stock_data)
        self.assertFalse(stock_data.empty)
        
        # Prepare features
        target_col = 'Stock_1'
        processed_data, transformers = prepare_features(
            stock_data,
            target_col=target_col,
            include_technical=False,  # Set to False for simplicity
            include_statistical=True,
            include_lags=True,
            normalize=True,
            reduce_dim=False,
            forecast_horizon=5
        )
        
        # Check if features are prepared
        self.assertIsNotNone(processed_data)
        self.assertIsNotNone(transformers)
        self.assertIn('Target_5', processed_data.columns)
        
        # Prepare data for training
        train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
            processed_data,
            target_col='Target_5',
            seq_length=10,
            forecast_horizon=1,
            batch_size=16
        )
        
        # Check if data loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        self.assertGreater(feature_dim, 0)
        
        # Create model
        model_type = 'lstm'
        hidden_dim = 32
        num_layers = 1
        output_dim = 1
        dropout_prob = 0.2
        
        model = create_model(model_type, feature_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        
        # Check if model is created
        self.assertIsNotNone(model)
        
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Train model (just 2 epochs for testing)
        num_epochs = 2
        model_save_path = os.path.join(self.test_dir, 'test_model.pth')
        
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, self.device, early_stopping_patience=5, model_save_path=model_save_path
        )
        
        # Check if training is completed
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertTrue(os.path.exists(model_save_path))
        
        # Evaluate model
        test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, self.device)
        
        # Check if evaluation is completed
        self.assertIsNotNone(test_loss)
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(targets)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        
        # Check if metrics are calculated
        self.assertIsNotNone(metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        
        # Plot predictions
        plot_path = os.path.join(self.test_dir, 'test_predictions.png')
        plot_predictions(targets, predictions, save_path=plot_path)
        
        # Check if plot is created
        self.assertTrue(os.path.exists(plot_path))

if __name__ == '__main__':
    unittest.main()
