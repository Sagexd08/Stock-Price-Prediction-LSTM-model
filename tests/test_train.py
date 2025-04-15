"""
Test module for train.py
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model
from src.train import (
    train_model,
    evaluate_model,
    calculate_metrics,
    plot_training_history,
    plot_predictions
)

class TestTrain(unittest.TestCase):
    """
    Test cases for train.py
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
        
        # Create sample data
        self.batch_size = 16
        self.seq_length = 10
        self.input_dim = 5
        self.hidden_dim = 32
        self.num_layers = 1
        self.output_dim = 1
        
        # Create model
        self.model = create_model('lstm', self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Create sample data loaders
        self.train_loader = self._create_data_loader()
        self.val_loader = self._create_data_loader()
        self.test_loader = self._create_data_loader()
        
        # Create loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    def tearDown(self):
        """
        Clean up test environment
        """
        # Remove test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
    
    def _create_data_loader(self):
        """
        Create a sample data loader
        """
        # Create random data
        X = torch.randn(self.batch_size * 3, self.seq_length, self.input_dim)
        y = torch.randn(self.batch_size * 3, self.output_dim)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return data_loader
    
    def test_train_model(self):
        """
        Test training a model
        """
        # Train model
        num_epochs = 2
        model_save_path = os.path.join(self.test_dir, 'test_model.pth')
        
        history = train_model(
            self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer, self.scheduler,
            num_epochs, self.device, early_stopping_patience=5, model_save_path=model_save_path
        )
        
        # Check if history is created
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('learning_rate', history)
        
        # Check if model is saved
        self.assertTrue(os.path.exists(model_save_path))
    
    def test_evaluate_model(self):
        """
        Test evaluating a model
        """
        # Evaluate model
        test_loss, predictions, targets = evaluate_model(self.model, self.test_loader, self.criterion, self.device)
        
        # Check if evaluation results are created
        self.assertIsNotNone(test_loss)
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(targets)
        
        # Check shapes
        self.assertEqual(predictions.shape[0], len(self.test_loader.dataset))
        self.assertEqual(targets.shape[0], len(self.test_loader.dataset))
    
    def test_calculate_metrics(self):
        """
        Test calculating metrics
        """
        # Create sample predictions and targets
        predictions = np.random.normal(0, 1, 100)
        targets = np.random.normal(0, 1, 100)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        
        # Check if metrics are calculated
        self.assertIsNotNone(metrics)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIn('R2', metrics)
    
    def test_plot_training_history(self):
        """
        Test plotting training history
        """
        # Create sample history
        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
            'learning_rate': [0.001, 0.001, 0.0005, 0.0005, 0.00025]
        }
        
        # Plot history
        save_path = os.path.join(self.test_dir, 'history_plot.png')
        plot_training_history(history, save_path)
        
        # Check if plot is saved
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_predictions(self):
        """
        Test plotting predictions
        """
        # Create sample predictions and targets
        predictions = np.random.normal(0, 1, 100)
        targets = np.random.normal(0, 1, 100)
        
        # Plot predictions
        save_path = os.path.join(self.test_dir, 'predictions_plot.png')
        plot_predictions(predictions, targets, save_path=save_path)
        
        # Check if plot is saved
        self.assertTrue(os.path.exists(save_path))

if __name__ == '__main__':
    unittest.main()
