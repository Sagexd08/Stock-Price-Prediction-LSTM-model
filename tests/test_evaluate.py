"""
Test module for evaluate.py
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model
from src.evaluate import (
    calculate_metrics,
    plot_predictions,
    plot_residuals,
    plot_error_distribution,
    plot_scatter,
    monte_carlo_dropout_prediction,
    plot_prediction_intervals,
    visualize_attention
)

class TestEvaluate(unittest.TestCase):
    """
    Test cases for evaluate.py
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
        self.y_true = np.random.normal(0, 1, 100)
        self.y_pred = np.random.normal(0, 1, 100)
        
        # Create sample model
        self.input_dim = 5
        self.hidden_dim = 32
        self.num_layers = 1
        self.output_dim = 1
        self.seq_length = 10
        
        # Create models
        self.lstm_model = create_model('lstm', self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        self.attention_model = create_model('lstm_attention', self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Create sample input
        self.X = torch.randn(1, self.seq_length, self.input_dim)
    
    def tearDown(self):
        """
        Clean up test environment
        """
        # Remove test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
    
    def test_calculate_metrics(self):
        """
        Test calculating metrics
        """
        # Calculate metrics
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        # Check if metrics are calculated
        self.assertIsNotNone(metrics)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIn('R2', metrics)
    
    def test_plot_predictions(self):
        """
        Test plotting predictions
        """
        # Plot predictions
        save_path = os.path.join(self.test_dir, 'predictions_plot.png')
        fig = plot_predictions(self.y_true, self.y_pred, title='Test Predictions', save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_residuals(self):
        """
        Test plotting residuals
        """
        # Plot residuals
        save_path = os.path.join(self.test_dir, 'residuals_plot.png')
        fig = plot_residuals(self.y_true, self.y_pred, title='Test Residuals', save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_error_distribution(self):
        """
        Test plotting error distribution
        """
        # Plot error distribution
        save_path = os.path.join(self.test_dir, 'error_distribution_plot.png')
        fig = plot_error_distribution(self.y_true, self.y_pred, title='Test Error Distribution', save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_scatter(self):
        """
        Test plotting scatter plot
        """
        # Plot scatter
        save_path = os.path.join(self.test_dir, 'scatter_plot.png')
        fig = plot_scatter(self.y_true, self.y_pred, title='Test Scatter Plot', save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_monte_carlo_dropout_prediction(self):
        """
        Test Monte Carlo dropout prediction
        """
        # Generate predictions with uncertainty
        mean_pred, std_pred = monte_carlo_dropout_prediction(self.lstm_model, self.X, n_samples=10, device=self.device)
        
        # Check if predictions are generated
        self.assertIsNotNone(mean_pred)
        self.assertIsNotNone(std_pred)
        
        # Check shapes
        self.assertEqual(mean_pred.shape, (1, self.output_dim))
        self.assertEqual(std_pred.shape, (1, self.output_dim))
    
    def test_plot_prediction_intervals(self):
        """
        Test plotting prediction intervals
        """
        # Create sample data
        y_true = np.random.normal(0, 1, 100)
        y_pred = np.random.normal(0, 1, 100)
        y_std = np.abs(np.random.normal(0, 0.2, 100))
        
        # Plot prediction intervals
        save_path = os.path.join(self.test_dir, 'prediction_intervals_plot.png')
        fig = plot_prediction_intervals(y_true, y_pred, y_std, confidence=0.95, title='Test Prediction Intervals', save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_visualize_attention(self):
        """
        Test visualizing attention weights
        """
        # Visualize attention
        save_path = os.path.join(self.test_dir, 'attention_plot.png')
        fig = visualize_attention(self.attention_model, self.X, save_path=save_path)
        
        # Check if plot is created and saved
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))

if __name__ == '__main__':
    unittest.main()
