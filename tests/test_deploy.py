"""
Test module for deploy.py
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
import json
import threading
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model
from src.deploy import ModelServer, ModelRetrainer

class TestDeploy(unittest.TestCase):
    """
    Test cases for deploy.py
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create model directory
        self.model_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create data directory
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        
        # Create sample model
        self.input_dim = 5
        self.hidden_dim = 32
        self.num_layers = 1
        self.output_dim = 1
        
        # Create model
        self.model = create_model('lstm', self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Save model
        self.model_path = os.path.join(self.model_dir, 'test_model.pth')
        torch.save(self.model.state_dict(), self.model_path)
        
        # Create model metadata
        self.metadata = {
            'model_type': 'lstm',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout_prob': 0.2,
            'feature_cols': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
            'seq_length': 10,
            'training_date': '20230101_000000',
            'test_metrics': {
                'MSE': 0.1,
                'RMSE': 0.316,
                'MAE': 0.25,
                'MAPE': 5.0,
                'R2': 0.8
            }
        }
        
        # Save metadata
        self.metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Feature1': np.random.normal(0, 1, 20),
            'Feature2': np.random.normal(0, 1, 20),
            'Feature3': np.random.normal(0, 1, 20),
            'Feature4': np.random.normal(0, 1, 20),
            'Feature5': np.random.normal(0, 1, 20)
        })
    
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
    
    def test_model_server_init(self):
        """
        Test initializing ModelServer
        """
        # Initialize model server
        model_server = ModelServer(self.model_path)
        
        # Check if model is loaded
        self.assertIsNotNone(model_server.model)
        
        # Check if metadata is loaded
        self.assertEqual(model_server.feature_cols, self.metadata['feature_cols'])
        self.assertEqual(model_server.seq_length, self.metadata['seq_length'])
    
    def test_model_server_predict(self):
        """
        Test making predictions with ModelServer
        """
        # Initialize model server
        model_server = ModelServer(self.model_path)
        
        # Make prediction
        result = model_server.predict(self.sample_data)
        
        # Check if prediction is made
        self.assertIsNotNone(result)
        self.assertIn('prediction', result)
        self.assertIn('timestamp', result)
    
    def test_model_retrainer_init(self):
        """
        Test initializing ModelRetrainer
        """
        # Initialize model retrainer
        retrainer = ModelRetrainer(self.model_dir, self.data_dir)
        
        # Check if retrainer is initialized
        self.assertIsNotNone(retrainer.scheduler)
        self.assertIsNone(retrainer.thread)
    
    def test_model_retrainer_start_stop(self):
        """
        Test starting and stopping ModelRetrainer
        """
        # Initialize model retrainer
        retrainer = ModelRetrainer(self.model_dir, self.data_dir)
        
        # Start retrainer
        retrainer.start()
        
        # Check if thread is started
        self.assertIsNotNone(retrainer.thread)
        self.assertTrue(retrainer.thread.is_alive())
        
        # Stop retrainer
        retrainer.stop()
        
        # Check if thread is stopped
        self.assertFalse(retrainer.thread.is_alive())

if __name__ == '__main__':
    unittest.main()
