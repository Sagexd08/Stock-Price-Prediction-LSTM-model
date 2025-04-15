"""
Test module for model.py
"""

import os
import sys
import unittest
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    LSTMModel,
    LSTMWithAttention,
    StackedLSTMWithAttention,
    MultiStepLSTM,
    create_model
)

class TestModel(unittest.TestCase):
    """
    Test cases for model.py
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Define test parameters
        self.input_dim = 10
        self.hidden_dim = 64
        self.num_layers = 2
        self.output_dim = 1
        self.batch_size = 32
        self.seq_length = 20
        
        # Create sample input
        self.x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
    
    def test_lstm_model(self):
        """
        Test LSTM model
        """
        # Create model
        model = LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Forward pass
        output = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
    
    def test_lstm_with_attention(self):
        """
        Test LSTM model with attention
        """
        # Create model
        model = LSTMWithAttention(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Forward pass
        output, attention_weights = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_length, 1))
    
    def test_stacked_lstm_with_attention(self):
        """
        Test stacked LSTM model with attention
        """
        # Create model
        model = StackedLSTMWithAttention(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        
        # Forward pass
        output, attention_weights = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_length, 1))
    
    def test_multi_step_lstm(self):
        """
        Test multi-step LSTM model
        """
        # Create model
        forecast_horizon = 5
        model = MultiStepLSTM(self.input_dim, self.hidden_dim, self.num_layers, forecast_horizon)
        
        # Forward pass
        output = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, forecast_horizon))
    
    def test_create_model(self):
        """
        Test model creation function
        """
        # Test creating different model types
        model_types = ['lstm', 'lstm_attention', 'stacked_lstm_attention', 'multi_step_lstm']
        
        for model_type in model_types:
            # Create model
            if model_type == 'multi_step_lstm':
                model = create_model(model_type, self.input_dim, self.hidden_dim, self.num_layers, 5)
            else:
                model = create_model(model_type, self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
            
            # Check if model is created
            self.assertIsNotNone(model)
            
            # Forward pass
            if model_type in ['lstm', 'multi_step_lstm']:
                output = model(self.x)
                self.assertIsNotNone(output)
            else:
                output, attention_weights = model(self.x)
                self.assertIsNotNone(output)
                self.assertIsNotNone(attention_weights)

if __name__ == '__main__':
    unittest.main()
