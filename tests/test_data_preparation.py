"""
Test module for data_preparation.py
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import (
    create_sequences,
    create_multistep_sequences,
    train_val_test_split,
    time_series_train_val_test_split,
    TimeSeriesDataset,
    create_data_loaders,
    prepare_data_for_training
)

class TestDataPreparation(unittest.TestCase):
    """
    Test cases for data_preparation.py
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200)
        self.data = pd.DataFrame({
            'Feature1': np.random.normal(0, 1, 200),
            'Feature2': np.random.normal(0, 1, 200),
            'Feature3': np.random.normal(0, 1, 200),
            'Target': np.random.normal(0, 1, 200)
        }, index=dates)
    
    def test_create_sequences(self):
        """
        Test creating sequences
        """
        # Create sequences
        seq_length = 10
        X, y = create_sequences(self.data, seq_length, 'Target')
        
        # Check shapes
        self.assertEqual(X.shape, (len(self.data) - seq_length, seq_length, len(self.data.columns) - 1))
        self.assertEqual(y.shape, (len(self.data) - seq_length,))
        
        # Check values
        self.assertTrue(np.array_equal(X[0, :, 0], self.data['Feature1'].values[:seq_length]))
        self.assertEqual(y[0], self.data['Target'].values[seq_length])
    
    def test_create_multistep_sequences(self):
        """
        Test creating multi-step sequences
        """
        # Create multi-step sequences
        seq_length = 10
        forecast_horizon = 5
        X, y = create_multistep_sequences(self.data, seq_length, 'Target', forecast_horizon)
        
        # Check shapes
        self.assertEqual(X.shape, (len(self.data) - seq_length - forecast_horizon + 1, seq_length, len(self.data.columns) - 1))
        self.assertEqual(y.shape, (len(self.data) - seq_length - forecast_horizon + 1, forecast_horizon))
        
        # Check values
        self.assertTrue(np.array_equal(X[0, :, 0], self.data['Feature1'].values[:seq_length]))
        self.assertTrue(np.array_equal(y[0], self.data['Target'].values[seq_length:seq_length+forecast_horizon]))
    
    def test_train_val_test_split(self):
        """
        Test train/val/test split
        """
        # Create sequences
        seq_length = 10
        X, y = create_sequences(self.data, seq_length, 'Target')
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_size=0.15, test_size=0.15)
        
        # Check shapes
        total_samples = len(X)
        expected_train_size = int(total_samples * 0.7)
        expected_val_size = int(total_samples * 0.15)
        expected_test_size = total_samples - expected_train_size - expected_val_size
        
        self.assertAlmostEqual(len(X_train), expected_train_size, delta=1)
        self.assertAlmostEqual(len(X_val), expected_val_size, delta=1)
        self.assertAlmostEqual(len(X_test), expected_test_size, delta=1)
    
    def test_time_series_train_val_test_split(self):
        """
        Test time series train/val/test split
        """
        # Split data
        train_data, val_data, test_data = time_series_train_val_test_split(self.data, val_size=0.15, test_size=0.15)
        
        # Check shapes
        total_samples = len(self.data)
        expected_train_size = int(total_samples * 0.7)
        expected_val_size = int(total_samples * 0.15)
        expected_test_size = total_samples - expected_train_size - expected_val_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(val_data), expected_val_size)
        self.assertEqual(len(test_data), expected_test_size)
        
        # Check if splits are in correct order
        self.assertTrue(train_data.index[0] < val_data.index[0])
        self.assertTrue(val_data.index[0] < test_data.index[0])
    
    def test_time_series_dataset(self):
        """
        Test TimeSeriesDataset
        """
        # Create sequences
        seq_length = 10
        X, y = create_sequences(self.data, seq_length, 'Target')
        
        # Create dataset
        dataset = TimeSeriesDataset(X, y)
        
        # Check length
        self.assertEqual(len(dataset), len(X))
        
        # Check item
        x, y_item = dataset[0]
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_tensor(y_item))
        self.assertEqual(x.shape, (seq_length, len(self.data.columns) - 1))
        self.assertEqual(y_item.shape, ())
    
    def test_create_data_loaders(self):
        """
        Test creating data loaders
        """
        # Create sequences
        seq_length = 10
        X, y = create_sequences(self.data, seq_length, 'Target')
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_size=0.15, test_size=0.15)
        
        # Create data loaders
        batch_size = 32
        train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
        
        # Check batch size
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(x_batch.shape[0], min(batch_size, len(X_train)))
        self.assertEqual(y_batch.shape[0], min(batch_size, len(y_train)))
    
    def test_prepare_data_for_training(self):
        """
        Test comprehensive data preparation
        """
        # Prepare data for training
        train_loader, val_loader, test_loader, feature_dim = prepare_data_for_training(
            self.data,
            target_col='Target',
            seq_length=10,
            forecast_horizon=1,
            val_size=0.15,
            test_size=0.15,
            batch_size=32
        )
        
        # Check if data loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check feature dimension
        self.assertEqual(feature_dim, len(self.data.columns) - 1)
        
        # Check batch shape
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(x_batch.shape[1], 10)  # seq_length
        self.assertEqual(x_batch.shape[2], feature_dim)

if __name__ == '__main__':
    unittest.main()
