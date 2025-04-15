"""
Test module for feature_engineering.py
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    add_technical_indicators,
    add_statistical_features,
    add_lag_features,
    normalize_features,
    prepare_features
)

class TestFeatureEngineering(unittest.TestCase):
    """
    Test cases for feature_engineering.py
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=100)
        self.stock_data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 100),
            'High': np.random.normal(105, 5, 100),
            'Low': np.random.normal(95, 5, 100),
            'Close': np.random.normal(100, 5, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
    
    def test_add_technical_indicators(self):
        """
        Test adding technical indicators
        """
        # Add technical indicators
        result_df = add_technical_indicators(self.stock_data)
        
        # Check if indicators are added
        expected_indicators = ['SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'MACD_12_26_9', 'RSI_14']
        for indicator in expected_indicators:
            self.assertIn(indicator, result_df.columns)
        
        # Check if data has expected shape
        self.assertEqual(result_df.shape[0], self.stock_data.shape[0])
        self.assertGreater(result_df.shape[1], self.stock_data.shape[1])
    
    def test_add_statistical_features(self):
        """
        Test adding statistical features
        """
        # Add statistical features
        result_df = add_statistical_features(self.stock_data, price_col='Close')
        
        # Check if features are added
        expected_features = ['Returns', 'LogReturns', 'Volatility_5', 'Skewness_5', 'Kurtosis_5']
        for feature in expected_features:
            self.assertIn(feature, result_df.columns)
        
        # Check if data has expected shape
        self.assertEqual(result_df.shape[0], self.stock_data.shape[0])
        self.assertGreater(result_df.shape[1], self.stock_data.shape[1])
    
    def test_add_lag_features(self):
        """
        Test adding lag features
        """
        # Add lag features
        result_df = add_lag_features(self.stock_data, cols_to_lag=['Close', 'Volume'])
        
        # Check if features are added
        expected_features = ['Close_lag_1', 'Close_lag_2', 'Volume_lag_1', 'Volume_lag_2']
        for feature in expected_features:
            self.assertIn(feature, result_df.columns)
        
        # Check if data has expected shape
        self.assertEqual(result_df.shape[0], self.stock_data.shape[0])
        self.assertGreater(result_df.shape[1], self.stock_data.shape[1])
    
    def test_normalize_features(self):
        """
        Test normalizing features
        """
        # Normalize features
        result_df, scaler = normalize_features(self.stock_data)
        
        # Check if data is normalized
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(scaler)
        
        # Check if data has expected shape
        self.assertEqual(result_df.shape, self.stock_data.shape)
        
        # Check if values are normalized (mean close to 0, std close to 1)
        for col in result_df.columns:
            self.assertAlmostEqual(result_df[col].mean(), 0, delta=0.5)
            self.assertAlmostEqual(result_df[col].std(), 1, delta=0.5)
    
    def test_prepare_features(self):
        """
        Test comprehensive feature preparation
        """
        # Prepare features
        result_df, transformers = prepare_features(
            self.stock_data,
            target_col='Close',
            include_technical=True,
            include_statistical=True,
            include_lags=True,
            normalize=True,
            reduce_dim=False,
            forecast_horizon=5
        )
        
        # Check if features are prepared
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(transformers)
        
        # Check if target column is created
        self.assertIn('Target_5', result_df.columns)
        
        # Check if scaler is in transformers
        self.assertIn('scaler', transformers)
        
        # Check if data has expected shape
        self.assertLess(result_df.shape[0], self.stock_data.shape[0])  # Some rows are dropped due to NaN values
        self.assertGreater(result_df.shape[1], self.stock_data.shape[1])

if __name__ == '__main__':
    unittest.main()
