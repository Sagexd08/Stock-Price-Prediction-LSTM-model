"""
Test module for data_acquisition.py
"""

import os
import sys
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_acquisition import download_stock_data, load_stock_data, merge_stock_with_index

class TestDataAcquisition(unittest.TestCase):
    """
    Test cases for data_acquisition.py
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Define test parameters
        self.ticker = 'AAPL'
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)  # Last 30 days
        self.test_file = os.path.join(self.test_dir, 'test_stock_data.csv')
    
    def tearDown(self):
        """
        Clean up test environment
        """
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_download_stock_data(self):
        """
        Test downloading stock data from Yahoo Finance
        """
        # Download data
        data = download_stock_data(
            self.ticker,
            self.start_date.strftime('%Y-%m-%d'),
            self.end_date.strftime('%Y-%m-%d'),
            save_path=self.test_file
        )
        
        # Check if data is downloaded
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        
        # Check if file is created
        self.assertTrue(os.path.exists(self.test_file))
        
        # Check if data has expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_load_stock_data(self):
        """
        Test loading stock data from CSV
        """
        # First download data
        download_stock_data(
            self.ticker,
            self.start_date.strftime('%Y-%m-%d'),
            self.end_date.strftime('%Y-%m-%d'),
            save_path=self.test_file
        )
        
        # Load data
        data = load_stock_data(self.test_file)
        
        # Check if data is loaded
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        
        # Check if data has expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_merge_stock_with_index(self):
        """
        Test merging stock data with market index data
        """
        # Create sample data
        stock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        index_data = pd.DataFrame({
            'Close': [1000, 1010, 1020, 1030, 1040],
            'Volume': [10000, 11000, 12000, 13000, 14000]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Merge data
        merged_data = merge_stock_with_index(stock_data, index_data)
        
        # Check if data is merged
        self.assertIsNotNone(merged_data)
        self.assertFalse(merged_data.empty)
        
        # Check if merged data has expected columns
        expected_columns = ['Close', 'Volume', 'Close_index', 'Volume_index']
        for col in expected_columns:
            self.assertIn(col, merged_data.columns)
        
        # Check if merged data has expected values
        self.assertEqual(merged_data.shape[0], 5)
        self.assertEqual(merged_data['Close'].iloc[0], 100)
        self.assertEqual(merged_data['Close_index'].iloc[0], 1000)

if __name__ == '__main__':
    unittest.main()
