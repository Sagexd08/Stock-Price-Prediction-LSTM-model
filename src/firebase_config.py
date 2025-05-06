"""
Firebase Configuration Module

This module provides functions for interacting with Firebase services,
including Firestore, Storage, and Authentication.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Firebase modules, with graceful fallback if not available
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage, auth
    FIREBASE_AVAILABLE = True
except ImportError:
    logger.warning("Firebase modules not available. Using local storage instead.")
    FIREBASE_AVAILABLE = False

class FirebaseManager:
    """
    Class for managing Firebase interactions.
    """
    def __init__(self, credential_path=None):
        """
        Initialize the Firebase manager.
        
        Parameters:
        -----------
        credential_path : str, optional
            Path to the Firebase service account credentials JSON file
        """
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase modules not available. Using local storage.")
            return
        
        try:
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                if credential_path and os.path.exists(credential_path):
                    # Initialize with service account
                    cred = credentials.Certificate(credential_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'stock-price-prediction-app.appspot.com'
                    })
                else:
                    # Try to initialize with default credentials
                    firebase_admin.initialize_app()
                
            # Initialize Firestore client
            self.db = firestore.client()
            
            # Initialize Storage client
            self.bucket = storage.bucket()
            
            self.initialized = True
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Firebase: {str(e)}")
    
    def save_stock_data(self, data, ticker):
        """
        Save stock data to Firestore and/or Storage.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Stock data to save
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self.initialized:
            # Save locally if Firebase is not available
            return self._save_local(data, ticker)
        
        try:
            # Convert DataFrame to dict for Firestore
            data_dict = data.reset_index().to_dict(orient='records')
            
            # Save to Firestore (limited to recent data due to document size limits)
            recent_data = data_dict[-30:] if len(data_dict) > 30 else data_dict
            
            # Add metadata
            metadata = {
                'ticker': ticker,
                'last_updated': firestore.SERVER_TIMESTAMP,
                'count': len(recent_data)
            }
            
            # Save to Firestore
            self.db.collection('stock_data').document(ticker).set(metadata)
            
            # Save recent data points as subcollection
            batch = self.db.batch()
            for i, record in enumerate(recent_data):
                # Convert date to string if it's a datetime
                if 'Date' in record and isinstance(record['Date'], (datetime, pd.Timestamp)):
                    record['Date'] = record['Date'].strftime('%Y-%m-%d')
                
                # Convert numpy types to Python native types
                for key, value in record.items():
                    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                        record[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32, np.float16)):
                        record[key] = float(value)
                
                doc_ref = self.db.collection('stock_data').document(ticker).collection('data').document(f'day_{i}')
                batch.set(doc_ref, record)
            
            batch.commit()
            
            # Save full dataset to Storage
            csv_content = data.reset_index().to_csv(index=False)
            blob = self.bucket.blob(f'stock_data/{ticker}.csv')
            blob.upload_from_string(csv_content, content_type='text/csv')
            
            logger.info(f"Stock data for {ticker} saved to Firebase")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stock data to Firebase: {str(e)}")
            # Fall back to local storage
            return self._save_local(data, ticker)
    
    def load_stock_data(self, ticker):
        """
        Load stock data from Firestore or Storage.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        pandas.DataFrame
            Stock data
        """
        if not self.initialized:
            # Load locally if Firebase is not available
            return self._load_local(ticker)
        
        try:
            # Try to load from Storage first (complete dataset)
            blob = self.bucket.blob(f'stock_data/{ticker}.csv')
            
            if blob.exists():
                # Download as string and convert to DataFrame
                csv_content = blob.download_as_string().decode('utf-8')
                data = pd.read_csv(pd.StringIO(csv_content))
                
                # Convert Date column to datetime and set as index
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
                
                logger.info(f"Stock data for {ticker} loaded from Firebase Storage")
                return data
            else:
                # If not in Storage, try Firestore (limited dataset)
                doc_ref = self.db.collection('stock_data').document(ticker)
                doc = doc_ref.get()
                
                if doc.exists:
                    # Get data subcollection
                    data_docs = doc_ref.collection('data').stream()
                    records = [doc.to_dict() for doc in data_docs]
                    
                    if records:
                        data = pd.DataFrame(records)
                        
                        # Convert Date column to datetime and set as index
                        if 'Date' in data.columns:
                            data['Date'] = pd.to_datetime(data['Date'])
                            data.set_index('Date', inplace=True)
                        
                        logger.info(f"Stock data for {ticker} loaded from Firestore")
                        return data
            
            # If data not found in Firebase, fall back to local
            logger.warning(f"Stock data for {ticker} not found in Firebase")
            return self._load_local(ticker)
            
        except Exception as e:
            logger.error(f"Error loading stock data from Firebase: {str(e)}")
            # Fall back to local storage
            return self._load_local(ticker)
    
    def save_model(self, model_path, metadata=None):
        """
        Save model to Firebase Storage.
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
        metadata : dict, optional
            Model metadata
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Firebase not initialized. Model not saved to Firebase.")
            return False
        
        try:
            # Upload model file to Storage
            blob = self.bucket.blob(f'models/{os.path.basename(model_path)}')
            blob.upload_from_filename(model_path)
            
            # Save metadata to Firestore if provided
            if metadata:
                # Add timestamp
                metadata['uploaded_at'] = firestore.SERVER_TIMESTAMP
                
                # Save to Firestore
                self.db.collection('models').document(os.path.basename(model_path)).set(metadata)
            
            logger.info(f"Model saved to Firebase: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to Firebase: {str(e)}")
            return False
    
    def _save_local(self, data, ticker):
        """
        Save data locally as fallback.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
            
            # Save to CSV
            file_path = os.path.join('data', 'raw', f'{ticker}_data.csv')
            data.reset_index().to_csv(file_path, index=False)
            
            logger.info(f"Stock data for {ticker} saved locally to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stock data locally: {str(e)}")
            return False
    
    def _load_local(self, ticker):
        """
        Load data locally as fallback.
        """
        try:
            # Try ticker-specific file first
            file_path = os.path.join('data', 'raw', f'{ticker}_data.csv')
            
            if not os.path.exists(file_path):
                # Fall back to generic stock_data.csv
                file_path = os.path.join('data', 'raw', 'stock_data.csv')
            
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                
                # Convert Date column to datetime and set as index
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
                
                logger.info(f"Stock data loaded locally from {file_path}")
                return data
            else:
                logger.warning(f"No local stock data found for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading stock data locally: {str(e)}")
            return None

# Initialize Firebase manager
firebase_manager = FirebaseManager()

# Export the instance for use in other modules
def get_firebase_manager():
    """
    Get the Firebase manager instance.
    
    Returns:
    --------
    FirebaseManager
        Firebase manager instance
    """
    return firebase_manager
