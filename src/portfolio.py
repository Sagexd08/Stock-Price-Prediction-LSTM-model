"""
Portfolio Management Module

This module provides functions for tracking and managing stock portfolios.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import yfinance as yf

# Import Firebase configuration
from firebase_config import get_firebase_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Class for managing user portfolios.
    """
    def __init__(self):
        """
        Initialize the portfolio manager.
        """
        self.firebase = get_firebase_manager()
    
    def get_user_portfolio(self, username):
        """
        Get a user's portfolio.
        
        Parameters:
        -----------
        username : str
            Username
        
        Returns:
        --------
        dict
            Portfolio data
        """
        try:
            # Try to load from Firebase if available
            if self.firebase.initialized:
                doc_ref = self.firebase.db.collection('portfolios').document(username)
                doc = doc_ref.get()
                
                if doc.exists:
                    return doc.to_dict()
            
            # Fall back to local storage
            return self._load_local_portfolio(username)
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {str(e)}")
            return self._load_local_portfolio(username)
    
    def _load_local_portfolio(self, username):
        """
        Load portfolio from local storage.
        """
        try:
            portfolio_file = os.path.join('data', 'portfolios', f'{username}.json')
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    return json.load(f)
            else:
                # Create empty portfolio
                empty_portfolio = {
                    'holdings': [],
                    'watchlist': [],
                    'transactions': [],
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)
                
                # Save empty portfolio
                with open(portfolio_file, 'w') as f:
                    json.dump(empty_portfolio, f, indent=4)
                
                return empty_portfolio
                
        except Exception as e:
            logger.error(f"Error loading local portfolio: {str(e)}")
            return {
                'holdings': [],
                'watchlist': [],
                'transactions': [],
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def save_portfolio(self, username, portfolio):
        """
        Save a user's portfolio.
        
        Parameters:
        -----------
        username : str
            Username
        portfolio : dict
            Portfolio data
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            portfolio['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to save to Firebase if available
            if self.firebase.initialized:
                self.firebase.db.collection('portfolios').document(username).set(portfolio)
            
            # Also save locally as backup
            return self._save_local_portfolio(username, portfolio)
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {str(e)}")
            return self._save_local_portfolio(username, portfolio)
    
    def _save_local_portfolio(self, username, portfolio):
        """
        Save portfolio to local storage.
        """
        try:
            portfolio_file = os.path.join('data', 'portfolios', f'{username}.json')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)
            
            # Save portfolio
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio, f, indent=4)
            
            return True
                
        except Exception as e:
            logger.error(f"Error saving local portfolio: {str(e)}")
            return False
    
    def add_to_watchlist(self, username, ticker):
        """
        Add a stock to the user's watchlist.
        
        Parameters:
        -----------
        username : str
            Username
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Get current portfolio
            portfolio = self.get_user_portfolio(username)
            
            # Add to watchlist if not already there
            if ticker not in portfolio['watchlist']:
                portfolio['watchlist'].append(ticker)
                
                # Save updated portfolio
                return self.save_portfolio(username, portfolio)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {str(e)}")
            return False
    
    def remove_from_watchlist(self, username, ticker):
        """
        Remove a stock from the user's watchlist.
        
        Parameters:
        -----------
        username : str
            Username
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Get current portfolio
            portfolio = self.get_user_portfolio(username)
            
            # Remove from watchlist if present
            if ticker in portfolio['watchlist']:
                portfolio['watchlist'].remove(ticker)
                
                # Save updated portfolio
                return self.save_portfolio(username, portfolio)
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            return False
    
    def add_transaction(self, username, transaction):
        """
        Add a transaction to the user's portfolio.
        
        Parameters:
        -----------
        username : str
            Username
        transaction : dict
            Transaction data (type, ticker, shares, price, date)
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Get current portfolio
            portfolio = self.get_user_portfolio(username)
            
            # Add transaction ID and timestamp
            transaction['id'] = len(portfolio['transactions']) + 1
            if 'date' not in transaction:
                transaction['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Add transaction
            portfolio['transactions'].append(transaction)
            
            # Update holdings
            self._update_holdings(portfolio)
            
            # Save updated portfolio
            return self.save_portfolio(username, portfolio)
            
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return False
    
    def _update_holdings(self, portfolio):
        """
        Update holdings based on transactions.
        
        Parameters:
        -----------
        portfolio : dict
            Portfolio data
        """
        # Reset holdings
        holdings = {}
        
        # Process all transactions
        for transaction in portfolio['transactions']:
            ticker = transaction['ticker']
            shares = transaction['shares']
            price = transaction['price']
            
            if transaction['type'] == 'buy':
                if ticker not in holdings:
                    holdings[ticker] = {
                        'ticker': ticker,
                        'shares': 0,
                        'cost_basis': 0,
                        'value': 0
                    }
                
                # Update shares and cost basis
                current_shares = holdings[ticker]['shares']
                current_cost = holdings[ticker]['cost_basis']
                
                new_shares = current_shares + shares
                new_cost = current_cost + (shares * price)
                
                holdings[ticker]['shares'] = new_shares
                holdings[ticker]['cost_basis'] = new_cost
                
            elif transaction['type'] == 'sell':
                if ticker in holdings:
                    # Update shares
                    holdings[ticker]['shares'] -= shares
                    
                    # If shares are zero or negative, remove from holdings
                    if holdings[ticker]['shares'] <= 0:
                        del holdings[ticker]
        
        # Convert to list
        portfolio['holdings'] = list(holdings.values())
    
    def get_portfolio_value(self, username):
        """
        Get the current value of a user's portfolio.
        
        Parameters:
        -----------
        username : str
            Username
        
        Returns:
        --------
        dict
            Portfolio value data
        """
        try:
            # Get current portfolio
            portfolio = self.get_user_portfolio(username)
            
            # Get current prices for holdings
            tickers = [holding['ticker'] for holding in portfolio['holdings']]
            
            if not tickers:
                return {
                    'total_value': 0,
                    'total_cost': 0,
                    'gain_loss': 0,
                    'gain_loss_percent': 0,
                    'holdings': []
                }
            
            # Get current prices
            current_prices = self._get_current_prices(tickers)
            
            # Calculate portfolio value
            total_value = 0
            total_cost = 0
            holdings_with_value = []
            
            for holding in portfolio['holdings']:
                ticker = holding['ticker']
                shares = holding['shares']
                cost_basis = holding['cost_basis']
                
                # Get current price
                current_price = current_prices.get(ticker, 0)
                
                # Calculate value
                value = shares * current_price
                
                # Update holding
                holding_with_value = holding.copy()
                holding_with_value['current_price'] = current_price
                holding_with_value['value'] = value
                holding_with_value['gain_loss'] = value - cost_basis
                holding_with_value['gain_loss_percent'] = (value - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
                
                holdings_with_value.append(holding_with_value)
                
                # Update totals
                total_value += value
                total_cost += cost_basis
            
            # Calculate overall gain/loss
            gain_loss = total_value - total_cost
            gain_loss_percent = (gain_loss / total_cost * 100) if total_cost > 0 else 0
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent,
                'holdings': holdings_with_value
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio value: {str(e)}")
            return {
                'total_value': 0,
                'total_cost': 0,
                'gain_loss': 0,
                'gain_loss_percent': 0,
                'holdings': []
            }
    
    def _get_current_prices(self, tickers):
        """
        Get current prices for a list of tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        
        Returns:
        --------
        dict
            Dictionary of ticker -> price
        """
        try:
            # Use yfinance to get current prices
            data = yf.download(tickers, period='1d')
            
            # If only one ticker, data will be a Series
            if len(tickers) == 1:
                return {tickers[0]: data['Close'][-1] if not data.empty else 0}
            
            # Get the last close price for each ticker
            prices = {}
            for ticker in tickers:
                if ticker in data['Close'].columns:
                    prices[ticker] = data['Close'][ticker][-1]
                else:
                    prices[ticker] = 0
            
            return prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {str(e)}")
            return {ticker: 0 for ticker in tickers}

# Initialize portfolio manager
portfolio_manager = PortfolioManager()

# Export the instance for use in other modules
def get_portfolio_manager():
    """
    Get the portfolio manager instance.
    
    Returns:
    --------
    PortfolioManager
        Portfolio manager instance
    """
    return portfolio_manager
