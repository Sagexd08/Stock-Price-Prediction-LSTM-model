"""
Visualization Module

This module provides functions for visualizing stock price data,
predictions, and model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_stock_prices(data, title='Stock Prices', save_path=None):
    """
    Plot stock prices over time.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock price data
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting stock prices")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for column in data.columns:
            ax.plot(data.index, data[column], label=column)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting stock prices: {str(e)}")
        raise

def plot_correlation_matrix(data, title='Correlation Matrix', save_path=None):
    """
    Plot correlation matrix of stock prices.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock price data
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting correlation matrix")
        
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise

def plot_technical_indicators(data, indicators, title='Technical Indicators', save_path=None):
    """
    Plot technical indicators.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock price and indicator data
    indicators : list
        List of indicator column names to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting technical indicators")
        
        n_indicators = len(indicators)
        fig, axes = plt.subplots(n_indicators, 1, figsize=(12, 4 * n_indicators), sharex=True)
        
        # If only one indicator, axes is not a list
        if n_indicators == 1:
            axes = [axes]
        
        # Plot price in the first subplot
        axes[0].plot(data.index, data['Close'], label='Close Price')
        axes[0].set_title('Close Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot indicators
        for i, indicator in enumerate(indicators):
            if indicator in data.columns:
                axes[i+1].plot(data.index, data[indicator], label=indicator)
                axes[i+1].set_title(indicator)
                axes[i+1].legend()
                axes[i+1].grid(True, alpha=0.3)
            else:
                logger.warning(f"Indicator {indicator} not found in data")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting technical indicators: {str(e)}")
        raise

def plot_feature_importance(feature_names, importances, title='Feature Importance', save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : list
        List of feature importance values
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting feature importance")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot feature importance
        sns.barplot(x='Importance', y='Feature', data=df.head(20), ax=ax)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def plot_attention_heatmap(attention_weights, dates=None, title='Attention Weights', save_path=None):
    """
    Plot attention weights as a heatmap.
    
    Parameters:
    -----------
    attention_weights : numpy.ndarray
        Attention weights with shape (batch_size, seq_length, 1)
    dates : list, optional
        List of dates corresponding to the sequence
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting attention heatmap")
        
        # Reshape attention weights
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.squeeze(-1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        sns.heatmap(attention_weights, cmap='viridis', ax=ax)
        
        # Set labels
        if dates is not None:
            ax.set_xticklabels(dates)
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sample')
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting attention heatmap: {str(e)}")
        raise

def plot_predictions_interactive(historical_data, predictions, ticker='Stock', output_file=None):
    """
    Create an interactive plot of historical data and predictions using Plotly.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing historical stock price data
    predictions : list
        List of predicted values
    ticker : str, optional
        Stock ticker symbol
    output_file : str, optional
        Path to save the HTML file
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    try:
        logger.info("Creating interactive prediction plot")
        
        # Create figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        # Add historical data
        if isinstance(historical_data, pd.DataFrame):
            if ticker in historical_data.columns:
                historical_values = historical_data[ticker]
            else:
                historical_values = historical_data.iloc[:, 0]  # Use first column
            
            dates = historical_data.index
        else:
            historical_values = historical_data
            dates = list(range(len(historical_data)))
        
        # Create future dates for predictions
        if isinstance(dates, pd.DatetimeIndex):
            last_date = dates[-1]
            future_dates = pd.date_range(start=last_date, periods=len(predictions) + 1)[1:]
        else:
            future_dates = list(range(len(dates), len(dates) + len(predictions)))
        
        # Add historical data trace
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_values,
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Add prediction trace
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x unified'
        )
        
        # Save the plot if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Interactive plot saved to {output_file}")
        
        # Show the plot
        fig.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive prediction plot: {str(e)}")
        raise

def plot_model_comparison(models, metrics, title='Model Comparison', save_path=None):
    """
    Plot model comparison.
    
    Parameters:
    -----------
    models : list
        List of model names
    metrics : dict
        Dictionary of metrics for each model
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        logger.info("Plotting model comparison")
        
        # Get metric names
        metric_names = list(metrics[models[0]].keys())
        
        # Create figure
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
        
        # If only one metric, axes is not a list
        if len(metric_names) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            
            # Add values on top of bars
            for j, value in enumerate(values):
                axes[i].text(j, value, f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting model comparison: {str(e)}")
        raise

def create_dashboard(data, predictions, attention_weights=None, title='Stock Price Prediction Dashboard', output_file=None):
    """
    Create an interactive dashboard using Plotly.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock price data
    predictions : dict
        Dictionary of predictions for each stock
    attention_weights : dict, optional
        Dictionary of attention weights for each stock
    title : str, optional
        Dashboard title
    output_file : str, optional
        Path to save the HTML file
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    try:
        logger.info("Creating interactive dashboard")
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stock Prices and Predictions', 'Attention Weights'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Add stock price and prediction traces
        for stock in predictions.keys():
            if stock in data.columns:
                # Add historical data trace
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[stock],
                        mode='lines',
                        name=f'{stock} Historical',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Create future dates for predictions
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date, periods=len(predictions[stock]) + 1)[1:]
                
                # Add prediction trace
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions[stock],
                        mode='lines+markers',
                        name=f'{stock} Prediction',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Add attention weights if available
                if attention_weights and stock in attention_weights:
                    fig.add_trace(
                        go.Heatmap(
                            z=[attention_weights[stock]],
                            x=data.index[-len(attention_weights[stock]):],
                            y=[stock],
                            colorscale='Viridis',
                            showscale=True,
                            name=f'{stock} Attention'
                        ),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x unified'
        )
        
        # Save the dashboard if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Interactive dashboard saved to {output_file}")
        
        # Show the dashboard
        fig.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    data = pd.DataFrame({
        'Stock_1': np.random.normal(100, 5, 100),
        'Stock_2': np.random.normal(50, 3, 100),
        'Stock_3': np.random.normal(200, 10, 100)
    }, index=dates)
    
    # Plot stock prices
    plot_stock_prices(data, save_path='results/stock_prices.png')
    
    # Plot correlation matrix
    plot_correlation_matrix(data, save_path='results/correlation_matrix.png')
    
    # Create sample predictions
    predictions = {
        'Stock_1': np.random.normal(100, 5, 10),
        'Stock_2': np.random.normal(50, 3, 10),
        'Stock_3': np.random.normal(200, 10, 10)
    }
    
    # Create sample attention weights
    attention_weights = {
        'Stock_1': np.random.rand(20),
        'Stock_2': np.random.rand(20),
        'Stock_3': np.random.rand(20)
    }
    
    # Create interactive dashboard
    create_dashboard(data, predictions, attention_weights, output_file='results/dashboard.html')
