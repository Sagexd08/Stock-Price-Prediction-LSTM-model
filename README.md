# Stock Price Prediction LSTM Model

An advanced stock price prediction system using LSTM (Long Short-Term Memory) neural networks with attention mechanisms. This project leverages deep learning techniques to forecast stock prices with high accuracy while providing comprehensive visualization and analysis tools.

## Features

- Multiple model architectures (LSTM, LSTM with Attention, Bidirectional LSTM, Stacked LSTM)
- Technical indicator analysis
- Interactive visualization dashboard
- Monte Carlo simulations for uncertainty estimation
- Real-time data fetching from Yahoo Finance
- Comprehensive performance analysis

## Project Structure

```
├── data/
│   ├── raw/         # Raw data files
│   └── processed/   # Processed and cleaned data
├── models/          # Trained model files
├── notebooks/       # Jupyter notebooks for analysis
├── src/            # Source code
├── tests/          # Unit tests
├── logs/           # Logging files
└── results/        # Model outputs and visualizations
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sagexd08/Stock-Price-Prediction-LSTM-model.git
cd Stock-Price-Prediction-LSTM-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Use the web interface to:
   - Load or download stock data
   - Configure model parameters
   - Train the model
   - Generate and visualize predictions
   - Analyze model performance

## Technologies Used

- Python 3.x
- PyTorch
- Streamlit
- Plotly
- YFinance
- Pandas
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Sohom (Sagexd08)
