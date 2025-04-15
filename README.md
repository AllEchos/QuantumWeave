# Quantum Weave

An advanced ensemble-based cryptocurrency trading system that weaves together multiple machine learning models for market prediction and automated trading.

## 🧠 Overview

Quantum Weave combines LSTM neural networks, traditional ML algorithms, and ensemble optimization to create a sophisticated trading system. The project uses TensorFlow for deep learning models and Binance API for market data and execution.

## 🔧 Key Components

- **LSTM Models**: Deep learning models with attention mechanisms for time-series prediction
- **Traditional ML Models**: Random Forest, Gradient Boosting, and Linear Regression
- **Ensemble Architecture**: Optimally weighted combination of models using Optuna
- **Triple-Barrier Labeling**: Advanced labeling technique for financial time series
- **Backtesting Engine**: Comprehensive performance analysis and visualization
- **Live Trading**: Real-time prediction and automated execution

## 📊 Features

- Multi-model ensemble prediction with optimized weights
- Comprehensive technical indicator generation
- Hyperparameter optimization for all models
- Docker containerization for easy deployment
- Backtesting with realistic trading simulation
- Testnet trading integration

## 📈 Performance Metrics

- Direction Accuracy: ~52%
- Sharpe Ratio: ~0.37
- Max Drawdown: ~175%
- Win Rate: ~54%
- Profit Factor: ~1.06

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- pandas, numpy, scikit-learn
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/AllEchos/QuantumWeave.git
cd QuantumWeave

# Install dependencies
pip install -r requirements.txt

# Download historical data
python binance_downloader.py

# Train models
python ensemble_model.py

# Start live trading
python live_trader.py
```

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t quantum-weave:latest .

# Run the container
docker run -v ./data:/app/data -v ./models:/app/models quantum-weave:latest
```

## 🧪 Future Development

- Integration with reinforcement learning via TensorTrade
- Enhanced feature engineering with sentiment analysis
- Improved position sizing based on prediction confidence
- Advanced risk management with dynamic stop-loss
- Integration of transformer models for sequence prediction

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Use at your own risk. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. 