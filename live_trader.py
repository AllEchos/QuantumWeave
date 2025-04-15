import numpy as np
import pandas as pd
import json
import time
import logging
import os
import warnings
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from binance.client import Client
from binance.exceptions import BinanceAPIException
import configparser

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, config_file=None, use_testnet=True):
        """
        Initialize the live trader.
        
        Parameters:
        - config_file: Path to config file with API keys
        - use_testnet: Whether to use Binance testnet
        """
        self.use_testnet = use_testnet
        self.client = self.initialize_client(config_file)
        self.models = {}
        self.ensemble_weights = {}
        self.feature_sets = {}
        self.scalers = {}
        self.window_size = 30
        self.symbol = "BTCUSDT"  # Default symbol
        self.interval = "1h"     # Default interval
        self.position = 0        # 0: no position, 1: long, -1: short
        self.trades = []
        self.last_signal = "HOLD"
        self.balance_history = []
        
        # Trading parameters
        self.position_size = 0.1  # 10% of balance per trade
        self.threshold = 0.01     # Signal threshold
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Load feature processing functions
        from lstm import compute_log_returns
        self.compute_log_returns = compute_log_returns
    
    def initialize_client(self, config_file=None):
        """Initialize Binance client with API keys."""
        api_key = ""
        api_secret = ""
        
        if config_file and os.path.exists(config_file):
            config = configparser.ConfigParser()
            config.read(config_file)
            api_key = config['BINANCE']['API_KEY']
            api_secret = config['BINANCE']['API_SECRET']
        
        if self.use_testnet:
            client = Client(api_key, api_secret, testnet=True)
            logger.info("Using Binance Testnet")
        else:
            client = Client(api_key, api_secret)
            logger.info("Using Binance Live Trading")
        
        return client
    
    def load_models(self, models_path="models"):
        """Load trained models and weights."""
        # Load ensemble weights
        try:
            with open('ensemble_weights.json', 'r') as f:
                self.ensemble_weights = json.load(f)
            
            logger.info(f"Loaded ensemble weights: {self.ensemble_weights}")
            
            # Custom objects dictionary for Keras model loading
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.losses.MeanAbsoluteError(),
                'huber': tf.keras.losses.Huber()
            }
            
            # Load models
            for model_name in self.ensemble_weights.keys():
                if model_name.startswith('lstm'):
                    # Load Keras model
                    model_path = os.path.join(models_path, f"{model_name}.h5")
                    if os.path.exists(model_path):
                        self.models[model_name] = load_model(model_path, custom_objects=custom_objects)
                        logger.info(f"Loaded LSTM model: {model_name}")
                else:
                    # Load scikit-learn model
                    model_path = os.path.join(models_path, f"{model_name}.joblib")
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded traditional model: {model_name}")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
            # Load feature sets
            from ensemble_model import create_feature_sets
            # Create a dummy dataframe to get feature sets
            dummy_df = pd.DataFrame({
                'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1], 
                'log_return': [0], 'rsi': [50], 'sma_10': [1], 'sma_30': [1],
                'macd': [0], 'macd_signal': [0], 'bb_upper': [1], 'bb_lower': [1], 'bb_width': [0],
                'volume_change': [0], 'volume_ma_ratio': [1], 'close_to_high': [1], 
                'close_to_low': [1], 'high_low_diff': [0], 'momentum_3': [0], 
                'momentum_6': [0], 'momentum_12': [0]
            })
            
            self.feature_sets = create_feature_sets(dummy_df)
            logger.info(f"Created feature sets: {list(self.feature_sets.keys())}")
            
            # Load scalers
            for set_name, features in self.feature_sets.items():
                scaler_path = os.path.join(models_path, f"scaler_{set_name}.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[set_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler for {set_name}")
            
            if not self.scalers:
                logger.warning("No scalers found, will use StandardScaler")
                from sklearn.preprocessing import StandardScaler
                for set_name in self.feature_sets:
                    self.scalers[set_name] = StandardScaler()
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_historical_data(self, lookback_hours=150):
        """Get historical data for the trading symbol."""
        try:
            # Calculate lookback time
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Convert to milliseconds timestamp
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            # Get historical klines
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_str=str(start_ts),
                end_str=str(end_ts)
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Keep only necessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Calculate technical indicators
            df = self.compute_log_returns(df)
            
            logger.info(f"Downloaded {len(df)} historical records")
            return df
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def create_features(self, df):
        """Create features for prediction."""
        feature_data = {}
        
        # Scale features for each feature set
        for set_name, features in self.feature_sets.items():
            if set_name in self.scalers:
                # Create a copy of the features to avoid modifying the original
                feature_df = df[features].copy()
                
                # Handle NaN values (shouldn't be any after compute_log_returns)
                feature_df = feature_df.fillna(0)
                
                # Scale features
                scaled_features = self.scalers[set_name].transform(feature_df.values)
                feature_data[set_name] = scaled_features
            else:
                logger.warning(f"No scaler found for {set_name}")
        
        return feature_data
    
    def create_windows(self, feature_data):
        """Create windowed data for LSTM models."""
        windowed_data = {}
        
        for set_name, features in feature_data.items():
            # Get the last window_size rows
            if len(features) >= self.window_size:
                window = features[-self.window_size:]
                # Reshape for LSTM (1, window_size, n_features)
                window_reshaped = window.reshape(1, self.window_size, window.shape[1])
                windowed_data[set_name] = window_reshaped
            else:
                logger.warning(f"Not enough data for {set_name} (need {self.window_size}, got {len(features)})")
        
        return windowed_data
    
    def flatten_windows(self, windowed_data):
        """Flatten windows for traditional models."""
        flattened_data = {}
        
        for set_name, window in windowed_data.items():
            # Reshape (1, window_size, n_features) to (1, window_size * n_features)
            flat = window.reshape(1, -1)
            flattened_data[set_name] = flat
        
        return flattened_data
    
    def predict(self, df):
        """Generate predictions from loaded models."""
        try:
            if not self.models:
                logger.error("No models loaded")
                return None
            
            # Create features
            feature_data = self.create_features(df)
            
            # Create windows
            windowed_data = self.create_windows(feature_data)
            
            # Flatten windows for traditional models
            flattened_data = self.flatten_windows(windowed_data)
            
            # Generate predictions from each model
            predictions = {}
            
            for model_name, model in self.models.items():
                # Extract feature set name from model name
                if "_" in model_name:
                    set_name = model_name.split("_", 1)[1]
                else:
                    set_name = "all"  # Default
                
                if set_name in windowed_data:
                    if model_name.startswith('lstm'):
                        # LSTM model - use windowed data
                        pred = model.predict(windowed_data[set_name], verbose=0).flatten()[0]
                        predictions[model_name] = pred
                    else:
                        # Traditional model - use flattened data
                        pred = model.predict(flattened_data[set_name])[0]
                        predictions[model_name] = pred
            
            # Check if we have all required predictions
            missing_models = set(self.ensemble_weights.keys()) - set(predictions.keys())
            if missing_models:
                logger.warning(f"Missing predictions from models: {missing_models}")
            
            # Generate ensemble prediction using available models
            available_weights = {}
            available_preds = {}
            weight_sum = 0
            
            for model_name, weight in self.ensemble_weights.items():
                if model_name in predictions:
                    available_weights[model_name] = float(weight)
                    available_preds[model_name] = predictions[model_name]
                    weight_sum += float(weight)
            
            # Normalize weights
            if weight_sum > 0:
                for model_name in available_weights:
                    available_weights[model_name] /= weight_sum
            
            # Calculate weighted average
            ensemble_pred = 0
            for model_name, pred in available_preds.items():
                ensemble_pred += pred * available_weights[model_name]
            
            logger.info(f"Ensemble prediction: {ensemble_pred:.6f}")
            
            # Generate signal
            if ensemble_pred > self.threshold:
                signal = "BUY"
            elif ensemble_pred < -self.threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            logger.info(f"Signal: {signal}")
            
            return {
                'timestamp': df['timestamp'].iloc[-1],
                'close': df['close'].iloc[-1],
                'predicted_return': ensemble_pred,
                'signal': signal,
                'individual_predictions': predictions
            }
        
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    def get_account_balance(self):
        """Get account balance."""
        try:
            if self.use_testnet:
                # In testnet, we'll simulate a balance
                return 10000.0
            else:
                account = self.client.get_account()
                balances = {asset['asset']: float(asset['free']) for asset in account['balances']}
                
                # Get USDT balance
                usdt_balance = balances.get('USDT', 0)
                
                # Get balance in trading pair asset
                asset = self.symbol.replace('USDT', '')
                asset_balance = balances.get(asset, 0)
                
                # Get current price of asset
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                price = float(ticker['price'])
                
                # Calculate total balance in USDT
                total_balance = usdt_balance + (asset_balance * price)
                
                return total_balance
        
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0
    
    def execute_trade(self, signal):
        """Execute a trade based on the signal."""
        try:
            if signal == self.last_signal:
                logger.info("Signal unchanged, no trade needed")
                return False
            
            # Get current balance
            balance = self.get_account_balance()
            trade_amount = balance * self.position_size
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            if signal == "BUY" and self.position <= 0:
                # Close short position if exists
                if self.position < 0:
                    if self.use_testnet:
                        # Simulate closing short
                        logger.info(f"Closing short position at {current_price}")
                    else:
                        # Execute real trade to close short
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side="BUY",
                            type="MARKET",
                            quantity=self.calculate_quantity(trade_amount, current_price)
                        )
                        logger.info(f"Closed short position: {order}")
                
                # Open long position
                if self.use_testnet:
                    # Simulate opening long
                    logger.info(f"Opening long position at {current_price}")
                    self.position = 1
                    self.trades.append({
                        'time': datetime.now(),
                        'price': current_price,
                        'type': 'BUY',
                        'amount': trade_amount
                    })
                else:
                    # Execute real trade to open long
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=self.calculate_quantity(trade_amount, current_price)
                    )
                    logger.info(f"Opened long position: {order}")
                    self.position = 1
                    self.trades.append({
                        'time': datetime.now(),
                        'price': current_price,
                        'type': 'BUY',
                        'amount': trade_amount,
                        'order_id': order['orderId']
                    })
            
            elif signal == "SELL" and self.position >= 0:
                # Close long position if exists
                if self.position > 0:
                    if self.use_testnet:
                        # Simulate closing long
                        logger.info(f"Closing long position at {current_price}")
                    else:
                        # Execute real trade to close long
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side="SELL",
                            type="MARKET",
                            quantity=self.calculate_quantity(trade_amount, current_price)
                        )
                        logger.info(f"Closed long position: {order}")
                
                # Open short position
                if self.use_testnet:
                    # Simulate opening short
                    logger.info(f"Opening short position at {current_price}")
                    self.position = -1
                    self.trades.append({
                        'time': datetime.now(),
                        'price': current_price,
                        'type': 'SELL',
                        'amount': trade_amount
                    })
                else:
                    # Execute real trade to open short
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=self.calculate_quantity(trade_amount, current_price)
                    )
                    logger.info(f"Opened short position: {order}")
                    self.position = -1
                    self.trades.append({
                        'time': datetime.now(),
                        'price': current_price,
                        'type': 'SELL',
                        'amount': trade_amount,
                        'order_id': order['orderId']
                    })
            
            elif signal == "HOLD" and self.position != 0:
                # Close any open position
                if self.position > 0:
                    # Close long
                    if self.use_testnet:
                        # Simulate closing long
                        logger.info(f"Closing long position at {current_price}")
                        self.position = 0
                    else:
                        # Execute real trade to close long
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side="SELL",
                            type="MARKET",
                            quantity=self.calculate_quantity(trade_amount, current_price)
                        )
                        logger.info(f"Closed long position: {order}")
                        self.position = 0
                
                elif self.position < 0:
                    # Close short
                    if self.use_testnet:
                        # Simulate closing short
                        logger.info(f"Closing short position at {current_price}")
                        self.position = 0
                    else:
                        # Execute real trade to close short
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side="BUY",
                            type="MARKET",
                            quantity=self.calculate_quantity(trade_amount, current_price)
                        )
                        logger.info(f"Closed short position: {order}")
                        self.position = 0
            
            # Update last signal
            self.last_signal = signal
            
            # Track balance
            current_balance = self.get_account_balance()
            self.balance_history.append({
                'time': datetime.now(),
                'balance': current_balance
            })
            
            return True
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def calculate_quantity(self, amount, price):
        """Calculate quantity based on amount and price, respecting exchange rules."""
        # Get symbol info
        info = self.client.get_symbol_info(self.symbol)
        
        # Extract lot size filter
        lot_size = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        
        if lot_size:
            min_qty = float(lot_size['minQty'])
            step_size = float(lot_size['stepSize'])
            
            # Calculate quantity
            qty = amount / price
            
            # Round to step size
            qty = round(qty / step_size) * step_size
            
            # Ensure minimum quantity
            qty = max(qty, min_qty)
            
            # Format to correct decimal places
            decimal_places = len(str(step_size).split('.')[-1].rstrip('0'))
            qty = float(f"%.{decimal_places}f" % qty)
            
            return qty
        else:
            # If no lot size filter, just divide and round to 6 decimal places
            return round(amount / price, 6)
    
    def save_trading_history(self):
        """Save trading history to file."""
        history = {
            'trades': self.trades,
            'balance_history': self.balance_history
        }
        
        with open('trading_history.json', 'w') as f:
            json.dump(history, f, default=str, indent=4)
        
        logger.info("Trading history saved")
    
    def run(self, interval_seconds=3600, max_iterations=None):
        """
        Run the live trader.
        
        Parameters:
        - interval_seconds: Seconds between trading decisions
        - max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info("Starting live trader")
        
        # Load models
        if not self.load_models():
            logger.error("Failed to load models, aborting")
            return
        
        iteration = 0
        
        try:
            while True:
                # Break if max iterations reached
                if max_iterations is not None and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations})")
                    break
                
                logger.info(f"\n--- Trading Iteration {iteration + 1} ---")
                
                # Get historical data
                df = self.get_historical_data()
                
                if df is None or len(df) < self.window_size:
                    logger.error("Not enough data for prediction")
                    time.sleep(60)  # Wait a bit before retrying
                    continue
                
                # Generate prediction
                prediction = self.predict(df)
                
                if prediction:
                    # Execute trade based on signal
                    self.execute_trade(prediction['signal'])
                
                # Save trading history
                self.save_trading_history()
                
                # Increment iteration counter
                iteration += 1
                
                # Sleep until next interval
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Trader stopped by user")
        
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        
        finally:
            # Save final trading history
            self.save_trading_history()
            logger.info("Trader stopped")

def main():
    # Configuration
    config_file = None  # Path to config file with API keys
    use_testnet = True  # Use Binance testnet
    interval_seconds = 3600  # 1 hour
    max_iterations = 24  # Run for 24 hours
    
    # Create and run trader
    trader = LiveTrader(config_file, use_testnet)
    trader.symbol = "BTCUSDT"  # Trading pair
    trader.interval = "1h"     # Timeframe
    trader.position_size = 0.1  # 10% of balance per trade
    trader.threshold = 0.01     # Signal threshold
    
    trader.run(interval_seconds, max_iterations)

if __name__ == "__main__":
    main() 