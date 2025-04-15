import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import optuna
import os

# Set up TensorFlow optimizations
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

def load_data(filepath="crypto_ohlcv.csv"):
    """Load data and compute returns and technical indicators."""
    # Import data processing functions from lstm.py
    from lstm import load_data, compute_log_returns, triple_barrier_labeling
    
    # Load and process data
    df = load_data(filepath)
    df = compute_log_returns(df)
    df = triple_barrier_labeling(df, threshold=0.02, hold_period=12)
    
    return df

def create_feature_sets(df, window_size=30):
    """
    Create feature sets for different models in the ensemble.
    Returns multiple feature sets to train different models on.
    """
    feature_sets = {}
    
    # Base features for all models
    base_features = ['open', 'high', 'low', 'close', 'volume', 'log_return']
    
    # Technical indicators
    tech_features = base_features + [
        'rsi', 'sma_10', 'sma_30', 'macd', 'macd_signal', 
        'bb_upper', 'bb_lower', 'bb_width'
    ]
    
    # Volume features
    volume_features = base_features + [
        'volume_change', 'volume_ma_ratio'
    ]
    
    # Price action features
    price_features = base_features + [
        'close_to_high', 'close_to_low', 'high_low_diff'
    ]
    
    # Momentum features
    momentum_features = base_features + [
        'momentum_3', 'momentum_6', 'momentum_12'
    ]
    
    # All features
    all_features = list(set(
        base_features + 
        tech_features + 
        volume_features + 
        price_features + 
        momentum_features
    ))
    
    # Define feature sets for different models
    feature_sets['base'] = base_features
    feature_sets['technical'] = tech_features
    feature_sets['volume'] = volume_features
    feature_sets['price_action'] = price_features
    feature_sets['momentum'] = momentum_features
    feature_sets['all'] = all_features
    
    return feature_sets

def create_windows(data, feature_cols, target_col, window_size=30):
    """
    Create windowed data for time series modeling.
    Returns X (features) and y (target).
    """
    n_samples = len(data) - window_size
    X = np.zeros((n_samples, window_size, len(feature_cols)))
    y = np.zeros(n_samples)
    
    # Extract feature and target data
    feature_data = data[feature_cols].values
    target_data = data[target_col].values
    
    # Create sliding windows
    for i in range(n_samples):
        X[i] = feature_data[i:i+window_size]
        y[i] = target_data[i+window_size]
    
    return X, y

def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, model_params=None):
    """
    Train an LSTM model with the given parameters.
    Returns the trained model.
    """
    if model_params is None:
        model_params = {
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50
        }
    
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units=model_params['lstm_units'],
            input_shape=input_shape,
            return_sequences=False
        ),
        tf.keras.layers.Dropout(model_params['dropout_rate']),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(model_params['dropout_rate']),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params['learning_rate'])
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def train_traditional_models(X_train_2d, y_train, X_val_2d, y_val):
    """
    Train traditional ML models (Random Forest, Gradient Boosting, etc.).
    Returns a dictionary of trained models.
    """
    models = {}
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_2d, y_train)
    models['random_forest'] = rf_model
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train_2d, y_train)
    models['gradient_boosting'] = gb_model
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_2d, y_train)
    models['linear_regression'] = lr_model
    
    return models

def flatten_windows(X):
    """
    Flatten 3D windowed data to 2D for traditional ML models.
    """
    # Get dimensions
    n_samples, time_steps, n_features = X.shape
    
    # Reshape to 2D
    X_2d = X.reshape(n_samples, time_steps * n_features)
    
    return X_2d

def evaluate_model(model, X_test, y_test, model_type='lstm'):
    """
    Evaluate a model and return metrics.
    """
    if model_type == 'lstm':
        y_pred = model.predict(X_test).flatten()
    else:
        # For traditional models
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def ensemble_predictions(model_predictions, weights=None):
    """
    Combine predictions from multiple models using weighted average.
    """
    if weights is None:
        # Equal weights if not specified
        weights = np.ones(len(model_predictions)) / len(model_predictions)
    
    # Stack predictions and calculate weighted average
    stacked_preds = np.column_stack(model_predictions)
    weighted_preds = np.sum(stacked_preds * weights.reshape(1, -1), axis=1)
    
    return weighted_preds

def optimize_ensemble_weights(predictions_list, y_true):
    """
    Use Optuna to find optimal weights for the ensemble.
    """
    def objective(trial):
        # Get weights for each model (sum to 1)
        weights = []
        for i in range(len(predictions_list)):
            w = trial.suggest_float(f"weight_{i}", 0.0, 1.0)
            weights.append(w)
        
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Get ensemble predictions
        ensemble_preds = ensemble_predictions(predictions_list, weights)
        
        # Calculate MSE
        mse = mean_squared_error(y_true, ensemble_preds)
        return mse
    
    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    # Get best weights
    best_params = study.best_params
    best_weights = np.array([best_params[f"weight_{i}"] for i in range(len(predictions_list))])
    
    # Normalize weights to sum to 1
    best_weights = best_weights / np.sum(best_weights)
    
    return best_weights

def save_models(models_dict, base_path="models"):
    """
    Save all models to disk.
    """
    os.makedirs(base_path, exist_ok=True)
    
    for model_name, model in models_dict.items():
        if model_name.startswith('lstm'):
            # Save Keras model
            model.save(os.path.join(base_path, f"{model_name}.h5"))
        else:
            # Save scikit-learn model
            joblib.dump(model, os.path.join(base_path, f"{model_name}.joblib"))
    
    print(f"All models saved to {base_path}")

def load_models(model_names, base_path="models"):
    """
    Load models from disk.
    """
    loaded_models = {}
    
    # Custom objects dictionary for Keras model loading
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'huber': tf.keras.losses.Huber()
    }
    
    for model_name in model_names:
        if model_name.startswith('lstm'):
            # Load Keras model
            loaded_models[model_name] = load_model(os.path.join(base_path, f"{model_name}.h5"), custom_objects=custom_objects)
        else:
            # Load scikit-learn model
            loaded_models[model_name] = joblib.load(os.path.join(base_path, f"{model_name}.joblib"))
    
    return loaded_models

def generate_trading_signals(predictions, threshold=0.01):
    """
    Convert predictions to trading signals (BUY, SELL, HOLD).
    """
    signals = np.where(
        predictions > threshold, 'BUY',
        np.where(predictions < -threshold, 'SELL', 'HOLD')
    )
    return signals

def backtest_model(df, predictions, threshold=0.01, initial_balance=10000):
    """
    Perform backtesting on predictions.
    Returns portfolio value, trades, and performance metrics.
    """
    # Create DataFrame for backtesting
    backtest_df = df.copy()
    backtest_df = backtest_df.iloc[len(backtest_df) - len(predictions):]  # Align indices
    backtest_df['predicted_return'] = predictions
    
    # Generate signals
    backtest_df['signal'] = generate_trading_signals(predictions, threshold)
    
    # Initialize tracking variables
    balance = initial_balance
    position = 0  # 0: no position, 1: long, -1: short
    trades = []
    portfolio_values = [initial_balance]
    
    # Simulate trading
    for i in range(1, len(backtest_df)):
        current_price = backtest_df['close'].iloc[i]
        signal = backtest_df['signal'].iloc[i-1]  # Use previous signal to trade current price
        
        # Execute trades based on signals
        if signal == 'BUY' and position <= 0:
            # Close short position if any
            if position < 0:
                # Calculate profit/loss
                entry_price = trades[-1]['entry_price']
                pl = entry_price - current_price
                balance += pl
                
                # Record trade
                trades[-1]['exit_price'] = current_price
                trades[-1]['exit_time'] = backtest_df.index[i]
                trades[-1]['pl'] = pl
            
            # Open long position
            position = 1
            trades.append({
                'entry_time': backtest_df.index[i],
                'entry_price': current_price,
                'type': 'BUY',
                'position_size': 1
            })
            
        elif signal == 'SELL' and position >= 0:
            # Close long position if any
            if position > 0:
                # Calculate profit/loss
                entry_price = trades[-1]['entry_price']
                pl = current_price - entry_price
                balance += pl
                
                # Record trade
                trades[-1]['exit_price'] = current_price
                trades[-1]['exit_time'] = backtest_df.index[i]
                trades[-1]['pl'] = pl
            
            # Open short position
            position = -1
            trades.append({
                'entry_time': backtest_df.index[i],
                'entry_price': current_price,
                'type': 'SELL',
                'position_size': 1
            })
        
        # Update portfolio value
        portfolio_value = balance
        if position == 1:
            portfolio_value += current_price - trades[-1]['entry_price']
        elif position == -1:
            portfolio_value += trades[-1]['entry_price'] - current_price
        
        portfolio_values.append(portfolio_value)
    
    # Close any open positions at the end
    if position != 0:
        current_price = backtest_df['close'].iloc[-1]
        
        if position == 1:
            # Close long position
            entry_price = trades[-1]['entry_price']
            pl = current_price - entry_price
        else:
            # Close short position
            entry_price = trades[-1]['entry_price']
            pl = entry_price - current_price
        
        balance += pl
        
        # Record trade
        trades[-1]['exit_price'] = current_price
        trades[-1]['exit_time'] = backtest_df.index[-1]
        trades[-1]['pl'] = pl
    
    # Calculate performance metrics
    backtest_df['portfolio_value'] = portfolio_values
    
    # Get completed trades
    completed_trades = [t for t in trades if 'exit_price' in t]
    
    # Calculate metrics
    if completed_trades:
        win_trades = sum(1 for t in completed_trades if t['pl'] > 0)
        loss_trades = sum(1 for t in completed_trades if t['pl'] <= 0)
        win_rate = win_trades / len(completed_trades) if completed_trades else 0
        
        # Calculate average win and loss
        avg_win = np.mean([t['pl'] for t in completed_trades if t['pl'] > 0]) if win_trades else 0
        avg_loss = np.mean([t['pl'] for t in completed_trades if t['pl'] <= 0]) if loss_trades else 0
        
        # Calculate profit factor
        profit_factor = sum(t['pl'] for t in completed_trades if t['pl'] > 0) / abs(sum(t['pl'] for t in completed_trades if t['pl'] < 0)) if sum(t['pl'] for t in completed_trades if t['pl'] < 0) != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Calculate returns
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance
    
    # Calculate drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = np.max(drawdown)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    # Return results
    results = {
        'backtest_df': backtest_df,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'metrics': {
            'total_trades': len(completed_trades),
            'win_trades': win_trades if completed_trades else 0,
            'loss_trades': loss_trades if completed_trades else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': portfolio_values[-1]
        }
    }
    
    return results

def plot_results(backtest_results, model_name):
    """
    Plot backtest results.
    """
    backtest_df = backtest_results['backtest_df']
    portfolio_values = backtest_results['portfolio_values']
    metrics = backtest_results['metrics']
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price and Signals
    axs[0].plot(backtest_df.index, backtest_df['close'], color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_signals = backtest_df[backtest_df['signal'] == 'BUY']
    axs[0].scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = backtest_df[backtest_df['signal'] == 'SELL']
    axs[0].scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell Signal')
    
    axs[0].set_title(f'Price and Trading Signals - {model_name}', fontsize=14)
    axs[0].set_ylabel('Price', fontsize=12)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Portfolio Value
    axs[1].plot(backtest_df.index, portfolio_values[1:], color='green')
    axs[1].set_title('Portfolio Value', fontsize=14)
    axs[1].set_ylabel('Value ($)', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    axs[2].fill_between(backtest_df.index, 0, drawdown[1:], color='red', alpha=0.3)
    axs[2].set_title('Drawdown', fontsize=14)
    axs[2].set_ylabel('Drawdown (%)', fontsize=12)
    axs[2].set_ylim(0, max(drawdown) * 1.1)  # Add some padding
    axs[2].grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = (
        f"Total Return: {metrics['total_return']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Total Trades: {metrics['total_trades']}"
    )
    
    axs[0].text(
        0.02, 0.05, metrics_text,
        transform=axs[0].transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        verticalalignment='bottom',
        fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_backtest_results.png")
    plt.show()

def main():
    # Load data
    print("Loading and processing data...")
    df = load_data()
    
    # Replace infinite values with NaN and then fill with reasonable values
    print("Cleaning data and handling infinite values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, fill NaN with either mean (for most features) or zeros
    for col in df.columns:
        if col == 'timestamp' or col == 'label':
            continue
        if df[col].isna().sum() > 0:
            if col in ['rsi', 'macd', 'macd_signal', 'bb_width']:
                # For indicators that should be in specific ranges
                df[col] = df[col].fillna(df[col].median())
            elif 'momentum' in col:
                df[col] = df[col].fillna(0)  # Momentum usually means no change
            elif col in ['volume_ma_ratio', 'close_to_high', 'close_to_low']:
                df[col] = df[col].fillna(1)  # Ratio indicators default to 1
            else:
                df[col] = df[col].fillna(df[col].mean())
    
    print(f"Data cleaning complete. Shape: {df.shape}")
    
    # Create feature sets
    feature_sets = create_feature_sets(df)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Initialize scalers for each feature set
    scalers = {}
    for set_name, features in feature_sets.items():
        scalers[set_name] = RobustScaler()  # Use RobustScaler instead of StandardScaler
        # Fit scaler only on training data
        scalers[set_name].fit(train_df[features])
        # Save scaler for live trading
        os.makedirs("models", exist_ok=True)
        joblib.dump(scalers[set_name], os.path.join("models", f"scaler_{set_name}.joblib"))
    
    # Scale the data
    scaled_train_df = train_df.copy()
    scaled_val_df = val_df.copy()
    scaled_test_df = test_df.copy()
    
    for set_name, features in feature_sets.items():
        scaled_train_df[features] = scalers[set_name].transform(train_df[features])
        scaled_val_df[features] = scalers[set_name].transform(val_df[features])
        scaled_test_df[features] = scalers[set_name].transform(test_df[features])
    
    # Define target column
    target_col = 'log_return'
    
    # Define window size
    window_size = 30
    
    # Create datasets for each feature set
    lstm_models = {}
    traditional_models = {}
    all_predictions = {}
    
    # For each feature set, train LSTM and traditional models
    for set_name, features in feature_sets.items():
        print(f"\nTraining models with {set_name} features...")
        
        # Create windowed data
        X_train, y_train = create_windows(scaled_train_df, features, target_col, window_size)
        X_val, y_val = create_windows(scaled_val_df, features, target_col, window_size)
        X_test, y_test = create_windows(scaled_test_df, features, target_col, window_size)
        
        # Train LSTM model
        print(f"Training LSTM model with {set_name} features...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm_model, history = train_lstm_model(X_train, y_train, X_val, y_val, input_shape)
        lstm_models[f'lstm_{set_name}'] = lstm_model
        
        # Evaluate LSTM model
        lstm_results = evaluate_model(lstm_model, X_test, y_test, model_type='lstm')
        print(f"LSTM {set_name} - Test MSE: {lstm_results['mse']:.6f}, MAE: {lstm_results['mae']:.6f}, R²: {lstm_results['r2']:.6f}")
        
        # Store predictions
        all_predictions[f'lstm_{set_name}'] = lstm_results['predictions']
        
        # Flatten data for traditional models
        X_train_2d = flatten_windows(X_train)
        X_val_2d = flatten_windows(X_val)
        X_test_2d = flatten_windows(X_test)
        
        # Train traditional models
        print(f"Training traditional models with {set_name} features...")
        trad_models = train_traditional_models(X_train_2d, y_train, X_val_2d, y_val)
        
        # Evaluate and store traditional models
        for model_name, model in trad_models.items():
            model_key = f"{model_name}_{set_name}"
            traditional_models[model_key] = model
            
            # Evaluate model
            trad_results = evaluate_model(model, X_test_2d, y_test, model_type='traditional')
            print(f"{model_name} {set_name} - Test MSE: {trad_results['mse']:.6f}, MAE: {trad_results['mae']:.6f}, R²: {trad_results['r2']:.6f}")
            
            # Store predictions
            all_predictions[model_key] = trad_results['predictions']
    
    # Create ensemble model
    print("\nCreating ensemble model...")
    
    # Get all predictions for the test set
    pred_list = list(all_predictions.values())
    
    # Optimize ensemble weights
    print("Optimizing ensemble weights...")
    best_weights = optimize_ensemble_weights(pred_list, y_test)
    
    # Get ensemble predictions
    ensemble_preds = ensemble_predictions(pred_list, best_weights)
    
    # Evaluate ensemble model
    ensemble_mse = mean_squared_error(y_test, ensemble_preds)
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    
    print(f"Ensemble - Test MSE: {ensemble_mse:.6f}, MAE: {ensemble_mae:.6f}, R²: {ensemble_r2:.6f}")
    
    # Save model weights
    weight_dict = {model_name: weight for model_name, weight in zip(all_predictions.keys(), best_weights)}
    with open('ensemble_weights.json', 'w') as f:
        json.dump(weight_dict, f, indent=4)
    
    # Save all models
    print("\nSaving models...")
    all_models = {**lstm_models, **traditional_models}
    save_models(all_models)
    
    # Backtest individual models and ensemble
    print("\nBacktesting models...")
    
    # Create test dataframe for backtesting
    backtest_df = test_df.copy()
    backtest_df = backtest_df.iloc[window_size:]  # Align with predictions
    
    # Backtest ensemble model
    ensemble_backtest = backtest_model(backtest_df, ensemble_preds)
    
    # Print ensemble backtest results
    print("\nEnsemble Backtest Results:")
    metrics = ensemble_backtest['metrics']
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    
    # Plot ensemble backtest results
    plot_results(ensemble_backtest, "Ensemble")
    
    # Optional: Backtest best individual model
    best_model_key = max(all_predictions.keys(), key=lambda k: r2_score(y_test, all_predictions[k]))
    best_model_preds = all_predictions[best_model_key]
    
    best_model_backtest = backtest_model(backtest_df, best_model_preds)
    print(f"\nBest Individual Model ({best_model_key}) Backtest Results:")
    metrics = best_model_backtest['metrics']
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    
    # Plot best model backtest results
    plot_results(best_model_backtest, best_model_key)
    
    # Export backtest predictions and signals for further analysis
    ensemble_backtest_df = ensemble_backtest['backtest_df']
    backtest_json = ensemble_backtest_df[["timestamp", "close", "log_return", "predicted_return", "signal"]].to_json(orient="records")
    
    with open("ensemble_backtest.json", "w") as f:
        f.write(backtest_json)
    
    print("Ensemble backtest results saved to ensemble_backtest.json")

if __name__ == "__main__":
    main() 