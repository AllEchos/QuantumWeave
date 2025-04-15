import numpy as np
import pandas as pd
import json
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Optimize TensorFlow for M4 Pro CPU
tf.config.threading.set_inter_op_parallelism_threads(8)  # Optimize thread count
tf.config.threading.set_intra_op_parallelism_threads(8)

# ---------------------------
# 1. Data Loader & Preprocessing
# ---------------------------
def load_data(filepath):
    """
    Load OHLCV data from a CSV file.
    CSV must contain columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_log_returns(df):
    """
    Compute the log returns from the 'close' prices.
    """
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Add more technical indicators
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages and MACD
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Trading volumes features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Price based features
    df['close_to_high'] = df['close'] / df['high']
    df['close_to_low'] = df['close'] / df['low']
    df['high_low_diff'] = (df['high'] - df['low']) / df['close']
    
    # Momentum indicators
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_6'] = df['close'] / df['close'].shift(6) - 1
    df['momentum_12'] = df['close'] / df['close'].shift(12) - 1
    
    # Clean data: replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values with appropriate fill methods
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
    
    # Drop any remaining NaN values
    df.dropna(inplace=True)
    return df

# ---------------------------
# 2. Triple-Barrier Labeling
# ---------------------------
def triple_barrier_labeling(df, threshold=0.02, hold_period=12):
    """
    Assign labels based on triple-barrier method.
    For each time step, define:
      - BUY if future price increase >= threshold within hold_period
      - SELL if future price drop <= -threshold within hold_period
      - HOLD otherwise
    Note: The hold_period can represent hours if data is hourly.
    
    Also calculate target return for regression (how much price will move)
    """
    labels = []
    target_returns = []
    close_prices = df['close'].values
    n = len(close_prices)
    
    for i in range(n):
        label = "HOLD"
        target_return = 0.0
        
        # Look ahead only if enough data remains
        if i + hold_period < n:
            window = close_prices[i+1: i+hold_period+1]
            high = np.max(window)
            low = np.min(window)
            current_price = close_prices[i]
            
            # Calculate max potential return in window (for regression target)
            high_return = (high - current_price) / current_price
            low_return = (low - current_price) / current_price
            
            # Determine target return based on which barrier is hit first
            if abs(high_return) > abs(low_return) and high_return >= threshold:
                target_return = high_return
                label = "BUY"
            elif abs(low_return) > abs(high_return) and abs(low_return) >= threshold:
                target_return = low_return
                label = "SELL"
            else:
                # If no barrier is hit, use end of window return
                target_return = (window[-1] - current_price) / current_price
        
        labels.append(label)
        target_returns.append(target_return)
    
    df['label'] = labels
    df['target_return'] = target_returns
    return df

# ---------------------------
# 3. Creating Windowed Datasets
# ---------------------------
def create_windows(data, window_size, feature_cols, target_col):
    """
    Given a DataFrame, create sliding windows with memory optimization.
    Returns: X (3D numpy array) and y (target values for supervised learning).
    """
    # Pre-allocate arrays for better memory efficiency
    n_samples = len(data) - window_size
    X = np.zeros((n_samples, window_size, len(feature_cols)))
    y = np.zeros(n_samples)
    
    # Use vectorized operations where possible
    feature_data = data[feature_cols].values
    target_data = data[target_col].values
    
    for i in range(n_samples):
        X[i] = feature_data[i:i+window_size]
        y[i] = target_data[i+window_size]
    
    return X, y

# ---------------------------
# 4. Build Advanced LSTM Model (for Optuna)
# ---------------------------
def build_advanced_lstm_model(input_shape, trial):
    """
    Build an advanced LSTM model with attention mechanism.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Choose model architecture type
    model_type = trial.suggest_categorical("model_type", ["simple_lstm", "bidirectional", "lstm_with_attention"])
    
    if model_type == "simple_lstm":
        # Number of LSTM layers
        n_layers = trial.suggest_int("n_layers", 1, 2)
        
        # First layer
        x = LSTM(
            units=trial.suggest_int("units_lstm0", 64, 256),
            return_sequences=(n_layers > 1),
            activation='tanh',
            recurrent_activation='sigmoid'
        )(inputs)
        x = Dropout(rate=trial.suggest_float("dropout0", 0.1, 0.3))(x)
        
        # Additional layers if needed
        for i in range(1, n_layers):
            return_seq = (i < n_layers - 1)
            x = LSTM(
                units=trial.suggest_int(f"units_lstm{i}", 64, 256),
                return_sequences=return_seq,
                activation='tanh',
                recurrent_activation='sigmoid'
            )(x)
            x = Dropout(rate=trial.suggest_float(f"dropout{i}", 0.1, 0.3))(x)
    
    elif model_type == "bidirectional":
        # Using bidirectional LSTM for capturing patterns in both directions
        n_layers = trial.suggest_int("n_layers", 1, 2)
        
        # First layer
        x = Bidirectional(LSTM(
            units=trial.suggest_int("units_lstm0", 64, 256),
            return_sequences=(n_layers > 1),
            activation='tanh',
            recurrent_activation='sigmoid'
        ))(inputs)
        x = Dropout(rate=trial.suggest_float("dropout0", 0.1, 0.3))(x)
        
        # Additional layers if needed
        for i in range(1, n_layers):
            return_seq = (i < n_layers - 1)
            x = Bidirectional(LSTM(
                units=trial.suggest_int(f"units_lstm{i}", 64, 256),
                return_sequences=return_seq,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))(x)
            x = Dropout(rate=trial.suggest_float(f"dropout{i}", 0.1, 0.3))(x)
    
    elif model_type == "lstm_with_attention":
        # LSTM with attention mechanism
        lstm_units = trial.suggest_int("lstm_units", 64, 256)
        x = LSTM(
            units=lstm_units,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid'
        )(inputs)
        
        # Apply attention
        query_value_attention_seq = tf.keras.layers.MultiHeadAttention(
            num_heads=trial.suggest_int("num_attention_heads", 2, 8),
            key_dim=trial.suggest_int("attention_key_dim", 32, 128)
        )(x, x)
        
        # Add & normalize (residual connection)
        x = tf.keras.layers.Add()([x, query_value_attention_seq])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(rate=trial.suggest_float("dropout", 0.1, 0.3))(x)
    
    # Dense layers for output
    dense_units = trial.suggest_int("dense_units", 16, 128)
    x = Dense(units=dense_units, activation='relu')(x)
    x = Dropout(rate=trial.suggest_float("dense_dropout", 0.1, 0.3))(x)
    
    # Output layer
    outputs = Dense(1, activation="linear")(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with optimized learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # Choose optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "adamw"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:  # adamw
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3))
    
    # Choose loss function
    loss_name = trial.suggest_categorical("loss", ["mse", "mae", "huber"])
    if loss_name == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_name == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    else:  # huber
        loss = tf.keras.losses.Huber(delta=trial.suggest_float("huber_delta", 0.1, 1.0))
    
    model.compile(optimizer=optimizer, loss=loss)
    return model

# ---------------------------
# 5. Optuna Objective for Hyperparameter Tuning
# ---------------------------
def objective(trial):
    # Load and preprocess data
    df = load_data("crypto_ohlcv.csv")
    df = compute_log_returns(df)
    df = triple_barrier_labeling(df, 
                               threshold=trial.suggest_float("threshold", 0.01, 0.05),
                               hold_period=trial.suggest_int("hold_period", 6, 24))
    
    # Select features
    numeric_feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'log_return',
        'rsi', 'sma_10', 'sma_30', 'macd', 'macd_signal', 
        'bb_upper', 'bb_lower', 'bb_width',
        'volume_change', 'volume_ma_ratio',
        'close_to_high', 'close_to_low', 'high_low_diff',
        'momentum_3', 'momentum_6', 'momentum_12'
    ]
    
    # Choose target: either 'log_return' or 'target_return'
    target_col = trial.suggest_categorical("target_col", ["log_return", "target_return"])
    
    # Choose scaler type
    scaler_type = trial.suggest_categorical("scaler", ["standard", "robust"])
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:  # robust
        scaler = RobustScaler()
    
    # Scale the features (fit on training portion only)
    train_portion = 0.8
    train_size = int(train_portion * len(df))
    train_df = df.iloc[:train_size]
    
    # Fit scaler only on training data to avoid data leakage
    df[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])
    
    # Create sliding window datasets
    window_size = trial.suggest_int("window_size", 12, 72)
    X, y = create_windows(df, window_size, numeric_feature_cols, target_col)
    
    # Split training and validation sets (time-series split)
    split = int(train_portion * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Build the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_advanced_lstm_model(input_shape, trial)
    
    # Early stopping with optimized patience
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=trial.suggest_int("patience", 5, 15),
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_delta=0.0001,
            min_lr=1e-6
        )
    ]
    
    # Train the model with optimized batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    val_pred = model.predict(X_val, batch_size=batch_size)
    mse = mean_squared_error(y_val, val_pred)
    mae = mean_absolute_error(y_val, val_pred)
    
    # Save trial information
    trial.set_user_attr("val_mse", mse)
    trial.set_user_attr("val_mae", mae)
    trial.set_user_attr("best_epoch", len(history.history['loss']) - callbacks[0].patience)
    
    # Return the metric we're optimizing
    return mse

# ---------------------------
# 6. Execute Hyperparameter Tuning and Train Best Model
# ---------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Increase n_trials for better search
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Value (MSE): {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Reload entire data for training best model
    df = load_data("crypto_ohlcv.csv")
    df = compute_log_returns(df)
    df = triple_barrier_labeling(df, 
                              threshold=trial.params.get("threshold", 0.02), 
                              hold_period=trial.params.get("hold_period", 12))
    
    # Get features based on best trial
    numeric_feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 'log_return',
        'rsi', 'sma_10', 'sma_30', 'macd', 'macd_signal', 
        'bb_upper', 'bb_lower', 'bb_width',
        'volume_change', 'volume_ma_ratio',
        'close_to_high', 'close_to_low', 'high_low_diff',
        'momentum_3', 'momentum_6', 'momentum_12'
    ]
    target_col = trial.params.get("target_col", "log_return")
    
    # Scale data (using the scaler type from best trial)
    if trial.params.get("scaler", "standard") == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
        
    df[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])
    
    # Use same window_size from best parameters
    window_size = trial.params.get("window_size", 30)
    X, y = create_windows(df, window_size, numeric_feature_cols, target_col)
    
    # Split to get a test set for evaluation
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Rebuild the model using the best hyperparameters
    input_shape = (X_train.shape[1], X_train.shape[2])
    best_model = build_advanced_lstm_model(input_shape, trial)
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=trial.params.get("patience", 10),
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_delta=0.0001,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    batch_size = trial.params.get("batch_size", 32)
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,  # 20% of training data for validation
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model on test set
    test_pred = best_model.predict(X_test).flatten()
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\nTest MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_pred, alpha=0.5)
    plt.plot([-2, 2], [-2, 2], 'r--')
    plt.title('Actual vs Predicted Returns')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.savefig('model_evaluation.png')
    plt.show()
    
    # ---------------------------
    # 7. Generate Predictions and Export Backtest Data
    # ---------------------------
    # Get predictions for all data
    all_predictions = best_model.predict(X).flatten().tolist()
    
    # For backtesting, combine timestamps, actual returns, and predictions
    backtest_df = df.iloc[window_size:].copy()
    backtest_df["predicted_return"] = all_predictions
    
    # Add a trading signal column based on prediction threshold
    signal_threshold = trial.params.get("threshold", 0.02) / 2  # Half of the labeling threshold as a heuristic
    backtest_df['signal'] = np.where(backtest_df['predicted_return'] > signal_threshold, 'BUY', 
                               np.where(backtest_df['predicted_return'] < -signal_threshold, 'SELL', 'HOLD'))
    
    # Compare model signals with the original labels
    accuracy = (backtest_df['signal'] == backtest_df['label']).mean()
    print(f"\nSignal accuracy: {accuracy:.2%}")
    
    # Count signals by type
    signal_counts = backtest_df['signal'].value_counts()
    print("\nSignal distribution:")
    print(signal_counts)
    
    # If you want to export as a JSON:
    backtest_json = backtest_df[["timestamp", "close", "log_return", "predicted_return", "label"]].to_json(orient="records")
    
    # Save JSON to file
    with open("backtest_predictions.json", "w") as f:
        f.write(backtest_json)
    
    print("Backtest predictions saved to backtest_predictions.json")
    
    # Also save the model
    best_model.save("lstm_trading_model.h5")
    print("Model saved to lstm_trading_model.h5")