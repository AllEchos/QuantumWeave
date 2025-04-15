import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_predictions(filepath):
    """Load and parse the predictions JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_predictions(df):
    """Analyze the predictions and calculate key metrics"""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate prediction accuracy
    correct_direction = np.sign(df['log_return']) == np.sign(df['predicted_return'])
    direction_accuracy = correct_direction.mean()
    
    # Calculate signal distribution
    signal_counts = df['label'].value_counts()
    
    # Calculate return statistics
    actual_returns = df['log_return'].describe()
    predicted_returns = df['predicted_return'].describe()
    
    # Calculate correlation
    correlation = df['log_return'].corr(df['predicted_return'])
    
    return {
        'direction_accuracy': direction_accuracy,
        'signal_distribution': signal_counts,
        'actual_returns': actual_returns,
        'predicted_returns': predicted_returns,
        'correlation': correlation
    }

def main():
    # Load predictions
    print("Loading predictions...")
    df = load_predictions('backtest_predictions.json')
    
    # Analyze
    print("\nAnalyzing predictions...")
    results = analyze_predictions(df)
    
    # Print results
    print("\n=== Prediction Analysis ===")
    print(f"\nDirection Accuracy: {results['direction_accuracy']:.2%}")
    print("\nSignal Distribution:")
    print(results['signal_distribution'])
    print("\nActual Returns Statistics:")
    print(results['actual_returns'])
    print("\nPredicted Returns Statistics:")
    print(results['predicted_returns'])
    print(f"\nCorrelation between actual and predicted returns: {results['correlation']:.4f}")
    
    # Print sample of recent predictions
    print("\n=== Recent Predictions Sample ===")
    print(df[['timestamp', 'close', 'log_return', 'predicted_return', 'label']].tail(10))

if __name__ == "__main__":
    main() 