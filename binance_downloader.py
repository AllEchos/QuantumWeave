import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import concurrent.futures
from tqdm import tqdm
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import configparser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binance_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_binance_data(symbol, interval, start_date, end_date=None, config_file=None, retry_count=3, retry_delay=10):
    """
    Download OHLCV data from Binance API with improved error handling.

    Parameters:
    - symbol: Trading pair (e.g., 'BTCUSDT')
    - interval: Timeframe (e.g., '1h', '4h', '1d')
    - start_date: Start date as string 'YYYY-MM-DD'
    - end_date: End date as string 'YYYY-MM-DD' (default: current time)
    - config_file: Path to config file with API keys (optional)
    - retry_count: Number of times to retry on failure
    - retry_delay: Seconds to wait between retries

    Returns:
    - DataFrame with OHLCV data
    """
    # Initialize Binance client
    if config_file and os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        api_key = config['BINANCE']['API_KEY']
        api_secret = config['BINANCE']['API_SECRET']
        client = Client(api_key, api_secret)
    else:
        # Initialize with empty strings for API keys - works for public data
        client = Client("", "")

    # Convert date strings to milliseconds timestamp
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)

    # Map interval string to milliseconds for API
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }

    # Download data in chunks to avoid API limitations
    klines = []
    
    # Optimize chunk size based on interval
    if interval in ['1m', '3m', '5m']:
        days_per_chunk = 7  # Smaller chunks for small intervals
    elif interval in ['15m', '30m', '1h']:
        days_per_chunk = 30
    else:
        days_per_chunk = 90  # Larger chunks for larger intervals
    
    chunk_size = timedelta(days=days_per_chunk).total_seconds() * 1000  # Convert to milliseconds

    current_start = start_ts
    total_chunks = int((end_ts - start_ts) / chunk_size) + 1
    
    with tqdm(total=total_chunks, desc=f"Downloading {symbol} {interval}") as pbar:
        while current_start < end_ts:
            current_end = min(current_start + chunk_size, end_ts)
            
            # Retry mechanism
            for attempt in range(retry_count):
                try:
                    # Convert timestamps to strings for API call
                    chunk = client.get_historical_klines(
                        symbol=symbol,
                        interval=interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR),
                        start_str=str(current_start),
                        end_str=str(current_end)
                    )
                    klines.extend(chunk)
                    
                    # Move to next chunk
                    current_start = current_end
                    pbar.update(1)
                    
                    # Avoid hitting API rate limits
                    time.sleep(1)
                    break
                    
                except BinanceAPIException as e:
                    logger.warning(f"API Error for {symbol} ({interval}): {e}")
                    if attempt < retry_count - 1:
                        logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{retry_count})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download chunk after {retry_count} attempts")
                        raise
                
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    if attempt < retry_count - 1:
                        logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{retry_count})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download chunk after {retry_count} attempts")
                        raise

    # Check if we got any data
    if not klines:
        logger.warning(f"No data returned for {symbol} {interval}")
        return pd.DataFrame()
        
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

    # Keep only the columns needed for the LSTM model
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df

def download_multiple_symbols(symbols, interval, start_date, end_date=None, config_file=None, max_workers=4):
    """
    Download data for multiple trading pairs in parallel.
    
    Parameters:
    - symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
    - interval: Timeframe (e.g., '1h', '4h', '1d')
    - start_date: Start date as string 'YYYY-MM-DD'
    - end_date: End date as string 'YYYY-MM-DD' (default: current time)
    - config_file: Path to config file with API keys (optional)
    - max_workers: Maximum number of concurrent downloads
    
    Returns:
    - Dictionary of DataFrames with symbol as key
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                download_binance_data, 
                symbol, 
                interval, 
                start_date, 
                end_date, 
                config_file
            ): symbol for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    logger.info(f"Successfully downloaded {symbol} data: {len(df)} records")
                else:
                    logger.warning(f"Empty dataset for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
    
    return results

def save_data(data_dict, base_dir="data"):
    """
    Save downloaded data to CSV files.
    
    Parameters:
    - data_dict: Dictionary of DataFrames with symbol as key
    - base_dir: Base directory to save files
    """
    os.makedirs(base_dir, exist_ok=True)
    
    for symbol, df in data_dict.items():
        # Create subdirectory for each symbol
        symbol_dir = os.path.join(base_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join(symbol_dir, f"{symbol}_ohlcv.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {symbol} data to {output_file}")

def get_all_usdt_pairs(client):
    """Get all available USDT trading pairs from Binance."""
    exchange_info = client.get_exchange_info()
    usdt_pairs = []
    
    for symbol_info in exchange_info['symbols']:
        symbol = symbol_info['symbol']
        if symbol.endswith('USDT') and symbol_info['status'] == 'TRADING':
            usdt_pairs.append(symbol)
    
    return usdt_pairs

if __name__ == "__main__":
    # Define parameters
    interval = '1h'  # Timeframe: 1h for hourly data
    start_date = '2021-01-01'  # Start date
    end_date = None  # End date (None = current time)
    
    # Optional config file path for API keys
    # config_file = 'config.ini'  # Uncomment to use API keys
    config_file = None
    
    # Initialize Binance client for getting symbol list
    client = Client("", "")
    
    # Choose which symbols to download
    # Option 1: Manually specify symbols
    target_symbols = [
        'BTCUSDT',  # Bitcoin
        'ETHUSDT',  # Ethereum
        'BNBUSDT',  # Binance Coin
        'ADAUSDT',  # Cardano
        'SOLUSDT',  # Solana
        'XRPUSDT',  # Ripple
        'DOTUSDT',  # Polkadot
        'DOGEUSDT', # Dogecoin
        'MATICUSDT', # Polygon
        'AVAXUSDT'  # Avalanche
    ]
    
    # Option 2: Get all USDT pairs (uncomment to use)
    # target_symbols = get_all_usdt_pairs(client)
    # logger.info(f"Found {len(target_symbols)} USDT trading pairs")
    
    logger.info(f"Starting download for {len(target_symbols)} symbols with {interval} interval from {start_date}")
    
    # Download data
    data_dict = download_multiple_symbols(
        target_symbols,
        interval,
        start_date,
        end_date,
        config_file,
        max_workers=4  # Adjust based on your system and API rate limits
    )
    
    # Save all data
    save_data(data_dict)
    
    # Also save the combined data for the LSTM model
    main_symbol = 'BTCUSDT'  # Use BTC as the main symbol
    if main_symbol in data_dict:
        output_file = "crypto_ohlcv.csv"
        data_dict[main_symbol].to_csv(output_file, index=False)
        logger.info(f"Main data saved to {output_file}")
        logger.info(f"Downloaded {len(data_dict[main_symbol])} records for {main_symbol}")
    
    # Save metadata
    metadata = {
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date if end_date else "current",
        "symbols": list(data_dict.keys()),
        "record_counts": {symbol: len(df) for symbol, df in data_dict.items()}
    }
    
    with open(os.path.join("data", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("Download completed successfully")
