"""
Real-time Data Collection Module
Handles stock price data fetching and preprocessing
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time

class StockDataCollector:
    """
    Collects real-time and historical stock data
    """
    
    def __init__(self, symbol='BTC-USD', interval='1h'):
        self.symbol = symbol
        self.interval = interval
        
    def fetch_realtime_data(self, period='7d'):
        """
        Fetch real-time stock/crypto data using yfinance
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 1y, 5y)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=self.interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Reset index to make datetime a column
            df = df.reset_index()
            
            # Rename columns for consistency
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_historical_data(self, start_date, end_date):
        """
        Fetch historical data for training
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval=self.interval)
            
            if df.empty:
                raise ValueError(f"No historical data found for {self.symbol}")
            
            df = df.reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators
        """
        if df is None or df.empty:
            return None
        
        try:
            # Calculate technical indicators
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_25'] = df['close'].rolling(window=25).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(periods=5)
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return df
    
    def prepare_features(self, df):
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame ready for model training/prediction
        """
        if df is None or df.empty:
            return None
        
        # Add target variable (next period's price movement)
        df['price_next'] = df['close'].shift(-1)
        df['target'] = (df['price_next'] > df['close']).astype(int)
        
        # Add time-based features
        if 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        elif 'date' in df.columns:
            df['hour'] = pd.to_datetime(df['date']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def get_latest_price(self):
        """
        Get the latest price for the symbol
        
        Returns:
            dict with latest price info
        """
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'symbol': self.symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching latest price: {e}")
            return None


# Sample usage and testing
if __name__ == "__main__":
    print("Testing Stock Data Collector...")
    
    # Initialize collector
    collector = StockDataCollector(symbol='BTC-USD', interval='1h')
    
    # Fetch real-time data
    print("\nFetching real-time data...")
    df = collector.fetch_realtime_data(period='30d')
    print(f"Fetched {len(df)} rows")
    print(df.head())
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    df = collector.add_technical_indicators(df)
    print(df.columns.tolist())
    
    # Prepare features
    print("\nPreparing features...")
    df = collector.prepare_features(df)
    print(f"Final dataset shape: {df.shape}")
    
    # Get latest price
    print("\nFetching latest price...")
    latest = collector.get_latest_price()
    print(latest)
