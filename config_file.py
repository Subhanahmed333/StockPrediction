"""
Configuration File
Central configuration for the Stock Price & Sentiment Predictor
"""

import os
from datetime import datetime

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_NAME = "Real-Time Stock Price & Sentiment Predictor"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Your Name"
APP_EMAIL = "your.email@example.com"

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Default symbols for quick access
DEFAULT_SYMBOLS = {
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD'],
    'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
    'indices': ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
}

# Data collection settings
DATA_CONFIG = {
    'default_symbol': 'BTC-USD',
    'default_interval': '1h',
    'default_period': '3mo',
    'supported_intervals': ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'],
    'supported_periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    'min_data_points': 200,  # Minimum required for training
    'max_api_retries': 3,
    'api_timeout': 30  # seconds
}

# ============================================================================
# MACHINE LEARNING SETTINGS
# ============================================================================

# Model training configuration
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1,  # Use all CPU cores
    'min_accuracy_threshold': 0.65,  # Minimum acceptable accuracy
    'hyperparameter_tuning': False,  # Enable for better accuracy (slower)
}

# Available model types
MODEL_TYPES = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'logistic_regression': {
        'max_iter': 1000
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    'technical_indicators': {
        'sma_periods': [7, 25, 50],
        'ema_periods': [12, 26],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'volume_sma': 20
    },
    'exclude_features': ['datetime', 'date', 'target', 'price_next', 
                         'dividends', 'stock_splits', 'capital_gains']
}

# ============================================================================
# SENTIMENT ANALYSIS SETTINGS
# ============================================================================

SENTIMENT_CONFIG = {
    'methods': ['textblob', 'vader', 'keywords'],
    'weights': {
        'textblob': 0.3,
        'vader': 0.4,
        'keywords': 0.3
    },
    'thresholds': {
        'positive': 0.1,
        'negative': -0.1
    },
    'financial_keywords': {
        'positive': [
            'bullish', 'rally', 'surge', 'gain', 'profit', 'growth',
            'breakthrough', 'moon', 'pump', 'buy', 'long', 'strong',
            'outperform', 'uptrend', 'breakout', 'support'
        ],
        'negative': [
            'bearish', 'crash', 'dump', 'loss', 'decline', 'fall',
            'sell', 'short', 'weak', 'fear', 'panic', 'collapse',
            'underperform', 'downtrend', 'breakdown', 'resistance'
        ]
    }
}

# ============================================================================
# DIRECTORY SETTINGS
# ============================================================================

# Create these directories if they don't exist
DIRECTORIES = {
    'models': 'models',
    'logs': 'logs',
    'data': 'data',
    'reports': 'reports',
    'cache': 'cache'
}

# File naming patterns
FILE_PATTERNS = {
    'model': 'best_model_{symbol}_{interval}.pkl',
    'metadata': 'metadata_{symbol}_{interval}.json',
    'features': 'features_{symbol}_{interval}.txt',
    'comparison': 'comparison_{symbol}_{interval}.csv',
    'logs': 'interaction_logs_{date}.csv',
    'predictions': 'predictions_{date}.csv'
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING_CONFIG = {
    'enabled': True,
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_file': os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d")}.log'),
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_interactions': True,
    'log_predictions': True,
    'log_errors': True
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

CHART_CONFIG = {
    'theme': 'plotly_white',
    'color_scheme': {
        'up': '#26a69a',
        'down': '#ef5350',
        'neutral': '#ffa726',
        'primary': '#1f77b4',
        'secondary': '#ff7f0e'
    },
    'chart_height': {
        'main': 500,
        'secondary': 300,
        'small': 200
    }
}

# ============================================================================
# API SETTINGS (Optional)
# ============================================================================

API_CONFIG = {
    'binance': {
        'enabled': False,
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', '')
    },
    'twitter': {
        'enabled': False,
        'api_key': os.getenv('TWITTER_API_KEY', ''),
        'api_secret': os.getenv('TWITTER_API_SECRET', ''),
        'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
        'access_secret': os.getenv('TWITTER_ACCESS_SECRET', '')
    }
}

# ============================================================================
# DEPLOYMENT SETTINGS
# ============================================================================

DEPLOYMENT_CONFIG = {
    'environment': os.getenv('ENVIRONMENT', 'development'),  # development, staging, production
    'debug': os.getenv('DEBUG', 'True').lower() == 'true',
    'port': int(os.getenv('PORT', 8501)),
    'host': os.getenv('HOST', 'localhost'),
    'enable_caching': True,
    'cache_ttl': 300,  # seconds
    'max_upload_size': 200,  # MB
    'session_timeout': 3600  # seconds
}

# ============================================================================
# ALERT SETTINGS (Future Enhancement)
# ============================================================================

ALERT_CONFIG = {
    'enabled': False,
    'email_alerts': False,
    'email_recipients': [],
    'sms_alerts': False,
    'alert_conditions': {
        'price_change_threshold': 5.0,  # percentage
        'confidence_threshold': 0.8,
        'sentiment_extreme': 0.5
    }
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

PERFORMANCE_CONFIG = {
    'enable_profiling': False,
    'cache_predictions': True,
    'parallel_processing': True,
    'batch_size': 1000,
    'memory_limit': '2GB'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(symbol, interval):
    """Get the path for a model file"""
    filename = FILE_PATTERNS['model'].format(
        symbol=symbol.replace('-', '_'),
        interval=interval
    )
    return os.path.join(DIRECTORIES['models'], filename)


def get_metadata_path(symbol, interval):
    """Get the path for a metadata file"""
    filename = FILE_PATTERNS['metadata'].format(
        symbol=symbol.replace('-', '_'),
        interval=interval
    )
    return os.path.join(DIRECTORIES['models'], filename)


def create_directories():
    """Create required directories if they don't exist"""
    for dir_name, dir_path in DIRECTORIES.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ensured: {dir_path}")


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check critical settings
    if DATA_CONFIG['min_data_points'] < 100:
        errors.append("min_data_points should be at least 100")
    
    if ML_CONFIG['test_size'] < 0.1 or ML_CONFIG['test_size'] > 0.5:
        errors.append("test_size should be between 0.1 and 0.5")
    
    # Check sentiment weights sum to 1
    weight_sum = sum(SENTIMENT_CONFIG['weights'].values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Sentiment weights sum to {weight_sum}, should be 1.0")
    
    if errors:
        print("⚠️  Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Configuration validated successfully")
    return True


def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Application: {APP_NAME} v{APP_VERSION}")
    print(f"Environment: {DEPLOYMENT_CONFIG['environment']}")
    print(f"Debug Mode: {DEPLOYMENT_CONFIG['debug']}")
    print(f"\nDefault Settings:")
    print(f"  Symbol: {DATA_CONFIG['default_symbol']}")
    print(f"  Interval: {DATA_CONFIG['default_interval']}")
    print(f"  Period: {DATA_CONFIG['default_period']}")
    print(f"\nML Configuration:")
    print(f"  Test Size: {ML_CONFIG['test_size']}")
    print(f"  CV Folds: {ML_CONFIG['cv_folds']}")
    print(f"  Min Accuracy: {ML_CONFIG['min_accuracy_threshold']}")
    print(f"\nDirectories:")
    for name, path in DIRECTORIES.items():
        print(f"  {name}: {path}")
    print("="*80 + "\n")


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize the application"""
    print("Initializing application...")
    
    # Create directories
    create_directories()
    
    # Validate configuration
    validate_config()
    
    # Print summary
    if DEPLOYMENT_CONFIG['debug']:
        print_config_summary()
    
    print("✓ Application initialized successfully\n")


# Run initialization when imported
if __name__ == "__main__":
    initialize()
else:
    # Auto-create directories when config is imported
    for dir_path in DIRECTORIES.values():
        os.makedirs(dir_path, exist_ok=True)
