"""
Model Training Script
Trains and saves ML models for stock price prediction
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data_collector import StockDataCollector
from ml_models import StockPricePredictor, ModelComparison
import warnings
warnings.filterwarnings('ignore')


def train_stock_prediction_models(symbol='BTC-USD', interval='1h', period='3mo'):
    """
    Complete training pipeline for stock prediction models
    
    Args:
        symbol: Stock/crypto symbol (e.g., 'BTC-USD', 'AAPL', 'TSLA')
        interval: Data interval (1h, 1d)
        period: Historical period (1mo, 3mo, 6mo, 1y)
    
    Returns:
        Trained model and performance metrics
    """
    
    print("="*80)
    print(f"STOCK PRICE PREDICTION MODEL TRAINING")
    print(f"Symbol: {symbol} | Interval: {interval} | Period: {period}")
    print("="*80)
    
    # Step 1: Data Collection
    print("\n[1/5] Collecting Data...")
    collector = StockDataCollector(symbol=symbol, interval=interval)
    
    df = collector.fetch_realtime_data(period=period)
    
    if df is None or df.empty:
        print("Error: No data collected. Exiting.")
        return None
    
    print(f"✓ Collected {len(df)} data points")
    print(f"  Date range: {df.iloc[0]['datetime']} to {df.iloc[-1]['datetime']}")
    
    # Step 2: Feature Engineering
    print("\n[2/5] Engineering Features...")
    df = collector.add_technical_indicators(df)
    df = collector.prepare_features(df)
    
    print(f"✓ Created {len(df.columns)} features")
    print(f"  Features: {', '.join(df.columns[:10].tolist())}...")
    
    # Step 3: Define Features and Target
    print("\n[3/5] Preparing Training Data...")
    
    # Select feature columns (exclude non-feature columns)
    exclude_cols = ['datetime', 'date', 'target', 'price_next', 'dividends', 
                    'stock_splits', 'capital_gains']
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and not df[col].isna().all()]
    
    print(f"✓ Selected {len(feature_cols)} features for training")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Check for class imbalance
    class_ratio = df['target'].value_counts().min() / df['target'].value_counts().max()
    print(f"  Class balance ratio: {class_ratio:.2f}")
    
    # Step 4: Train and Compare Models
    print("\n[4/5] Training Models...")
    
    comparison = ModelComparison()
    comparison_results = comparison.compare_models(
        df, 
        feature_cols, 
        target_col='target',
        model_types=['random_forest', 'gradient_boosting', 'xgboost']
    )
    
    # Select best model
    best_model_name = comparison_results.iloc[0]['model']
    best_model = comparison.results[best_model_name]
    
    print(f"\n✓ Best model: {best_model_name}")
    print(f"  Accuracy: {comparison_results.iloc[0]['test_accuracy']:.4f}")
    
    # Step 5: Save Models
    print("\n[5/5] Saving Models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save best model
    model_filename = f"models/best_model_{symbol.replace('-', '_')}_{interval}.pkl"
    best_model.save_model(model_filename)
    
    # Save comparison results
    comparison_filename = f"models/comparison_{symbol.replace('-', '_')}_{interval}.csv"
    comparison_results.to_csv(comparison_filename, index=False)
    print(f"✓ Comparison results saved to {comparison_filename}")
    
    # Save feature list
    feature_filename = f"models/features_{symbol.replace('-', '_')}_{interval}.txt"
    with open(feature_filename, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"✓ Feature list saved to {feature_filename}")
    
    # Create model metadata
    metadata = {
        'symbol': symbol,
        'interval': interval,
        'period': period,
        'best_model': best_model_name,
        'accuracy': float(comparison_results.iloc[0]['test_accuracy']),
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'training_date': datetime.now().isoformat(),
        'model_file': model_filename
    }
    
    metadata_filename = f"models/metadata_{symbol.replace('-', '_')}_{interval}.json"
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata saved to {metadata_filename}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {metadata['accuracy']:.4f}")
    print(f"Model saved to: {model_filename}")
    print("="*80)
    
    return best_model, metadata


def train_multiple_symbols(symbols, interval='1h', period='3mo'):
    """
    Train models for multiple stock symbols
    
    Args:
        symbols: List of stock/crypto symbols
        interval: Data interval
        period: Historical period
    """
    results = {}
    
    for symbol in symbols:
        print(f"\n\nTraining model for {symbol}...")
        try:
            model, metadata = train_stock_prediction_models(symbol, interval, period)
            results[symbol] = {
                'success': True,
                'model': model,
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error training {symbol}: {e}")
            results[symbol] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for symbol, result in results.items():
        if result['success']:
            print(f"✓ {symbol}: Accuracy {result['metadata']['accuracy']:.4f}")
        else:
            print(f"✗ {symbol}: Failed - {result['error']}")
    
    return results


if __name__ == "__main__":
    """
    Main execution
    Usage: python train_models.py [symbol] [interval] [period]
    Example: python train_models.py BTC-USD 1h 3mo
    """
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = 'BTC-USD'
    
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    else:
        interval = '1h'
    
    if len(sys.argv) > 3:
        period = sys.argv[3]
    else:
        period = '3mo'
    
    # Train model
    try:
        model, metadata = train_stock_prediction_models(symbol, interval, period)
        
        # Optional: Train for multiple symbols
        # symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'GOOGL']
        # train_multiple_symbols(symbols, interval='1d', period='1y')
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
