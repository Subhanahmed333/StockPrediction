"""
Quick Demo Script
Tests all components of the stock predictor system
"""

import warnings
warnings.filterwarnings('ignore')

def demo_data_collection():
    """Demo data collection functionality"""
    print("\n" + "="*80)
    print("DEMO 1: DATA COLLECTION")
    print("="*80)
    
    from data_collector import StockDataCollector
    
    print("\n[1] Fetching Bitcoin data...")
    collector = StockDataCollector(symbol='BTC-USD', interval='1h')
    
    df = collector.fetch_realtime_data(period='7d')
    if df is not None:
        print(f"✓ Successfully fetched {len(df)} data points")
        print(f"\nSample data:")
        print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())
    
    print("\n[2] Adding technical indicators...")
    df = collector.add_technical_indicators(df)
    print(f"✓ Added indicators: {', '.join(df.columns[6:12].tolist())}")
    
    print("\n[3] Getting latest price...")
    latest = collector.get_latest_price()
    if latest:
        print(f"✓ Current BTC Price: ${latest['price']:,.2f}")
        print(f"  24h Change: {latest['change_percent']:.2f}%")


def demo_sentiment_analysis():
    """Demo sentiment analysis functionality"""
    print("\n" + "="*80)
    print("DEMO 2: SENTIMENT ANALYSIS")
    print("="*80)
    
    from sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    
    # Test samples
    samples = {
        "Bullish": "Bitcoin surges past $50,000! Strong buying momentum continues. 🚀",
        "Bearish": "Market crash warning! Major sell-off expected. Get out now!",
        "Neutral": "Bitcoin trading sideways around support level. Waiting for direction."
    }
    
    print("\nAnalyzing different sentiment types...\n")
    
    for label, text in samples.items():
        result = analyzer.analyze_comprehensive(text)
        
        print(f"[{label} Example]")
        print(f"Text: {text}")
        print(f"Detected Sentiment: {result['sentiment'].upper()}")
        print(f"Score: {result['score']:.3f}")
        print(f"Confidence: {result['confidence']:.2%}")
        print()


def demo_ml_prediction():
    """Demo ML prediction functionality"""
    print("\n" + "="*80)
    print("DEMO 3: MACHINE LEARNING PREDICTION")
    print("="*80)
    
    import os
    from data_collector import StockDataCollector
    from ml_models import StockPricePredictor
    
    symbol = 'BTC-USD'
    interval = '1h'
    model_file = f"models/best_model_{symbol.replace('-', '_')}_{interval}.pkl"
    
    # Check if model exists
    if not os.path.exists(model_file):
        print(f"\n⚠️  No trained model found for {symbol}")
        print("Training a new model (this may take a few minutes)...\n")
        
        try:
            from train_models import train_stock_prediction_models
            model, metadata = train_stock_prediction_models(symbol, interval, '1mo')
            print("\n✓ Model trained successfully!")
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            print("Skipping prediction demo...")
            return
    
    # Load model
    print(f"\n[1] Loading trained model...")
    predictor = StockPricePredictor()
    predictor.load_model(model_file)
    print("✓ Model loaded successfully")
    
    # Fetch latest data
    print("\n[2] Fetching latest data...")
    collector = StockDataCollector(symbol, interval)
    df = collector.fetch_realtime_data(period='30d')
    df = collector.add_technical_indicators(df)
    df = collector.prepare_features(df)
    print(f"✓ Data prepared: {len(df)} samples")
    
    # Make prediction
    print("\n[3] Making prediction...")
    latest_features = df[predictor.feature_names].iloc[-1:]
    prediction = predictor.predict(latest_features)
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Symbol: {symbol}")
    print(f"Prediction: {prediction['prediction']} {'📈' if prediction['prediction'] == 'UP' else '📉'}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Probability UP: {prediction['probability_up']:.2%}")
    print(f"Probability DOWN: {prediction['probability_down']:.2%}")
    print(f"Timestamp: {prediction['timestamp']}")
    print("-"*60)


def demo_model_comparison():
    """Demo model comparison functionality"""
    print("\n" + "="*80)
    print("DEMO 4: MODEL COMPARISON")
    print("="*80)
    
    from data_collector import StockDataCollector
    from ml_models import ModelComparison
    
    print("\n[1] Collecting training data...")
    collector = StockDataCollector('BTC-USD', '1h')
    df = collector.fetch_realtime_data(period='1mo')
    df = collector.add_technical_indicators(df)
    df = collector.prepare_features(df)
    
    print(f"✓ Dataset prepared: {len(df)} samples\n")
    
    # Define features
    exclude_cols = ['datetime', 'date', 'target', 'price_next', 'dividends', 
                    'stock_splits', 'capital_gains']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and not df[col].isna().all()]
    
    print("[2] Training and comparing models...")
    print("(This will take 2-3 minutes)\n")
    
    comparison = ModelComparison()
    results = comparison.compare_models(
        df, 
        feature_cols,
        model_types=['random_forest', 'gradient_boosting', 'logistic_regression']
    )
    
    print("\n✓ Model comparison complete!")


def demo_full_pipeline():
    """Demo complete pipeline"""
    print("\n" + "="*80)
    print("DEMO 5: COMPLETE PIPELINE")
    print("="*80)
    
    from data_collector import StockDataCollector
    from sentiment_analyzer import SentimentAnalyzer
    from ml_models import StockPricePredictor
    import os
    
    symbol = 'BTC-USD'
    
    print("\n[1] Data Collection...")
    collector = StockDataCollector(symbol, '1h')
    df = collector.fetch_realtime_data('7d')
    latest_price = collector.get_latest_price()
    print(f"✓ Current {symbol}: ${latest_price['price']:,.2f}")
    
    print("\n[2] Sentiment Analysis...")
    analyzer = SentimentAnalyzer()
    news_text = "Bitcoin shows strong bullish momentum with institutional adoption rising"
    sentiment = analyzer.analyze_comprehensive(news_text)
    print(f"✓ Market Sentiment: {sentiment['sentiment'].upper()} ({sentiment['score']:.3f})")
    
    print("\n[3] Price Prediction...")
    model_file = f"models/best_model_BTC_USD_1h.pkl"
    
    if os.path.exists(model_file):
        predictor = StockPricePredictor()
        predictor.load_model(model_file)
        
        df = collector.add_technical_indicators(df)
        df = collector.prepare_features(df)
        latest_features = df[predictor.feature_names].iloc[-1:]
        prediction = predictor.predict(latest_features)
        
        print(f"✓ Price Prediction: {prediction['prediction']} ({prediction['confidence']:.2%})")
    else:
        print("⚠️  Model not found. Run training first.")
    
    print("\n[4] Combined Analysis...")
    print("-"*60)
    print("MARKET ANALYSIS SUMMARY")
    print("-"*60)
    print(f"Asset: {symbol}")
    print(f"Current Price: ${latest_price['price']:,.2f}")
    print(f"24h Change: {latest_price['change_percent']:.2f}%")
    print(f"Market Sentiment: {sentiment['sentiment'].upper()}")
    if os.path.exists(model_file):
        print(f"ML Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
    print("-"*60)


def main():
    """Run all demos"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║    Real-Time Stock Price & Sentiment Predictor - DEMO         ║
    ║                                                                ║
    ║    This demo showcases all components of the system           ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect demo to run:")
    print("1. Data Collection Demo")
    print("2. Sentiment Analysis Demo")
    print("3. ML Prediction Demo")
    print("4. Model Comparison Demo")
    print("5. Complete Pipeline Demo")
    print("6. Run All Demos")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    demos = {
        '1': demo_data_collection,
        '2': demo_sentiment_analysis,
        '3': demo_ml_prediction,
        '4': demo_model_comparison,
        '5': demo_full_pipeline
    }
    
    if choice == '6':
        # Run all demos
        for demo_func in demos.values():
            try:
                demo_func()
            except Exception as e:
                print(f"\n✗ Demo failed: {e}")
            input("\nPress Enter to continue to next demo...")
    elif choice in demos:
        try:
            demos[choice]()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '0':
        print("\nGoodbye!")
        return
    else:
        print("\n✗ Invalid choice")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo run the full application, use: streamlit run app.py")
    print("To train models, use: python train_models.py [SYMBOL] [INTERVAL] [PERIOD]")
    print("\nThank you for trying the Real-Time Stock Price & Sentiment Predictor!")


if __name__ == "__main__":
    main()
