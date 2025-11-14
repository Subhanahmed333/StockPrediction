"""
Real-Time Stock Price & Sentiment Predictor
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
from data_collector import StockDataCollector
from sentiment_analyzer import SentimentAnalyzer
from ml_models import StockPricePredictor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price & Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .up-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .down-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = []

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def log_interaction(action, details):
    """Log user interactions"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details
    }
    st.session_state.logs.append(log_entry)
    
    # Save to CSV
    df_logs = pd.DataFrame(st.session_state.logs)
    df_logs.to_csv('logs/interaction_logs.csv', index=False)


def load_model(symbol, interval='1h'):
    """Load trained model for symbol"""
    model_file = f"models/best_model_{symbol.replace('-', '_')}_{interval}.pkl"
    
    if os.path.exists(model_file):
        predictor = StockPricePredictor()
        predictor.load_model(model_file)
        return predictor
    return None


def plot_candlestick(df):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'] if 'datetime' in df.columns else df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])
    
    fig.update_layout(
        title='Price Chart',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_technical_indicators(df):
    """Plot technical indicators"""
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(
        x=df['datetime'] if 'datetime' in df.columns else df['date'],
        y=df['close'],
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # SMAs
    if 'sma_7' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'] if 'datetime' in df.columns else df['date'],
            y=df['sma_7'],
            name='SMA 7',
            line=dict(color='orange', dash='dash')
        ))
    
    if 'sma_25' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'] if 'datetime' in df.columns else df['date'],
            y=df['sma_25'],
            name='SMA 25',
            line=dict(color='green', dash='dash')
        ))
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'] if 'datetime' in df.columns else df['date'],
            y=df['bb_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dot'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=df['datetime'] if 'datetime' in df.columns else df['date'],
            y=df['bb_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dot'),
            fill='tonexty',
            opacity=0.3
        ))
    
    fig.update_layout(
        title='Technical Indicators',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_rsi(df):
    """Plot RSI indicator"""
    if 'rsi' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['datetime'] if 'datetime' in df.columns else df['date'],
        y=df['rsi'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        xaxis_title='Date',
        template='plotly_white',
        height=300
    )
    
    return fig


def main():
    """Main application"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Header
    st.markdown('<div class="main-header">📈 Real-Time Stock Price & Sentiment Predictor</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Symbol selection
    symbol_options = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
    symbol = st.sidebar.selectbox("Select Symbol", symbol_options)
    
    # Interval selection
    interval = st.sidebar.selectbox("Select Interval", ['1h', '1d'])
    
    # Period selection
    period = st.sidebar.selectbox("Select Period", ['7d', '1mo', '3mo', '6mo', '1y'])
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🤖 Prediction", 
        "💭 Sentiment Analysis",
        "📈 Analytics",
        "📋 Logs"
    ])
    
    # Initialize components
    collector = StockDataCollector(symbol=symbol, interval=interval)
    analyzer = SentimentAnalyzer()
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Market Overview")
        
        # Fetch data
        with st.spinner("Fetching market data..."):
            df = collector.fetch_realtime_data(period=period)
            latest_price = collector.get_latest_price()
        
        if df is not None and not df.empty:
            # Current price metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${latest_price['price']:,.2f}" if latest_price else "N/A",
                    delta=f"{latest_price['change_percent']:.2f}%" if latest_price else None
                )
            
            with col2:
                st.metric(
                    label="24h Change",
                    value=f"${latest_price['change']:,.2f}" if latest_price else "N/A"
                )
            
            with col3:
                st.metric(
                    label="Volume",
                    value=f"{latest_price['volume']:,.0f}" if latest_price else "N/A"
                )
            
            with col4:
                st.metric(
                    label="Market Cap",
                    value=f"${latest_price['market_cap']:,.0f}" if latest_price and latest_price['market_cap'] else "N/A"
                )
            
            # Charts
            st.plotly_chart(plot_candlestick(df), use_container_width=True)
            
            # Add technical indicators
            df = collector.add_technical_indicators(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plot_technical_indicators(df), use_container_width=True)
            
            with col2:
                rsi_fig = plot_rsi(df)
                if rsi_fig:
                    st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Recent data table
            st.subheader("Recent Data")
            st.dataframe(
                df[['datetime' if 'datetime' in df.columns else 'date', 
                    'open', 'high', 'low', 'close', 'volume']].tail(10),
                use_container_width=True
            )
            
            log_interaction("view_dashboard", {'symbol': symbol, 'period': period})
        
        else:
            st.error("Failed to fetch data. Please try again.")
    
    # Tab 2: Prediction
    with tab2:
        st.header("Price Movement Prediction")
        
        # Check if model exists
        model = load_model(symbol, interval)
        
        if model is None:
            st.warning(f"⚠️ No trained model found for {symbol}. Please train a model first.")
            
            if st.button("🎓 Train Model Now"):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        from train_models import train_stock_prediction_models
                        trained_model, metadata = train_stock_prediction_models(
                            symbol=symbol,
                            interval=interval,
                            period='3mo'
                        )
                        st.success(f"✅ Model trained successfully! Accuracy: {metadata['accuracy']:.4f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error training model: {e}")
        else:
            st.success("✅ Model loaded successfully!")
            
            # Fetch current data
            with st.spinner("Fetching latest data for prediction..."):
                df = collector.fetch_realtime_data(period='30d')
                df = collector.add_technical_indicators(df)
                df = collector.prepare_features(df)
            
            if df is not None and not df.empty:
                # Get latest features
                latest_features = df[model.feature_names].iloc[-1:].copy()
                
                # Make prediction
                prediction = model.predict(latest_features)
                
                # Display prediction
                pred_class = "up-prediction" if prediction['prediction'] == 'UP' else "down-prediction"
                
                st.markdown(f"""
                    <div class="prediction-box {pred_class}">
                        <h2>Prediction: {prediction['prediction']}</h2>
                        <h3>Confidence: {prediction['confidence']:.2%}</h3>
                        <p>Probability UP: {prediction['probability_up']:.2%}</p>
                        <p>Probability DOWN: {prediction['probability_down']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Store prediction
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence']
                })
                
                # Feature importance
                if hasattr(model.model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'feature': model.feature_names,
                        'importance': model.model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    
                    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                title='Top 10 Important Features')
                    st.plotly_chart(fig, use_container_width=True)
                
                log_interaction("make_prediction", {
                    'symbol': symbol,
                    'prediction': prediction['prediction'],
                    'confidence': float(prediction['confidence'])
                })
            else:
                st.error("Failed to fetch data for prediction.")
    
    # Tab 3: Sentiment Analysis
    with tab3:
        st.header("Sentiment Analysis")
        
        st.write("Analyze market sentiment from text (news, social media, etc.)")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze (news headline, tweet, etc.)",
            height=150,
            placeholder="Example: Bitcoin surges past $50,000 as institutional adoption grows..."
        )
        
        if st.button("Analyze Sentiment"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    result = analyzer.analyze_comprehensive(text_input)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_color = {
                        'positive': '🟢',
                        'negative': '🔴',
                        'neutral': '🟡'
                    }
                    st.metric(
                        "Sentiment",
                        f"{sentiment_color[result['sentiment']]} {result['sentiment'].upper()}"
                    )
                
                with col2:
                    st.metric("Score", f"{result['score']:.3f}")
                
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                # Detailed results
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['textblob']:
                        st.write("**TextBlob Analysis:**")
                        st.json(result['textblob'])
                
                with col2:
                    if result['vader']:
                        st.write("**VADER Analysis:**")
                        st.json(result['vader'])
                
                log_interaction("sentiment_analysis", {
                    'text_length': len(text_input),
                    'sentiment': result['sentiment'],
                    'score': float(result['score'])
                })
            else:
                st.warning("Please enter some text to analyze.")
        
        # Batch analysis
        st.subheader("Batch Sentiment Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV with text column", type=['csv'])
        
        if uploaded_file is not None:
            df_text = pd.read_csv(uploaded_file)
            st.write("Preview:", df_text.head())
            
            text_column = st.selectbox("Select text column", df_text.columns)
            
            if st.button("Analyze Batch"):
                with st.spinner("Analyzing batch..."):
                    texts = df_text[text_column].tolist()
                    summary = analyzer.get_sentiment_summary(texts)
                
                st.subheader("Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Positive", summary['positive'])
                with col2:
                    st.metric("Negative", summary['negative'])
                with col3:
                    st.metric("Neutral", summary['neutral'])
                
                st.metric("Overall Sentiment", summary['overall_sentiment'].upper())
                st.metric("Average Score", f"{summary['average_score']:.3f}")
    
    # Tab 4: Analytics
    with tab4:
        st.header("Analytics & Insights")
        
        if st.session_state.prediction_history:
            df_pred = pd.DataFrame(st.session_state.prediction_history)
            
            # Prediction distribution
            st.subheader("Prediction Distribution")
            pred_counts = df_pred['prediction'].value_counts()
            
            fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="UP vs DOWN Predictions"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence over time
            st.subheader("Confidence Over Time")
            fig = px.line(
                df_pred,
                x='timestamp',
                y='confidence',
                color='prediction',
                title="Prediction Confidence"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions made yet. Go to the Prediction tab to make predictions.")
    
    # Tab 5: Logs
    with tab5:
        st.header("System Logs")
        
        if st.session_state.logs:
            df_logs = pd.DataFrame(st.session_state.logs)
            
            st.subheader(f"Total Interactions: {len(df_logs)}")
            
            # Action distribution
            action_counts = df_logs['action'].value_counts()
            
            fig = px.bar(
                x=action_counts.index,
                y=action_counts.values,
                title="Actions Distribution",
                labels={'x': 'Action', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent logs
            st.subheader("Recent Activity")
            st.dataframe(df_logs.tail(20), use_container_width=True)
            
            # Download logs
            if st.button("Download Logs"):
                csv = df_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No logs available yet.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
        **About**
        
        This application provides real-time stock price prediction
        and sentiment analysis using machine learning models.
        
        **Features:**
        - Real-time data streaming
        - ML-based price prediction
        - NLP sentiment analysis
        - Interactive visualizations
        - Comprehensive logging
    """)


if __name__ == "__main__":
    main()
