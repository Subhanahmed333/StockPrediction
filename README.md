# Real-Time Stock Price & Sentiment Predictor

A machine learning-powered application for real-time stock price prediction and sentiment analysis.

## Features

- 📊 **Real-time Dashboard**: Interactive candlestick charts with technical indicators
- 🤖 **ML Predictions**: Price movement predictions using Random Forest, Gradient Boosting, and XGBoost
- 💬 **Sentiment Analysis**: Analyze market sentiment from news and social media
- 📈 **Analytics**: Track prediction history and performance metrics
- 📝 **Logging**: Monitor system activity and interactions

## Live Demo

[Add your Streamlit Cloud URL here after deployment]

## Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn, XGBoost
- **Data**: yfinance, python-binance
- **NLP**: NLTK, TextBlob, VADER
- **Visualization**: Plotly

## Local Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
4. Train models: `python train_models.py`
5. Run app: `streamlit run app.py`

## Usage

See [setup_instructions.md](setup_instructions.md) for detailed instructions.

## Model Training

Train custom models for any stock/crypto:

```bash
python train_models.py SYMBOL INTERVAL PERIOD
# Example: python train_models.py AAPL 1d 1y
```

## Deployment

This app is deployed on Streamlit Cloud. See deployment section in setup_instructions.md.

## License

MIT License

## Disclaimer

This application is for educational purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.
