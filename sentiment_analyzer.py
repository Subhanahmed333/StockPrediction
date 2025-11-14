"""
Sentiment Analysis Module
Analyzes text sentiment using multiple NLP approaches
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
import pandas as pd
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class SentimentAnalyzer:
    """
    Multi-method sentiment analysis for financial text
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial keywords for enhanced analysis
        self.positive_keywords = [
            'bullish', 'rally', 'surge', 'gain', 'profit', 'growth',
            'breakthrough', 'moon', 'pump', 'buy', 'long', 'strong'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'dump', 'loss', 'decline', 'fall',
            'sell', 'short', 'weak', 'fear', 'panic', 'collapse'
        ]
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for Twitter-like content)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            dict with polarity and subjectivity
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'method': 'textblob',
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'score': polarity
            }
            
        except Exception as e:
            print(f"TextBlob error: {e}")
            return None
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            dict with compound score and sentiment
        """
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            # Classify sentiment
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'method': 'vader',
                'compound': compound,
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg'],
                'sentiment': sentiment,
                'score': compound
            }
            
        except Exception as e:
            print(f"VADER error: {e}")
            return None
    
    def analyze_financial_keywords(self, text):
        """
        Analyze sentiment based on financial keywords
        
        Args:
            text: Text to analyze
        
        Returns:
            dict with keyword-based sentiment
        """
        text_lower = text.lower()
        
        pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
        neg_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            score = 0
            sentiment = 'neutral'
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                sentiment = 'positive'
            elif score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        return {
            'method': 'keywords',
            'positive_count': pos_count,
            'negative_count': neg_count,
            'score': score,
            'sentiment': sentiment
        }
    
    def analyze_comprehensive(self, text):
        """
        Comprehensive sentiment analysis using all methods
        
        Args:
            text: Text to analyze
        
        Returns:
            dict with combined analysis
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'score': 0,
                'confidence': 0,
                'timestamp': datetime.now()
            }
        
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Run all analysis methods
        textblob_result = self.analyze_textblob(cleaned_text)
        vader_result = self.analyze_vader(cleaned_text)
        keyword_result = self.analyze_financial_keywords(cleaned_text)
        
        # Combine results with weighted average
        scores = []
        sentiments = []
        
        if textblob_result:
            scores.append(textblob_result['score'] * 0.3)
            sentiments.append(textblob_result['sentiment'])
        
        if vader_result:
            scores.append(vader_result['score'] * 0.4)
            sentiments.append(vader_result['sentiment'])
        
        if keyword_result:
            scores.append(keyword_result['score'] * 0.3)
            sentiments.append(keyword_result['sentiment'])
        
        # Calculate final score
        final_score = sum(scores) if scores else 0
        
        # Determine final sentiment
        sentiment_counts = pd.Series(sentiments).value_counts()
        final_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else 'neutral'
        
        # Calculate confidence
        confidence = sentiment_counts.max() / len(sentiments) if sentiments else 0
        
        return {
            'sentiment': final_sentiment,
            'score': final_score,
            'confidence': confidence,
            'textblob': textblob_result,
            'vader': vader_result,
            'keywords': keyword_result,
            'timestamp': datetime.now(),
            'original_text': text[:100] + '...' if len(text) > 100 else text
        }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of text strings
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for text in texts:
            result = self.analyze_comprehensive(text)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, texts):
        """
        Get overall sentiment summary for multiple texts
        
        Args:
            texts: List of text strings
        
        Returns:
            dict with summary statistics
        """
        df = self.analyze_batch(texts)
        
        return {
            'total_texts': len(texts),
            'positive': (df['sentiment'] == 'positive').sum(),
            'negative': (df['sentiment'] == 'negative').sum(),
            'neutral': (df['sentiment'] == 'neutral').sum(),
            'average_score': df['score'].mean(),
            'average_confidence': df['confidence'].mean(),
            'overall_sentiment': df['sentiment'].mode()[0] if not df.empty else 'neutral'
        }


# Sample usage and testing
if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test samples
    samples = [
        "Bitcoin is showing strong bullish momentum! Great time to buy!",
        "The market is crashing, expect major losses ahead.",
        "Price consolidating around support level. Neutral outlook.",
        "🚀 Moon time! BTC to the moon! Best investment ever!",
        "Bearish trend continues. Sell signal confirmed."
    ]
    
    print("\n" + "="*60)
    print("Individual Text Analysis")
    print("="*60)
    
    for text in samples:
        print(f"\nText: {text}")
        result = analyzer.analyze_comprehensive(text)
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Score: {result['score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    print("\n" + "="*60)
    print("Batch Analysis Summary")
    print("="*60)
    
    summary = analyzer.get_sentiment_summary(samples)
    for key, value in summary.items():
        print(f"{key}: {value}")
