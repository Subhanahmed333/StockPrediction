"""
Machine Learning Models Module
Contains multiple ML models for price prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StockPricePredictor:
    """
    Ensemble of ML models for stock price movement prediction
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = []
        
    def create_model(self, model_type=None):
        """
        Create ML model based on type
        
        Args:
            model_type: Type of model to create
        
        Returns:
            Initialized model
        """
        if model_type is None:
            model_type = self.model_type
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        }
        
        return models.get(model_type, models['random_forest'])
    
    def prepare_data(self, df, feature_cols, target_col='target'):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Select features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, df, feature_cols, target_col='target', optimize=False):
        """
        Train the model
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            optimize: Whether to perform hyperparameter tuning
        
        Returns:
            dict with training results
        """
        print(f"\nTraining {self.model_type} model...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, feature_cols, target_col)
        
        # Create model
        self.model = self.create_model()
        
        # Hyperparameter optimization
        if optimize and self.model_type == 'random_forest':
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                self.model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train model
            self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.evaluate_model(y_train, y_train_pred)
        test_metrics = self.evaluate_model(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                feature_cols, 
                self.model.feature_importances_
            ))
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        # Store training history
        results = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'feature_importance': feature_importance,
            'n_features': len(feature_cols),
            'n_samples': len(df)
        }
        
        self.training_history.append(results)
        
        # Print results
        print("\n" + "="*60)
        print("Training Results")
        print("="*60)
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"CV Score:       {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Precision:      {test_metrics['precision']:.4f}")
        print(f"Recall:         {test_metrics['recall']:.4f}")
        print(f"F1 Score:       {test_metrics['f1_score']:.4f}")
        
        if feature_importance:
            print("\nTop 5 Important Features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
                print(f"{i}. {feature}: {importance:.4f}")
        
        return results
    
    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            dict with evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def predict(self, features):
        """
        Make predictions on new data
        
        Args:
            features: DataFrame or array of features
        
        Returns:
            dict with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame if needed
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=self.feature_names)
        
        # Ensure correct feature order
        features = features[self.feature_names]
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probabilities),
            'probability_up': probabilities[1] if len(probabilities) > 1 else 0,
            'probability_down': probabilities[0],
            'timestamp': datetime.now()
        }
    
    def predict_batch(self, features):
        """
        Make predictions on multiple samples
        
        Args:
            features: DataFrame of features
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct feature order
        features = features[self.feature_names]
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        results = pd.DataFrame({
            'prediction': ['UP' if p == 1 else 'DOWN' for p in predictions],
            'confidence': probabilities.max(axis=1),
            'probability_up': probabilities[:, 1] if probabilities.shape[1] > 1 else 0,
            'probability_down': probabilities[:, 0]
        })
        
        return results
    
    def save_model(self, filepath='model.pkl'):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.training_history = model_data.get('training_history', [])
        
        print(f"Model loaded from {filepath}")


class ModelComparison:
    """
    Compare multiple ML models
    """
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, df, feature_cols, target_col='target', model_types=None):
        """
        Train and compare multiple models
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            model_types: List of model types to compare
        
        Returns:
            DataFrame with comparison results
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'xgboost', 
                          'logistic_regression', 'decision_tree']
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = []
        
        for model_type in model_types:
            print(f"\nTraining {model_type}...")
            
            try:
                predictor = StockPricePredictor(model_type=model_type)
                result = predictor.train(df, feature_cols, target_col)
                
                results.append({
                    'model': model_type,
                    'test_accuracy': result['test_metrics']['accuracy'],
                    'test_precision': result['test_metrics']['precision'],
                    'test_recall': result['test_metrics']['recall'],
                    'test_f1': result['test_metrics']['f1_score'],
                    'cv_score': result['cv_mean_score']
                })
                
                self.results[model_type] = predictor
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df


# Sample usage
if __name__ == "__main__":
    print("Testing ML Models...")
    
    # This would normally be called from the main application
    # with real data from the data collector
    print("\nML Models module loaded successfully!")
    print("Available models: random_forest, gradient_boosting, xgboost,")
    print("                 logistic_regression, svm, decision_tree")
