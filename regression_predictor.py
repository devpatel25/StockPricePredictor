"""
Regression-Based Price Prediction Module
CS 5800 - Algorithms Final Project

Implements linear and polynomial regression for price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.impute import SimpleImputer
import joblib
from typing import Tuple, Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

class RegressionPredictor:
    """
    Price prediction using regression algorithms.
    CS 5800 Reference: Optimization algorithms (Module 7)
    """
    
    def __init__(self, lookback_days: int = 5):
        """
        Initialize predictor with lookback window.
        
        Args:
            lookback_days: Number of previous days to use for prediction
        """
        self.lookback_days = lookback_days
        self.models = {}
        self.feature_names = []
        self.best_model = None
        self.scaler = None
        self.imputer = None
        
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix and target vector.
        
        Time Complexity: O(n*m) where m is number of features
        Space Complexity: O(n*m)
        
        CS 5800: Feature engineering as preprocessing step
        """
        features = []
        targets = []
        
        for i in range(self.lookback_days, len(df)):
            # Price features (last n days)
            price_features = df['Close'].iloc[i-self.lookback_days:i].values.tolist()
            
            # Volume feature (normalized)
            volume_feature = df['Volume'].iloc[i-1] / 1e9
            
            # Technical features
            volatility = df['Volatility'].iloc[i-1] if 'Volatility' in df else 0
            rsi = df['RSI'].iloc[i-1] if 'RSI' in df else 50
            
            # Moving average features
            sma_20 = df['SMA_20'].iloc[i-1] if 'SMA_20' in df else df['Close'].iloc[i-1]
            ema_12 = df['EMA_12'].iloc[i-1] if 'EMA_12' in df else df['Close'].iloc[i-1]
            
            # Price ratios
            price_to_sma = df['Close'].iloc[i-1] / sma_20 if sma_20 != 0 else 1
            
            # Combine all features
            feature_vector = (price_features + 
                            [volume_feature, volatility, rsi, 
                             price_to_sma, ema_12])
            
            features.append(feature_vector)
            targets.append(df['Close'].iloc[i])
        
        # Store feature names for interpretation
        self.feature_names = ([f'price_lag_{i+1}' for i in range(self.lookback_days)] +
                             ['volume_norm', 'volatility', 'rsi', 
                              'price_to_sma', 'ema_12'])
        
        return np.array(features), np.array(targets)
    
    def train_linear_regression(self, X_train: np.ndarray, 
                               y_train: np.ndarray) -> LinearRegression:
        """
        Train standard linear regression model.
        
        Time Complexity: O(n*m^2) for normal equation
        Space Complexity: O(m^2) for covariance matrix
        
        CS 5800: Least squares optimization
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear'] = model
        return model
    
    def train_ridge_regression(self, X_train: np.ndarray, 
                              y_train: np.ndarray, 
                              alpha: float = 1.0) -> Ridge:
        """
        Train Ridge regression with L2 regularization.
        
        CS 5800: Regularization to prevent overfitting
        """
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
        return model
    
    def train_polynomial_regression(self, X_train: np.ndarray, 
                                   y_train: np.ndarray, 
                                   degree: int = 2) -> Tuple:
        """
        Train polynomial regression model.
        
        Time Complexity: O(n*m^degree) for feature expansion
        Space Complexity: O(n*m^degree)
        
        CS 5800: Non-linear function approximation
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_train)
        
        # Train model on polynomial features
        model = LinearRegression()
        model.fit(X_poly, y_train)
        
        self.models[f'polynomial_{degree}'] = (poly, model)
        return poly, model
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, 
                           model, cv_folds: int = 5) -> Dict:
        """
        Perform time series cross-validation.
        
        CS 5800: Model validation using divide-and-conquer
        """
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Calculate cross-validation scores
        mse_scores = -cross_val_score(model, X, y, 
                                     cv=tscv, 
                                     scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, 
                                   cv=tscv, 
                                   scoring='r2')
        
        return {
            'mse_mean': mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
    
    def predict_next_day(self, model, current_features: np.ndarray) -> float:
        """
        Predict next day's closing price.
        
        Time Complexity: O(m) for prediction
        Space Complexity: O(1)
        """
        # Handle polynomial models
        if isinstance(model, tuple):
            poly, linear_model = model
            current_features = poly.transform(current_features.reshape(1, -1))
            prediction = linear_model.predict(current_features)[0]
        else:
            prediction = model.predict(current_features.reshape(1, -1))[0]
        
        return prediction
    
    def generate_trading_signal(self, current_price: float, 
                              predicted_price: float, 
                              threshold: float = 0.01) -> str:
        """
        Generate trading signal based on prediction.
        
        CS 5800: Decision algorithm based on threshold
        """
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > threshold:
            return "BUY"
        elif price_change < -threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of different regression models.
        
        CS 5800: Algorithm comparison and selection
        """
        results = []
        
        # Train and evaluate linear regression
        print("Training Linear Regression...")
        linear_model = self.train_linear_regression(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        
        results.append({
            'Model': 'Linear Regression',
            'MSE': mean_squared_error(y_test, y_pred_linear),
            'MAE': mean_absolute_error(y_test, y_pred_linear),
            'R2': r2_score(y_test, y_pred_linear),
            'Complexity': 'O(nm²)'
        })
        
        # Train and evaluate Ridge regression
        print("Training Ridge Regression...")
        ridge_model = self.train_ridge_regression(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)
        
        results.append({
            'Model': 'Ridge Regression',
            'MSE': mean_squared_error(y_test, y_pred_ridge),
            'MAE': mean_absolute_error(y_test, y_pred_ridge),
            'R2': r2_score(y_test, y_pred_ridge),
            'Complexity': 'O(nm²)'
        })
        
        # Train and evaluate polynomial regression (degree 2)
        print("Training Polynomial Regression (degree 2)...")
        poly2, model2 = self.train_polynomial_regression(X_train, y_train, degree=2)
        X_test_poly2 = poly2.transform(X_test)
        y_pred_poly2 = model2.predict(X_test_poly2)
        
        results.append({
            'Model': 'Polynomial Regression (2)',
            'MSE': mean_squared_error(y_test, y_pred_poly2),
            'MAE': mean_absolute_error(y_test, y_pred_poly2),
            'R2': r2_score(y_test, y_pred_poly2),
            'Complexity': 'O(nm^d)'
        })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R2', ascending=False)
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        if 'Linear' in best_model_name:
            self.best_model = self.models['linear']
        elif 'Ridge' in best_model_name:
            self.best_model = self.models['ridge']
        else:
            self.best_model = self.models['polynomial_2']
        
        return results_df
    
    def save_model(self, filepath: str):
        """Save the best model to disk."""
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

# Main execution
if __name__ == "__main__":
    # Load data with signals
    df = pd.read_csv('data/signals/AAPL_ma_signals.csv', 
                     index_col='Date', parse_dates=True)
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/predictions', exist_ok=True)
    
    # Initialize predictor
    predictor = RegressionPredictor(lookback_days=5)
    
    # Create features
    print("Creating features...")
    X, y = predictor.create_features(df)
    
    # Split data (80/20)
    split_index = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training samples: {len(X_train_raw)}")
    print(f"Testing samples: {len(X_test_raw)}")
    print(f"Features per sample: {X_train_raw.shape[1]}")
    
    # Impute missing values (fit on training, apply to both)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    predictor.imputer = imputer  # store for potential later use
    
    # Evaluate all models
    print("\nEvaluating models...")
    results_df = predictor.evaluate_models(X_train, y_train, X_test, y_test)
    
    print("\nModel Comparison:")
    print(results_df.to_string())
    
    # Make predictions with best model
    print(f"\nBest model selected: {results_df.iloc[0]['Model']}")
    
    # Generate signals for test set
    signals = []
    predictions = []
    
    for i in range(len(X_test)):
        # current_price: use previous actual (test series) or last training target for first test sample
        current_price = y_test[i] if i > 0 else y_train[-1]
        # features are already imputed
        current_features = X_test[i]
        predicted_price = predictor.predict_next_day(predictor.best_model, current_features)
        signal = predictor.generate_trading_signal(current_price, predicted_price)
        
        signals.append(signal)
        predictions.append(predicted_price)
    
    # Calculate signal distribution
    signal_counts = pd.Series(signals).value_counts()
    print(f"\nSignal Distribution:")
    print(signal_counts)
    
    # Save model
    predictor.save_model('models/best_regression_model.pkl')
    
    # Save predictions (align index if desired)
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'Signal': signals
    })
    test_results.to_csv('data/predictions/regression_predictions.csv', index=False)
    print("\nPredictions saved to data/predictions/regression_predictions.csv")