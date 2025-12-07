"""
Data Preprocessing Module
CS 5800 - Algorithms Final Project

Handles data cleaning, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Preprocesses stock data for algorithm consumption.
    CS 5800 Reference: Implements divide-and-conquer for data processing (Module 2)
    """
    
    def __init__(self, fill_method: str = 'forward'):
        """
        Initialize preprocessor with configuration.
        
        Args:
            fill_method: Method for handling missing values
        """
        self.fill_method = fill_method
        self.scaler = MinMaxScaler()
        self.features_created = False
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Time Complexity: O(n) where n is number of rows
        Space Complexity: O(1) - in-place operation
        """
        # Count missing values
        missing_before = df.isnull().sum().sum()
        
        if self.fill_method == 'forward':
            # Forward fill for time series continuity
            df = df.fillna(method='ffill')
        elif self.fill_method == 'interpolate':
            # Linear interpolation for smoother filling
            df = df.interpolate(method='linear')
        else:
            # Backward fill as fallback
            df = df.fillna(method='bfill')
        
        # Fill any remaining NaN with column mean
        df = df.fillna(df.mean())
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values reduced from {missing_before} to {missing_after}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], 
                       z_threshold: float = 3) -> pd.DataFrame:
        """
        Detect and handle outliers using Z-score method.
        
        CS 5800 Reference: Statistical analysis for data quality (Module 6)
        Time Complexity: O(n*m) where m is number of columns
        """
        outliers_removed = 0
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_indices = np.where(z_scores > z_threshold)[0]
                
                if len(outlier_indices) > 0:
                    # Cap outliers at threshold instead of removing
                    mean = df[col].mean()
                    std = df[col].std()
                    upper_limit = mean + (z_threshold * std)
                    lower_limit = mean - (z_threshold * std)
                    
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                    outliers_removed += len(outlier_indices)
        
        print(f"Handled {outliers_removed} outliers")
        return df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators as features.
        
        Time Complexity: O(n) for each indicator
        Space Complexity: O(n) for new features
        """
        # Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        
        # Volume features
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        # Volatility features (20-day rolling)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Price position features
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Momentum features
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        self.features_created = True
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        CS 5800 Reference: Sliding window technique (Module 4)
        Time Complexity: O(n)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def normalize_features(self, df: pd.DataFrame, 
                         columns_to_normalize: List[str]) -> pd.DataFrame:
        """
        Normalize specified columns using MinMax scaling.
        
        Time Complexity: O(n*m) where m is number of columns
        """
        for col in columns_to_normalize:
            if col in df.columns:
                df[f'{col}_normalized'] = self.scaler.fit_transform(
                    df[[col]].values.reshape(-1, 1)
                )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_col: str = 'Close', 
                           n_lags: int = 5) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        CS 5800 Reference: Dynamic programming approach to feature creation
        Time Complexity: O(n*k) where k is number of lags
        """
        for i in range(1, n_lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                                test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets (time-aware).
        
        CS 5800 Reference: Divide-and-conquer strategy (Module 2)
        """
        # Remove NaN rows created by feature engineering
        df = df.dropna()
        
        # Calculate split index
        split_index = int(len(df) * (1 - test_size))
        
        # Time series split (no shuffling)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Testing set: {len(test_df)} samples")
        print(f"Features created: {len(df.columns)}")
        
        return train_df, test_df
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate correlation-based feature importance.
        
        Time Complexity: O(n*m²) for correlation matrix
        """
        # Calculate correlation with target (Close price)
        if 'Close' in df.columns:
            correlations = df.corr()['Close'].abs().sort_values(ascending=False)
            return correlations[1:11]  # Top 10 features excluding Close itself
        return pd.Series()

# Main execution
if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('data/raw/AAPL_historical.csv', index_col='Date', parse_dates=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(fill_method='forward')
    
    # Apply preprocessing pipeline
    print("Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Step 2: Detect and handle outliers
    df = preprocessor.detect_outliers(df, ['Volume', 'Close'])
    
    # Step 3: Create technical features
    df = preprocessor.create_technical_features(df)
    
    # Step 4: Create lag features
    df = preprocessor.create_lag_features(df, n_lags=5)
    
    # Step 5: Normalize features
    df = preprocessor.normalize_features(df, ['Volume', 'Close', 'High', 'Low'])
    
    # Step 6: Split data
    train_df, test_df = preprocessor.prepare_train_test_split(df)
    
    # Step 7: Get feature importance
    print("\nTop 10 Important Features:")
    print(preprocessor.get_feature_importance(df))
    
    # Save preprocessed data
    train_df.to_csv('data/processed/AAPL_train.csv')
    test_df.to_csv('data/processed/AAPL_test.csv')
    print("\nPreprocessing complete!")