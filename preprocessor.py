"""
Data Preprocessing Module
CS 5800 - Algorithms Final Project

Handles data cleaning, normalization, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# Robust import for scikit-learn (package name is scikit-learn; import path is sklearn)
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except Exception:
    try:
        from scikit_learn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
    except Exception:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

from typing import Tuple, Optional, List
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
        self.features_created = False

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        missing_before = df.isnull().sum().sum()

        if self.fill_method == 'forward':
            df = df.fillna(method='ffill')
        elif self.fill_method == 'interpolate':
            df = df.interpolate(method='linear')
        else:
            df = df.fillna(method='bfill')

        # Fill any remaining NaN with column mean (numeric only)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        missing_after = df.isnull().sum().sum()
        print(f"Missing values reduced from {missing_before} to {missing_after}")

        return df

    def detect_outliers(self, df: pd.DataFrame, columns: List[str],
                       z_threshold: float = 3) -> pd.DataFrame:
        """
        Detect and handle outliers using Z-score method (capping).
        """
        outliers_handled = 0

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if series.empty:
                    continue
                # compute zscores safely
                z_scores = np.abs(stats.zscore(series))
                if np.isnan(z_scores).all():
                    continue
                outlier_mask = z_scores > z_threshold

                if outlier_mask.any():
                    mean = series.mean()
                    std = series.std()
                    if std == 0 or np.isnan(std):
                        continue
                    upper_limit = mean + (z_threshold * std)
                    lower_limit = mean - (z_threshold * std)
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                    outliers_handled += int(outlier_mask.sum())

        print(f"Handled {outliers_handled} outliers")
        return df

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators as features.
        """
        # Ensure numeric columns cleaned for operations
        for col in ['High', 'Low', 'Close', 'Open', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')

        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']

        df['Volume_SMA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)

        df['RSI'] = self.calculate_rsi(df['Close'])
        # avoid division by zero
        denom = (df['High'] - df['Low']).replace(0, np.nan)
        df['Price_Position'] = (df['Close'] - df['Low']) / denom

        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100

        # Replace infinities and excessive NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.features_created = True
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def normalize_features(self, df: pd.DataFrame,
                           columns_to_normalize: List[str]) -> pd.DataFrame:
        """
        Normalize specified columns using MinMax scaling.
        Coerces to numeric, fills remaining NaNs, and uses a fresh scaler per column.
        """
        for col in columns_to_normalize:
            if col in df.columns:
                # Remove common non-numeric characters then coerce
                cleaned = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
                # Fill missing values with median to avoid errors during scaling
                if cleaned.isna().all():
                    # skip if column has no numeric data
                    continue
                fill_value = cleaned.median()
                cleaned = cleaned.fillna(fill_value)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(cleaned.values.reshape(-1, 1)).flatten()
                df[f'{col}_normalized'] = scaled

        return df

    def create_lag_features(self, df: pd.DataFrame,
                           target_col: str = 'Close',
                           n_lags: int = 5) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        """
        for i in range(1, n_lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)

        return df

    def prepare_train_test_split(self, df: pd.DataFrame,
                                 test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets (time-aware).
        """
        df = df.dropna()
        split_index = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

        print(f"Training set: {len(train_df)} samples")
        print(f"Testing set: {len(test_df)} samples")
        print(f"Features created: {len(df.columns)}")

        return train_df, test_df

    def get_feature_importance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate correlation-based feature importance.
        """
        if 'Close' in df.columns:
            numeric = df.select_dtypes(include=[np.number])
            if 'Close' not in numeric.columns:
                return pd.Series()
            correlations = numeric.corr()['Close'].abs().sort_values(ascending=False)
            # Exclude Close itself
            correlations = correlations.drop(labels=['Close'], errors='ignore')
            return correlations.head(10)
        return pd.Series()


# Main execution
if __name__ == "__main__":
    # Load raw data
    raw_path = 'data/raw/AAPL_historical.csv'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)

    preprocessor = DataPreprocessor(fill_method='forward')

    print("Starting preprocessing pipeline...")

    df = preprocessor.handle_missing_values(df)
    df = preprocessor.detect_outliers(df, ['Volume', 'Close'])
    df = preprocessor.create_technical_features(df)
    df = preprocessor.create_lag_features(df, n_lags=5)
    df = preprocessor.normalize_features(df, ['Volume', 'Close', 'High', 'Low'])

    train_df, test_df = preprocessor.prepare_train_test_split(df)

    print("\nTop 10 Important Features:")
    print(preprocessor.get_feature_importance(df))

    train_df.to_csv(os.path.join(processed_dir, 'AAPL_train.csv'))
    test_df.to_csv(os.path.join(processed_dir, 'AAPL_test.csv'))
    print(f"\nPreprocessing complete! Processed files saved to {processed_dir}")