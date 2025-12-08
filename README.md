# StockPricePredictor

[![GitHub](https://img.shields.io/badge/GitHub-devpatel25/StockPricePredictor-blue)](https://github.com/devpatel25/StockPricePredictor)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive stock price prediction and trading strategy evaluation system built with Python. This project implements multiple machine learning algorithms, technical analysis indicators, and backtesting capabilities to analyze and predict stock price movements.

## 🚀 Features

### Core Functionality
- **Data Collection**: Automated stock data fetching using Yahoo Finance API
- **Data Preprocessing**: Robust data cleaning and feature engineering
- **Multiple Prediction Models**:
  - Linear Regression with feature selection
  - Dynamic Programming optimization
  - Moving Average analysis
- **Signal Generation**: Technical indicators and trading signals
- **Backtesting Engine**: Realistic trading simulation with commissions and slippage
- **Performance Metrics**: Comprehensive evaluation of trading strategies

### Technical Indicators
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- MACD (Moving Average Convergence Divergence)
- Golden/Death Cross signals
- Price vs SMA signals
- EMA crossover signals

### Visualization & Dashboard
- Interactive web dashboard built with Streamlit
- Performance charts and backtest results
- Real-time data visualization
- Trade log analysis

## 📊 Project Structure

```
StockPricePredictor/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── scripts/                   # Core modules
│   ├── data_collector.py      # Stock data collection
│   ├── preprocessor.py        # Data preprocessing
│   ├── regression_predictor.py # ML regression models
│   ├── dynamic_programming.py # DP optimization
│   ├── moving_averages.py     # Technical analysis
│   ├── signal_generator.py    # Trading signals
│   ├── backtester.py          # Backtesting engine
│   └── performance_metrics.py # Performance evaluation
├── data/                      # Data storage
│   ├── raw/                   # Raw stock data
│   ├── processed/             # Cleaned data
│   ├── predictions/           # Model predictions
│   └── signals/               # Generated signals
├── models/                    # Trained models
├── results/                   # Backtest results
└── README.md                  # Project documentation
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/devpatel25/StockPricePredictor.git
   cd StockPricePredictor
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### Quick Start
Run the main application:
```bash
python main.py
```

### Individual Components

#### Data Collection
```python
from scripts.data_collector import DataCollector

collector = DataCollector()
data = collector.fetch_stock_data('AAPL', '2020-01-01', '2023-12-31')
```

#### Model Training
```python
from scripts.regression_predictor import RegressionPredictor

predictor = RegressionPredictor()
predictor.train_model('data/processed/AAPL_train.csv')
predictions = predictor.predict('data/processed/AAPL_test.csv')
```

#### Backtesting
```python
from scripts.backtester import Backtester

backtester = Backtester(initial_capital=10000)
metrics = backtester.run_backtest(df_with_signals)
backtester.plot_results(metrics)
```

#### Web Dashboard
```bash
streamlit run dashboard.py  # If dashboard file exists
```

## 🔧 Configuration

### Model Parameters
- **SMA Windows**: [20, 50, 200] (configurable in MovingAverageAnalyzer)
- **EMA Spans**: [12, 26] (configurable in MovingAverageAnalyzer)
- **Backtest Settings**: Capital, commission, slippage (configurable in Backtester)

### Data Sources
- Primary: Yahoo Finance (yfinance library)
- Supported symbols: AAPL, AMZN, GOOGL, MSFT, NVDA (and more)

## 📊 Results & Performance

### Sample Backtest Results (AAPL)
- **Initial Capital**: $10,000
- **Final Value**: Varies by strategy
- **Sharpe Ratio**: Calculated for risk-adjusted returns
- **Max Drawdown**: Risk assessment metric
- **Win Rate**: Trading success percentage

### Key Metrics
- Total Return
- Annual Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Buy & Hold Comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data API
- Scikit-learn for machine learning algorithms
- Streamlit for web dashboard framework
- Technical analysis libraries (ta, pandas-ta)

## 📞 Contact

**Dev Patel**
- GitHub: [@devpatel25](https://github.com/devpatel25)
- Project Link: [https://github.com/devpatel25/StockPricePredictor](https://github.com/devpatel25/StockPricePredictor)

## 🔄 Future Enhancements

- [ ] Deep learning models (LSTM, CNN)
- [ ] Real-time trading integration
- [ ] Portfolio optimization
- [ ] Risk management modules
- [ ] Additional technical indicators
- [ ] Multi-asset support
- [ ] Cloud deployment options

---

**Disclaimer**: This project is for educational and research purposes only. Not intended for actual trading or investment decisions. Past performance does not guarantee future results.
