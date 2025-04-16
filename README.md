# Smart Trading Assistant

Smart Trading Assistant is an application that performs technical analysis and generates trading recommendations for both stocks and cryptocurrencies. 
It combines financial data, deep learning (LSTM), and large language models to deliver intelligent insights and portfolio suggestions.

([Demo Video](https://github.com/user-attachments/assets/8864f275-c03e-4246-8d16-b1a81e788952))

## Features

- **Dual Asset Analysis**: Supports both stock and cryptocurrency analysis
- **Real-time Data Integration**:
  - Stocks via Yahoo Finance
  - Cryptocurrency via Binance
  - Blockchain wallet integration
- **Advanced Analysis**:
  - LSTM-based price prediction
  - Technical indicators
  - Volume analysis
  - Trend identification
- **AI-Powered Recommendations**:
  - Market trend analysis
  - Risk assessment
  - Entry/exit points
  - Position sizing suggestions
- **Interactive Web Interface**:
  - Built with Streamlit
  - Easy symbol selection
  - Popular preset options
  - Real-time analysis updates

## Prerequisites

- Python 3.8+
- Required API keys:
  - Hugging Face API token
  - Binance API credentials
  - Web3 provider URL

## Installation

1. Clone the repository
2. Install required packages:
```cmd
pip install numpy pandas yfinance scikit-learn python-dotenv ccxt python-binance web3 tensorflow langchain streamlit
```

3. Create a `.env` file with your API credentials:
```dotenv
HUGGINGFACEHUB_API_TOKEN=your_token_here
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
WEB3_PROVIDER=your_web3_provider_url
```

## Usage
Run the application:
```bash
streamlit run app.py
```

1. Select asset type (Stock/Crypto) from the sidebar
2. Choose a symbol from presets or enter your own
3. Click "Analyze" to get detailed insights

## Components
- `StockAnalysisAgent` : Handles stock data fetching and analysis
- `CryptoAnalysisAgent` : Manages cryptocurrency data and analysis
- `RecommendationAgent` : Generates AI-powered trading recommendations
- `EnhancedTradingSystem` : Coordinates all components and provides unified analysis

## Model Architecture
- LSTM-based neural network
- Dual-layer LSTM configuration
- MinMaxScaler for data normalization
- Dense layers for final prediction

## Security Features
- Environment variable-based credential management
- Secure API integrations
- Error handling and validation

## Technical Details
- Uses TensorFlow for machine learning
- Implements Mistral-7B for NLP tasks
- Streamlit for web interface
- Real-time data fetching and processing

## Error Handling
The system includes comprehensive error handling for:
- Data fetching failures
- API connection issues
- Invalid symbols
- Insufficient data scenarios


