import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import ccxt
from binance.client import Client
from web3 import Web3
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from forex_python.converter import CurrencyRates


# Load environment variables from .env file
load_dotenv()

class StockAnalysisAgent:
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not self.api_key:
            raise ValueError("Hugging Face API token not found in .env file")
        
        self.model = self._build_lstm_model()
        self.scaler = MinMaxScaler()
    
    def _build_lstm_model(self):
        inputs = tf.keras.Input(shape=(60, 1))
        x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(50, return_sequences=False)(x)
        x = tf.keras.layers.Dense(25)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fetch_stock_data(self, symbol, period='1y'):
        """
        Fetch stock data using yfinance
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period to fetch data for (default: '1y')
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty:
                print(f"No data found for symbol {symbol}")
                return None
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def prepare_data(self, data):
        """
        Prepare data for LSTM model
        
        Args:
            data (pandas.DataFrame): Stock data with 'Close' prices
            
        Returns:
            tuple: (X, y) arrays for training
        """
        if data is None or 'Close' not in data.columns:
            return np.array([]), np.array([])
            
        prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)


class RecommendationAgent:
    def __init__(self):
        api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not api_token:
            raise ValueError("Hugging Face API token not found in .env file")
            
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=api_token,
            task="text-generation",
            temperature=0.7,
            max_new_tokens=512
        )
        
        # prompt template optimized for Mistral-7B-Instruct
        self.recommendation_prompt = PromptTemplate(
            input_variables=["stock", "analysis"],
            template="""<s>[INST] You are a professional financial advisor. Based on this technical analysis for {stock}:
            {analysis}

            Provide a detailed trading recommendation covering:
            1. Current market trends and price action
            2. Risk assessment and potential downside
            3. Specific entry and exit price points
            4. Position sizing and portfolio diversification advice

            Format your response in clear, actionable bullet points. [/INST]</s>
            """
        )
    
    def generate_recommendation(self, stock, analysis):
        try:
            chain = LLMChain(llm=self.llm, prompt=self.recommendation_prompt)
            return chain.run(stock=stock, analysis=analysis)
        except Exception as e:
            return f"Error generating recommendation: {str(e)}\nPlease try again in a few moments."



class CryptoAnalysisAgent:
    def __init__(self):
        # Initialize exchange connections
        self.binance = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY')
        })
        self.model = self._build_lstm_model()
        self.scaler = MinMaxScaler()

    def _build_lstm_model(self):
        # Same as StockAnalysisAgent's model
        inputs = tf.keras.Input(shape=(60, 1))
        x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(50, return_sequences=False)(x)
        x = tf.keras.layers.Dense(25)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fetch_crypto_data(self, symbol, timeframe='1d', limit=365):
        try:
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching crypto data: {e}")
            return None

    def prepare_data(self, data):
        # Same as StockAnalysisAgent's prepare_data method
        if data is None or 'close' not in data.columns:
            return np.array([]), np.array([])
            
        prices = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

class WalletIntegration:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER')))
        
    def get_wallet_balance(self, address, token_address=None):
        try:
            if token_address:
                # Get ERC20 token balance
                abi = self.get_token_abi(token_address)
                contract = self.web3.eth.contract(address=token_address, abi=abi)
                balance = contract.functions.balanceOf(address).call()
                return balance
            else:
                # Get ETH balance
                balance = self.web3.eth.get_balance(address)
                return self.web3.from_wei(balance, 'ether')
        except Exception as e:
            print(f"Error getting wallet balance: {e}")
            return None

class ForexAnalysisAgent:
    def __init__(self):
        self.c = CurrencyRates()
        self.model = self._build_lstm_model()
        self.scaler = MinMaxScaler()

    def _build_lstm_model(self):
        inputs = tf.keras.Input(shape=(60, 1))
        x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(50, return_sequences=False)(x)
        x = tf.keras.layers.Dense(25)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fetch_forex_data(self, pair, days=365):
        try:
            base_currency, quote_currency = pair.split('/')
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    rate = self.c.get_rate(base_currency, quote_currency, current_date)
                    data.append({
                        'Date': current_date,
                        'Close': rate,
                        'Volume': 0  # Forex doesn't have traditional volume
                    })
                except:
                    pass
                current_date += timedelta(days=1)
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching forex data: {e}")
            return None

    def prepare_data(self, data):
        if data is None or 'Close' not in data.columns:
            return np.array([]), np.array([])
            
        prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

class EnhancedTradingSystem:
    def __init__(self):
        self.stock_agent = StockAnalysisAgent()
        self.crypto_agent = CryptoAnalysisAgent()
        self.forex_agent = ForexAnalysisAgent()  # Add forex agent
        self.wallet_integration = WalletIntegration()
        self.recommendation_agent = RecommendationAgent()

    def analyze_asset(self, symbol, asset_type='stock'):
        if asset_type == 'stock':
            data = self.stock_agent.fetch_stock_data(symbol)
            agent = self.stock_agent
        elif asset_type == 'crypto':
            data = self.crypto_agent.fetch_crypto_data(symbol)
            agent = self.crypto_agent
        else:  # forex
            data = self.forex_agent.fetch_forex_data(symbol)
            agent = self.forex_agent

        if data is None or len(data) < 60:
            return None
        
        X, y = agent.prepare_data(data)
        if len(X) == 0:
            return None
            
        agent.model.fit(X, y, batch_size=32, epochs=10, verbose=0)
        
        last_60_days = agent.scaler.transform(
            data['Close' if asset_type == 'stock' else 'close'].values[-60:].reshape(-1, 1)
        )
        next_day_pred = agent.model.predict(
            last_60_days.reshape(1, 60, 1)
        )
        
        current_price = data['Close' if asset_type == 'stock' else 'close'].iloc[-1]
        predicted_price = agent.scaler.inverse_transform(next_day_pred)[0][0]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'volume': data['Volume' if asset_type == 'stock' else 'volume'].iloc[-1],
            'trend': 'Bullish' if price_change > 0 else 'Bearish'
        }

    def get_recommendation(self, symbol, asset_type='stock'):
        analysis = self.analyze_asset(symbol, asset_type)
        if analysis is None:
            return f"Unable to analyze {asset_type} data"
        
        recommendation = self.recommendation_agent.generate_recommendation(
            symbol, str(analysis)
        )
        
        return {
            'analysis': analysis,
            'recommendation': recommendation
        }

def main():
    st.set_page_config(page_title="Financial Analyzer", layout="wide")
    st.title("TradeMatrix  System")

    # Sidebar for trading times
    st.sidebar.title("Market Hours")
    
    # Get current time in different time zones
    now = datetime.now()
    ny_tz = pytz.timezone('America/New_York')
    london_tz = pytz.timezone('Europe/London')
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    sydney_tz = pytz.timezone('Australia/Sydney')

    # Get current UTC time
    utc_now = datetime.now(pytz.utc)

    # Convert to each timezone
    ny_time = utc_now.astimezone(ny_tz).strftime('%H:%M:%S')
    london_time = utc_now.astimezone(london_tz).strftime('%H:%M:%S')
    tokyo_time = utc_now.astimezone(tokyo_tz).strftime('%H:%M:%S')
    sydney_time = utc_now.astimezone(sydney_tz).strftime('%H:%M:%S')

    # Display market hours in sidebar
    st.sidebar.markdown("### Current Market Times")
    st.sidebar.markdown(f"New York: {ny_time}")
    st.sidebar.markdown(f"London: {london_time}")
    st.sidebar.markdown(f"Tokyo: {tokyo_time}")
    st.sidebar.markdown(f"Sydney: {sydney_time}")

    # Market Status
    st.sidebar.markdown("### Market Status")
    ny_hour = int(now.strftime("%H"))
    if 9 <= ny_hour < 16:  # New York trading hours
        st.sidebar.success("🟢 New York Market: Open")
    else:
        st.sidebar.error("🔴 New York Market: Closed")

    # Asset type selection
    asset_type = st.selectbox(
        "Select Asset Type",
        ["Stocks", "Cryptocurrency", "Forex"]
    )

    trading_system = EnhancedTradingSystem()

    if asset_type == "Stocks":
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOGL)")
        
    elif asset_type == "Cryptocurrency":
        symbol = st.text_input("Enter Trading Pair (e.g., BTC/USDT, ETH/USDT)")
        
        # Wallet Integration
        wallet_address = st.text_input("Enter Wallet Address (Optional)")
        if wallet_address:
            balance = trading_system.wallet_integration.get_wallet_balance(wallet_address)
            if balance:
                st.info(f"Wallet Balance: {balance} ETH")
                
    else:  # Forex
        # Common forex pairs
        forex_pairs = {
            "Major Pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"],
            "Minor Pairs": ["EUR/GBP", "GBP/JPY", "EUR/JPY"],
            "Commodity Pairs": ["USD/CAD", "AUD/USD", "NZD/USD"]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            pair_category = st.selectbox("Select Pair Category", list(forex_pairs.keys()))
        with col2:
            symbol = st.selectbox("Select Currency Pair", forex_pairs[pair_category])

        # Show forex-specific market hours
        st.sidebar.markdown("### Forex Trading Sessions")
        st.sidebar.markdown("Sydney: 5:00 PM - 2:00 AM ET")
        st.sidebar.markdown("Tokyo: 7:00 PM - 4:00 AM ET")
        st.sidebar.markdown("London: 3:00 AM - 12:00 PM ET")
        st.sidebar.markdown("New York: 8:00 AM - 5:00 PM ET")

    if st.button("Analyze"):
        with st.spinner("Fetching data and analyzing..."):
            result = trading_system.get_recommendation(
                symbol.upper(),
                asset_type.lower()
            )

        if isinstance(result, str):
            st.error(result)
        else:
            st.success("Analysis Complete!")
            analysis = result['analysis']

            col1, col2, col3 = st.columns(3)
            
            # Format price based on asset type
            if asset_type == "Forex":
                price_prefix = ""
                price_suffix = f" {symbol.split('/')[1]}"
            else:
                price_prefix = "$"
                price_suffix = ""
                
            col1.metric(
                "Current Price", 
                f"{price_prefix}{analysis['current_price']:.4f}{price_suffix}"
            )
            col2.metric(
                "Predicted Price", 
                f"{price_prefix}{analysis['predicted_price']:.4f}{price_suffix}"
            )
            col3.metric(
                "Expected Change",
                f"{analysis['price_change']:.2f}%",
                delta=analysis['price_change'],
                delta_color="inverse"
            )
            
            st.markdown(f"**Volume:** {int(analysis['volume']) if analysis['volume'] != 'N/A' else 'N/A'}")
            st.markdown(f"**📈 Trend:** {analysis['trend']}")

            st.subheader("AI Trading Recommendation")
            st.markdown(result['recommendation'])

if __name__ == "__main__":
    main()
