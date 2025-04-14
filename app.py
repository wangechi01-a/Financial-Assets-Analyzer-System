import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import ccxt
from binance.client import Client
from web3 import Web3
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

# Load environment variables
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
        if data is None or 'Close' not in data.columns:
            return np.array([]), np.array([])
        prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

class CryptoAnalysisAgent:
    def __init__(self):
        self.binance = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY')
        })
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
                abi = self.get_token_abi(token_address)
                contract = self.web3.eth.contract(address=token_address, abi=abi)
                balance = contract.functions.balanceOf(address).call()
                return balance
            else:
                balance = self.web3.eth.get_balance(address)
                return self.web3.from_wei(balance, 'ether')
        except Exception as e:
            print(f"Error getting wallet balance: {e}")
            return None

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
        self.recommendation_prompt = PromptTemplate(
            input_variables=["stock", "analysis"],
            template="""<s>[INST] You are a professional financial advisor. Based on this technical analysis for {stock}:
            {analysis}

            Provide a detailed trading recommendation covering:
            1. Current market trends and price action
            2. Risk assessment and potential downside
            3. Specific entry and exit price points
            4. Position sizing and portfolio diversification advice
            5. Your advise on this matter

            Format your response in clear, actionable bullet points. [/INST]</s>
            """
        )
    
    def generate_recommendation(self, stock, analysis):
        try:
            chain = LLMChain(llm=self.llm, prompt=self.recommendation_prompt)
            return chain.run(stock=stock, analysis=analysis)
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

class EnhancedTradingSystem:
    def __init__(self):
        self.stock_agent = StockAnalysisAgent()
        self.crypto_agent = CryptoAnalysisAgent()
        self.wallet_integration = WalletIntegration()
        self.recommendation_agent = RecommendationAgent()

    def analyze_asset(self, symbol, asset_type='stock'):
        if asset_type == 'stock':
            data = self.stock_agent.fetch_stock_data(symbol)
            agent = self.stock_agent
        else:  # crypto
            data = self.crypto_agent.fetch_crypto_data(symbol)
            agent = self.crypto_agent

        if data is None or len(data) < 60:
            return None

        X, y = agent.prepare_data(data)
        if len(X) == 0:
            return None

        agent.model.fit(X, y, batch_size=32, epochs=10, verbose=0)

        last_60 = data['Close' if asset_type == 'stock' else 'close'].values[-60:].reshape(-1, 1)
        last_60_scaled = agent.scaler.transform(last_60)

        next_day_pred = agent.model.predict(last_60_scaled.reshape(1, 60, 1))
        current_price = last_60[-1][0]
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
        recommendation = self.recommendation_agent.generate_recommendation(symbol, str(analysis))
        return {'analysis': analysis, 'recommendation': recommendation}
    
    
def main():
    st.set_page_config(page_title="Smart Trading Assistant", layout="centered")
    st.title("Smart Trading Assistant")

    st.sidebar.header("🔎 Select Asset Type")
    asset_type = st.sidebar.selectbox("Choose Asset Type", ["Stock", "Crypto"])

    # Popular symbol presets
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "NFLX", "META", "BABA"]
    popular_crypto = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "MATIC/USDT", "EUR/USD"]

    st.sidebar.markdown("### Select a symbol or type your own:")

    default_list = popular_stocks if asset_type == "Stock" else popular_crypto
    selected_symbol = st.sidebar.selectbox("Choose from popular symbols", options=default_list)
    custom_symbol = st.sidebar.text_input("Or type your own", value=selected_symbol)

    symbol = custom_symbol.upper()

    if st.sidebar.button("Analyze"):
        st.info("Fetching and analyzing data... Please wait.")
        system = EnhancedTradingSystem()
        result = system.get_recommendation(symbol, asset_type.lower())

        if isinstance(result, str):
            st.error(result)
        else:
            analysis = result['analysis']
            st.subheader("📊 Technical Summary")
            st.metric("Current Price", f"${analysis['current_price']:.2f}")
            st.metric("Predicted Price", f"${analysis['predicted_price']:.2f}")
            st.metric("Price Change", f"{analysis['price_change']:.2f}%")
            st.write(f"**Trend:** {analysis['trend']}")
            st.write(f"**Volume:** {analysis['volume']:,}")

            st.subheader("📌 AI Recommendation")
            st.markdown(result['recommendation'])

if __name__ == "__main__":
    main()