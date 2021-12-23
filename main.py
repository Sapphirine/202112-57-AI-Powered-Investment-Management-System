#import libraries
import sqlalchemy
import pymysql
import ta
import pandas as pd
import numpy as np
import requests
import tweepy
import nltk
import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from nltk.corpus import stopwords
from textblob import Word, TextBlob


TWITTER_CONSUMER_KEY = 'credentials'
TWITTER_CONSUMER_SECRET = 'credentials'
TWITTER_ACCESS_TOKEN = 'credentials'
TWITTER_ACCESS_TOKEN_SECRET = 'credentials'


auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

pymysql.install_as_MySQLdb()
engine = sqlalchemy.create_engine('mysql://root:12345678@localhost:3306/')


#Stock recommendation class
class StocksRecommender:
    engine = sqlalchemy.create_engine('mysql://root:12345678@localhost:3306/')

    def __init__(self, index):
        self.index = index

    def gettables(self):
        query = f"""SELECT table_name FROM information_schema.tables WHERE table_schema = '{self.index}'"""
        df = pd.read_sql(query, self.engine)
        df['Schema'] = self.index
        return df

    def getprices(self):
        prices = []
        for table, schema in zip(self.gettables().TABLE_NAME, self.gettables().Schema):
            sql = schema +'.'+ f'`{table}`'
            prices.append(pd.read_sql(f"SELECT Date, Open, High, Low, Close FROM {sql}", self.engine))
        return prices

    def MACDdecision(self, df):#Technical indicator: MACD decision
        df['MACD_diff'] = ta.trend.macd_diff(df.Close)
        df['Decision MACD'] = np.where((df.MACD_diff > 0) & (df.MACD_diff.shift(1) < 0), True, False)

    def Goldencrossdecision(self, df): #Technical indicator: golden cross decision
        df['SMA50'] = ta.trend.sma_indicator(df.Close, window = 50)
        df['SMA200'] = ta.trend.sma_indicator(df.Close, window = 200)
        df['Signal'] = np.where(df['SMA50'] > df['SMA200'], True, False)
        df['Decision GC'] = df.Signal.diff()

    def RSI_SMAdecision(self, df):#Technical indicator: RSI decision
        df['RSI'] = ta.momentum.rsi(df.Close, window = 6)
        df['Decision RSI/SMA'] = np.where((df.Close > df.SMA200) & (df.RSI < 30), True, False)

    def applytechnicals(self):
        prices = self.getprices()
        for frame in prices:
            self.MACDdecision(frame)
            self.Goldencrossdecision(frame)
            self.RSI_SMAdecision(frame)
        return prices

    def recommender(self):
        buy_stocks = []
        indicators = ['Decision MACD', 'Decision GC', 'Decision RSI/SMA']
        for symbol, frame in zip(self.gettables().TABLE_NAME, self.applytechnicals()):
            if frame.empty is False:
                for indicator in indicators:
                    if frame[indicator].iloc[-1] == True:
                        buy_stocks.append(symbol)
        return tuple(buy_stocks)


sp500 = StocksRecommender('SP500')
stocks_for_buying = sp500.recommender()
risk_preference = ('High', 'Medium', 'Low', 'No choice')


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Intelligent Investment Management')


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.sidebar.title('Current Portfolio') #current portfolio dialogbox
a = st.sidebar.button('AAPL')
if a == True:
    data_X = load_data('AAPL')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()


b = st.sidebar.button('MSFT')
if b == True:
    data_X = load_data('MSFT')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()


c = st.sidebar.button('GOOGL')
if c == True:
    data_X = load_data('GOOGL')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()


d = st.sidebar.button('SBUX')
if d == True:
    data_X = load_data('SBUX')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

e = st.sidebar.button('TSLA')
if e == True:
    data_X = load_data('TSLA')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

f = st.sidebar.button('FB')
if f == True:
    data_X = load_data('FB')
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data_X['Date'], y=data_X['Close'], name="stock_close"))
        fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()



st.sidebar.title('Recommended S&P 500 Stock')
stock_buying_option = st.sidebar.selectbox('', stocks_for_buying)
option = st.sidebar.selectbox('Chart Type', ('Candle Stick', 'Line Chart', 'Stocktwits', 'Forecasting', 'Twitter')) #Drop down for sidebar


n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365


data = load_data(stock_buying_option) #load data
if option == 'Stocktwits':

    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{stock_buying_option}.json")

    data_stockwits = r.json()

    for message in data_stockwits['messages']:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])


if option == 'Line Chart': #line chart visualisation
    data_line_chart = data
    def plot_raw_data():
        fig = go.Figure()
    	fig.add_trace(go.Scatter(x=data_line_chart['Date'], y=data_line_chart['Open'], name="stock_open"))
    	fig.add_trace(go.Scatter(x=data_line_chart['Date'], y=data_line_chart['Close'], name="stock_close"))
    	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)

    plot_raw_data()

    data_forecast = data
    df_train = data_forecast[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)


if option == 'Forecasting': #forecasting visualisation
    data_forecast = data
    df_train = data_forecast[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)


if option == 'Candle Stick': #Candle stick visualisation
    symbol = stock_buying_option

    sql = 'SP500' +'.'+ f'`{stock_buying_option}`'

    data_cs = pd.read_sql(f"SELECT Date, Open, High, Low, Close FROM {sql}", engine)
    data_cs = data_cs.iloc[-100:]
    print(data_cs)


    fig_cs = go.Figure(data=[go.Candlestick(x=data_cs['Date'],
                    open=data_cs['Open'],
                    high=data_cs['High'],
                    low=data_cs['Low'],
                    close=data_cs['Close'],
                    increasing_line_color = 'green',
                    decreasing_line_color = 'red')])



    fig_cs.update_layout(height=800)

    st.plotly_chart(fig_cs, use_container_width=True)

    st.write(data_cs)


if option == 'Twitter': #Twitter sentiment
    query = tweepy.Cursor(api.search_tweets, q=stock_buying_option, lang = 'en').items(1000)
    tweets = [{'Tweet':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]
    df = pd.DataFrame.from_dict(tweets)
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = stopwords.words('english')
    custom_stopwords = ['RT']
    def preprocess_tweets(tweet, custom_stopwords):
        processed_tweet = tweet
        processed_tweet.replace('[^\w\s]', '')
        processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
        processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
        processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
        return(processed_tweet)

    df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x, custom_stopwords))
    df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
    df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])
    stock = df[['Timestamp', 'polarity']]
    stock = stock.sort_values(by='Timestamp', ascending=True)
    stock['MA Polarity'] = stock.polarity.rolling(10, min_periods=3).mean()


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock['Timestamp'], y=stock['MA Polarity'], name="Sentiment"))
    fig1.layout.update(title_text='Twitter Sentiment', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

left, right, w = st.columns([0.5,1,1.5])

#Buy/sell/watch options
with left:
    buy = st.sidebar.button('Buy')

with right:
    sell = st.sidebar.button('Sell')

with w:
    watch = st.sidebar.button('Watch')
