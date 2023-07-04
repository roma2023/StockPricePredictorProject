from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import datetime
import pandas as pd
import time
 


def getDataSet(SYMBOL, NUM_DAYS, INTERVAL):
    startTime = (datetime.date.today() - datetime.timedelta( NUM_DAYS-1 ) )
    endTime = datetime.datetime.today().date()
    start = time.mktime(startTime.timetuple())
    end= time.mktime(endTime.timetuple()) 
    url = 'https://finance.yahoo.com/quote/'+SYMBOL+'/history?period1=' + str(int(start)) + '&period2='+str(int(end))+'&interval='+INTERVAL+'&filter=history&includeAdjustedClose=true'
    minus = []

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options)
    browser.get(url)
    html = browser.page_source

    soup = BeautifulSoup(html, 'html.parser')

    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df = df[:-1]
    df = df[~df['Open'].str.contains('Dividend', regex=False)]
    df.rename(columns={"Close*": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open', 'Adj Close**': 'Adj Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open'] = df['open'].astype(float)
    df['Adj Close'] = df['Adj Close'].astype(float)
    return df.iloc[::-1]