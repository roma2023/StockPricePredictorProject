from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
from openpyxl import Workbook
import os
import time
from itertools import islice
def getDF():
    url = 'https://finance.yahoo.com/trending-tickers'
    minus = [4, 8, 9, 10, 11]

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options)
    browser.get(url)
    try:
        browser.find_element(By.NAME, "agree").click()
    except:
        pass
    html = browser.page_source

    soup = BeautifulSoup(html, 'html.parser')

    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df.drop(df.iloc[:, 7:], inplace=True, axis=1)
    df['Symbol'] = df['Symbol'].astype(str)
    df['Name'] = df['Name'].astype(str)
    df['Last Price'] = df['Last Price'].astype(float)
    df['Market Time'] = df['Market Time'].astype(str)
    df['Change'] = df['Change'].astype(float)
    df['% Change'] = df['% Change'].str.strip('%').astype(float)
    df['Volume'] = df['Volume'].astype(str)
    return df