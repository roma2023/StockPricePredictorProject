import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score  
import Scraper
import train
import warnings
warnings.filterwarnings(action='ignore')

class SMP:
    def __init__(self, numdays=100, interval='1d', symbol = 'SPY', stationarity = False, window=15):
        self.numdays = numdays
        self.interval = interval
        self.symbol = symbol
        self.stationarity = stationarity
        self.window = window
        self.live_pred_data = []
        self.bestGlobalModel =  None
        self.globalGlobalAccuracy = 0
        # List of symbols for technical indicators
        self.INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']
        self.start = (datetime.date.today() - datetime.timedelta(self.numdays))
        self.end = datetime.datetime.today()
        self.data = Scraper.getDataSet(symbol,numdays,interval)
        self.data = yf.download(symbol, start=self.start, end=self.end, interval=self.interval)
        try:
            self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
        except:
            pass
    def outliers(self):
        # calculate the IQR for each column
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1

        # remove outliers from each column
        self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
    def makeStationary(self):
        if(self.stationarity == True):
            df_stationary = pd.DataFrame()
            # Loop through each column and make it stationary with STL
            for col in self.data.columns:
                # Perform STL decomposition
                result = STL(self.data[col], seasonal=13, period=100).fit()

                # Obtain detrended data
                detrended_data = self.data[col] - result.trend
                df_stationary[col] = detrended_data - result.seasonal
                df_stationary.dropna(inplace=True) # Remove NaN values

                # Obtain deseasonalized data
                deseasonal_data = self.data[col] - result.seasonal
            self.data = df_stationary
    
    def smoothData(self, alpha=0.65):
        self.data = self.data.ewm(alpha=alpha).mean()

    def getIndicatorData(self):
        """
        Function that uses the finta API to calculate technical indicators used as the features
        :return:
        """
        for indicator in self.INDICATORS:
            ind_data = eval('TA.' + indicator + '(self.data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            self.data = self.data.merge(ind_data, left_index=True, right_index=True)
        self.data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        self.data['ema50'] = self.data['close'] / self.data['close'].ewm(50).mean()
        self.data['ema21'] = self.data['close'] / self.data['close'].ewm(21).mean()
        self.data['ema15'] = self.data['close'] / self.data['close'].ewm(14).mean()
        self.data['ema5'] = self.data['close'] / self.data['close'].ewm(5).mean()

        # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        self.data['normVol'] = self.data['volume'] / self.data['volume'].ewm(5).mean()

        # Remove columns that won't be used as features
        del (self.data['open'])
        del (self.data['high'])
        del (self.data['low'])
        del (self.data['volume'])
        del (self.data['Adj Close'])
        self.data = self.data.dropna()
    def producePrediction(self):
        """
        Function that produces the 'truth' values
        At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
        :param window: number of days, or rows to look ahead to see what the price did
        """
        self.live_pred_data = self.data.iloc[-self.window:]
        prediction = (self.data.shift(-self.window)['close'] >= self.data['close'])
        prediction = prediction.iloc[:-self.window]
        self.data['pred'] = prediction.astype(int)
        del(self.data['close'])
        self.data = self.data.dropna() 
        del(self.live_pred_data['close'])
    def getModelAccuracy(self):
        self.bestGlobalModel, self.bestGlobalAccuracy= train.cross_Validation_ADA(self.data)
        print("Model successfuly generated with precision ", self.bestGlobalAccuracy,"!")
    def makePrediction(self):
        predictions = self.bestGlobalModel.predict(self.live_pred_data)
        return self.bestGlobalAccuracy, predictions
        
        