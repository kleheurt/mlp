import pandas as pd
import ta as ta

#Imports data from csv to pandas dataFrame
def get_data():
    dataset = r".\data.csv"
    dataset = pd.read_csv(dataset, sep=",")
    return dataset

#Produces the indicator vectors used by Nakano et al
#Using Technical Analysis library as ta
def get_indicators(dataset):
    ret = ta.others.DailyReturnIndicator(close = dataset['close'])
    ema2 = ta.trend.EMAIndicator(window = 2, close= dataset['close'])
    ema4 = ta.trend.EMAIndicator(window = 4, close= dataset['close'])
    ema12 = ta.trend.EMAIndicator(window = 12, close= dataset['close'])
    ema24 = ta.trend.EMAIndicator(window = 24, close= dataset['close'])
    rsi12 = ta.momentum.RSIIndicator(close = dataset['close'], window = 12)
    rsi24 = ta.momentum.RSIIndicator(close = dataset['close'], window = 24)
    rsi48 = ta.momentum.RSIIndicator(close = dataset['close'], window = 48)

    ret = pd.Series(ret.daily_return())
    ema2 = pd.Series(ema2.ema_indicator())
    ema4 = pd.Series(ema4.ema_indicator())
    ema12 = pd.Series(ema12.ema_indicator())
    ema24 = pd.Series(ema24.ema_indicator())
    rsi12 = pd.Series(rsi12.rsi())
    rsi24 = pd.Series(rsi24.rsi())
    rsi48 = pd.Series(rsi48.rsi())

    processed_data = pd.DataFrame({ "Return":ret,
                                    "EMA2":ema2,
                                    "EMA4":ema4,
                                    "EMA12":ema12,
                                    "EMA24":ema24,
                                    "RSI12":rsi12,
                                    "RSI24":rsi24,
                                    "RSI48":rsi48
                                    })

    return processed_data[48:] #Dropping NaN values derived from indicator calculations
     
raw_data = get_data()
raw_data = raw_data[["close"]].astype("float64")
ready_data = get_indicators(raw_data)
print(ready_data)

ready_data.to_csv(path_or_buf=r".\processed_data.csv")