import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import time
import numpy as np
import datetime

# data

# samsung, sk 필요 데이터 추출
samsung = pd.read_csv('D:\study\samsung\삼성전자 주가 20210721.csv', sep=',',index_col='일자', 
header=0,engine='python', encoding='cp949')
samsung = samsung[['시가','고가','저가','종가', '거래량']]
samsung = samsung.sort_values(by='일자', ascending=True)
samsung = samsung.query('"2011/01/03"<= 일자 <= "2021/07/21"')
print(samsung)

sk = pd.read_csv('D:\study\samsung\SK주가 20210721.csv', sep=',', index_col='일자', header=0,
engine='python', encoding='CP949')
sk = sk[['시가','고가','저가','종가','거래량']]
sk = sk.sort_values(by='일자', ascending=False)
sk = sk.query('"2011/01/03"<= 일자 <= "2021/07/21"')
print(sk)

# 판다스 -> 넘파이 변환

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): 
        subset = dataset[i : (i+size)] 
        aaa.append(subset)
    return np.array(aaa)

size = 5

samsung = split_x(samsung, size) 
samsung = print(samsung.shape) # (2597, 5, 5)
samsung.reshape(2597*5, 5)
sk = split_x(sk, size)
sk = sk.reshape(2597*5, 5)

x1 = samsung[:10000]
print(x1)
x2 = sk[:10000]

