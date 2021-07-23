import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler
from tensorflow.python.keras.engine.training import Model

### 1.데이터 ###

# read_csv
datasets_samsung =  pd.read_csv('D:\study\samsung\삼성전자 주가 20210721.csv',header=0 ,usecols=['시가','고가', '저가', '종가', '거래량'], nrows=2602 ,encoding='cp949')
datasets_sk = pd.read_csv('D:\study\samsung\SK주가 20210721.csv', header=0, usecols=['시가','고가', '저가', '종가', '거래량'], nrows=2602 ,encoding='cp949')

# sort
datasets_samsung_sort = datasets_samsung.sort_index(ascending=False)
datasets_sk_sort = datasets_sk.sort_index(ascending=False)

# scaling - samsung
data1_ss = datasets_samsung_sort.iloc[:, :-1]
data2_ss = datasets_samsung_sort.iloc[:, -1:]

scaler = MinMaxScaler()
data1_ss = scaler.fit_transform(data1_ss)
data2_ss = scaler.fit_transform(data2_ss)

data_ss = np.concatenate([data1_ss, data2_ss], axis=1)

print(data_ss)

# scaling - sk
data1_sk = datasets_sk_sort.iloc[:, :-1]
data2_sk = datasets_sk_sort.iloc[:, -1:]

scaler = MinMaxScaler()
data1_sk = scaler.fit_transform(data1_sk)
data2_sk = scaler.fit_transform(data2_sk)

data_sk = np.concatenate([data1_sk, data2_sk], axis=1)
print(data_sk)


# target
target = data_ss[:, [-2]] # 삼성의 종가

# split function


x1 = [] # 삼성
x2 = [] # SK
y = [] # target
size = 5
for i in range(len(target) - size + 1):
    x1.append(data_ss[i: (i + size) ])
    x2.append(data_sk[i: (i + size) ])
    y.append(target[i + (size - 1)]) 
x1_pred = [data_ss[len(data_ss) - size : ]]
x2_pred = [data_sk[len(data_sk) - size : ]]

# numpy 배열화
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)
print(x1.shape, x2.shape, y.shape, x1_pred.shape, x2_pred.shape) # (2553, 50, 5) (2553, 50, 5) (2553, 1) (1, 50, 5) (1, 50, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, shuffle='false', random_state=66)


### 2.모델링 ###
 
#2-1. 모델1
input1 = Input(shape=(50,1)) 
xx = LSTM(units=20, activation='relu', input_shape=(50, 1))(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
output1 = Dense(10)(xx)

#2-2. 모델2
input2 = Input(shape=(50,1))
xx = LSTM(units=20, activation='relu', input_shape=(50, 1))(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
output2 = Dense(10)(xx)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2]) 
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

### 3.컴파일, 훈련 ###
model.compile(loss='mse', optimizer='adam')

# Save Module
####################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath = './_save/ModelCheckPoint/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k47_", date_time, "_", filename])
####################################################################


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                    restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath=filepath)
import time
start_time = time.time()
model.fit([x1_train, x2_train], y, epochs=100, batch_size=1, callbacks=[es, mcp])
end_time = time.time() - start_time

### 4.평가, 예측 ###
loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_pred, x2_pred])
y_pred = scaler.inverse_transform(y_pred)

print('loss = ', loss)
print("predict = ", y_pred)
print('time : ', end_time)


'''
ValueError: Data cardinality is ambiguous:
  x sizes: 2042, 2042
  y sizes: 2553
'''




