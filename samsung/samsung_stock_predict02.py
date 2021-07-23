import pandas as pd
import numpy as np

# 1. 데이터

filepath = './_data/'
fname_ss = '삼성전자 주가 20210721.csv'
fname_sk = 'SK주가 20210721.csv'

df_ss = pd.read_csv(filepath + fname_ss, encoding='cp949')

# print(df_ss)
# print(df_ss.columns) # ['일자', '시가', '고가', '저가', '종가', '종가 단순 5', '10', '20', '60', '120', '거래량','단순 5', '20.1', '60.1', '120.1', 'Unnamed: 15]'
# print(df_ss.loc[:,['일자', '시가', '고가', '저가', '종가', '거래량']])

# df_ss.drop(df_ss.index[2601:3601], inplace=True)

dropdate = df_ss[ (df_ss['일자'] < '2011/01/03')].index
df_ss.drop(dropdate, inplace=True)
df_ss.drop(df_ss.index[2601], inplace=True)

df_ss = df_ss.sort_values(by='일자', axis=0)
df_ss = df_ss.reset_index()
df_ss = df_ss.loc[:,['시가', '고가', '저가', '거래량', '종가']]


# print(df_ss['종가']) # (2601, 5)

df_sk = pd.read_csv(filepath + fname_sk, encoding='cp949')

dropdate = df_sk[ (df_sk['일자'] < '2011/01/03')].index
df_sk.drop(dropdate, inplace=True)
df_sk.drop(df_sk.index[2601], inplace=True)

df_sk = df_sk.sort_values(by='일자', axis=0)
df_sk = df_sk.reset_index()
df_sk = df_sk.loc[:,['시가', '고가', '저가', '거래량', '종가']]

x1 = df_ss
x2 = df_sk

# 1_1. dataframe_to_numpy.array

x1 = x1.to_numpy()
x2 = x2.to_numpy()

# print(x1.shape) # (2601, 5)

size = 5

def split_x(a, num):
    aaa = []
    for i in range(len(a) - num + 1 ): 
        subset = a[i : (i + num )] 
        aaa.append(subset)
    return np.array(aaa)

samsung = split_x(x1, size)
sk = split_x(x2, size)

x1_predict = samsung[[-1,-2]]
x2_predict = sk[[-1,-2]]

print(x1_predict.shape, x2_predict.shape)

samsung = np.delete(samsung,[-1,-2],0)
sk = np.delete(sk,[-1,-2],0)

print(samsung.shape) # (2597, 5, 5)
print(sk.shape) # (2597, 5, 5)

y = x1[:,4]
y = np.delete(y,[0,1,2,3,4,5],0)
print(y.shape)

# 1_3. train_test_split

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(samsung, sk, y,
                        train_size=0.8, random_state=66)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM , Conv1D, Input

input1 = Input(shape=(5, 5))
qq = Conv1D(64, kernel_size=5)(input1)
qq = Dense(64, activation='relu')(qq)
qq = Dense(128, activation='relu')(qq)
qq = Dropout(0.2)(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dropout(0.2)(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dense(128, activation='relu')(qq)
qq = Dense(64, activation='relu')(qq)
output1 = Dense(16, activation='relu')(qq)

input2 = Input(shape=(5, 5))
qq = Conv1D(64, kernel_size=5)(input2)
qq = Dense(64, activation='relu')(qq)
qq = Dense(128, activation='relu')(qq)
qq = Dropout(0.2)(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dropout(0.2)(qq)
qq = Dense(256, activation='relu')(qq)
qq = Dense(64, activation='relu')(qq)
qq = Dense(32, activation='relu')(qq)
output2 = Dense(16, activation='relu')(qq)

from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2])
qq = Dense(32, activation='relu')(merge1)
qq = Dense(16, activation='relu')(qq)
qq = Dense(8, activation='relu')(qq)
last_output = Dense(1)(qq)

model = Model(inputs=[input1, input2], outputs=last_output)

import time

start_time = time.time()
filepath = './_save/samsung/day1/'
fname = 'samsung0723_0259_.0011-47753212.0000.hdf5'
model = load_model(filepath + fname)
end_time = time.time() - start_time

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

######################################################################

import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/samsung/day1/'
filename = '.{epoch:04d}-{loss:.4f}.hdf5'
modelpath = "".join([filepath, "samsung", date_time, "_", filename])

######################################################################

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                filepath=modelpath)

model.compile(loss='mse', optimizer='adam')

# start_time = time.time()
# model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=4, verbose=1,
#              callbacks=[es, mcp], validation_split=0.1)
# end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y_test)
results = model.predict([x1_predict, x2_predict])

print('소요 시간 : ', end_time)
print('loss : ', loss)
print('예상주가 :', results[0])