import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Model

# 1. data 
path = 'D:\study_603\dacon\_data/'

data_train = pd.read_csv(path + "train_data.csv")
data_test = pd.read_csv(path + "test_data.csv")

print(data_train.shape) # (45654, 3)
print(data_test.shape) # (9131, 2) 

x = data_train['title']
y = data_train['topic_idx']
x_pred = data_test['title']

# token, sequence
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)


# padding

print("x의 최대길이 : ", max(len(i) for i in x)) 
print("x의 평균길이 : ", sum(map(len, x)) / len(x)) 
print("x_pred의 최대길이 : ", max(len(i) for i in x_pred)) 
print("x_pred의 평균길이 : ", sum(map(len, x_pred)) / len(x_pred)) 
# x의 최대길이 :  13
# x의 평균길이 :  6.623954089455469
# x_pred의 최대길이 :  11
# x_pred의 평균길이 :  5.127696856861242

# preprocessing, to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x = pad_sequences(x, maxlen=10, padding='pre') 
x_pred = pad_sequences(x_pred, maxlen=10, padding='pre')
print(x.shape, x_pred.shape) # (45654, 10) (9131, 10)

word_size = len(token.word_index)
print(word_size) # 101081

# y to categorical
print(np.unique(y)) # [0 1 2 3 4 5 6]
y = to_categorical(y)
print(np.unique(y)) # [0. 1.]

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, test_size=0.2, random_state=66)

print(x_train.shape)

#2. model
from tensorflow.keras import models, Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
input = Input(shape=(10,)) 
em = Embedding(input_dim=101081, output_dim=77)(input)
em = LSTM(units=32, activation='relu')(em)
output = Dense(7, activation='softmax')(em)

model = Model(inputs=input, outputs=output)

model.summary()

# 3. compile, fit
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Save Module
####################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath = 'D:\study_603\dacon\_save\MCP/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "TEST_", date_time, "_", filename])
####################################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time 

start_time = time.time()
model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=2, validation_split=0.02)
end_time = time.time() - start_time

#4. Evaluate
loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('time : ', end_time)

#5. Predict
prediction = model.predict(x_pred)
prediction = np.argmax(prediction, axis=1) # to_categorical -> original

#6. submit
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131,)
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('D:\study_603\dacon\_data/new.csv,', header=['index', 'topic_idx'], index=False)

'''
loss: 0.0070 - acc: 0.9980 - val_loss: 1.1705 - val_acc: 0.7155
'''

'''
Errors may have originated from an input operation.
Input Source operations connected to node model/embedding/embedding_lookup:
 model/embedding/embedding_lookup/2746 (defined at C:\ProgramData\Anaconda3\lib\contextlib.py:113)

Input Source operations connected to node model/embedding/embedding_lookup:
 model/embedding/embedding_lookup/2746 (defined at C:\ProgramData\Anaconda3\lib\contextlib.py:113)

Function call stack:
test_function -> test_function
'''