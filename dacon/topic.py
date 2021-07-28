import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential

#read_csv
train = pd.read_csv('D:/study_603/dacon/_data/train_data.csv')
test = pd.read_csv("D:\study_603\dacon\_data/test_data.csv")
topic_dict = pd.read_csv("D:\study_603\dacon\_data/topic_dict.csv")

#preprocessing
def clean_text(sent):
  sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
  return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()
train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')

# npy save
np.save('D:\study_603\_save\_npy/ntg_x.npy', arr=x_data)
np.save('./_save/_npy/NTG_y.npy', arr=y_data)
np.save('./_save/_npy/NTG_x_pred.npy', arr=x_pred)

#model
def dnn_model():
  model = Sequential()
  model.add(Dense(128, input_dim = 150000, activation = "relu"))
  model.add(Dropout(0.8))
  model.add(Dense(7, activation = "softmax"))
  return model

model = dnn_model()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.optimizers.Adam(0.001), metrics = ['accuracy'])

history = model.fit(x = train_tf_text[:40000], y = train_label[:40000],
                    validation_data =(train_tf_text[40000:], train_label[40000:]),
                    epochs = 4)

tmp_pred = model.predict(test_tf_text)
pred = np.argmax(tmp_pred, axis = 1)
