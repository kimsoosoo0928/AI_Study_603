'''
커맨드창에서
cd \
cd study_603
cd _save
cd _graph
dir/w
tensorboard --logdir=.

웹을 키고
http://127.0.0.1:6006 or
http://localhost:6006/
'''












from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([1,2,3,4,5]) 
y = np.array([1,2,4,3,5])

#2. 모델
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=1)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

from tensorflow.keras.callbacks import TensorBoard

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                write_graph=True, write_images=True)


model.fit(x, y, epochs=50 batch_size=1, callbacks=[tb], val)

#4. 평가, 예측
loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)

result = model.predict([6]) 
print('6의 예측값 : ', result) 
