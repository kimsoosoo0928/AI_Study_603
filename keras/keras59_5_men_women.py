# 실습 1.
# men, women 데이터로 모델링을 구성할 것!

# 실습 2. 과제
# 본인 사진으로 predict 하시오 ! d:\data 안에 사진을 넣고 

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

men = train_datagen.flow_from_directory(
    '../data/men_women/men',
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary',
    shuffle=True
)


women = test_datagen.flow_from_directory(
    '../data/men_women/women',
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary'
)

