
#import os
#os.environ['KAGGLE_USERNAME'] = 'username' # username
#os.environ['KAGGLE_KEY'] = 'key' # key

#!kaggle datasets download -d moltean/fruits
#!unzip -q fruits.zip


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

train_datagen = ImageDataGenerator(
rescale=1./255, # 일반화
rotation_range=10, # 랜덤하게 이미지를 회전 (단위: 도, 0-180)
zoom_range=0.1, # 랜덤하게 이미지 확대 (%)
width_shift_range=0.1, # 랜덤하게 이미지를 수평으로 이동 (%)
height_shift_range=0.1, # 랜덤하게 이미지를 수직으로 이동 (%)
horizontal_flip=True # 랜덤하게 이미지를 수평으로 뒤집기
)
test_datagen = ImageDataGenerator(
rescale=1./255 # 일반화
)
train_gen = train_datagen.flow_from_directory(
'fruits-360/Training',
target_size=(224, 224), # (height, width)
batch_size=32,
seed=2021,
class_mode='categorical',
shuffle=True
)
test_gen = test_datagen.flow_from_directory(
'fruits-360/Test',
target_size=(224, 224), # (height, width)
batch_size=32,
seed=2021,
class_mode='categorical',
shuffle=False
)

from pprint import pprint
pprint(train_gen.class_indices)
preview_batch = train_gen.__getitem__(0)
preview_imgs, preview_labels = preview_batch
plt.title(str(preview_labels[0]))
plt.imshow(preview_imgs[0])

from tensorflow.keras.applications.inception_v3 import InceptionV3
input = Input(shape=(224, 224, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input, pooling='max')
x = base_model.output
x = Dropout(rate=0.25)(x)
x = Dense(256, activation='relu')(x)
output = Dense(131, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

from tensorflow.keras.models import load_model
model = load_model('model.h5')

test_imgs, test_labels = test_gen.__getitem__(100)
y_pred = model.predict(test_imgs)
classes = dict((v, k) for k, v in test_gen.class_indices.items())
fig, axes = plt.subplots(4, 8, figsize=(20, 12))
for img, test_label, pred_label, ax in zip(test_imgs, test_labels, y_pred, axes.flatten()):
test_label = classes[np.argmax(test_label)]
pred_label = classes[np.argmax(pred_label)]
ax.set_title('GT:%s\nPR:%s' % (test_label, pred_label))
ax.imshow(img)
