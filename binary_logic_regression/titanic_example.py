#import os
#os.environ['KAGGLE_USERNAME'] = 'username' # username
#os.environ['KAGGLE_KEY'] = 'key' # key
#!kaggle datasets download -d heptapod/titanic
#!unzip titanic.zip


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train_and_test2.csv')

df = pd.read_csv('train_and_test2.csv', usecols=[
'Age', # 나이
'Fare', # 승차 요금
'Sex', # 성별
'sibsp', # 타이타닉에 탑승한 형제자매, 배우자의 수
'Parch', # 타이타니게 탑승한 부모, 자식의 수
'Pclass', # 티켓 등급 (1, 2, 3등석)
'Embarked', # 탑승국
'2urvived' # 생존 여부 (0: 사망, 1: 생존)
])

df = df.dropna()

x_data = df.drop(columns=['2urvived'], axis=1)
x_data = x_data.astype(np.float32)
y_data = df[['2urvived']]
y_data = y_data.astype(np.float32)

scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2021)

model = Sequential([
Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
model.fit(
x_train,
y_train,
validation_data=(x_val, y_val), # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
epochs = 20
)
