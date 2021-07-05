#여러 X값을 이용하여 매출 예측하기

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam, SGD 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 

df = pd.read_csv('advertising.csv') 
x_data = np.array(df[['TV', 'Newspaper', 'Radio']], dtype=np.float32) 
y_data = np.array(df['Sales'], dtype=np.float32) 

x_data = x_data.reshape((-1, 3)) 
y_data = y_data.reshape((-1, 1)) 

print(x_data.shape) 
print(y_data.shape) 

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2021) 

print(x_train.shape, x_val.shape) 
print(y_train.shape, y_val.shape) 

model = Sequential([ Dense(1) ]) 
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1)) 
model.fit( x_train, y_train, validation_data=(x_val, y_val), # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증 epochs=100 # epochs 복수형으로 쓰기! )
