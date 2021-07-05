#환경변수 지정하기
# import os os.environ['KAGGLE_USERNAME'] = '[내_캐글_username]' # username os.environ['KAGGLE_KEY'] = '[내_캐글_key]' # key
#원하는 데이터셋의 API를 복사해 와 실행하기
# !kaggle datasets download -d ashydv/advertising-dataset
#데이터셋 압축 풀어주기
# !unzip /content/advertising-dataset.zip

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam, SGD 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split

df = pd.read_csv('advertising.csv') 
df.head(5)
print(df.shape)

sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars=['Sales'], height=4)

#데이터셋 가공하기

x_data = np.array(df[['TV']], dtype=np.float32) 
y_data = np.array(df['Sales'], dtype=np.float32) 
print(x_data.shape) 
print(y_data.shape)
x_data = x_data.reshape((-1, 1)) 
y_data = y_data.reshape((-1, 1)) 
print(x_data.shape) 
print(y_data.shape)

#데이터셋을 학습 데이터와 검증 데이터로 분할하기

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2021) 
print(x_train.shape, x_val.shape) 
print(y_train.shape, y_val.shape)

#학습시키기

model = Sequential([ Dense(1) ]) 
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1)) 
model.fit( x_train, y_train, validation_data=(x_val, y_val), 
# 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증 epochs=100 # epochs 복수형으로 쓰기! )

#검증 데이터로 예측하기

y_pred = model.predict(x_val) 
plt.scatter(x_val, y_val) 
plt.scatter(x_val, y_pred, color='r') 
plt.show()
