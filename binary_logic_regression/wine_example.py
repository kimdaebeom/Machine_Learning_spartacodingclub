#import os
#os.environ['KAGGLE_USERNAME'] = 'username' # username
#os.environ['KAGGLE_KEY'] = 'key' # key
#!kaggle datasets download -d brynja/wineuci
#!unzip wineuci.zip

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('Wine.csv')

df = pd.read_csv('Wine.csv', names=[
'name'
,'alcohol'
,'malicAcid'
,'ash'
,'ashalcalinity'
,'magnesium'
,'totalPhenols'
,'flavanoids'
,'nonFlavanoidPhenols'
,'proanthocyanins'
,'colorIntensity'
,'hue'
,'od280_od315'
,'proline'
])
x_data = df.drop(columns=['name'], axis=1)
x_data = x_data.astype(np.float32)
y_data = df[['name']]
y_data = y_data.astype(np.float32)
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)
encoder = OneHotEncoder()
y_data_encoded = encoder.fit_transform(y_data).toarray()
x_train, x_val, y_train, y_val = train_test_split(x_data_scaled, y_data_encoded, test_size=0.2, random_state=2021)

model = Sequential([
Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.02), metrics=['acc'])
model.fit(
x_train,
y_train,
validation_data=(x_val, y_val),
epochs=20
)
