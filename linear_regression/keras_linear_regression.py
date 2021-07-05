import numpy as np 
from tensorflow.keras.models 
import Sequential from tensorflow.keras.layers 
import Dense from tensorflow.keras.optimizers 
import Adam, SGD 
x_data = np.array([[1], [2], [3]]) 
y_data = np.array([[10], [20], [30]]) 
model = Sequential([ Dense(1) ]) 
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1)) 
model.fit(x_data, y_data, epochs=100) # epochs 복수형으로 쓰기!
y_pred = model.predict([[4]]) 
print(y_pred)
