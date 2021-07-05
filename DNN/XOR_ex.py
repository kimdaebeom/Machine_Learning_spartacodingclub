import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 이진 논리 회귀

model = Sequential([
Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
model.fit(x_data, y_data, epochs=1000, verbose=0)
y_pred = model.predict(x_data)

# MLP

model = Sequential([
Dense(8, activation='relu'),
Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
model.fit(x_data, y_data, epochs=1000, verbose=0)
y_pred = model.predict(x_data)
