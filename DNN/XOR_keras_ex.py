import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD

input = Input(shape=(2,))
hidden = Dense(8, activation='relu')(input)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
model.summary()
model.fit(x_data, y_data, epochs=1000, verbose=0)

y_pred = model.predict(x_data)
