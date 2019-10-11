import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import pandas as pd

store_data = pd.read_csv('Drinks.csv', sep=',')
# print(store_data)
scaled_train_samples = []
train_labels = []
train_labels.append([str(store_data.values[0,j]) for j in range(1, 13)])

for i in range(0, 177):
    scaled_train_samples.append(str(store_data.values[i,j]) for j in range(1, 13))
    train_labels.append(str(store_data.values[i, j]) for j in range(13, 13))
layers = [
    Dense(8, input_shape=(13,), activation='tanh'),
    Dense(5, activation='tanh'),
    Dense(3, activation='softmax')
]

model = Sequential(layers)

a = model.compile(
    Adam(lr=.7),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    scaled_train_samples,
    train_labels,
    batch_size=10,
    epochs=20,
    shuffle=True,
    verbose=3
)