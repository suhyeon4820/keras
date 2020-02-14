from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
print(y_train.shape) #(50000, 1)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 96)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 96)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)

model = Sequential()
model.add(Conv2D(32, (2, 2), padding ='same', input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (2, 2)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (2, 2)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics = accuracy(딱 떨어지는 값에 쓰임)
early_stopping = EarlyStopping(monitor = 'loss', patience=20)
model.fit(x_train, y_train, validation_split=0.7, epochs=100, batch_size=8, verbose=1, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)
print(acc)