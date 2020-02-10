from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000, )

x_train = x_train.reshape(x_train.shape[0], 784).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float')/255

print(type(x_train))

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_test.shape) #(10000, 10)

model = Sequential()
model.add(Dense(32, input_shape=(784, ))) 
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics = accuracy(딱 떨어지는 값에 쓰임)
early_stopping = EarlyStopping(monitor = 'loss', patience=20)
model.fit(x_train, y_train, validation_split=0.7, epochs=100, batch_size=8, verbose=1, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)
print(acc)