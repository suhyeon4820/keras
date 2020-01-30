from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y1 = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], [50, 60, 70],
           [60, 70, 80], [70, 80, 90], [80, 90, 100], [90, 100, 110], [100, 110, 120], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
#y2 = array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 5, 6, 7])
y2 = array([40, 50, 60, 70, 80, 90, 100, 110, 5, 6, 7, 8, 9])

print(x1.shape) #(13, 3)
print(y1.shape) #(13,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 3, 1)
print(x1.shape)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
print(x1.shape)

# model_1
input1= Input(shape = (3, 1))
dense1 = LSTM(20, activation = 'relu')(input1)
dense1 = Dense(18)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(14)(dense1)
dense1 = Dense(12)(dense1)
output1 = Dense(1)(dense1)

# model_2
input2= Input(shape = (3, 1))
dense2 = LSTM(20, activation = 'relu')(input2)
dense2 = Dense(18)(dense2)
dense2 = Dense(16)(dense2)
dense2 = Dense(14)(dense2)
dense2 = Dense(12)(dense2)
output2 = Dense(1)(dense2)

from keras.layers import Concatenate, concatenate, Add
#merge1 = concatenate([output1, output2])
merge1 = Add()([output1, output2])
middle1 = Dense(7)(merge1)
middle1 = Dense(7)(middle1)
output3 = Dense(1)(middle1)

# outout_1
output1 = Dense(4)(output3)
middle1 = Dense(7)(output1)
middle1 = Dense(7)(middle1)
output_1 = Dense(1)(middle1)

# output_2
output2 = Dense(4)(output3)
middle2 = Dense(7)(output2)
middle2 = Dense(7)(middle2)
output_2 = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = [output_1, output_2])
model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') # monitor=loss(mode=auto/min), monitor=accuracy(mode=max)
model.fit([x1, x2], [y1, y2], epochs=10000, batch_size = 1, validation_split = 0, verbose = 1, callbacks=[early_stopping]) # verbose(장황한) : 표시(1)/제거(0)/막대제거(2)/epoch만(3이상)

# 평가예측(loss : 4개(총 loss포함), mse : 3개(metrix))
aaa = model.evaluate([x1, x2], [y1, y2], batch_size = 1)
print('aaa:', aaa)


loss = model.evaluate([x1, x2], [y1, y2], batch_size = 1)
print('loss:', loss)


x1_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) #(3, 3)
x2_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) #(3, 3)
print(x1_input.shape)
x1_input = x1_input.reshape(4, 3, 1)
x2_input = x2_input.reshape(4, 3, 1)

y_pred = model.predict([x1_input, x2_input])
print('y_pred', y_pred)