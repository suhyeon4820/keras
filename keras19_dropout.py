from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, BatchNormalization

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print(x.shape) #(13, 3)
print(y.shape) #(13,)

x = x.reshape(x.shape[0], x.shape[1], 1) #(13, 3, 1)
print(x.shape)


model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))  # True -> output(2차원)을 3차원으로 변환
model.add(Dense(100))
model.add(Dense(100))
model.add(Dropout(0.2)) #가중치에서 불필요한 것 제거
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5, activation = 'linear')) #activation default = linear
model.add(Dense(1))

model.summary()

# # 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용

# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor = 'loss', patience = 40, mode = 'min') # monitor=loss(mode=auto/min), monitor=accuracy(mode=max)
# model.fit(x, y, epochs=10000, batch_size = 1, validation_split = 0, verbose = 1, callbacks=[early_stopping]) # verbose(장황한) : 표시(1)/제거(0)/막대제거(2)/epoch만(3이상)

# # 평가예측(loss : 4개(총 loss포함), mse : 3개(metrix))
# aaa = model.evaluate(x, y, batch_size = 1)
# print('aaa:', aaa)

# loss, mse = model.evaluate(x, y, batch_size = 1)
# print('loss, mae :', loss, mse)

# x_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) #(3, 3)
# print(x_input.shape)
# x_input = x_input.reshape(4, 3, 1)

# y_pred = model.predict(x_input)
# print('y_pred', y_pred)

tensorboard --logdir=./graph