from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
           [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print(x.shape) #(13, 3)
print(y.shape) #(13,)

x = x.reshape(x.shape[0], x.shape[1], 1) #(13, 3, 1)
print(x.shape)

model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape=(3, 1))) #1은 x를 1개씩 잘라서 연산하겠다는 뜻 (몇개씩 자르는지)
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x, y, epochs=100, batch_size = 1, validation_split = 0, verbose = 99) # verbose(장황한) : 표시(1)/제거(0)/막대제거(2)/epoch(3)

# 평가예측(loss : 4개(총 loss포함), mse : 3개(metrix))
aaa = model.evaluate(x, y, batch_size = 1)
print('aaa:', aaa)

loss, mse = model.evaluate(x, y, batch_size = 1)
print('loss, mae :', loss, mse)

x_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]]) #(3, 3)
print(x_input.shape)
x_input = x_input.reshape(4, 3, 1)

y_pred = model.predict(x_input)
print('y_pred', y_pred)