from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = array([4, 5, 6, 7, 8])
print(x.shape) #(5, 3)
print(y.shape) #(5,)

x = x.reshape(x.shape[0], x.shape[1], 1) #(5, 3, 1)
print(x.shape)

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3, 1))) #1은 x를 1개씩 잘라서 연산하겠다는 뜻 (몇개씩 자르는지)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x, y, epochs=400, batch_size = 1, validation_split = 0) 

# 평가예측(loss : 4개(총 loss포함), mse : 3개(metrix))
aaa = model.evaluate(x, y, batch_size = 1)
print('aaa:', aaa)

loss, mse = model.evaluate(x, y, batch_size = 1)
print('loss, mae :', loss, mse)

x_input = array([6, 7, 8])  #(3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(1, 3, 1)

y_pred = model.predict(x_input)
print('y_pred', y_pred)

