from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

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
model.add(LSTM(2, activation='relu', return_sequences=True)) 
model.add(LSTM(3, activation='relu', return_sequences=True)) 
model.add(LSTM(4, activation='relu', return_sequences=True)) 
model.add(LSTM(5, activation='relu', return_sequences=True)) 
model.add(LSTM(5, activation='relu', return_sequences=True)) 
model.add(LSTM(30, activation='relu', return_sequences=False)) # return_sequences=False : default값
model.add(Reshape((15, 2)))  ### 정말 중요! 위 hidden layer의 노드수와 합이 맞아야 함
model.add(LSTM(20, activation = 'relu', return_sequences=False)) # Whether to return the last output in the output sequence, or the full sequence.
model.add(Dense(16)) #activation default = linear
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') # monitor=loss(mode=auto/min), monitor=accuracy(mode=max)
model.fit(x, y, epochs=100, batch_size = 1, validation_split = 0, verbose = 1, callbacks=[early_stopping]) # verbose(장황한) : 표시(1)/제거(0)/막대제거(2)/epoch만(3이상)

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