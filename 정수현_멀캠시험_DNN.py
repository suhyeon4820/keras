import numpy as np
from numpy import array
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

data_samsung = pd.read_csv('samsung.csv',encoding='cp949', thousands = ',')
print(data_samsung.shape) #(426, 6)

x_data = data_samsung[['시가', '고가', '저가', '종가', '거래량']]
x_data = np.array(x_data)
print(x_data.shape) #(426, 5)

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):     # 10
        end_ix = i + n_steps           # 0 + 4 = 4   n_steps : 몇개씩 자르는지
        if end_ix > len(sequence)-1:   # 4 > 10-1
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix, 4] # 0, 1, 2, 3 / 4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = x_data
n_steps = 5
x, y = split_sequence(dataset, n_steps)
print(x)
print(x.shape) #(421, 5, 5)
print('================================')
y = data_samsung['종가']
y = np.array(y)
y = y[5:]
print(y)
print(y.shape) #(421, )

x = x.reshape(421, 25)
model = Sequential()
model.add(Dense(20, input_shape=(25,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x, y, epochs=100, batch_size = 1, validation_split = 0) 

# 평가예측
loss, mse = model.evaluate(x, y, batch_size = 1)
print('loss:', loss)

# 평가예측
x_input = array([x[0]])  
x_input = x_input.reshape(1, 25)
print(x_input.shape)

y_pred = model.predict(x_input, batch_size = 1)
print(y_pred)