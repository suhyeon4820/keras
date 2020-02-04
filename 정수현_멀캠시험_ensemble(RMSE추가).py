import numpy as np
from numpy import array
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

data_samsung = pd.read_csv('samsung.csv',encoding='cp949', thousands = ',')
print(data_samsung.shape) #(426, 6)
x_data = data_samsung[['시가', '고가', '저가', '종가', '거래량']]
x_data = np.array(x_data)
print(x_data.shape) #(426, 5)

data_kospi = pd.read_csv('kospi200.csv',encoding='cp949', thousands = ',')
x1_data = data_kospi[['시가', '고가', '저가', '현재가', '거래량']]
x1_data = np.array(x1_data)
print(x1_data.shape) #(426, 5)

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

# 삼성전자 Data
dataset = x_data
n_steps = 5
x1, y = split_sequence(dataset, n_steps)
print(x1)
print(x1.shape) #(421, 5, 5)
x1 = x1.reshape(421, 25)

# Kospi Data
dataset = x1_data
n_steps = 5
x2, y = split_sequence(dataset, n_steps)
print(x2)
print(x2.shape) #(421, 5, 5)
x2 = x2.reshape(421, 25)

print('================================')
y = data_samsung['종가']
y = np.array(y)
y = y[5:]
print(y)
print(y.shape) #(421, )

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.6, random_state=66, shuffle = False)

# model 1
input1 = Input(shape=(25, ))
dense1 = Dense(32)(input1)
dense1 = Dense(16)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)

# model 2
input2 = Input(shape=(25, ))
dense2 = Dense(32)(input2)
dense2 = Dense(16)(dense2)
dense2 = Dense(8)(dense2)
dense2 = Dense(4)(dense2)
output2 = Dense(1)(dense2)

from keras.layers import Concatenate, concatenate
merge1 = Concatenate(axis=1)([output1, output2])
middle1 = Dense(16)(merge1)
middle2 = Dense(8)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit([x1, x2], y, epochs=100, batch_size = 1, validation_split = 0) 

# 평가예측
loss, mse = model.evaluate([x1, x2], y, batch_size = 1)
print('loss:', loss)

# 평가예측
x1_input = array(array([x1[0]]))  
x1_input = x1_input.reshape(1, 25)
print(x1_input.shape)

x2_input = array(array([x2[0]]))  
x2_input = x2_input.reshape(1, 25)
print(x2_input.shape)

y_pred = model.predict([x1_input, x2_input], batch_size = 1)
print(y_pred)

y_pred = model.predict([x1_test, x2_test], batch_size = 1)
print(y_pred)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred))