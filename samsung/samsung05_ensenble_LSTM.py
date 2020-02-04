import numpy as np
import pandas as pd

samsung = np.load('./samsung/data/samsung.npy')
kospi200 = np.load('./samsung/data/kospi200.npy')

print(samsung) #(426, 5)
print(kospi200) #(426, 5)
print(samsung.shape) #(426, 5)
print(kospi200.shape) #(426, 5)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps   
        y_end_number = x_end_number + y_column   
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :] # x값(5x5)
        tmp_y = dataset[x_end_number : y_end_number, 3] # y값(종가)
        x.append(tmp_x) 
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)
print(x1.shape) #(421, 5, 5)
print(y1.shape) #(421, 1)
print(x1[0, :], '\n', y1[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 1, test_size = 0.3, shuffle = False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state = 1, test_size = 0.3, shuffle = False)

print(x1_train.shape) #(294, 5, 5)
print(x1_test.shape)  #(127, 5, 5)
print(x2_train.shape) #(294, 5, 5)
print(x2_test.shape)  #(127, 5, 5)
print(y1_train.shape) #(294, 5, 5)
print(y1_test.shape)  #(127, 5, 5)

# 데이터 전처리
# StandardScaler
x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
print(x1_train[0, :])

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
print(x2_train[0, :])

x1_train =  x1_train.reshape(294, 5, 5)
x1_test = x1_test.reshape(127, 5, 5)
x2_train =  x2_train.reshape(294, 5, 5)
x2_test = x2_test.reshape(127, 5, 5)

# 모델
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# model 1
input1 = Input(shape=(5, 5))
dense1 = LSTM(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(1)(dense1)

# model 2
input2 = Input(shape=(5, 5))
dense2 = LSTM(64)(input2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
output2 = Dense(1)(dense2)

from keras.layers import Concatenate, concatenate
merge1 = concatenate([output1, output2])
middle1 = Dense(16)(merge1)
middle2 = Dense(8)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train, x2_train], y1_train, epochs=50, batch_size = 1, validation_split = 0, callbacks=[early_stopping]) 

loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)
print('loss:', loss)

y_pred = model.predict([x1_test, x2_test])

for i in range(5):
    print('종가:', y1_test[i], 'y예측값:', y_pred[i])
    
# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y1_test, y_pred))
print('RMSE : ', RMSE(y1_test, y_pred))