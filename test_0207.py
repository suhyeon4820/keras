import numpy as np
from numpy import array
import pandas as pd

# 데이터 준비 ===========================================================================================================
samsung = np.load('./test/samsung.npy')
kospi200 = np.load('./test/kospi200.npy')
print(samsung) #(430, 5)
print(samsung.shape) #(430, 5)
print(kospi200) #(430, 5)
print(kospi200.shape) #(430, 5)

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
# samsung(x1, y1)
x1, y1 = split_xy5(samsung, 5, 1)
print(x1.shape) #(425, 5, 5)
print(y1.shape) #(425, 1), 삼성전자 종가
print(x1[-1, :], '\n', y1[-1])

# 데이터셋 나누기 =======================================================================================================
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 1, test_size = 0.3, shuffle = False)

print(x1_train.shape) #(297, 5, 5)
print(x1_test.shape)  #(128, 5, 5)
print(y1_train.shape) #(297, 1)
print(y1_test.shape)  #(128, 1)

# 데이터 전처리 =========================================================================================================
x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
# print(x1_train[0, :])

x1_train = x1_train.reshape(297, 5, 5)
x1_test = x1_test.reshape(128, 5, 5)

# 모델 ====================================================================================================================
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(5, 5)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()

# 훈련 =========================================================================================================================
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x1_train, y1_train, epochs=100, batch_size = 10, validation_split = 0, callbacks=[early_stopping]) 

loss, mse = model.evaluate(x1_test, y1_test, batch_size = 10)
print('loss:', loss)


# x1_input = array(array([x1[424]]))  
x1_input = array([[57800, 58400, 56400, 56400, 19749457],
                  [55500, 57400, 55200, 57200, 23995260],
                  [57100, 59000, 56800, 58900, 21800192],
                  [60000, 60200, 58900, 59500, 19278165],
                  [60100, 61100, 59700, 61100, 14727159]])
x1_input = x1_input.reshape(1, 5, 5)


y_pred = model.predict(x1_input, batch_size = 10)
print('삼성전자 종가:', y_pred)
    
## 종가 : 7786 ㅜ