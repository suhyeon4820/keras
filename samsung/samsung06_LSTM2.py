import numpy as np
import pandas as pd

samsung = np.load('./samsung/data/samsung.npy')
kospi200 = np.load('./samsung/data/kospi200.npy')

# print(samsung) #(426, 5)
# print(samsung.shape) #(426, 5)
# print(kospi200) #(426, 5)
# print(kospi200.shape) #(426, 5)


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps    # 0 + 5 = 5 (x의 끝 숫자)
        y_end_number = x_end_number + y_column  # 5 + 1 =6 (y의 끝 숫자)
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number : y_end_number, 3]   # 3번째 열이 '종가'==> y값
        x.append(tmp_x) 
        y.append(tmp_y)
    return np.array(x), np.array(y) 

x, y = split_xy5(samsung, 25, 1)
print(x.shape) #(401, 25, 5)
print(y.shape) #(401, 1)
print(x[0,:],'\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, test_size=0.3, shuffle = False)

print(x_train.shape) #(280, 25, 5)
print(x_test.shape)  #(121, 25, 5)



# 데이터 전처리
# StandardScaler
#1. 3차원 => 2차원
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

print(x_train.shape)
print(x_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

x_train_scaled = x_train_scaled.reshape(280, 25, 5)
x_test_scaled = x_test_scaled.reshape(121, 25, 5)

print(x_train_scaled.shape)
print(x_test_scaled.shape)

-

# 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape = (25, 5)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, epochs=100, batch_size = 1, validation_split = 0, callbacks=[early_stopping]) 

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size = 1)
print('loss:', loss)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가:', y_test[i], 'y예측값:', y_pred[i])

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred))