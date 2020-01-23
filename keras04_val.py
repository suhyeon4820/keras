import numpy as np    
# 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 102, 103, 104, 105])
# print(x.shape)

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # 연속적인 model layer
model.add(Dense(5, input_shape = (1, ))) # input_shape : 2차원 
model.add(Dense(2))
model.add(Dense(3)) # layer 2 추가(node : 10개)
model.add(Dense(1)) # outputlayer(node : 1개)

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x_train, y_train, epochs=150, batch_size = 1, validation_split = 0, validation_data = (x_val, y_val)) 

# 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print('mse:', mse)

# 예측
# x_pred = np.array([11, 12, 13])
aaa = model.predict(x_test, batch_size = 1)
print(aaa)


bbb = model.predict(x_val, batch_size = 1)
print(bbb)
