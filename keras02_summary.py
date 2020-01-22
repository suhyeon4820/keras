import numpy as np    
# 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # 연속적인 model layer
model.add(Dense(5, input_dim = 1)) # layer 1 추가(node : 20개)
model.add(Dense(2))
model.add(Dense(3)) # layer 2 추가(node : 10개)
model.add(Dense(1)) # layer 3 추가(node : 1개)

model.summary()

# # 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
# model.fit(x, y, epochs=150, batch_size = 1) 

# # 평가예측
# loss, mse = model.evaluate(x, y, batch_size = 1)
# print('mse:', mse)

# # 예측
# x_pred = np.array([11, 12, 13])
# aaa = model.predict(x_pred, batch_size = 1)
# print(aaa)

# bbb = model.predict(x, batch_size = 1)
# print(bbb)
